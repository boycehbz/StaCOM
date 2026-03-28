'''
 @FileName    : cmaes.py
 @EditTime    : 2021-10-22 11:21:47
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from utils.cmaes import CMA
import torch
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compensate_system_error(env, cur_state, ref_states, keyp_2d):
    ref_states = np.asarray(ref_states)
    cur_state  = np.asarray(cur_state)
    W, N = ref_states.shape[0], ref_states.shape[1]

    res       = np.zeros((W, N, 57), dtype=np.float32)
    tar_pose  = ref_states[:, :, :57].copy() + res
    ext_force = np.zeros((W, N, 3), dtype=np.float32)

    obs, reward, done, info = env.step([tar_pose, ref_states, cur_state, ext_force, keyp_2d])
    sys_error = ref_states[:, :, :57] - obs[:, :, :57]
    return sys_error


class CMAES():
    def __init__(self, env, times=20, device=torch.device('cpu'), window_size=1, **kwargs):
        self.env      = env
        self.bc       = self.env.unwrapped._p
        self.env.reset()
        self.times     = times
        self.n_windows = window_size
        self.n_agents  = len(getattr(self.env, "robots",
                                     [getattr(self.env, "robot", None)]))
        if self.n_agents <= 0:
            self.n_agents = 1

        AT = 0.15
        AB = 0.15
        AC = 0.20
        AS = 0.60
        AE = 0.50

        bounds_one = np.array([
            [-AT, AT], [-AT, AT], [-AT, AT],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [-AB, AB], [-AB, AB], [-AB, AB],
            [-AB, AB], [-AB, AB], [-AB, AB],
            [-AB, AB], [-AB, AB], [-AB, AB],
            [0, 0], [0, 0], [0, 0],
            [0, 0], [0, 0], [0, 0],
            [-AC, AC], [-AC, AC], [-AC, AC],
            [-AS, AS], [-AS, AS], [-AS, AS],
            [-AE, AE], [-AE, AE], [-AE, AE],
            [-AC, AC], [-AC, AC], [-AC, AC],
            [-AS, AS], [-AS, AS], [-AS, AS],
            [-AE, AE], [-AE, AE], [-AE, AE],
        ], dtype=np.float32)

        assert bounds_one.shape[0] == 57, \
            f"bounds_one should be (57,2), got {bounds_one.shape}"

        self.bounds = np.tile(bounds_one, (self.n_windows * self.n_agents, 1))
        self.data_type = torch.float32
        self.np_type   = np.float32
        self.device    = device

    def __call__(
        self,
        mean,
        sigma,
        cur_state,
        ref_state,
        sys_error,
        keyp_2d,
        show_progress=False,
        progress_desc=None,
    ):
        mean      = np.asarray(mean,      dtype=np.float32)
        cur_state = np.asarray(cur_state)
        ref_state = np.asarray(ref_state)
        sys_error = np.asarray(sys_error)

        W = ref_state.shape[0]
        N = ref_state.shape[1]
        assert W == self.n_windows
        assert N == self.n_agents
        assert mean.shape[0] == 57 * W * N

        effective_sigma = min(float(sigma), 0.30)

        optimizer = CMA(
            mean=mean.copy(),
            sigma=effective_sigma,
            bounds=self.bounds,
            population_size=12,
        )

        value_min   = np.inf
        best_result = None
        generation_summaries = []

        generation_iter = range(self.times)
        if show_progress:
            generation_iter = tqdm(
                generation_iter,
                total=self.times,
                desc=progress_desc or 'CMAES',
                leave=False,
            )

        for generation in generation_iter:
            generation_values          = []
            generation_contact_costs   = []
            generation_stability_costs = []
            generation_win_contact     = []
            generation_win_contact_pa  = []
            generation_win_contact_pw  = []
            generation_win_stability   = []
            solutions = []

            for _ in range(optimizer.population_size):
                x   = optimizer.ask()
                res = x.copy().reshape(W, N, 57)

                tar_pose  = ref_state[:, :, :57].copy() + res + sys_error
                ext_force = np.zeros((W, N, 3), dtype=np.float32)

                obs, value, done, info = self.env.step(
                    [tar_pose, ref_state, cur_state, ext_force, keyp_2d]
                )

                value_f = float(value)
                solutions.append((x, value_f))
                generation_values.append(value_f)

                if isinstance(info, dict):
                    ct  = info.get("contact_cost_total")
                    st  = info.get("stability_cost_total")
                    cw  = info.get("contact_costs")
                    cpa = info.get("contact_costs_per_agent")
                    cpw = info.get("contact_costs_per_wrist")
                    sw  = info.get("stability_costs")
                    if ct  is not None: generation_contact_costs.append(float(ct))
                    if st  is not None: generation_stability_costs.append(float(st))
                    if cw  is not None:
                        generation_win_contact.extend([float(v) for v in cw])
                    if cpa is not None:
                        generation_win_contact_pa.extend(
                            [[float(v) for v in w] for w in cpa])
                    if cpw is not None:
                        generation_win_contact_pw.extend(
                            [[[float(v) for v in wrist] for wrist in w]
                             for w in cpw])
                    if sw  is not None:
                        generation_win_stability.extend([float(v) for v in sw])

                if value_f < value_min:
                    value_min   = value_f
                    best_result = {
                        'value':    value_f,
                        'residual': res.copy(),
                        'obs':      np.asarray(obs).copy(),
                        'tar_pose': np.asarray(tar_pose).copy(),
                        'info':     info if isinstance(info, dict) else None,
                    }

            optimizer.tell(solutions)

            vstat   = np.asarray(generation_values, dtype=np.float32)
            summary = {
                'generation':  int(generation + 1),
                'value_mean':  float(vstat.mean()),
                'value_min':   float(vstat.min()),
                'value_max':   float(vstat.max()),
            }
            if generation_contact_costs:
                cs = np.asarray(generation_contact_costs, dtype=np.float32)
                summary.update(contact_cost_mean=float(cs.mean()),
                               contact_cost_min=float(cs.min()),
                               contact_cost_max=float(cs.max()))
            if generation_stability_costs:
                ss = np.asarray(generation_stability_costs, dtype=np.float32)
                summary.update(stability_cost_mean=float(ss.mean()),
                               stability_cost_min=float(ss.min()),
                               stability_cost_max=float(ss.max()))
            generation_summaries.append(summary)

            if show_progress and hasattr(generation_iter, 'set_postfix'):
                generation_iter.set_postfix(
                    vmin=f"{summary['value_min']:.4f}",
                    vmean=f"{summary['value_mean']:.4f}",
                )

            log_parts = [
                f"value_mean={summary['value_mean']:.6f}",
                f"value_min={summary['value_min']:.6f}",
                f"value_max={summary['value_max']:.6f}",
            ]
            if 'contact_cost_mean' in summary:
                log_parts.append("contact={:.4f}/{:.4f}".format(
                    summary['contact_cost_mean'], summary['contact_cost_min']))
            if 'stability_cost_mean' in summary:
                log_parts.append("stability={:.4f}/{:.4f}".format(
                    summary['stability_cost_mean'], summary['stability_cost_min']))
            logger.info("CMAES generation %d/%d: %s",
                        generation + 1, self.times, " | ".join(log_parts))

        return {
            'optimizer':            optimizer,
            'best':                 best_result,
            'best_value':           float(value_min),
            'generation_summaries': generation_summaries,
        }