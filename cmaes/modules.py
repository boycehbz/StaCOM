'''
 @FileName    : modules.py
 @EditTime    : 2021-10-14 20:36:14
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import time
import os
import torch
import random
import numpy as np
import yaml
from utils.CSVLogger import Logger
from utils.smpl_torch_batch import SMPLModel
import gym
import vclrl_envs
from vclrl_envs.data.amass_config import joint_names

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def init(note='occlusion', dtype=torch.float32, device=torch.device('cpu'), viz=False, **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join('output', note)
    out_dir = os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Create the log for the current experiment
    # logger = Logger(os.path.join(out_dir, 'log.csv'), title="vposer")
    # logger.set_names([note])
    # logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    # logger.set_names(joint_names)
    logger = Logger(os.path.join(out_dir, 'log.csv'), title="vposer")

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/SMPL_NEUTRAL.pkl', 
                        data_type=dtype,
                    )
    
    # Load env
    env_name = "AMASSSampleEnv-v0"
    # env = gym.make(env_name, render=viz, data=kwargs.get('data_folder'), smpl=model_smpl, vis_motion=False)
    env = gym.make(
        env_name,
        render=viz,
        data=kwargs.get('data_folder'),
        smpl=model_smpl,
        vis_motion=False,
        object_mesh_path=kwargs.get('object_mesh_path'),
        object_pose_path=kwargs.get('object_pose_path'),
        object_mass=5.0,
        object_stability_seconds= 0.01,
        object_stability_weight= 10.0
    )
    num_agents = getattr(env, "num_agents", 1)
    log_names = []
    for aid in range(num_agents):
        log_names.extend([f"agent{aid}_{name}" for name in joint_names])
    logger.set_names(log_names)
    return out_dir, logger, model_smpl, env


class DatasetLoader():
    def __init__(self, trainset=None, testset=None, smpl_model=None, generator=None, data_folder='./data', dtype=torch.float32, frame_length=16, **kwargs):
        self.data_folder = data_folder
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.dtype = dtype
        self.model = smpl_model
        self.generator = generator
        self.frame_length = frame_length

    def load_trainset(self):
        train_dataset = []
        for i in range(len(self.trainset)):
            train_dataset.append(VideoData(True, self.data_folder, self.model, self.trainset[i], self.frame_length))
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        return train_dataset

    def load_testset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            test_dataset.append(VideoData(False, self.data_folder, self.model, self.testset[i], self.frame_length))
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset

    def load_evalset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            test_dataset.append(VideoData(False, self.data_folder, self.model, self.testset[i], self.frame_length))
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset

class ModelLoader():
    def __init__(self, model=None, lr=0.001, device=torch.device('cpu'), pretrain=False, pretrain_dir='', output='', smpl=None, frame_length=16, **kwargs):
        self.smpl = smpl
        self.output = output
        # self.log = open(os.path.join(self.output, 'results.txt'), 'w')
        self.data_shape = [1,kwargs.pop('data_shape'),3]
        self.num_neurons = kwargs.pop('num_neurons')
        self.latentD = kwargs.get('latentD')
        self.frame_length = frame_length
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(self.latentD)

        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count:', model_params)

        self.device = device
        #if uv_mask:
        self.uv_mask = cv2.imread('./data/MASK.png')
        if self.uv_mask.max() > 1:
            self.uv_mask = self.uv_mask / 255.

        print('load model: %s' %self.model_type)

        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=1, verbose=True)

        # load segnet model
        if False:
            segmodel_dir = 'pretrain_model/poseseg_epoch007.pkl'
            fixmodel_dict = torch.load(segmodel_dir).state_dict()
            model_dict = self.model.state_dict()
            fixmodel_dict = {'poseseg.' + k: v for k, v in fixmodel_dict.items() if 'poseseg.' + k in model_dict}
            model_dict.update(fixmodel_dict)
            self.model.load_state_dict(model_dict)
            for param in self.model.poseseg.parameters():
                param.requires_grad = False
            print("load seg model")


        # load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            self.optimizer.load_state_dict(params['optimizer'])
            print("load pretrain parameters from %s" %pretrain_dir)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


    def save_model(self, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, '%s_epoch%03d.pkl' %(task, epoch))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save model to %s' % model_name)

    def save_results(self, pred_params, gt_params, i, batchsize):
        # save trained model
        output = os.path.join(self.output, 'meshes')
        if not os.path.exists(output):
            os.makedirs(output)

        shape = torch.zeros((1, 10), device=pred_params.device, dtype=pred_params.dtype)
        trans = torch.zeros((1, 3), device=pred_params.device, dtype=pred_params.dtype)

        for f, (pred, gt) in enumerate(zip(pred_params, gt_params)):
            pred = pred.reshape(1, 72)
            gt = gt.reshape(1, 72)

            pre_mesh, _ = self.smpl(shape, pred, trans)
            gt_mesh, _ = self.smpl(shape, gt, trans)
            pre_mesh = pre_mesh[0]
            gt_mesh = gt_mesh[0]

            pre_path = os.path.join(output, '%02d_pre.obj' %(i*batchsize + f))
            self.smpl.write_obj(pre_mesh, pre_path)
            gt_path = os.path.join(output, '%02d_gt.obj' %(i*batchsize + f))
            self.smpl.write_obj(gt_mesh, gt_path)

    def save_eval_results(self, pred_uv, gt_uv, i, batchsize, generator, loss):
        # save trained model
        output = os.path.join(self.output, 'meshes')
        if not os.path.exists(output):
            os.makedirs(output)
        
        self.log.write(str(i) + ' ' + str(loss) + '\n')
        self.log.flush()
        if loss < 3000:
            return
        for ind, (op, uv) in enumerate(zip(pred_uv, gt_uv)):
            op = op.transpose(1, 2, 0)  # H*W*C
            uv = uv.transpose(1, 2, 0)  # H*W*C

            # cv2.imshow('pre', op + 0.5)
            # cv2.imshow('img', uv + 0.5)
            # cv2.waitKey(0)

            pre_mesh = resample_np(generator, op)
            gt_mesh = resample_np(generator, uv)
            # Scale to original size---hbz 07/17
            pre_path = os.path.join(output, '%05d_pre.obj' %(i*batchsize+ind))
            self.smpl.write_obj(pre_mesh, pre_path)
            gt_path = os.path.join(output, '%05d_gt.obj' %(i*batchsize+ind))
            self.smpl.write_obj(gt_mesh, gt_path)

            preim_path = os.path.join(output, '%05d_preim.jpg' %(i*batchsize+ind))
            cv2.imwrite(preim_path, (op+0.5)*255)
            gtim_path = os.path.join(output, '%05d_gtim.jpg' %(i*batchsize+ind))
            cv2.imwrite(gtim_path, (uv+0.5)*255)


class LossLoader():
    def __init__(self, train_loss='L1', test_loss='L1', device=torch.device('cpu'),  batchsize=1, dtype=torch.float32, generator=None, data_folder=None, testset=None, viz=False, **kwargs):
        env_name = "AMASSSampleEnv-v0" #HumanoidStandEnv HumanoidWalkerEnv
        character_path = os.path.join(data_folder, '%s.urdf' %testset)
        scene_path = os.path.join(data_folder, '%s_scene.urdf' %testset)
        data_path = os.path.join(data_folder, testset)
        viz = not viz
        self.env = gym.make(env_name, render=viz, character=character_path, scene=scene_path, data=data_path)
        self.bc = self.env.unwrapped._p
        self.env.reset()
        self.sample = Sample(self.bc)

        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.device = device
        self.generator = generator
        self.smpl = SMPLModel(
                                device=device,
                                model_path='./data/SMPL_NEUTRAL.pkl', 
                                data_type=dtype,
                            )
        self.smpl_gpu = SMPLModel(
                                device=torch.device('cuda'),
                                model_path='./data/SMPL_NEUTRAL.pkl', 
                                data_type=dtype,
                            )
        self.kl_coef = kwargs.get('kl_coef')
        self.latentD = kwargs.get('latentD')
        self.frame_length = 16
        self.train_loss = {}
        self.geodesic_loss = geodesic_loss_R(reduction='mean')
        for loss in self.train_loss_type:
            if loss == 'L1':
                self.train_loss.update(L1=nn.L1Loss(size_average=False).to(self.device))
            elif loss == 'partloss':
                self.train_loss.update(partloss=part_loss(self.generator).to(self.device))
            elif loss == 'weight_L1':
                self.train_loss.update(w_L1=weight_L1(self.device))
            elif loss == 'L2':
                self.train_loss.update(L2=nn.MSELoss(size_average=True).to(self.device))
            elif loss == 'LPloss':
                self.train_loss.update(LPloss=LPloss(self.device))
            elif loss == 'boneloss':            
                self.train_loss.update(boneloss=boneloss(self.generator).to(self.device))
            elif loss == 'shapeloss':            
                self.train_loss.update(shapeloss=shapeloss(self.generator).to(self.device))
            elif loss == 'resampledloss':            
                self.train_loss.update(resampledloss=resampledloss(self.generator, self.smpl).to(self.device))

        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=nn.L1Loss(size_average=False).to(self.device))

    def aa2matrot(self, pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 3, 3)
        return pose_body_matrot

    def simulate(self, cur_state, kin_state, pred):
        assert cur_state.shape[0] == 1
        actions = pred #['mean'] #pred['action']

        # Phys loss
        cur_state = cur_state.detach().cpu().numpy()
        kin_state = kin_state.detach().cpu().numpy()
        action = actions.detach().cpu().numpy()

        out_actions = []
        for i, (c, k, a) in enumerate(zip(cur_state, kin_state, action)):
            # apply sampling
            tar_pose = k.copy()
            tar_pose[6:57] = tar_pose[6:57] + a

            # transform to quaternion
            tar_pose_quan = self.sample.Euler2Quaternion(tar_pose)[:75]
            # tar_pose_quan = self.sample.Quaternion2AxisAngle(tar_pose_quan)
            cur_state_quan = self.sample.Euler2Quaternion(c)
            kin_pose_quan = self.sample.Euler2Quaternion(k)
            kin_pose_quan[75:] = 0

            obs, reward, done, action_output = self.env.step([tar_pose_quan, kin_pose_quan, cur_state_quan, True])
            obs = self.sample.Quaternion2Euler(np.array(obs))
            out_actions.append(action_output)
        #     sim_results.append(obs[:43])
        # sim_results = torch.tensor(sim_results, dtype=dtype, device=device)
        return obs, done, np.array(out_actions), reward

    def calcul_trainloss(self, cur_state, kin_state, mean, sigma, pred, dist_last):

        SCALER = 1000
        pred_params = pred['param']
        actions = pred['action']
        batch_size, dim = pred_params.size()
        device = pred_params.device
        dtype = pred_params.dtype
        kin_pose = kin_state[:,:57]

        # dist smooth
        l_mean = dist_last['mean']
        l_sigma = dist_last['std']
        smooth_loss = torch.norm(mean - l_mean)
        smooth_loss += torch.norm(sigma - l_sigma)

        # Phys loss
        cur_state = cur_state.detach().cpu().numpy()
        kin_state = kin_state.detach().cpu().numpy()
        action = actions.detach().cpu().numpy()
        sim_results = []
        for c, k, a in zip(cur_state, kin_state, action):
            # apply sampling
            tar_pose = k.copy()
            tar_pose[6:57] = tar_pose[6:57] + a

            # transform to quaternion
            tar_pose_quan = self.sample.Euler2Quaternion(tar_pose)[:75]
            cur_state_quan = self.sample.Euler2Quaternion(c)
            kin_pose_quan = self.sample.Euler2Quaternion(k)

            obs, _, _, _ = self.env.step([tar_pose_quan, kin_pose_quan, cur_state_quan])
            obs = self.sample.Quaternion2Euler(np.array(obs))
            sim_results.append(obs[:57])
        sim_results = torch.tensor(sim_results, dtype=dtype, device=device)

        # pred_params = pred_params.view(-1, 72)
        # gt_params = gt_params.view(-1, 72)

        # shape = torch.zeros((pred_params.size(0), 10), device=device, dtype=dtype)
        # trans = torch.zeros((pred_params.size(0), 3), device=device, dtype=dtype)

        # pred_mesh, _ = self.smpl_gpu(shape, pred_params, trans)
        # with torch.no_grad():
        #     gt_mesh, _ = self.smpl_gpu(shape, gt_params, trans)

        rec_loss = 0
        phys_loss = 0
        for ltype in self.train_loss:
            if ltype == 'L2':
                rec_loss = self.train_loss['L2'](pred_params, kin_pose)
                phys_loss = self.train_loss['L2'](pred_params, sim_results)
                # loss += self.train_loss['L2'](pred_mesh, gt_mesh)
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass

        rec_loss = (1. - self.kl_coef) * rec_loss * SCALER
        phys_loss = (1. - self.kl_coef) * phys_loss * SCALER

        # KL loss
        q_z = torch.distributions.normal.Normal(pred['mean'], pred['std'])
        p_z = torch.distributions.normal.Normal(mean, sigma)
        loss_kl = self.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        n_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl += self.kl_coef * 10 * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, n_z), dim=[1]))

        # loss_matrot = 10 * self.geodesic_loss(self.aa2matrot(gt_params.view(-1, 3)), self.aa2matrot(pred_params.view(-1, 3)))

        # local linear loss
        # mean = pred['mean']
        # inter = (mean[:,:-2,:] + mean[:,2:,:]) / 2
        # middle = mean[:,1:-1,:]
        # latent_linear = ((inter - middle)**2).mean() * 10

        # pred_mesh = pred_mesh.view(batch_size, frame_length, -1, 3)
        # inter = (pred_mesh[:,:-2,:] + pred_mesh[:,2:,:]) / 2
        # middle = pred_mesh[:,1:-1,:]
        # latent_linear += ((inter - middle)**2).mean() * 100
        # latent_linear = 0.

        loss_dict = {'loss_kl': loss_kl,
                     'loss_rec': rec_loss,
                     'loss_phy': phys_loss,
                     'loss_smooth': smooth_loss,
                    #  'loss_matrot': loss_matrot,
                    #  'latent_linear': latent_linear,
                     }

        # if epoch < 10:
        #     loss_dict['loss_pose_rec'] = (1. - self.kl_coef) * torch.mean(torch.sum(torch.pow(porig - prec, 2), dim=[1, 2, 3]))

        loss_total = torch.stack(list(loss_dict.values())).sum()

        for k in loss_dict:
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 6)

        return loss_total, loss_dict

    def calcul_testloss(self, cur_state, kin_state, mean, sigma, pred):
        
        SCALER = 1000
        pred_params = pred['param']
        batch_size = pred_params.size(0)
        device = pred_params.device
        dtype = pred_params.dtype
        kin_pose = kin_state[:,:57]
        # q_z = torch.distributions.normal.Normal(pred['mean'], pred['std'])

        # pred_params = pred_params.reshape(-1, 72)
        # gt_params = gt_params.reshape(-1, 72)

        # shape = torch.zeros((pred_params.size(0), 10), device=device, dtype=dtype)
        # trans = torch.zeros((pred_params.size(0), 3), device=device, dtype=dtype)

        # pred_mesh, _ = self.smpl_gpu(shape, pred_params, trans)
        # gt_mesh, _ = self.smpl_gpu(shape, gt_params, trans)

        loss = 0
        for ltype in self.train_loss:
            if ltype == 'L2':
                loss += self.train_loss['L2'](pred_params, kin_pose)
                # loss += self.train_loss['L2'](pred_mesh, gt_mesh)
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass

        loss_mesh_rec = loss * SCALER

        loss_dict = {'loss_mesh_rec': loss_mesh_rec,
                     }

        loss_total = torch.stack(list(loss_dict.values())).sum()

        for k in loss_dict:
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 6)

        return loss_total, loss_dict

    def calcul_KLloss(self, drec):
        
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        porig = drec['uv_decode']

        device = porig.device
        dtype = porig.dtype
        batch_size = porig.size(0)

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, self.latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl = self.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        loss_dict = {
                     'loss_kl': loss_kl
                     }

        #if self.epochs_completed < 10:
        # loss_dict['loss_pose_rec'] = (1. - self.kl_coef) * torch.mean(torch.sum(torch.pow(porig - prec, 2), dim=[1, 2, 3]))

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def calcul_maskloss(self, m0, m1, m2, m3, mask2, mask3, mask4):
        loss = 0.
        loss += (self.train_loss['L2'](m0, mask2) * 10)
        loss += (self.train_loss['L2'](m1, mask2) * 10)
        loss += (self.train_loss['L2'](m2, mask3) * 10)
        loss += (self.train_loss['L2'](m3, mask4) * 10)
        return loss

    def calcul_latentloss(self, pred, gt):
        loss = 0.
        for ltype in self.train_loss:
            if ltype == 'L1':
                loss += self.train_loss['L1'](pred, gt)
            elif ltype == 'L2':
                pass
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass
        return loss







