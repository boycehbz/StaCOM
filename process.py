
import torch
import numpy as np
import cv2
from tqdm import tqdm
import time
import os
from utils.geometry import batch_rodrigues

def merge_gt(data):
    batch_size, frame_length, agent_num = data['pose'].shape[:3]

    data['data_shape'] = data['pose'].shape[:3]
    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, 6890, 3)
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)
    data['x'] = data['x'].reshape(batch_size*frame_length*agent_num, -1)
    data['valid'] = data['valid'].reshape(batch_size*frame_length*agent_num,)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid(data, is_train=False):
    batch_size, frame_length, agent_num = data['pose'].shape[:3]

    data['data_shape'] = data['pose'].shape[:3]
    data['valid'] = data['valid'].reshape(batch_size*frame_length*agent_num,)

    if not is_train:
        data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, -1, 3)
        data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)

    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)

    imgname = (np.array(data['imgname']).T).reshape(batch_size, frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['norm_img'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'].reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]

    # imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    # data['imgname'] = imgname.tolist()

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname'], 'obj_path':data['obj_path']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname', 'obj_path']}
    data = {**imnames, **data}
    # data['img'] = data['img'].to(device)
    # data['pose'] = data['pose'].to(device)
    # data['betas'] = data['betas'].to(device)
    # data['gt_cam_t'] = data['gt_cam_t'].to(device)
    # data['keypoints'] = data['keypoints'].to(device)
    # data['pred_keypoints'] = data['pred_keypoints'].to(device)
    # data['verts'] = data['verts'].to(device)
    # data['joints'] = data['joints'].to(device)
    
    # data["center"] = data["center"].to(device).float()
    # data["scale"] = data["scale"].to(device).float()
    # data["img_h"] = data["img_h"].to(device).float()
    # data["img_w"] = data["img_w"].to(device).float()
    # data["focal_length"] = data["focal_length"].to(device).float()

    return data


def hoi_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        batchsize = data['pose'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data, is_train=True)

        # forward
        pred = model.model(data)

        # calculate loss
        if 'Single_AMP_Loss' in loss_func.train_loss_type and loss_func.single_amp_trainer is not None:
            single_module = loss_func.train_loss.get('Single_AMP_Loss')
            if single_module is not None:
                fake_body_rotmat = single_module.latest_fake_body_rotmat
                fake_betas = single_module.latest_fake_betas
                real_body_rotmat = single_module.latest_real_body_rotmat if not loss_func.single_amp_trainer.uses_external_real else None
                real_betas = single_module.latest_real_betas if not loss_func.single_amp_trainer.uses_external_real else None

                if fake_body_rotmat is not None and fake_betas is not None:
                    loss_func.single_amp_trainer.train_discriminator_step(
                        real_body_rotmat,
                        real_betas,
                        fake_body_rotmat,
                        fake_betas
                    )

        if 'Interact_AMP_Loss' in loss_func.train_loss_type and loss_func.interact_amp_trainer is not None:
            interact_module = loss_func.train_loss.get('Interact_AMP_Loss')
            if interact_module is not None:
                fake_features = interact_module.latest_fake_features
                real_features = interact_module.latest_real_features

                if fake_features is not None and real_features is not None and fake_features.numel() > 0 and real_features.numel() > 0:
                    loss_func.interact_amp_trainer.train_discriminator_step(
                        real_features,
                        fake_features
                    )

        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)
            # cur_loss_dict.update({
            #     'disc_loss': disc_metrics['disc_loss'],
            #     'disc_acc': disc_metrics['accuracy']
            # })
           

        debug = False
        if debug:
            results = {}
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            model.save_generated_interaction(results, i, batchsize)

        debug = False
        if debug:
            results = {}
            results.update(imgs=data['imgname'])
            results.update(data_shape=data['data_shape'])
            results.update(obj_path=data['obj_path'])
            results.update(obj_pose=data['obj_pose'].detach().cpu().numpy().astype(np.float32))
            # results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            # results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'MPJPE_instance' in cur_loss_dict.keys():
                results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach() #/ batchsize
        tqdm.write('epoch: %d/%d, batch: %d/%d, loss: %.6f %s' %(epoch, num_epoch, i, len_data, loss_batch, cur_loss_dict))
        train_loss += loss_batch

        if (epoch + 1) % 10 == 0 and i % 4000 == 0:
            disc_save_dir = os.path.join(model.output, 'amp_checkpoints')
            os.makedirs(disc_save_dir, exist_ok=True)

            if loss_func.single_amp_trainer is not None:
                single_path = os.path.join(disc_save_dir, f'single_amp_epoch{epoch + 1}.pth')
                loss_func.single_amp_trainer.save_discriminator(single_path)

            if loss_func.interact_amp_trainer is not None:
                interact_path = os.path.join(disc_save_dir, f'interact_amp_epoch{epoch + 1}.pth')
                loss_func.interact_amp_trainer.save_discriminator(interact_path)


    return train_loss/len_data

def hoi_test(model, loss_func, loader, epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    loss_func.reset_accumulators()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):

            batchsize = data['pose'].shape[0]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if i == 0: #loss.max() > 100:# 23,25,26,49,6,11,14
                results = {}
                results.update(imgs=data['imgname'])
                results.update(data_shape=data['data_shape'])
                results.update(obj_path=data['obj_path'])
                results.update(obj_pose=data['obj_pose'].detach().cpu().numpy().astype(np.float32))
                # results.update(single_person=data['single_person'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                # results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                if 'MPJPE_instance' in cur_loss_dict.keys():
                    results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize)

            loss_batch = loss.mean().detach() #/ batchsize
            tqdm.write('batch: %d/%d, loss: %.6f %s' %(i, len(loader), loss_batch, cur_loss_dict))
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        final_metrics = loss_func.compute_final_metrics()
        print("Final accumulated metrics:", final_metrics)
        return final_metrics

def hoi_eval(model, loader, loss_func, device=torch.device('cpu')):

    print('-' * 10 + 'model eval' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    output = {'pose':{}, 'shape':{}, 'trans':{}}
    gt = {'pose':{}, 'shape':{}, 'trans':{}, 'gender':{}, 'valid':{}}
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            # if i > 1:
            #     break
            batchsize = data['keypoints'].shape[0]
            seq_id = data['seq_id']
            frame_id = torch.cat(data['frame_id']).reshape(-1, batchsize)
            frame_id = frame_id.detach().cpu().numpy().T

            batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

            del data['seq_id']
            del data['frame_id']
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)

            pred_pose = pred['pred_pose'].reshape(batch_size, frame_length, agent_num, -1)
            pred_shape = pred['pred_shape'].reshape(batch_size, frame_length, agent_num, -1)
            pred_trans = pred['pred_cam_t'].reshape(batch_size, frame_length, agent_num, -1)

            pred_pose = pred_pose.detach().cpu().numpy()
            pred_shape = pred_shape.detach().cpu().numpy()
            pred_trans = pred_trans.detach().cpu().numpy()

            gt_pose = data['pose'].reshape(batch_size, frame_length, agent_num, -1)
            gt_shape = data['betas'].reshape(batch_size, frame_length, agent_num, -1)
            gt_trans = data['gt_cam_t'].reshape(batch_size, frame_length, agent_num, -1)
            gt_gender = data['gender'].reshape(batch_size, frame_length, agent_num)
            valid = data['valid'].reshape(batch_size, frame_length, agent_num)

            gt_pose = gt_pose.detach().cpu().numpy()
            gt_shape = gt_shape.detach().cpu().numpy()
            gt_trans = gt_trans.detach().cpu().numpy()
            gt_gender = gt_gender.detach().cpu().numpy()
            valid = valid.detach().cpu().numpy()

            for batch in range(batchsize):
                s_id = str(int(seq_id[batch]))
                for f in range(frame_length):

                    if s_id not in output['pose'].keys():
                        output['pose'][s_id] = [pred_pose[batch][f]]
                        output['shape'][s_id] = [pred_shape[batch][f]]
                        output['trans'][s_id] = [pred_trans[batch][f]]

                        gt['pose'][s_id] = [gt_pose[batch][f]]
                        gt['shape'][s_id] = [gt_shape[batch][f]]
                        gt['trans'][s_id] = [gt_trans[batch][f]]
                        gt['gender'][s_id] = [gt_gender[batch][f]]
                        gt['valid'][s_id] = [valid[batch][f]]
                    else:
                        output['pose'][s_id].append(pred_pose[batch][f])
                        output['shape'][s_id].append(pred_shape[batch][f])
                        output['trans'][s_id].append(pred_trans[batch][f])

                        gt['pose'][s_id].append(gt_pose[batch][f])
                        gt['shape'][s_id].append(gt_shape[batch][f])
                        gt['trans'][s_id].append(gt_trans[batch][f])
                        gt['gender'][s_id].append(gt_gender[batch][f])
                        gt['valid'][s_id].append(valid[batch][f])
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize)

        return output, gt

