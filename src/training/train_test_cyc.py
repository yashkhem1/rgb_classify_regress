import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
from common.utils import AverageMeter
from common.camera_utils import h36m_cameras_extrinsic_params as h36m_cam_params, rotate_emb
from models.criterion_list import calculate_soft_ngh, cal_cyc_mu_sig
from models.criterion_mean_bone import MeanBoneLenLoss
from common.error_metrics import un_normalise_pose, cal_avg_l2_jnt_dist, scale_norm_pose, normalise_pose

valid_reg_jnts = torch.cat((torch.arange(0,18), torch.arange(21,48)))


def run_epoch(epoch, opt, data_loader, model, optimizer=None, split='train'):

    if split == 'train':
        model['backbone'].train()
        model['temporal'].train()
        # model['pose'].train()
        bn_momentum = opt.bn_momentum
        assert optimizer is not None
        # if epoch == 1:
        data_loader.dataset.init_epoch()
    else:
        if epoch == 1:
            data_loader.dataset.init_epoch()
        model['backbone'].train()
        model['temporal'].train()
        # model['pose'].eval()
        bn_momentum = 0.0

    for _, m in model['backbone'].named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = bn_momentum
        if isinstance(m, torch.nn.BatchNorm3d):
            m.momentum = bn_momentum

    for _, m in model['temporal'].named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = bn_momentum
        if isinstance(m, torch.nn.BatchNorm3d):
            m.momentum = bn_momentum

    loss_met_avg, loss_bone_avg, loss_pose_avg, acc_avg, mpjpe_avg, nmpjpe_avg = AverageMeter(), AverageMeter(), \
                                          AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    loss_reg_avg = AverageMeter()
    loss_avg = AverageMeter()

    max_itrs = 1500
    if split == 'test':
        max_itrs = 500

    n_iters_epoch = min(len(data_loader), max_itrs)

    if epoch < 10:
        neg_pose_th = 100.0
    else:
        neg_pose_th = 100.0

    dataset = data_loader.dataset

    mean = torch.from_numpy(dataset.mean_3d).float().view(1, -1)
    std = torch.from_numpy(dataset.std_3d).float().view(1, -1)

    mean_bone_len = torch.from_numpy(dataset.mean_bone_len).float().view(1, -1)
    skeleton_idx = dataset.skeleton_idx
    skeleton_wt = dataset.skeleton_wt
    n_joints = dataset.n_joints

    allowed_subject_list_reg = list(map(lambda x: dataset.subject_map[x], dataset.allowed_subject_list_reg))

    criterion_pose = nn.L1Loss().cuda()
    criterion_bone_len = MeanBoneLenLoss(skeleton_idx, skeleton_wt, mean_bone_len, mean, std)

    beta_1 = 1.0
    beta_2 = 1.0
    id_range = 5.0

    pbar = tqdm(total=n_iters_epoch, ascii=True, ncols=200)
    # torch.cuda.empty_cache()

    for i, (inp_reg, inp_met, tar_3d, tar_3d_un, meta) in enumerate(data_loader):

        if i > n_iters_epoch:
            break

        pbar.update(1)

        batch_size_reg = inp_reg.shape[0] // opt.chunk_size

        batch_size_met = -1
        if opt.no_emb is False:
            batch_size_met = inp_met.shape[0] // opt.chunk_size
            batch_ratio = inp_met.shape[1] // 2

        assert batch_size_reg == batch_size_met or batch_size_met == -1

        if batch_size_reg <= 1 or (opt.no_emb is False and batch_size_met <= 1):
            continue

        # For regression
        subject_reg = meta['subj_reg'].view(batch_size_reg * opt.chunk_size)

        tar_3d_flat = tar_3d.view(batch_size_reg, opt.chunk_size, n_joints*3)
        tar_3d_un = tar_3d_un.view(batch_size_reg, opt.chunk_size, -1, 3)
        tar_3d_un_flat = tar_3d_un.view(batch_size_reg, opt.chunk_size, n_joints*3).numpy()

        inp_reg_ten = inp_reg.permute(0, 3, 1, 2).float()  # .to(torch.device("cuda:0"))

        tar_3d_flat = tar_3d_flat.float().to(torch.device("cuda:0"))

        subject_reg_ten = subject_reg.long()

        # For metric learning
        if opt.no_emb is False:
            inp_met = inp_met.view(batch_size_met * opt.chunk_size * batch_ratio * 2,
                                   inp_met.shape[2], inp_met.shape[3], inp_met.shape[4])

            subject_met = meta['subj_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2)
            unq_id_met = meta['unq_id_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2)
            pose_un_met = meta['pose_un_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2, 16, 3)
            pose_met = meta['pose_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2, 16, 3)
            cam_id_met = meta['cam_id_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2)
            cam_rot_met = meta['cam_rot_met'].view(batch_size_met * opt.chunk_size * batch_ratio * 2, 4)

            # inp_met_ten = inp_met.permute(0, 3, 1, 2).float()  # .to(torch.device("cuda:0"))

            subject_met_ten = subject_met.long()
            cam_id_met_ten = cam_id_met.long()

            # separating views for weak supervision
            inp_a_met = torch.FloatTensor(batch_size_met * opt.chunk_size * batch_ratio,
                                          inp_met.shape[1], inp_met.shape[2], inp_met.shape[3])
            inp_p_met = torch.FloatTensor(batch_size_met * opt.chunk_size * batch_ratio,
                                          inp_met.shape[1], inp_met.shape[2], inp_met.shape[3])
            pose_a_met = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 16, 3)
            pose_p_met = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 16, 3)

            subj_met_a = torch.IntTensor(batch_size_met * batch_ratio, opt.chunk_size)
            subj_met_p = torch.IntTensor(batch_size_met * batch_ratio, opt.chunk_size)

            cam_rot_met_a = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 4)
            cam_rot_met_p = torch.FloatTensor(batch_size_met * batch_ratio, opt.chunk_size, 4)

            chunk_count = 0
            for b in range(0, batch_size_met * 2 * opt.chunk_size * batch_ratio, 2 * opt.chunk_size):
                inp_a_met[chunk_count*opt.chunk_size:(chunk_count+1)*opt.chunk_size] = inp_met[b:b + opt.chunk_size]
                pose_a_met[chunk_count, :, :, :] = pose_met[b:b + opt.chunk_size]
                subj_met_a[chunk_count, :, ] = subject_met[b:b + opt.chunk_size]
                cam_rot_met_a[chunk_count, :, :] = cam_rot_met[b:b + opt.chunk_size]

                inp_p_met[chunk_count*opt.chunk_size:(chunk_count+1)*opt.chunk_size] = inp_met[b + opt.chunk_size:b + 2 * opt.chunk_size]
                pose_p_met[chunk_count, :, :, :] = pose_met[b + opt.chunk_size:b + 2 * opt.chunk_size]
                subj_met_p[chunk_count, :, ] = subject_met[b + opt.chunk_size:b + 2 * opt.chunk_size]
                cam_rot_met_p[chunk_count, :, :] = cam_rot_met[b + opt.chunk_size:b + 2 * opt.chunk_size]

                chunk_count = chunk_count + 1

            assert chunk_count == batch_size_met * batch_ratio

            inp_a_met_ten = inp_a_met.permute(0, 3, 1, 2).contiguous().float()
            inp_p_met_ten = inp_a_met.permute(0, 3, 1, 2).contiguous().float()

        # concatenating inputs
        if opt.no_emb is False:
            inp_ten = torch.cat((inp_reg_ten, inp_a_met_ten, inp_p_met_ten), dim=0)
            # inp_ten = torch.cat((inp_a_met_ten, inp_p_met_ten), dim=0)
        else:
            inp_ten = inp_reg_ten

        inp_ten = inp_ten.to(torch.device("cuda:0"))

        if split == 'train':
            resnet_feat = model['backbone'](inp_ten)

        else:
            with torch.no_grad():
                resnet_feat = model['backbone'](inp_ten)

        n_channel = resnet_feat.shape[1]
        h = resnet_feat.shape[2]
        w = resnet_feat.shape[3]

        if opt.no_emb is False:
            resnet_feat_reg, resnet_feat_a_met, resnet_feat_p_met = torch.split(resnet_feat,
                                                                                [inp_reg_ten.shape[0],
                                                                                 inp_a_met_ten.shape[0],
                                                                                 inp_p_met_ten.shape[0]])

            resnet_feat_reg = resnet_feat_reg.view(batch_size_reg, opt.chunk_size, n_channel, h, w)
            resnet_feat_reg = resnet_feat_reg.permute(0, 2, 1, 3, 4).contiguous()

            # resnet_feat_a_met, resnet_feat_p_met = torch.split(resnet_feat, [inp_a_met_ten.shape[0], inp_p_met_ten.shape[0]])

            resnet_feat_a_met = resnet_feat_a_met.view(batch_size_met * batch_ratio, opt.chunk_size, n_channel, h, w)
            resnet_feat_a_met = resnet_feat_a_met.permute(0, 2, 1, 3, 4).contiguous()

            resnet_feat_p_met = resnet_feat_p_met.view(batch_size_met * batch_ratio, opt.chunk_size, n_channel, h, w)
            resnet_feat_p_met = resnet_feat_p_met.permute(0, 2, 1, 3, 4).contiguous()

            subj_met_a = subj_met_a.view(batch_size_met * batch_ratio, opt.chunk_size)
            subj_met_p = subj_met_p.view(batch_size_met * batch_ratio, opt.chunk_size)
        else:
            resnet_feat_reg = resnet_feat
            resnet_feat_reg = resnet_feat_reg.view(batch_size_reg, opt.chunk_size, n_channel, h, w)
            resnet_feat_reg = resnet_feat_reg.permute(0, 2, 1, 3, 4).contiguous()

        acc = 0
        if opt.no_emb is False and epoch >= opt.sch_emb:
            desp_a_met = model['temporal'](resnet_feat_a_met)
            desp_p_met = model['temporal'](resnet_feat_p_met)

            desp_a_met = desp_a_met.view(batch_size_met * batch_ratio, opt.chunk_size, -1)
            desp_p_met = desp_p_met.view(batch_size_met * batch_ratio, opt.chunk_size, -1)

            loss_met = 0
            loss_reg = 0
            loss_mean_bone = 0
            loss_mean_bone_count = 0
            acc = 0

            n_iters_batch = batch_size_met * batch_ratio

            target_ids = id_range * torch.arange(0, opt.chunk_size).float() / opt.chunk_size
            target_ids = target_ids.view(1, -1)
            target_ids_wgt = target_ids

            for j in range(n_iters_batch):
                assert subj_met_a[j, 0] == subj_met_p[j, 0]

                emb_a_met_chunk = desp_a_met[j]
                emb_p_met_chunk = desp_p_met[j]

                emb_a_met_un_chunk = un_normalise_pose(emb_a_met_chunk, mean, std)
                emb_p_met_un_chunk = un_normalise_pose(emb_p_met_chunk, mean, std)

                emb_a_met_un_chunk = emb_a_met_un_chunk.view(opt.chunk_size, -1, 3)
                emb_p_met_un_chunk = emb_p_met_un_chunk.view(opt.chunk_size, -1, 3)

                pose_a_met_chunk = pose_a_met[j]
                pose_p_met_chunk = pose_p_met[j]

                pose_a_met_chunk = pose_a_met_chunk.view(opt.chunk_size, -1)
                pose_p_met_chunk = pose_p_met_chunk.view(opt.chunk_size, -1)

                pose_a_met_chunk.requires_grad = False
                pose_p_met_chunk.requires_grad = False

                pose_a_met_un_chunk = un_normalise_pose(pose_a_met_chunk, mean, std)
                pose_p_met_un_chunk = un_normalise_pose(pose_p_met_chunk, mean, std)

                pose_a_met_un_chunk = pose_a_met_un_chunk.view(opt.chunk_size, -1, 3)
                pose_p_met_un_chunk = pose_p_met_un_chunk.view(opt.chunk_size, -1, 3)

                cam_rot_chunk_a = cam_rot_met_a[j]
                cam_rot_chunk_p = cam_rot_met_p[j]

                pred_id_mu, pred_id_sig = cal_cyc_mu_sig(emb_a_met_chunk, emb_p_met_chunk,
                                                         emb_a_met_un_chunk, emb_p_met_un_chunk,
                                                         cam_rot_chunk_a, cam_rot_chunk_p, target_ids_wgt, mean, std,
                                                         beta_1, beta_2, opt)

                gt_id_mu, gt_id_sig = cal_cyc_mu_sig(pose_a_met_chunk, pose_p_met_chunk,
                                                     pose_a_met_un_chunk, pose_p_met_un_chunk,
                                                     cam_rot_chunk_a, cam_rot_chunk_p, target_ids_wgt, mean, std,
                                                     1.0, 1.0, opt)

                gt_id_sig = torch.sqrt(gt_id_sig).to(torch.device("cuda:0"))

                # diff_th = (id_range / (2 * opt.chunk_size))

                target = target_ids_wgt
                if pred_id_mu.is_cuda is True:
                    target = target.to(torch.device("cuda:0"))

                diff = torch.abs((pred_id_mu[:, 2:-2] - target[:, 2:-2]))

                mask = diff.detach() > gt_id_sig[:, 2:-2]
                mask = mask.float()

                diff = torch.mul(diff, mask)
                diff = diff ** 2

                # loss_met = loss_met + torch.mean(torch.div(diff, pred_id_sig[:, 2:-2]))
                loss_met = loss_met + torch.mean(diff)
                loss_reg = loss_reg + torch.mean(0.5 * torch.log(pred_id_sig[:, 2:-2]))

                if opt.no_mean_bone is False:
                    loss_mean_bone = loss_mean_bone + 0.5 * (criterion_bone_len(emb_a_met_un_chunk) +
                                                             criterion_bone_len(emb_p_met_un_chunk))

            loss_met = loss_met / n_iters_batch
            loss_reg = loss_reg / n_iters_batch
            loss_mean_bone = loss_mean_bone / n_iters_batch

            if torch.isnan(loss_met) is True or torch.isnan(loss_reg) or torch.isnan(loss_mean_bone):
                import ipdb
                ipdb.set_trace()

            loss_met_avg.update(loss_met.item())
            loss_bone_avg.update(loss_mean_bone.item())
            loss_reg_avg.update(loss_reg.item())
        else:
            loss_met = torch.FloatTensor(1).fill_(0).to(torch.device("cuda:0"))
            loss_reg = torch.FloatTensor(1).fill_(0).to(torch.device("cuda:0"))
            loss_mean_bone = torch.FloatTensor(1).fill_(0).to(torch.device("cuda:0"))

        if opt.no_pose is False and epoch >= opt.sch_pose:

            desp_reg = model['temporal'](resnet_feat_reg)
            loss_pose = 0.
            mpjpe = 0.
            nmpjpe = 0.

            for j in range(batch_size_reg):
                pred_3d_reg_flat = desp_reg[j].view(opt.chunk_size, -1)

                loss_pose = loss_pose + criterion_pose(pred_3d_reg_flat, tar_3d_flat[j])

                pred_3d_reg_un = un_normalise_pose(pred_3d_reg_flat.detach().cpu(), mean, std)

                mpjpe = mpjpe + cal_avg_l2_jnt_dist(pred_3d_reg_un.numpy(), tar_3d_un[j].numpy())

                pred_reg_scaled = scale_norm_pose(pred_3d_reg_un, tar_3d_un[j].float())

                nmpjpe = nmpjpe + cal_avg_l2_jnt_dist(pred_reg_scaled.numpy(), tar_3d_un[j].numpy())

            loss_pose = loss_pose / batch_size_reg
            loss_pose_avg.update(loss_pose.item())

            mpjpe = mpjpe / batch_size_reg
            nmpjpe = nmpjpe / batch_size_reg

        else:
            loss_pose = torch.FloatTensor(1).fill_(0).to(torch.device("cuda:0"))
            mpjpe = 0.
            nmpjpe = 0.

        if loss_met.is_cuda is False:
            loss_pose = loss_pose.cpu()

        loss = opt.emb_wt * loss_met + opt.bone_wt * loss_mean_bone + opt.pose_wt * loss_pose
        #
        #
        # acc_avg.update(acc)
        mpjpe_avg.update(mpjpe)
        nmpjpe_avg.update(nmpjpe)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model['backbone'].parameters(), max_norm=1.)
            # torch.nn.utils.clip_grad_norm_(model['temporal'].parameters(), max_norm=1.)
            optimizer.step()
            # classifier_optimizer.zero_grad()

        pbar_suffix = 'Ep {}: [{}]| L MET {:.2f} | L REG {:.2f} | L BONE {:.3f} | AC: {:.3f} | L POSE {:.2f} | ' \
                      '| MPJPE {:.2f} | NMPJPE {:.2f} )'.format(split, epoch, loss_met_avg.avg, loss_reg_avg.avg, loss_bone_avg.avg,
                                                                acc_avg.avg, loss_pose_avg.avg, mpjpe_avg.avg, nmpjpe_avg.avg)
        pbar.set_description(pbar_suffix)

        # if split == 'train':
        #     import ipdb
        #     ipdb.set_trace()

    pbar.close()

    results = dict()
    results['loss_met'] = loss_met_avg.avg
    results['acc'] = -acc_avg.avg
    results['mpjpe'] = mpjpe_avg.avg
    results['nmpjpe'] = nmpjpe_avg.avg

    return results
