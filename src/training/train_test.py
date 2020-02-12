import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
from common.utils import AverageMeter
from common.error_metrics import un_normalise_pose, cal_avg_l2_jnt_dist, scale_norm_pose, normalise_pose


def run_epoch(epoch, opt, data_loader, model, optimizer=None, split='train'):

    if split == 'train':
        model['backbone'].train()
        model['pose'].train()
        # model['pose'].train()
        bn_momentum = opt.bn_momentum
        assert optimizer is not None
        # if epoch == 1:
        data_loader.dataset.init_epoch(split)
    else:
        if epoch == 1:
            data_loader.dataset.init_epoch(split)
        model['backbone'].train()
        model['pose'].train()
        # model['pose'].eval()
        bn_momentum = 0.0

    for _, m in model['backbone'].named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = bn_momentum
        if isinstance(m, torch.nn.BatchNorm3d):
            m.momentum = bn_momentum

    for _, m in model['pose'].named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = bn_momentum
        if isinstance(m, torch.nn.BatchNorm3d):
            m.momentum = bn_momentum

    loss_l1_avg,  mpjpe_avg, nmpjpe_avg = AverageMeter(), AverageMeter(), AverageMeter()

    max_itrs = 2500
    if split == 'test':
        max_itrs = 1500

    n_iters_epoch = min(len(data_loader), max_itrs)  #TODO: Ask rahul here

    dataset = data_loader.dataset

    mean = torch.from_numpy(dataset.mean_3d).float().view(1, -1)
    std = torch.from_numpy(dataset.std_3d).float().view(1, -1)

    n_joints = dataset.n_joints

    criterion_pose = nn.L1Loss(reduction='mean').to(torch.device("cuda:0"))

    pbar = tqdm(total=n_iters_epoch, ascii=True, ncols=200)
    # torch.cuda.empty_cache()

    for i, (inp_reg, tar_3d, tar_3d_un, meta) in enumerate(data_loader):

        if i > n_iters_epoch:
            break

        pbar.update(1)

        batch_size_reg = inp_reg.shape[0]

        # For regression
        # subject_reg = meta['subj_reg'].view(batch_size_reg * opt.chunk_size)

        tar_3d_flat = tar_3d.view(batch_size_reg, n_joints*3)
        tar_3d_un = tar_3d_un.view(batch_size_reg, -1, 3)
        tar_3d_un_flat = tar_3d_un.view(batch_size_reg, n_joints*3).numpy()

        # convert to channels first format
        inp_reg_ten = inp_reg.permute(0, 3, 1, 2).float()

        tar_3d_flat = tar_3d_flat.float().to(torch.device("cuda:0"))

        # subject_reg_ten = subject_reg.long()

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

        resnet_feat_reg = resnet_feat

        acc = 0
        desp_reg = model['pose'](resnet_feat_reg)

        # loss_l1 = 0.
        # mpjpe = 0.
        # nmpjpe = 0.

        pred_3d_reg_flat = desp_reg.view(batch_size_reg, -1)

        loss_l1 = criterion_pose(pred_3d_reg_flat, tar_3d_flat)

        pred_3d_reg_un = un_normalise_pose(pred_3d_reg_flat.detach().cpu(), mean, std)

        mpjpe = cal_avg_l2_jnt_dist(pred_3d_reg_un.numpy(), tar_3d_un.numpy(), avg=True)

        pred_reg_scaled = scale_norm_pose(pred_3d_reg_un, tar_3d_un.float())

        nmpjpe = cal_avg_l2_jnt_dist(pred_reg_scaled.numpy(), tar_3d_un.numpy(), avg=True)

        loss_l1_avg.update(loss_l1.item())

        loss = loss_l1
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

        pbar_suffix = 'Ep {}: [{}]| L L1 {:.4f} ' \
                      '| MPJPE {:.2f} | NMPJPE {:.2f} )'.format(split, epoch, loss_l1_avg.avg, mpjpe_avg.avg, nmpjpe_avg.avg)
        pbar.set_description(pbar_suffix)

        # if split == 'train':
        #     import ipdb
        #     ipdb.set_trace()

    pbar.close()

    results = dict()
    results['loss_l1'] = loss_l1_avg.avg
    results['mpjpe'] = mpjpe_avg.avg
    results['nmpjpe'] = nmpjpe_avg.avg

    return results
