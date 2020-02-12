import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
from common.utils import AverageMeter
from common.error_metrics import calculate_correct_mask
from common.error_metrics import un_normalise_pose, cal_avg_l2_jnt_dist, scale_norm_pose, normalise_pose


def run_epoch(epoch, opt, data_loader, model, optimizer=None, split='train'):

    if split == 'train':
        model['backbone'].train()
        model['classifier'].train()
        # model['pose'].train()
        bn_momentum = opt.bn_momentum
        assert optimizer is not None
        # if epoch == 1:
        data_loader.dataset.init_epoch(split)
    else:
        if epoch == 1:
            data_loader.dataset.init_epoch(split)
        model['backbone'].eval()
        model['classifier'].eval()
        # model['pose'].eval()
        bn_momentum = 0.0

    for _, m in model['backbone'].named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = bn_momentum
        if isinstance(m, torch.nn.BatchNorm3d):
            m.momentum = bn_momentum

    for _, m in model['classifier'].named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = bn_momentum
        if isinstance(m, torch.nn.BatchNorm3d):
            m.momentum = bn_momentum

    loss_class, acc, acc_x, acc_y, acc_z = AverageMeter(), AverageMeter(), AverageMeter(), \
                                            AverageMeter(), AverageMeter()

    max_itrs = 2500
    if split == 'test':
        max_itrs = 1500

    n_iters_epoch = min(len(data_loader), max_itrs)  # TODO: Ask rahul here

    dataset = data_loader.dataset

    mean = torch.from_numpy(dataset.mean_3d).float().view(1, -1)
    std = torch.from_numpy(dataset.std_3d).float().view(1, -1)

    n_joints = dataset.n_joints

    # criterion_pose = nn.L1Loss(reduction='mean').to(torch.device("cuda:0"))
    criterion_class_x = nn.CrossEntropyLoss(reduction='mean').to(torch.device("cuda:0"))
    criterion_class_y = nn.CrossEntropyLoss(reduction='mean').to(torch.device("cuda:0"))
    criterion_class_z = nn.CrossEntropyLoss(reduction='mean').to(torch.device("cuda:0"))

    pbar = tqdm(total=n_iters_epoch, ascii=True, ncols=200)
    # torch.cuda.empty_cache()

    for i, (inp, tar_x, tar_y, tar_z, meta) in enumerate(data_loader):

        if i > n_iters_epoch:
            break

        pbar.update(1)

        batch_size_reg = inp.shape[0]

        # For regression
        # subject_reg = meta['subj_reg'].view(batch_size_reg * opt.chunk_size)

        # convert to channels first format
        inp_ten = inp.permute(0, 3, 1, 2).float()

        inp_ten = inp_ten.to(torch.device("cuda:0"))

        if split == 'train':
            resnet_feat = model['backbone'](inp_ten)
        else:
            with torch.no_grad():
                resnet_feat = model['backbone'](inp_ten)

        n_channel = resnet_feat.shape[1]
        h = resnet_feat.shape[2]
        w = resnet_feat.shape[3]

        x_out, y_out, z_out = model['classifier'](resnet_feat)

        x_out = x_out.view(-1, opt.n_bins_x)
        y_out = y_out.view(-1, opt.n_bins_y)
        z_out = z_out.view(-1, opt.n_bins_z)

        tar_x = tar_x.view(-1).long().to(torch.device("cuda:0"))
        tar_y = tar_y.view(-1).long().to(torch.device("cuda:0"))
        tar_z = tar_z.view(-1).long().to(torch.device("cuda:0"))

        loss_class_x = criterion_class_x(x_out, tar_x)
        loss_class_y = criterion_class_y(y_out, tar_y)
        loss_class_z = criterion_class_z(z_out, tar_z)

        loss = loss_class_x + loss_class_y

        if opt.only_xy is False:
            loss = loss + loss_class_z

        loss_class.update(loss.item())

        x_out_cpu = x_out.detach().cpu()
        y_out_cpu = y_out.detach().cpu()
        z_out_cpu = z_out.detach().cpu()

        tar_x_cpu = tar_x.detach().cpu()
        tar_y_cpu = tar_y.detach().cpu()
        tar_z_cpu = tar_z.detach().cpu()

        correct_x_batch = calculate_correct_mask(x_out_cpu, tar_x_cpu)
        correct_y_batch = calculate_correct_mask(y_out_cpu, tar_y_cpu)

        acc_x.update(correct_x_batch.float().mean().item())
        acc_y.update(correct_y_batch.float().mean().item())

        if opt.only_xy is True:
            acc_batch = torch.mul(correct_x_batch, correct_y_batch).float().mean()
        else:
            correct_z_batch = correct_x_batch = calculate_correct_mask(z_out_cpu, tar_z_cpu)
            acc_z.update(correct_z_batch.float().mean().item())

            acc_batch = torch.mul((torch.mul(correct_x_batch, correct_y_batch), correct_z_batch)).float().mean()

        acc.update(acc_batch.item())

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model['backbone'].parameters(), max_norm=1.)
            # torch.nn.utils.clip_grad_norm_(model['temporal'].parameters(), max_norm=1.)
            optimizer.step()
            # classifier_optimizer.zero_grad()

        pbar_suffix = 'Ep {}: [{}]| L CLASS {:.3f} ' \
                      '| ACC {:.3f} | ACC X {:.3f} | ACC Y {:.3f} | ACC Z {:.3f})'.format(split, epoch, loss_class.avg,
                                                                                          acc.avg, acc_x.avg, acc_y.avg,
                                                                                          acc_z.avg)
        pbar.set_description(pbar_suffix)

        # if split == 'train':
        #     import ipdb
        #     ipdb.set_trace()

    pbar.close()

    results = dict()
    results['loss_class'] = loss_class.avg
    results['acc'] = -acc.avg
    results['acc_x'] = -acc_x.avg
    results['acc_y'] = -acc_y.avg
    results['acc_z'] = -acc_z.avg

    return results
