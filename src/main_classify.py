import sys
sys.path.append('./')

import torch
import torch.utils.data
import numpy as np
from opts import opts
from models.models_list import BackBone, PoseClassifier
from data_loader.h36m_classify import H36M
from common.logger import Logger
# from utils.utils import adjust_learning_rate

# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    opt = opts().parse()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    from training.train_test_classify import run_epoch

    train_dataset = H36M(opt, split='train',
                         train_stats={},
                         allowed_subj_list=opt.sub_list_reg,
                         )

    test_dataset = H36M(opt, split='test',
                        train_stats={'mean_3d': train_dataset.mean_3d,
                                     'std_3d': train_dataset.std_3d,
                                    },
                        allowed_subj_list=[9, 11])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
        # worker_init_fn=worker_init_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.train_batch,
        shuffle=False,
        num_workers=1,
        # worker_init_fn=worker_init_fn
    )

    opt.n_bins_x = train_dataset.n_bins_x
    opt.n_bins_y = train_dataset.n_bins_y
    opt.n_bins_z = train_dataset.n_bins_z
    opt.n_joints = train_dataset.n_joints

    model = dict()
    model['backbone'] = BackBone(opt, spatial_size=2)
    model['classifier'] = PoseClassifier(opt, in_feat=model['backbone'].out_feats, h=model['backbone'].out_feat_h)

    opt.bn_momentum = 0.1
    opt.bn_decay = 0.9

    model_dict=None

    if opt.load_model != 'none':
        model_dict = torch.load(opt.load_model)
        model['backbone'].load_state_dict(model_dict['backbone'])
        model['classifier'].load_state_dict(model_dict['pose'])

    if opt.data_par is True:
        print('Using data parallel')
        model['backbone'] = torch.nn.DataParallel(model['backbone'])

    model['backbone'].to(torch.device("cuda:0"))   # change device here if we want
    model['classifier'].to(torch.device("cuda:0"))

    if opt.test is True:
        run_epoch(1, opt, test_loader, model, optimizer=None, split='test')

        exit(0)

    optimizer = torch.optim.Adam(list(model['backbone'].parameters()) + list(model['classifier'].parameters()),
                                 opt.lr,
                                 betas=(0.9, 0.999), #can use beta from opt
                                 weight_decay=0.00,
                                 amsgrad=False,
                                 )

    if opt.resume is True:
        assert model_dict is not None
        optimizer.load_state_dict(model_dict['optimiser_emb'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    logger = Logger(opt.save_dir + '/logs')

    opt_metric_val = 9999.0

    opt.global_step = 0

    for epoch in range(1, opt.n_epochs + 1):
        loss_train = 0
        nmpjpe_train = 0
        mpjpe_train = 0

        result_train = run_epoch(epoch, opt, train_loader, model, optimizer=optimizer,
                                 split='train')

        if epoch % opt.val_intervals == 0:
            result_val = run_epoch(epoch, opt, test_loader, model, optimizer=None,
                                   split='test')

            logger.write('LTr {:.3f}  AccTr {:.2f} AccXTr {:.2f} AccYTr {:.2f} AccZTr {:.2f} LVal {:.5f} '
                         'AccVal {:.2f} AccXVal {:.2f} AccYVal {:.2f} AccZVal {:.2f}\n'.format(result_train['loss_class'],
                        -result_train['acc'], -result_train['acc_x'], -result_train['acc_y'], -result_train['acc_z'],
                        result_val['loss_class'], -result_val['acc'],
                        -result_val['acc_x'], -result_val['acc_y'], -result_val['acc_z']))

            print('Saving last epoch model')
            save_last_model = dict()
            if opt.data_par is True:
                save_last_model['backbone'] = model['backbone'].module.state_dict()
            else:
                save_last_model['backbone'] = model['backbone'].state_dict()

            save_last_model['classifier'] = model['classifier'].state_dict()
            save_last_model['tr_stat'] = train_dataset.train_stats
            save_last_model['optimiser'] = optimizer.state_dict()

            torch.save(save_last_model, opt.save_dir + '/model_last.pth')

            metric_val = result_val[opt.e_metric]  # Metric value from opt

            if opt_metric_val > metric_val:
                print('Saving model best model')
                logger.write('Saving best model\n')

                save_best_model = dict()
                if opt.data_par is True:
                    save_best_model['backbone'] = model['backbone'].module.state_dict()
                else:
                    save_best_model['backbone'] = model['backbone'].state_dict()

                save_best_model['classifier'] = model['classifier'].state_dict()
                save_best_model['tr_stat'] = train_dataset.train_stats

                torch.save(save_best_model, opt.save_dir + '/model_best.pth')
                opt_metric_val = metric_val
        else:
            logger.write('LTr {:.3f}  AccTr {:.2f} AccXTr {:.2f} AccYTr {:.2f} AccZTr {:.2f}\n'.format(
                result_train['loss_class'], -result_train['acc'], -result_train['acc_x'], -result_train['acc_y'],
                -result_train['acc_z']))

        opt.global_step = opt.global_step + 1
        if opt.global_step % opt.lr_step == 0:
            print('Applying LR Decay')
            scheduler.step()

        # opt.bn_momentum = opt.bn_momentum * opt.bn_decay

    logger.close()


if __name__ == '__main__':
    main()
