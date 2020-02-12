import argparse
import os


class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
  def init(self):
    self.parser.add_argument('--exp_id', default='default', help='Experiment ID')
    self.parser.add_argument('--test', action='store_true', help='test')
    self.parser.add_argument('--data_dir', default='../data/annot_h36m_cam_reg/', help='data directory')
    self.parser.add_argument('--img_dir', default='../data/Human3.6/images', help='image directory')
    self.parser.add_argument('--bin_info_pre', default='annot_bins_sc_norm_True_10_10_10', help='bin info file prefix')

    # check if image directory present

    self.parser.add_argument('--load_model', default='none', help='Provide full path to a previously trained model')
    self.parser.add_argument('--resume', action='store_true', help='resume training')

    self.parser.add_argument('--lr', type=float, default=1.0e-4, help='Learning Rate')
    self.parser.add_argument('--lr_step', type=int, default=15, help='drop LR')
    self.parser.add_argument('--n_epochs', type=int, default=25, help='#training epochs')
    self.parser.add_argument('--val_intervals', type=int, default=1, help='#valid intervel')
    self.parser.add_argument('--train_batch', type=int, default=64, help='Mini-batch size')
    # self.parser.add_argument('--chunk_size', type=int, default=20, help='Mini-batch size')
    self.parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    self.parser.add_argument('--sub_list_reg', default='[1,5,6,7,8]', help='supervision sub list')
    # self.parser.add_argument('--sub_list_met', default='[1,5,6,7,8]', help='metric sub list')
    # self.parser.add_argument('--d_frac', type=float, default=1.0, help='fraction of supervision')
    self.parser.add_argument('--inp_img_size', type=float, default=224.0, help='fraction of supervision')

    # self.parser.add_argument('--beta', type=float, default=5.0, help='temperature for softmax')
    # self.parser.add_argument('--reg_wt', type=float, default=0.1, help='weight for regulariser')
    # self.parser.add_argument('--emb_wt', type=float, default=0.1, help='weight for metric learning')
    # self.parser.add_argument('--bone_wt', type=float, default=0.1, help='weight for bone ratio loss')
    # self.parser.add_argument('--pose_wt', type=float, default=1.0, help='weight for pose loss')

    self.parser.add_argument('--arch', default='resnet18', help='resnet18 | resnet34 | ...')
    # self.parser.add_argument('--use_temporal', action='store_true', help='use temporal info')

    self.parser.add_argument('--num_output', type=int, default=16, help='num joints')
    # self.parser.add_argument('--desp_dim', type=int, default=48, help='descriptor dimension')
    self.parser.add_argument('--bn_aff', action='store_true', help='turns on bn affine params for resnet')
    self.parser.add_argument('--reg_bias', action='store_true', help='uses bias at regression')
    self.parser.add_argument('--res_norm', action='store_true', help='uses imagenet data norm for inp norm')
    self.parser.add_argument('--inp_norm', action='store_true', help='turns on inp normalization')
    self.parser.add_argument('--data_par', action='store_true', help='uses data parallel')

    self.parser.add_argument('--only_xy', action='store_true', help='uses only x and y dimensions')

    # self.parser.add_argument('--no_emb', action='store_true', help='disables embedding loss')
    # self.parser.add_argument('--no_pose', action='store_true', help='disables pose loss')
    # self.parser.add_argument('--no_mean_bone', action='store_true', help='disables mean bone len loss')
    #
    # self.parser.add_argument('--sch_emb', type=int, default=1, help='schedule for emb loss')
    # self.parser.add_argument('--sch_pose', type=int, default=1, help='schedule for emb loss')

    self.parser.add_argument('--e_metric', default='nmpjpe', help='eval model metric')

  def parse(self):
    self.init()
    self.opt = self.parser.parse_args()
    self.opt.save_dir = os.path.join('..','exp', self.opt.exp_id)
    self.opt.data_dir = self.opt.data_dir
    # self.opt.downSample = list(map(int,self.opt.downSample.strip('[]').split(',')))
    self.opt.sub_list_reg = list(map(int, self.opt.sub_list_reg.strip('[]').split(',')))
    # self.opt.sub_list_met = list(map(int, self.opt.sub_list_met.strip('[]').split(',')))
    # self.opt.camPairs = [list(map(int,x.strip(',(').split(',')))
    # for x in self.opt.camPairs.strip(')]').strip('[').split(')')]

    args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                if not name.startswith('_'))
    # refs = dict((name, getattr(ref, name)) for name in dir(ref)
    #             if not name.startswith('_'))

    if self.opt.test is True:
      assert self.opt.load_model!='none'

    if self.opt.test is False:
      if not os.path.exists(self.opt.save_dir):
        os.makedirs(self.opt.save_dir)

      if not os.path.exists(os.path. join(self.opt.save_dir, 'visualize')):
        os.makedirs(os.path.join(self.opt.save_dir, 'visualize'))

      file_name = os.path.join(self.opt.save_dir, 'opt.txt')
      with open(file_name, 'wt') as opt_file:
        opt_file.write('==> Args:\n')
        for k, v in sorted(args.items()):
          opt_file.write('  %s: %s\n' % (str(k), str(v)))
        opt_file.write('==> Args:\n')
        # for k, v in sorted(refs.items()):
        #   opt_file.write('  %s: %s\n' % (str(k), str(v)))

    return self.opt
