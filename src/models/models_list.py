import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from easydict import EasyDict as edict
from models.resnet_modules import model_urls, resnet_spec, ResNet


def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 34
    config.num_deconv_layers = 3
    config.num_deconv_filters = 256
    config.num_deconv_kernel = 4
    config.final_conv_kernel = 1
    config.depth_dim = 1
    config.input_channel = 3
    return config


def resnet_base(model_type='resnet18', bn_aff=False, pretrained=True, **kwargs):

    """Constructs a ResNet-34 model.

    Args:
        model_type: resnet model type
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        bn_aff: Turns on BN for RenNet if True
    """
    print('Using backend {}'.format(model_type))

    block_type, layers, _, _ = resnet_spec[model_type]

    model = ResNet(block_type, layers, bn_aff, **kwargs)
    print('Resnet model created')

    if pretrained:
        org_resnet = model_zoo.load_url(model_urls[model_type])

        # if 'fc.weight' in org_resnet:
        #     org_resnet.pop('fc.weight', None)
        #
        # if 'fc.bias' in org_resnet:
        #     org_resnet.pop('fc.bias', None)
        #
        # transfer_partial_weights(org_resnet, model)
        model.load_state_dict(org_resnet, strict=False)
        org_resnet = None
        print('Initialised with ImageNet weights')
    return model


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class BackBone(nn.Module):
    def __init__(self, opt, spatial_size=2):

        super(BackBone, self).__init__()
        self.opt = opt
        self.inp_norm = self.opt.inp_norm

        self.resnet_backbone = resnet_base(model_type=self.opt.arch, bn_aff=self.opt.bn_aff)

        for param in self.resnet_backbone.parameters():
            param.requires_grad = True

        _, _, n_feats, _ = resnet_spec[self.opt.arch]
        self.out_feats = n_feats[-1]

        if self.inp_norm is True:
            print('Normalising Input')

        cfg = get_default_network_config()

        self.spatial_size = spatial_size

        self.pool_1 = nn.AvgPool2d(kernel_size=self.spatial_size)

        self.out_feat_h = 7 // self.spatial_size

        # self.emb_encoder = nn.Sequential(nn.Conv2d(in_channels=out_feats, out_channels=126,
        #                                            kernel_size=spatial_size, bias=False),
        #                                  nn.BatchNorm2d(126),
        #                                  )

        # self.emb_encoder.apply(weight_init)
        self.eps = 1e-10

    def fix_weights(self, lock, lock2=False):
        for param in self.resnet_backbone.conv1.parameters():
            param.requires_grad = not lock
        for param in self.resnet_backbone.bn1.parameters():
            param.requires_grad = not lock
        for param in self.resnet_backbone.layer1.parameters():
            param.requires_grad = not lock
        for param in self.resnet_backbone.layer2.parameters():
            param.requires_grad = not lock
        for param in self.resnet_backbone.layer3.parameters():
            param.requires_grad = not lock

    def input_norm(self, x):
        batch_size = x.size(0)
        n_channel = x.size(1)
        flat = x.view(batch_size, -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7

        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).expand_as(x)

    def forward(self, x):
        if self.inp_norm is True:
            x = self.input_norm(x)

        resnet_out = self.resnet_backbone(x)

        resnet_out_down = self.pool_1(resnet_out)

        return resnet_out_down


class PoseRegressor(nn.Module):
    def __init__(self, opt, in_feat=512, h=1):
        super(PoseRegressor, self).__init__()

        conv_kernels = [h, 1]

        self.out_feats = [512, opt.desp_dim]

        spatial_dim = h

        self.conv_1 = nn.Conv2d(in_channels=in_feat, out_channels=self.out_feats[0], kernel_size=conv_kernels[0],
                               padding=(0, 0), bias=False)

        spatial_dim = spatial_dim - (conv_kernels[0]-1)

        self.batch_norm = nn.BatchNorm2d(self.out_feats[0], affine=False)

        self.fcn = nn.Conv2d(in_channels=self.out_feats[0], out_channels=self.out_feats[1], kernel_size=1, bias=False)

    def forward(self, x):

        batch_size = x.shape[0]
        in_feat = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]

        y = self.conv_1(x)
        y = self.batch_norm(y)

        y = nn.functional.relu(y)  #Why not relu(y)?

        # y = self.conv_2(y)
        #
        # y = self.pool1(y)

        y = self.fcn(y)

        return y
