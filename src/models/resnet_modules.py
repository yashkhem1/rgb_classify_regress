import torch.nn as nn
import math
import torch
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

set_affine = False


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=set_affine)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=set_affine)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, bn_aff=False):
        if bn_aff is True:
            global set_affine
            set_affine = True
            print('BN Affine set True')

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=set_affine)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, mean=0, std=0.001)
                pass

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=set_affine),  # change here to turn affine false
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


'''
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
#    model = ResNet(Bottleneck, [3, 4, 6, 1], **kwargs)
    model = ResNetTwoStream(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading image net weights...")
        copied_param_names, fixed_param_names = transfer_partial_weights(model_zoo.load_url(model_urls['resnet50']), model)
        print("Done image net weights...")
        return model, copied_param_names, fixed_param_names
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model,[],[]
'''


# def transfer_partial_weights(state_dict_other, obj, submodule=0, prefix=None, add_prefix=''):
#     print('Transferring weights...')
#
#     if 0:
#         print('\nStates source\n')
#         for name, param in state_dict_other.items():
#             print(name)
#         print('\nStates target\n')
#         for name, param in obj.state_dict().items():
#             print(name)
#
#     own_state = obj.state_dict()
#     copyCount = 0
#     skipCount = 0
#     paramCount = len(own_state)
#     # for name_raw, param in own_state.items():
#     #    paramCount += param.view(-1).size()[0]
#     # for name_raw, param in state_dict_other.items():
#     #    print("param",param)
#     copied_param_names = []
#     skipped_param_names = []
#     for name_raw, param in state_dict_other.items():
#         if isinstance(param, torch.nn.Parameter):
#             # backwards compatibility for serialized parameters
#             param = param.data
#             # print('.data conversion for ',name)
#         if prefix is not None and not name_raw.startswith(prefix):
#             # print("skipping {} because of prefix {}".format(name_raw, prefix))
#             continue
#
#         # remove the path of the submodule from which we load
#         name = add_prefix + ".".join(name_raw.split('.')[submodule:])
#
#         if name in own_state:
#             if hasattr(own_state[name], 'copy_'):  # isinstance(own_state[name], torch.Tensor):
#                 # print('copy_ ',name)
#                 if own_state[name].size() == param.size():
#                     own_state[name].copy_(param)
#                     copyCount += 1
#                     copied_param_names.append(name)
#                 else:
#                     print('Invalid param size(own={} vs. source={}), skipping {}'.format(own_state[name].size(),
#                                                                                          param.size(), name))
#                     skipCount += 1
#                     skipped_param_names.append(name)
#
#             elif hasattr(own_state[name], 'copy'):
#                 own_state[name] = param.copy()
#                 copyCount += 1
#                 copied_param_names.append(name)
#             else:
#                 print('training.utils: Warning, unhandled element type for name={}, name_raw={}'.format(name, name_raw))
#                 print(type(own_state[name]))
#                 skipCount += 1
#                 skipped_param_names.append(name)
#                 IPython.embed()
#         else:
#             skipCount += 1
#             print('Warning, no match for {}, ignoring'.format(name))
#             skipped_param_names.append(name)
#             # print(' since own_state.keys() = ',own_state.keys())
#
#     print('Copied {} elements, {} skipped, and {} target params without source'.format(copyCount, skipCount,
#                                                                                        paramCount - copyCount))
#     return copied_param_names, skipped_param_names


# def copy_weights(model_pretrained, model):
#     pretrained_dict = model_pretrained.state_dict()
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(pretrained_dict)


resnet_spec = {'resnet18': (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 18),
               'resnet34': (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 34),
               'resnet50': (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 50),
               'resnet101': (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 101),
               'resnet152': (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 152)}
