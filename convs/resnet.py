import torch
import torch.nn as nn

from rational.torch import Rational

from .modules import get_child_dict, Module, Conv2d, Linear, BatchNorm2d, Sequential


__all__ = ['resnet18']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.ratl_1 = Rational(approx_func='relu', cuda=False)
        self.ratl_2 = Rational(approx_func='relu', cuda=False)

    def forward(self, x, params=None, episode=None, activation='relu'):
        identity = x
        
        out = self.conv1(x, get_child_dict(params, 'conv1'))
        out = self.bn1(out, get_child_dict(params, 'bn1'), episode)
        out = self.relu(out) if activation == 'relu' else self.ratl_1(out)

        out = self.conv2(out, get_child_dict(params, 'conv2'))
        out = self.bn2(out, get_child_dict(params, 'bn2'), episode)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out) if activation == 'relu' else self.ratl_2(out)

        return out


class ResNet18(Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.ratl = Rational(approx_func='relu', cuda=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer_1_1 = BasicBlock(inplanes=64, planes=64, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        self.layer_1_2 = BasicBlock(inplanes=64, planes=64, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        
        downsample = Sequential(conv1x1(64, 128, 2), BatchNorm2d(128))
        self.layer_2_1 = BasicBlock(inplanes=64, planes=128, stride=2, downsample=downsample, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        self.layer_2_2 = BasicBlock(inplanes=128, planes=128, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        
        downsample = Sequential(conv1x1(128, 256, 2), BatchNorm2d(256))
        self.layer_3_1 = BasicBlock(inplanes=128, planes=256, stride=2, downsample=downsample, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        self.layer_3_2 = BasicBlock(inplanes=256, planes=256, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)

        downsample = Sequential(conv1x1(256, 512, 2), BatchNorm2d(512))
        self.layer_4_1 = BasicBlock(inplanes=256, planes=512, stride=2, downsample=downsample, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        self.layer_4_2 = BasicBlock(inplanes=512, planes=512, groups=1, base_width=64, dilation=1, norm_layer=BatchNorm2d)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * BasicBlock.expansion

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x, params=None, activation='relu'):
        x = self.conv1(x, get_child_dict(params, 'conv1'))
        x = self.bn1(x, get_child_dict(params, 'bn1'))
        x = self.relu(x) if activation == 'relu' else self.ratl(x)
        x = self.maxpool(x)

        x_1 = self.layer_1_1(x, get_child_dict(params, 'layer_1_1'), activation=activation)
        x_1 = self.layer_1_2(x_1, get_child_dict(params, 'layer_1_2'), activation=activation)
        x_2 = self.layer_2_1(x_1, get_child_dict(params, 'layer_2_1'), activation=activation)
        x_2 = self.layer_2_2(x_2, get_child_dict(params, 'layer_2_2'), activation=activation)
        x_3 = self.layer_3_1(x_2, get_child_dict(params, 'layer_3_1'), activation=activation)
        x_3 = self.layer_3_2(x_3, get_child_dict(params, 'layer_3_2'), activation=activation)
        x_4 = self.layer_4_1(x_3, get_child_dict(params, 'layer_4_1'), activation=activation)
        x_4 = self.layer_4_2(x_4, get_child_dict(params, 'layer_4_2'), activation=activation)

        pooled = self.avgpool(x_4)
        features = torch.flatten(pooled, 1)

        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }

    def forward(self, x, params=None, activation='relu'):
        assert activation in ['relu', 'ratl']
        return self._forward_impl(x, params, activation)

    @property
    def last_conv(self):
        return self.layer4_2.conv2


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet18()
