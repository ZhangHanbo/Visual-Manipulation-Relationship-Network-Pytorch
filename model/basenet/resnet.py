from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from .feat_extractor import featExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

from collections import OrderedDict

from model.utils.net_utils import set_bn_fix, set_bn_eval


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
             'resnet152']

# TODO: write auto-downloading code
model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
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


class ResNet(featExtractor):
    def __init__(self, block, layers, num_classes=1000, feat_list = ("conv4",),
                 pretrained_model_path = None):
        self.inplanes = 64
        self.expansion = block.expansion
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # it is slightly better whereas slower to set stride = 1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained_model_path is not None:
            print("loading pretrained model: " + pretrained_model_path)

            state_dict = torch.load(pretrained_model_path)
            # self.load_state_dict({k:v for k,v in state_dict.items() if k in self.state_dict()})
            self.load_state_dict(state_dict)

        self.feat_list = feat_list

        self._init_modules()

        self.feat_layer["conv1"] = [self.conv1, self.bn1, self.relu]
        self.feat_layer["maxpool"] = self.maxpool
        self.feat_layer["conv2"] = self.layer1
        self.feat_layer["conv3"] = self.layer2
        self.feat_layer["conv4"] = self.layer3
        self.feat_layer["conv5"] = self.layer4
        self.feat_layer["fc"] = self.avgpool
        self.feat_layer["cscore"] = self.fc

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = []

        for m_name, m in self.feat_layer.items():
            if isinstance(m, list):
                for l in m:
                    x = l(x)
            else:
                x = m(x)
            if m_name == "fc":
                x = x.view(x.size(0), -1)
            if m_name in self.feat_list:
                feats.append(x)
            if m_name == self.feat_list[-1]:
                break
        if len(feats) == 1:
            feats = feats[0]
        return feats

    def _init_modules(self):

        # Fix blocks
        for p in self.conv1.parameters(): p.requires_grad = False
        for p in self.bn1.parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.layer3.parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.layer2.parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.layer1.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def train(self, mode = True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:

            # Set fixed blocks to be in eval mode
            self.conv1.eval()
            self.bn1.eval()
            self.relu.eval()
            self.maxpool.eval()

            if cfg.RESNET.FIXED_BLOCKS >= 1:
                self.layer1.eval()
            if cfg.RESNET.FIXED_BLOCKS >= 2:
                self.layer2.eval()
            if cfg.RESNET.FIXED_BLOCKS >= 3:
                self.layer3.eval()

            self.apply(set_bn_eval)

def resnet18(feat_list, pretrained_model_path):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], feat_list=feat_list, pretrained_model_path=pretrained_model_path)
    return model


def resnet34(feat_list, pretrained_model_path):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], feat_list=feat_list, pretrained_model_path=pretrained_model_path)
    return model


def resnet50(feat_list, pretrained_model_path):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], feat_list=feat_list, pretrained_model_path=pretrained_model_path)
    return model


def resnet101(feat_list, pretrained_model_path):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], feat_list=feat_list, pretrained_model_path=pretrained_model_path)
    return model


def resnet152(feat_list, pretrained_model_path):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], feat_list=feat_list, pretrained_model_path=pretrained_model_path)
    return model

def resnet_initializer(name, feat_list, pretrained = False):
    if cfg.PRETRAIN_TYPE == "pytorch":
        local_model_paths = {
            'res152': 'data/pretrained_model/resnet152-b121ed2d.pth',
            'res101': 'data/pretrained_model/resnet101-5d3b4d8f.pth',
            'res50': 'data/pretrained_model/resnet50-19c8e357.pth',
            'res34': 'data/pretrained_model/resnet34-333f7ec4.pth',
            'res18': 'data/pretrained_model/resnet18-5c106cde.pth',
        }
    elif cfg.PRETRAIN_TYPE == "caffe":
        local_model_paths = {
            'res152': 'data/pretrained_model/resnet152_caffe.pth',
            'res101': 'data/pretrained_model/resnet101_caffe.pth',
            'res50': 'data/pretrained_model/resnet50_caffe.pth',
            'res34': 'data/pretrained_model/resnet34_caffe.pth',
            'res18': 'data/pretrained_model/resnet18_caffe.pth',
        }
    else:
        raise RuntimeError("Please specify caffe or pytorch pretrained model to use.")
    cfg_dict = {
        "res18": {"block": BasicBlock, "layer_cfg": [2, 2, 2, 2]},
        "res34": {"block": BasicBlock, "layer_cfg": [3, 4, 6, 3]},
        "res50": {"block": Bottleneck, "layer_cfg": [3, 4, 6, 3]},
        "res101": {"block": Bottleneck, "layer_cfg": [3, 4, 23, 3]},
        "res152": {"block": Bottleneck, "layer_cfg": [3, 8, 36, 3]},
    }
    model = ResNet(cfg_dict[name]["block"], cfg_dict[name]["layer_cfg"],
                   feat_list=feat_list, pretrained_model_path=local_model_paths[name] if pretrained else None)
    return model



