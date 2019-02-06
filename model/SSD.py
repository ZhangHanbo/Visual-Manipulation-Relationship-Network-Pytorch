"""
# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# Modified from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
# --------------------------------------------------------
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from model.ssd.default_bbox_generator import PriorBox
import torch.nn.init as init
from model.ssd.multi_bbox_loss import MultiBoxLoss
from model.utils.config import cfg
from basenet.resnet import resnet18,resnet34,resnet50,resnet101,resnet152
import torchvision as tv

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class _SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, classes):
        super(_SSD, self).__init__()

        self.size = cfg.TRAIN.COMMON.INPUT_SIZE
        self.classes = classes
        self.num_classes = len(self.classes)
        self.priors_cfg = self._init_prior_cfg()
        self.priorbox = PriorBox(self.priors_cfg)
        self.priors_xywh = Variable(self.priorbox.forward())
        self.priors_xywh.detach()

        self.priors = torch.cat([
            self.priors_xywh[:, 0:1] - 0.5 * self.priors_xywh[:, 2:3],
            self.priors_xywh[:, 1:2] - 0.5 * self.priors_xywh[:, 3:4],
            self.priors_xywh[:, 0:1] + 0.5 * self.priors_xywh[:, 2:3],
            self.priors_xywh[:, 1:2] + 0.5 * self.priors_xywh[:, 3:4]
        ], 1)

        self.priors = self.priors * self.size
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = MultiBoxLoss(self.num_classes)

    def forward(self, x, im_info, gt_boxes, num_boxes):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        if isinstance(self.base, nn.ModuleList):
            for layer in self.base:
                x = layer(x)
        else:
            x = self.base(x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for conv in self.SSD_feat_layers:
            x = conv(x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        if self.training:
            predictions = (
                loc,
                conf,
                self.priors.type_as(loc)
            )
            # targets = torch.cat([gt_boxes[:,:,:4] / self.size, gt_boxes[:,:,4:5]],dim=2)
            targets = gt_boxes
            SSD_loss_bbox, SSD_loss_cls = self.criterion(predictions, targets, num_boxes)
        else:
            SSD_loss_cls = 0
            SSD_loss_bbox = 0

        conf = self.softmax(conf)

        return loc, conf, SSD_loss_bbox, SSD_loss_cls

    def create_architecture(self):
        self._init_modules()
        def weights_init(m):
            def xavier(param):
                init.xavier_uniform(param)
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()
        # initialize newly added layers' weights with xavier method
        self.extras.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)

    def _init_prior_cfg(self):
        prior_cfg = {
            'min_dim': self.size,
            'feature_maps': cfg.SSD.FEATURE_MAPS,
            'min_sizes': cfg.SSD.PRIOR_MIN_SIZE,
            'max_sizes': cfg.SSD.PRIOR_MAX_SIZE,
            'steps': cfg.SSD.PRIOR_STEP,
            'aspect_ratios':cfg.SSD.PRIOR_ASPECT_RATIO,
            'clip':cfg.SSD.PRIOR_CLIP
        }
        return prior_cfg

class vgg16(_SSD):
    # TODO: SUPPORT VGG19
    def __init__(self, num_classes, pretrained=False):
        super(vgg16 , self).__init__(num_classes)
        self._pretrained = pretrained
        self.module_path = "data/pretrained_model/vgg16_reducedfc.pth"
        self._bbox_dim = 4

    def _init_modules(self):
        # TODO: ADD CONFIGS INTO CONFIT.PY
        base_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                    512, 512, 512]

        extras_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

        mbox_cfg = []
        for i in cfg.SSD.PRIOR_ASPECT_RATIO:
            mbox_cfg.append(2 * len(i) + 2)

        base, extras, head = self.multibox(self.vgg(base_cfg, 3),
                                         self.add_extras(extras_cfg, 1024),
                                         mbox_cfg, self.num_classes, self._bbox_dim)
        # init base net
        self.base = nn.ModuleList(base)
        vgg_weights = torch.load(self.module_path)

        if self._pretrained:
            self.base.load_state_dict(vgg_weights)

        self.SSD_feat_layers = self.base[23:]
        self.base = self.base[:23]

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def add_extras(self, cfg, i, batch_norm=False):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            # if last layer is S, skip to next v.
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers

    def multibox(self, vgg, extra_layers, cfg, num_classes, bbox_dim):
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * bbox_dim, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * bbox_dim, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return vgg, extra_layers, (loc_layers, conf_layers)

    # This function is derived from torchvision VGG make_layers()
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    def vgg(self, cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

