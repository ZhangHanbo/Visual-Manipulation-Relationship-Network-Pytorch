# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from model.ssd.default_bbox_generator import PriorBox
import torch.nn.init as init
from model.ssd.multi_bbox_loss import MultiBoxLoss
from model.utils.config import cfg
import torchvision as tv

from model.utils.net_utils import weights_xavier_init
from Detectors import objectDetector

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

class SSD(objectDetector):

    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv3', 'conv4'), pretrained = True):
        super(SSD, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)
        self.FeatExt.feat_layer["conv3"][0].ceil_mode = True
        ##### Important to set model to eval mode before evaluation ####
        self.FeatExt.eval()
        rand_img = torch.Tensor(1, 3, 300, 300)
        rand_feat = self.FeatExt(rand_img)
        self.FeatExt.train()
        n_channels = [f.size(1) for f in rand_feat]

        self.size = cfg.SCALES[0]
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
        self.criterion = MultiBoxLoss(self.n_classes)

        mbox_cfg = []
        for i in cfg.SSD.PRIOR_ASPECT_RATIO:
            mbox_cfg.append(2 * len(i) + 2)

        self.extra_conv = nn.ModuleList()
        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()

        # conv 4_3 detector
        self.loc.append(
            nn.Conv2d(n_channels[0], mbox_cfg[0] * 4 if self.class_agnostic else mbox_cfg[0] * 4 * self.n_classes
                      , kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(n_channels[0], mbox_cfg[0] * self.n_classes, kernel_size=3, padding=1))

        # conv 7 detector
        self.extra_conv.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)))
        self.loc.append(nn.Conv2d(1024, mbox_cfg[1] * 4 if self.class_agnostic else mbox_cfg[1] * 4 * self.n_classes,
                                  kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(1024, mbox_cfg[1] * self.n_classes, kernel_size=3, padding=1))

        def add_extra_conv(extra_conv, loc, conf, in_c, mid_c, out_c, downsamp, mbox, n_cls, cag):
            extra_conv.append(nn.Sequential(
                nn.Conv2d(in_c, mid_c, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, out_c, kernel_size=3, stride=2 if downsamp else 1, padding=1 if downsamp else 0),
                nn.ReLU(inplace=True),
            ))
            loc.append(nn.Conv2d(out_c, mbox * 4 if cag else mbox * 4 * n_cls, kernel_size=3, padding=1))
            conf.append(nn.Conv2d(out_c, mbox * n_cls, kernel_size=3, padding=1))

        add_extra_conv(self.extra_conv, self.loc, self.conf, 1024, 256, 512, True, mbox_cfg[2], self.n_classes, self.class_agnostic)
        add_extra_conv(self.extra_conv, self.loc, self.conf, 512, 128, 256, True, mbox_cfg[3], self.n_classes, self.class_agnostic)
        add_extra_conv(self.extra_conv, self.loc, self.conf, 256, 128, 256, False, mbox_cfg[4], self.n_classes, self.class_agnostic)
        add_extra_conv(self.extra_conv, self.loc, self.conf, 256, 128, 256, False, mbox_cfg[5], self.n_classes, self.class_agnostic)

        self.iter_counter = 0

    def _get_obj_det_result(self, sources):
        loc = []
        conf = []
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.n_classes)
        return loc, conf

    def forward(self, data_batch):
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
        x = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        if self.training:
            self.iter_counter += 1

        sources = []

        s0, x = self.FeatExt(x)
        s0 = self.L2Norm(s0)
        sources.append(s0)

        for m in self.extra_conv:
            x = m(x)
            sources.append(x)

        loc, conf = self._get_obj_det_result(sources)
        SSD_loss_cls, SSD_loss_bbox = 0, 0
        if self.training:
            predictions = (
                loc,
                conf,
                self.priors.type_as(loc)
            )
            SSD_loss_bbox, SSD_loss_cls = self.criterion(predictions, gt_boxes, num_boxes)
        conf = self.softmax(conf)

        return loc, conf, SSD_loss_bbox, SSD_loss_cls

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _init_modules(self):
        pass

    def _init_weights(self):
        from functools import partial
        xavier_init = partial(weights_xavier_init, gain=1., bias=0., distribution='uniform')

        # initialize newly added layers' weights with xavier method
        self.extra_conv.apply(xavier_init)
        self.loc.apply(xavier_init)
        self.conf.apply(xavier_init)

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
