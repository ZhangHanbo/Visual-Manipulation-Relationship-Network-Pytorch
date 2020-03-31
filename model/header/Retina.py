from functools import partial

import numpy as np
import torch.nn as nn

from model.utils.net_utils import bias_init_with_prob, weights_normal_init
from six.moves import map, zip

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class retinaHeader(nn.Module):
    """
    An anchor-based head used in [1]_.
    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.
    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 stacked_convs=4,
                 class_agnostic = True):
        super(retinaHeader, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.stacked_convs = stacked_convs
        self.cls_out_channels = num_classes
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self.class_agnostic = class_agnostic
        self._init_modules()

    def _init_modules(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.cls_convs.append(nn.BatchNorm2d(self.feat_channels))
            self.cls_convs.append(nn.ReLU())
            self.reg_convs.append(nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.reg_convs.append(nn.BatchNorm2d(self.feat_channels))
            self.reg_convs.append(nn.ReLU())

        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        bbox_dim = 4 if self.class_agnostic else 4 * self.cls_out_channels
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * bbox_dim, 3, padding=1)
        self.output_act = nn.Sigmoid()

    def _init_weights(self):
        for m in self.cls_convs:
            weights_normal_init(m.conv, dev=0.01)
        for m in self.reg_convs:
            weights_normal_init(m.conv, dev=0.01)
        bias_cls = bias_init_with_prob(0.01)
        weights_normal_init(self.retina_cls, dev=0.01, bias=bias_cls)
        weights_normal_init(self.retina_reg, dev=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        cls_score = self.output_act(cls_score)
        # out is B x C x W x H, with C = n_classes + n_anchors
        cls_score = cls_score.permute(0, 2, 3, 1)
        batch_size, width, height, channels = cls_score.shape
        cls_score = cls_score.view(
            batch_size, width, height, self.num_anchors, self.num_classes)
        cls_score = cls_score.contiguous().view(x.size(0), -1, self.num_classes)

        bbox_pred = self.retina_reg(reg_feat)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        bbox_pred = bbox_pred.contiguous().view(bbox_pred.size(0), -1, 4 if self.class_agnostic else 4 * self.num_classes)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
