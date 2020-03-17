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

import torch.nn.init as init
from model.utils.config import cfg


class _Classifier(nn.Module):
    def __init__(self, din, bbox_dim, anchor_scales, anchor_ratios, anchor_angles):

        super(_Classifier, self).__init__()
        self._bbox_dim = bbox_dim
        self._num_classes = 2
        self._scales = anchor_scales
        self._ratios = anchor_ratios
        self._angles = anchor_angles
        self._num_anchors = len(self._scales) * len(self._ratios) * len(self._angles)

        self.loc = nn.Conv2d(din, self._num_anchors * self._bbox_dim, kernel_size=3, padding=1)
        self.conf = nn.Conv2d(din, self._num_anchors * self._num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        loc = self.loc(x).permute(0,2,3,1)
        conf = self.conf(x).permute(0,2,3,1)
        return loc, conf

