"""
# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init

from header import header

class _RFCN_header(header):
    def __init__(self, input_dim, n_classes, class_ag, k = 3):
        """
        :param input_dim: feature map channel number
        :param n_classes:
        :param class_ag:
        :param k: grid size
        """
        super(_RFCN_header,self).__init__(input_dim, n_classes, class_ag)

        self.position_sensitive_score_map = \
            nn.Conv2d(input_dim, k ** 2 * n_classes, kernel_size=1)
        if class_ag:
            self.position_sensitive_bbox_map = \
                nn.Conv2d(input_dim, k ** 2 * 4, kernel_size=1)
        else:
            self.position_sensitive_bbox_map = \
                nn.Conv2d(input_dim, k ** 2 * 4 * n_classes, kernel_size=1)

        self.k = k

    def forward(self, x):
        """
        :param feat: [batch_size, channel, H, W]
        :param rois: [batch_size, num, 5]
        :return:
        """

        feat = x[0]
        rois = x[1]

        score_map = self.position_sensitive_score_map(feat)
        bbox_map = self.position_sensitive_bbox_map(feat)


