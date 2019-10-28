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

from header import _RCNN_header

class _RCNN_fc_header(_RCNN_header):
    def __init__(self, input_dim, n_classes, class_ag, hidden_number = (1024,1024,1024), include_bn = False):
        super(_RCNN_fc_header,self).__init__(input_dim, n_classes, class_ag)

        self.layer_number = len(hidden_number)
        self.bottoms = [input_dim,] + hidden_number[:-1]
        self.tops = hidden_number

        self.hiddens = nn.ModuleList()
        for i in range(self.bottoms):
            if include_bn:
                self.hiddens.append(
                    nn.Sequential(
                        nn.Linear(self.bottoms[i], self.tops[i]),
                        nn.BatchNorm1d(),
                        nn.ReLU()
                    )
                )
            else:
                self.hiddens.append(
                    nn.Sequential(
                        nn.Linear(self.bottoms[i], self.tops[i]),
                        nn.ReLU()
                    )
                )

        if class_ag:
            self.bbox_predictor = nn.Linear(self.tops[-1], 4)
        else:
            self.bbox_predictor = nn.Linear(self.tops[-1], 4 * self.n_classes)

        self.cls_predictor = nn.Linear(self.tops[-1], self.n_classes)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        for i in range(len(self.hiddens)):
            x = self.hiddens[i](x)

        bbox_delta = self.bbox_predictor(x)
        cls_score = self.cls_predictor(x)

        return bbox_delta, cls_score

    def _init_weights(self):
        # xavier initializer
        def xavier_init(m):
            def xavier(param):
                init.xavier_uniform(param)
            if type(m) == nn.Linear:
                xavier(m.weight.data)
                m.bias.data.zero_()
        self.hiddens.apply(xavier_init)
