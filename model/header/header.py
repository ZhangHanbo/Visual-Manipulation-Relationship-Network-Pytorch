"""
# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------
"""

import torch.nn as nn
import abc

class header(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, n_classes, class_agnostic):
        super(header, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_agnostic = class_agnostic
        self.header = None
        self.conf = None
        self.loc = None

    def forward(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def _make_header(self):
        raise NotImplementedError
