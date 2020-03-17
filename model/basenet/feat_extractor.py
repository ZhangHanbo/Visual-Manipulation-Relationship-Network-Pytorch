import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import abc

class featExtractor(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, feat_list=('conv5',)):
        """
        :param feat_list: a sub-list of ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc', 'cscore')
        """
        super(featExtractor, self).__init__()
        self.feat_list = feat_list

        # initialize feat_layer
        self.feat_layer = OrderedDict()
        for key in ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc', 'cscore'):
            self.feat_layer[key] = None



