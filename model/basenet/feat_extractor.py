import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from model.utils.config import cfg

import abc

# TODO: Now for each feature extractor, the useless layers are also created, which will waste the memory of GPU.

class featExtractor(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, feat_list=('conv5',)):
        """
        :param feat_list: a sub-list of ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc', 'cscore')
        """
        super(featExtractor, self).__init__()
        self.feat_list = feat_list

        self.feat_layer = OrderedDict()

