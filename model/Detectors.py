import torch
from torch import nn
import torch.nn.functional as F

from utils.config import cfg
from utils.net_utils import set_bn_fix, set_bn_eval

import abc

from basenet.resnet import resnet_initializer
from basenet.vgg import vgg_initializer

class detector(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, feat_name='res101', feat_list=('conv4',), pretrained=True):
        super(detector, self).__init__()
        self.feat_name = feat_name
        self.feat_list = feat_list
        self.pretrained = pretrained

        self.FeatExt = self._init_feature_extractor()

    def _init_feature_extractor(self):
        # init resnet feature extractor
        if self.feat_name in {'res18', 'res34', 'res50', 'res101', 'res152'}:
            return resnet_initializer(self.feat_name, self.feat_list, self.pretrained)
        elif self.feat_name in {'vgg11', 'vgg13', 'vgg16', 'vgg19'}:
            return vgg_initializer(self.feat_name, self.feat_list, self.pretrained)

class graspDetector(detector):
    __metaclass__ = abc.ABCMeta
    def __init__(self, feat_name='res101', feat_list=('conv4',), pretrained=True):
        super(graspDetector, self).__init__(feat_name, feat_list, pretrained)

class objectDetector(detector):
    __metaclass__ = abc.ABCMeta
    def __init__(self, classes, class_agnostic,
                 feat_name = 'res101', feat_list = ('conv4',), pretrained = True):
        super(objectDetector, self).__init__(feat_name, feat_list, pretrained)

        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic



