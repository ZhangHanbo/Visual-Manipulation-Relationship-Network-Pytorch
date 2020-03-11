import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from model.ssd.default_bbox_generator import PriorBox
import torch.nn.init as init
from torchvision import models

from model.utils.config import cfg
from basenet.resnet import resnet18,resnet34,resnet50,resnet101,resnet152

from model.fully_conv_grasp.classifier import _Classifier
from model.fully_conv_grasp.grasp_proposal_target import _GraspTargetLayer
from model.fully_conv_grasp.bbox_transform_grasp import \
    points2labels,labels2points,grasp_encode, grasp_decode

from model.fully_conv_grasp.bbox_transform_grasp import points2labels
from model.utils.net_utils import _smooth_l1_loss

import numpy as np

from model.fully_conv_grasp.generate_grasp_anchors import generate_oriented_anchors

import pdb

class _FCGN(nn.Module):

    def __init__(self):
        super(_FCGN, self).__init__()
        self.size = cfg.TRAIN.COMMON.INPUT_SIZE

        self._as = cfg.FCGN.ANCHOR_SCALES
        self._ar = cfg.FCGN.ANCHOR_RATIOS
        self._aa = cfg.FCGN.ANCHOR_ANGLES
        self._fs = cfg.FCGN.FEAT_STRIDE[0]

        # for resnet
        if self.dout_base_model is None:
            if self._fs == 16:
                self.dout_base_model = 256 * self.expansions
            elif self._fs == 32:
                self.dout_base_model = 512 * self.expansions

        self.classifier = _Classifier(self.dout_base_model, 5, self._as, self._ar, self._aa)
        self.proposal_target = _GraspTargetLayer(self._fs, self._ar, self._as, self._aa)

        self._anchors = torch.from_numpy(generate_oriented_anchors(base_size=self._fs,
                                    scales=np.array(self._as), ratios=np.array(self._ar),
                                    angles=np.array(self._aa))).float()

        self._num_anchors = self._anchors.size(0)

        # [x1, y1, x2, y2] -> [xc, yc, w, h]
        self._anchors = torch.cat([
            (self._anchors[:, 0:1] + self._anchors[:, 2:3]) / 2,
            (self._anchors[:, 1:2] + self._anchors[:, 3:4]) / 2,
            self._anchors[:, 2:3] - self._anchors[:, 0:1] + 1,
            self._anchors[:, 3:4] - self._anchors[:, 1:2] + 1,
            self._anchors[:, 4:5]
        ], dim=1)

        self.iter_counter = 0

    def forward(self, data_batch):
        x = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        if self.training:
            self.iter_counter += 1

        # features
        x = self.base(x)
        pred = self.classifier(x)
        loc, conf = pred
        self.batch_size = loc.size(0)

        all_anchors = self._generate_anchors(conf.size(1), conf.size(2))
        all_anchors = all_anchors.type_as(gt_boxes)
        all_anchors = all_anchors.expand(self.batch_size, all_anchors.size(1),all_anchors.size(2))

        loc = loc.contiguous().view(loc.size(0), -1, 5)
        conf = conf.contiguous().view(conf.size(0), -1, 2)
        prob = F.softmax(conf, 2)

        bbox_loss = 0
        cls_loss = 0
        conf_label = None
        if self.training:
            # inside weights indicate which bounding box should be regressed
            # outside weidhts indicate two things:
            # 1. Which bounding box should contribute for classification loss,
            # 2. Balance cls loss and bbox loss
            gt_xywhc = points2labels(gt_boxes)
            loc_label, conf_label, iw, ow = self.proposal_target(conf, gt_xywhc, all_anchors)

            keep = Variable(conf_label.view(-1).ne(-1).nonzero().view(-1))
            conf = torch.index_select(conf.view(-1, 2), 0, keep.data)
            conf_label = torch.index_select(conf_label.view(-1), 0, keep.data)
            cls_loss = F.cross_entropy(conf, conf_label)

            iw = Variable(iw)
            ow = Variable(ow)
            loc_label = Variable(loc_label)
            bbox_loss = _smooth_l1_loss(loc, loc_label, iw, ow, dim = [2,1])

        return loc, prob, bbox_loss , cls_loss, conf_label, all_anchors

    def _generate_anchors(self, feat_height, feat_width):
        shift_x = np.arange(0, feat_width) * self._fs
        shift_y = np.arange(0, feat_height) * self._fs
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = torch.cat([
            shifts,
            torch.zeros(shifts.size(0), 3).type_as(shifts)
        ], dim = 1)
        shifts = shifts.contiguous().float()

        A = self._num_anchors
        K = shifts.size(0)

        # anchors = self._anchors.view(1, A, 5) + shifts.view(1, K, 5).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 5) + shifts.view(K, 1, 5)
        anchors = anchors.view(1, K * A, 5)

        return anchors

    def create_architecture(self):
        self._init_modules()

        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def weights_init(m):
            def xavier(param):
                init.xavier_uniform(param)
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()

        # initialize newly added layers' weights with xavier method
        # self.classifier.loc.apply(weights_init)
        # self.classifier.conf.apply(weights_init)
        normal_init(self.classifier.conf, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.classifier.loc, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)

class resnet(_FCGN):
    def __init__(self, num_layers = 101, pretrained=False):

        self.num_layers = num_layers
        self.model_path = 'data/pretrained_model/resnet' + str(num_layers) + '_caffe.pth'

        self.pretrained = pretrained
        self._bbox_dim = 5
        self.dout_base_model = None
        if num_layers == 18 or num_layers == 34:
            self.expansions = 1
        elif num_layers == 50 or num_layers == 101 or num_layers == 152:
            self.expansions = 4
        else:
            assert 0, "network not defined"

        super(resnet, self).__init__()

    def _init_modules(self):
        if self.num_layers == 18:
            resnet = resnet18()
        elif self.num_layers == 34:
            resnet = resnet34()
        elif self.num_layers == 50:
            resnet = resnet50()
        elif self.num_layers == 101:
            resnet = resnet101()
        elif self.num_layers == 152:
            resnet = resnet152()
        else:
            assert 0, "network not defined"

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet.
        if self._fs == 16:
            self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
        elif self._fs == 32:
            self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)


class vgg16(_FCGN):
    def __init__(self, pretrained=False):

        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self._bbox_dim = 5
        super(vgg16, self).__init__()

    def _init_modules(self):
        vgg = models.vgg16()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        if self._fs == 16:
            self.base = vgg.features[:24]
        elif self._fs == 32:
            self.base = vgg.features