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

from model.utils.config import cfg

from model.fcgn.classifier import _Classifier
from model.fcgn.grasp_proposal_target import _GraspTargetLayer

from model.fcgn.bbox_transform_grasp import points2labels
from model.utils.net_utils import _smooth_l1_loss, weights_normal_init

import numpy as np

from model.fcgn.generate_grasp_anchors import generate_oriented_anchors
from model.Detectors import graspDetector

import pdb

class FCGN(graspDetector):

    def __init__(self, feat_name = 'res101', feat_list = ('conv4',), pretrained = True):
        super(FCGN, self).__init__(feat_name, feat_list, pretrained)
        ##### Important to set model to eval mode before evaluation ####
        self.FeatExt.eval()
        rand_img = torch.zeros(size=(1, 3, 224, 224))
        rand_feat = self.FeatExt(rand_img)
        self.FeatExt.train()
        self.dout_base_model = rand_feat.size(1)

        self.size = cfg.SCALES[0]
        self.FCGN_as = cfg.FCGN.ANCHOR_SCALES
        self.FCGN_ar = cfg.FCGN.ANCHOR_RATIOS
        self.FCGN_aa = cfg.FCGN.ANCHOR_ANGLES
        self.FCGN_fs = cfg.FCGN.FEAT_STRIDE[0]

        self.FCGN_classifier = _Classifier(self.dout_base_model, 5, self.FCGN_as, self.FCGN_ar, self.FCGN_aa)
        self.FCGN_proposal_target = _GraspTargetLayer(self.FCGN_fs, self.FCGN_ar, self.FCGN_as, self.FCGN_aa)

        self.FCGN_anchors = torch.from_numpy(generate_oriented_anchors(base_size=self.FCGN_fs,
                                    scales=np.array(self.FCGN_as), ratios=np.array(self.FCGN_ar),
                                    angles=np.array(self.FCGN_aa))).float()

        self.FCGN_num_anchors = self.FCGN_anchors.size(0)
        # [x1, y1, x2, y2] -> [xc, yc, w, h]
        self.FCGN_anchors = self._grasp_anchor_transform()

        self.iter_counter = 0

    def _grasp_anchor_transform(self):
        return torch.cat([
            (self.FCGN_anchors[:, 0:1] + self.FCGN_anchors[:, 2:3]) / 2,
            (self.FCGN_anchors[:, 1:2] + self.FCGN_anchors[:, 3:4]) / 2,
            self.FCGN_anchors[:, 2:3] - self.FCGN_anchors[:, 0:1] + 1,
            self.FCGN_anchors[:, 3:4] - self.FCGN_anchors[:, 1:2] + 1,
            self.FCGN_anchors[:, 4:5]
        ], dim=1)

    def forward(self, data_batch):
        x = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        if self.training:
            self.iter_counter += 1

        # features
        x = self.FeatExt(x)
        pred = self.FCGN_classifier(x)
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
            loc_label, conf_label, iw, ow = self.FCGN_proposal_target(conf, gt_xywhc, all_anchors)

            keep = conf_label.view(-1).ne(-1).nonzero().view(-1)
            conf = torch.index_select(conf.view(-1, 2), 0, keep.data)
            conf_label = torch.index_select(conf_label.view(-1), 0, keep.data)
            cls_loss = F.cross_entropy(conf, conf_label)

            bbox_loss = _smooth_l1_loss(loc, loc_label, iw, ow, dim = [2,1])

        return loc, prob, bbox_loss , cls_loss, conf_label, all_anchors

    def _generate_anchors(self, feat_height, feat_width):
        shift_x = np.arange(0, feat_width) * self.FCGN_fs
        shift_y = np.arange(0, feat_height) * self.FCGN_fs
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = torch.cat([
            shifts,
            torch.zeros(shifts.size(0), 3).type_as(shifts)
        ], dim = 1)
        shifts = shifts.contiguous().float()

        A = self.FCGN_num_anchors
        K = shifts.size(0)

        # anchors = self.FCGN_anchors.view(1, A, 5) + shifts.view(1, K, 5).permute(1, 0, 2).contiguous()
        anchors = self.FCGN_anchors.view(1, A, 5).type_as(shifts) + shifts.view(K, 1, 5)
        anchors = anchors.view(1, K * A, 5)

        return anchors

    def create_architecture(self):
        self._init_weights()

    def _init_weights(self):
        weights_normal_init(self.FCGN_classifier.conf, 0.01, 0.)
        weights_normal_init(self.FCGN_classifier.loc, 0.001, 0.)
