# --------------------------------------------------------
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# Modified from https://github.com/toandaominh1997/EfficientDet.Pytorch
# --------------------------------------------------------

import math
import numpy as np

from model.featnet.BiFPN import BIFPN
from model.header.Retina import retinaHeader
from model.utils.net_utils import weight_kaiming_init, _focal_loss, _smooth_l1_loss
from model.rpn.generate_anchors import generate_anchors
from model.rpn.bbox_transform import *
from model.utils.config import cfg

from Detectors import objectDetector
import torch.nn as nn
import torch

class EfficientDet(objectDetector):
    def __init__(self, classes, class_agnostic,
                 feat_name = 'efficientnet-b2', feat_list = ('conv3', 'conv4', 'conv5', 'conv6', 'conv7'), pretrained = True,
                 D_bifpn=3, W_bifpn=88):
        super(EfficientDet, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)

        self.ED_BIFPN = BIFPN(in_channels=self.FeatExt.get_list_features(),
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)
        self.ED_retinahead = retinaHeader(num_classes=self.n_classes,
                                    in_channels=W_bifpn, class_agnostic=self.class_agnostic)

        self._anchor_scales = cfg.RCNN_COMMON.ANCHOR_SCALES
        self._anchor_ratios = cfg.RCNN_COMMON.ANCHOR_RATIOS
        self._feat_stride = cfg.RCNN_COMMON.FEAT_STRIDE

        self._anchors = []
        self._num_anchors = []
        for i in self._feat_stride:
            if isinstance(self._anchor_scales[0], list):
                scales = self._anchor_scales[i]
            else:
                scales = self._anchor_scales
            if isinstance(self._anchor_ratios[0], list):
                ratios = self._anchor_ratios[i]
            else:
                ratios = self._anchor_ratios
            anchor = torch.from_numpy(
                generate_anchors(base_size=i, scales=np.array(scales), ratios=np.array(ratios))).float()
            self._anchors.append(anchor)
            self._num_anchors.append(anchor.size(0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # self.freeze_bn()

        self.iter_counter = 0

    def _generate_anchors(self, feat_height, feat_width):

        anchors = []
        assert isinstance(feat_width, list) \
               and len(feat_width) == len(feat_height) \
               and isinstance(self._feat_stride, list) \
               and len(self._feat_stride) == len(feat_height),\
            "feat height, feat weight, feat stride should be all lists or ints, and length of them should be equal"

        for i in range(len(feat_height)):
            shift_x = np.arange(0, feat_width[i]) * self._feat_stride[i]
            shift_y = np.arange(0, feat_height[i]) * self._feat_stride[i]
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
            shifts = shifts.contiguous().type_as(self._anchors[i]).float()
            A = self._num_anchors[i]
            K = shifts.size(0)

            # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
            anchor = self._anchors[i].view(1, A, 4) + shifts.view(K, 1, 4)
            anchor = anchor.view(1, K * A, 4)

            anchors.append(anchor)
        anchors = torch.cat(anchors , dim = 1).squeeze()

        return anchors

    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        self.batch_size = im_data.shape[0]
        if self.training:
            self.iter_counter += 1

        x = self.FeatExt(im_data)
        x = self.ED_BIFPN(x)
        outs = self.ED_retinahead(x)
        cls_prob = torch.cat([out for out in outs[0]], dim=1)
        bbox_pred = torch.cat([out for out in outs[1]], dim=1)

        feat_height = [f.size(2) for f in x]
        feat_width = [f.size(3) for f in x]
        all_anchors = self._generate_anchors(feat_height, feat_width).type_as(gt_boxes)
        self.priors = all_anchors

        loss_bbox, loss_cls = 0, 0
        if self.training:
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            # batch_size x N_a x N_gt
            overlaps = bbox_overlaps_batch(all_anchors, gt_boxes)
            ov_max, ov_argmax = torch.max(overlaps, dim=2)  # batch_size x N_a x 1
            # positive sample indices
            pos_ind = torch.ge(ov_max, 0.5)
            bbox_targets = torch.cat([gt_boxes[i][ov_argmax[i]].unsqueeze(0) for i in range(self.batch_size)], dim = 0)
            transformed_targets = bbox_transform_batch(all_anchors, bbox_targets)

            labels = torch.Tensor(cls_prob.shape[:2]).view(-1).zero_().type_as(gt_boxes).long()
            labels[pos_ind.view(-1)] = bbox_targets[:,:,4].view(-1)[pos_ind.view(-1)].long()
            loss_cls = _focal_loss(cls_prob.view(-1, cls_prob.size(-1)), labels,
                                    gamma=cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA, alpha=cfg.TRAIN.COMMON.FOCAL_LOSS_ALPHA)
            # compute the loss for regression
            # if pos_ind.sum() > 0:
            bbox_inside_weights = bbox_pred.new(bbox_pred.shape).zero_()
            bbox_outside_weights = bbox_pred.new(bbox_pred.shape).zero_()

            bbox_inside_weights[pos_ind] = 1.
            bbox_outside_weights[pos_ind] = 1.

            loss_bbox = _smooth_l1_loss(bbox_pred, transformed_targets,
                                        bbox_inside_weights, bbox_outside_weights, 1./3., dim = [1])

        return bbox_pred, cls_prob, loss_bbox, loss_cls

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def create_architecture(self):
        pass