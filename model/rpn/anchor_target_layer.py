from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        self._ratios = ratios

        if isinstance(feat_stride,list):
            self._anchors = []
            self._num_anchors = []
            for i in feat_stride:
                if isinstance(self._scales[0], list):
                    scales = self._scales[i]
                if isinstance(self._ratios[0], list):
                    ratios = self._ratios[i]
                anchor = torch.from_numpy(generate_anchors(base_size=i,
                          scales=np.array(scales), ratios=np.array(ratios))).float()
                self._anchors.append(anchor)
                self._num_anchors.append(anchor.size(0))
        else:
            self._anchors = torch.from_numpy(generate_anchors(base_size=feat_stride,
                          scales=np.array(scales), ratios=np.array(ratios))).float()
            self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        batch_size = gt_boxes.size(0)

        if isinstance(rpn_cls_score,list):
            feat_width = []
            feat_height = []
            for i in range(len(rpn_cls_score)):
                feat_height.append(rpn_cls_score[i].size(2))
                feat_width.append(rpn_cls_score[i].size(3))
        else:
            feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        A = self._num_anchors

        if isinstance(self._anchors, list):
            for i in range(len(self._anchors)):
                self._anchors[i] = self._anchors[i].type_as(gt_boxes) # move to specific gpu.
        else:
            self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.

        all_anchors = self._generate_anchors(feat_height, feat_width)
        all_anchors = all_anchors.type_as(gt_boxes)

        total_anchors = int(all_anchors.size(0))

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not cfg.TRAIN.RCNN_COMMON.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RCNN_COMMON.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RCNN_COMMON.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RCNN_COMMON.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RCNN_COMMON.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RCNN_COMMON.RPN_FG_FRACTION * cfg.TRAIN.RCNN_COMMON.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RCNN_COMMON.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RCNN_COMMON.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = cfg.TRAIN.RCNN_COMMON.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RCNN_COMMON.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RCNN_COMMON.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RCNN_COMMON.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        # labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        # labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        # bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        # bbox_targets = bbox_targets.view(batch_size, height, width, A * 4)
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        # bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
        #                     .permute(0,3,1,2).contiguous()
        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        #bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
        #                    .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _generate_anchors(self, feat_height, feat_width):

        if isinstance(feat_height,list):
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
        else:
            shift_x = np.arange(0, feat_width) * self._feat_stride
            shift_y = np.arange(0, feat_height) * self._feat_stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
            shifts = shifts.contiguous().type_as(self._anchors).float()

            A = self._num_anchors
            K = shifts.size(0)

            # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
            anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
            anchors = anchors.view(1, K * A, 4).squeeze()
        return anchors

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


