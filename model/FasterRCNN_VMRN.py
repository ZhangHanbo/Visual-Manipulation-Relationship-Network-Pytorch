# --------------------------------------------------------
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# Modified from R. B. G.'s faster_rcnn.py
# --------------------------------------------------------

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from utils.config import cfg
from rpn.rpn import _RPN
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_crop.modules.roi_crop import _RoICrop
from roi_align.modules.roi_align import RoIAlignAvg
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms.nms_wrapper import nms
import time
import pdb
from utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, set_bn_eval, set_bn_fix
from utils.net_utils import objdet_inference
from model.rpn.bbox_transform import bbox_overlaps
import copy

from FasterRCNN import fasterRCNN

from model.op2l.object_pairing_layer import _ObjPairLayer
from model.op2l.rois_pair_expanding_layer import  _RoisPairExpandingLayer
from model.op2l.op2l import _OP2L

class vmrn_rel_classifier(nn.Module):
    def __init__(self, obj_pair_feat_dim):
        super(vmrn_rel_classifier,self).__init__()
        self._input_dim = obj_pair_feat_dim
        self.fc1 = nn.Linear(self._input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.outlayer = nn.Linear(2048,3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.outlayer(x)
        return x


class fasterRCNN_VMRN(fasterRCNN):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained = True):
        super(fasterRCNN_VMRN, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)

        self._isex = cfg.TRAIN.VMRN.ISEX
        self.VMRN_rel_op2l = _OP2L(cfg.VMRN.OP2L_POOLING_SIZE, cfg.VMRN.OP2L_POOLING_SIZE, 1.0 / 16.0, self._isex)

        self.iter_counter = 0

    def _object_detection(self, rois, cls_prob, bbox_pred, batch_size, im_info):
        det_results = torch.Tensor([]).type_as(rois[0])
        obj_num = []

        for i in range(batch_size):
            obj_boxes = torch.Tensor(objdet_inference(cls_prob[i], bbox_pred[i], im_info[i], rois[i][:, 1:5],
                                                      class_agnostic = self.class_agnostic, n_classes = self.n_classes,
                                                      for_vis = True)).type_as(det_results)
            obj_num.append(obj_boxes.size(0))
            if obj_num[-1] > 0 :
                # add image index
                img_ind = i * torch.ones(obj_boxes.size(0), 1).type_as(det_results)
                det_results = torch.cat([det_results, torch.cat([img_ind, obj_boxes], 1)], 0)

        return det_results, torch.Tensor(obj_num).type_as(det_results).long()

    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]
        rel_mat = data_batch[4]

        # object detection
        if self.training:
            self.iter_counter += 1
        self.batch_size = im_data.size(0)

        # feed image data to base model to obtain base feature map
        base_feat = self.FeatExt(im_data)
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # rois preprocess
        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = \
                self._get_header_train_data(rois, gt_boxes, num_boxes)
            pos_rois_labels = [(rois_label[i * rois.size(1): (i + 1) * rois.size(1)] > 0) for i in range(self.batch_size)]
            od_rois = [rois[i][pos_rois_labels[i]].data for i in range(self.batch_size)]
        else:
            rois_label, rois_target, rois_inside_ws, rois_outside_ws = None, None, None, None
            od_rois = rois.data

        pooled_feat = self._roi_pooling(base_feat, rois)
        cls_score, cls_prob, bbox_pred = self._get_det_rslt(pooled_feat)
        RCNN_loss_bbox, RCNN_loss_cls = 0, 0
        if self.training:
            RCNN_loss_bbox, RCNN_loss_cls = self._loss_comp(cls_score, cls_prob, bbox_pred, rois_label, rois_target,
                                                            rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.contiguous().view(self.batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.contiguous().view(self.batch_size, rois.size(1), -1)
        # for object detection before relationship detection
        if self.training:
            od_cls_prob = [cls_prob[i][pos_rois_labels[i]].data for i in range(self.batch_size)]
            od_bbox_pred = [bbox_pred[i][pos_rois_labels[i]].data for i in range(self.batch_size)]
        else:
            od_cls_prob = cls_prob.data
            od_bbox_pred = bbox_pred.data

        # online data
        obj_rois, obj_num = self._object_detection(od_rois, od_cls_prob, od_bbox_pred, self.batch_size, im_info.data)

        # offline data
        if self.training:
            for i in range(self.batch_size):
                img_ind = (i * torch.ones(num_boxes[i].item(),1)).type_as(gt_boxes)
                obj_rois = torch.cat([obj_rois, torch.cat([img_ind, (gt_boxes[i][:num_boxes[i]])],1)])
                obj_num = torch.cat([obj_num,torch.Tensor([num_boxes[i]]).type_as(obj_num)])

        obj_labels = torch.Tensor([]).type_as(gt_boxes).long()
        if obj_rois.size(0) > 0:
            obj_labels = obj_rois[:, 5]
            obj_rois = obj_rois[:, :5]

        if (obj_num > 1).sum().item() > 0:
            # filter out the detection of only one object instance
            obj_pair_feat = self.VMRN_rel_op2l(base_feat, obj_rois, self.batch_size, obj_num)
            obj_pair_feat = obj_pair_feat.detach()
            obj_pair_feat = self._rel_head_to_tail(obj_pair_feat)
            rel_cls_score = self.VMRN_rel_cls_score(obj_pair_feat)

            rel_cls_prob = F.softmax(rel_cls_score)

            VMRN_rel_loss_cls = 0
            if self.training:
                self.rel_batch_size = rel_cls_prob.size(0)

                obj_pair_rel_label = self._generate_rel_labels(obj_rois, gt_boxes, obj_num, rel_mat)
                obj_pair_rel_label = obj_pair_rel_label.type_as(gt_boxes).long()

                rel_not_keep = (obj_pair_rel_label == 0)

                # no relationship is kept
                if (rel_not_keep == 0).sum().item() > 0:
                    rel_keep = torch.nonzero(rel_not_keep == 0).view(-1)

                    rel_cls_score = rel_cls_score[rel_keep]
                    obj_pair_rel_label = obj_pair_rel_label[rel_keep]

                    obj_pair_rel_label -= 1

                    VMRN_rel_loss_cls = F.cross_entropy(rel_cls_score, obj_pair_rel_label)
            else:
                if (not cfg.TEST.VMRN.ISEX) and cfg.TRAIN.VMRN.ISEX:
                    rel_cls_prob = rel_cls_prob[::2,:]

        else:
            VMRN_rel_loss_cls = 0
            # no detected relationships
            rel_cls_prob = Variable(torch.Tensor([]).type_as(cls_prob))

        rel_result = None
        if not self.training:
            if obj_rois.numel() > 0:
                pred_boxes = obj_rois.data[:,1:5]
                pred_boxes[:, 0::2] /= im_info[0][3].item()
                pred_boxes[:, 1::2] /= im_info[0][2].item()
                rel_result = (pred_boxes, obj_labels, rel_cls_prob.data)
            else:
                rel_result = (obj_rois.data, obj_labels, rel_cls_prob.data)

        return rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, VMRN_rel_loss_cls, rois_label

    def _generate_rel_labels(self, obj_rois, gt_boxes, obj_num, rel_mat):

        obj_pair_rel_label = torch.Tensor(self.rel_batch_size).type_as(gt_boxes).zero_().long()
        # generate online data labels
        cur_pair = 0
        for i in range(obj_num.size(0)):
            img_index = i % self.batch_size
            if obj_num[i] <=1 :
                continue
            begin_ind = torch.sum(obj_num[:i])
            overlaps = bbox_overlaps(obj_rois[begin_ind:begin_ind + obj_num[i]][:, 1:5],
                                     gt_boxes[img_index][:, 0:4])
            max_overlaps, max_inds = torch.max(overlaps, 1)
            for o1ind in range(obj_num[i]):
                for o2ind in range(o1ind + 1, obj_num[i]):
                    o1_gt = int(max_inds[o1ind].item())
                    o2_gt = int(max_inds[o2ind].item())
                    if o1_gt == o2_gt:
                        # skip invalid pairs
                        if self._isex:
                            cur_pair += 2
                        else:
                            cur_pair += 1
                        continue
                    # some labels are leaved out when labeling
                    if rel_mat[img_index][o1_gt, o2_gt].item() == 0:
                        if rel_mat[img_index][o2_gt, o1_gt].item() == 3:
                            rel_mat[img_index][o1_gt, o2_gt] = rel_mat[img_index][o2_gt, o1_gt]
                        else:
                            rel_mat[img_index][o1_gt, o2_gt] = 3 - rel_mat[img_index][o2_gt, o1_gt]
                    obj_pair_rel_label[cur_pair] = rel_mat[img_index][o1_gt, o2_gt]

                    cur_pair += 1
                    if self._isex:
                        # some labels are leaved out when labeling
                        if rel_mat[img_index][o2_gt, o1_gt].item() == 0:
                            if rel_mat[img_index][o1_gt, o2_gt].item() == 3:
                                rel_mat[img_index][o2_gt, o1_gt] = rel_mat[img_index][o1_gt, o2_gt]
                            else:
                                rel_mat[img_index][o2_gt, o1_gt] = 3 - rel_mat[img_index][o1_gt, o2_gt]
                        obj_pair_rel_label[cur_pair] = rel_mat[img_index][o2_gt, o1_gt]
                        cur_pair += 1

        return obj_pair_rel_label


    def _init_weights(self):
        fasterRCNN._init_weights(self)
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

        normal_init(self.VMRN_rel_cls_score.fc1, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.fc2, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.outlayer, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _init_modules_resnet(self):
        fasterRCNN._init_modules_resnet(self)

        # VMRN layers
        if cfg.VMRN.SHARE_WEIGHTS:
            self.VMRN_rel_top = copy.deepcopy(self.FeatExt.layer4)
        else:
            self.VMRN_rel_top_o1 = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_o2 = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_union = copy.deepcopy(self.FeatExt.layer4)

        self.VMRN_rel_cls_score = vmrn_rel_classifier(2048 * 3)

    def _init_modules_vgg(self):
        fasterRCNN._init_modules_vgg(self)

        def rel_pipe():
            return nn.Sequential(
                nn.Conv2d(512, 128, 1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, 1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                )

        if cfg.VMRN.SHARE_WEIGHTS:
            self.VMRN_rel_top = rel_pipe()
        else:
            self.VMRN_rel_top_o1 = rel_pipe()
            self.VMRN_rel_top_o2 = rel_pipe()
            self.VMRN_rel_top_union = rel_pipe()

        self.VMRN_rel_cls_score = vmrn_rel_classifier(64 * 7 * 7 * 3)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        fasterRCNN.train(self, mode)
        if mode and self.feat_name[:3] == 'res':
            if cfg.VMRN.SHARE_WEIGHTS:
                self.VMRN_rel_top.apply(set_bn_eval)
            else:
                self.VMRN_rel_top_o1.apply(set_bn_eval)
                self.VMRN_rel_top_o2.apply(set_bn_eval)
                self.VMRN_rel_top_union.apply(set_bn_eval)

    def _rel_head_to_tail(self, pooled_pair):
        if self.feat_name[:3] == 'res':
            return self._rel_head_to_tail_resnet(pooled_pair)
        elif self.feat_name[:3] == 'vgg':
            return self._rel_head_to_tail_vgg(pooled_pair)

    def _rel_head_to_tail_resnet(self, pooled_pair):
        # box_type: o1, o2, union
        opfc = []
        if cfg.VMRN.SHARE_WEIGHTS:
            for box_type in range(pooled_pair.size(1)):
                cur_box = pooled_pair[:,box_type]
                opfc[box_type] = self.VMRN_rel_top(cur_box)
        else:
            opfc.append(self.VMRN_rel_top_o1(pooled_pair[:,0]).mean(3).mean(2))
            opfc.append(self.VMRN_rel_top_o2(pooled_pair[:, 1]).mean(3).mean(2))
            opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).mean(3).mean(2))
        return torch.cat(opfc, 1)

    def _rel_head_to_tail_vgg(self, pooled_pair):
        # box_type: o1, o2, union
        opfc = []
        if cfg.VMRN.SHARE_WEIGHTS:
            for box_type in range(pooled_pair.size(1)):
                cur_box = pooled_pair[:, box_type]
                opfc[box_type] = self.VMRN_rel_top(cur_box)
        else:
            opfc.append(self.VMRN_rel_top_o1(pooled_pair[:, 0]).view(pooled_pair.size(0), -1))
            opfc.append(self.VMRN_rel_top_o2(pooled_pair[:, 1]).view(pooled_pair.size(0), -1))
            opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).view(pooled_pair.size(0), -1))
        return torch.cat(opfc,1)
