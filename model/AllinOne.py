# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
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
import torch.nn.init as init

import numpy as np
from utils.config import cfg
from rpn.rpn import _RPN
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_crop.modules.roi_crop import _RoICrop
from roi_align.modules.roi_align import RoIAlignAvg
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.fully_conv_grasp.bbox_transform_grasp import points2labels
from model.nms.nms_wrapper import nms
from model.basenet.resnet import Bottleneck

from model.fully_conv_grasp.classifier import _Classifier
from model.fully_conv_grasp.grasp_proposal_target import _GraspTargetLayer
from model.fully_conv_grasp.generate_grasp_anchors import generate_oriented_anchors

import time
import pdb
from utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from basenet.resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from model.rpn.bbox_transform import bbox_overlaps, bbox_overlaps_batch
import copy

import FasterRCNN

from model.op2l.object_pairing_layer import _ObjPairLayer
from model.op2l.rois_pair_expanding_layer import  _RoisPairExpandingLayer
from model.op2l.op2l import _OP2L
from model.fully_conv_grasp.bbox_transform_grasp import labels2points, grasp_decode

class _All_in_One(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_All_in_One, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self._fs = cfg.FCGN.FEAT_STRIDE[0]
        # for resnet
        if self.dout_base_model is None:
            if self._fs == 16:
                self.dout_base_model = 256 * self.expansions
            elif self._fs == 32:
                self.dout_base_model = 512 * self.expansions

        # loss
        self.VMRN_obj_loss_cls = 0
        self.VMRN_obj_loss_bbox = 0

        # define rpn
        self.VMRN_obj_rpn = _RPN(self.dout_base_model,
                             anchor_scales=cfg.RCNN_COMMON.ANCHOR_SCALES,
                             anchor_ratios=cfg.RCNN_COMMON.ANCHOR_RATIOS,
                             feat_stride=self._fs)

        self.VMRN_obj_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.VMRN_obj_roi_pool = _RoIPooling(cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE, 1.0/16.0)
        self.VMRN_obj_roi_align = RoIAlignAvg(cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.RCNN_COMMON.POOLING_SIZE * 2 if cfg.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL else cfg.RCNN_COMMON.POOLING_SIZE
        self.VMRN_obj_roi_crop = _RoICrop()

        self._isex = cfg.TRAIN.VMRN.ISEX
        self.VMRN_rel_op2l = _OP2L(cfg.VMRN.OP2L_POOLING_SIZE, cfg.VMRN.OP2L_POOLING_SIZE, 1.0 / 16.0, self._isex)

        self._train_iter_conter = 0

        self._MGN_as = cfg.FCGN.ANCHOR_SCALES
        self._MGN_ar = cfg.FCGN.ANCHOR_RATIOS
        self._MGN_aa = cfg.FCGN.ANCHOR_ANGLES

        # grasp detection components
        self.MGN_classifier = _Classifier(self.dout_base_model, 5, self._MGN_as,
                                          self._MGN_ar, self._MGN_aa)
        self.MGN_proposal_target = _GraspTargetLayer(self._fs, self._MGN_ar,
                                                     self._MGN_as, self._MGN_aa)
        self._MGN_anchors = torch.from_numpy(generate_oriented_anchors(base_size=self._fs,
                                                       scales=np.array(self._MGN_as),
                                                       ratios=np.array(self._MGN_ar),
                                                       angles=np.array(self._MGN_aa))).float()
        self._MGN_num_anchors = self._MGN_anchors.size(0)
        # [x1, y1, x2, y2] -> [xc, yc, w, h]
        self._MGN_anchors = torch.cat([
            0 * self._MGN_anchors[:, 0:1],
            0 * self._MGN_anchors[:, 1:2],
            self._MGN_anchors[:, 2:3] - self._MGN_anchors[:, 0:1] + 1,
            self._MGN_anchors[:, 3:4] - self._MGN_anchors[:, 1:2] + 1,
            self._MGN_anchors[:, 4:5]
        ], dim=1)
        self._MGN_USE_POOLED_FEATS = cfg.MGN.USE_POOLED_FEATS

    def rel_forward_with_gtbox(self, im_data, gt):
        # object detection
        if self.training:
            self._train_iter_conter += 1
        self.batch_size = im_data.size(0)

        gt_boxes = gt['boxes']
        num_boxes = gt['num_boxes']
        im_info = gt['im_info']

        # feed image data to base model to obtain base feature map
        base_feat = self.VMRN_base(im_data)

        # offline data
        obj_rois = []
        for i in range(self.batch_size):
            obj_rois.append(torch.cat([(i * torch.ones(num_boxes[i].item(), 1)).type_as(gt_boxes),
                                       (gt_boxes[i][:num_boxes[i]][:, 0:4])], 1))
        obj_rois = torch.cat(obj_rois, dim=0)
        obj_num = num_boxes
        obj_rois = Variable(obj_rois)
        obj_labels = gt_boxes[:, :, -1].view(-1)
        obj_labels = obj_labels[obj_labels > 0]

        # relationship forward
        if (obj_num > 1).sum().item() > 0:
            # filter out the detection of only one object instance
            obj_pair_feat = self.VMRN_rel_op2l(base_feat, obj_rois, self.batch_size, obj_num)
            obj_pair_feat = obj_pair_feat.detach()
            obj_pair_feat = self._rel_head_to_tail(obj_pair_feat)
            rel_cls_score = self.VMRN_rel_cls_score(obj_pair_feat)
            rel_cls_prob = F.softmax(rel_cls_score)
            VMRN_rel_loss_cls = 0
            if (not cfg.TEST.VMRN.ISEX) and cfg.TRAIN.VMRN.ISEX:
                rel_cls_prob = rel_cls_prob[::2, :]
        else:
            VMRN_rel_loss_cls = 0
            # no detected relationships
            rel_cls_prob = Variable(torch.Tensor([]).type_as(base_feat))

        # grasp detection forward
        pooled_feat = self._roi_pooing(base_feat, obj_rois)
        grasp_feat = self._MGN_head_to_tail(pooled_feat)
        grasp_pred = self.MGN_classifier(grasp_feat)
        # bs*N x K*A x 5, bs*N x K*A x 2
        grasp_loc, grasp_conf = grasp_pred
        grasp_all_anchors = self._generate_anchors(grasp_conf.size(1), grasp_conf.size(2), obj_rois.unsqueeze(0))
        # filter out negative samples
        grasp_all_anchors = grasp_all_anchors.type_as(gt_boxes)
        # reshape grasp_loc and grasp_conf
        grasp_loc = grasp_loc.contiguous().view(grasp_loc.size(0), -1, 5)
        grasp_conf = grasp_conf.contiguous().view(grasp_conf.size(0), -1, 2)
        grasp_batch_size = grasp_loc.size(0)
        # bs*N x K*A x 2
        grasp_prob = F.softmax(grasp_conf, 2)

        # relationship postprocess
        rel_result = None
        if not self.training:
            if obj_rois.numel() > 0:
                pred_boxes = obj_rois.data[:, 1:5].clone()
                pred_boxes[:, 0::2] /= im_info[0][3].item()
                pred_boxes[:, 1::2] /= im_info[0][2].item()
                rel_result = (pred_boxes, obj_labels, rel_cls_prob.data)
            else:
                rel_result = (obj_rois.data, obj_labels, rel_cls_prob.data)

        # grasp postprocess
        grasp_box_deltas = grasp_loc.data
        grasp_scores = grasp_prob.data
        grasp_box_deltas = grasp_box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
        grasp_box_deltas = grasp_box_deltas.view(grasp_all_anchors.size())
        # bs*N x K*A x 5
        grasp_pred = grasp_decode(grasp_box_deltas, grasp_all_anchors)
        # bs*N x K*A x 1
        rois_w = (obj_rois[:, 3] - obj_rois[:, 1]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 0:1])
        rois_h = (obj_rois[:, 4] - obj_rois[:, 2]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 1:2])
        keep_mask = (grasp_pred[:, :, 0:1] > 0) & (grasp_pred[:, :, 1:2] > 0) & \
                    (grasp_pred[:, :, 0:1] < rois_w) & (grasp_pred[:, :, 1:2] < rois_h)
        grasp_scores = (grasp_scores).contiguous(). \
            view(obj_rois.size(0), -1, 2)
        # bs*N x 1 x 1
        xleft = obj_rois[:, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
        ytop = obj_rois[:, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
        # rois offset
        grasp_pred[:, :, 0:1] = grasp_pred[:, :, 0:1] + xleft
        grasp_pred[:, :, 1:2] = grasp_pred[:, :, 1:2] + ytop
        # bs*N x K*A x 8
        grasp_pred_boxes = labels2points(grasp_pred).contiguous().view(obj_rois.size(0), -1, 8)
        # bs*N x K*A
        grasp_pos_scores = grasp_scores[:, :, 1]
        # bs*N x K*A
        _, grasp_score_idx = torch.sort(grasp_pos_scores, dim=1, descending=True)
        _, grasp_idx_rank = torch.sort(grasp_score_idx)
        # bs*N x K*A mask
        topn_grasp = 5
        grasp_maxscore_mask = (grasp_idx_rank < topn_grasp)
        # bs*N x topN
        grasp_maxscores = grasp_scores[:, :, 1][grasp_maxscore_mask].contiguous(). \
            view(obj_rois.size()[:1] + (topn_grasp,))
        # scores = scores * grasp_maxscores[:, :, 0:1]
        # bs*N x topN x 8
        grasp_pred_boxes = grasp_pred_boxes[grasp_maxscore_mask].view(obj_rois.size()[:1] + (topn_grasp, 8))
        grasp_pred_boxes[:, :, 0::2] /= im_info[0][3].item()
        grasp_pred_boxes[:, :, 1::2] /= im_info[0][2].item()
        grasp_pred_boxes = torch.cat([grasp_pred_boxes, grasp_maxscores.unsqueeze(-1)], dim = -1)
        return rel_result, grasp_pred_boxes

    def forward(self, im_data, gt):
        # object detection
        if self.training:
            self._train_iter_conter += 1
        self.batch_size = im_data.size(0)

        gt_boxes = gt['boxes']
        gt_grasps = gt['grasps']
        gt_grasp_inds = gt['grasp_inds']
        num_boxes = gt['num_boxes']
        num_grasps = gt['num_grasps']
        im_info = gt['im_info']
        rel_mat = gt['rel_mat']

        # feed image data to base model to obtain base feature map
        base_feat = self.VMRN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.VMRN_obj_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # rois preprocess
        if self.training:
            obj_det_rois = rois[:,:cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET]
            roi_data = self.VMRN_obj_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            grasp_rois = rois.clone()
            rois = torch.cat([obj_det_rois,rois],1)

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        pooled_feat = self._roi_pooing(base_feat, rois)
        print(pooled_feat.mean().item(), pooled_feat.max().item(), pooled_feat.min().item(), pooled_feat.std().item())

        if self.training:
            pooled_feat_shape = pooled_feat.size()
            pooled_feat = pooled_feat.contiguous().view((self.batch_size, -1) + pooled_feat_shape[1:])
            grasp_feat = pooled_feat[:, cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET:].\
                contiguous().view((-1,) + pooled_feat_shape[1:])
            pooled_feat = pooled_feat.view(pooled_feat_shape)
            if self._MGN_USE_POOLED_FEATS:
                rois_overlaps = bbox_overlaps_batch(rois[:, cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET:], gt_boxes)
                # bs x N_{rois}
                _, rois_inds = torch.max(rois_overlaps, dim=2)
                rois_inds += 1
                grasp_rois_mask = rois_label.view(-1) > 0
            else:
                raise NotImplementedError

        ############################################
        # OBJECT DETECTION
        ############################################
        # feed pooled features to top model
        VMRN_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.VMRN_obj_bbox_pred(VMRN_feat)
        if self.training:
            if self.class_agnostic:
                bbox_pred = bbox_pred.contiguous().view(self.batch_size, -1, 4)
            else:
                bbox_pred = bbox_pred.contiguous().view(self.batch_size, -1, 4 * self.n_classes)
            obj_det_bbox_pred = bbox_pred[:,:cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET]
            bbox_pred = bbox_pred[:,cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET:]
            if self.class_agnostic:
                obj_det_bbox_pred = obj_det_bbox_pred.contiguous().view(-1, 4)
                bbox_pred = bbox_pred.contiguous().view(-1, 4)
            else:
                obj_det_bbox_pred = obj_det_bbox_pred.contiguous().view(-1, 4 * self.n_classes)
                bbox_pred = bbox_pred.contiguous().view(-1, 4 * self.n_classes)

        # compute object classification probability
        cls_score = self.VMRN_obj_cls_score(VMRN_feat)
        cls_prob = F.softmax(cls_score)
        if self.training:
            cls_score = cls_score.contiguous().view(self.batch_size, -1, self.n_classes)
            obj_det_cls_score = cls_score[:, :cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET]
            cls_score = cls_score[:, cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET:]
            obj_det_cls_score = obj_det_cls_score.contiguous().view(-1, self.n_classes)
            cls_score = cls_score.contiguous().view(-1, self.n_classes)

            cls_prob = cls_prob.contiguous().view(self.batch_size, -1, self.n_classes)
            obj_det_cls_prob = cls_prob[:, :cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET]
            cls_prob = cls_prob[:, cfg.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET:]
            obj_det_cls_prob = obj_det_cls_prob.contiguous().view(-1, self.n_classes)
            cls_prob = cls_prob.contiguous().view(-1, self.n_classes)

        VMRN_obj_loss_cls = 0
        VMRN_obj_loss_bbox = 0

        # compute object detector loss
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        if self.training:
            # classification loss
            VMRN_obj_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            VMRN_obj_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        ############################################
        # VISUAL MANIPULATION RELATIONSHIP
        ############################################
        # online data
        if self.training:
            if self._train_iter_conter > cfg.TRAIN.VMRN.ONLINEDATA_BEGIN_ITER:
                obj_rois, obj_num = self._obj_det(obj_det_rois,
                        obj_det_cls_prob.contiguous().view(self.batch_size, -1, self.n_classes),
                        obj_det_bbox_pred.contiguous().view(self.batch_size,
                                -1, 4 if self.class_agnostic else 4 * self.n_classes),
                        self.batch_size, im_info)
                obj_rois = obj_rois.type_as(gt_boxes)
                obj_num = obj_num.type_as(num_boxes)
            else:
                obj_rois = torch.FloatTensor([]).type_as(gt_boxes)
                obj_num = torch.LongTensor([]).type_as(num_boxes)
            obj_labels = None
        else:
            # when testing, this is object detection results
            # TODO: SUPPORT MULTI-IMAGE BATCH
            obj_rois, obj_num = self._obj_det(rois,
                    cls_prob.contiguous().view(self.batch_size, -1, self.n_classes),
                    bbox_pred.contiguous().view(self.batch_size,
                                -1, 4 if self.class_agnostic else 4 * self.n_classes),
                    self.batch_size, im_info)
            if obj_rois.numel() > 0:
                obj_labels = obj_rois[:,5]
                obj_rois = obj_rois[:,:5]
                obj_rois = obj_rois.type_as(gt_boxes)
                obj_num = obj_num.type_as(num_boxes)
            else:
                # there is no object detected
                obj_labels = torch.Tensor([]).type_as(gt_boxes).long()
                obj_rois = obj_rois.type_as(gt_boxes)
                obj_num = obj_num.type_as(num_boxes)

        # offline data
        if self.training:
            for i in range(self.batch_size):
                obj_rois = torch.cat([obj_rois,
                                  torch.cat([(i * torch.ones(num_boxes[i].item(),1)).type_as(gt_boxes),
                                             (gt_boxes[i][:num_boxes[i]][:,0:4])],1)
                                  ])
                obj_num = torch.cat([obj_num,torch.Tensor([num_boxes[i]]).type_as(obj_num)])

        obj_rois = Variable(obj_rois)

        if obj_rois.size(0)>1:
            # filter out the detection of only one object instance
            obj_pair_feat = self.VMRN_rel_op2l(base_feat, obj_rois, self.batch_size, obj_num)
            # obj_pair_feat = obj_pair_feat.detach()
            obj_pair_feat = self._rel_head_to_tail(obj_pair_feat)
            rel_cls_score = self.VMRN_rel_cls_score(obj_pair_feat)

            rel_cls_prob = F.softmax(rel_cls_score)

            VMRN_rel_loss_cls = 0
            if self.training:
                self.rel_batch_size = rel_cls_prob.size(0)

                obj_pair_rel_label = self._generate_rel_labels(obj_rois, gt_boxes, obj_num, rel_mat)
                obj_pair_rel_label = obj_pair_rel_label.type_as(gt_boxes).long()

                rel_not_keep = (obj_pair_rel_label == 0)
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
            rel_cls_prob = Variable(torch.Tensor([]).type_as(obj_labels))

        rel_result = None
        if not self.training:
            if obj_rois.numel() > 0:
                pred_boxes = obj_rois.data[:,1:5]
                pred_boxes[:, 0::2] /= im_info[0][3].item()
                pred_boxes[:, 1::2] /= im_info[0][2].item()
                rel_result = (pred_boxes, obj_labels, rel_cls_prob.data)
            else:
                rel_result = (obj_rois.data, obj_labels, rel_cls_prob.data)

        ############################################
        # ROI-BASED GRASP DETECTION
        ############################################
        if self.training:
            if (grasp_rois_mask > 0).sum().item() > 0:
                grasp_feat = self._MGN_head_to_tail(grasp_feat[grasp_rois_mask])
            else:
                # when there are no one positive rois:
                grasp_loc = Variable(torch.Tensor([]).type_as(gt_grasps))
                grasp_prob = Variable(torch.Tensor([]).type_as(gt_grasps))
                grasp_bbox_loss = Variable(torch.Tensor([0]).type_as(VMRN_obj_loss_bbox))
                grasp_cls_loss = Variable(torch.Tensor([0]).type_as(VMRN_obj_loss_cls))
                grasp_conf_label = torch.Tensor([-1]).type_as(rois_label)
                grasp_all_anchors = torch.Tensor([]).type_as(gt_grasps)
                return rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_bbox, \
                    VMRN_obj_loss_cls, VMRN_obj_loss_bbox, VMRN_rel_loss_cls, rois_label, \
                   grasp_loc, grasp_prob, grasp_bbox_loss , grasp_cls_loss, grasp_conf_label, grasp_all_anchors
        else:
            grasp_feat = self._MGN_head_to_tail(pooled_feat)

        grasp_pred = self.MGN_classifier(grasp_feat)
        # bs*N x K*A x 5, bs*N x K*A x 2
        grasp_loc, grasp_conf = grasp_pred

        # generate anchors
        # bs*N x K*A x 5
        if self.training:
            grasp_all_anchors = self._generate_anchors(grasp_conf.size(1), grasp_conf.size(2), grasp_rois)
        else:
            grasp_all_anchors = self._generate_anchors(grasp_conf.size(1), grasp_conf.size(2), rois)
        # filter out negative samples
        grasp_all_anchors = grasp_all_anchors.type_as(gt_grasps)
        if self.training:
            grasp_all_anchors = grasp_all_anchors[grasp_rois_mask]
            # bs*N x 1 x 1
            rois_w = (grasp_rois[:, :, 3] - grasp_rois[:, :, 1]).data.view(-1).unsqueeze(1).unsqueeze(2)
            rois_h = (grasp_rois[:, :, 4] - grasp_rois[:, :, 2]).data.view(-1).unsqueeze(1).unsqueeze(2)
            rois_w = rois_w[grasp_rois_mask]
            rois_h = rois_h[grasp_rois_mask]
            # bs*N x 1 x 1
            fsx = rois_w / grasp_conf.size(1)
            fsy = rois_h / grasp_conf.size(2)
            # bs*N x 1 x 1
            xleft = grasp_rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
            ytop = grasp_rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
            xleft = xleft[grasp_rois_mask]
            ytop = ytop[grasp_rois_mask]

        # reshape grasp_loc and grasp_conf
        grasp_loc = grasp_loc.contiguous().view(grasp_loc.size(0), -1, 5)
        grasp_conf = grasp_conf.contiguous().view(grasp_conf.size(0), -1, 2)
        grasp_batch_size = grasp_loc.size(0)

        # bs*N x K*A x 2
        grasp_prob = F.softmax(grasp_conf, 2)

        grasp_bbox_loss = 0
        grasp_cls_loss = 0
        grasp_conf_label = None
        if self.training:
            # inside weights indicate which bounding box should be regressed
            # outside weidhts indicate two things:
            # 1. Which bounding box should contribute for classification loss,
            # 2. Balance cls loss and bbox loss
            grasp_gt_xywhc = points2labels(gt_grasps)
            # bs*N x N_{Gr_gt} x 5
            grasp_gt_xywhc = self._assign_rois_grasps(grasp_gt_xywhc, gt_grasp_inds, rois_inds)
            # filter out negative samples
            grasp_gt_xywhc = grasp_gt_xywhc[grasp_rois_mask]

            # absolute coords to relative coords
            grasp_gt_xywhc[:, :, 0:1] -= xleft
            grasp_gt_xywhc[:, :, 0:1] = torch.clamp(grasp_gt_xywhc[:, :, 0:1], min = 0)
            grasp_gt_xywhc[:, :, 0:1] = torch.min(grasp_gt_xywhc[:, :, 0:1], rois_w)
            grasp_gt_xywhc[:, :, 1:2] -= ytop
            grasp_gt_xywhc[:, :, 1:2] = torch.clamp(grasp_gt_xywhc[:, :, 1:2], min = 0)
            grasp_gt_xywhc[:, :, 1:2] = torch.min(grasp_gt_xywhc[:, :, 1:2], rois_h)

            # grasp training data
            grasp_loc_label, grasp_conf_label, grasp_iw, grasp_ow = self.MGN_proposal_target(grasp_conf,
                                        grasp_gt_xywhc, grasp_all_anchors,xthresh = fsx/2, ythresh = fsy/2)

            grasp_keep = Variable(grasp_conf_label.view(-1).ne(-1).nonzero().view(-1))
            grasp_conf = torch.index_select(grasp_conf.view(-1, 2), 0, grasp_keep.data)
            grasp_conf_label = torch.index_select(grasp_conf_label.view(-1), 0, grasp_keep.data)
            grasp_cls_loss = F.cross_entropy(grasp_conf, grasp_conf_label)

            grasp_iw = Variable(grasp_iw)
            grasp_ow = Variable(grasp_ow)
            grasp_loc_label = Variable(grasp_loc_label)
            grasp_bbox_loss = _smooth_l1_loss(grasp_loc, grasp_loc_label, grasp_iw, grasp_ow, dim = [2,1])

        return rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_bbox, \
                VMRN_obj_loss_cls, VMRN_obj_loss_bbox, VMRN_rel_loss_cls, rois_label, \
                grasp_loc, grasp_prob, grasp_bbox_loss , grasp_cls_loss, grasp_conf_label, grasp_all_anchors

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

    def _obj_det(self, rois, cls_prob, bbox_pred, batch_size, im_info):
        det_results = torch.Tensor([]).type_as(rois)
        obj_num = []
        if not self.training:
            det_labels = torch.Tensor([]).type_as(rois).long()

        for i in range(batch_size):
            cur_rois = rois[i:i+1]
            cur_cls_prob = cls_prob[i:i+1]
            cur_bbox_pred = bbox_pred[i:i+1]
            cur_im_info = im_info[i:i+1]
            obj_boxes = self._get_single_obj_det_results(cur_rois, cur_cls_prob, cur_bbox_pred, cur_im_info)
            obj_num.append(obj_boxes.size(0))
            if obj_num[-1] > 0 :
                det_results = torch.cat([det_results,
                                     torch.cat([i * torch.ones(obj_boxes.size(0), 1).type_as(det_results),
                                                obj_boxes], 1)
                                     ], 0)
        return det_results, torch.LongTensor(obj_num)

    # TODO: support batch detection
    def _get_single_obj_det_results(self, rois, cls_prob, bbox_pred, im_info):
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        results = []
        if cfg.TEST.COMMON.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).type_as(box_deltas) \
                                     + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).type_as(box_deltas)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).type_as(box_deltas) \
                                     + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).type_as(box_deltas)
                    box_deltas = box_deltas.view(1, -1, 4 * self.n_classes)
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        thresh = 0
        for j in xrange(1, self.n_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.COMMON.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                final_keep = torch.nonzero(cls_dets[:, -1] > cfg.TEST.COMMON.OBJ_DET_THRESHOLD).squeeze()
                result = cls_dets[final_keep]
                # unsqueeze result to 2 dims
                if result.numel()>0 and result.dim() == 1:
                    result = result.unsqueeze(0)
                # in testing, concat object labels
                if final_keep.numel() > 0:
                    if self.training:
                        result = result[:,:4]
                    else:
                        result = torch.cat([result[:,:4],
                                j * torch.ones(result.size(0),1).type_as(result)],1)
                if result.numel() > 0:
                    results.append(result)

        if len(results):
            final = torch.cat(results, 0)
        else:
            final = torch.Tensor([]).type_as(rois)

        return final

    def _roi_pooing(self, base_feat, rois):
        # do roi pooling based on predicted rois
        if cfg.RCNN_COMMON.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.VMRN_obj_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.RCNN_COMMON.POOLING_MODE == 'align':
            pooled_feat = self.VMRN_obj_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.RCNN_COMMON.POOLING_MODE == 'pool':
            pooled_feat = self.VMRN_obj_roi_pool(base_feat, rois.view(-1, 5))
        return pooled_feat

    def _generate_anchors(self, feat_height, feat_width, rois):
        # feat stride x, dim: bs x N x 1
        fsx = ((rois[:, :, 3:4] - rois[:, :, 1:2]) / feat_width).data.cpu().numpy()
        # feat stride y, dim: bs x N x 1
        fsy = ((rois[:, :, 4:5] - rois[:, :, 2:3]) / feat_height).data.cpu().numpy()

        # bs x N x W, center point of each cell
        shift_x = np.arange(0, feat_width) * fsx + fsx / 2
        # bs x N x H, center point of each cell
        shift_y = np.arange(0, feat_height) * fsy + fsy / 2

        # [bs x N x W x H (x coords), bs x N x W x H (y coords)]
        shift_x, shift_y = (
            np.repeat(np.expand_dims(shift_x, 2), shift_y.shape[2], axis=2),
            np.repeat(np.expand_dims(shift_y, 3), shift_x.shape[2], axis=3)
        )
        # bs x N x W*H x 2
        shifts = torch.cat([torch.from_numpy(shift_x).unsqueeze(4), torch.from_numpy(shift_y).unsqueeze(4)],4)
        shifts = shifts.contiguous().view(rois.size(0), rois.size(1), -1, 2)
        # bs x N x W*H x 5
        shifts = torch.cat([
            shifts,
            torch.zeros(shifts.size()[:-1] + (3,)).type_as(shifts)
        ], dim = -1)
        shifts = shifts.contiguous().float()

        A = self._MGN_num_anchors
        # K = W*H
        K = shifts.size(-2)

        # anchors = self._anchors.view(1, A, 5) + shifts.view(1, K, 5).permute(1, 0, 2).contiguous()

        if cfg.MGN.USE_ADAPTIVE_ANCHOR:
            # bs x N x 1
            anchor_size = torch.sqrt(torch.pow(torch.from_numpy(fsx), 2) +
                                     torch.pow(torch.from_numpy(fsy), 2)).type_as(shifts)
            anchor_size = anchor_size * self._MGN_as[0]
            # bs x N x 1 x 1 x 5
            anchor_size = anchor_size.unsqueeze(3).unsqueeze(4)
            anchors = torch.zeros(anchor_size.size()[:-1] + (5,))
            anchors[:, :, :, :, 2:4] = anchor_size
            # 1 x A x 5
            angle = torch.Tensor(cfg.FCGN.ANCHOR_ANGLES).contiguous().view(1, 1, -1).permute(0, 2, 1).type_as(shifts)
            anchor_angle = torch.zeros(1, angle.size(1), 5)
            anchor_angle[:, :, 4:5] = angle
            # bs x N x 1 x 1 x 5 + 1 x A x 5 -> bs x N x 1 x A x 5
            anchors = anchors + anchor_angle
            # bs x N x 1 x A x 5 + bs x N x K x 1 x 5 -> bs x N x K x A x 5
            anchors = anchors + shifts.unsqueeze(-2)
        else:
            # bs x N x K x A x 5
            anchors = self._MGN_anchors.view(1, A, 5) + shifts.unsqueeze(-2)
        # bs*N x K*A x 5
        anchors = anchors.view(rois.size(0) * rois.size(1) , K * A, 5)

        return anchors

    def _assign_rois_grasps(self, grasp, grasp_inds, rois_inds):
        """
        :param grasp: bs x N_{Gr_gt} x Gdim
        :param grasp_inds: bs x N_{Gr_gt}
        :param rois_inds: bs x N_{rois}
        :return: grasp: bs x N_{rois} x N_{Gr_gt} x Gdim
        """
        # bs x N x N_{Gr_gt} x 1
        grasp_mask = (grasp_inds.unsqueeze(-2) == rois_inds.unsqueeze(-1)).unsqueeze(3).float()
        # bs x 1 x N_{Gr_gt} x 5
        grasp = grasp.unsqueeze(1)
        # bs*N x N_{Gr_gt} x 5
        grasp_out = (grasp_mask * grasp).contiguous().\
            view(rois_inds.size(0)*rois_inds.size(1), grasp_inds.size(1), -1)
        return grasp_out

    def _init_weights(self):
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

        normal_init(self.VMRN_obj_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_obj_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_obj_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_obj_cls_score, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_obj_bbox_pred, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.fc1, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.fc2, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.outlayer, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)

        normal_init(self.MGN_classifier.conf, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.MGN_classifier.loc, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def resume_iter(self, epoch, iter_per_epoch):
        self._train_iter_conter = epoch * iter_per_epoch

class resnet(_All_in_One):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):

        self.num_layers = num_layers
        self.model_path = 'data/pretrained_model/resnet' + str(num_layers) + '_caffe.pth'

        self.dout_base_model = None
        if num_layers == 18 or num_layers == 34:
            self.expansions = 1
        elif num_layers == 50 or num_layers == 101 or num_layers == 152:
            self.expansions = 4
        else:
            assert 0, "network not defined"

        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        super(resnet, self).__init__(classes, class_agnostic)

    def _init_modules(self):
        resnet = resnet101()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet.
        self.VMRN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                       resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)

        self.VMRN_obj_top = nn.Sequential(resnet.layer4)

        # VMRN layers
        if cfg.VMRN.SHARE_WEIGHTS:
            self.VMRN_rel_top = copy.deepcopy(nn.Sequential(resnet.layer4))
        else:
            self.VMRN_rel_top_o1 = copy.deepcopy(nn.Sequential(resnet.layer4))
            self.VMRN_rel_top_o2 = copy.deepcopy(nn.Sequential(resnet.layer4))
            self.VMRN_rel_top_union = copy.deepcopy(nn.Sequential(resnet.layer4))

        self.VMRN_obj_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.VMRN_obj_bbox_pred = nn.Linear(2048, 4)
        else:
            self.VMRN_obj_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        self.VMRN_rel_cls_score = vmrn_rel_classifier(2048 * 3)

        # Fix blocks
        for p in self.VMRN_base[0].parameters(): p.requires_grad = False
        for p in self.VMRN_base[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.VMRN_base[6].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.VMRN_base[5].parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.VMRN_base[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.VMRN_base.apply(set_bn_fix)
        self.VMRN_obj_top.apply(set_bn_fix)

        self.MGN_top = nn.Sequential(
            Bottleneck(256 * self.expansions, 64 * self.expansions),
            Bottleneck(256 * self.expansions, 64 * self.expansions),
            Bottleneck(256 * self.expansions, 64 * self.expansions)
        )

        # initialize grasp top
        def kaiming_init(m):
            def xavier(param):
                init.kaiming_normal(param, nonlinearity='relu')

            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)

        self.MGN_top.apply(kaiming_init)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.VMRN_base.eval()
            self.VMRN_base[5].train()
            self.VMRN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.VMRN_base.apply(set_bn_eval)
            self.VMRN_obj_top.apply(set_bn_eval)

            if cfg.VMRN.SHARE_WEIGHTS:
                self.VMRN_rel_top.apply(set_bn_eval)
            else:
                self.VMRN_rel_top_o1.apply(set_bn_eval)
                self.VMRN_rel_top_o2.apply(set_bn_eval)
                self.VMRN_rel_top_union.apply(set_bn_eval)

            self.MGN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.VMRN_obj_top(pool5).mean(3).mean(2)
        return fc7

    def _rel_head_to_tail(self, pooled_pair):
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

    def _MGN_head_to_tail(self, feats):
        return self.MGN_top(feats)

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

