# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
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
from object_detector import objectDetector
import time
import pdb
from utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, weights_normal_init,\
    set_bn_eval, set_bn_fix

class fasterRCNN(objectDetector):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained = True):

        super(fasterRCNN, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)
        # loss
        rand_img = torch.Tensor(1, 3, 224, 224)
        rand_feat = self.feat_extractor(rand_img)
        self.dout_base_model = rand_feat.size(1)

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model,
                             anchor_scales=cfg.RCNN_COMMON.ANCHOR_SCALES,
                             anchor_ratios=cfg.RCNN_COMMON.ANCHOR_RATIOS,
                             feat_stride=cfg.RCNN_COMMON.FEAT_STRIDE[0])

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.RCNN_COMMON.POOLING_SIZE * 2 if cfg.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL else cfg.RCNN_COMMON.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.iter_counter = 0

    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        batch_size = im_data.size(0)
        if self.training:
            self.iter_counter += 1

        # feed image data to base model to obtain base feature map
        base_feat = self.feat_extractor(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

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
        # do roi pooling based on predicted rois

        if cfg.RCNN_COMMON.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.RCNN_COMMON.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.RCNN_COMMON.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            if cfg.TRAIN.COMMON.USE_FOCAL_LOSS:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduce=False)
                focal_loss_factor = torch.pow((1 - cls_prob[range(int(cls_prob.size(0))), rois_label])
                                            ,cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA)
                RCNN_loss_cls = torch.mean(RCNN_loss_cls * focal_loss_factor)
            else:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        weights_normal_init(self.RCNN_rpn.RPN_Conv, 0.01)
        weights_normal_init(self.RCNN_rpn.RPN_cls_score, 0.01)
        weights_normal_init(self.RCNN_rpn.RPN_bbox_pred, 0.01)
        weights_normal_init(self.RCNN_cls_score, 0.01)
        weights_normal_init(self.RCNN_bbox_pred, 0.001)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _init_modules(self):
        objectDetector._init_modules(self)
        if self.feat_name[:3] == 'res':
            self._init_modules_resnet()
        elif self.feat_name[:3] == 'vgg':
            self._init_modules_vgg()

    def _init_modules_resnet(self):

        self.RCNN_top = self.feat_extractor.feat_layer["conv5"]
        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        self.RCNN_top.apply(set_bn_fix)

    def _init_modules_vgg(self):

        self.RCNN_top = self.feat_extractor.feat_layer["fc"]
        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def train(self, mode=True):
        objectDetector.train(self, mode = mode)
        if mode:
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        if self.feat_name[:3] == 'res':
            return self._head_to_tail_resnet(pool5)
        elif self.feat_name[:3] == 'vgg':
            return self._head_to_tail_vgg(pool5)

    def _head_to_tail_resnet(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7

    def _head_to_tail_vgg(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)
        return fc7


