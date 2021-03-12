# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# based on code from Jiasen Lu, Jianwei Yang, Ross Girshick
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import cfg
from rpn.rpn import _RPN
from model.roi_layers import RoIAlignAvg, RoIAlignMax, ROIPool
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from Detectors import objectDetector
from utils.net_utils import _smooth_l1_loss, weights_normal_init

class fasterRCNN(objectDetector):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained = True):

        super(fasterRCNN, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)
        ##### Important to set model to eval mode before evaluation ####
        self.FeatExt.eval()
        rand_img = torch.zeros(size=(1, 3, 224, 224))
        rand_feat = self.FeatExt(rand_img)
        self.FeatExt.train()
        self.dout_base_model = rand_feat.size(1)

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model,
                             anchor_scales=cfg.RCNN_COMMON.ANCHOR_SCALES,
                             anchor_ratios=cfg.RCNN_COMMON.ANCHOR_RATIOS,
                             feat_stride=cfg.RCNN_COMMON.FEAT_STRIDE[0])

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg((cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE), 1.0 / 16.0, 0)

        self.grid_size = cfg.RCNN_COMMON.POOLING_SIZE * 2 if cfg.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL else cfg.RCNN_COMMON.POOLING_SIZE

        self.iter_counter = 0

    def _get_header_train_data(self, rois, gt_boxes, num_boxes):
        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        rois_label = rois_label.view(-1).long()
        rois_target = rois_target.view(-1, rois_target.size(2))
        rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
        rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))
        return rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws

    def _roi_pooling(self, base_feat, rois):
        if cfg.RCNN_COMMON.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.RCNN_COMMON.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        return pooled_feat

    def _obj_head_to_tail(self, pool5):
        if self.feat_name[:3] == 'res':
            return self._obj_head_to_tail_resnet(pool5)
        elif self.feat_name[:3] == 'vgg':
            return self._obj_head_to_tail_vgg(pool5)

    def _obj_head_to_tail_resnet(self, pool5):
        fc7 = self.FeatExt.layer4(pool5).mean(3).mean(2)
        return fc7

    def _obj_head_to_tail_vgg(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.FeatExt.feat_layer["fc"](pool5_flat)
        return fc7

    def _get_obj_det_result(self, pooled_feat):
        # feed pooled features to top model
        pooled_feat = self._obj_head_to_tail(pooled_feat)
        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        return cls_score, cls_prob, bbox_pred

    def _obj_det_loss_comp(self, cls_score, cls_prob, bbox_pred, rois_label, rois_target, rois_inside_ws, rois_outside_ws):
        # classification loss
        if cfg.TRAIN.COMMON.USE_FOCAL_LOSS:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduce=False)
            focal_loss_factor = torch.pow((1 - cls_prob[range(int(cls_prob.size(0))), rois_label])
                                          , cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA)
            RCNN_loss_cls = torch.mean(RCNN_loss_cls * focal_loss_factor)
        else:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        if not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
        # bounding box regression L1 loss
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        return RCNN_loss_cls, RCNN_loss_bbox

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _init_weights(self):
        weights_normal_init(self.RCNN_rpn.RPN_Conv, 0.01, 0.)
        weights_normal_init(self.RCNN_rpn.RPN_cls_score, 0.01, 0.)
        weights_normal_init(self.RCNN_rpn.RPN_bbox_pred, 0.01, 0.)
        weights_normal_init(self.RCNN_cls_score, 0.01, 0.)
        weights_normal_init(self.RCNN_bbox_pred, 0.001, 0.)

    def _init_modules(self):

        if self.feat_name[:3] == 'res':
            self._init_modules_resnet()
        elif self.feat_name[:3] == 'vgg':
            self._init_modules_vgg()

    def _init_modules_resnet(self):

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    def _init_modules_vgg(self):

        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        batch_size = im_data.size(0)
        if self.training:
            self.iter_counter += 1

        # feed image data to base model to obtain base feature map
        # base_feat = self.FeatExt(im_data)
        base_feat = self.FeatExt(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = \
                self._get_header_train_data(rois, gt_boxes, num_boxes)
        else:
            rois_label, rois_target, rois_inside_ws, rois_outside_ws = None, None, None, None

        pooled_feat = self._roi_pooling(base_feat, rois)
        cls_score, cls_prob, bbox_pred = self._get_obj_det_result(pooled_feat)

        RCNN_loss_bbox, RCNN_loss_cls = 0, 0
        if self.training:
            RCNN_loss_bbox, RCNN_loss_cls = self._obj_det_loss_comp(cls_score, cls_prob, bbox_pred, rois_label, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label