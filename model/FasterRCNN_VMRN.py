# --------------------------------------------------------
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# Modified from R. B. G.'s faster_rcnn.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from utils.config import cfg

from FasterRCNN import fasterRCNN
from Detectors import VMRN

class fasterRCNN_VMRN(fasterRCNN, VMRN):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained = True):
        super(fasterRCNN_VMRN, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)

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
        cls_score, cls_prob, bbox_pred = self._get_obj_det_result(pooled_feat)
        RCNN_loss_bbox, RCNN_loss_cls = 0, 0
        if self.training:
            RCNN_loss_bbox, RCNN_loss_cls = self._obj_det_loss_comp(cls_score, cls_prob, bbox_pred, rois_label, rois_target,
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

        # generate object RoIs.
        obj_rois, obj_num = torch.Tensor([]).type_as(rois), torch.Tensor([]).type_as(num_boxes)
        # online data
        if not self.training or (cfg.TRAIN.VMRN.TRAINING_DATA in {'all', 'online'}):
            obj_rois, obj_num = self._object_detection(od_rois, od_cls_prob, od_bbox_pred, self.batch_size, im_info.data)
        # offline data
        if self.training and (cfg.TRAIN.VMRN.TRAINING_DATA in {'all', 'offline'}):
            for i in range(self.batch_size):
                img_ind = (i * torch.ones(num_boxes[i].item(),1)).type_as(gt_boxes)
                obj_rois = torch.cat([obj_rois, torch.cat([img_ind, (gt_boxes[i][:num_boxes[i]])],1)])
            obj_num = torch.cat([obj_num, num_boxes])

        obj_labels = torch.Tensor([]).type_as(gt_boxes).long()
        if obj_rois.size(0) > 0:
            obj_labels = obj_rois[:, 5]
            obj_rois = obj_rois[:, :5]

        VMRN_rel_loss_cls = 0
        if (obj_num > 1).sum().item() > 0:
            rel_cls_score, rel_cls_prob = self._get_rel_det_result(base_feat, obj_rois, obj_num)
            if self.training:
                obj_pair_rel_label = self._generate_rel_labels(obj_rois, gt_boxes, obj_num, rel_mat, rel_cls_prob.size(0))
                VMRN_rel_loss_cls = self._rel_det_loss_comp(obj_pair_rel_label.type_as(gt_boxes).long(), rel_cls_score)
            else:
                rel_cls_prob = self._rel_cls_prob_post_process(rel_cls_prob)
        else:
            rel_cls_prob = torch.Tensor([]).type_as(cls_prob)

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

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _init_modules_resnet(self):
        fasterRCNN._init_modules_resnet(self)
        VMRN._init_modules_resnet(self)

    def _init_modules_vgg(self):
        fasterRCNN._init_modules_vgg(self)
        VMRN._init_modules_vgg(self)

    def _init_weights(self):
        fasterRCNN._init_weights(self)
        VMRN._init_weights(self)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        VMRN.train(self, mode)
