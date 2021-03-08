# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import torch
import torch.nn as nn
from model.utils.config import cfg
from model.SSD import SSD
from model.Detectors import VMRN

import numpy as np

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

class SSD_VMRN(SSD, VMRN):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv3', 'conv4'), pretrained = True):
        super(SSD_VMRN, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)

    def forward(self, data_batch):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        x = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]
        rel_mat = data_batch[4]

        if self.training:
            self.iter_counter += 1

        sources = []

        base_feat, x = self.FeatExt(x)
        base_feat = self.L2Norm(base_feat)
        sources.append(base_feat)

        for m in self.extra_conv:
            x = m(x)
            sources.append(x)

        loc, conf = self._get_obj_det_result(sources)
        SSD_loss_cls, SSD_loss_bbox = 0, 0
        if self.training:
            predictions = (
                loc,
                conf,
                self.priors.type_as(loc)
            )
            SSD_loss_bbox, SSD_loss_cls = self.criterion(predictions, gt_boxes, num_boxes)
        conf = self.softmax(conf)

        # generate object RoIs.
        obj_rois, obj_num = torch.Tensor([]).type_as(loc), torch.Tensor([]).type_as(num_boxes)
        # online data
        if not self.training or (cfg.TRAIN.VMRN.TRAINING_DATA == 'all' or 'online'):
            obj_rois, obj_num = self._object_detection(self.priors.type_as(loc), conf, loc, self.batch_size,
                                                       im_info.data)
        # offline data
        if self.training and (cfg.TRAIN.VMRN.TRAINING_DATA == 'all' or 'offline'):
            for i in range(self.batch_size):
                img_ind = (i * torch.ones(num_boxes[i].item(), 1)).type_as(gt_boxes)
                obj_rois = torch.cat([obj_rois, torch.cat([img_ind, (gt_boxes[i][:num_boxes[i]])], 1)])
            obj_num = torch.cat([obj_num, num_boxes])

        obj_labels = torch.Tensor([]).type_as(gt_boxes).long()
        if obj_rois.size(0) > 0:
            obj_labels = obj_rois[:, 5]
            obj_rois = obj_rois[:, :5]

        VMRN_rel_loss_cls, reg_loss = 0, 0
        if (obj_num > 1).sum().item() > 0:
            rel_cls_score, rel_cls_prob, reg_loss = self._get_rel_det_result(base_feat, obj_rois, obj_num, im_info)
            if self.training:
                obj_pair_rel_label = self._generate_rel_labels(obj_rois, gt_boxes, obj_num, rel_mat,
                                                               rel_cls_prob.size(0))
                VMRN_rel_loss_cls = self._rel_det_loss_comp(obj_pair_rel_label.type_as(gt_boxes).long(), rel_cls_score)
            else:
                rel_cls_prob = self._rel_cls_prob_post_process(rel_cls_prob)
        else:
            rel_cls_prob = torch.Tensor([]).type_as(conf)

        rel_result = None
        if not self.training:
            if obj_rois.numel() > 0:
                pred_boxes = obj_rois.data[:, 1:5]
                pred_boxes[:, 0::2] /= im_info[0][3].item()
                pred_boxes[:, 1::2] /= im_info[0][2].item()
                rel_result = (pred_boxes, obj_labels, rel_cls_prob.data)
            else:
                rel_result = (obj_rois.data, obj_labels, rel_cls_prob.data)

        return loc, conf, rel_result, SSD_loss_bbox, SSD_loss_cls, VMRN_rel_loss_cls, reg_loss

    def _init_modules(self):
        assert cfg.TRAIN.VMRN.USE_REL_CLS_GRADIENTS, "No gradients are applied to relationship convolutional layers."
        SSD._init_modules(self)
        VMRN._init_modules(self)

    def _init_weights(self):
        SSD._init_weights(self)
        VMRN._init_weights(self)

    def create_architecture(self, object_detector_path=''):
        # TODO: support pretrained object detector here
        pass