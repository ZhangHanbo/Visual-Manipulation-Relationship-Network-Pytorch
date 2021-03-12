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
import torch.nn.init as init
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_overlaps_batch
from model.fcgn.bbox_transform_grasp import points2labels
from model.utils.net_utils import _smooth_l1_loss, _affine_grid_gen, weight_kaiming_init, set_bn_eval, set_bn_fix
from model.basenet.resnet import Bottleneck
import numpy as np
from model.FasterRCNN import fasterRCNN
from model.FCGN import FCGN

import pdb

class MGN(fasterRCNN, FCGN):

    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained = True):
        super(MGN, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)
        self.iter_counter = 0

        self.use_objdet_branch = cfg.TRAIN.COMMON.BBOX_REG

        if not self.use_objdet_branch:
            # if do not use object detection branch, RPN plays the role of object instance detection.
            self.RCNN_rpn.RPN_proposal._include_rois_score = True

        self._fix_fasterRCNN = cfg.MGN.FIX_OBJDET
        if self._fix_fasterRCNN:
            self._fixed_keys = []

    def _grasp_anchor_transform(self):
        return torch.cat([
            0 * self.FCGN_anchors[:, 0:1],
            0 * self.FCGN_anchors[:, 1:2],
            self.FCGN_anchors[:, 2:3] - self.FCGN_anchors[:, 0:1] + 1,
            self.FCGN_anchors[:, 3:4] - self.FCGN_anchors[:, 1:2] + 1,
            self.FCGN_anchors[:, 4:5]
        ], dim=1)

    def _grasp_loss_comp(self, rois, grasp_conf, grasp_loc, grasp_gt, grasp_anchors, fh, fw):

        rois_w = (rois[:, 3] - rois[:, 1]).data.unsqueeze(1).unsqueeze(2)
        rois_h = (rois[:, 4] - rois[:, 2]).data.unsqueeze(1).unsqueeze(2)
        # bs*N x 1 x 1
        fsx = rois_w / fw
        fsy = rois_h / fh
        # bs*N x 1 x 1
        xleft = rois[:, 1].data.unsqueeze(1).unsqueeze(2)
        ytop = rois[:, 2].data.unsqueeze(1).unsqueeze(2)

        # absolute coords to relative coords
        grasp_gt[:, :, 0:1] -= xleft
        grasp_gt[:, :, 0:1] = torch.clamp(grasp_gt[:, :, 0:1], min=0)
        grasp_gt[:, :, 0:1] = torch.min(grasp_gt[:, :, 0:1], rois_w)
        grasp_gt[:, :, 1:2] -= ytop
        grasp_gt[:, :, 1:2] = torch.clamp(grasp_gt[:, :, 1:2], min=0)
        grasp_gt[:, :, 1:2] = torch.min(grasp_gt[:, :, 1:2], rois_h)

        # inside weights indicate which bounding box should be regressed
        # outside weights indicate two things:
        # 1. Which bounding box should contribute for classification loss,
        # 2. Balance cls loss and bbox loss
        # grasp training data
        grasp_loc_label, grasp_conf_label, grasp_iw, grasp_ow = self.FCGN_proposal_target(grasp_conf,
                                                        grasp_gt, grasp_anchors, xthresh=fsx / 2, ythresh=fsy / 2)

        grasp_keep = grasp_conf_label.view(-1).ne(-1).nonzero().view(-1)
        grasp_conf = torch.index_select(grasp_conf.view(-1, 2), 0, grasp_keep.data)
        grasp_conf_label = torch.index_select(grasp_conf_label.view(-1), 0, grasp_keep.data)
        grasp_cls_loss = F.cross_entropy(grasp_conf, grasp_conf_label)

        grasp_bbox_loss = _smooth_l1_loss(grasp_loc, grasp_loc_label, grasp_iw, grasp_ow, dim=[2, 1])
        return grasp_bbox_loss , grasp_cls_loss, grasp_conf_label

    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        gt_grasps = data_batch[3]
        num_boxes = data_batch[4]
        num_grasps = data_batch[5]
        gt_grasp_inds = data_batch[6]
        batch_size = im_data.size(0)
        if self.training:
            self.iter_counter += 1

        # for jacquard dataset, the bounding box labels are set to -1. For training, we set them to 1, which does not
        # affect the training process.
        if self.training:
            if gt_boxes[:, :, -1].sum().item() < 0:
                gt_boxes[:, :, -1] = 1

        for i in range(batch_size):
            if torch.sum(gt_grasp_inds[i]).item() == 0:
                gt_grasp_inds[i, :num_grasps[i].item()] = 1

        # features
        base_feat = self.FeatExt(im_data)

        # generate rois of RCNN
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        if not self.use_objdet_branch:
            rois_scores = rois[:, :, 5:].clone()
            rois = rois[:, :, :5].clone()

        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = \
                self._get_header_train_data(rois, gt_boxes, num_boxes)
        else:
            rois_label, rois_target, rois_inside_ws, rois_outside_ws = None, None, None, None
        pooled_feat = self._roi_pooling(base_feat, rois)

        cls_prob, bbox_pred, RCNN_loss_bbox, RCNN_loss_cls = \
            None, None, torch.tensor([0]).type_as(rois), torch.tensor([0]).type_as(rois)
        if self.use_objdet_branch:
            # object detection branch
            cls_score, cls_prob, bbox_pred = self._get_obj_det_result(pooled_feat)
            if self.training:
                RCNN_loss_bbox, RCNN_loss_cls = self._obj_det_loss_comp(cls_score, cls_prob, bbox_pred, rois_label, rois_target,
                                                                        rois_inside_ws, rois_outside_ws)
            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        else:
            cls_prob = torch.cat([1-rois_scores, rois_scores], dim = -1)

        # grasp detection branch
        # 1. obtaining grasp features of the positive ROIs and prepare grasp training data
        if self.training:
            rois_overlaps = bbox_overlaps_batch(rois, gt_boxes)
            # bs x N_{rois}
            _, rois_inds = torch.max(rois_overlaps, dim=2)
            rois_inds += 1
            grasp_rois_mask = rois_label.view(-1) > 0

            if (grasp_rois_mask > 0).sum().item() > 0:
                grasp_feat = self._MGN_head_to_tail(pooled_feat[grasp_rois_mask])
                grasp_rois = rois.view(-1, 5)[grasp_rois_mask]
                # process grasp ground truth, return: N_{gr_rois} x N_{Gr_gt} x 5
                grasp_gt_xywhc = points2labels(gt_grasps)
                grasp_gt_xywhc = self._assign_rois_grasps(grasp_gt_xywhc, gt_grasp_inds, rois_inds)
                grasp_gt_xywhc = grasp_gt_xywhc[grasp_rois_mask]
            else:
                # when there are no one positive rois, return dummy results
                grasp_loc = torch.tensor([]).type_as(gt_grasps)
                grasp_prob = torch.tensor([]).type_as(gt_grasps)
                grasp_bbox_loss = torch.tensor([0]).type_as(gt_grasps)
                grasp_cls_loss = torch.tensor([0]).type_as(gt_grasps)
                grasp_conf_label = torch.tensor([-1]).type_as(rois_label)
                grasp_all_anchors = torch.tensor([]).type_as(gt_grasps)
                return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
                   grasp_loc, grasp_prob, grasp_bbox_loss , grasp_cls_loss, grasp_conf_label, grasp_all_anchors
        else:
            grasp_feat = self._MGN_head_to_tail(pooled_feat)

        # N_{gr_rois} x W x H x A*5, N_{gr_rois} x W x H x A*2
        grasp_loc, grasp_conf = self.FCGN_classifier(grasp_feat)
        feat_height, feat_width = grasp_conf.size(1), grasp_conf.size(2)
        # reshape grasp_loc and grasp_conf
        grasp_loc = grasp_loc.contiguous().view(grasp_loc.size(0), -1, 5)
        grasp_conf = grasp_conf.contiguous().view(grasp_conf.size(0), -1, 2)
        grasp_prob = F.softmax(grasp_conf, 2)

        # 2. calculate grasp loss
        grasp_bbox_loss, grasp_cls_loss, grasp_conf_label = 0, 0, None
        if self.training:
            # N_{gr_rois} x K*A x 5
            grasp_all_anchors = self._generate_anchors(feat_height, feat_width, grasp_rois)
            grasp_bbox_loss, grasp_cls_loss, grasp_conf_label = self._grasp_loss_comp(grasp_rois,
                grasp_conf, grasp_loc, grasp_gt_xywhc, grasp_all_anchors, feat_height, feat_width)
        else:
            # bs*N x K*A x 5
            grasp_all_anchors = self._generate_anchors(feat_height, feat_width, rois.view(-1, 5))

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
               grasp_loc, grasp_prob, grasp_bbox_loss , grasp_cls_loss, grasp_conf_label, grasp_all_anchors

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

    def _generate_anchors(self, feat_height, feat_width, rois):
        # feat stride x, dim: bs*N x 1
        fsx = ((rois[:, 3:4] - rois[:, 1:2]) / feat_width).data.cpu().numpy()
        # feat stride y, dim: bs*N x 1
        fsy = ((rois[:, 4:5] - rois[:, 2:3]) / feat_height).data.cpu().numpy()

        # bs*N x W, center point of each cell
        shift_x = np.arange(0, feat_width) * fsx + fsx / 2
        # bs*N x H, center point of each cell
        shift_y = np.arange(0, feat_height) * fsy + fsy / 2

        # [bs x N x W x H (x coords), bs x N x W x H (y coords)]
        shift_x, shift_y = (
            np.repeat(np.expand_dims(shift_x, 1), shift_y.shape[1], axis=1),
            np.repeat(np.expand_dims(shift_y, 2), shift_x.shape[1], axis=2)
        )
        # bs*N x W*H x 2
        shifts = torch.cat([torch.from_numpy(shift_x).unsqueeze(3), torch.from_numpy(shift_y).unsqueeze(3)],3)
        shifts = shifts.contiguous().view(rois.size(0), -1, 2).type_as(rois)
        # bs*N x W*H x 5
        shifts = torch.cat([
            shifts,
            torch.zeros(shifts.size()[:-1] + (3,)).type_as(shifts)
        ], dim = -1)
        shifts = shifts.contiguous().float()

        A = self.FCGN_num_anchors
        # K = W*H
        K = shifts.size(-2)

        # anchors = self._anchors.view(1, A, 5) + shifts.view(1, K, 5).permute(1, 0, 2).contiguous()

        if cfg.MGN.USE_ADAPTIVE_ANCHOR:
            # bs*N x 1
            anchor_size = torch.sqrt(torch.pow(torch.from_numpy(fsx), 2) +
                                     torch.pow(torch.from_numpy(fsy), 2)).type_as(shifts)
            anchor_size = anchor_size * self.FCGN_as[0]
            # bs*N x 1 x 1 x 5
            anchor_size = anchor_size.unsqueeze(2).unsqueeze(3)
            anchors = torch.zeros(anchor_size.size()[:-1] + (5,)).type_as(shifts)
            anchors[:, :, :, :, 2:4] = anchor_size
            # 1 x A x 5
            angle = torch.tensor(cfg.FCGN.ANCHOR_ANGLES).contiguous().view(1, 1, -1).permute(0, 2, 1).type_as(shifts)
            anchor_angle = torch.zeros(1, angle.size(1), 5).type_as(shifts)
            anchor_angle[:, :, 4:5] = angle
            # bs*N x 1 x 1 x 5 + 1 x A x 5 -> bs*N x 1 x A x 5
            anchors = anchors + anchor_angle
            # bs*N x 1 x A x 5 + bs*N x K x 1 x 5 -> bs*N x K x A x 5
            anchors = anchors + shifts.unsqueeze(-2)
        else:
            # bs*N x K x A x 5
            anchors = self.FCGN_anchors.view(1, A, 5).type_as(shifts) + shifts.unsqueeze(-2)
        # bs*N x K*A x 5
        anchors = anchors.view(rois.size(0), K * A, 5)

        return anchors

    def create_architecture(self, object_detector_path=''):
        self._init_modules()
        self._init_weights()

        if self._fix_fasterRCNN:
            assert object_detector_path != '', "An pretrained object detector should be specified for VMRN."
            object_detector = torch.load(object_detector_path)
            self._load_and_fix_object_detector(object_detector['model'])

    def _load_and_fix_object_detector(self, object_model):
        """
        To use this function, you need to make sure that all keys in object_model match the ones in the target model.
        """
        self._fixed_keys = set([key.split('.')[0] for key in object_model.keys()])
        self.load_state_dict(object_model, strict=False)
        for name, module in self.named_children():
            if name in self._fixed_keys:
                for p in module.parameters(): p.requires_grad = False

    def _init_modules_resnet(self):
        fasterRCNN._init_modules_resnet(self)
        self.MGN_top = nn.Sequential(
            Bottleneck(self.dout_base_model, self.dout_base_model / 4),
            Bottleneck(self.dout_base_model, self.dout_base_model / 4),
            Bottleneck(self.dout_base_model, self.dout_base_model / 4)
        )

    def _init_modules_vgg(self):
        fasterRCNN._init_modules_vgg(self)
        if 'bn' in self.feat_name:
            self.MGN_top = nn.Sequential(
                nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.dout_base_model),
                nn.ReLU(),
                nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.dout_base_model),
                nn.ReLU(),
                nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.dout_base_model),
                nn.ReLU(),
            )
        else:
            self.MGN_top = nn.Sequential(
                nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, padding=1),
                nn.ReLU(),
            )

    def _init_weights(self):
        fasterRCNN._init_weights(self)
        FCGN._init_weights(self)
        self.MGN_top.apply(weight_kaiming_init)

    def _MGN_head_to_tail(self, feats):
        return self.MGN_top(feats)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode and self._fix_fasterRCNN:
            for name, module in self.named_children():
                if name in self._fixed_keys:
                    module.eval()