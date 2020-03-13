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
import os
from model.ssd.default_bbox_generator import PriorBox
import torch.nn.init as init
from torchvision import models

from model.utils.config import cfg
from basenet.resnet import resnet18,resnet34,resnet50,resnet101,resnet152

from roi_pooling.modules.roi_pool import _RoIPooling
from roi_crop.modules.roi_crop import _RoICrop
from roi_align.modules.roi_align import RoIAlignAvg
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from rpn.rpn import _RPN

from model.fully_conv_grasp.classifier import _Classifier
from model.fully_conv_grasp.grasp_proposal_target import _GraspTargetLayer
from model.fully_conv_grasp.bbox_transform_grasp import \
    points2labels,labels2points,grasp_encode, grasp_decode
from model.rpn.bbox_transform import bbox_overlaps_batch

from model.fully_conv_grasp.bbox_transform_grasp import points2labels
from model.utils.net_utils import _smooth_l1_loss, _affine_grid_gen

from model.basenet.resnet import Bottleneck

import numpy as np
import copy

from model.fully_conv_grasp.generate_grasp_anchors import generate_oriented_anchors

import pdb

class _MGN(nn.Module):

    def __init__(self, classes, class_agnostic):
        super(_MGN, self).__init__()
        self.n_classes = len(classes)
        self._MGN_as = cfg.FCGN.ANCHOR_SCALES
        self._MGN_ar = cfg.FCGN.ANCHOR_RATIOS
        self._MGN_aa = cfg.FCGN.ANCHOR_ANGLES
        self._fs = cfg.FCGN.FEAT_STRIDE[0]
        # for resnet
        if self.dout_base_model is None:
            if self._fs == 16:
                self.dout_base_model = 256 * self.expansions
            elif self._fs == 32:
                self.dout_base_model = 512 * self.expansions

        # grasp detection components
        self.MGN_classifier = _Classifier(self.dout_base_model, 5, self._MGN_as,
                                      self._MGN_ar, self._MGN_aa)
        self.MGN_proposal_target = _GraspTargetLayer(self._fs, self._MGN_ar,
                                                 self._MGN_as, self._MGN_aa)
        self._MGN_anchors = torch.from_numpy(generate_oriented_anchors(base_size=self._fs,
                                    scales=np.array(self._MGN_as), ratios=np.array(self._MGN_ar),
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

        # object detection components
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model,
                             anchor_scales=cfg.RCNN_COMMON.ANCHOR_SCALES,
                             anchor_ratios=cfg.RCNN_COMMON.ANCHOR_RATIOS,
                             feat_stride=cfg.RCNN_COMMON.FEAT_STRIDE[0])

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE, 1.0 / 16.0)
        self.grid_size = cfg.RCNN_COMMON.POOLING_SIZE * 2 if cfg.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL else cfg.RCNN_COMMON.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.iter_counter = 0

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

        # features
        base_feat = self.base(im_data)

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
        # grasp top
        if self.training:
            if self._MGN_USE_POOLED_FEATS:
                rois_overlaps = bbox_overlaps_batch(rois, gt_boxes)
                # bs x N_{rois}
                _, rois_inds = torch.max(rois_overlaps, dim=2)
                rois_inds += 1
                grasp_rois_mask = rois_label.view(-1) > 0
            else:
                raise NotImplementedError
        # bbox top
        bbox_feat = self._RCNN_head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(bbox_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(bbox_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            if cfg.TRAIN.COMMON.USE_FOCAL_LOSS:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduce=False)
                focal_loss_factor = torch.pow((1 - cls_prob[range(int(cls_prob.size(0))), rois_label])
                                              , cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA)
                RCNN_loss_cls = torch.mean(RCNN_loss_cls * focal_loss_factor)
            else:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            if (grasp_rois_mask > 0).sum().item() > 0:
                grasp_feat = self._MGN_head_to_tail(pooled_feat[grasp_rois_mask])
            else:
                # when there are no one positive rois:
                grasp_loc = Variable(torch.Tensor([]).type_as(gt_grasps))
                grasp_prob = Variable(torch.Tensor([]).type_as(gt_grasps))
                grasp_bbox_loss = Variable(torch.Tensor([0]).type_as(RCNN_loss_bbox))
                grasp_cls_loss = Variable(torch.Tensor([0]).type_as(RCNN_loss_cls))
                grasp_conf_label = torch.Tensor([-1]).type_as(rois_label)
                grasp_all_anchors = torch.Tensor([]).type_as(gt_grasps)
                return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
                   grasp_loc, grasp_prob, grasp_bbox_loss , grasp_cls_loss, grasp_conf_label, grasp_all_anchors
        else:
            grasp_feat = self._MGN_head_to_tail(pooled_feat)

        grasp_pred = self.MGN_classifier(grasp_feat)
        # bs*N x K*A x 5, bs*N x K*A x 2
        grasp_loc, grasp_conf = grasp_pred

        # generate anchors
        # bs*N x K*A x 5
        grasp_all_anchors = self._generate_anchors(grasp_conf.size(1), grasp_conf.size(2), rois)
        # filter out negative samples
        grasp_all_anchors = grasp_all_anchors.type_as(gt_grasps)
        if self.training:
            grasp_all_anchors = grasp_all_anchors[grasp_rois_mask]
            # bs*N x 1 x 1
            rois_w = (rois[:, :, 3] - rois[:, :, 1]).data.view(-1).unsqueeze(1).unsqueeze(2)
            rois_h = (rois[:, :, 4] - rois[:, :, 2]).data.view(-1).unsqueeze(1).unsqueeze(2)
            rois_w = rois_w[grasp_rois_mask]
            rois_h = rois_h[grasp_rois_mask]
            # bs*N x 1 x 1
            fsx = rois_w / grasp_conf.size(1)
            fsy = rois_h / grasp_conf.size(2)
            # bs*N x 1 x 1
            xleft = rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
            ytop = rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
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

    def create_architecture(self):
        self._init_modules()

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

        normal_init(self.MGN_classifier.conf, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.MGN_classifier.loc, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)

class resnet(_MGN):
    def __init__(self, classes, num_layers = 101, pretrained=False, class_agnostic=False):

        self.num_layers = num_layers
        self.model_path = 'data/pretrained_model/resnet' + str(num_layers) + '_caffe.pth'

        self.pretrained = pretrained
        self._bbox_dim = 5
        self.dout_base_model = None
        if num_layers == 18 or num_layers == 34:
            self.expansions = 1
        elif num_layers == 50 or num_layers == 101 or num_layers == 152:
            self.expansions = 4
        else:
            assert 0, "network not defined"

        super(resnet, self).__init__(classes, class_agnostic)

    def _init_modules(self):
        if self.num_layers == 18:
            resnet = resnet18()
        elif self.num_layers == 34:
            resnet = resnet34()
        elif self.num_layers == 50:
            resnet = resnet50()
        elif self.num_layers == 101:
            resnet = resnet101()
        elif self.num_layers == 152:
            resnet = resnet152()
        else:
            assert 0, "network not defined"

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet.
        if self._fs == 16:
            self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
            self.RCNN_top = nn.Sequential(resnet.layer4)

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

        else:
            assert 0, "only support stride 16."

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        # fix batch normalization
        self.base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def _RCNN_head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7

    def _MGN_head_to_tail(self, feats):
        return self.MGN_top(feats)