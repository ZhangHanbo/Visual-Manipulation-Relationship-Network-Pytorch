from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, anchor_scales, anchor_ratios, feat_stride, include_rois_score = False):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.feat_stride = feat_stride

        self._num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = self._num_anchors * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = self._num_anchors * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer, which is used to filter good proposals (default number: 300)
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, include_rois_score)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        """1 x (d*A) x H x W -> 1 x d x (A*H) x W """
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    @staticmethod
    def reshapeII(x, d):
        """1 x (d*A) x H x W -> 1 x d x A x H x W"""
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1]) / float(d)),
            input_shape[2],
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        # check input
        if isinstance(base_feat, list):
            assert isinstance(self.feat_stride,list) and len(self.feat_stride) == len(base_feat), \
                "Input of multi-scale RPN should be a list of feature maps that match cfg.RCNN_COMMON.FEAT_STRIDE"
            for feat in base_feat:
                assert feat.size(1) == self.din, \
                    "Numbers of channels of all feature maps should be equal to RPN.din"
            return self._forward_multi_feature_maps(base_feat, im_info, gt_boxes, num_boxes)

        else:
            return self._forward_single_feature_maps(base_feat, im_info, gt_boxes, num_boxes)

    def _forward_single_feature_maps(self,base_feat, im_info, gt_boxes, num_boxes):
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_score_reshapeII = self.reshapeII(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        rpn_bbox_pred_reshape = rpn_bbox_pred.permute(0,2,3,1).contiguous().view(batch_size, -1, 4)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                  im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshapeII.permute(0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)

            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            # FOCAL LOSS
            if cfg.TRAIN.RCNN_COMMON.RPN_USE_FOCAL_LOSS:
                rpn_cls_prob = self.reshapeII(rpn_cls_prob, 2)\
                    .permute(0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)
                rpn_cls_prob = torch.index_select(rpn_cls_prob.view(-1, 2), 0, rpn_keep)
                self.rpn_loss_cls  = F.cross_entropy(rpn_cls_score, rpn_label, reduce=False)
                focal_loss_factor = torch.pow((1 -  rpn_cls_prob[range(int(rpn_cls_prob.size(0))),rpn_label])
                                             ,cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA)
                self.rpn_loss_cls = torch.mean(self.rpn_loss_cls * focal_loss_factor)
            else:
                self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred_reshape, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2])

        return rois, self.rpn_loss_cls, self.rpn_loss_box

    def _forward_multi_feature_maps(self,base_feat, im_info, gt_boxes, num_boxes):
        batch_size = base_feat[0].size(0)

        rpn_cls_score_list = []
        rpn_cls_prob_list = []
        rpn_bbox_pred_list = []

        if self.training:
            rpn_cls_score_reshape_list = []
            rpn_bbox_pred_reshape_list = []
            rpn_cls_prob_reshape_list = []


        for feat in base_feat:
            # return feature map after convrelu layer
            rpn_conv1 = F.relu(self.RPN_Conv(feat), inplace=True)
            # get rpn classification score
            rpn_cls_score = self.RPN_cls_score(rpn_conv1)
            rpn_cls_score_list.append(rpn_cls_score)

            rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
            if self.training:
                # 1 x 2 x A x H x W -> 1 x H x W x A x 2 -> 1 x (H*W*A) x 2
                rpn_cls_score_reshape_list.append(self.reshapeII(rpn_cls_score, 2).
                                                  permute(0,3,4,2,1).contiguous().view(batch_size,-1,2))

            rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim = 1)
            rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
            if self.training:
                # 1 x 2 x A x H x W -> 1 x H x W x A x 2 -> 1 x (H*W*A) x 2
                rpn_cls_prob_reshape_list.append(self.reshapeII(rpn_cls_prob, 2).
                                                  permute(0,3,4,2,1).contiguous().view(batch_size,-1,2))
            rpn_cls_prob_list.append(rpn_cls_prob)

            # get rpn offsets to the anchor boxes
            rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
            rpn_bbox_pred_list.append(rpn_bbox_pred)
            if self.training:
                # 1 x (A*4) x H x W ->  1 x H x W x (A*4) -> 1 x (H*W*A) x 4
                rpn_bbox_pred_reshape_list.append(rpn_bbox_pred.permute(0,2,3,1).
                                                  contiguous().view(batch_size, -1, 4))

        if self.training:
            all_rpn_cls_score = torch.cat(rpn_cls_score_reshape_list, 1)
            all_rpn_bbox_pred = torch.cat(rpn_bbox_pred_reshape_list, 1)
            all_rpn_cls_prob = torch.cat(rpn_cls_prob_reshape_list, 1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal(([cls.data for cls in rpn_cls_prob_list],
                                  [bbox.data for bbox in rpn_bbox_pred_list],
                                  im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target(([cls.data for cls in rpn_cls_score_list], gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(all_rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            # print('rpn fg/bg:'+ str(int((rpn_label==1).sum())) +'/' +str(int((rpn_label==0).sum())))
            # FOCAL LOSS
            if cfg.TRAIN.RCNN_COMMON.RPN_USE_FOCAL_LOSS:
                rpn_cls_prob = torch.index_select(all_rpn_cls_prob.view(-1, 2), 0, rpn_keep)
                self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label, reduce=False)
                focal_loss_factor = torch.pow((1 - rpn_cls_prob[range(int(rpn_cls_prob.size(0))), rpn_label])
                                              , cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA)
                self.rpn_loss_cls = torch.sum(self.rpn_loss_cls * focal_loss_factor)
            else:
                self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(all_rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2])

        return rois, self.rpn_loss_cls, self.rpn_loss_box