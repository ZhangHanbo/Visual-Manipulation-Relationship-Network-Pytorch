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

import torch.nn.init as init
from model.utils.config import cfg

import numpy as np

import pdb
import time


from .bbox_transform_grasp import labels2points, points2labels, \
    grasp_encode, grasp_decode,jaccard_overlap

class _GraspTargetLayer(nn.Module):
    def __init__(self, feat_stride, ratios, scales, angles):
        super(_GraspTargetLayer, self).__init__()

        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS)

        self.negpos_ratio = cfg.TRAIN.FCGN.NEG_POS_RATIO

        self._feat_stride = feat_stride

    def forward(self, conf, gt, priors, xthresh = None, ythresh = None, angle_thresh = None):
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt)

        self.batch_size = gt.size(0)

        if xthresh is None:
            xthresh = self._feat_stride / 2

        if ythresh is None:
            ythresh = self._feat_stride / 2

        if angle_thresh is None:
            angle_thresh = cfg.TRAIN.FCGN.ANGLE_THRESH

        if cfg.TRAIN.FCGN.ANGLE_MATCH:
            loc_t, conf_t = self._match_gt_prior(priors, gt, xthresh, ythresh, angle_thresh)
        else:
            loc_t, conf_t = self._match_gt_prior_IoUbased(priors, gt)
        iw, ow = self._mine_hard_samples(conf_t, conf)

        if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            loc_t = ((loc_t - self.BBOX_NORMALIZE_MEANS.expand_as(loc_t))
                        / self.BBOX_NORMALIZE_STDS.expand_as(loc_t))

        #if ((conf_t == 0).sum()/(conf_t == 1).sum()).item() != 3:
        #    pdb.set_trace()

        return loc_t, conf_t, iw, ow

    def _match_gt_prior(self, priors, gt, xthresh, ythresh, angle_thresh):
        """
        :param priors: bs x K x 5
        :param gt: bs x N x 5
        :param angle_thresh:
        :return:
        """

        num_priors = priors.size(1)

        x_gt = gt[:, :, 0:1].transpose(2,1)
        y_gt = gt[:, :, 1:2].transpose(2,1)
        ang_gt = gt[:, :, 4:5].transpose(2,1)
        mask_gt = (torch.sum(gt==0, 2, keepdim = True) != gt.size(2)).transpose(2,1)

        xdiff = torch.abs(priors[:, : ,0:1] - x_gt)
        ydiff = torch.abs(priors[:, :, 1:2] - y_gt)
        angdiff = torch.abs(priors[:, :, 4:5] - ang_gt)

        mask = torch.zeros_like(xdiff) + mask_gt.float()

        match_mat = (xdiff <= xthresh) \
                    & (ydiff <= ythresh) \
                    & (angdiff <= angle_thresh) \
                    & (mask != 0)

        match_num = torch.sum(match_mat, 2, keepdim = True)
        label = torch.zeros(self.batch_size, num_priors).type_as(gt).long()
        label[(torch.sum(match_mat, 2) > 0)] = 1

        # bs x N x K ->  K x bs x N ->  K x bs x N x 1
        match_mat = match_mat.permute(2,0,1).unsqueeze(3)
        # bs x K x 5 ->  K x bs x 5 ->  K x bs x 1 x 5
        gt = gt.permute(1,0,2).unsqueeze(2)
        # K x bs x N x 5 -> bs x N x 5
        # When a prior matches multi gts, it will use
        # the mean of all matched gts as its target.
        loc = torch.sum(match_mat.float() * gt, dim = 0) + cfg.EPS

        # make all nans zeros
        keep = (match_num > 0).squeeze()
        loc[keep] /= match_num[keep].float()
        loc_encode = grasp_encode(loc, priors)

        return loc_encode, label

    def _match_gt_prior_IoUbased(self, priors, gt):
        """
        :param priors: bs x K x 5
        :param gt: bs x N x 5
        :param angle_thresh:
        :return:
        """
        num_priors = priors.size(1)

        x_gt = gt[:, :, 0:1].transpose(2,1)
        y_gt = gt[:, :, 1:2].transpose(2,1)
        #ang_gt = gt[:, :, 4:5].transpose(2, 1)

        mask_gt = (torch.sum(gt==0, 2, keepdim = True) != gt.size(2)).transpose(2,1)

        xdiff = torch.abs(priors[:, : ,0:1] - x_gt)
        ydiff = torch.abs(priors[:, :, 1:2] - y_gt)
        #angdiff = torch.abs(priors[:, :, 4:5] - ang_gt)
        mask = torch.zeros_like(xdiff) + mask_gt.float()

        match_mat = (xdiff <= self._feat_stride / 2) \
                    & (ydiff <= self._feat_stride / 2) \
                    & (mask != 0)

        iou_ind = torch.nonzero(match_mat).data.cpu()
        for i in iou_ind:
            rec1 = np.array(priors[i[0].item(),i[1].item(),:])
            rec2 = np.array(gt[i[0].item(),i[2].item(),:])
            if jaccard_overlap(rec1,rec2) < cfg.TRAIN.FCGN.JACCARD_THRESH:
                match_mat[i[0].item(),i[1].item(),i[2].item()] = 0

        match_num = torch.sum(match_mat, 2, keepdim = True)
        label = torch.zeros(self.batch_size, num_priors).type_as(gt).long()
        label[(torch.sum(match_mat, 2) > 0)] = 1

        # bs x N x K ->  K x bs x N ->  K x bs x N x 1
        match_mat = match_mat.permute(2,0,1).unsqueeze(3)
        # bs x K x 5 ->  K x bs x 5 ->  K x bs x 1 x 5
        gt = gt.permute(1,0,2).unsqueeze(2)
        # K x bs x N x 5 -> bs x N x 5
        # When a prior matches multi gts, it will use
        # the mean of all matched gts as its target.
        loc = torch.sum(match_mat.float() * gt, dim = 0) + cfg.EPS

        # make all nans zeros
        keep = (match_num > 0).squeeze()
        loc[keep] /= match_num[keep].float()
        loc_encode = grasp_encode(loc, priors)

        return loc_encode, label

    def _mine_hard_samples(self, conf_t, conf):
        """
        :param loc_t: bs x N x 5
        :param conf_t: bs x N
        :param conf: bs x N x 2
        :return:
        """
        pos = (conf_t > 0)
        batch_conf = conf.data.view(-1, 2)
        loss_c = self._log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(self.batch_size, -1)

        loss_c[pos] = -1  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        # To find element indexes that indicate elements which have highest confidence loss
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = self.negpos_ratio * num_pos
        neg = (idx_rank < num_neg.expand_as(idx_rank)) & (pos != 1)

        conf_t[neg.eq(0) & pos.eq(0)] = -1

        iw = pos.gt(0).float() * cfg.TRAIN.FCGN.BBOX_INSIDE_WEIGHTS[0]
        iw = iw.unsqueeze(2).expand(conf.size(0), -1, 5)

        if cfg.TRAIN.FCGN.BBOX_POSITIVE_WEIGHTS < 0:
            ow = (pos + neg).gt(0).float() / ((num_pos + num_neg)|1).float()
            ow = ow.unsqueeze(2).expand(conf.size(0), -1, 5)
        else:
            ow = (pos.gt(0).float() * cfg.TRAIN.FCGN.BBOX_POSITIVE_WEIGHTS \
                + neg.gt(0).float()) / ((num_pos + num_neg)|1).float()
            ow = ow.unsqueeze(2).expand(conf.size(0), -1, 5)

        if (ow != ow).sum().item() > 0:
            pdb.set_trace()

        if (neg.gt(0) & pos.gt(0)).sum().item() > 0:
            pdb.set_trace()

        return iw, ow

    def _log_sum_exp(self,x):
        """Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence loss across
        all examples in a batch.
        Args:
            x (Variable(tensor)): conf_preds from conf layers
        """
        x_max, _ = x.data.max(dim = 1, keepdim = True)
        return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max