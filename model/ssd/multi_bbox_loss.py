# -*- coding: utf-8 -*-
"""
borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/modules/multibox_loss.py
modified and reorganized by Hanbo Zhang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils.config import cfg

from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.negpos_ratio = cfg.TRAIN.SSD.NEG_POS_RATIO

        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS)


    def forward(self, predictions, gt_bboxes, num_boxes):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_bboxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_bboxes)

        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t, conf_t = self._match_priors_gt(priors, gt_bboxes, cfg.TRAIN.COMMON.BBOX_THRESH, num_boxes)
        conf_t = conf_t.long()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = self._log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(num, -1)

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        # To find element indexes that indicate elements which have highest confidence loss
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = float(num_pos.data.sum().item())
        loss_l /=  N
        loss_c /=  N
        return loss_l, loss_c

    def _match_priors_gt(self, priors, gt, thresh, num_boxes):
        batch_size = gt.size(0)
        num_priors = priors.size(0)
        overlaps = bbox_overlaps_batch(priors, gt)

        # [b, num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1)
        # [b, num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(2)

        matches = torch.zeros(batch_size, num_priors, 5).type_as(priors)
        for num in range(batch_size):
            # select valid best prior idx
            best_prior_idx_valid = best_prior_idx[num][:num_boxes[num]]
            best_truth_overlap[num].index_fill_(0, best_prior_idx_valid, 2)  # ensure best prior
            # TODO refactor: index  best_prior_idx with long tensor
            # ensure every gt matches with its prior of max overlap
            for j in range(best_prior_idx_valid.size(0)):
                best_truth_idx[num][best_prior_idx_valid[j]] = j
            matches[num] = gt[num][best_truth_idx[num]]

        loc = matches[:,:,:-1]  # Shape: [bs, num_priors,4]
        conf = matches[:,:,-1]  # Shape: [bs, num_priors]
        conf[best_truth_overlap < thresh] = 0  # label as background
        encoded_loc = bbox_transform_batch(priors, loc)
        if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            encoded_loc = ((encoded_loc - self.BBOX_NORMALIZE_MEANS.expand_as(encoded_loc))
                        / self.BBOX_NORMALIZE_STDS.expand_as(encoded_loc))
        return encoded_loc, conf

    def _log_sum_exp(self,x):
        """Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence loss across
        all examples in a batch.
        Args:
            x (Variable(tensor)): conf_preds from conf layers
        """
        x_max = x.data.max()
        return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max
