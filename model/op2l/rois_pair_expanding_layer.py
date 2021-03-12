# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.autograd import Variable

DEBUG = False

class _RoisPairExpandingLayer(nn.Module):
    def __init__(self):
        super(_RoisPairExpandingLayer,self).__init__()

    def forward(self, rois, batch_size, obj_num):
        """
        :param rois: region of intrests list
        :param batch_size: image number in one batch
        :param obj_num: a Tensor that indicates object numbers in each image
        :return:
        """
        self._rois = torch.tensor([]).type_as(rois).float()
        for imgnum in range(obj_num.size(0)):
            begin_idx = obj_num[:imgnum].sum().item()
            if obj_num[imgnum] == 1:
                cur_rois = rois[int(begin_idx):int(begin_idx + obj_num[imgnum].item())][:, 1:5]
                cur_rois = torch.cat([((imgnum % batch_size) * torch.ones(cur_rois.size(0), 1)).type_as(cur_rois),
                                      cur_rois], 1)
                self._rois = torch.cat([self._rois, cur_rois], 0)
            elif obj_num[imgnum] >1:
                cur_rois = rois[int(begin_idx):int(begin_idx + obj_num[imgnum].item())][:, 1:5]
                cur_rois = self._single_image_expand(cur_rois)
                cur_rois = torch.cat([((imgnum % batch_size) * torch.ones(cur_rois.size(0), 1)).type_as(cur_rois),
                                      cur_rois], 1)
                self._rois = torch.cat([self._rois, cur_rois], 0)

        return self._rois

    def backward(self):
        pass

    def _single_image_expand(self, rois):
        _rois = rois
        _rois_num = _rois.size(0)
        for b1 in range(_rois_num):
            for b2 in range(b1+1, _rois_num):
                if b1 != b2:
                    box1 = rois[b1]
                    box2 = rois[b2]
                    tmax = torch.max(box1[2:4], box2[2:4])
                    tmin = torch.min(box1[0:2], box2[0:2])
                    unionbox = torch.cat([tmin, tmax],0)
                    unionbox = torch.reshape(unionbox, (-1, 4))
                    _rois = torch.cat([_rois, unionbox], 0)
        return _rois
