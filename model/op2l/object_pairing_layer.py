# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg

import pdb

DEBUG = False

class _ObjPairLayer(nn.Module):
    def __init__(self, isex):
        super(_ObjPairLayer, self).__init__()
        self._isex = isex

    def forward(self, roi_pooled_feats, batch_size, obj_num):
        """
        :param roi_pooled_feats: feature maps after roi pooling.
          The first obj_num features are single-object features.
          dim: BS*N+N(N-1) x C x W x H
        :param obj_num: object number
        :return: obj_pair_feats: dim: BS*N(N-1) x 3 x C x W x H
        """

        _paired_feats = torch.tensor([]).type_as(roi_pooled_feats)
        for imgnum in range(obj_num.size(0)):
            if obj_num[imgnum] <= 1:
                continue
            begin_idx = (0.5 * obj_num[:imgnum].float() ** 2 + 0.5 * obj_num[:imgnum].float()).sum().item()
            cur_img_feats = roi_pooled_feats[int(begin_idx):\
                int(begin_idx + 0.5 * float(obj_num[imgnum]) ** 2 + 0.5 * float(obj_num[imgnum]))]
            cur_img_feats = self._single_image_pair(cur_img_feats, int(obj_num[imgnum]))
            _paired_feats = torch.cat([_paired_feats, cur_img_feats], 0)

        return _paired_feats

    def _single_image_pair(self, feats, objnum):
        obj_feats = feats[:objnum]
        union_feats = feats[objnum:]
        pair_feats = torch.tensor([]).type_as(feats)

        cur_union = 0
        for o1 in range(objnum):
            for o2 in range(o1+1, objnum):
                pair_feats = torch.cat([pair_feats,
                                        torch.cat([obj_feats[o1:o1+1],
                                                   obj_feats[o2:o2+1],
                                                   union_feats[cur_union:cur_union+1]],
                                                  0).unsqueeze(0)]
                                       ,0)
                if self._isex:
                    pair_feats = torch.cat([pair_feats,
                                            torch.cat([obj_feats[o2:o2 + 1],
                                                       obj_feats[o1:o1 + 1],
                                                       union_feats[cur_union:cur_union+1]],
                                                      0).unsqueeze(0)]
                                           , 0)
                cur_union += 1

        return pair_feats

