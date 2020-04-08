import torch
from torch import nn
import torch.nn.functional as F

from utils.config import cfg
from utils.net_utils import set_bn_fix, set_bn_eval

import abc

from basenet.resnet import resnet_initializer
from basenet.vgg import vgg_initializer
from basenet.efficientnet import EfficientNet
from model.op2l.op2l import _OP2L
from model.rpn.bbox_transform import bbox_overlaps
from utils.net_utils import objdet_inference

import copy


class detector(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, feat_name='res101', feat_list=('conv4',), pretrained=True):
        super(detector, self).__init__()
        self.feat_name = feat_name
        self.feat_list = feat_list
        self.pretrained = pretrained

        self.FeatExt = self._init_feature_extractor()

    def _init_feature_extractor(self):
        # init resnet feature extractor
        if 'res' in self.feat_name:
            return resnet_initializer(self.feat_name, self.feat_list, self.pretrained)
        elif 'vgg' in self.feat_name:
            return vgg_initializer(self.feat_name, self.feat_list, self.pretrained)
        elif 'efcnet' in self.feat_name:
            if self.pretrained:
                return EfficientNet.from_pretrained(self.feat_name.replace("efcnet", "efficientnet-"),
                                                    feat_list = self.feat_list)
            else:
                return EfficientNet.from_name(self.feat_name.replace("efcnet", "efficientnet-"),
                                              override_params={'feat_list': self.feat_list})

class graspDetector(detector):
    __metaclass__ = abc.ABCMeta
    def __init__(self, feat_name='res101', feat_list=('conv4',), pretrained=True):
        super(graspDetector, self).__init__(feat_name, feat_list, pretrained)

class objectDetector(detector):
    __metaclass__ = abc.ABCMeta
    def __init__(self, num_classes, class_agnostic,
                 feat_name = 'res101', feat_list = ('conv4',), pretrained = True):
        super(objectDetector, self).__init__(feat_name, feat_list, pretrained)

        self.n_classes = num_classes
        self.class_agnostic = class_agnostic

class vmrn_rel_classifier(nn.Module):
    def __init__(self, obj_pair_feat_dim):
        super(vmrn_rel_classifier,self).__init__()
        self._input_dim = obj_pair_feat_dim
        self.fc1 = nn.Linear(self._input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.outlayer = nn.Linear(2048,3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.outlayer(x)
        return x

class VMRN(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(VMRN,self).__init__()
        self._isex = cfg.TRAIN.VMRN.ISEX
        self.VMRN_rel_op2l = _OP2L(cfg.VMRN.OP2L_POOLING_SIZE, cfg.VMRN.OP2L_POOLING_SIZE, 1.0 / 16.0, self._isex)
        self.iter_counter = 0

    def _object_detection(self, rois, cls_prob, bbox_pred, batch_size, im_info):
        det_results = torch.Tensor([]).type_as(rois[0])
        obj_num = []

        for i in range(batch_size):
            obj_boxes = torch.Tensor(objdet_inference(cls_prob[i], bbox_pred[i], im_info[i], rois[i][:, 1:5],
                                                      class_agnostic = self.class_agnostic,
                                                      for_vis = True, recover_imscale=False)).type_as(det_results)
            obj_num.append(obj_boxes.size(0))
            if obj_num[-1] > 0 :
                # add image index
                img_ind = i * torch.ones(obj_boxes.size(0), 1).type_as(det_results)
                det_results = torch.cat([det_results, torch.cat([img_ind, obj_boxes], 1)], 0)

        return det_results, torch.Tensor(obj_num).type_as(det_results).long()

    def _get_rel_det_result(self, base_feat, obj_rois, obj_num):
        # filter out the detection of only one object instance
        obj_pair_feat = self.VMRN_rel_op2l(base_feat, obj_rois, self.batch_size, obj_num)
        if not cfg.TRAIN.VMRN.USE_REL_GRADIENTS:
            obj_pair_feat = obj_pair_feat.detach()
        obj_pair_feat = self._rel_head_to_tail(obj_pair_feat)
        rel_cls_score = self.VMRN_rel_cls_score(obj_pair_feat)
        rel_cls_prob = F.softmax(rel_cls_score)
        return rel_cls_score, rel_cls_prob

    def _check_rel_mat(self, rel_mat, o1, o2):
        # some labels are neglected when the dataset was labeled
        if rel_mat[o1, o2].item() == 0:
            if rel_mat[o2, o1].item() == 3:
                rel_mat[o1, o2] = rel_mat[o2, o1]
            else:
                rel_mat[o1, o2] = 3 - rel_mat[o2, o1]
        return rel_mat

    def _generate_rel_labels(self, obj_rois, gt_boxes, obj_num, rel_mat, rel_batch_size):

        obj_pair_rel_label = torch.Tensor(rel_batch_size).type_as(gt_boxes).zero_().long()
        # generate online data labels
        cur_pair = 0
        for i in range(obj_num.size(0)):
            img_index = i % self.batch_size
            if obj_num[i] <=1 :
                continue
            begin_ind = torch.sum(obj_num[:i])
            overlaps = bbox_overlaps(obj_rois[begin_ind:begin_ind + obj_num[i]][:, 1:5],
                                     gt_boxes[img_index][:, 0:4])
            max_overlaps, max_inds = torch.max(overlaps, 1)
            for o1ind in range(obj_num[i]):
                for o2ind in range(o1ind + 1, obj_num[i]):
                    o1_gt = int(max_inds[o1ind].item())
                    o2_gt = int(max_inds[o2ind].item())
                    if o1_gt == o2_gt:
                        # skip invalid pairs
                        if self._isex:
                            cur_pair += 2
                        else:
                            cur_pair += 1
                        continue
                    # some labels are neglected when the dataset was labeled
                    rel_mat[img_index] = self._check_rel_mat(rel_mat[img_index], o1_gt, o2_gt)
                    obj_pair_rel_label[cur_pair] = rel_mat[img_index][o1_gt, o2_gt]
                    cur_pair += 1

                    if self._isex:
                        rel_mat[img_index] = self._check_rel_mat(rel_mat[img_index], o2_gt, o1_gt)
                        obj_pair_rel_label[cur_pair] = rel_mat[img_index][o2_gt, o1_gt]
                        cur_pair += 1

        return obj_pair_rel_label

    def _rel_det_loss_comp(self, obj_pair_rel_label, rel_cls_score):
        obj_pair_rel_label = obj_pair_rel_label
        # filter out all invalid data (e.g. two ROIs are matched to the same ground truth)
        rel_not_keep = (obj_pair_rel_label == 0)
        if (rel_not_keep == 0).sum().item() > 0:
            rel_keep = torch.nonzero(rel_not_keep == 0).view(-1)
            rel_cls_score = rel_cls_score[rel_keep]
            obj_pair_rel_label = obj_pair_rel_label[rel_keep]
            obj_pair_rel_label -= 1
            VMRN_rel_loss_cls = F.cross_entropy(rel_cls_score, obj_pair_rel_label)
        return VMRN_rel_loss_cls

    def _rel_cls_prob_post_process(self, rel_cls_prob):
        if (not cfg.TEST.VMRN.ISEX) and cfg.TRAIN.VMRN.ISEX:
            rel_cls_prob = rel_cls_prob[::2, :]
        elif cfg.TEST.VMRN.ISEX and cfg.TRAIN.VMRN.ISEX:
            rel_cls_prob_1 = rel_cls_prob[0::2, :]
            rel_cls_prob_2 = rel_cls_prob[1::2, :]
            rel_cls_prob = rel_cls_prob_1.new(rel_cls_prob_1.shape).zero_()
            rel_cls_prob[:, 0] = (rel_cls_prob_1[:, 0] + rel_cls_prob_2[:, 1]) / 2
            rel_cls_prob[:, 1] = (rel_cls_prob_1[:, 1] + rel_cls_prob_2[:, 0]) / 2
            rel_cls_prob[:, 2] = (rel_cls_prob_1[:, 2] + rel_cls_prob_2[:, 2]) / 2
        return rel_cls_prob

    def _rel_head_to_tail(self, pooled_pair):
        if self.feat_name[:3] == 'res':
            return self._rel_head_to_tail_resnet(pooled_pair)
        elif self.feat_name[:3] == 'vgg':
            return self._rel_head_to_tail_vgg(pooled_pair)

    def _rel_head_to_tail_resnet(self, pooled_pair):
        # box_type: o1, o2, union
        opfc = []
        if cfg.VMRN.SHARE_WEIGHTS:
            for box_type in range(pooled_pair.size(1)):
                cur_box = pooled_pair[:,box_type]
                opfc[box_type] = self.VMRN_rel_top(cur_box)
        else:
            opfc.append(self.VMRN_rel_top_o1(pooled_pair[:,0]).mean(3).mean(2))
            opfc.append(self.VMRN_rel_top_o2(pooled_pair[:, 1]).mean(3).mean(2))
            opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).mean(3).mean(2))
        return torch.cat(opfc, 1)

    def _rel_head_to_tail_vgg(self, pooled_pair):
        # box_type: o1, o2, union
        opfc = []
        if cfg.VMRN.SHARE_WEIGHTS:
            for box_type in range(pooled_pair.size(1)):
                cur_box = pooled_pair[:, box_type]
                opfc[box_type] = self.VMRN_rel_top(cur_box)
        else:
            opfc.append(self.VMRN_rel_top_o1(pooled_pair[:, 0]).view(pooled_pair.size(0), -1))
            opfc.append(self.VMRN_rel_top_o2(pooled_pair[:, 1]).view(pooled_pair.size(0), -1))
            opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).view(pooled_pair.size(0), -1))
        return torch.cat(opfc,1)

    def _init_modules_resnet(self):
        # VMRN layers
        if cfg.VMRN.SHARE_WEIGHTS:
            self.VMRN_rel_top = copy.deepcopy(self.FeatExt.layer4)
        else:
            self.VMRN_rel_top_o1 = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_o2 = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_union = copy.deepcopy(self.FeatExt.layer4)

        self.VMRN_rel_cls_score = vmrn_rel_classifier(2048 * 3)

    def _init_modules_vgg(self):
        def rel_pipe():
            return nn.Sequential(
                nn.Conv2d(512, 128, 1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, 1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                )

        if cfg.VMRN.SHARE_WEIGHTS:
            self.VMRN_rel_top = rel_pipe()
        else:
            self.VMRN_rel_top_o1 = rel_pipe()
            self.VMRN_rel_top_o2 = rel_pipe()
            self.VMRN_rel_top_union = rel_pipe()

        self.VMRN_rel_cls_score = vmrn_rel_classifier(64 * 7 * 7 * 3)

    def _init_weights(self):
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

        normal_init(self.VMRN_rel_cls_score.fc1, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.fc2, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.VMRN_rel_cls_score.outlayer, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)

    def train(self, mode=True):
        if mode and self.feat_name[:3] == 'res':
            if cfg.VMRN.SHARE_WEIGHTS:
                self.VMRN_rel_top.apply(set_bn_eval)
            else:
                self.VMRN_rel_top_o1.apply(set_bn_eval)
                self.VMRN_rel_top_o2.apply(set_bn_eval)
                self.VMRN_rel_top_union.apply(set_bn_eval)