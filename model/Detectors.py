import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.config import cfg
from utils.net_utils import set_bn_fix, set_bn_eval, set_bn_unfix, set_bn_train

import abc

from basenet.resnet import resnet_initializer
from basenet.vgg import vgg_initializer
from model.op2l.op2l import _OP2L
from model.rpn.bbox_transform import bbox_overlaps
from utils.net_utils import objdet_inference, weights_normal_init
from utils.crf_utils import crf, RelaTransform
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

class rel_classifier(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, obj_pair_feat_dim, using_bn=False, grad_backprop=False):
        super(rel_classifier, self).__init__()
        self._input_dim = obj_pair_feat_dim
        self._using_bn = using_bn
        self._grad_backprop = grad_backprop

    def _build_fc(self, in_d, out_d):
        if self._using_bn:
            return nn.Sequential(nn.Linear(in_d, out_d),
                                 nn.BatchNorm1d(out_d),
                                 nn.ReLU(inplace=True))
        else:
            return nn.Sequential(nn.Linear(in_d, out_d),
                                 nn.ReLU(inplace=True))

class vmrn_rel_classifier(rel_classifier):
    def __init__(self, obj_pair_feat_dim, num_rel = 3, using_bn = True, grad_backprop = True):
        super(vmrn_rel_classifier,self).__init__(obj_pair_feat_dim, using_bn, grad_backprop)
        # self.fc1 = self._build_fc(self._input_dim * 3, 2048)
        # self.fc2 = self._build_fc(2048, 2048)
        self.fc1 = nn.Linear(self._input_dim * 3, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.outlayer = nn.Linear(2048, num_rel)

    def forward(self, x):
        x = torch.cat(x[:3], 1)
        if not self._grad_backprop: x = x.detach()
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.outlayer(x)
        reg_loss = 0
        return x, reg_loss

class uvtranse_classifier(rel_classifier):
    def __init__(self, obj_pair_feat_dim, num_rel = 3, using_bn = False, grad_backprop = True):
        super(uvtranse_classifier,self).__init__(obj_pair_feat_dim, using_bn, grad_backprop)
        # initialize UVTransE weights
        self.f_s = nn.Sequential(self._build_fc(self._input_dim, 512),
                                 self._build_fc(512, 256))
        self.f_o = copy.deepcopy(self.f_s)
        self.f_u = copy.deepcopy(self.f_s)
        self.vis_embeds = [self.f_s, self.f_o, self.f_u]

        self.Output = nn.Sequential(self._build_fc(256 + 19, 256),
                                    nn.Linear(256, num_rel))
        self._regular = cfg.VMRN.UVTRANSE_REGULARIZATION
        
    def forward(self, x):
        # generate vis feats
        for i in range(3):
            if not self._grad_backprop:
                x[i] = x[i].detach()
            x[i] = self.vis_embeds[i](x[i])

        reg_loss = 0
        if self.training:
            reg_loss = cfg.VMRN.UVTRANSE_REGULARIZATION * (torch.clamp(x[0].pow(2).sum(1) - 1, min = 0) +
                                                           torch.clamp(x[1].pow(2).sum(1) - 1, min = 0) +
                                                           torch.clamp(x[2].pow(2).sum(1) - 1, min = 0)).mean()

        # p = union - s - o
        appear_feat = x[2] - x[1] - x[0]

        x[3]["pair_num"] = x[0].shape[0]
        loc_feat = self._generate_loc_feat(x[3])
        assert appear_feat.dim() == 2 and loc_feat.dim() == 2
        x = torch.cat([appear_feat, loc_feat], dim = 1)
        x = self.Output(x)
        return x, reg_loss

    def _generate_loc_feat(self, locs):
        obj_num = locs["obj_num"]
        pair_num = locs["pair_num"]
        boxes = locs["box"]
        is_ex = locs["is_ex"]
        im_info = locs["im_info"]
        batch_size = im_info.shape[0]

        loc_feats = torch.zeros(size=(pair_num ,14)).float()
        pair_counter = 0
        for im_ind in range(obj_num.size(0)):
            if obj_num[im_ind] <= 1:
                continue
            begin_idx = (0.5 * obj_num[:im_ind].float() ** 2 + 0.5 * obj_num[:im_ind].float()).sum().item()
            im_boxes = boxes[int(begin_idx):\
                int(begin_idx + 0.5 * float(obj_num[im_ind]) ** 2 + 0.5 * float(obj_num[im_ind]))]

            im_obj_boxes = im_boxes[:obj_num[im_ind]]
            im_union_boxes = im_boxes[obj_num[im_ind]:]

            union_counter = 0
            for o1 in range(obj_num[im_ind]):
                for o2 in range(o1+1, obj_num[im_ind]):
                    o1_box = im_obj_boxes[o1][1:5]
                    o2_box = im_obj_boxes[o2][1:5]
                    union_box = im_union_boxes[union_counter][1:5]
                    loc_feats[pair_counter] = torch.cat((o1_box, o2_box, union_box, im_info[im_ind % batch_size][:2]))
                    pair_counter += 1
                    if is_ex:
                        loc_feats[pair_counter] = torch.cat((o2_box, o1_box, union_box, im_info[im_ind % batch_size][:2]))
                        pair_counter += 1

        loc_feats = self._encode_loc_feat(loc_feats)
        loc_feats = loc_feats.type_as(boxes)
        return loc_feats

    def _encode_loc_feat(self, loc_feats):
        encoded = torch.zeros(size=(loc_feats.shape[0] ,19)).float()
        im_info = loc_feats[:, -2:]

        w_i, h_i = im_info[:, 1], im_info[:, 0]

        x_s, y_s, w_s, h_s = (loc_feats[:, 0] + loc_feats[:, 2]) / 2., (loc_feats[:, 1] + loc_feats[:, 3]) / 2., \
                                loc_feats[:, 2] - loc_feats[:, 0], loc_feats[:, 3] - loc_feats[:, 1]
        a_s = w_s * h_s
        x_o, y_o, w_o, h_o = (loc_feats[:, 4] + loc_feats[:, 6]) / 2., (loc_feats[:, 5] + loc_feats[:, 7]) / 2., \
                             loc_feats[:, 6] - loc_feats[:, 4], loc_feats[:, 7] - loc_feats[:, 5]
        a_o = w_o * h_o
        a_u = (loc_feats[:, 10] - loc_feats[:, 8]) * (loc_feats[:, 11] - loc_feats[:, 9])

        encoded[:, 0] = x_s / w_i
        encoded[:, 1] = y_s / h_i
        encoded[:, 2] = (x_s + w_s) / w_i
        encoded[:, 3] = (y_s + h_s) / h_i
        encoded[:, 4] = a_s / (w_i * h_i)
        encoded[:, 5] = x_o / w_i
        encoded[:, 6] = y_o / h_i
        encoded[:, 7] = (x_o + w_o) / w_i
        encoded[:, 8] = (y_o + h_o) / h_i
        encoded[:, 9] = a_o / (w_i * h_i)
        encoded[:, 10] = (x_s - x_o) / w_o
        encoded[:, 11] = (y_s - y_o) / h_o
        encoded[:, 12] = torch.log(w_s / w_o)
        encoded[:, 13] = torch.log(h_s / h_o)
        encoded[:, 14] = (x_o - x_s) / w_s
        encoded[:, 15] = (y_o - y_s) / h_s
        encoded[:, 16] = torch.log(w_o / w_s)
        encoded[:, 17] = torch.log(h_o / h_s)
        encoded[:, 18] = a_u / (w_i * h_i)

        return encoded

class VMRN(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(VMRN,self).__init__()
        self._isex = cfg.TRAIN.VMRN.ISEX
        self.VMRN_rel_op2l = _OP2L(cfg.VMRN.OP2L_POOLING_SIZE, cfg.VMRN.OP2L_POOLING_SIZE, 1.0 / 16.0, self._isex)
        self.using_crf = cfg.VMRN.USE_CRF
        self.iter_counter = 0

    def _object_detection(self, rois, cls_prob, bbox_pred, batch_size, im_info):
        det_results = torch.tensor([]).type_as(rois[0])
        obj_num = []

        for i in range(batch_size):
            obj_boxes = torch.tensor(objdet_inference(cls_prob[i], bbox_pred[i], im_info[i], rois[i][:, 1:5],
                                                      class_agnostic = self.class_agnostic, for_vis = True,
                                                      recover_imscale=False, with_cls_score=True)).type_as(det_results)
            # obj_boxes = obj_boxes[torch.argsort(obj_boxes[:, 4], descending=True)]
            # obj_boxes = obj_boxes[:6] # maximum box number : 6
            obj_num.append(obj_boxes.size(0))
            if obj_num[-1] > 0:
                obj_boxes = torch.cat([obj_boxes[:, :4], obj_boxes[:, -1:]], dim=1)
                # add image index
                img_ind = i * torch.ones(obj_boxes.size(0), 1).type_as(det_results)
                det_results = torch.cat([det_results, torch.cat([img_ind, obj_boxes], 1)], 0)

        return det_results, torch.tensor(obj_num).type_as(det_results).long()

    def _get_rel_det_result(self, base_feat, obj_rois, obj_num, im_info):
        # filter out the detection of only one object instance
        obj_pair_feat, paired_rois = self.VMRN_rel_op2l(base_feat, obj_rois, self.batch_size, obj_num)
        if not cfg.TRAIN.VMRN.USE_REL_GRADIENTS:
            obj_pair_feat = obj_pair_feat.detach()
        obj_pair_feat = self._rel_head_to_tail(obj_pair_feat)
        obj_pair_feat.append({"box":paired_rois, "obj_num":obj_num, "is_ex":self._isex, "im_info":im_info})
        rel_cls_score, reg_loss = self.VMRN_rel_cls_score(obj_pair_feat)
        if cfg.VMRN.SCORE_POSTPROC:
            rel_cls_score = self._rel_cls_score_post_process(rel_cls_score)
        if self.using_crf:
            rel_cls_score = crf(rel_cls_score, obj_num, 7, True)
        rel_cls_prob = F.softmax(rel_cls_score)
        return rel_cls_score, rel_cls_prob, reg_loss

    def _check_rel_mat(self, rel_mat, o1, o2):
        # some labels are neglected when the dataset was labeled
        if rel_mat[o1, o2].item() == 0:
            if rel_mat[o2, o1].item() == 3:
                rel_mat[o1, o2] = rel_mat[o2, o1]
            else:
                rel_mat[o1, o2] = 3 - rel_mat[o2, o1]
        return rel_mat

    def _generate_rel_labels(self, obj_rois, gt_boxes, obj_num, rel_mat, rel_batch_size):
        if self.using_crf:
            rel_mat = RelaTransform(rel_mat)
        obj_pair_rel_label = torch.zeros(rel_batch_size).type_as(gt_boxes).long()
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

    def _select_pairs(self, obj_rois, obj_num):
        # in each image, only 2 rois are preserved.
        obj_num = obj_num[:self.batch_size].zero_() + 2
        selected_rois = []
        for im_ind in range(self.batch_size):
            rois = obj_rois[obj_rois[:, 0] == im_ind]
            for _ in range(5):
                selected = rois[np.random.choice(np.arange(rois.shape[0]), size=2, replace=False)]
                # check if the selected two boxes are same.
                if bbox_overlaps(selected[0:1][:, 1:5], selected[1:2][:, 1:5]).item() > 0.7:
                    continue
                else:
                    break
            selected_rois.append(selected.clone())
        selected_rois = torch.cat(selected_rois, dim=0)
        return selected_rois, obj_num

    def _rel_det_loss_comp(self, obj_pair_rel_label, rel_cls_score):
        obj_pair_rel_label = obj_pair_rel_label
        # filter out all invalid data (e.g. two ROIs are matched to the same ground truth)
        rel_not_keep = (obj_pair_rel_label == 0)
        VMRN_rel_loss_cls = 0
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
            if self.using_crf:
                rel_cls_prob[:, 3] = (rel_cls_prob_1[:, 3] + rel_cls_prob_2[:, 4]) / 2
                rel_cls_prob[:, 4] = (rel_cls_prob_1[:, 4] + rel_cls_prob_2[:, 3]) / 2
        return rel_cls_prob

    def _rel_cls_score_post_process(self, rel_cls_score):
        if cfg.TRAIN.VMRN.ISEX:
            rel_cls_score_conj = rel_cls_score.reshape(-1, 2, rel_cls_score.shape[-1])
            rel_cls_score_conj = torch.cat([rel_cls_score_conj[:, 1:2], rel_cls_score_conj[:, 0:1]], dim=1)
            rel_cls_score_conj = rel_cls_score_conj.reshape(rel_cls_score.shape)
            rel_cls_score[:, 0] = (rel_cls_score[:, 0] + rel_cls_score_conj[:, 1]) / 2
            rel_cls_score[:, 1] = (rel_cls_score[:, 1] + rel_cls_score_conj[:, 0]) / 2
            rel_cls_score[:, 2] = (rel_cls_score[:, 2] + rel_cls_score_conj[:, 2]) / 2
            if self.using_crf:
                rel_cls_score[:, 3] = (rel_cls_score[:, 3] + rel_cls_score_conj[:, 4]) / 2
                rel_cls_score[:, 4] = (rel_cls_score[:, 4] + rel_cls_score_conj[:, 3]) / 2
        return rel_cls_score

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
                opfc.append(self.VMRN_rel_top(pooled_pair[:,box_type]).mean(3).mean(2))
        else:
            opfc.append(self.VMRN_rel_top_o1(pooled_pair[:,0]).mean(3).mean(2))
            opfc.append(self.VMRN_rel_top_o2(pooled_pair[:, 1]).mean(3).mean(2))
            opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).mean(3).mean(2))
        return opfc

    def _rel_head_to_tail_vgg(self, pooled_pair):
        # box_type: o1, o2, union
        opfc = []
        if cfg.VMRN.SHARE_WEIGHTS:
            for box_type in range(pooled_pair.size(1)):
                cur_box = pooled_pair[:, box_type]
                opfc.append(self.VMRN_rel_top(cur_box))
        else:
            opfc.append(self.VMRN_rel_top_o1(pooled_pair[:, 0]).view(pooled_pair.size(0), -1))
            opfc.append(self.VMRN_rel_top_o2(pooled_pair[:, 1]).view(pooled_pair.size(0), -1))
            opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).view(pooled_pair.size(0), -1))
        return opfc

    def _init_modules_resnet(self):
        # VMRN layers
        if cfg.VMRN.SHARE_WEIGHTS:
            self.VMRN_rel_top = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top.apply(set_bn_unfix)
        else:
            self.VMRN_rel_top_o1 = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_o2 = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_union = copy.deepcopy(self.FeatExt.layer4)
            self.VMRN_rel_top_o1.apply(set_bn_unfix)
            self.VMRN_rel_top_o2.apply(set_bn_unfix)
            self.VMRN_rel_top_union.apply(set_bn_unfix)

        num_rel = 3 if not self.using_crf else 5
        if cfg.VMRN.RELATION_CLASSIFIER == "UVTransE":
            self.VMRN_rel_cls_score = uvtranse_classifier(2048, num_rel=num_rel,
                                                          grad_backprop=cfg.TRAIN.VMRN.USE_REL_CLS_GRADIENTS)
        else:
            self.VMRN_rel_cls_score = vmrn_rel_classifier(2048, num_rel=num_rel,
                                                          grad_backprop=cfg.TRAIN.VMRN.USE_REL_CLS_GRADIENTS)

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
            self.VMRN_rel_top = self.FeatExt.feat_layer["fc"]
            objfeat_dim = 4096
        else:
            self.VMRN_rel_top_o1 = rel_pipe()
            self.VMRN_rel_top_o2 = rel_pipe()
            self.VMRN_rel_top_union = rel_pipe()
            objfeat_dim = 64 * 7 * 7

        num_rel = 3 if not self.using_crf else 5
        if cfg.VMRN.RELATION_CLASSIFIER == "UVTransE":
            self.VMRN_rel_cls_score = uvtranse_classifier(objfeat_dim, num_rel=num_rel,
                                                          grad_backprop=cfg.TRAIN.VMRN.USE_REL_CLS_GRADIENTS)
        else:
            self.VMRN_rel_cls_score = vmrn_rel_classifier(objfeat_dim, num_rel=num_rel,
                                                          grad_backprop=cfg.TRAIN.VMRN.USE_REL_CLS_GRADIENTS)

    def _init_weights(self):
        self.VMRN_rel_cls_score.apply(weights_normal_init)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode and self.feat_name[:3] == 'res':
            if cfg.VMRN.SHARE_WEIGHTS:
                self.VMRN_rel_top.apply(set_bn_eval)
            else:
                self.VMRN_rel_top_o1.apply(set_bn_eval)
                self.VMRN_rel_top_o2.apply(set_bn_eval)
                self.VMRN_rel_top_union.apply(set_bn_eval)