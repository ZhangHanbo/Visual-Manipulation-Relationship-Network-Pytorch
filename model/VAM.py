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
import torchvision as tv
import os
from model.ssd.default_bbox_generator import PriorBox
import torch.nn.init as init
from model.ssd.multi_bbox_loss import MultiBoxLoss
from model.utils.config import cfg
from model.roi_layers.nms import nms
from model.rpn.bbox_transform import bbox_overlaps

from model.op2l.op2l import _OP2L

import torchvision
import pdb

import numpy as np

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

t2i = torchvision.transforms.ToPILImage()
i2t = torchvision.transforms.ToTensor()
trans = torchvision.transforms.Compose(
            [t2i, torchvision.transforms.Resize(size=[224,224]), i2t]
        )

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class SSD(nn.Module):
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

    def __init__(self, classes):
        super(SSD, self).__init__()
        self.size = cfg.SCALES[0]
        self.classes = classes
        self.num_classes = len(self.classes)
        self.priors_cfg = self._init_prior_cfg()
        self.priorbox = PriorBox(self.priors_cfg)
        self.priors_xywh = Variable(self.priorbox.forward(), volatile=True)
        self.priors = torch.cat([
            self.priors_xywh[:, 0:1] - 0.5 * self.priors_xywh[:, 2:3],
            self.priors_xywh[:, 1:2] - 0.5 * self.priors_xywh[:, 3:4],
            self.priors_xywh[:, 0:1] + 0.5 * self.priors_xywh[:, 2:3],
            self.priors_xywh[:, 1:2] + 0.5 * self.priors_xywh[:, 3:4]
        ], 1)

        self.priors = self.priors * self.size
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        self.softmax = nn.Softmax(dim=-1)

        self._isex = cfg.TRAIN.VMRN.ISEX
        self.VMRN_rel_op2l = _OP2L(cfg.VMRN.OP2L_POOLING_SIZE, cfg.VMRN.OP2L_POOLING_SIZE, 1.0/8.0, self._isex)

        self.iter_counter = 0

        self.criterion = MultiBoxLoss(self.num_classes)

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

        input_imgs = x.clone()

        sources = list()
        loc = list()
        conf = list()

        self.batch_size = x.size(0)

        # apply vgg up to conv4_3 relu
        if isinstance(self.base, nn.ModuleList):
            for k,v in enumerate(self.base):
                x = v(x)
        else:
            x = self.base(x)

        s = self.L2Norm(x)
        sources.append(s)
        base_feat = s

        # apply vgg up to fc7
        if isinstance(self.conv5, nn.ModuleList):
            for k,v in enumerate(self.conv5):
                x = v(x)
        else:
            x = self.conv5(x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        SSD_loss_cls = 0
        SSD_loss_bbox = 0
        if self.training:
            predictions = (
                loc,
                conf,
                self.priors.type_as(loc)
            )
            # targets = torch.cat([gt_boxes[:,:,:4] / self.size, gt_boxes[:,:,4:5]],dim=2)
            targets = gt_boxes
            SSD_loss_bbox, SSD_loss_cls = self.criterion(predictions, targets, num_boxes)

        conf = self.softmax(conf)

        # online data
        if self.training:
            if self.iter_counter > cfg.TRAIN.VMRN.ONLINEDATA_BEGIN_ITER:
                obj_rois, obj_num = self._obj_det(conf, loc, self.batch_size, im_info)
                obj_rois = obj_rois.type_as(gt_boxes)
                obj_num = obj_num.type_as(num_boxes)
            else:
                obj_rois = torch.FloatTensor([]).type_as(gt_boxes)
                obj_num = torch.LongTensor([]).type_as(num_boxes)
            obj_labels = None
        else:
            # when testing, this is object detection results
            # TODO: SUPPORT MULTI-IMAGE BATCH
            obj_rois, obj_num = self._obj_det(conf, loc, self.batch_size, im_info)
            if obj_rois.numel() > 0:
                obj_labels = obj_rois[:, 5]
                obj_rois = obj_rois[:, :5]
                obj_rois = obj_rois.type_as(gt_boxes)
                obj_num = obj_num.type_as(num_boxes)
            else:
                # there is no object detected
                obj_labels = torch.Tensor([]).type_as(gt_boxes).long()
                obj_rois = obj_rois.type_as(gt_boxes)
                obj_num = obj_num.type_as(num_boxes)

        if self.training:
            # offline data
            for i in range(self.batch_size):
                obj_rois = torch.cat([obj_rois,
                                      torch.cat([(i * torch.ones(num_boxes[i].item(), 1)).type_as(gt_boxes),
                                                 (gt_boxes[i][:num_boxes[i]][:, 0:4])], 1)
                                      ])
                obj_num = torch.cat([obj_num, torch.Tensor([num_boxes[i]]).type_as(obj_num)])


        obj_rois = Variable(obj_rois)

        VMRN_rel_loss_cls = 0
        rel_cls_prob = torch.Tensor([]).type_as(obj_rois)
        if (obj_num > 1).sum().item() > 0:

            obj_pair_feat = self.VMRN_obj_pair_feat_extractor(input_imgs, obj_rois, self.batch_size, obj_num)
            # obj_pair_feat = obj_pair_feat.detach()
            rel_cls_score = self.VMRN_rel_cls_score(obj_pair_feat)

            rel_cls_prob = F.softmax(rel_cls_score)

            self.rel_batch_size = obj_pair_feat.size(0)

            if self.training:
                obj_pair_rel_label = self._generate_rel_labels(obj_rois, gt_boxes, obj_num, rel_mat)
                obj_pair_rel_label = obj_pair_rel_label.type_as(gt_boxes).long()

                rel_not_keep = (obj_pair_rel_label == 0)
                # no relationship is kept
                if (rel_not_keep == 0).sum().item() > 0:
                    rel_keep = torch.nonzero(rel_not_keep == 0).view(-1)

                    rel_cls_score = rel_cls_score[rel_keep]

                    obj_pair_rel_label = obj_pair_rel_label[rel_keep]
                    obj_pair_rel_label -= 1
                    VMRN_rel_loss_cls = F.cross_entropy(rel_cls_score, obj_pair_rel_label)
            else:
                if (not cfg.TEST.VMRN.ISEX) and cfg.TRAIN.VMRN.ISEX:
                    rel_cls_prob = rel_cls_prob[::2, :]

        rel_result = None
        if not self.training:
            if obj_rois.numel() > 0:
                pred_boxes = obj_rois.data[:,1:5]
                pred_boxes[:, 0::2] /= im_info[0][3].item()
                pred_boxes[:, 1::2] /= im_info[0][2].item()
                rel_result = (pred_boxes, obj_labels, rel_cls_prob.data)
            else:
                rel_result = (obj_rois.data, obj_labels, rel_cls_prob.data)

        return loc, conf, rel_result, SSD_loss_bbox, SSD_loss_cls, VMRN_rel_loss_cls

    def _generate_rel_labels(self, obj_rois, gt_boxes, obj_num, rel_mat):

        obj_pair_rel_label = torch.Tensor(self.rel_batch_size).type_as(gt_boxes).zero_().long()
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
                    # some labels are leaved out when labeling
                    if rel_mat[img_index][o1_gt, o2_gt].item() == 0:
                        if rel_mat[img_index][o2_gt, o1_gt].item() == 3:
                            rel_mat[img_index][o1_gt, o2_gt] = rel_mat[img_index][o2_gt, o1_gt]
                        else:
                            rel_mat[img_index][o1_gt, o2_gt] = 3 - rel_mat[img_index][o2_gt, o1_gt]
                    obj_pair_rel_label[cur_pair] = rel_mat[img_index][o1_gt, o2_gt]

                    cur_pair += 1
                    if self._isex:
                        # some labels are leaved out when labeling
                        if rel_mat[img_index][o2_gt, o1_gt].item() == 0:
                            if rel_mat[img_index][o1_gt, o2_gt].item() == 3:
                                rel_mat[img_index][o2_gt, o1_gt] = rel_mat[img_index][o1_gt, o2_gt]
                            else:
                                rel_mat[img_index][o2_gt, o1_gt] = 3 - rel_mat[img_index][o1_gt, o2_gt]
                        obj_pair_rel_label[cur_pair] = rel_mat[img_index][o2_gt, o1_gt]
                        cur_pair += 1

        return obj_pair_rel_label

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def _obj_det(self, conf, loc, batch_size, im_info):
        det_results = torch.Tensor([]).type_as(loc)
        obj_num = []
        if not self.training:
            det_labels = torch.Tensor([]).type_as(loc).long()

        for i in range(batch_size):
            cur_cls_prob = conf[i:i + 1]
            cur_bbox_pred = loc[i:i + 1]
            cur_im_info = im_info[i:i + 1]
            obj_boxes = self._get_single_obj_det_results(cur_cls_prob, cur_bbox_pred, cur_im_info)
            obj_num.append(obj_boxes.size(0))
            if obj_num[-1] > 0:
                det_results = torch.cat([det_results,
                                         torch.cat([i * torch.ones(obj_boxes.size(0), 1).type_as(det_results),
                                                    obj_boxes], 1)
                                         ], 0)
        return det_results, torch.LongTensor(obj_num)

    def _get_single_obj_det_results(self, cls_prob, bbox_pred, im_info):

        scores = cls_prob.data
        thresh = 0.05  # filter out low confidence boxes for acceleration
        results = []

        if cfg.TEST.COMMON.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).type_as(box_deltas) \
                                + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).type_as(box_deltas)
                box_deltas = box_deltas.view(1, -1, 4)
            pred_boxes = bbox_transform_inv(self.priors.type_as(bbox_pred).data, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(self.priors.data, (1, scores.shape[1]))

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        for j in xrange(1, self.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds, :]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets[:, :4], cls_dets[:, 4], cfg.TEST.COMMON.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                final_keep = torch.nonzero(cls_dets[:, -1] > cfg.TEST.COMMON.OBJ_DET_THRESHOLD).squeeze()
                result = cls_dets[final_keep]

                if result.numel()>0 and result.dim() == 1:
                    result = result.unsqueeze(0)
                # in testing, concat object labels
                if final_keep.numel() > 0:
                    if self.training:
                        result = result[:,:4]
                    else:
                        result = torch.cat([result[:,:4],
                                j * torch.ones(result.size(0),1).type_as(result)],1)
                if result.numel() > 0:
                    results.append(result)

        if len(results):
            final = torch.cat(results, 0)
        else:
            final = torch.Tensor([]).type_as(bbox_pred)
        return final

    def VMRN_obj_pair_feat_extractor(self, input_imgs, obj_rois, batch_size, obj_num):
        bbox_imgs = torch.Tensor([]).type_as(input_imgs)
        union_bbox_imgs = torch.Tensor([]).type_as(input_imgs)
        for i in range(obj_num.size(0)):
            img_index = i % batch_size
            cur_img = input_imgs[img_index]
            if obj_num[i] <=1 :
                continue
            begin_ind = torch.sum(obj_num[:i])
            cur_obj_bboxes = obj_rois[begin_ind:(begin_ind+obj_num[i])]
            for o1ind in range(obj_num[i]):
                o1_bbox = cur_obj_bboxes[o1ind][1:5].long()
                o1_img = cur_img[:, o1_bbox[1]:o1_bbox[3], o1_bbox[0]:o1_bbox[2]].unsqueeze(0)
                o1_img = F.upsample_bilinear(o1_img, size = [224,224])
                bbox_imgs = torch.cat((bbox_imgs, o1_img), dim = 0)
                for o2ind in range(o1ind + 1, obj_num[i]):
                    o2_bbox = cur_obj_bboxes[o2ind][1:5].long()
                    union_bbox = torch.cat((torch.min(o1_bbox[0:2],o2_bbox[0:2]),
                                            torch.max(o1_bbox[2:4],o2_bbox[2:4])), 0)
                    # 1 x 3 x H x W
                    union_img = cur_img[:, union_bbox[1]:union_bbox[3],
                                union_bbox[0]:union_bbox[2]].unsqueeze(0)
                    union_img = F.upsample_bilinear(union_img, size = [224,224])
                    union_bbox_imgs = torch.cat((union_bbox_imgs, union_img), dim=0)

        bbox_num = bbox_imgs.size(0)
        union_bbox_num = union_bbox_imgs.size(0)
        if isinstance(self.rel_base, nn.ModuleList):
            for k,v in enumerate(self.rel_base):
                bbox_imgs = self.rel_base(bbox_imgs)
                union_bbox_imgs = self.rel_base(union_bbox_imgs)
        else:
            bbox_imgs = self.rel_base(bbox_imgs).reshape(bbox_num, -1)
            union_bbox_imgs = self.rel_base(union_bbox_imgs).reshape(union_bbox_num, -1)

        obj_pair_feats = torch.Tensor([]).type_as(input_imgs)

        union_counter = 0
        for i in range(obj_num.size(0)):
            for o1ind in range(obj_num[i]):
                for o2ind in range(o1ind + 1, obj_num[i]):
                    cur_pair_feat = torch.cat((bbox_imgs[o1ind], bbox_imgs[o2ind],
                                               union_bbox_imgs[union_counter]),dim = 0).unsqueeze(0)
                    obj_pair_feats = torch.cat((obj_pair_feats, cur_pair_feat), dim = 0)
                    if self._isex:
                        cur_pair_feat = torch.cat((bbox_imgs[o2ind], bbox_imgs[o1ind],
                                                   union_bbox_imgs[union_counter]), dim=0).unsqueeze(0)
                        obj_pair_feats = torch.cat((obj_pair_feats, cur_pair_feat), dim=0)
                    union_counter += 1
        return obj_pair_feats

    def create_architecture(self):
        self._init_modules()
        def weights_init(m):
            def xavier(param):
                init.xavier_uniform(param)
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()
        # initialize newly added layers' weights with xavier method
        self.extras.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)

    def _init_prior_cfg(self):
        prior_cfg = {
            'min_dim': self.size,
            'feature_maps': cfg.SSD.FEATURE_MAPS,
            'min_sizes': cfg.SSD.PRIOR_MIN_SIZE,
            'max_sizes': cfg.SSD.PRIOR_MAX_SIZE,
            'steps': cfg.SSD.PRIOR_STEP,
            'aspect_ratios':cfg.SSD.PRIOR_ASPECT_RATIO,
            'clip':cfg.SSD.PRIOR_CLIP
        }
        return prior_cfg

    def resume_iter(self, epoch, iter_per_epoch):
        self.iter_counter = epoch * iter_per_epoch

class vgg16(SSD):
    def __init__(self, num_classes, pretrained=False):
        super(vgg16 , self).__init__(num_classes)
        self._pretrained = pretrained
        self.module_path = "data/pretrained_model/vgg16_reducedfc.pth"
        self.rel_module_path = "data/pretrained_model/vgg16_caffe.pth"
        self._bbox_dim = 4

    def _init_modules(self):
        base_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                    512, 512, 512]

        extras_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

        # TODO: get mbox_cfg from cfg.SSD_PRIOR_ASPECT_RATIO
        mbox_cfg = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location

        base, extras, head = self.multibox(self.vgg(base_cfg, 3),
                                         self.add_extras(extras_cfg, 1024),
                                         mbox_cfg, self.num_classes)
        # init base net
        base = nn.ModuleList(base)
        vgg_weights = torch.load(self.module_path)
        base.load_state_dict(vgg_weights)

        self.base = nn.ModuleList(base[:23])
        self.conv5 = nn.ModuleList(base[23:])

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        rel_base = torchvision.models.vgg16()
        vgg_weights = torch.load(self.rel_module_path)
        rel_base.load_state_dict(vgg_weights)

        self.rel_base = rel_base.features

        self.VMRN_rel_cls_score = vmrn_rel_classifier(512 * 7 * 7 * 3)

    def _rel_head_to_tail(self, pooled_pair):
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

    def add_extras(self, cfg, i, batch_norm=False):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers

    def multibox(self, vgg, extra_layers, cfg, num_classes):
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return vgg, extra_layers, (loc_layers, conf_layers)

    # This function is derived from torchvision VGG make_layers()
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    def vgg(self, cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

class resnet(SSD):
    def __init__(self, num_classes, pretrained=False, layer_num = 50):
        super(resnet , self).__init__(num_classes)
        self.layer_num = layer_num
        self._pretrained = pretrained
        if layer_num == 50:
            self.module_path = "data/pretrained_model/resnet50_caffe.pth"
        elif layer_num == 101:
            self.module_path = "data/pretrained_model/resnet101_caffe.pth"
        self._bbox_dim = 4

    def _init_modules(self):
        extras_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

        # TODO: get mbox_cfg from cfg.SSD_PRIOR_ASPECT_RATIO
        mbox_cfg = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location

        base, extras, head = self.multibox(self.resnet(),
                                           self.add_extras(extras_cfg, 1024),
                                           mbox_cfg, self.num_classes)

        # init base net
        base = nn.ModuleList(base)

        self.base = nn.ModuleList(base[:6])
        self.conv5 = nn.ModuleList(base[6:])

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.layer_num == 50:
            rel_base = torchvision.models.resnet50()
        elif self.layer_num==101:
            rel_base = torchvision.models.resnet101()
        else:
            assert 0, "This ResNet is not defined."
        res_weights = torch.load(self.module_path)
        rel_base.load_state_dict(res_weights)

        self.rel_base = nn.ModuleList([rel_base.conv1, rel_base.bn1, rel_base.relu, rel_base.maxpool,
                    rel_base.layer1, rel_base.layer2, rel_base.layer3, rel_base.layer4, rel_base.avgpool])

        self.VMRN_rel_cls_score = vmrn_rel_classifier(2048 * 3)

    def _rel_head_to_tail(self, pooled_pair):
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
        return torch.cat(opfc, 1)

    def add_extras(self, cfg, i, batch_norm=False):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            # if last layer is S, skip to next v.
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers

    def multibox(self, resnet, extra_layers, cfg, num_classes):
        loc_layers = []
        conf_layers = []

        # stride 8
        loc_layers += [nn.Conv2d(512, cfg[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, cfg[0] * num_classes, kernel_size=3, padding=1)]
        # stride 16
        loc_layers += [nn.Conv2d(1024, cfg[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(1024, cfg[1] * num_classes, kernel_size=3, padding=1)]

        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return resnet, extra_layers, (loc_layers, conf_layers)

    def resnet(self):

        res_full = None
        if self.layer_num == 50:
            res_full = tv.models.resnet50()
        elif self.layer_num == 101:
            res_full = tv.models.resnet101()
        else:
            assert 0, "This ResNet is not defined."

        if self._pretrained:
            res_weights = torch.load(self.module_path)
            res_full.load_state_dict(res_weights)

        layers = [res_full.conv1, res_full.bn1, res_full.relu, res_full.maxpool,
                  res_full.layer1, res_full.layer2, res_full.layer3]
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(1024, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

class vmrn_rel_classifier(nn.Module):
    def __init__(self, obj_pair_feat_dim):
        super(vmrn_rel_classifier, self).__init__()
        self._input_dim = obj_pair_feat_dim
        self.fc1 = nn.Linear(self._input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.outlayer = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.outlayer(x)
        return x
