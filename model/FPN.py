import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from utils.config import cfg
from rpn.rpn import _RPN
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_crop.modules.roi_crop import _RoICrop
from roi_align.modules.roi_align import RoIAlignAvg
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from basenet.resnet import resnet18,resnet34,resnet50,resnet101,resnet152

import torch.nn.init as init

class _FPN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpns
        self._share_rpn = cfg.FPN.SHARE_RPN
        self._share_header = cfg.FPN.SHARE_HEADER

        self._num_pyramid_layers = len(cfg.RCNN_COMMON.FEAT_STRIDE)
        if self._share_rpn:
            self.RCNN_rpn = _RPN(self.dout_base_model,
                                 anchor_scales=cfg.RCNN_COMMON.ANCHOR_SCALES,
                                 anchor_ratios=cfg.RCNN_COMMON.ANCHOR_RATIOS,
                                 feat_stride=cfg.RCNN_COMMON.FEAT_STRIDE)
        else:
            self.RCNN_rpns = nn.ModuleList()
            for i in range(len(cfg.RCNN_COMMON.FEAT_STRIDE)):
                self.RCNN_rpns.append(
                    _RPN(self.dout_base_model,
                         anchor_scales=cfg.RCNN_COMMON.ANCHOR_SCALES,
                         anchor_ratios=cfg.RCNN_COMMON.ANCHOR_RATIOS,
                         feat_stride=cfg.RCNN_COMMON.FEAT_STRIDE[i])
                )

        self.RCNN_roi_aligns = nn.ModuleList()
        self.RCNN_roi_pools = nn.ModuleList()
        for i in range(len(cfg.RCNN_COMMON.FEAT_STRIDE)):
            self.RCNN_roi_aligns.append(
                RoIAlignAvg(cfg.RCNN_COMMON.POOLING_SIZE,
                            cfg.RCNN_COMMON.POOLING_SIZE,
                            1.0 / float(cfg.RCNN_COMMON.FEAT_STRIDE[i]))
            )

            self.RCNN_roi_pools.append(
                _RoIPooling(cfg.RCNN_COMMON.POOLING_SIZE,
                            cfg.RCNN_COMMON.POOLING_SIZE,
                            1.0 / float(cfg.RCNN_COMMON.FEAT_STRIDE[i]))
            )

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.iter_counter = 0


    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]

        batch_size = im_data.size(0)

        if self.training:
            self.iter_counter += 1

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        C = [base_feat]
        for i, layer in enumerate(self.RCNN_feat_layers):
            C.append(layer(C[i]))

        # C2 C3 C4 C5 C6
        C256 = []
        for i, newconv in enumerate(self.RCNN_newconvs):
            C256.append(newconv(C[i+1]))

        source = [C256[3]]
        for i, upsampleconv in enumerate(self.RCNN_upsampleconvs):
            if cfg.FPN.UPSAMPLE_CONV:
                source.append(F.upsample(upsampleconv(source[i]),size=(C256[2 - i].size(-2), C256[2 - i].size(-1)), mode = 'bilinear') + C256[2 - i])
            else:
                source.append(F.upsample(source[i],size=(C256[2 - i].size(-2), C256[2 - i].size(-1)), mode = 'bilinear') + C256[2 - i])
        # reverse ups list
        source = source[::-1]

        # P2 P3 P4 P5 P6
        source.append(C256[4])

        for i in range(len(source)):
            source[i] = self.RCNN_mixconvs[i](source[i])

        # feed base feature map tp RPN to obtain rois
        if self._share_rpn:
            rois_rpn, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(source, im_info, gt_boxes, num_boxes)
        else:
            rois_rpn = []
            rpn_loss_bbox = []
            rpn_loss_cls = []
            for i in range(len(source)):
                r, cl, bl = self.RCNN_rpns[i](source[i], im_info, gt_boxes, num_boxes)
                rois_rpn.append(r)
                rpn_loss_cls.append(cl)
                rpn_loss_bbox.append(bl)
            rpn_loss_bbox = sum(rpn_loss_bbox) / len(rpn_loss_bbox)
            rpn_loss_cls = sum(rpn_loss_cls) / len(rpn_loss_cls)
            rois_rpn = torch.cat(rois_rpn, dim = 1)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois_rpn, gt_boxes, num_boxes)
            # outputs is a tuple of list.
            roi_data = self._assign_layer(roi_data)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = [i for i in rois_label if i.numel()>0]
            rois_target = [i for i in rois_target if i.numel() > 0]
            rois_inside_ws = [i for i in rois_inside_ws if i.numel() > 0]
            rois_outside_ws = [i for i in rois_outside_ws if i.numel() > 0]

        else:
            rois = self._assign_layer(rois_rpn)
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = [0]
            rpn_loss_bbox = [0]

        for i in range(len(rois)):
            rois[i] = Variable(rois[i])
        # do roi pooling based on predicted rois
        pooled_feat = []
        if cfg.RCNN_COMMON.POOLING_MODE == 'align':
            for i in range(len(source)):
                if rois[i].numel()>0:
                    pooled_feat.append(self.RCNN_roi_aligns[i](source[i], rois[i].view(-1, 5)))
        elif cfg.RCNN_COMMON.POOLING_MODE == 'pool':
            for i in range(len(source)):
                if rois[i].numel() > 0:
                    pooled_feat.append(self.RCNN_roi_pools[i](source[i], rois[i].view(-1, 5)))

        rois = torch.cat(rois, dim = 0)
        img_inds = rois[:,0]
        pooled_feat = torch.cat(pooled_feat, dim = 0)
        if self.training:
            rois_label = torch.cat(rois_label, dim=0)
            rois_target = torch.cat(rois_target, dim=0)
            rois_inside_ws = torch.cat(rois_inside_ws, dim=0)
            rois_outside_ws = torch.cat(rois_outside_ws, dim=0)

        # put all rois belonging to the same image together
        inds = []
        for i in range(batch_size):
            # rois indexes in ith image
            rois_num_i = int(torch.sum(img_inds == i))
            _, inds_i = torch.sort(img_inds == i, descending=True)
            inds.append(inds_i[:rois_num_i])
        inds = torch.cat(inds,dim = 0)

        rois = rois[inds]
        pooled_feat = pooled_feat[inds]
        if self.training:
            rois_label = rois_label[inds]
            rois_target = rois_target[inds]
            rois_inside_ws = rois_inside_ws[inds]
            rois_outside_ws = rois_outside_ws[inds]

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1)
                                            .expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            if cfg.TRAIN.COMMON.USE_FOCAL_LOSS:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, reduce=False)
                focal_loss_factor = torch.pow((1 -  cls_prob[range(int(cls_prob.size(0))),rois_label])
                                             ,cfg.TRAIN.COMMON.FOCAL_LOSS_GAMMA)
                RCNN_loss_cls = torch.mean(RCNN_loss_cls * focal_loss_factor)
            else:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.contiguous().view(batch_size, -1, 5)
        cls_prob = cls_prob.contiguous().view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.contiguous().view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, \
               rpn_loss_cls, rpn_loss_bbox, \
               RCNN_loss_cls, RCNN_loss_bbox, \
               rois_label

    def _assign_layer(self,roi_data):
        k0 = cfg.FPN.K
        if isinstance(roi_data,tuple):
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # rois: bs x N x 5. The first column is image index
            w = rois[:, :, 3] - rois[:, :, 1]
            h = rois[:, :, 4] - rois[:, :, 2]
            log2 = np.log(2)
            # in paper k = 2,3,4,5,6, but in this code k = 0,1,2,3,4
            k = torch.floor(torch.clamp(k0 + torch.log(torch.sqrt(w * h) / 224) / log2, min = 0, max = 4))

            rois_new = []
            rois_label_new = []
            rois_target_new = []
            rois_inside_ws_new = []
            rois_outside_ws_new = []
            for i in range(self._num_pyramid_layers):
                inds = (k == i)
                rois_new.append(Variable(rois[inds]))
                rois_label_new.append(Variable(rois_label[inds].view(-1).long()))
                rois_target_new.append(Variable(rois_target[inds].view(-1, rois_target.size(2))))
                rois_inside_ws_new.append(Variable(rois_inside_ws[inds].view(-1,rois_inside_ws.size(2))))
                rois_outside_ws_new.append(Variable(rois_outside_ws[inds].view(-1,rois_outside_ws.size(2))))

            roi_data_new = (rois_new, rois_label_new, rois_target_new, rois_inside_ws_new, rois_outside_ws_new)
            return roi_data_new
        else:
            rois = roi_data
            # rois: (bs*N) x 5. The first column is image index
            w = rois[:, :, 3] - rois[:,:, 1]
            h = rois[:,:, 4] - rois[:,:, 2]
            log2 = np.log(2)
            # in paper k = 2,3,4,5,6, but in this code k = 0,1,2,3,4
            k = torch.floor(torch.clamp(k0 + torch.log(torch.sqrt(w * h) / 224) / log2, min = 0, max = 4))

            rois_new = []
            for i in range(self._num_pyramid_layers):
                inds = (k == i)
                rois_new.append(Variable(rois[inds, :]))

            roi_data_new = rois_new
            return roi_data_new


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

        if self._share_rpn:
            normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
            normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
            normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        else:
            for rpn in self.RCNN_rpns:
                normal_init(rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
                normal_init(rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
                normal_init(rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.COMMON.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)

        # FPN layers init
        for newconv in self.RCNN_newconvs:
            normal_init(newconv, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)
        for deconv in self.RCNN_upsampleconvs:
            normal_init(deconv, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)
        for mixconv in self.RCNN_mixconvs:
            normal_init(mixconv, 0, 0.001, cfg.TRAIN.COMMON.TRUNCATED)

        def xavier_init(m):

            def xavier(param):
                init.xavier_uniform(param)
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()

        self.RCNN_top.apply(xavier_init)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

class resnet(_FPN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/resnet' + str(num_layers) + '_caffe.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.num_layers = num_layers

    if num_layers == 18 or num_layers == 34:
        self.expansions = 1
    elif num_layers == 50 or num_layers == 101 or num_layers == 152:
        self.expansions = 4
    else:
        assert 0, "network not defined"

    _FPN.__init__(self, classes, class_agnostic)

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
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool)

    self.RCNN_feat_layers = nn.ModuleList()
    self.RCNN_feat_layers.append(resnet.layer1)
    self.RCNN_feat_layers.append(resnet.layer2)
    self.RCNN_feat_layers.append(resnet.layer3)
    self.RCNN_feat_layers.append(resnet.layer4)
    self.RCNN_feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

    # FPN channel reducing layers
    self.RCNN_newconvs = nn.ModuleList()
    self.RCNN_newconvs.append(nn.Conv2d(64 * self.expansions, 256, 1, stride=1))
    self.RCNN_newconvs.append(nn.Conv2d(128 * self.expansions, 256, 1, stride=1))
    self.RCNN_newconvs.append(nn.Conv2d(256 * self.expansions, 256, 1, stride=1))
    self.RCNN_newconvs.append(nn.Conv2d(512 * self.expansions, 256, 1, stride=1))
    self.RCNN_newconvs.append(nn.Conv2d(512 * self.expansions, 256, 1, stride=1))

    # FPN mix conv layers
    self.RCNN_mixconvs = nn.ModuleList()
    self.RCNN_mixconvs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
    self.RCNN_mixconvs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
    self.RCNN_mixconvs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
    self.RCNN_mixconvs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
    self.RCNN_mixconvs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))

    # FPN upsample conv layers (whether to attach a conv layer after upsampling, totally 3)
    self.RCNN_upsampleconvs = nn.ModuleList()
    for i in range(3):
        self.RCNN_upsampleconvs.append(
            nn.Conv2d(256, 256, 1, stride=1)
        )
    hidden_num = 1024
    # fully connected classifier and regressor
    self.RCNN_top = nn.Sequential(
        nn.Linear(self.expansions * 64 * 7 * 7, hidden_num),
        nn.ReLU(),
        nn.Linear(hidden_num, hidden_num),
        nn.ReLU())

    self.RCNN_cls_score = nn.Linear(hidden_num, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(hidden_num, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_feat_layers[2].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_feat_layers[1].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_feat_layers[0].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_feat_layers.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_feat_layers[0].eval()
      self.RCNN_feat_layers[1].train()
      self.RCNN_feat_layers[2].train()
      self.RCNN_feat_layers[3].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_feat_layers.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)
    return fc7

