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
from model.ssd.multi_bbox_loss import MultiBoxLoss
from model.utils.config import cfg
import torchvision as tv

from model.utils.net_utils import weights_xavier_init
from ObjectDetector import objectDetector

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

class SSD(objectDetector):
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

    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv3', 'conv4'), pretrained = True):
        super(SSD, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)
        self.FeatExt.feat_layer["conv3"][0].ceil_mode = True
        ##### Important to set model to eval mode before evaluation ####
        self.FeatExt.eval()
        rand_img = torch.Tensor(1, 3, 300, 300)
        rand_feat = self.FeatExt(rand_img)
        self.FeatExt.train()
        n_channels = [f.size(1) for f in rand_feat]

        self.size = cfg.SCALES[0]
        self.classes = classes
        self.num_classes = len(self.classes)
        self.priors_cfg = self._init_prior_cfg()
        self.priorbox = PriorBox(self.priors_cfg)
        self.priors_xywh = Variable(self.priorbox.forward())
        self.priors_xywh.detach()

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
        self.criterion = MultiBoxLoss(self.num_classes)

        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()
        mbox_cfg = []
        for i in cfg.SSD.PRIOR_ASPECT_RATIO:
            mbox_cfg.append(2 * len(i) + 2)

        self.loc.append(
            nn.Conv2d(n_channels[0], mbox_cfg[0] * 4 if self.class_agnostic else mbox_cfg[0] * 4 * self.num_classes
                      , kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(n_channels[0], mbox_cfg[0] * self.num_classes, kernel_size=3, padding=1))

        self.SSD_conv7 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True))
        self.loc.append(nn.Conv2d(1024, mbox_cfg[1] * 4 if self.class_agnostic else mbox_cfg[1] * 4 * self.num_classes,
                                  kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(1024, mbox_cfg[1] * self.num_classes, kernel_size=3, padding=1))

        self.SSD_conv8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.loc.append(nn.Conv2d(512, mbox_cfg[2] * 4 if self.class_agnostic else mbox_cfg[2] * 4 * self.num_classes,
                                  kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(512, mbox_cfg[2] * self.num_classes, kernel_size=3, padding=1))

        self.SSD_conv9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.loc.append(nn.Conv2d(256, mbox_cfg[3] * 4 if self.class_agnostic else mbox_cfg[3] * 4 * self.num_classes,
                                  kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(256, mbox_cfg[3] * self.num_classes, kernel_size=3, padding=1))

        self.SSD_conv10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.loc.append(nn.Conv2d(256, mbox_cfg[4] * 4 if self.class_agnostic else mbox_cfg[4] * 4 * self.num_classes,
                                  kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(256, mbox_cfg[4] * self.num_classes, kernel_size=3, padding=1))

        self.SSD_conv11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.loc.append(nn.Conv2d(256, mbox_cfg[5] * 4 if self.class_agnostic else mbox_cfg[5] * 4 * self.num_classes,
                                  kernel_size=3, padding=1))
        self.conf.append(nn.Conv2d(256, mbox_cfg[5] * self.num_classes, kernel_size=3, padding=1))

        self.iter_counter = 0

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

        if self.training:
            self.iter_counter += 1

        sources = []
        loc = []
        conf = []

        s0, x = self.FeatExt(x)

        s0 = self.L2Norm(s0)
        sources.append(s0)

        # apply vgg up to fc7
        x = self.SSD_conv7(x)
        sources.append(x)

        x = self.SSD_conv8(x)
        sources.append(x)

        x = self.SSD_conv9(x)
        sources.append(x)

        x = self.SSD_conv10(x)
        sources.append(x)

        x = self.SSD_conv11(x)
        sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        if self.training:
            predictions = (
                loc,
                conf,
                self.priors.type_as(loc)
            )
            # targets = torch.cat([gt_boxes[:,:,:4] / self.size, gt_boxes[:,:,4:5]],dim=2)
            targets = gt_boxes
            SSD_loss_bbox, SSD_loss_cls = self.criterion(predictions, targets, num_boxes)
        else:
            SSD_loss_cls = 0
            SSD_loss_bbox = 0

        conf = self.softmax(conf)

        return loc, conf, SSD_loss_bbox, SSD_loss_cls

    def create_architecture(self):
        # initialize newly added layers' weights with xavier method
        self.SSD_conv7.apply(weights_xavier_init)
        self.SSD_conv8.apply(weights_xavier_init)
        self.SSD_conv9.apply(weights_xavier_init)
        self.SSD_conv10.apply(weights_xavier_init)
        self.SSD_conv11.apply(weights_xavier_init)
        self.loc.apply(weights_xavier_init)
        self.conf.apply(weights_xavier_init)

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
