# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_path
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import *
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.data_viewer import dataViewer

import model.FasterRCNN as FasterRCNN
import model.FPN as FPN
import model.VMRN as VMRN
import model.FullyConvGrasp as FCGN
import model.SSD as SSD
import model.SSD_VMRN as SSD_VMRN
import model.MultiGrasp as MGN
import model.AllinOne as ALL_IN_ONE
import model.RoIGrasp as ROIGN
import model.VAM as VAM

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms.nms_wrapper import nms
from model.fully_conv_grasp.bbox_transform_grasp import labels2points, grasp_decode

import warnings

# implemented-algorithm list
LEGAL_FRAMES = {"faster_rcnn", "ssd", "fpn", "faster_rcnn_vmrn", "ssd_vmrn", "all_in_one", "fcgn", "roign", "vam"}

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--frame', dest='frame',
                    help='faster_rcnn, fpn, ssd, faster_rcnn_vmrn, ssd_vmrn, fcgn, mgn, allinone, roign',
                    default='faster_rcnn', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=0, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=0, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="output",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--GPU', dest='GPU',
                      help='GPU number (Only for model saving.)',
                      default=0, type=int)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=0, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=None, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=None, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=None, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)
  parser.add_argument('--vis', dest='vis',
                      help='whether to visualize training data',
                      default=False, type=bool)

  args = parser.parse_args()
  return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def makeCudaData(data_list):
    for i, data in enumerate(data_list):
        data_list[i] = data.cuda()
    return data_list

def read_cfgs():
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '50']
    elif args.dataset == 'vmrdcompv1':
        args.imdb_name = "vmrd_compv1_trainval"
        args.imdbval_name = "vmrd_compv1_test"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']
    elif args.dataset == 'bdds':
        args.imdb_name = "bdds_trainval"
        args.imdbval_name = "bdds_test"
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']
    elif args.dataset[:7] == 'cornell':
        cornell = args.dataset.split('_')
        args.imdb_name = 'cornell_{}_{}_trainval_{}'.format(cornell[1],cornell[2],cornell[3])
        args.imdbval_name = 'cornell_{}_{}_test_{}'.format(cornell[1],cornell[2],cornell[3])
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '50']
    elif args.dataset[:8] == 'jacquard':
        jacquard = args.dataset.split('_')
        args.imdb_name = 'jacquard_{}_trainval_{}'.format(jacquard[1], jacquard[2])
        args.imdbval_name = 'jacquard_{}_test_{}'.format(jacquard[1], jacquard[2])
        args.set_cfgs = ['MAX_NUM_GT_GRASPS', '1000']
    if args.dataset[:7] == 'cornell':
        args.cfg_file = "cfgs/cornell_{}_{}_ls.yml".format(args.frame, args.net) if args.large_scale \
        else "cfgs/cornell_{}_{}.yml".format(args.frame, args.net)
    elif args.dataset[:8] == 'jacquard':
        args.cfg_file = "cfgs/jacquard_{}_{}_ls.yml".format(args.frame, args.net) if args.large_scale \
        else "cfgs/jacquard_{}_{}.yml".format(args.frame, args.net)
    else:
        args.cfg_file = "cfgs/{}_{}_{}_ls.yml".format(args.dataset, args.frame, args.net) if args.large_scale \
        else "cfgs/{}_{}_{}.yml".format(args.dataset, args.frame, args.net)
    print("Using cfg file: " + args.cfg_file)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if not args.disp_interval:
        args.disp_interval = cfg.TRAIN.COMMON.DISPLAY
    if not args.batch_size:
        args.batch_size = cfg.TRAIN.COMMON.IMS_PER_BATCH
    if not args.lr_decay_step:
        args.lr_decay_step = cfg.TRAIN.COMMON.LR_DECAY_STEPSIZE[0]
    if not args.lr:
        args.lr = cfg.TRAIN.COMMON.LEARNING_RATE
    if not args.lr_decay_gamma:
        args.lr_decay_gamma = cfg.TRAIN.COMMON.GAMMA
    if not args.max_epochs:
        args.max_epochs = cfg.TRAIN.COMMON.MAX_EPOCH
    print('Using config:')
    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.COMMON.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    pprint.pprint(cfg)
    if args.cuda:
        cfg.CUDA = True

    return args

def init_network(args, n_cls, resume = None):
    """
    :param args: define hyperparameters
    :param n_cls: number of object classes for initializing network output layers
    :return:
    """
    # initilize the network here.'
    if args.frame == 'faster_rcnn':
        if args.net == 'vgg16':
            Network = FasterRCNN.vgg16(n_cls, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            Network = FasterRCNN.resnet(n_cls, 101, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            Network = FasterRCNN.resnet(n_cls, 50, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            Network = FasterRCNN.resnet(n_cls, 152, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'faster_rcnn_vmrn':
        if args.net == 'vgg16':
            Network = VMRN.vgg16(n_cls, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            Network = VMRN.resnet(n_cls, 101, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            Network = VMRN.resnet(n_cls, 50, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            Network = VMRN.resnet(n_cls, 152, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'fpn':
        if args.net == 'res101':
            Network = FPN.resnet(n_cls, 101, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            Network = FPN.resnet(n_cls, 50, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            Network = FPN.resnet(n_cls, 152, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'fcgn':
        if args.net == 'res101':
            Network = FCGN.resnet(num_layers=101, pretrained=True)
        elif args.net == 'res50':
            Network = FCGN.resnet(num_layers=50, pretrained=True)
        elif args.net == 'res34':
            Network = FCGN.resnet(num_layers=34, pretrained=True)
        elif args.net == 'vgg16':
            Network = FCGN.vgg16(pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'roign':
        if args.net == 'res101':
            Network = ROIGN.resnet(n_cls, 101, pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'mgn':
        if args.net == 'res101':
            Network = MGN.resnet(n_cls, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'all_in_one':
        if args.net == 'res101':
            Network = ALL_IN_ONE.resnet(n_cls, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'ssd':
        if args.net == 'vgg16':
            Network = SSD.vgg16(n_cls, pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'ssd_vmrn':
        if args.net == 'vgg16':
            Network = SSD_VMRN.vgg16(n_cls, pretrained=True)
        elif args.net == 'res50':
            Network = SSD_VMRN.resnet(n_cls, layer_num=50, pretrained=True)
        elif args.net == 'res101':
            Network = SSD_VMRN.resnet(n_cls, layer_num=101, pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'vam':
        if args.net == 'vgg16':
            Network = VAM.vgg16(n_cls, pretrained=True)
        elif args.net == 'res50':
            Network = VAM.resnet(n_cls, layer_num=50, pretrained=True)
        elif args.net == 'res101':
            Network = VAM.resnet(n_cls, layer_num=101, pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    else:
        print("frame is not defined")
        pdb.set_trace()
    Network.create_architecture()

    lr = args.lr
    # tr_momentum = cfg.TRAIN.COMMON.MOMENTUM
    # tr_momentum = args.momentum

    if resume is not None:
        output_dir = resume['load_dir']
        iters_per_epoch = resume['iters_per_epoch']
        load_name = os.path.join(output_dir,
                                 args.frame + '_{}_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                        args.checkpoint, args.GPU))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        Network.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

        Network.iter_counter = (args.start_epoch - 1) * iters_per_epoch
        print("start iteration:", Network.iter_counter)

    if args.cuda:
        Network.cuda()

    if args.mGPUs:
        Network = nn.DataParallel(Network)

    params = []
    for key, value in dict(Network.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.COMMON.DOUBLE_BIAS + 1),
                          'weight_decay': cfg.TRAIN.COMMON.BIAS_DECAY and cfg.TRAIN.COMMON.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.COMMON.WEIGHT_DECAY}]

    # init optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.COMMON.MOMENTUM)
    if resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return Network, optimizer

def evalute_model(Network, namedb, args):
    max_per_image = 100

    # load test dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(namedb, False)
    if args.dataset[:4] == 'vmrd' or args.dataset[:7] == 'cornell' or args.dataset == 'jacquard':
        if args.frame != 'faster_rcnn' and args.frame !='ssd':
            if cfg.TRAIN.COMMON.AUGMENTATION:
                warnings.warn('########Grasps may be not rectangles due to augmentation!!!########')
    if args.frame in {"fpn", "faster_rcnn"}:
        dataset = objdetMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    elif args.frame in {"ssd"}:
        dataset = objdetRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    elif args.frame in {"ssd_vmrn", "vam"}:
        dataset = vmrdetRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    elif args.frame in {"faster_rcnn_vmrn"}:
        dataset = vmrdetMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    elif args.frame in {"fcgn"}:
        dataset = graspdetRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    elif args.frame in {"all_in_one"}:
        dataset = allInOneMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    elif args.frame in {"roign", "mgn"}:
        dataset = roigdetMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes)
    else:
        raise RuntimeError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)
    num_images = len(imdb.image_index)

    start = time.time()
    thresh = 0.01

    # init variables
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_rel = []
    all_grasp = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    Network.eval()
    empty_array= np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i in range(num_images):

        data_batch = next(data_iter)
        if args.cuda:
            data_batch = makeCudaData(data_batch)

        det_tic = time.time()
        if args.frame == 'faster_rcnn' or args.frame == 'fpn':
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            net_loss_cls, net_loss_bbox, \
            rois_label = Network(data_batch)

            boxes = rois.data[:, :, 1:5]
        elif args.frame == 'ssd':
            bbox_pred, cls_prob, \
            net_loss_bbox, net_loss_cls = Network(data_batch)

            boxes = Network.priors.type_as(bbox_pred)

        elif args.frame == 'faster_rcnn_vmrn':
            rois, cls_prob, bbox_pred, rel_result, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, RCNN_rel_loss_cls, \
            rois_label = Network(data_batch)

            boxes = rois.data[:, :, 1:5]

            all_rel.append(rel_result)

        elif args.frame == 'ssd_vmrn' or args.frame == 'vam':
            bbox_pred, cls_prob, rel_result, \
            loss_bbox, loss_cls, rel_loss_cls = Network(data_batch)

            boxes = Network.priors.type_as(bbox_pred)

            all_rel.append(rel_result)

        elif args.frame == 'fcgn':
            bbox_pred, cls_prob, loss_bbox, \
            loss_cls, rois_label, boxes = \
                Network(data_batch)

        elif args.frame == 'roign':
            rois, rpn_loss_cls, rpn_loss_box, rois_label, \
            grasp_loc, grasp_prob, grasp_bbox_loss, grasp_cls_loss, \
            grasp_conf_label, grasp_all_anchors = Network(data_batch)

            boxes = rois.data[:, :, 1:5]

        elif args.frame == 'mgn':
            rois, cls_prob, bbox_pred, rpn_loss_cls, \
            rpn_loss_box, loss_cls, loss_bbox, rois_label, \
            grasp_loc, grasp_prob, grasp_bbox_loss, \
            grasp_cls_loss, grasp_conf_label, grasp_all_anchors \
                = Network(data_batch)

            boxes = rois.data[:, :, 1:5]

        elif args.frame == 'all_in_one':
            rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_box, \
            loss_cls, loss_bbox, rel_loss_cls, rois_label, \
            grasp_loc, grasp_prob, grasp_bbox_loss, \
            grasp_cls_loss, grasp_conf_label, grasp_all_anchors \
                = Network(data_batch)

            boxes = rois.data[:, :, 1:5]

            all_rel.append(rel_result)

        # bs x N x N_class
        if args.frame != 'roign':
            scores = cls_prob.data
        if args.frame == 'mgn' or args.frame == 'all_in_one' or args.frame == 'roign':
            # bs*N x K*A x 2
            grasp_scores = grasp_prob.data

        if cfg.TEST.COMMON.BBOX_REG:
            # Apply bounding-box regression deltas
            if args.frame != 'roign':
                box_deltas = bbox_pred.data
            if args.frame == 'mgn' or args.frame == 'all_in_one' or args.frame == 'roign':
                grasp_box_deltas = grasp_loc.data
            if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.frame == 'fcgn':
                    box_deltas = box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
                    box_deltas = box_deltas.view(1, -1, 5)
                    pred_label = grasp_decode(box_deltas, boxes)
                    pred_boxes = labels2points(pred_label)
                    imshape = np.tile(np.array([cfg.TRAIN.COMMON.INPUT_SIZE,cfg.TRAIN.COMMON.INPUT_SIZE])
                                      ,(int(pred_boxes.size(1)),int(pred_boxes.size(2) / 2)))
                    imshape = torch.from_numpy(imshape).type_as(pred_boxes)
                    keep = (((pred_boxes > imshape) | (pred_boxes < 0)).sum(2) == 0)
                    pred_boxes = pred_boxes[keep]
                    scores = scores[keep]
                elif args.frame == 'mgn' or args.frame == 'all_in_one' or args.frame == 'roign':
                    grasp_box_deltas = grasp_box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
                    grasp_box_deltas = grasp_box_deltas.view(grasp_all_anchors.size())
                    # bs*N x K*A x 5
                    grasp_pred = grasp_decode(grasp_box_deltas, grasp_all_anchors)
                    # bs*N x K*A x 1
                    rois_w = (rois[:, :, 3] - rois[:, :, 1]).data.view(-1). \
                        unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 0:1])
                    rois_h = (rois[:, :, 4] - rois[:, :, 2]).data.view(-1). \
                        unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 1:2])
                    keep_mask = (grasp_pred[:, :, 0:1] > 0) & (grasp_pred[:, :, 1:2] > 0) &\
                                (grasp_pred[:, :, 0:1] < rois_w) & (grasp_pred[:, :, 1:2] < rois_h)
                    grasp_scores = (grasp_scores).contiguous().\
                        view(rois.size(0),rois.size(1), -1, 2)
                    # bs*N x 1 x 1
                    xleft = rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
                    ytop = rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
                    # rois offset
                    grasp_pred[:, :, 0:1] = grasp_pred[:, :, 0:1] + xleft
                    grasp_pred[:, :, 1:2] = grasp_pred[:, :, 1:2] + ytop
                    # bs x N x K*A x 8
                    grasp_pred_boxes = labels2points(grasp_pred).contiguous().view(rois.size(0), rois.size(1), -1, 8)
                    # bs x N x K*A
                    grasp_pos_scores = grasp_scores[:, :, :, 1]
                    # bs x N x K*A
                    _, grasp_score_idx = torch.sort(grasp_pos_scores, dim = 2, descending=True)
                    _, grasp_idx_rank = torch.sort(grasp_score_idx)
                    # bs x N x K*A mask
                    topn_grasp = 1
                    grasp_maxscore_mask = (grasp_idx_rank < topn_grasp)
                    # bs x N x topN
                    grasp_maxscores = grasp_scores[:, :, :, 1][grasp_maxscore_mask].contiguous().\
                        view(rois.size()[:2] + (topn_grasp,))
                    # scores = scores * grasp_maxscores[:, :, 0:1]
                    # bs x N x topN x 8
                    grasp_pred_boxes = grasp_pred_boxes[grasp_maxscore_mask].view(rois.size()[:2] + (topn_grasp, 8))
                    if args.frame != 'roign':
                        if args.class_agnostic:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                            pred_boxes = clip_boxes(pred_boxes, data_batch[1].data, 1)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                            pred_boxes = clip_boxes(pred_boxes, data_batch[1].data, 1)

                elif args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, data_batch[1].data, 1)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, data_batch[1].data, 1)
        else:
            if args.frame != 'roign':
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if args.frame != 'roign':
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            pred_boxes[:, 0::4] /= data_batch[1][0][3].item()
            pred_boxes[:, 1::4] /= data_batch[1][0][2].item()
            pred_boxes[:, 2::4] /= data_batch[1][0][3].item()
            pred_boxes[:, 3::4] /= data_batch[1][0][2].item()
        if args.frame == 'mgn' or args.frame == 'all_in_one' or args.frame == 'roign':
            grasp_pred_boxes = grasp_pred_boxes.squeeze()
            grasp_scores = grasp_scores.squeeze()
            if grasp_pred_boxes.dim() == 2:
                grasp_pred_boxes[:, 0::4] /= data_batch[1][0][3].item()
                grasp_pred_boxes[:, 1::4] /= data_batch[1][0][2].item()
                grasp_pred_boxes[:, 2::4] /= data_batch[1][0][3].item()
                grasp_pred_boxes[:, 3::4] /= data_batch[1][0][2].item()
            elif grasp_pred_boxes.dim() == 3:
                grasp_pred_boxes[:, :, 0::4] /= data_batch[1][0][3].item()
                grasp_pred_boxes[:, :, 1::4] /= data_batch[1][0][2].item()
                grasp_pred_boxes[:, :, 2::4] /= data_batch[1][0][3].item()
                grasp_pred_boxes[:, :, 3::4] /= data_batch[1][0][2].item()
            elif grasp_pred_boxes.dim() == 4:
                grasp_pred_boxes[:, :, :, 0::4] /= data_batch[1][0][3].item()
                grasp_pred_boxes[:, :, :, 1::4] /= data_batch[1][0][2].item()
                grasp_pred_boxes[:, :, :, 2::4] /= data_batch[1][0][3].item()
                grasp_pred_boxes[:, :, :, 3::4] /= data_batch[1][0][2].item()

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if args.frame != 'roign':
            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic or args.frame == 'fcgn':
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    if args.frame == 'mgn' or args.frame == 'all_in_one':
                        cur_grasp = grasp_pred_boxes[inds, :]
                        cur_grasp = cur_grasp[order]

                    if args.frame != 'fcgn':
                        keep = nms(cls_dets, cfg.TEST.COMMON.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]

                        if args.frame == 'mgn' or args.frame == 'all_in_one':
                            cur_grasp = cur_grasp[keep.view(-1).long()]

                        cls_dets = cls_dets.cpu().numpy()

                    else:
                        cls_dets = cls_dets[0:1]
                        cls_dets = cls_dets.cpu().numpy()
                        # for cornell grasp dataset, when testing, the input image is cropped from (100, 100), therefore,
                        # the coordinates of grasp rectangles should be added to this offset.
                        if args.dataset[:7] == 'cornell':
                            cls_dets[:, :8] += np.tile(np.array([[100, 100]]), 4)

                    all_boxes[j][i] = cls_dets
                    if args.frame == 'mgn' or args.frame == 'all_in_one':
                        all_grasp[j][i] = [cls_dets.copy(), cur_grasp]
                else:
                    all_boxes[j][i] = empty_array
        else:
            # N x 8
            if grasp_pred_boxes.dim() != 2:
                raise NotImplementedError
            cls_dets = grasp_pred_boxes[0:1]
            cls_dets = cls_dets.cpu().numpy()
            # for cornell grasp dataset, when testing, the input image is cropped from (100, 100), therefore,
            # the coordinates of grasp rectangles should be added to this offset.
            if args.dataset[:7] == 'cornell':
                cls_dets[:, :8] += np.tile(np.array([[100, 100]]), 4)
            all_boxes[1][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    if args.frame == 'mgn' or args.frame == 'all_in_one':
                        all_grasp[j][i][0] = all_grasp[j][i][0][keep, :]
                        all_grasp[j][i][1] = all_grasp[j][i][1][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                     .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    print('Evaluating detections')
    result = imdb.evaluate_detections(all_boxes, output_dir)

    if args.frame[-4:] == 'vmrn':
        print('Evaluating relationships')
        orec, oprec, imgprec, imgprec_difobjnum = imdb.evaluate_relationships(all_rel)
        print("object recall:   \t%.4f" % orec)
        print("object precision:\t%.4f" % oprec)
        print("image acc:       \t%.4f" % imgprec)
        print("image acc for images with different object numbers (2,3,4,5):")
        print("%s\t%s\t%s\t%s\t" % tuple(imgprec_difobjnum))
        result = imgprec

    if args.frame == 'mgn':
        print('Evaluating grasp detection results')
        grasp_MRFPPI, mean_MRFPPI = imdb.evaluate_multigrasp_detections(all_grasp)
        print('Mean Log-Average Miss Rate: %.4f' % np.mean(np.array(mean_MRFPPI)))
        result = mean_MRFPPI

    end = time.time()
    print("test time: %0.4fs" % (end - start))
    return result

if __name__ == '__main__':

    # init arguments
    args = read_cfgs()

    # check algorithm
    assert args.frame in LEGAL_FRAMES, "Illegal algorithm name."

    # check cuda devices
    if not torch.cuda.is_available():
        assert RuntimeError("Training can only be done by GPU. Please use --cuda to enable training.")
    if torch.cuda.is_available() and not args.cuda:
        assert RuntimeError("You have a CUDA device, so you should probably run with --cuda")

    # init random seed
    np.random.seed(cfg.RNG_SEED)

    # init logger
    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        current_t = time.time()
        logger = Logger(os.path.join('.', 'logs', str(current_t)))

    # init dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sampler_batch = sampler(train_size, args.batch_size)
    iters_per_epoch = int(train_size / args.batch_size)
    if args.frame in {"fpn", "faster_rcnn"}:
        dataset = objdetMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"ssd"}:
        dataset = objdetRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"ssd_vmrn", "vam"}:
        dataset = vmrdetRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"faster_rcnn_vmrn"}:
        dataset = vmrdetMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"fcgn"}:
        dataset = graspdetRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"all_in_one"}:
        dataset = allInOneMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"roign", "mgn"}:
        dataset = roigdetMulInSizeRoibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, cls_list=imdb.classes, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    else:
        raise RuntimeError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    # init output directory for model saving
    output_dir = args.save_dir + "/" + args.dataset + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.vis:
        visualizer = dataViewer(imdb.classes)
        data_vis_dir = os.path.join(args.save_dir, args.dataset, 'data_vis')
        if not os.path.exists(data_vis_dir):
            os.makedirs(data_vis_dir)

    # init network
    resume = None
    if args.resume:
        resume = {}
        resume['load_dir'] = output_dir
        resume['iters_per_epoch'] = iters_per_epoch
    Network, optimizer = init_network(args, imdb.classes, resume=resume)

    # init variables
    current_result, best_result, loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rel_pred, \
        loss_grasp_box, loss_grasp_cls, fg_cnt, bg_cnt, fg_grasp_cnt, bg_grasp_cnt = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
    save_flag, rois, rpn_loss_cls, rpn_loss_box, rel_loss_cls, cls_prob, bbox_pred, rel_cls_prob, loss_bbox, loss_cls, \
        rois_label, grasp_cls_loss, grasp_bbox_loss, grasp_conf_label = \
            False, None,None,None,None,None,None,None,None,None,None,None,None,None

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        Network.train()

        start = time.time()

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):

            # get data batch
            data_batch = next(data_iter)
            if args.vis:
                for i in range(data_batch[0].size(0)):
                    im_vis = visualizer.draw_graspdet_with_owner(data_batch[0][i].permute(1,2,0).numpy() + cfg.PIXEL_MEANS,
                        data_batch[2][i].numpy(),data_batch[3][i].numpy(),data_batch[7][i].numpy())
                    cv2.imwrite(os.path.join(data_vis_dir, "vis_batch" + str(step) + "_img" + str(i) + ".png"), im_vis)
            # ship to cuda
            if args.cuda:
                data_batch = makeCudaData(data_batch)

            # network forward
            Network.zero_grad()

            # forward process
            if args.frame == 'faster_rcnn_vmrn':
                rois, cls_prob, bbox_pred, rel_cls_prob, rpn_loss_cls, rpn_loss_box, loss_cls, \
                            loss_bbox, rel_loss_cls,rois_label = Network(data_batch)
                if rel_loss_cls == 0:
                    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + loss_cls.mean() + loss_bbox.mean()
                else:
                    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + loss_cls.mean() + loss_bbox.mean() + rel_loss_cls.mean()
            elif args.frame == 'faster_rcnn' or args.frame == 'fpn':
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, \
                            rois_label = Network(data_batch)
                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + loss_cls.mean() + loss_bbox.mean()
            elif args.frame == 'fcgn':
                bbox_pred, cls_prob, loss_bbox, loss_cls, rois_label,rois = Network(data_batch)
                loss = loss_bbox.mean() + loss_cls.mean()
            elif args.frame == 'roign':
                rois, rpn_loss_cls, rpn_loss_box, rois_label, grasp_loc, grasp_prob, grasp_bbox_loss, grasp_cls_loss, \
                            grasp_conf_label, grasp_all_anchors = Network(data_batch)
                loss = rpn_loss_box.mean() + rpn_loss_cls.mean() \
                       + cfg.MGN.OBJECT_GRASP_BALANCE * (grasp_bbox_loss.mean() + grasp_cls_loss.mean())
            elif args.frame == 'mgn':
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rois_label, grasp_loc, \
                grasp_prob, grasp_bbox_loss, grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
                loss = rpn_loss_box.mean() + rpn_loss_cls.mean() + loss_cls.mean() + loss_bbox.mean() + \
                       cfg.MGN.OBJECT_GRASP_BALANCE * (grasp_bbox_loss.mean() + grasp_cls_loss.mean())
            elif args.frame == 'all_in_one':
                rois, cls_prob, bbox_pred, rel_cls_prob, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rel_loss_cls, rois_label, \
                grasp_loc, grasp_prob, grasp_bbox_loss,grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
                loss = rpn_loss_box.mean() + rpn_loss_cls.mean() + loss_cls.mean() + loss_bbox.mean() + rel_loss_cls.mean() + \
                       cfg.MGN.OBJECT_GRASP_BALANCE * grasp_bbox_loss.mean() + grasp_cls_loss.mean()
            elif args.frame == 'ssd':
                bbox_pred, cls_prob, loss_bbox, loss_cls = Network(data_batch)
                loss = loss_bbox.mean() + loss_cls.mean()
            elif args.frame == 'ssd_vmrn' or args.frame == 'vam':
                bbox_pred, cls_prob, rel_result, loss_bbox, loss_cls, rel_loss_cls = Network(data_batch)
                if rel_loss_cls==0:
                    loss = loss_cls.mean() + loss_bbox.mean()
                else:
                    loss = loss_cls.mean() + loss_bbox.mean() + rel_loss_cls.mean()
            loss_temp += loss.data.item()

            # backward process
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                 clip_gradient(Network, 10.)
            optimizer.step()

            # record training information
            if args.mGPUs:
                if rpn_loss_cls is not None:
                    loss_rpn_cls += rpn_loss_cls.mean().data[0].item()
                if rpn_loss_box is not None:
                    loss_rpn_box += rpn_loss_box.mean().data[0].item()
                if loss_cls is not None:
                    loss_rcnn_cls += loss_cls.mean().data[0].item()
                if loss_bbox is not None:
                    loss_rcnn_box += loss_bbox.mean().data[0].item()
                if rel_loss_cls is not None and rel_loss_cls!=0:
                    loss_rel_pred += rel_loss_cls.mean().data[0].item()
                if grasp_cls_loss is not None:
                    loss_grasp_cls += grasp_cls_loss.mean().data[0].item()
                if grasp_bbox_loss is not None:
                    loss_grasp_box += grasp_bbox_loss.mean().data[0].item()
                if rois_label is not None:
                    tempfg = torch.sum(rois_label.data.ne(0))
                    fg_cnt += tempfg
                    bg_cnt += (rois_label.data.numel() - tempfg)
                if grasp_conf_label is not None:
                    tempfg = torch.sum(grasp_conf_label.data.ne(0))
                    fg_grasp_cnt += tempfg
                    bg_grasp_cnt += (grasp_conf_label.data.numel() - tempfg)
            else:
                if rpn_loss_cls is not None:
                    loss_rpn_cls += rpn_loss_cls.item()
                if rpn_loss_cls is not None:
                    loss_rpn_box += rpn_loss_box.item()
                if loss_cls is not None:
                    loss_rcnn_cls += loss_cls.item()
                if loss_bbox is not None:
                    loss_rcnn_box += loss_bbox.item()
                if rel_loss_cls is not None and rel_loss_cls != 0:
                    loss_rel_pred += rel_loss_cls.item()
                if grasp_cls_loss is not None:
                    loss_grasp_cls += grasp_cls_loss.item()
                if grasp_bbox_loss is not None:
                    loss_grasp_box += grasp_bbox_loss.item()
                if rois_label is not None:
                    tempfg = torch.sum(rois_label.data.ne(0))
                    fg_cnt += tempfg
                    bg_cnt += (rois_label.data.numel() - tempfg)
                if grasp_conf_label is not None:
                    tempfg = torch.sum(grasp_conf_label.data.ne(0))
                    fg_grasp_cnt += tempfg
                    bg_grasp_cnt += (grasp_conf_label.data.numel() - tempfg)

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval
                    loss_rpn_cls /= args.disp_interval
                    loss_rpn_box /= args.disp_interval
                    loss_rcnn_cls /= args.disp_interval
                    loss_rcnn_box /= args.disp_interval
                    loss_rel_pred /= args.disp_interval
                    loss_grasp_cls /= args.disp_interval
                    loss_grasp_box /= args.disp_interval

                print("[session %d][epoch %2d][iter %4d/%4d] \n\t\t\tloss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, optimizer.param_groups[0]['lr']))
                print('\t\t\ttime cost: %f' % (end - start,))
                if rois_label is not None:
                    print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt, bg_cnt))
                if grasp_conf_label is not None:
                    print("\t\t\tgrasp_fg/grasp_bg=(%d/%d)" % (fg_grasp_cnt, bg_grasp_cnt))
                if rpn_loss_box is not None and rpn_loss_cls is not None:
                    print("\t\t\trpn_cls: %.4f\n\t\t\trpn_box: %.4f\n\t\t\trcnn_cls: %.4f\n\t\t\trcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                else:
                    print("\t\t\trcnn_cls: %.4f\n\t\t\trcnn_box %.4f" \
                          % (loss_rcnn_cls, loss_rcnn_box))
                if rel_loss_cls is not None:
                    print("\t\t\trel_loss %.4f" \
                          % (loss_rel_pred,))
                if grasp_cls_loss is not None and grasp_bbox_loss is not None:
                    print("\t\t\tgrasp_cls: %.4f\n\t\t\tgrasp_box %.4f" \
                          % (loss_grasp_cls, loss_grasp_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                    }
                    if rpn_loss_cls:
                        info['loss_rpn_cls'] = loss_rpn_cls
                    if rpn_loss_box:
                        info['loss_rpn_box'] = loss_rpn_box
                    if rel_loss_cls:
                        info['loss_rel_pred'] = loss_rel_pred
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, Network.iter_counter)

                loss_temp = 0.
                loss_rpn_cls = 0.
                loss_rpn_box = 0.
                loss_rcnn_cls = 0.
                loss_rcnn_box = 0.
                loss_rel_pred = 0.
                loss_grasp_box = 0.
                loss_grasp_cls = 0.
                fg_cnt = 0.
                bg_cnt = 0.
                fg_grasp_cnt = 0.
                bg_grasp_cnt = 0.
                start = time.time()

            # adjust learning rate
            if args.lr_decay_step == 0:
                # clr = lr / (1 + decay * n) -> lr_n / lr_n+1 = (1 + decay * (n+1)) / (1 + decay * n)
                decay = (1 + args.lr_decay_gamma * Network.iter_counter) / (1 + args.lr_decay_gamma * (Network.iter_counter + 1))
                adjust_learning_rate(optimizer, decay)
            elif Network.iter_counter % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)

            # test and save
            if Network.iter_counter % cfg.TRAIN.COMMON.SNAPSHOT_ITERS == 0:
                # test network and record results
                Network.eval()
                current_result = evalute_model(Network, args.imdbval_name, args)
                if args.use_tfboard:
                    logger.scalar_summary('mAP', current_result, Network.iter_counter)
                Network.train()

                if cfg.TRAIN.COMMON.SNAPSHOT_AFTER_TEST:
                    if current_result > best_result:
                        best_result = current_result
                        save_flag = True
                else:
                    save_flag = True

                if save_flag:
                    if args.mGPUs:
                        save_name = os.path.join(output_dir,
                                                 args.frame + '_{}_{}_{}_{}.pth'.format(args.session, epoch, step,
                                                                                        args.imdb_name))
                        save_checkpoint({
                            'session': args.session,
                            'epoch': epoch + 1,
                            'model': Network.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'pooling_mode': cfg.RCNN_COMMON.POOLING_MODE,
                            'class_agnostic': args.class_agnostic,
                        }, save_name)
                    else:
                        save_name = os.path.join(output_dir,
                                                 args.frame + '_{}_{}_{}_{}.pth'.format(args.session, epoch, step, args.GPU))
                        save_checkpoint({
                            'session': args.session,
                            'epoch': epoch + 1,
                            'model': Network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'pooling_mode': cfg.RCNN_COMMON.POOLING_MODE,
                            'class_agnostic': args.class_agnostic,
                        }, save_name)
                    print('save model: {}'.format(save_name))
                    save_flag = False

        end = time.time()
        print(end - start)
