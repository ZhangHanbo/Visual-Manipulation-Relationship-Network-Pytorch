# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.fully_conv_grasp.bbox_transform_grasp import labels2points, grasp_decode
from model.utils.net_utils import save_net, load_net, vis_detections, draw_grasp
from model.FasterRCNN import vgg16
from model.FasterRCNN import resnet
from model import SSD
from model import FPN
from model import VMRN
import model.SSD_VMRN as SSD_VMRN
import model.FullyConvGrasp as FCGN
import model.MultiGrasp as MGN
import model.AllinOne as ALL_IN_ONE

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/pascal_voc_faster_rcnn_res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--frame', dest='frame',
                        help='faster_rcnn, ssd, faster_rcnn_vmrn, ssd_vmrn',
                        default='faster_rcnn', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="output",
                        type=str)
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
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=2504, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.COMMON.LEARNING_RATE
momentum = cfg.TRAIN.COMMON.MOMENTUM
weight_decay = cfg.TRAIN.COMMON.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.frame == 'ssd' or args.frame == 'ssd_vmrn':
        args.class_agnostic = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        #args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        #args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == 'vmrdcompv1':
        args.imdb_name = "vmrd_compv1_trainval"
        args.imdbval_name = "vmrd_compv1_test"
        #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == 'bdds':
        args.imdb_name = "bdds_trainval"
        args.imdbval_name = "bdds_test"
        #args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset[:7] == 'cornell':
        cornell = args.dataset.split('_')
        args.frame = 'fcgn'
        args.imdb_name = 'cornell_{}_{}_trainval_{}'.format(cornell[1],cornell[2],cornell[3])
        args.imdbval_name = 'cornell_{}_{}_test_{}'.format(cornell[1],cornell[2],cornell[3])
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '50']
    elif args.dataset[:8] == 'jacquard':
        jacquard = args.dataset.split('_')
        args.imdb_name = 'jacquard_{}_trainval_{}'.format(jacquard[1], jacquard[2])
        args.imdbval_name = 'jacquard_{}_test_{}'.format(jacquard[1], jacquard[2])
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '1000']

    if args.dataset[:7] == 'cornell':
        args.cfg_file = "cfgs/cornell_{}_{}_ls.yml".format(args.frame, args.net) if args.large_scale \
        else "cfgs/cornell_{}_{}.yml".format(args.frame, args.net)
    elif args.dataset[:8] == 'jacquard':
        args.cfg_file = "cfgs/jacquard_{}_{}_ls.yml".format(args.frame, args.net) if args.large_scale \
        else "cfgs/jacquard_{}_{}.yml".format(args.frame, args.net)
    else:
        args.cfg_file = "cfgs/{}_{}_{}_ls.yml".format(args.dataset, args.frame, args.net) if args.large_scale \
        else "cfgs/{}_{}_{}.yml".format(args.dataset, args.frame, args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.COMMON.USE_FLIPPED = False
    cfg.TRAIN.COMMON.USE_VERTICAL_ROTATED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)

    load_name = os.path.join(input_dir,
                        args.frame + '_{}_{}_{}_{}.pth'.format(args.checksession,
                                    args.checkepoch, args.checkpoint, args.GPU))

    # initilize the network here.
    if args.frame == 'ssd':
        if args.net == 'vgg16':
            Network = SSD.vgg16(imdb.classes)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'ssd_vmrn':
        if args.net == 'vgg16':
            Network = SSD_VMRN.vgg16(imdb.classes)
        elif args.net == 'res50' :
            Network = SSD_VMRN.resnet(imdb.classes, layer_num=50)
        elif args.net == 'res101' :
            Network = SSD_VMRN.resnet(imdb.classes, layer_num=101)
        else:
            print("network is not defined")
            pdb.set_trace()
    if args.frame == 'fpn':
        if args.net == 'res101':
            Network = FPN.resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'faster_rcnn':
        if args.net == 'vgg16':
            Network = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            Network = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            Network = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            Network = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'faster_rcnn_vmrn':
        if args.net == 'vgg16':
            Network = VMRN.vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            Network = VMRN.resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            Network = VMRN.resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            Network = VMRN.resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'fcgn':
        if args.net == 'res101':
            Network = FCGN.resnet(num_layers = 101, pretrained=True)
        elif args.net == 'res50':
            Network = FCGN.resnet(num_layers = 50, pretrained=True)
        elif args.net == 'res34':
            Network = FCGN.resnet(num_layers = 34, pretrained=True)
        elif args.net == 'vgg16':
            Network = FCGN.vgg16(pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'mgn':
        if args.net == 'res101':
            Network = MGN.resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    elif args.frame == 'all_in_one':
        if args.net == 'res101':
            Network = ALL_IN_ONE.resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()
    Network.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    Network.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    num_grasps = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    gt_grasps = torch.FloatTensor(1)
    # visual manipulation relationship matrix
    rel_mat = torch.FloatTensor(1)
    gt_grasp_inds = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        num_grasps = num_grasps.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_grasps = gt_grasps.cuda()
        rel_mat = rel_mat.cuda()
        gt_grasp_inds = gt_grasp_inds.cuda()

    # make variable
    im_data = Variable(im_data,requires_grad = False)
    im_info = Variable(im_info,requires_grad = False)
    num_grasps = Variable(num_grasps,requires_grad = False)
    num_boxes = Variable(num_boxes,requires_grad = False)
    gt_boxes = Variable(gt_boxes,requires_grad = False)
    gt_grasps = Variable(gt_grasps,requires_grad = False)
    rel_mat = Variable(rel_mat,requires_grad = False)
    gt_grasp_inds = Variable(gt_grasp_inds,requires_grad = False)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        Network.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0
    else:
        thresh = 0.05

    save_name = args.frame + args.net + args.dataset
    num_images = len(roidb)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    # store relationship detection results
    all_rel = []
    # for multi-grasp network
    all_grasp = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    Network.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    if vis:
        color_pool = [
            (255, 0, 0),
            (255, 102, 0),
            (255, 153, 0),
            (255, 204, 0),
            (255, 255, 0),
            (204, 255, 0),
            (153, 255, 0),
            (0, 255, 51),
            (0, 255, 153),
            (0, 255, 204),
            (0, 255, 255),
            (0, 204, 255),
            (0, 153, 255),
            (0, 102, 255),
            (102, 0, 255),
            (153, 0, 255),
            (204, 0, 255),
            (255, 0, 204),
            (187, 68, 68),
            (187, 116, 68),
            (187, 140, 68),
            (187, 163, 68),
            (187, 187, 68),
            (163, 187, 68),
            (140, 187, 68),
            (68, 187, 92),
            (68, 187, 140),
            (68, 187, 163),
            (68, 187, 187),
            (68, 163, 187),
            (68, 140, 187),
            (68, 116, 187),
            (116, 68, 187),
            (140, 68, 187),
            (163, 68, 187),
            (187, 68, 163),
            (255, 119, 119),
            (255, 207, 136),
            (119, 255, 146),
            (153, 214, 255)
        ]
        np.random.shuffle(color_pool)
        np.random.shuffle(color_pool)
        color_dict = {}
        for i, clsname in enumerate(imdb.classes):
            color_dict[clsname] = color_pool[i]

    for i in range(num_images):

        # get data batch
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        gt_grasps.data.resize_(data[3].size()).copy_(data[3])
        num_boxes.data.resize_(data[4].size()).copy_(data[4])
        num_grasps.data.resize_(data[5].size()).copy_(data[5])
        rel_mat.data.resize_(data[6].size()).copy_(data[6])
        gt_grasp_inds.data.resize_(data[7].size()).copy_(data[7])

        det_tic = time.time()
        if args.frame == 'faster_rcnn' or args.frame == 'fpn':
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            net_loss_cls, net_loss_bbox, \
            rois_label = Network(im_data, im_info, gt_boxes, num_boxes)

            boxes = rois.data[:, :, 1:5]
        elif args.frame == 'ssd':
            bbox_pred, cls_prob, \
            net_loss_bbox, net_loss_cls = Network(im_data, im_info, gt_boxes, num_boxes)

            boxes = Network.priors.type_as(bbox_pred)

        elif args.frame == 'ssd_vmrn':
            bbox_pred, cls_prob, rel_result,\
            net_loss_bbox, net_loss_cls, rel_loss_cls = Network(im_data, im_info, gt_boxes, num_boxes, rel_mat)

            boxes = Network.priors.type_as(bbox_pred)

            all_rel.append(rel_result)

        elif args.frame[-4:] == 'vmrn':
            rois, cls_prob, bbox_pred, rel_result, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, RCNN_rel_loss_cls, \
            rois_label = Network(im_data, im_info, gt_boxes, num_boxes, rel_mat)

            boxes = rois.data[:, :, 1:5]

            all_rel.append(rel_result)
        elif args.frame == 'fcgn':
            bbox_pred, cls_prob, loss_bbox, \
            loss_cls, rois_label, boxes = \
                Network(im_data, im_info, gt_grasps, num_boxes)

        elif args.frame == 'mgn':
            gt = {
                'boxes': gt_boxes,
                'grasps': gt_grasps,
                'grasp_inds': gt_grasp_inds,
                'num_boxes': num_boxes,
                'num_grasps': num_grasps,
                'im_info': im_info
            }
            rois, cls_prob, bbox_pred, rpn_loss_cls, \
            rpn_loss_box, loss_cls, loss_bbox, rois_label, \
            grasp_loc, grasp_prob, grasp_bbox_loss, \
            grasp_cls_loss, grasp_conf_label, grasp_all_anchors \
                = Network(im_data, gt)

            boxes = rois.data[:, :, 1:5]

        elif args.frame == 'all_in_one':
            gt = {
                'boxes': gt_boxes,
                'grasps': gt_grasps,
                'grasp_inds': gt_grasp_inds,
                'num_boxes': num_boxes,
                'num_grasps': num_grasps,
                'im_info': im_info,
                'rel_mat': rel_mat
            }
            rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_box, \
            loss_cls, loss_bbox, rel_loss_cls, rois_label, \
            grasp_loc, grasp_prob, grasp_bbox_loss, \
            grasp_cls_loss, grasp_conf_label, grasp_all_anchors \
                = Network(im_data, gt)

            boxes = rois.data[:, :, 1:5]

            all_rel.append(rel_result)

        # bs x N x N_class
        scores = cls_prob.data
        if args.frame == 'mgn' or args.frame == 'all_in_one':
            # bs*N x K*A x 2
            grasp_scores = grasp_prob.data

        if cfg.TEST.COMMON.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if args.frame == 'mgn' or args.frame == 'all_in_one':
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
                elif args.frame == 'mgn' or args.frame == 'all_in_one' :
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
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

                elif args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        pred_boxes[:, 0::4] /= data[1][0][3].item()
        pred_boxes[:, 1::4] /= data[1][0][2].item()
        pred_boxes[:, 2::4] /= data[1][0][3].item()
        pred_boxes[:, 3::4] /= data[1][0][2].item()
        if args.frame == 'mgn' or args.frame == 'all_in_one':
            grasp_pred_boxes = grasp_pred_boxes.squeeze()
            grasp_scores = grasp_scores.squeeze()
            if grasp_pred_boxes.dim() == 2:
                grasp_pred_boxes[:, 0::4] /= data[1][0][3].item()
                grasp_pred_boxes[:, 1::4] /= data[1][0][2].item()
                grasp_pred_boxes[:, 2::4] /= data[1][0][3].item()
                grasp_pred_boxes[:, 3::4] /= data[1][0][2].item()
            elif grasp_pred_boxes.dim() == 3:
                grasp_pred_boxes[:, :, 0::4] /= data[1][0][3].item()
                grasp_pred_boxes[:, :, 1::4] /= data[1][0][2].item()
                grasp_pred_boxes[:, :, 2::4] /= data[1][0][3].item()
                grasp_pred_boxes[:, :, 3::4] /= data[1][0][2].item()
            elif grasp_pred_boxes.dim() == 4:
                grasp_pred_boxes[:, :, :, 0::4] /= data[1][0][3].item()
                grasp_pred_boxes[:, :, :, 1::4] /= data[1][0][2].item()
                grasp_pred_boxes[:, :, :, 2::4] /= data[1][0][3].item()
                grasp_pred_boxes[:, :, :, 3::4] /= data[1][0][2].item()

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        im2show_gr = None
        im2show_obj = None
        obj_index_begin = 0
        if vis:
            im = cv2.imread(imdb.image_path_at(roidb[i]['img_id']))
            im2show_gr = np.copy(im)
            im2show_obj = np.copy(im)
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
                    obj_index_end = obj_index_begin + (cls_dets[:, -1] > cfg.TEST.COMMON.OBJ_DET_THRESHOLD).sum()

                    if args.frame == 'mgn' or args.frame == 'all_in_one':
                        cur_grasp = cur_grasp[keep.view(-1).long()]
                    if vis:
                        if args.frame == 'mgn' or args.frame == 'all_in_one':
                            im2show_obj, im2show_gr = vis_detections(im2show_obj, im2show_gr,
                                                     imdb.classes[j], cls_dets.cpu().numpy(),
                                                     cfg.TEST.COMMON.OBJ_DET_THRESHOLD, cur_grasp.cpu().numpy(),
                                                     color_dict=color_dict, index=range(obj_index_begin, obj_index_end))
                        else:
                            im2show_obj, im2show_gr = vis_detections(im2show_obj, im2show_gr,
                                                     imdb.classes[j], cls_dets.cpu().numpy(),
                                                     cfg.TEST.COMMON.OBJ_DET_THRESHOLD,
                                                     color_dict=color_dict, index=range(obj_index_begin, obj_index_end))
                            
                    cls_dets = cls_dets.cpu().numpy()
                    obj_index_begin = obj_index_end
                    
                else:
                    cls_dets = cls_dets[0:1]
                    if vis:
                        im2show_gr = np.array(im_data[0].data.permute(1,2,0)) + cfg.PIXEL_MEANS
                        im2show_gr = draw_grasp(im2show_gr, cls_dets)
                    cls_dets = cls_dets.cpu().numpy()
                    # offset comming from crop
                    cls_dets[:, :8] += np.tile(np.array([[100, 100]]), 4)
                all_boxes[j][i] = cls_dets
                if args.frame == 'mgn' or args.frame == 'all_in_one':
                    all_grasp[j][i] = [cls_dets.copy(), cur_grasp]
            else:
                all_boxes[j][i] = empty_array

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

        if vis:
            result_dir = output_dir + '/' + args.frame + args.net + args.dataset
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if im2show_obj is not None:
                cv2.imwrite(result_dir + '/' + '{:d}obj_det.png'.format(i), im2show_obj)
            if im2show_gr is not None:
                cv2.imwrite(result_dir + '/' + '{:d}gr_det.png'.format(i), im2show_gr)
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        if args.frame == 'mgn':
            pickle.dump((all_boxes, all_grasp), f, pickle.HIGHEST_PROTOCOL)
        elif args.frame[-4:] == 'vmrn':
            pickle.dump((all_boxes, all_rel), f, pickle.HIGHEST_PROTOCOL)
        elif args.frame == 'all_in_one':
            pickle.dump((all_boxes, all_grasp, all_rel), f, pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    if args.frame[-4:] == 'vmrn' or args.frame == 'all_in_one':
        print('Evaluating relationships')
        orec, oprec, imgprec, imgprec_difobjnum = imdb.evaluate_relationships(all_rel)
        print("object recall:   \t%.4f" %  orec)
        print("object precision:\t%.4f" %  oprec)
        print("image acc:       \t%.4f" %  imgprec)
        print("image acc for images with different object numbers (2,3,4,5):")
        print("%s\t%s\t%s\t%s\t" % tuple(imgprec_difobjnum))

    if args.frame == 'mgn' or args.frame == 'all_in_one':
        print('Evaluating grasp detection results')
        grasp_MRFPPI, mean_MRFPPI, keypoint = imdb.evaluate_multigrasp_detections(all_grasp)
        with open('MRFPPI.pkl', 'wb') as f:
            pickle.dump(grasp_MRFPPI, f, pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
