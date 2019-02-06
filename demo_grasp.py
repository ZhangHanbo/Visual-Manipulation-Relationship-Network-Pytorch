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

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.FasterRCNN import vgg16
from model.FasterRCNN import resnet
import pdb
import model.FullyConvGrasp as FCGN

from model.fully_conv_grasp.bbox_transform_grasp import labels2points, grasp_decode
from model.utils.net_utils import save_net, load_net, vis_detections, draw_grasp

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Use trained Faster R-CNN network to detect')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/pascal_voc_faster_rcnn_res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="output")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
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
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    args = parser.parse_args()
    return args

def _get_fixed_size_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    h,w,_ = im_orig.shape

    if h > w:
        dif = (h - w)/2
        im_orig = im_orig[int(dif):int(w+dif),:,:]
        im_scale = cfg.TRAIN.COMMON.INPUT_SIZE / w
    elif h<w:
        dif = (w - h) / 2
        im_orig = im_orig[:, int(dif):int((h + dif)), :]
        im_scale = cfg.TRAIN.COMMON.INPUT_SIZE / h
    else:
        im_scale = cfg.TRAIN.COMMON.INPUT_SIZE / h

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)

    return im

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])

    target_size = 600
    im_scale = float(target_size) / float(im_size_min)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im

if __name__ == '__main__':

    args = parse_args()

    args.cfg_file =  "cfgs/cornell_{}_{}.yml".format('fcgn', args.net)

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'fcgn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__','grasp'])

    # initilize the network here.
    if args.net == 'vgg16':
        Network = FCGN.vgg16(pretrained=True)
    elif args.net == 'res101':
        Network = FCGN.resnet(101, pretrained=False)
    elif args.net == 'res50':
        Network = FCGN.resnet(50, pretrained=False)
    else:
        print("network is not defined")
        pdb.set_trace()

    Network.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    Network.load_state_dict(checkpoint['model'])

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        Network.cuda()

    Network.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    while (num_images >= 0):
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Load the demo image
        im_file = os.path.join(args.image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs = _get_image_blob(im)

        im_data_pt = torch.from_numpy(blobs)
        im_data_pt = im_data_pt.unsqueeze(0)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        bbox_pred, cls_prob, loss_bbox, \
        loss_cls, rois_label, boxes = \
            Network(im_data, None, gt_boxes, num_boxes)

        scores = cls_prob.data

        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
        box_deltas = box_deltas.view(1, -1, 5)
        pred_label = grasp_decode(box_deltas, boxes)
        pred_boxes = labels2points(pred_label)
        imshape = np.tile(np.array([800,800])
                                      ,(int(pred_boxes.size(1)),int(pred_boxes.size(2) / 2)))
        imshape = torch.from_numpy(imshape).type_as(pred_boxes)
        keep = (((pred_boxes > imshape) | (pred_boxes < 0)).sum(2) == 0)
        pred_boxes = pred_boxes[keep]
        scores = scores[keep]


        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(blobs) + cfg.PIXEL_MEANS
        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds, :]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                cls_dets = cls_dets[cls_dets[:,-1]>0.2]
                if vis:
                    im2show = draw_grasp(im2show, cls_dets)
                cls_dets = cls_dets.cpu().numpy()

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                            .format(num_images + 1, len(imglist), detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
