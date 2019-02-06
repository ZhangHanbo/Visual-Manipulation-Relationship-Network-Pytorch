# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network.
   Modified by Hanbo Zhang to support Visual Manipulation Relationship Network
   and Visual Manipulation Relationship Dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob,prep_im_for_blob_fixed_size,prep_im_for_blob_aug, im_list_to_blob
import pdb
import cv2
from model.utils.net_utils import draw_grasp

def get_minibatch(roidb, num_classes, training = True):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    gt_boxes = None
    gt_grasps = None

    if 'boxes' in roidb[0]:
        box_dim = roidb[0]['boxes'].shape[1]

    assert(cfg.TRAIN.RCNN_COMMON.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.RCNN_COMMON.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    if (training and not cfg.TRAIN.COMMON.AUGMENTATION and not cfg.TRAIN.COMMON.FIXED_INPUT_SIZE) \
            or (not training and not cfg.TEST.COMMON.FIXED_INPUT_SIZE and not cfg.TEST.COMMON.AUGMENTATION):

        # Sample random scales to use for each image in this batch
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.RCNN_COMMON.SCALES),
                                        size=num_images)
        im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

        # deal with object bbox
        if 'boxes' in roidb[0]:
            if cfg.TRAIN.COMMON.USE_ALL_GT:
                # Include all ground truth boxes
                gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
            else:
                # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
                gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
            # gt boxes: (x1, y1, x2, y2, cls)
            gt_boxes = np.empty((len(gt_inds), box_dim + 1), dtype=np.float32)
            gt_boxes[:, :box_dim] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
            gt_boxes[:, box_dim] = roidb[0]['gt_classes'][gt_inds]
            gt_bk = None

        # deal with grasps
        if 'grasps' in roidb[0] and roidb[0]['grasps'].size > 0:
            gt_grasps = roidb[0]['grasps'].astype(np.float32) * im_scales[0]
            if 'grasp_inds' in roidb[0]:
                gt_grasp_inds = roidb[0]['grasp_inds']

    elif (training and cfg.TRAIN.COMMON.FIXED_INPUT_SIZE and not cfg.TRAIN.COMMON.AUGMENTATION) \
            or (not training and cfg.TEST.COMMON.FIXED_INPUT_SIZE and not cfg.TEST.COMMON.AUGMENTATION):

        im_blob, im_scales = _get_image_blob(roidb)

        # deal with object bbox
        if 'boxes' in roidb[0]:
            if cfg.TRAIN.COMMON.USE_ALL_GT:
                # Include all ground truth boxes
                gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
            else:
                # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
                gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
            gt_boxes = np.empty((len(gt_inds), box_dim + 1), dtype=np.float32)

            gt_boxes[:, :box_dim] = roidb[0]['boxes'][gt_inds, :]
            gt_boxes[:, :box_dim][:, 0::2] *= im_scales[0]['x']
            gt_boxes[:, :box_dim][:, 1::2] *= im_scales[0]['y']
            gt_boxes[:, box_dim] = roidb[0]['gt_classes'][gt_inds]
            gt_bk = None

        # deal with grasps
        if 'grasps' in roidb[0] and roidb[0]['grasps'].size > 0:
            gt_grasps = roidb[0]['grasps'].astype(np.float32)
            gt_grasps[:, 0::2] *= im_scales[0]['x']
            gt_grasps[:, 1::2] *= im_scales[0]['y']
            if 'grasp_inds' in roidb[0]:
                gt_grasp_inds = roidb[0]['grasp_inds']

    elif training and cfg.TRAIN.COMMON.FIXED_INPUT_SIZE and cfg.TRAIN.COMMON.AUGMENTATION \
            or (not training and cfg.TEST.COMMON.FIXED_INPUT_SIZE and cfg.TEST.COMMON.AUGMENTATION):
        im_blob, im_scales, gt_b, gt_c, gt_g, gt_bk, gt_gk =  \
            _get_image_blob_with_aug(roidb, training=training)

        if gt_b is not None:
            if cfg.TRAIN.COMMON.USE_ALL_GT:
                # Include all ground truth boxes
                gt_inds = np.where(gt_c[0] != 0)[0]
            else:
                # TODO: For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
                pass
            gt_boxes = np.empty((len(gt_inds), box_dim + 1), dtype=np.float32)
            gt_boxes[:, :box_dim] = gt_b[0][gt_inds, :]
            gt_boxes[:, :box_dim][:, 0::2] *= im_scales[0]['x']
            gt_boxes[:, :box_dim][:, 1::2] *= im_scales[0]['y']
            gt_boxes[:, box_dim] = gt_c[0][gt_inds]

        if gt_g is not None:
            gt_grasps = gt_g[0].astype(np.float32)
            gt_grasps[:, 0::2] *= im_scales[0]['x']
            gt_grasps[:, 1::2] *= im_scales[0]['y']
            if 'grasp_inds' in roidb[0]:
                gt_grasp_inds = roidb[0]['grasp_inds'][gt_gk]

    elif training and not cfg.TRAIN.COMMON.FIXED_INPUT_SIZE and cfg.TRAIN.COMMON.AUGMENTATION \
            or (not training and not cfg.TEST.COMMON.FIXED_INPUT_SIZE and cfg.TEST.COMMON.AUGMENTATION):
        # Sample random scales to use for each image in this batch
        # TODO: SUPPORT MULTI-IMAGE BATCH
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.RCNN_COMMON.SCALES),
                                      size=num_images)
        im_blob, im_scales, gt_b, gt_c, gt_g, gt_bk, gt_gk = \
            _get_image_blob_with_aug(roidb, random_scale_inds, training)

        if gt_b is not None:
            if cfg.TRAIN.COMMON.USE_ALL_GT:
                # Include all ground truth boxes
                gt_inds = np.where(gt_c[0] != 0)[0]
            else:
                # TODO: For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
                pass
            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            # gt boxes: (x1, y1, x2, y2, cls)
            gt_boxes[:, :box_dim] = gt_b[0][gt_inds, :] * im_scales[0]
            gt_boxes[:, box_dim] = roidb[0]['gt_classes'][gt_inds]

        if gt_g is not None:
            gt_grasps = gt_g[0].astype(np.float32) * im_scales[0]
            if 'grasp_inds' in roidb[0]:
                gt_grasp_inds = roidb[0]['grasp_inds'][gt_gk]
    else:
        assert 0, "logic error"

    blobs = {}
    blobs['data'] = im_blob
    if gt_boxes is not None:
        blobs['gt_boxes'] = gt_boxes
    if gt_grasps is not None:
        blobs['gt_grasps'] = gt_grasps

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    if isinstance(im_scales[0],dict):
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]['y'], im_scales[0]['x']]],
            dtype=np.float32)
    else:
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0], im_scales[0]]],
            dtype=np.float32)

    # TODO: MAKE SURE THAT OBJECTS DELETED ARE CONSIDERED
    if 'nodeinds' in roidb[0]:
        if gt_bk is not None:
            blobs['nodeinds'] = roidb[0]['nodeinds'][gt_bk[0]][gt_inds]
        else:
            blobs['nodeinds'] = roidb[0]['nodeinds'][gt_inds]
    if 'fathers' in roidb[0]:
        if gt_bk is not None:
            blobs['fathers'] = [roidb[0]['fathers'][f_ind] for f_ind in list(gt_bk[0][gt_inds])]
        else:
            blobs['fathers'] = [roidb[0]['fathers'][f_ind] for f_ind in list(gt_inds)]
    if 'children' in roidb[0]:
        if gt_bk is not None:
            blobs['children'] = [roidb[0]['children'][c_ind] for c_ind in list(gt_bk[0][gt_inds])]
        else:
            blobs['children'] = [roidb[0]['children'][c_ind] for c_ind in list(gt_inds)]
    if 'grasp_inds' in roidb[0]:
        blobs['gt_grasp_inds'] = gt_grasp_inds
    blobs['img_id'] = roidb[0]['img_id']
    blobs['img_path'] = roidb[0]['image']

    return blobs

def _get_image_blob(roidb, scale_inds = -1):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        #im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])

        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im,im,im), axis=2)

        # rgb -> bgr
        im = im[:, :, ::-1]

        # flip the channel, since the original one using cv2
        im = np.rot90(im, roidb[i]['rotated'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        if scale_inds == -1:
            # origion code
            target_size = cfg.TRAIN.COMMON.INPUT_SIZE
            im, im_scale = prep_im_for_blob_fixed_size(im, cfg.PIXEL_MEANS, target_size)
        else:
            # origion code
            target_size = cfg.TRAIN.RCNN_COMMON.SCALES[scale_inds[i]]
            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.COMMON.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _get_image_blob_with_aug(roidb, scale_inds = -1, training = True):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    gt_boxes = None
    gt_boxes_keep = None
    gt_classes = None
    gt_grasps = None
    gt_grasps_keep = None

    if 'boxes' in roidb[0]:
        gt_boxes = []
        gt_classes = []
        gt_boxes_keep = []

    if 'grasps' in roidb[0] and roidb[0]['grasps'].size > 0 :
        gt_grasps = []
        gt_grasps_keep=[]

    for i in range(num_images):
        #im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])

        boxes = None
        cls = None
        boxes_keep = None
        if 'boxes' in roidb[i]:
            boxes = np.array(roidb[i]['boxes'], dtype=np.int32)
            cls = roidb[i]['gt_classes']
            boxes_keep = np.array(range(boxes.shape[0]), dtype=np.int32)

        grasps = None
        grasps_keep = None
        # grasps should be floats
        if 'grasps' in roidb[i] and roidb[i]['grasps'].size > 0:
            grasps = np.array(roidb[i]['grasps'], dtype=np.int32)
            grasps_keep = np.array(range(grasps.shape[0]), dtype=np.int32)

        # flip the channel, since the original one using cv2
        im = np.rot90(im, roidb[i]['rotated'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im, boxes, cls, grasps, boxes_keep, grasps_keep = \
            prep_im_for_blob_aug(im, boxes, cls, grasps, boxes_keep, grasps_keep, training)

        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im,im,im), axis=2)

        # rgb -> bgr
        im = im[:, :, ::-1]

        # origion code
        if scale_inds == -1:
            target_size = cfg.TRAIN.COMMON.INPUT_SIZE
            im, im_scale = prep_im_for_blob_fixed_size(im, cfg.PIXEL_MEANS, target_size)
        else:
            target_size = cfg.TRAIN.RCNN_COMMON.SCALES[scale_inds[i]]
            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.COMMON.MAX_SIZE)

        processed_ims.append(im)
        im_scales.append(im_scale)

        if gt_boxes is not None:
            gt_boxes.append(np.array(boxes, dtype=np.int32))
            gt_classes.append(cls)
            gt_boxes_keep.append(np.array(boxes_keep,dtype=np.uint16))

        if gt_grasps is not None:
            gt_grasps.append(np.array(grasps, dtype=np.int32))
            gt_grasps_keep.append(np.array(grasps_keep,dtype=np.uint16))

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, gt_boxes, gt_classes, gt_grasps, gt_boxes_keep, gt_grasps_keep
