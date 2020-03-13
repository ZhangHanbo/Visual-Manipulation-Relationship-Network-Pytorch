# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
import cv2

def get_minibatch_objdet(roidb):
    box_dim = roidb['boxes'].shape[1]

    # To support object-agnostic dataset that do not include object-specific classes (e.g. jacquard),
    # We set -1 label for these boxes.
    if 'gt_classes' not in roidb:
        roidb['gt_classes'] = -np.ones(roidb['boxes'].shape[0], dtype=np.int32)

    im_blob = _get_image_blob(roidb)

    # deal with object bbox
    if cfg.TRAIN.COMMON.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb['gt_classes'] != 0) & np.all(roidb['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), box_dim + 1), dtype=np.float32)
    gt_boxes[:, :box_dim] = roidb['boxes'][gt_inds, :]
    gt_boxes[:, box_dim] = roidb['gt_classes'][gt_inds]

    blobs = {}
    blobs['data'] = im_blob
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], 1., 1.],
        dtype=np.float32)
    blobs['img_id'] = roidb['img_id']
    blobs['img_path'] = roidb['image']
    return blobs

def get_minibatch_vmrdet(roidb):
    blobs = get_minibatch_objdet(roidb)
    # TODO: deal with the situation that some objects are filtered out (like in COCO, the ones that are ''iscrowd'')
    blobs['node_inds'] = roidb['node_inds']
    blobs['parent_lists'] = roidb['parent_lists']
    blobs['child_lists'] = roidb['child_lists']
    assert(blobs['gt_boxes'].shape[0] == blobs['node_inds'].shape[0])
    return blobs

def get_minibatch_graspdet(roidb):
    num_images = len(roidb)
    assert (cfg.TRAIN.RCNN_COMMON.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.TRAIN.RCNN_COMMON.BATCH_SIZE)

    im_blob = _get_image_blob(roidb)
    gt_grasps = roidb['grasps'].astype(np.float32)

    blobs = {}
    blobs['data'] = im_blob
    blobs['gt_grasps'] = gt_grasps
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], 1., 1.],
        dtype=np.float32)
    blobs['img_id'] = roidb['img_id']
    blobs['img_path'] = roidb['image']
    return blobs

def get_minibatch_roigdet(roidb):
    blobs = get_minibatch_objdet(roidb)
    # TODO: deal with the situation that some objects are filtered out (like in COCO, the ones that are ''iscrowd'')
    blobs['gt_grasps'] = roidb['grasps'].astype(np.float32)
    blobs['gt_grasp_inds'] = roidb['grasp_inds']
    blobs['node_inds'] = roidb['node_inds']
    return blobs

def get_minibatch_allinone(roidb):
    blobs = get_minibatch_vmrdet(roidb)
    blobs['gt_grasps'] = roidb['grasps'].astype(np.float32)
    blobs['gt_grasp_inds'] = roidb['grasp_inds']
    return blobs

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    im = imread(roidb['image'])

    if len(im.shape) == 2:
        im = im[:,:,np.newaxis]
        im = np.concatenate((im,im,im), axis=2)

    im = im.astype(np.float32, copy=False)
    return im
