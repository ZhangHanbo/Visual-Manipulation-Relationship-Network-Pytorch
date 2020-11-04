# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# based on code from Jiasen Lu, Jianwei Yang, Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
import cv2
import warnings

def get_minibatch_objdet(roidb):
    box_dim = roidb['boxes'].shape[1]

    # To support object-agnostic dataset that do not include object-specific classes (e.g. jacquard and cornell),
    # We set 1 label for these boxes.
    if 'gt_classes' not in roidb:
        roidb['gt_classes'] = np.ones(roidb['boxes'].shape[0], dtype=np.int32)

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
        [im_blob.shape[0], im_blob.shape[1], 1., 1., roidb['img_id']],
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
    im_blob = _get_image_blob(roidb)
    gt_grasps = roidb['grasps'].astype(np.float32)

    blobs = {}
    blobs['data'] = im_blob
    blobs['gt_grasps'] = gt_grasps
    blobs['im_info'] = np.array(
        [im_blob.shape[0], im_blob.shape[1], 1., 1., roidb['img_id']],
        dtype=np.float32)
    blobs['img_id'] = roidb['img_id']
    blobs['img_path'] = roidb['image']
    return blobs

def get_minibatch_roigdet(roidb):
    blobs = get_minibatch_objdet(roidb)
    # for cornell and jacquard, the roidb does not contain grasp_inds. But it is obvious that all grasps belonging
    # to the only object.
    if 'grasp_inds' not in roidb:
        roidb['grasp_inds'] = np.ones(roidb['grasps'].shape[0], dtype=np.float32)
        roidb['node_inds'] = np.ones(1, dtype=np.float32)
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

    # remember: cv2.imread will load picture in the order of BGR
    im = cv2.imread(roidb['image'])
    im = np.rot90(im, roidb['rotated'])

    def check_and_modify_image(im, roidb):
        # For some images, the size of PIL.Image.open does not match that of cv2.imread.
        # For now, this is just an expedient but not the perfect solution.
        w_r = roidb["width"]
        h_r = roidb["height"]
        if w_r == im.shape[0] and h_r == im.shape[1] and w_r != h_r:
            warnings.warn("The size of PIL.Image.open does not match that of cv2.imread. "
                          "Rotating the image by 90 degrees clockwise. Image: " + roidb["image"])
            im = np.rot90(im, 3)
        assert w_r == im.shape[1] and h_r == im.shape[0]
        return im

    im = check_and_modify_image(im, roidb)

    if len(im.shape) == 2:
        im = im[:,:,np.newaxis]
        im = np.concatenate((im,im,im), axis=2)

    # BGR to RGB
    if cfg.PRETRAIN_TYPE == "pytorch":
        im = im[:, :, ::-1]

    im = im.astype(np.float32, copy=False)
    return im
