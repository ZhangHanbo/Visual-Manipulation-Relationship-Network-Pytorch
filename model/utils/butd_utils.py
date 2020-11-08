# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Hanbo Zhang
# --------------------------------------------------------

"""This file provides interface for using pycaffe pretrained object detector and features"""

import caffe
import torch
import numpy as np
import cv2

from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

def load_caffemodel(prototxt_path, model_path, use_gpu = torch.cuda.is_available(), gpu_id = 0):
    if use_gpu:
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    # caffe.TEST for testing
    loaded_network = caffe.Net(prototxt_path, model_path, caffe.TEST)
    return loaded_network

def rcnn_im_detect_with_gtbox(net, im, boxes, feat_list = ()):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)
        feat_list: a list that contains feature names you need. (SUPPORT: conv1-conv5, fc, and logit)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
        attr_scores (ndarray): R x M array of attribute class scores
    """
    feat_dict = {
        "conv1": "conv1",
        "conv2": "res2c",
        "conv3": "res3b3",
        "conv4": "res4b22",
        "conv5": "res5c",
        "fc":"pool5_flat",
        "logit":"cls_score"
    }

    blobs, im_scales = _get_blobs(im, boxes)

    # Purpose: save computation resource for duplicated ROIs.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    if 'im_info' in net.blobs:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    if 'im_info' in net.blobs:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    feats = []
    if len(feat_list) > 0:
        for f in feat_list:
            feats.append(net.blobs[feat_dict[f]])

    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']

    if cfg.TEST.COMMON.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    if 'attr_prob' in net.blobs:
        attr_scores = blobs_out['attr_prob']
    else:
        attr_scores = None

    if 'rel_prob' in net.blobs:
        rel_scores = blobs_out['rel_prob']
    else:
        rel_scores = None

    return scores, pred_boxes, attr_scores, rel_scores, feats