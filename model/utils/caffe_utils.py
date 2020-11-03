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

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.COMMON.MAX_SIZE:
            im_scale = float(cfg.TEST.COMMON.SMAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def rcnn_conv_feat_ext(net, im, n_conv = 4):
    pass

def rcnn_box_feat_ext(net, im, boxes):
    pass

def rcnn_im_detect(net, im, boxes, feat_list = ()):
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

if __name__ == '__main__':
    model_path = "/data0/svc4/code/Visual-Manipulation-Relationship-Network-Pytorch/data/pretrained_model/resnet101_faster_rcnn_final.caffemodel"
    prototxt_path = "/data0/svc4/code/Visual-Manipulation-Relationship-Network-Pytorch/data/pretrained_model/test_gt.prototxt"

    print("Loading Caffe Model: " + model_path)
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    net = load_caffemodel(prototxt_path, model_path, gpu_id=0)
    boxes = np.random.rand(10, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes *= 150
    _, _, _, _, feats = rcnn_im_detect(net, im, boxes, feat_list=("conv4", "conv5", "logit"))
    pass