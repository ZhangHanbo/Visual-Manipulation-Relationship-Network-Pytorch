# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
# from scipy.misc import imread, imresize
import cv2

from model.utils.config import cfg
import torch

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


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

    return blob

def prep_im_for_blob(im, target_size, max_size, fix_size = False):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im_shape = im.shape
    im_scale = {}
    if not fix_size:
        im_size_min = np.min(im_shape[0:2])
        im_scale['x'] = float(target_size) / float(im_size_min)
        im_scale['y'] = float(target_size) / float(im_size_min)
    else:
        im_size_y, im_size_x = im.shape[:2]
        im_scale['x'] = float(target_size) / float(im_size_x)
        im_scale['y'] = float(target_size) / float(im_size_y)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale['x'], fy=im_scale['y'],
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def image_normalize(im, mean=None, std=None):
    if mean is None:
        mean = cfg.PIXEL_MEANS if cfg.PRETRAIN_TYPE=="pytorch" else cfg.PIXEL_MEANS_CAFFE
    if std is None:
        std = cfg.PIXEL_STDS if cfg.PRETRAIN_TYPE=="pytorch" else np.array([[[1., 1., 1.]]])
    if cfg.PRETRAIN_TYPE=="pytorch":
        im /= 255.
    im  = (im - mean) / (std + 1e-8)
    return im.astype(np.float32)

def image_unnormalize(im, mean=None, std=None):
    if mean is None:
        mean = cfg.PIXEL_MEANS if cfg.PRETRAIN_TYPE=="pytorch" else cfg.PIXEL_MEANS_CAFFE
    if std is None:
        std = cfg.PIXEL_STDS if cfg.PRETRAIN_TYPE=="pytorch" else np.array([[[1., 1., 1.]]])
    im = im * (std + 1e-8) + mean
    if cfg.PRETRAIN_TYPE == "pytorch":
        im *= 255.
    return im.astype(np.float32)

def prepare_data_batch_from_cvimage(cv_img, is_cuda = True, frame = "faster_rcnn_vmrn"):
    def prepare_data_for_faster_rcnn_vmrn(cv_img, is_cuda = True):
        # BGR to RGB
        if cfg.PRETRAIN_TYPE == "pytorch":
            cv_img = cv_img[:, :, ::-1]
        image, im_scale = prep_im_for_blob(cv_img, cfg.SCALES[0], cfg.TRAIN.COMMON.MAX_SIZE)
        image = image_normalize(image)

        im_info = np.array(
            [image.shape[0], image.shape[1], im_scale['y'], im_scale['x'], -1],
            dtype=np.float32)

        data = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
        im_info = torch.from_numpy(im_info)
        gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
        num_boxes = torch.FloatTensor([0])
        rel_mat = torch.FloatTensor([0])

        data_batch = [data, im_info, gt_boxes, num_boxes, rel_mat]

        for i, d in enumerate(data_batch):
            if is_cuda:
                d = d.cuda()
            if i in {3}:
                data_batch[i] = d
            else:
                data_batch[i] = d.unsqueeze(0)
        return data_batch

    def prepare_data_for_all_in_one(cv_img, is_cuda = True):
        # BGR to RGB
        if cfg.PRETRAIN_TYPE == "pytorch":
            cv_img = cv_img[:, :, ::-1]
        image, im_scale = prep_im_for_blob(cv_img, cfg.SCALES[0], cfg.TRAIN.COMMON.MAX_SIZE)
        image = image_normalize(image)

        im_info = np.array(
            [image.shape[0], image.shape[1], im_scale['y'], im_scale['x'], -1],
            dtype=np.float32)

        data = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
        im_info = torch.from_numpy(im_info)
        gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
        gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
        gt_grasp_inds = torch.LongTensor([0])
        num_boxes = torch.FloatTensor([0])
        num_grasps = torch.FloatTensor([0])
        rel_mat = torch.FloatTensor([0])

        data_batch = [data, im_info, gt_boxes, gt_grasps, num_boxes, num_grasps, rel_mat, gt_grasp_inds]

        for i, d in enumerate(data_batch):
            if is_cuda:
                d = d.cuda()
            if i in {4, 5}:
                data_batch[i] = d
            else:
                data_batch[i] = d.unsqueeze(0)
        return data_batch

    if frame in {"faster_rcnn_vmrn"}:
        data_batch = prepare_data_for_faster_rcnn_vmrn(cv_img, is_cuda)
    elif frame in {"all_in_one"}:
        data_batch = prepare_data_for_all_in_one(cv_img, is_cuda)
    else:
        raise NotImplementedError("Data preprocessing has not been implemented for current framework.")

    return data_batch