# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

# Here we use cv2 instead of PIL.Image because in very rare cases, the results
# of cv2.imread and PIL.Image.open are different, which will cause errors later.
import cv2
import PIL

from model.utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from model.utils.config import cfg
import pdb
import copy

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
  """Image database."""

  def __init__(self, name, classes=None):
    self._name = name
    self._num_classes = 0
    if not classes:
      self._classes = []
    else:
      self._classes = classes
    self._image_index = []
    self._obj_proposer = 'gt'
    self._roidb = None
    self._widths = None
    self._heights = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}
    self._class_to_ind = {}

    self.set_proposal_method(cfg.TRAIN.COMMON.PROPOSAL_METHOD)

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
    return self._image_index

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries for data loading.
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def image_path_at(self, i):
    raise NotImplementedError

  def image_id_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def evaluate_detections(self, all_boxes, output_dir):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  def _get_widths_and_heights(self):
    ws = []
    hs = []
    print("Initialize image widths and heights...")
    for i in range(self.num_images):
      im = PIL.Image.open(self.image_path_at(i))
      ws.append(im.size[0])
      hs.append(im.size[1])
    return ws, hs

  @property
  def widths(self):
    if self._widths is not None:
      return self._widths
    self._widths, self._heights = self._get_widths_and_heights()
    return self._widths

  @property
  def heights(self):
    if self._heights is not None:
      return self._heights
    self._widths, self._heights = self._get_widths_and_heights()
    return self._heights

  def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                      area='all', limit=None):
    """Evaluate detection proposal recall metrics.

    Returns:
        results: dictionary of results with keys
            'ar': average recall
            'recalls': vector recalls at each IoU overlap threshold
            'thresholds': vector of IoU overlap thresholds
            'gt_overlaps': vector of all ground-truth overlaps
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
             '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                   [0 ** 2, 32 ** 2],  # small
                   [32 ** 2, 96 ** 2],  # medium
                   [96 ** 2, 1e5 ** 2],  # large
                   [96 ** 2, 128 ** 2],  # 96-128
                   [128 ** 2, 256 ** 2],  # 128-256
                   [256 ** 2, 512 ** 2],  # 256-512
                   [512 ** 2, 1e5 ** 2],  # 512-inf
                   ]
    assert area in areas, 'unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for i in range(self.num_images):
      # Checking for max_overlaps == 1 avoids including crowd annotations
      # (...pretty hacking :/)
      max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
      gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                         (max_gt_overlaps == 1))[0]
      gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
      gt_areas = self.roidb[i]['seg_areas'][gt_inds]
      valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                               (gt_areas <= area_range[1]))[0]
      gt_boxes = gt_boxes[valid_gt_inds, :]
      num_pos += len(valid_gt_inds)

      if candidate_boxes is None:
        # If candidate_boxes is not supplied, the default is to use the
        # non-ground-truth boxes from this roidb
        non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
        boxes = self.roidb[i]['boxes'][non_gt_inds, :]
      else:
        boxes = candidate_boxes[i]
      if boxes.shape[0] == 0:
        continue
      if limit is not None and boxes.shape[0] > limit:
        boxes = boxes[:limit, :]

      overlaps = bbox_overlaps(boxes.astype(np.float),
                               gt_boxes.astype(np.float))

      _gt_overlaps = np.zeros((gt_boxes.shape[0]))
      for j in range(gt_boxes.shape[0]):
        # find which proposal box maximally covers each gt box
        argmax_overlaps = overlaps.argmax(axis=0)
        # and get the iou amount of coverage for each gt box
        max_overlaps = overlaps.max(axis=0)
        # find which gt box is 'best' covered (i.e. 'best' = most iou)
        gt_ind = max_overlaps.argmax()
        gt_ovr = max_overlaps.max()
        assert (gt_ovr >= 0)
        # find the proposal box that covers the best covered gt box
        box_ind = argmax_overlaps[gt_ind]
        # record the iou coverage of this gt box
        _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        assert (_gt_overlaps[j] == gt_ovr)
        # mark the proposal box and the gt box as used
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1
      # append recorded iou coverage level
      gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
      step = 0.05
      thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
      recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps}

  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass

  def _update_roidb(self):
    for r in self._roidb:
      for i in range(r['gt_classes'].shape[0]):
        cls_name = self._classes[r['gt_classes'][i]]
        r['gt_classes'][i] = self._class_to_ind[cls_name]
