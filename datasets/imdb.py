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
    self._num_origin = None
    self._origin_flag = True
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}

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
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
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

  def evaluate_detections(self, all_boxes, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  def _get_widths(self):
    return [PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)]

  @property
  def widths(self):
    if self._widths is not None:
      return self._widths
    self._widths = self._get_widths()
    return self._widths

  @property
  def heights(self):
    if self._heights is not None:
      return self._heights
    self._heights = self._get_heights()
    return self._heights

  @property
  def num_origin(self):
    if self._num_origin is not None:
      return self._num_origin
    assert self._origin_flag, "image index has been changed before calling num_origin."
    self._num_origin = self.num_images
    return self._num_origin

  def _get_heights(self):
    return [PIL.Image.open(self.image_path_at(i)).size[1]
            for i in range(self.num_images)]

  def append_flipped_images(self):
    num_images = self.num_images
    # this line can not be removed since you should make sure that
    # num_origin is initialized before modifying image_index
    num_origin = self.num_origin
    widths = self.widths
    heights = self.heights

    for i in range(num_images):
      entry = {'flipped': True}
      entry['rotated'] = self.roidb[i]['rotated']
      w = widths[i]
      if 'boxes' in self.roidb[i]:
        boxes = self.roidb[i]['boxes'].copy()
        boxes[:, 0::2] = w - (boxes[:, 0::2].copy())[:,::-1] - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry['boxes'] = boxes
        entry['gt_classes'] = self.roidb[i]['gt_classes']
      if 'grasps' in self.roidb[i]:
        grasps = self.roidb[i]['grasps'].copy()
        if grasps.size>0:
          grasps[:, 0::2] = w - grasps[:, 0::2].copy() - 1
        entry['grasps'] = grasps
        if 'grasp_inds' in self.roidb[i]:
          entry['grasp_inds'] = self.roidb[i]['grasp_inds']
      if 'gt_overlaps' in self.roidb[i]:
        entry['gt_overlaps'] = self.roidb[i]['gt_overlaps']
      # vmrd data entry
      if 'nodeinds' in self.roidb[i]:
        entry['nodeinds'] = self.roidb[i]['nodeinds'].copy()
      if 'fathers' in self.roidb[i]:
        entry['fathers'] = copy.deepcopy(self.roidb[i]['fathers'])
      if 'children' in self.roidb[i]:
        entry['children'] = copy.deepcopy(self.roidb[i]['children'])
      self.roidb.append(entry)
    self._image_index = self._image_index * 2
    self._origin_flag = False
    self._widths = self._widths * 2
    self._heights = self._heights * 2

  def append_rotated_images(self):
    num_images = self.num_images
    # this line can not be removed since you should make sure that
    # num_origin is initialized before modifying image_index
    num_origin = self.num_origin
    widths = self.widths
    heights = self.heights

    # rotate coordinates of bounding boxes and grasps
    def rotcoords(coords, rot, w, h, isbbox=False):
      new_coords = np.zeros(coords.shape)
      # (y, w-x)
      if rot == 1:
        new_coords[:, 0::2] = coords[:, 1::2]
        new_coords[:, 1::2] = w - coords[:, 0::2] - 1
      # (w-x, h-y)
      elif rot == 2:
        new_coords[:, 0::2] = w - coords[:, 0::2] - 1
        new_coords[:, 1::2] = h - coords[:, 1::2] - 1
      # (h-y,x)
      elif rot == 3:
        new_coords[:, 0::2] = h - coords[:, 1::2] - 1
        new_coords[:, 1::2] = coords[:, 0::2]
      if isbbox:
        new_coords = np.concatenate(
          (np.minimum(new_coords[:, 0:1], new_coords[:, 2:3]),
           np.minimum(new_coords[:, 1:2], new_coords[:, 3:4]),
           np.maximum(new_coords[:, 0:1], new_coords[:, 2:3]),
           np.maximum(new_coords[:, 1:2], new_coords[:, 3:4]))
          , axis=1)
      return new_coords
    # totally 3 rotation angles
    for r in range(1,4):
      for i in range(num_images):
        entry = {}
        assert not self.roidb[i]['flipped'], "Images should be rotated first."
        entry['flipped'] = False
        entry['rotated'] = r
        if 'boxes' in self.roidb[i]:
          boxes = self.roidb[i]['boxes'].copy()
          boxes = rotcoords(boxes, r, widths[i], heights[i], True)
          assert (boxes[:, 2] >= boxes[:, 0]).all()
          entry['boxes'] = boxes
          entry['gt_classes'] = self.roidb[i]['gt_classes']
        if 'grasps' in self.roidb[i]:
          grasps = self.roidb[i]['grasps'].copy()
          if grasps.size > 0:
            grasps = rotcoords(grasps, r, widths[i], heights[i], False)
          entry['grasps'] = grasps
          if 'grasp_inds' in self.roidb[i]:
            entry['grasp_inds'] = self.roidb[i]['grasp_inds']
        if 'gt_overlaps' in self.roidb[i]:
          entry['gt_overlaps'] = self.roidb[i]['gt_overlaps']
        # vmrd data entry
        if 'nodeinds' in self.roidb[i]:
          entry['nodeinds'] = self.roidb[i]['nodeinds'].copy()
        if 'fathers' in self.roidb[i]:
          entry['fathers'] = copy.deepcopy(self.roidb[i]['fathers'])
        if 'children' in self.roidb[i]:
          entry['children'] = copy.deepcopy(self.roidb[i]['children'])
        self.roidb.append(entry)

    self._image_index = self._image_index * 4
    self._origin_flag = False
    self._widths, self._heights = \
      (self._widths + self._heights) * 2, (self._heights + self._widths) * 2

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

  def create_roidb_from_box_list(self, box_list, gt_roidb):
    assert len(box_list) == self.num_images, \
      'Number of boxes must match number of ground-truth images'
    roidb = []
    for i in range(self.num_images):
      boxes = box_list[i]
      num_boxes = boxes.shape[0]
      overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
        gt_boxes = gt_roidb[i]['boxes']
        gt_classes = gt_roidb[i]['gt_classes']
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({
        'boxes': boxes,
        'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
      })
    return roidb

  @staticmethod
  def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
      a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                     b[i]['seg_areas']))
    return a

  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass
