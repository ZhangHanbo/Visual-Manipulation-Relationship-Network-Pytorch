# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import pickle
import json

# COCO API
from pycocotools.refcoco import REFER

class refcoco(imdb):
  def __init__(self, image_set, split = 'umd', version = 'g'):
    """
    :param image_set: train, val, test
    :param split: chosen from: {'unc', 'google', 'umd'}
    :param version: chosen from: {'', '+', 'g'}
    """
    imdb.__init__(self, 'refcoco' + version + '_' + split + '_' + image_set)
    self._image_set = image_set
    self._version = version
    self._split = split
    self._data_path = osp.join(cfg.DATA_DIR, 'COCO')
    # load COCO API, classes, class <-> id mappings
    self._refCOCO = REFER(self._data_path, dataset='refcoco' + self._version, splitBy=self._split)
    cats = self._refCOCO.loadCats(self._refCOCO.getCatIds())
    self._classes = list(['__background__'] + cats)
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._class_to_coco_cat_id = dict(list(zip(cats, self._refCOCO.getCatIds())))
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self.set_proposal_method('gt')

    # image id to the directory that includes the corresponding image
    train_img_list = os.listdir(os.path.join(self._data_path, 'train2017'))
    val_img_list = os.listdir(os.path.join(self._data_path, 'val2017'))
    self._image_name_to_dir = {}
    for name in train_img_list:
        if name[-3:] == 'jpg':
            self._image_name_to_dir[name] = 'train2017'
    for name in val_img_list:
        if name[-3:] == 'jpg':
            self._image_name_to_dir[name] = 'val2017'

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = []
    for ref in self._refCOCO.Refs.values():
      print(ref['image_id'])
      if ref['split'] == self._image_set:
        image_ids.append(ref['image_id'])

    return image_ids

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = (str(index).zfill(12) + '.jpg')
    image_path = osp.join(self._data_path, self._image_name_to_dir[file_name], file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_refcoco_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_refcoco_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    im_ann = self._refCOCO.loadImgs(index)[0]
    width = im_ann['width']
    height = im_ann['height']

    annIds = self._refCOCO.getAnnIds(image_ids=index)
    objs = self._refCOCO.loadAnns(annIds)
    refIds = self._refCOCO.getRefIds(image_ids=index)
    refs = self._refCOCO.loadRefs(refIds)
    for ref in refs:
      assert ref['ann_id'] in annIds
    refannIds = [ref['ann_id'] for ref in refs]

    # Sanitize bboxes -- some are invalid
    valid_objs = []
    caps = []
    for obj in objs:
      x1 = np.max((0, obj['bbox'][0]))
      y1 = np.max((0, obj['bbox'][1]))
      x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
      y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
      if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
        # if there is a corresponding caption for obj
        if obj['id'] in refannIds:
          ref = self._refCOCO.annToRef[obj['id']]['sentences']
          randSel = np.random.randint(len(ref))
          ref = ref[randSel]
          caps.append(ref['tokens'])
        # if there is no caption for obj, an empty string is used.
        else:
          caps.append([])
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.int32)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Lookup table to map from COCO category ids to our internal class
    # indices
    coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                      self._class_to_ind[cls])
                                     for cls in self._classes[1:]])

    for ix, obj in enumerate(objs):
      cls = coco_cat_id_to_class_ind[obj['category_id']]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      if obj['iscrowd']:
        # Set overlap to -1 for all classes for crowd objects
        # so they will be excluded during training
        overlaps[ix, :] = -1.0
      else:
        overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'rotated': 0,
            'captions': caps}

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls in self.classes:
      cls_ind = self._class_to_ind[cls]
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()
    return ap_default

  def _do_detection_eval(self, res_file, output_dir):
    ann_type = 'bbox'
    coco_dt = self._refCOCO.loadRes(res_file)
    coco_eval = COCOeval(self._refCOCO, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    map = self._print_detection_eval_metrics(coco_eval)
    eval_file = osp.join(output_dir, 'detection_results.pkl')
    with open(eval_file, 'wb') as fid:
      pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
    print('Wrote COCO eval results to: {}'.format(eval_file))
    return map

  def _coco_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      results.extend(
        [{'image_id': index,
          'category_id': cat_id,
          'bbox': [xs[k], ys[k], ws[k], hs[k]],
          'score': scores[k]} for k in range(dets.shape[0])])
    return results

  def _write_coco_results_file(self, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls in self.classes:
      cls_ind = self._class_to_ind[cls]
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                       self.num_classes - 1))
      coco_cat_id = self._class_to_coco_cat_id[cls]
      results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                     coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def evaluate_detections(self, all_boxes, output_dir):
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._year +
                                     '_results'))

    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      map = self._do_detection_eval(res_file, output_dir)

    return map
