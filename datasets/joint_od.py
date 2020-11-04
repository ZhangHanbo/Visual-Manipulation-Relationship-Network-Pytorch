from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Copyright (c) 2020 Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------


import os
import numpy as np
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

from .imdb import imdb

class joint_od(imdb):
    def __init__(self, db_list):
        """
        :param db_list: datasets that will joint together
        """
        self.db_names = [db.name for db in db_list]
        self.db_list = db_list
        super(joint_od, self).__init__(name = '+'.join(self.db_names))

        # currently this synsets only contain classes in VMRD and COCO
        self._object_class_synsets = (["remote", "remote controller"],
                                      ["glasses", "eye glasses"],
                                      ["cell phone", "mobile phone"])
                                      # ["book", "notebook"])
        self._init_classes()
        self._init_image_index()

    def _init_classes(self):
        self._classes = list(self.db_list[0]._classes)
        self._class_to_ind = dict(self.db_list[0]._class_to_ind)
        for db in self.db_list[1:]:
            self._combine_classes(db)

    def _combine_classes(self, combined_with):
        """
        :param combined_with: a dataset that you want to combine this dataset with
        :return:
        """
        for cls in combined_with._classes:
            if cls not in self._classes:

                ismatch = False
                for cls_synset in self._object_class_synsets:
                    if cls in cls_synset:
                        for syn in cls_synset:
                            if syn in self._classes:
                                combined_with._class_to_ind[cls] = self._class_to_ind[syn]
                                ismatch = True
                                break
                        break

                if not ismatch:
                    self._classes.append(cls)
                    self._class_to_ind[cls] = len(self._classes) - 1
                    combined_with._class_to_ind[cls] = self._class_to_ind[cls]

            else:
                combined_with._class_to_ind[cls] = self._class_to_ind[cls]

    def _init_image_index(self):
        self._image_index = list(self.db_list[0]._image_index)
        self._index_seg_point = [0, len(self._image_index)]
        for db in self.db_list[1:]:
            self._image_index += db._image_index
            self._index_seg_point.append(self._index_seg_point[-1] + len(db._image_index))

    def _get_sub_gt_roidb(self, db, update = True):
        roidb = db.roidb
        if update:
            db._update_roidb()
        if not db.name.startswith('coco'):
            widths = db.widths
            heights = db.heights
            for i in range(len(db.image_index)):
                roidb[i]['width'] = widths[i]
                roidb[i]['height'] = heights[i]
        return roidb

    def gt_roidb(self):
        gt_roidb = list(self._get_sub_gt_roidb(self.db_list[0], update=False))
        for db in self.db_list[1:]:
            roidb = self._get_sub_gt_roidb(db)
            gt_roidb.extend(roidb)
        return gt_roidb

    def image_id_at(self, i):
        belong_to = 0
        for p_seg in self._index_seg_point[1:]:
            if i < p_seg:
                break
            belong_to += 1
        return self.db_list[belong_to].image_id_at(i - self._index_seg_point[belong_to])

    def image_path_at(self, i):
        belong_to = 0
        for p_seg in self._index_seg_point[1:]:
            if i < p_seg:
                break
            belong_to += 1
        return self.db_list[belong_to].image_path_at(i - self._index_seg_point[belong_to])

    def evaluate_detections(self, all_boxes, output_dir):
        maps = []
        for i, db in enumerate(self.db_list):
            seg_p_start = self._index_seg_point[i]
            seg_p_end = self._index_seg_point[i + 1]
            map = db.evaluate_detections([cls_box[seg_p_start:seg_p_end] for cls_box in all_boxes], output_dir)
            maps.append(map)
        # TODO: A more general metric should be applied here.
        return maps[0]

