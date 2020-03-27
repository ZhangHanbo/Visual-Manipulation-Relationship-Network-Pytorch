from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------


import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import pickle
from .imdb import imdb
import cv2

import pdb

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class jacquard(imdb):
    def __init__(self, image_set, version = 'rgb', testfold = 5, devkit_path=None):
        imdb.__init__(self, 'jacquard_' + version + '_' + image_set + '_' + str(testfold))

        if image_set == 'test':
            self._image_set = 'trainval_' + str(testfold)

        if image_set == 'trainval':
            self._image_set = []
            for i in range(1,6):
                if i != testfold:
                    self._image_set.append('trainval_' + str(i))

        # rgb rgd or depth
        self._version = version

        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        # Example Cornell/origin
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'obj')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._devkit_path), \
            'Cornell path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        For our Jacquard dataset, the index is the absolute path of the cooresponding file.
        """
        if self._version == 'rgb':
            image_path = index + '_RGB.png'
        elif self._version == 'rgd':
            image_path = index + '_RGD_.png'
        elif self._version == 'depth':
            image_path = index + '_Depth.png'
        else:
            raise NotImplemented

        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /Cornell/ImageSets/test.txt
        if isinstance(self._image_set,list):
            image_index = []
            for file in self._image_set:
                image_set_file = os.path.join(self._data_path, file + '.txt')
                assert os.path.exists(image_set_file), \
                    'Path does not exist: {}'.format(image_set_file)
                with open(image_set_file) as f:
                    image_index += [x.strip() for x in f.readlines()]
        else:
            image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'Jacquard')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_label_file(self, filename):
        label_file = open(filename, 'r')
        labels = label_file.readlines()
        label_list = []
        for label in labels:
            label = label.split(';')
            label[4] = label[4][:-2]
            label[2], label[3], label[4] = label[3], label[4], label[2]
            label_list += [label]
        return np.array(label_list).astype(np.float32)

    def _labels_to_points(self, labels):
        x = labels[:, 0:1]
        y = labels[:, 1:2]
        w = labels[:, 2:3]
        h = labels[:, 3:4]
        a = -labels[:, 4:5]
        a = a / 180 * np.pi
        vec1x = w / 2 * np.cos(a) + h / 2 * np.sin(a)
        vec1y = -w / 2 * np.sin(a) + h / 2 * np.cos(a)
        vec2x = w / 2 * np.cos(a) - h / 2 * np.sin(a)
        vec2y = -w / 2 * np.sin(a) - h / 2 * np.cos(a)
        return np.concatenate([x + vec1x, y + vec1y, x - vec2x, y - vec2y,
                          x - vec1x, y - vec1y, x + vec2x, y + vec2y], 1)

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        pos_filename = index + '_grasps.txt'
        grasps = self._load_label_file(pos_filename)
        grasps = self._labels_to_points(grasps)
        # load segmentation
        seg_name = index + '_mask.png'
        seg = cv2.imread(seg_name)
        seg = np.where(seg > 0)
        xmin = np.min(seg[1])
        xmax = np.max(seg[1])
        ymin = np.min(seg[0])
        ymax = np.max(seg[0])
        # object bounding boxes from segmentaion
        boxes = np.array([[xmin, ymin, xmax, ymax]], dtype = np.int32)


        # self._show_label(index, grasps)
        return {'grasps': grasps,
                'boxes': boxes,
                'rotated': 0}

    def evaluate_detections(self, all_boxes, output_dir=None):
        print('-----------------------------------------------------')
        print('Computing results of Jacquard Grasp Detection.')
        print('-----------------------------------------------------')
        # all_boxes[0] indicates backgrounds which in this case is empty.
        grasps = all_boxes[1]
        # 5 angle thresholds, 4 jaccard thretholds
        rights = np.zeros((5, 4))
        right = 0.
        total = 0.
        # 5 angle thresholds, 4 jaccard thretholds
        accs = np.zeros((5, 4))
        failed_list = []
        for im_ind, index in enumerate(self.image_index):
            total += 1
            det_result = grasps[im_ind]
            if det_result.size != 0:
                anno = self._load_annotation(index)
                anno = anno['grasps']
                det_result = self.points2label(det_result)
                anno = self.points2label(anno)
                ovs = []
                for i in range(anno.shape[0]):
                    ovs.append(self._jaccard_overlap(det_result[0], anno[i]))
                ovs = np.array(ovs)

                for i1, angth in enumerate(range(10, 35, 5)):
                    for i2,jacth in enumerate(range(20, 40, 5)):
                        jacth = float(jacth) / 100.0
                        keep = (ovs > jacth)
                        angdiff = np.abs((det_result[:, 4] - anno[:, 4])[keep]) % 180
                        if angdiff.size > 0 and ((angdiff < angth) | (angdiff > 180 - angth)).sum() > 0:
                            rights[i1,i2] += 1

                keep = (ovs > cfg.TEST.FCGN.JACCARD_OVERLAP_THRESH)
                angdiff = np.abs((det_result[:,4] - anno[:,4])[keep]) % 180
                if angdiff.size > 0 and ((angdiff < 30) | (angdiff > 150)).sum() > 0:
                    right += 1
                else:
                    failed_list.append(im_ind)
            else:
                failed_list.append(im_ind)
        wrong = total - right
        acc = right / total
        accs = rights / total
        print('x axis: Jaccard Thresholds: 20%, 25%, 30%, 35%')
        print('y axis: Angle Thresholds: 10, 15, 20, 25, 30')
        print(accs)
        print('Failed List:', failed_list)
        print('Right/Wrong/Total:{:d}/{:d}/{:d}'.format(int(right),int(wrong),int(total)))
        print('Final Accuracy:\t%.4f' %  acc)
        return acc


    def _jaccard_overlap(self, pred, gt):
        r1 = ((pred[0], pred[1]),(pred[2], pred[3]), pred[4])
        area_r1 = pred[2] * pred[3]
        r2 = ((gt[0], gt[1]), (gt[2], gt[3]), gt[4])
        area_r2 = gt[2] * gt[3]
        int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
        if int_pts is not None:
            order_pts = cv2.convexHull(int_pts, returnPoints=True)
            int_area = cv2.contourArea(order_pts)
            ovr = int_area * 1.0 / (area_r1 + area_r2 - int_area)
            return ovr
        else:
            return 0

    def points2label(self, points):
        """
        :param points: bs x n x 8 point array. Each line represents a grasp
        :return: label: bs x n x 5 label array: xc, yc, w, h, Theta
        """
        if points.shape[1] < 8:
            pdb.set_trace()
        label = np.zeros((points.shape[0], 5))
        label[:, 0] = (points[:, 0] + points[:, 4]) / 2
        label[:, 1] = (points[:, 1] + points[:, 5]) / 2
        label[:, 2] = np.sqrt(np.power((points[:, 2] - points[:, 0]), 2)
                                    + np.power((points[:, 3] - points[:, 1]), 2))
        label[:, 3] = np.sqrt(np.power((points[:, 2] - points[:, 4]), 2)
                                    + np.power((points[:, 3] - points[:, 5]), 2))
        label[:, 4] = np.arctan((points[:, 3] - points[:, 1]) / (points[:, 2] - points[:, 0]))
        label[:, 4] = label[:, 4] / np.pi * 180
        return label


if __name__ == '__main__':
    d = jacquard('trainval', testfold=5)
    res = d.roidb
    print(res)
