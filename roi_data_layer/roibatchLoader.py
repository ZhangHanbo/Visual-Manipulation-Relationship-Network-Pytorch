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
import torch
import torch.utils.data as data

from model.utils.config import cfg
from roi_data_layer.minibatch import *
from model.utils.blob import prep_im_for_blob, image_normalize
import abc

import cv2
import os

from model.utils.augmentations import *

import pdb

class roibatchLoader(data.Dataset):
    __metaclass__ = abc.ABCMeta
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, cls_list=None,
                 augmentation = False):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.max_num_grasp = cfg.MAX_NUM_GT_GRASPS
        self.training = training
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)
        self.cls_list = cls_list

        self.pixel_means = cfg.PIXEL_MEANS if cfg.PRETRAIN_TYPE == "pytorch" else cfg.PIXEL_MEANS_CAFFE
        self.pixel_stds = cfg.PIXEL_STDS if cfg.PRETRAIN_TYPE == "pytorch" else np.array([[[1., 1., 1.]]])

        self.augmentation = augmentation
        if self.augmentation:
            self.augImageOnly = None
            self.augObjdet = None

    @abc.abstractmethod
    def _imagePreprocess(self, blob, fix_size):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self._roidb)

class objdetRoibatchLoader(roibatchLoader):
    __metaclass__ = abc.ABCMeta
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):

        super(objdetRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _imagePreprocess(self, blob, fix_size = True):
        keep = np.arange(blob['gt_boxes'].shape[0])
        if self.augmentation:
            if self.augImageOnly is not None: blob['data'] = self.augImageOnly(blob['data'])
            if self.augObjdet is not None: blob['data'], blob['gt_boxes'], _, _, _ = \
                self.augObjdet(image=blob['data'], boxes=blob['gt_boxes'], boxes_keep=keep)
        # choose one predefined size, TODO: support multi-instance batch
        random_scale_ind = np.random.randint(0, high=len(cfg.SCALES))
        blob['data'], im_scale = prep_im_for_blob(blob['data'], cfg.SCALES[random_scale_ind], cfg.TRAIN.COMMON.MAX_SIZE, fix_size)
        # modify bounding boxes according to resize parameters
        blob['im_info'][:2] = (blob['data'].shape[0], blob['data'].shape[1])
        blob['im_info'][2:4] = (im_scale['y'], im_scale['x'])
        blob['gt_boxes'][:, :-1][:, 0::2] *= im_scale['x']
        blob['gt_boxes'][:, :-1][:, 1::2] *= im_scale['y']
        blob['data'] = image_normalize(blob['data'], mean=self.pixel_means, std=self.pixel_stds)
        return blob

    def _boxPostProcess(self, gt_boxes):
        gt_boxes_padding = torch.FloatTensor(self.max_num_box, 5).zero_()
        not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
        keep = torch.nonzero(not_keep == 0).view(-1)
        num_boxes = min(keep.size(0), self.max_num_box)
        keep = keep[:num_boxes]
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            gt_boxes_padding[:num_boxes, :] = gt_boxes
        return gt_boxes_padding, keep

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_objdet(minibatch_db)
        # preprocess images
        blobs = self._imagePreprocess(blobs)

        data = torch.from_numpy(blobs['data'].copy())
        data = data.permute(2, 0, 1).contiguous()
        im_info = torch.from_numpy(blobs['im_info'])
        if self.training:
            # object detection data
            # 4 coordinates (xmin, ymin, xmax, ymax) and 1 label
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            gt_boxes, keep = self._boxPostProcess(gt_boxes)
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_boxes, keep.size(0)
        else:
            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0
            return data, im_info, gt_boxes, num_boxes

class graspdetRoibatchLoader(roibatchLoader):
    __metaclass__ = abc.ABCMeta
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(graspdetRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _imagePreprocess(self, blob, fix_size = True):
        keep = np.arange(blob['gt_grasps'].shape[0])
        if self.augmentation:
            if self.augImageOnly is not None: blob['data'] = self.augImageOnly(blob['data'])
            if self.augObjdet is not None: blob['data'], _, blob['gt_grasps'], _, _ = \
                self.augmGraspdet(image=blob['data'], grasps=blob['gt_grasps'], grasps_keep=keep)
        # choose one predefined size, TODO: support multi-instance batch
        random_scale_ind = np.random.randint(0, high=len(cfg.SCALES))
        blob['data'], im_scale = prep_im_for_blob(blob['data'], cfg.SCALES[random_scale_ind], cfg.TRAIN.COMMON.MAX_SIZE, fix_size)
        blob['im_info'][:2] = (blob['data'].shape[0], blob['data'].shape[1])
        blob['im_info'][2:4] = (im_scale['y'], im_scale['x'])
        blob['gt_grasps'][:, 0::2] *= im_scale['x']
        blob['gt_grasps'][:, 1::2] *= im_scale['y']
        blob['data'] = image_normalize(blob['data'], mean=self.pixel_means, std=self.pixel_stds)
        return blob

    def _graspPostProcess(self, gt_grasps, gt_grasp_inds = None):
        gt_grasps_padding = torch.FloatTensor(self.max_num_grasp, 8).zero_()
        num_grasps = min(gt_grasps.size(0), self.max_num_grasp)
        gt_grasps_padding[:num_grasps, :] = gt_grasps[:num_grasps]
        if gt_grasp_inds is not None:
            gt_grasp_inds_padding = torch.LongTensor(self.max_num_grasp).zero_()
            gt_grasp_inds_padding[:num_grasps] = gt_grasp_inds[:num_grasps]
            return gt_grasps_padding, num_grasps, gt_grasp_inds_padding
        return gt_grasps_padding, num_grasps

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_graspdet(minibatch_db)
        blobs = self._imagePreprocess(blobs)

        data = torch.from_numpy(blobs['data'].copy())
        data = data.permute(2, 0, 1).contiguous()
        im_info = torch.from_numpy(blobs['im_info'])

        if self.training:
            np.random.shuffle(blobs['gt_grasps'])
            gt_grasps = torch.from_numpy(blobs['gt_grasps'])
            gt_grasps, num_grasps = self._graspPostProcess(gt_grasps)
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_grasps, num_grasps
        else:
            gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            num_grasps = 0
            return data, im_info, gt_grasps, num_grasps

class vmrdetRoibatchLoader(objdetRoibatchLoader):
    __metaclass__ = abc.ABCMeta

    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):

        super(vmrdetRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _imagePreprocess(self, blob, fix_size=True):
        keep = np.arange(blob['gt_boxes'].shape[0])
        if self.augmentation:
            if self.augImageOnly is not None: blob['data'] = self.augImageOnly(blob['data'])
            if self.augObjdet is not None: blob['data'], blob['gt_boxes'], _, keep, _ = \
                self.augObjdet(image=blob['data'], boxes=blob['gt_boxes'], boxes_keep=keep)

        # choose one predefined size, TODO: support multi-instance batch
        random_scale_ind = np.random.randint(0, high=len(cfg.SCALES))
        blob['data'], im_scale = prep_im_for_blob(blob['data'], cfg.SCALES[random_scale_ind], cfg.TRAIN.COMMON.MAX_SIZE, fix_size)
        # modify bounding boxes according to resize parameters
        blob['im_info'][:2] = (blob['data'].shape[0], blob['data'].shape[1])
        blob['im_info'][2:4] = (im_scale['y'], im_scale['x'])
        blob['gt_boxes'][:, :-1][:, 0::2] *= im_scale['x']
        blob['gt_boxes'][:, :-1][:, 1::2] *= im_scale['y']
        blob['data'] = image_normalize(blob['data'], mean=self.pixel_means, std=self.pixel_stds)
        blob['node_inds'] = blob['node_inds'][keep]
        blob['parent_lists'] = [blob['parent_lists'][p_ind] for p_ind in list(keep)]
        blob['child_lists'] = [blob['child_lists'][c_ind] for c_ind in list(keep)]
        return blob

    def _genRelMat(self, obj_list, node_inds, child_lists, parent_lists):
        num_boxes = obj_list.size(0)
        rel_mat = torch.FloatTensor(self.max_num_box, self.max_num_box).zero_()
        # get relationship matrix
        for o1 in range(num_boxes):
            for o2 in range(num_boxes):
                ind_o1 = node_inds[obj_list[o1].item()]
                ind_o2 = node_inds[obj_list[o2].item()]
                if ind_o2 == ind_o1 or rel_mat[o1, o2].item() != 0:
                    continue
                o1_children = child_lists[obj_list[o1].item()]
                o1_fathers = parent_lists[obj_list[o1].item()]
                if ind_o2 in o1_children:
                    # o1 is o2's father
                    rel_mat[o1, o2] = cfg.VMRN.FATHER
                elif ind_o2 in o1_fathers:
                    # o1 is o2's child
                    rel_mat[o1, o2] = cfg.VMRN.CHILD
                else:
                    # o1 and o2 has no relationship
                    rel_mat[o1, o2] = cfg.VMRN.NOREL
        return rel_mat

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_vmrdet(minibatch_db)
        # preprocess images
        blobs = self._imagePreprocess(blobs)

        data = torch.from_numpy(blobs['data'].copy())
        data = data.permute(2, 0, 1).contiguous()
        im_info = torch.from_numpy(blobs['im_info'])
        if self.training:
            # object detection data
            # 4 coordinates (xmin, ymin, xmax, ymax) and 1 label
            shuffle_inds = range(blobs['gt_boxes'].shape[0])
            np.random.shuffle(shuffle_inds)
            shuffle_inds = torch.LongTensor(shuffle_inds)
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            gt_boxes = gt_boxes[shuffle_inds]
            gt_boxes, keep = self._boxPostProcess(gt_boxes)
            shuffle_inds = shuffle_inds[keep]
            rel_mat = self._genRelMat(shuffle_inds, blobs['node_inds'], blobs['child_lists'], blobs['parent_lists'])
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_boxes, keep.size(0), rel_mat
        else:
            if cfg.TRAIN.COMMON.USE_ODLOSS:
                gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
                num_boxes = 0
            else:
                gt_boxes = torch.from_numpy(blobs['gt_boxes'])
                num_boxes = gt_boxes.shape[0]
            rel_mat = torch.FloatTensor([0])
            return data, im_info, gt_boxes, num_boxes, rel_mat

class mulInSizeRoibatchLoader(roibatchLoader):
    __metaclass__ = abc.ABCMeta
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, cls_list=None, augmentation = False):
        super(mulInSizeRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                                            training, cls_list, augmentation)
        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.FloatTensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i * batch_size
            right_idx = min((i + 1) * batch_size - 1, self.data_size - 1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx + 1)] = target_ratio

    def _cropImage(self, data, gt_boxes, target_ratio):
        data_height, data_width = data.size(0), data.size(1)
        x_s, y_s = 0, 0
        if target_ratio < 1:
            # this means that data_width << data_height, we need to crop the
            # data_height
            min_y = int(torch.min(gt_boxes[:, :-1][:, 1::2]))
            max_y = int(torch.max(gt_boxes[:, :-1][:, 1::2]))
            trim_size = int(np.floor(data_width / target_ratio))
            if trim_size > data_height:
                trim_size = data_height
            box_region = max_y - min_y + 1
            if min_y > 0:
                if (box_region - trim_size) < 0:
                    y_s_min = max(max_y - trim_size, 0)
                    y_s_max = min(min_y, data_height - trim_size)
                    if y_s_min == y_s_max:
                        y_s = y_s_min
                    else:
                        y_s = np.random.choice(range(y_s_min, y_s_max))
                else:
                    y_s_add = int((box_region - trim_size) / 2)
                    if y_s_add == 0:
                        y_s = min_y
                    else:
                        y_s = np.random.choice(range(min_y, min_y + y_s_add))
            elif min_y < 0:
                raise RuntimeError
            # crop the image
            data = data[y_s:(y_s + trim_size), :, :]
        else:
            # this means that data_width >> data_height, we need to crop the
            # data_width
            min_x = int(torch.min(gt_boxes[:, :-1][:, 0::2]))
            max_x = int(torch.max(gt_boxes[:, :-1][:, 0::2]))
            trim_size = int(np.ceil(data_height * target_ratio))
            if trim_size > data_width:
                trim_size = data_width
            box_region = max_x - min_x + 1
            if min_x > 0:
                if (box_region - trim_size) < 0:
                    x_s_min = max(max_x - trim_size, 0)
                    x_s_max = min(min_x, data_width - trim_size)
                    if x_s_min == x_s_max:
                        x_s = x_s_min
                    else:
                        x_s = np.random.choice(range(x_s_min, x_s_max))
                else:
                    x_s_add = int((box_region - trim_size) / 2)
                    if x_s_add == 0:
                        x_s = min_x
                    else:
                        x_s = np.random.choice(range(min_x, min_x + x_s_add))
            elif min_x < 0:
                raise RuntimeError
            # crop the image
            data = data[:, x_s:(x_s + trim_size), :]
        return data, (x_s, y_s)

    def _paddingImage(self, data, im_info, target_ratio):
        data_height, data_width = data.size(0), data.size(1)
        if target_ratio < 1:
            # this means that data_width < data_height
            padding_data = torch.FloatTensor(int(np.ceil(data_width / target_ratio)), \
                                             data_width, 3).zero_()
            padding_data[:data_height, :, :] = data
            im_info[0] = padding_data.size(0)
        elif target_ratio > 1:
            # this means that data_width > data_height
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * target_ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data
            im_info[1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = data[:trim_size, :trim_size, :]
            im_info[0] = trim_size
            im_info[1] = trim_size

        return padding_data, im_info

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

class objdetMulInSizeRoibatchLoader(objdetRoibatchLoader, mulInSizeRoibatchLoader):
    __metaclass__ = abc.ABCMeta

    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(objdetMulInSizeRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _cropBox(self, data, coord_s, gt_boxes):
        # shift y coordiante of gt_boxes
        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 1::2] -= float(coord_s[1])
        # update gt bounding box according the trip
        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 1::2].clamp_(0, data.size(0) - 1)
        # shift x coordiante of gt_boxes
        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 0::2] -= float(coord_s[0])
        # update gt bounding box according the trip
        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 0::2].clamp_(0, data.size(1) - 1)
        return gt_boxes

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_objdet(minibatch_db)

        # preprocess images
        blobs = self._imagePreprocess(blobs, False)

        data = torch.from_numpy(blobs['data'].copy())
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(0), data.size(1)
        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])

            # if batch_size > 1, all images need to be processed to have the same size
            if self.batch_size > 1:
                ratio = self.ratio_list_batch[index]
                # if the image need to crop, crop to the target size.
                coord_s = (0, 0)
                # TODO: currently no crop is applied since the target ratio is equal to the original ratio.
                if self._roidb[index_ratio]['need_crop']:
                    data, coord_s = self._cropImage(data, gt_boxes, ratio)
                # based on the ratio, padding the image.
                data, im_info = self._paddingImage(data, im_info, ratio)
                # crpo bbox according to cropped image
                gt_boxes = self._cropBox(data, coord_s, gt_boxes)

            gt_boxes, keep = self._boxPostProcess(gt_boxes)
            # permute trim_data to adapt to downstream processing
            data = data.permute(2, 0, 1).contiguous()
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_boxes, keep.size(0)

        else:
            data = data.permute(2, 0, 1).contiguous()
            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0
            return data, im_info, gt_boxes, num_boxes

class graspMulInSizeRoibatchLoader(graspdetRoibatchLoader, mulInSizeRoibatchLoader):
    __metaclass__ = abc.ABCMeta

    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation=False):
        super(graspMulInSizeRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _cropGrasp(self, data, coord_s, gt_grasps, gt_grasp_inds = None):
        # shift y coordiante of gt_boxes
        gt_grasps[:, 1::2] -= float(coord_s[1])
        # shift x coordiante of gt_boxes
        gt_grasps[:, 0::2] -= float(coord_s[0])

        # filter out illegal grasps. TWO OPTIONS:
        # 1) filter out all grasps that have any vertices out of the range of the image.
        # keep = (((gt_grasps[:, 0::2] > 0) & (gt_grasps[:, 0::2] < data.size(1))).sum(1) == 4) & \
        #        (((gt_grasps[:, 1::2] > 0) & (gt_grasps[:, 1::2] < data.size(0))).sum(1) == 4)
        # 2) filter out all grasps whose centers are out of the range of the image.
        gc_x = gt_grasps[:, 0::2].sum(1) / 4
        gc_y = gt_grasps[:, 1::2].sum(1) / 4
        keep = (gc_x > 0) & (gc_x < data.size(1)) & (gc_y > 0)& (gc_y < data.size(0))

        gt_grasps = gt_grasps[keep]
        if gt_grasp_inds is not None:
            gt_grasp_inds = gt_grasp_inds[keep]
            return gt_grasps, keep, gt_grasp_inds
        return gt_grasps, keep

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_graspdet(minibatch_db)
        blobs = self._imagePreprocess(blobs, False)

        data = torch.from_numpy(blobs['data'].copy())
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(0), data.size(1)
        if self.training:
            np.random.shuffle(blobs['gt_grasps'])
            gt_grasps = torch.from_numpy(blobs['gt_grasps'])

            # if batch_size > 1, all images need to be processed to have the same size
            if self.batch_size > 1:
                ratio = self.ratio_list_batch[index]
                # if the image need to crop, crop to the target size.
                coord_s = (0, 0)
                if self._roidb[index_ratio]['need_crop']:
                    data, coord_s = self._cropImage(data, gt_grasps, ratio)
                # based on the ratio, padding the image.
                data, im_info = self._paddingImage(data, im_info, ratio)
                # crpo bbox according to cropped image
                gt_grasps, _ = self._cropGrasp(data, coord_s, gt_grasps)

            gt_grasps, num_grasps = self._graspPostProcess(gt_grasps)
            # permute trim_data to adapt to downstream processing
            data = data.permute(2, 0, 1).contiguous()
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_grasps, num_grasps
        else:
            data = data.permute(2, 0, 1).contiguous()
            gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            num_grasps = 0
            return data, im_info, gt_grasps, num_grasps

class vmrdetMulInSizeRoibatchLoader(vmrdetRoibatchLoader, objdetMulInSizeRoibatchLoader):
    __metaclass__ = abc.ABCMeta

    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation=False):
        super(vmrdetMulInSizeRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_vmrdet(minibatch_db)
        # preprocess images
        blobs = self._imagePreprocess(blobs, False)

        data = torch.from_numpy(blobs['data'].copy())
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(0), data.size(1)
        if self.training:
            shuffle_inds = range(blobs['gt_boxes'].shape[0])
            np.random.shuffle(shuffle_inds)
            shuffle_inds = torch.LongTensor(shuffle_inds)

            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            gt_boxes = gt_boxes[shuffle_inds]

            # if batch_size > 1, all images need to be processed to have the same size
            if self.batch_size > 1:
                ratio = self.ratio_list_batch[index]
                # if the image need to crop, crop to the target size.
                coord_s = (0, 0)
                if self._roidb[index_ratio]['need_crop']:
                    data, coord_s = self._cropImage(data, gt_boxes, ratio)
                # based on the ratio, padding the image.
                data, im_info = self._paddingImage(data, im_info, ratio)
                # crpo bbox according to cropped image
                gt_boxes = self._cropBox(data, coord_s, gt_boxes)

            gt_boxes, keep = self._boxPostProcess(gt_boxes)

            shuffle_inds = shuffle_inds[keep]
            rel_mat = self._genRelMat(shuffle_inds, blobs['node_inds'], blobs['child_lists'], blobs['parent_lists'])

            # permute trim_data to adapt to downstream processing
            data = data.permute(2, 0, 1).contiguous()
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_boxes, keep.size(0), rel_mat

        else:
            data = data.permute(2, 0, 1).contiguous()
            if cfg.TRAIN.COMMON.USE_ODLOSS:
                gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
                num_boxes = 0
            else:
                gt_boxes = torch.from_numpy(blobs['gt_boxes'])
                num_boxes = gt_boxes.shape[0]
            rel_mat = torch.FloatTensor([0])
            return data, im_info, gt_boxes, num_boxes, rel_mat

class roigdetMulInSizeRoibatchLoader(graspMulInSizeRoibatchLoader, objdetMulInSizeRoibatchLoader):
    __metaclass__ = abc.ABCMeta

    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation=False):
        super(roigdetMulInSizeRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _imagePreprocess(self, blob, fix_size = False):
        assert not fix_size, "When grasp labels are included, the input image can not be fixed-size."
        keep_b = np.arange(blob['gt_boxes'].shape[0])
        keep_g = np.arange(blob['gt_grasps'].shape[0])
        if self.augmentation:
            if self.augImageOnly is not None: blob['data'] = self.augImageOnly(blob['data'])
            if self.augObjdet is not None: blob['data'], blob['gt_boxes'], blob['gt_grasps'], keep_b, keep_g = \
                self.augObjdet(image=blob['data'], boxes=blob['gt_boxes'], grasps=blob['gt_grasps'],
                               boxes_keep=keep_b, grasps_keep=keep_g)

        # choose one predefined size, TODO: support multi-instance batch
        random_scale_ind = np.random.randint(0, high=len(cfg.SCALES))
        blob['data'], im_scale = prep_im_for_blob(blob['data'], cfg.SCALES[random_scale_ind], cfg.TRAIN.COMMON.MAX_SIZE, fix_size)
        blob['im_info'][:2] = (blob['data'].shape[0], blob['data'].shape[1])
        blob['im_info'][2:4] = (im_scale['y'], im_scale['x'])
        # modify bounding boxes according to resize parameters
        blob['gt_boxes'][:, :-1][:, 0::2] *= im_scale['x']
        blob['gt_boxes'][:, :-1][:, 1::2] *= im_scale['y']
        blob['gt_grasps'][:, 0::2] *= im_scale['x']
        blob['gt_grasps'][:, 1::2] *= im_scale['y']
        blob['node_inds'] = blob['node_inds'][keep_b]
        blob['gt_grasp_inds'] = blob['gt_grasp_inds'][keep_g]
        blob['data'] = image_normalize(blob['data'], mean=self.pixel_means, std=self.pixel_stds)
        return blob

    def _graspIndsPostProcess(self, grasp_inds, shuffle_inds, node_inds):
        grasp_inds_ori = grasp_inds.clone()
        order2inds = dict(enumerate(node_inds))
        inds2order = dict(zip(order2inds.values(), order2inds.keys()))
        shuffle2order = dict(enumerate(shuffle_inds))
        order2shuffle = dict(zip(shuffle2order.values(), shuffle2order.keys()))
        # make box index begins with 1
        for key in order2shuffle.keys():
            order2shuffle[key] += 1
        for ind in node_inds:
            grasp_inds[grasp_inds_ori == float(ind)] = float(order2shuffle[inds2order[ind]])
        return grasp_inds

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_roigdet(minibatch_db)
        blobs = self._imagePreprocess(blobs)

        data = torch.from_numpy(blobs['data'].copy())
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(0), data.size(1)
        if self.training:
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            gt_grasps = torch.from_numpy(blobs['gt_grasps'])
            gt_grasp_inds = torch.from_numpy(blobs['gt_grasp_inds'])

            # shuffle boxes
            shuffle_inds_b = range(blobs['gt_boxes'].shape[0])
            np.random.shuffle(shuffle_inds_b)
            shuffle_inds_b = torch.LongTensor(shuffle_inds_b)
            gt_boxes = gt_boxes[shuffle_inds_b]
            gt_grasp_inds = self._graspIndsPostProcess(gt_grasp_inds, shuffle_inds_b.data.numpy(), blobs['node_inds'])

            # shuffle grasps
            shuffle_inds_g = range(blobs['gt_grasps'].shape[0])
            np.random.shuffle(shuffle_inds_g)
            shuffle_inds_g = torch.LongTensor(shuffle_inds_g)
            gt_grasps = gt_grasps[shuffle_inds_g]
            gt_grasp_inds = gt_grasp_inds[shuffle_inds_g]

            # if batch_size > 1, all images need to be processed to have the same size
            if self.batch_size > 1:
                ratio = self.ratio_list_batch[index]
                # if the image need to crop, crop to the target size.
                coord_s = (0, 0)
                if self._roidb[index_ratio]['need_crop']:
                    # here image cropping is according to both gt_boxes and gt_grasps
                    data, coord_s = self._cropImage(data, torch.cat((gt_grasps, gt_boxes), dim=-1), ratio)
                # based on the ratio, padding the image.
                data, im_info = self._paddingImage(data, im_info, ratio)
                # crpo bbox according to cropped image
                gt_boxes = self._cropBox(data, coord_s, gt_boxes)
                gt_grasps, _, gt_grasp_inds = self._cropGrasp(data, coord_s, gt_grasps, gt_grasp_inds)

            gt_boxes, keep = self._boxPostProcess(gt_boxes)
            gt_grasps, num_grasps, gt_grasp_inds = self._graspPostProcess(gt_grasps, gt_grasp_inds)

            # permute trim_data to adapt to downstream processing
            data = data.permute(2, 0, 1).contiguous()
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_boxes, gt_grasps, keep.size(0), num_grasps, gt_grasp_inds
        else:
            data = data.permute(2, 0, 1).contiguous()
            if cfg.TRAIN.COMMON.USE_ODLOSS:
                gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
                num_boxes = 0
            else:
                gt_boxes = torch.from_numpy(blobs['gt_boxes'])
                num_boxes = gt_boxes.shape[0]
            gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            gt_grasp_inds = torch.LongTensor([0])
            num_grasps = 0
            return data, im_info, gt_boxes, gt_grasps, num_boxes, num_grasps, gt_grasp_inds

class allInOneMulInSizeRoibatchLoader(roigdetMulInSizeRoibatchLoader, vmrdetMulInSizeRoibatchLoader):
    __metaclass__ = abc.ABCMeta

    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation= False):
        super(allInOneMulInSizeRoibatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes, training,
                 cls_list, augmentation)

    def _imagePreprocess(self, blob, fix_size = False):
        assert not fix_size, "When grasp labels are included, the input image can not be fixed-size."
        keep_b = np.arange(blob['gt_boxes'].shape[0])
        keep_g = np.arange(blob['gt_grasps'].shape[0])
        if self.augmentation:
            blob['data'] = self.augImageOnly(blob['data'])
            blob['data'], blob['gt_boxes'], blob['gt_grasps'], keep_b, keep_g = \
                self.augObjdet(image=blob['data'], boxes=blob['gt_boxes'], grasps=blob['gt_grasps'],
                                boxes_keep=keep_b, grasps_keep=keep_g)

        # choose one predefined size, TODO: support multi-instance batch
        random_scale_ind = np.random.randint(0, high=len(cfg.SCALES))
        blob['data'], im_scale = prep_im_for_blob(blob['data'], cfg.SCALES[random_scale_ind], cfg.TRAIN.COMMON.MAX_SIZE, fix_size)
        blob['im_info'][:2] = (blob['data'].shape[0], blob['data'].shape[1])
        blob['im_info'][2:4] = (im_scale['y'], im_scale['x'])
        # modify bounding boxes according to resize parameters
        blob['gt_boxes'][:, :-1][:, 0::2] *= im_scale['x']
        blob['gt_boxes'][:, :-1][:, 1::2] *= im_scale['y']
        blob['gt_grasps'][:, 0::2] *= im_scale['x']
        blob['gt_grasps'][:, 1::2] *= im_scale['y']
        blob['gt_grasp_inds'] = blob['gt_grasp_inds'][keep_g]
        blob['data'] = image_normalize(blob['data'], mean=self.pixel_means, std=self.pixel_stds)
        blob['node_inds'] = blob['node_inds'][keep_b]
        blob['parent_lists'] = [blob['parent_lists'][p_ind] for p_ind in list(keep_b)]
        blob['child_lists'] = [blob['child_lists'][c_ind] for c_ind in list(keep_b)]
        return blob

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = self._roidb[index_ratio]
        blobs = get_minibatch_allinone(minibatch_db)
        blobs = self._imagePreprocess(blobs)

        data = torch.from_numpy(blobs['data'].copy())
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(0), data.size(1)
        if self.training:
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            gt_grasps = torch.from_numpy(blobs['gt_grasps'])
            gt_grasp_inds = torch.from_numpy(blobs['gt_grasp_inds'])

            # shuffle boxes
            shuffle_inds_b = range(blobs['gt_boxes'].shape[0])
            np.random.shuffle(shuffle_inds_b)
            shuffle_inds_b = torch.LongTensor(shuffle_inds_b)
            gt_boxes = gt_boxes[shuffle_inds_b]
            gt_grasp_inds = self._graspIndsPostProcess(gt_grasp_inds, shuffle_inds_b.data.numpy(), blobs['node_inds'])

            # shuffle grasps
            shuffle_inds_g = range(blobs['gt_grasps'].shape[0])
            np.random.shuffle(shuffle_inds_g)
            shuffle_inds_g = torch.LongTensor(shuffle_inds_g)
            gt_grasps = gt_grasps[shuffle_inds_g]
            gt_grasp_inds = gt_grasp_inds[shuffle_inds_g]

            # if batch_size > 1, all images need to be processed to have the same size
            if self.batch_size > 1:
                ratio = self.ratio_list_batch[index]
                # if the image need to crop, crop to the target size.
                coord_s = (0, 0)
                if self._roidb[index_ratio]['need_crop']:
                    # here image cropping is according to both gt_boxes and gt_grasps
                    data, coord_s = self._cropImage(data, torch.cat((gt_grasps, gt_boxes), dim=-1), ratio)
                # based on the ratio, padding the image.
                data, im_info = self._paddingImage(data, im_info, ratio)
                # crpo bbox according to cropped image
                gt_boxes = self._cropBox(data, coord_s, gt_boxes)
                gt_grasps, _, gt_grasp_inds = self._cropGrasp(data, coord_s, gt_grasps, gt_grasp_inds)

            gt_boxes, keep = self._boxPostProcess(gt_boxes)
            gt_grasps, num_grasps, gt_grasp_inds = self._graspPostProcess(gt_grasps, gt_grasp_inds)

            shuffle_inds_b = shuffle_inds_b[keep]
            rel_mat = self._genRelMat(shuffle_inds_b, blobs['node_inds'], blobs['child_lists'], blobs['parent_lists'])

            # permute trim_data to adapt to downstream processing
            data = data.permute(2, 0, 1).contiguous()
            assert data.size(1) == im_info[0] and data.size(2) == im_info[1]
            return data, im_info, gt_boxes, gt_grasps, keep.size(0), num_grasps, rel_mat, gt_grasp_inds
        else:
            data = data.permute(2, 0, 1).contiguous()
            if cfg.TRAIN.COMMON.USE_ODLOSS:
                gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
                num_boxes = 0
            else:
                gt_boxes = torch.from_numpy(blobs['gt_boxes'])
                num_boxes = gt_boxes.shape[0]
            gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            gt_grasp_inds = torch.LongTensor([0])
            num_grasps = 0
            rel_mat = torch.FloatTensor([0])
            return data, im_info, gt_boxes, gt_grasps, num_boxes, num_grasps, rel_mat, gt_grasp_inds

class ssdbatchLoader(objdetRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(ssdbatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train SSD without any augmentation.")
        else:
            self.augImageOnly = ComposeImageOnly([
                ConvertToFloats(),
                PhotometricDistort(),
            ])
            self.augObjdet = Compose([
                RandomMirror(),
                Expand(mean = self.pixel_means * 255. if cfg.PRETRAIN_TYPE == "pytorch" else self.pixel_means),
                RandomSampleCrop(),
            ])

class fcgnbatchLoader(graspdetRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(fcgnbatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train FCGN without any augmentation.")
        else:
            self.augImageOnly = ComposeImageOnly([
                ConvertToFloats(),
                PhotometricDistort(),
            ])
            self.augmGraspdet = Compose([
                RandomRotate(),
                RandomMirror(),
                RandomCropKeepBoxes(keep_shape=True),
            ])

class svmrnbatchLoader(vmrdetRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(svmrnbatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train S-VMRN without any augmentation.")
        else:
            self.augImageOnly = ComposeImageOnly([
                ConvertToFloats(),
                PhotometricDistort(),
            ])
            self.augObjdet = Compose([
                RandomMirror(),
                Expand(mean= self.pixel_means * 255. if cfg.PRETRAIN_TYPE == "pytorch" else self.pixel_means),
                # TODO: allow to damage bounding boxes while prevent deleting them when doing random crop
                RandomCropKeepBoxes(),
            ])

class fasterrcnnbatchLoader(objdetMulInSizeRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(fasterrcnnbatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train Faster-RCNN without flipped images.")
        else:
            self.augObjdet = Compose([
                RandomMirror(),
            ])

class fvmrnbatchLoader(vmrdetMulInSizeRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(fvmrnbatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train F-VMRN without any augmentation.")
        else:
            # self.augImageOnly = ComposeImageOnly([
            #     ConvertToFloats(),
            #     PhotometricDistort(),
            # ])
            self.augObjdet = Compose([
                RandomMirror(),
                # TODO: allow to damage bounding boxes while prevent deleting them when doing random crop
                # RandomCropKeepBoxes(keep_shape=True),
                # RandomCropKeepBoxes(),
                # Expand(mean=self.pixel_means * 255. if cfg.PRETRAIN_TYPE == "pytorch" else self.pixel_means, keep_size=True),
            ])


class roignbatchLoader(roigdetMulInSizeRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(roignbatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train ROI-GN without any augmentation.")
        else:
            self.augImageOnly = ComposeImageOnly([
                ConvertToFloats(),
                PhotometricDistort(),
            ])
            self.augObjdet = Compose([
                RandomMirror(),
                # TODO: allow to damage bounding boxes while prevent deleting them when doing random crop
                RandomCropKeepBoxes(keep_shape=True),
                Expand(mean = self.pixel_means * 255. if cfg.PRETRAIN_TYPE == "pytorch" else self.pixel_means, keep_size=True),
            ])

class fallinonebatchLoader(allInOneMulInSizeRoibatchLoader):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True,
                 cls_list=None, augmentation = False):
        super(fallinonebatchLoader, self).__init__(roidb, ratio_list, ratio_index, batch_size, num_classes,
                                             training, cls_list, augmentation)
        if not self.augmentation and self.training:
            warnings.warn("You are going to train ROI-GN without any augmentation.")
        else:
            self.augImageOnly = ComposeImageOnly([
                ConvertToFloats(),
                PhotometricDistort(),
            ])
            self.augObjdet = Compose([
                RandomMirror(),
                # TODO: allow to damage bounding boxes while prevent deleting them when doing random crop
                # RandomCropKeepBoxes(keep_shape=True),
                RandomCropKeepBoxes(),
                Expand(mean = self.pixel_means * 255. if cfg.PRETRAIN_TYPE == "pytorch" else self.pixel_means, keep_size=True),
            ])
