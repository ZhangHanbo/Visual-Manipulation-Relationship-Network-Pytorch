"""
The data layer used during training to train a Fast R-CNN network.
Modified by Hanbo Zhang to support Visual Manipulation Relationship Network
and Visual Manipulation Relationship Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.utils.data as data

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch
from model.utils.net_utils import draw_grasp

import cv2
import os

import pdb

class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.RCNN_COMMON.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.RCNN_COMMON.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.max_num_grasp = cfg.MAX_NUM_GT_GRASPS
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

    def __getitem__(self, index):
        if cfg.TRAIN.COMMON.FIXED_INPUT_SIZE:
            return self._getitem_fixed_size(index)
        else:
            return self._getitem_unfixed_size(index)

    def _getitem_fixed_size(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes, self.training)
        data = torch.from_numpy(blobs['data'])

        data = data.squeeze(0).permute(2, 0, 1).contiguous()
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)
        if self.training:
            # grasp data
            num_grasps = 0
            gt_grasps_padding = torch.FloatTensor(self.max_num_grasp, 8).zero_()
            gt_grasp_inds_padding = torch.FloatTensor(self.max_num_grasp).zero_()

            if 'gt_grasps' in blobs:
                shuffle_inds_gr = range(blobs['gt_grasps'].shape[0])
                np.random.shuffle(shuffle_inds_gr)
                shuffle_inds_gr = torch.LongTensor(shuffle_inds_gr)

                gt_grasps = torch.from_numpy(blobs['gt_grasps'])
                gt_grasps = gt_grasps[shuffle_inds_gr]

                if 'gt_grasp_inds' in blobs:
                    gt_grasp_inds = torch.from_numpy(blobs['gt_grasp_inds'])
                    gt_grasp_inds = gt_grasp_inds[shuffle_inds_gr]

                num_grasps = min(gt_grasps.size(0), self.max_num_grasp)
                gt_grasps_padding[:num_grasps, :] = gt_grasps[:num_grasps]
                if 'gt_grasp_inds' in blobs:
                    gt_grasp_inds_padding[:num_grasps] = gt_grasp_inds[:num_grasps]

            # object detection data
            # 4 coordinates (xmin, ymin, xmax, ymax) and 1 label
            num_boxes = 0
            gt_boxes_padding = torch.FloatTensor(self.max_num_box, 5).zero_()
            rel_mat = torch.FloatTensor(self.max_num_box, self.max_num_box).zero_()

            if 'gt_boxes' in blobs:
                shuffle_inds_bb = range(blobs['gt_boxes'].shape[0])
                np.random.shuffle(shuffle_inds_bb)
                shuffle_inds_bb = torch.LongTensor(shuffle_inds_bb)

                gt_boxes = torch.from_numpy(blobs['gt_boxes'])
                gt_boxes = gt_boxes[shuffle_inds_bb]

                not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
                keep = torch.nonzero(not_keep == 0).view(-1)

                if keep.numel() != 0:
                    gt_boxes = gt_boxes[keep]
                    shuffle_inds_bb = shuffle_inds_bb[keep]

                    num_boxes = min(gt_boxes.size(0), self.max_num_box)
                    gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]

                    # get relationship matrix
                    if 'nodeinds' in blobs:
                        for o1 in range(num_boxes):
                            for o2 in range(num_boxes):
                                ind_o1 = blobs['nodeinds'][shuffle_inds_bb[o1].item()]
                                ind_o2 = blobs['nodeinds'][shuffle_inds_bb[o2].item()]
                                if ind_o2 == ind_o1 or rel_mat[o1, o2].item() != 0:
                                    continue
                                o1_children = blobs['children'][shuffle_inds_bb[o1].item()]
                                o1_fathers = blobs['fathers'][shuffle_inds_bb[o1].item()]
                                if ind_o2 in o1_children:
                                    # o1 is o2's father
                                    rel_mat[o1, o2] = cfg.VMRN.FATHER
                                elif ind_o2 in o1_fathers:
                                    # o1 is o2's child
                                    rel_mat[o1, o2] = cfg.VMRN.CHILD
                                else:
                                    # o1 and o2 has no relationship
                                    rel_mat[o1, o2] = cfg.VMRN.NOREL

            # transfer index into sequence number of boxes returned, and filter out grasps belonging to dropped boxes.
            if 'gt_grasp_inds' in blobs:
                gt_grasp_inds_padding_ori = gt_grasp_inds_padding.clone()
                order2inds = dict(enumerate(blobs['nodeinds']))
                inds2order = dict(zip(order2inds.values(), order2inds.keys()))
                shuffle2order = dict(enumerate(shuffle_inds_bb.data.numpy()))
                order2shuffle = dict(zip(shuffle2order.values(), shuffle2order.keys()))

                # make box index begins with 1
                for key in order2shuffle.keys():
                    order2shuffle[key] += 1

                for ind in blobs['nodeinds']:
                    gt_grasp_inds_padding[gt_grasp_inds_padding_ori == \
                                          float(ind)] = float(order2shuffle[inds2order[ind]])

            im_info = im_info.view(4)

            # im2show = data.clone()
            # label = gt_grasps[::10].clone()
            # print(blobs['img_id'])
            # self._show_label(im2show=im2show,gt_boxes=label, filename = os.path.basename(blobs['img_path']))

            return data, im_info, gt_boxes_padding, gt_grasps_padding, num_boxes, num_grasps, rel_mat, gt_grasp_inds_padding

        else:
            im_info = im_info.view(4)
            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            gt_grasp_inds =  torch.FloatTensor([0])
            num_boxes = 0
            num_grasps = 0
            rel_mat = torch.FloatTensor([0])

            return data, im_info, gt_boxes, gt_grasps, num_boxes, num_grasps, rel_mat, gt_grasp_inds

    def _getitem_unfixed_size(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes, self.training)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)
        if self.training:

            gt_grasps = None

            if 'gt_grasps' in blobs:
                shuffle_inds_gr = range(blobs['gt_grasps'].shape[0])
                np.random.shuffle(shuffle_inds_gr)
                shuffle_inds_gr = torch.LongTensor(shuffle_inds_gr)

                gt_grasps = torch.from_numpy(blobs['gt_grasps'])
                gt_grasps = gt_grasps[shuffle_inds_gr]

                if 'gt_grasp_inds' in blobs:
                    gt_grasps_inds = torch.from_numpy(blobs['gt_grasp_inds'])
                    gt_grasps_inds = gt_grasps_inds[shuffle_inds_gr]

            gt_boxes = None
            if 'gt_boxes' in blobs:
                shuffle_inds_bb = range(blobs['gt_boxes'].shape[0])
                np.random.shuffle(shuffle_inds_bb)
                shuffle_inds_bb = torch.LongTensor(shuffle_inds_bb)

                gt_boxes = torch.from_numpy(blobs['gt_boxes'])
                gt_boxes = gt_boxes[shuffle_inds_bb]

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################

            # NOTE1: need to cope with the case where a group cover both conditions. (done)
            # NOTE2: need to consider the situation for the tail samples. (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list_batch[index]

            if self._roidb[index_ratio]['need_crop']:
                if ratio < 1:
                    # this means that data_width << data_height, we need to crop the
                    # data_height
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
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
                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]
                    if gt_boxes is not None:
                        # shift y coordiante of gt_boxes
                        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 1::2] -= float(y_s)
                        # update gt bounding box according the trip
                        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 1::2].clamp_(0, trim_size - 1)

                    if gt_grasps is not None:
                        gt_grasps[:, 1::2] -= float(y_s)
                        keep = (((gt_grasps[:,1::2] > 0) & (gt_grasps[:,1::2] < trim_size - 1)).sum(1) == 4)
                        gt_grasps = gt_grasps[keep]
                        shuffle_inds_gr = shuffle_inds_gr[keep]
                        if 'gt_grasp_inds' in blobs:
                            gt_grasps_inds = gt_grasps_inds[keep]

                else:
                    # this means that data_width >> data_height, we need to crop the
                    # data_width
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                      trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                      x_s = 0
                    else:
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
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    if gt_boxes is not None:
                        # shift x coordiante of gt_boxes
                        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 0::2] -= float(x_s)
                        # update gt bounding box according the trip
                        gt_boxes[:, :(gt_boxes.size(1) - 1)][:, 0::2].clamp_(0, trim_size - 1)

                    if gt_grasps is not None:
                        gt_grasps[:, 0::2] -= float(x_s)
                        keep = (((gt_grasps[:,0::2] > 0) & (gt_grasps[:,1::2] < trim_size - 1)).sum(1) == 4)
                        gt_grasps = gt_grasps[keep]
                        shuffle_inds_gr = shuffle_inds_gr[keep]
                        if 'gt_grasp_inds' in blobs:
                            gt_grasps_inds = gt_grasps_inds[keep]

            # based on the ratio, padding the image.
            if ratio < 1:
                # this means that data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                               data_width, 3).zero_()

                padding_data[:data_height, :, :] = data[0]
                # update im_info
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.FloatTensor(data_height, \
                                               int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                if gt_boxes is not None:
                    # gt_boxes.clamp_(0, trim_size)
                    gt_boxes[:, :(gt_boxes.size(1) - 1)].clamp_(0, trim_size)
                if gt_grasps is not None:
                    keep = (((gt_grasps > 0) & (gt_grasps < trim_size)).sum(1) == 8)
                    gt_grasps = gt_grasps[keep]
                    shuffle_inds_gr = shuffle_inds_gr[keep]
                    if 'gt_grasp_inds' in blobs:
                        gt_grasps_inds = gt_grasps_inds[keep]
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size


            # grasp data
            num_grasps = 0
            gt_grasps_padding = torch.FloatTensor(self.max_num_grasp, 8).zero_()
            gt_grasp_inds_padding = torch.FloatTensor(self.max_num_grasp).zero_()

            if 'gt_grasps' in blobs:
                num_grasps = min(gt_grasps.size(0), self.max_num_grasp)
                gt_grasps_padding[:num_grasps, :] = gt_grasps[:num_grasps]
                if 'gt_grasp_inds' in blobs:
                    gt_grasp_inds_padding[:num_grasps] = gt_grasps_inds[:num_grasps]


            # object detection data
            # 4 coordinates (xmin, ymin, xmax, ymax) and 1 label
            num_boxes = 0
            gt_boxes_padding = torch.FloatTensor(self.max_num_box, 5).zero_()
            rel_mat = torch.FloatTensor(self.max_num_box, self.max_num_box).zero_()

            if 'gt_boxes' in blobs:
                # check the bounding box:
                not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
                keep = torch.nonzero(not_keep == 0).view(-1)

                gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
                rel_mat = torch.FloatTensor(self.max_num_box, self.max_num_box).zero_()

                if keep.numel() != 0:
                    gt_boxes = gt_boxes[keep]
                    shuffle_inds_bb = shuffle_inds_bb[keep]

                    num_boxes = min(gt_boxes.size(0), self.max_num_box)
                    gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]

                    # get relationship matrix
                    if 'nodeinds' in blobs:
                        for o1 in range(num_boxes):
                            for o2 in range(num_boxes):
                                ind_o1 = blobs['nodeinds'][shuffle_inds_bb[o1].item()]
                                ind_o2 = blobs['nodeinds'][shuffle_inds_bb[o2].item()]
                                if ind_o2 == ind_o1 or rel_mat[o1, o2].item() != 0:
                                    continue
                                o1_children = blobs['children'][shuffle_inds_bb[o1].item()]
                                o1_fathers = blobs['fathers'][shuffle_inds_bb[o1].item()]
                                if ind_o2 in o1_children:
                                    # o1 is o2's father
                                    rel_mat[o1, o2] = cfg.VMRN.FATHER
                                elif ind_o2 in o1_fathers:
                                    # o1 is o2's child
                                    rel_mat[o1, o2] = cfg.VMRN.CHILD
                                else:
                                    # o1 and o2 has no relationship
                                    rel_mat[o1, o2] = cfg.VMRN.NOREL

            # transfer index into sequence number of boxes returned, and filter out grasps belonging to dropped boxes.
            if 'gt_grasp_inds' in blobs:
                gt_grasp_inds_padding_ori = gt_grasp_inds_padding.clone()
                order2inds = dict(enumerate(blobs['nodeinds']))
                inds2order = dict(zip(order2inds.values(), order2inds.keys()))
                shuffle2order = dict(enumerate(shuffle_inds_bb.data.numpy()))
                order2shuffle = dict(zip(shuffle2order.values(), shuffle2order.keys()))

                # make box index begins with 1
                for key in order2shuffle.keys():
                    order2shuffle[key] += 1

                for ind in blobs['nodeinds']:
                    gt_grasp_inds_padding[gt_grasp_inds_padding_ori == \
                                          float(ind)] = float(order2shuffle[inds2order[ind]])

            # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(4)
            #if num_grasps < 10:
            #im2show = padding_data.clone()
            #label = gt_grasps.clone()
            #print(blobs['img_id'])
            #self._show_label(im2show=im2show, gt_boxes=label, filename=os.path.basename(blobs['img_path']))

            return padding_data, im_info, gt_boxes_padding, gt_grasps_padding, num_boxes, \
                   num_grasps, rel_mat, gt_grasp_inds_padding

        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            im_info = im_info.view(4)

            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            gt_grasps = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
            gt_grasp_inds = torch.FloatTensor([0])
            num_boxes = 0
            num_grasps = 0
            rel_mat = torch.FloatTensor([0])

            return data, im_info, gt_boxes, gt_grasps, num_boxes, num_grasps, rel_mat, gt_grasp_inds


    def _show_label(self,im2show, gt_boxes, filename = 'labelshow.png'):
        if gt_boxes.size(1) == 5:
            label = np.array(torch.cat([
                gt_boxes[:, 0:2],
                gt_boxes[:, 0:1],
                gt_boxes[:, 3:4],
                gt_boxes[:, 2:4],
                gt_boxes[:, 2:3],
                gt_boxes[:, 1:2]
            ],1))
        else:
            label = np.array(gt_boxes[:, :8])
        im2show = np.array(im2show.squeeze().permute(1, 2, 0))+cfg.PIXEL_MEANS
        cv2.imwrite("origin.png", im2show)
        im2show = draw_grasp(im2show, label)
        cv2.imwrite(filename, im2show)
        pdb.set_trace()

    def __len__(self):
        return len(self._roidb)
