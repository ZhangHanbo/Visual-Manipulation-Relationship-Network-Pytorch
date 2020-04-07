# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import torch
from model.utils.config import cfg
import numpy as np
import cv2

def points2labels(points):
    """
    :param points: bs x n x 8 point array. Each line represents a grasp
    :return: label: bs x n x 5 label array: xc, yc, w, h, Theta
    """
    batch_size = points.size(0)
    label = torch.Tensor(batch_size, points.size(1), 5).type_as(points)
    label[:, :, 0] = (points[:, :, 0] + points[:, :, 4]) / 2
    label[:, :, 1] = (points[:, :, 1] + points[:, :, 5]) / 2
    label[:, :, 2] = torch.sqrt(torch.pow((points[:, :, 2] - points[:, :, 0]), 2)
                                + torch.pow((points[:, :, 3] - points[:, :, 1]), 2))
    label[:, :, 3] = torch.sqrt(torch.pow((points[:, :, 2] - points[:, :, 4]), 2)
                                + torch.pow((points[:, :, 3] - points[:, :, 5]), 2))
    label[:, :, 4] = - torch.atan((points[:, :, 3] - points[:, :, 1]) / (points[:, :, 2] - points[:, :, 0]))
    label[:, :, 4] = label[:, :, 4] / np.pi * 180
    label[:, :, 4][label[:, :, 4] != label[:, :, 4]] = 0
    return label

def labels2points(label):
    if label.dim() == 2:
        x = label[:, 0:1]
        y = label[:, 1:2]
        w = label[:, 2:3]
        h = label[:, 3:4]
        a = label[:, 4:5]
        a = a / 180 * np.pi
    elif label.dim() == 3:
        x = label[:,:,0:1]
        y = label[:,:,1:2]
        w = label[:,:,2:3]
        h = label[:,:,3:4]
        a = label[:,:,4:5]
        a = a / 180 * np.pi
    vec1x = w/2*torch.cos(a) + h/2*torch.sin(a)
    vec1y = -w/2*torch.sin(a) + h/2*torch.cos(a)
    vec2x = w/2*torch.cos(a) - h/2*torch.sin(a)
    vec2y = -w/2*torch.sin(a) - h/2*torch.cos(a)
    return torch.cat([x + vec1x,y + vec1y, x - vec2x,y - vec2y,
        x - vec1x,y - vec1y, x + vec2x,y + vec2y,],-1)

def grasp_encode(label, ref):
    assert label.dim() == ref.dim()
    if ref.dim() == 2:
        ref_widths = ref[:, 2]
        ref_heights = ref[:, 3]
        ref_ctr_x = ref[:, 0]
        ref_ctr_y = ref[:, 1]
        ref_angle = ref[:, 4]

        gt_widths = label[:, 2]
        gt_heights = label[:, 3]
        gt_ctr_x = label[:, 0]
        gt_ctr_y = label[:, 1]
        gt_angle = label[:, 4]

    elif ref.dim() == 3:
        ref_widths = ref[:, :, 2]
        ref_heights = ref[:,:, 3]
        ref_ctr_x = ref[:, :, 0]
        ref_ctr_y = ref[:, :, 1]
        ref_angle = ref[:, :, 4]

        gt_widths = label[:, :, 2]
        gt_heights = label[:, :, 3]
        gt_ctr_x = label[:, :, 0]
        gt_ctr_y = label[:, :, 1]
        gt_angle = label[:, :, 4]
    else:
        raise ValueError('ref_roi input dimension is not correct.')

    targets_dx = (gt_ctr_x - ref_ctr_x) / ref_widths
    targets_dy = (gt_ctr_y - ref_ctr_y) / ref_heights
    targets_dw = torch.log(gt_widths / ref_widths)
    targets_dh = torch.log(gt_heights / ref_heights)
    targets_da = torch.div(gt_angle - ref_angle, cfg.TRAIN.FCGN.ANGLE_THRESH)


    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_da),-1)

    return targets

def grasp_decode(encoded_label, ref):
    assert encoded_label.dim() == ref.dim()
    if ref.dim() == 2:
        ref_widths = ref[:, 2]
        ref_heights = ref[:, 3]
        ref_ctr_x = ref[:, 0]
        ref_ctr_y = ref[:, 1]
        ref_angle = ref[:, 4]

        gt_widths = encoded_label[:, 2]
        gt_heights = encoded_label[:, 3]
        gt_ctr_x = encoded_label[:, 0]
        gt_ctr_y = encoded_label[:, 1]
        gt_angle = encoded_label[:, 4]

    elif ref.dim() == 3:
        ref_widths = ref[:, :, 2]
        ref_heights = ref[:,:, 3]
        ref_ctr_x = ref[:, :, 0]
        ref_ctr_y = ref[:, :, 1]
        ref_angle = ref[:, :, 4]

        gt_widths = encoded_label[:, :, 2]
        gt_heights = encoded_label[:, :, 3]
        gt_ctr_x = encoded_label[:, :, 0]
        gt_ctr_y = encoded_label[:, :, 1]
        gt_angle = encoded_label[:, :, 4]
    else:
        raise ValueError('ref_roi input dimension is not correct.')

    targets_dx = gt_ctr_x * ref_widths + ref_ctr_x
    targets_dy = gt_ctr_y * ref_heights + ref_ctr_y
    targets_dw = torch.exp(gt_widths) * ref_widths
    targets_dh = torch.exp(gt_heights) * ref_heights
    targets_da = gt_angle * cfg.TRAIN.FCGN.ANGLE_THRESH + ref_angle


    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_da),-1)

    return targets


def jaccard_overlap(pred, gt):
    r1 = ((pred[0], pred[1]), (pred[2], pred[3]), pred[4])
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