import rosapi
from rosapi import getco
from rosapi import kinect_reader
import numpy as np
import cv2
import os
from sensor_msgs.msg import PointCloud2, PointField
import rospy
from baxter_grasp.srv import *
from rosapi import GetGraspOri
import baxter_interface
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from baxter_interface import CHECK_VERSION
import datetime
import time

import _init_path
import sys
import argparse
import pprint
import pdb
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv, bbox_overlaps
from model.fully_conv_grasp.bbox_transform_grasp import labels2points, grasp_decode
from model.utils.net_utils import save_net, load_net, vis_detections, draw_grasp, draw_single_grasp, draw_single_bbox
from model.FasterRCNN import vgg16
from model.FasterRCNN import resnet
from model import SSD
from model import FPN
from model import VMRN
import model.FullyConvGrasp as FCGN
import model.MultiGrasp as MGN
import model.AllinOne as ALL_IN_ONE
import model.SSD_VMRN as SSD_VMRN

import torch.backends.cudnn as cudnn  # DA
from scipy.linalg import solve

lua_grasp_detec_file = 'demo.lua'

h_offset = 0.025
topn_grasp = 3


def calibrate_kinect(robot_coordinate):
    rgb = 'cali_crgb'
    d = 'cali_cd'
    kinect1.save_image(rgb, d)
    Img = cv2.imread('output/rob_result/kinectImg/' + rgb + '.jpg')
    Dep = cv2.imread('output/rob_result/kinectImg/' + d + '.png', cv2.IMREAD_UNCHANGED)
    image_coordinate = getco.Get_Image_Co(Img, Dep)
    print(robot_coordinate)
    print(image_coordinate)
    trans_matrix = rosapi.Calibrate(image_coordinate, robot_coordinate)
    print trans_matrix
    matfile = open('output/rob_result/' + 'trans_mat.txt', 'w+')
    for row in range(3):
        for rank in range(4):
            matfile.write(str(trans_matrix[row][rank]) + ' ')
        matfile.write('\n')


def test_calibrate(trans_matrix, kinect_reader, baxter_limb):
    img_co = getco.Get_Image_Co(kinect_reader.image_color, kinect_reader.image_depth)
    target_image = np.hstack((img_co, np.array([[1]]))).reshape(4, 1)
    # target_robot = np.array([0.591494798211500,-0.0958040978661000,-0.127865387076800]).reshape(3,1)
    pos = np.dot(trans_matrix, target_image)
    ori = [0, 1, 0, 0]
    move_limb_to_point(pos, ori, limb=baxter_limb, vel=1.0)


def grasprec_mask(grec, shape):
    # 3-D mask generater
    boundlines = np.zeros([4, 2])
    for t in range(4):
        yx1 = np.expand_dims(grec[t, :], 0)
        yx2 = np.expand_dims(grec[(t + 1) % 4, :], 0)
        yx = np.concatenate((yx1, yx2), 0)
        x = np.concatenate((np.expand_dims(yx[:, 1], 1), np.ones([2, 1])), 1)
        y = yx[:, 0]
        boundlines[t, :] = np.linalg.solve(x, y)
    mask = np.zeros(shape)
    x = np.arange(shape[1]).reshape(shape[1], 1) + 1
    # x: n x 2
    x = np.concatenate((x, np.ones((shape[1], 1))), axis=1)
    # boundlines: 4 x 2
    # bound: n x 4
    bound = np.dot(x, boundlines.T)
    # y: m x 1
    y = np.arange(shape[0]).reshape(shape[0], 1) + 1
    # totally: m x n points, therefore, totally m x n x 4 bound checking
    # mask: m x n x 4
    mask = (np.expand_dims(bound, 0) > np.expand_dims(y, 1))
    mask = (mask[:, :, 0] & mask[:, :, 2]) | (mask[:, :, 1] & mask[:, :, 3]) | \
           ((1 - mask[:, :, 0]) & (1 - mask[:, :, 2])) | ((1 - mask[:, :, 1]) & (1 - mask[:, :, 3]))
    mask = 1 - mask
    if len(shape) == 2:
        return np.expand_dims(mask, 2)
    elif len(shape) == 3:
        return mask


def kinect_grasp_depth(kg, depth):
    xmin = int(np.max([np.floor(np.min(kg[:, 1])), 1]))
    xmax = int(np.min([np.ceil(np.max(kg[:, 1])), depth.shape[1]]))
    ymin = int(np.max([np.floor(np.min(kg[:, 0])), 1]))
    ymax = int(np.min([np.ceil(np.max(kg[:, 0])), depth.shape[0]]))
    temp_kg = np.copy(kg)
    temp_kg[:, 1] = kg[:, 1] - xmin
    temp_kg[:, 0] = kg[:, 0] - ymin
    h = int(ymax - ymin)
    w = int(xmax - xmin)
    kgmask = grasprec_mask(temp_kg, (h, w))
    graspdepthpatch = depth[ymin:ymax, xmin:xmax]
    xyoffset = {'x': xmin, 'y': ymin}
    return graspdepthpatch, kgmask, xyoffset


def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg


def gen_robot_xyz(kgdepth, xyoffset, transmat):
    xoffset = xyoffset['x']
    yoffset = xyoffset['y']
    # h,w,_ = kgdepth.shape
    h, w = kgdepth.shape
    robotpc = np.zeros([w, h, 3])
    for x in range(robotpc.shape[0]):
        for y in range(robotpc.shape[1]):
            depth = kgdepth[y, x]
            xori = x + xoffset
            yori = y + yoffset
            robotxyz = (np.dot(transmat, np.array([[xori], [yori], [depth], [1]]))).squeeze()
            robotpc[x, y, :] = robotxyz
    return robotpc


def min_ignore_zero(array):
    tarray = np.copy(array)
    tarray = (tarray == 0) * 4096 + tarray
    return np.min(tarray), np.where(tarray == np.min(tarray))


def normal_estimate_client(cloud):
    rospy.wait_for_service("normal_estimation")
    try:
        normal_estimate = rospy.ServiceProxy("normal_estimation", NormEsti)
        resp1 = normal_estimate(cloud)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def image_color_client():
    rospy.wait_for_service("kinect_image_color_server")
    try:
        color_image_getter = rospy.ServiceProxy("kinect_image_color_server", ColorImg)
        start = True
        resp1 = color_image_getter(start)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def image_depth_client():
    rospy.wait_for_service("kinect_image_depth_server")
    try:
        depth_image_getter = rospy.ServiceProxy("kinect_image_depth_server", DepthImg)
        start = True
        resp1 = depth_image_getter(start)
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def normalmsg_to_array(normal_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = [(f.name, np.float32) for f in normal_msg.fields]
    print(dtype_list)
    dtype_list = []
    for name in ['n_x', 'n_y', 'n_z', 'pad_1', 'curvature', 'pad_2', 'pad_3', 'pad_4']:
        dtype_list.append((name, np.float32))
    print(dtype_list)
    # parse the cloud into an array
    cloud_arr = np.fromstring(normal_msg.data, dtype_list)
    if squeeze and normal_msg.height == 1:
        return np.reshape(cloud_arr, (normal_msg.width,))
    else:
        return np.reshape(cloud_arr, (normal_msg.height, normal_msg.width))


def fit_plane(points):
    # input: n x 3 points, np.array
    # output: 3-d surface normal, with z > 0

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    xy = np.sum(x * y)
    xz = np.sum(x * z)
    yz = np.sum(y * z)
    xx = np.sum(x * x)
    yy = np.sum(y * y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    z_sum = np.sum(z)

    n_points = points.shape[0]
    a = np.array([[xx, xy, x_sum],
                  [xy, yy, y_sum],
                  [x_sum, y_sum, n_points]])
    b = np.array([xz, yz, z_sum])
    # plane: Ax + By + C = z, surf_norm = (A, B, -1)
    ABC = solve(a, b)
    surf_norm = np.array([-ABC[0], -ABC[1], 1])
    surf_norm = surf_norm / np.linalg.norm(surf_norm)
    return surf_norm


def image_grasp_to_robot_v2(kgrec, transmat, depth):
    kgdepth, kgmask, xyoffset = kinect_grasp_depth(kgrec, depth)

    kgmask = kgmask[:, :, 0]

    mindepth, gpoints = min_ignore_zero(kgdepth * kgmask)
    print(mindepth, gpoints)
    gpoints = np.array([gpoints[0][0], gpoints[1][0]])
    grasp_patch = kgdepth[(gpoints[0] - 2): (gpoints[0] + 3),
                  (gpoints[1] - 2): (gpoints[1] + 3)]
    xyoffset['x'] += gpoints[0] - 2
    xyoffset['y'] += gpoints[1] - 2
    robotpc = gen_robot_xyz(grasp_patch, xyoffset, transmat)
    robotgvec = fit_plane(robotpc.reshape(-1, 3))
    if np.abs(robotgvec[2]) > 0:
        robotgvec = np.array([0., 0., -1.])
    # grasp rec angle
    gp1 = np.flip(np.mean(kgrec[0:2, :], 0), 0)
    gp2 = np.flip(np.mean(kgrec[2:4, :], 0), 0)
    gdiff = gp2 - gp1
    robotgang = np.arctan2(gdiff[1], -gdiff[0])
    # robot grasp point
    gcent = (gp1 + gp2) / 2
    robotgpoint = np.dot(transmat, np.array([gcent[0], gcent[1], mindepth, 1])).squeeze()
    print(robotgpoint)
    robotori = GetGraspOri(robotgvec, -robotgang)
    return robotgpoint, robotori, robotgvec


def image_grasp_to_robot(kgrec, transmat, depth):
    # grasp rec of kinect depth image
    kgdepth, kgmask, xyoffset = kinect_grasp_depth(kgrec, depth)
    # robot point cloud
    robotpc = gen_robot_xyz(kgdepth, xyoffset, transmat)
    # estimate robot surface normals of grasp rec
    robotpcmsg = xyz_array_to_pointcloud2(robotpc)
    normalmsg = normal_estimate_client(robotpcmsg)
    normalmsg = normalmsg.normals
    normals = normalmsg_to_array(normalmsg)
    # find grasp point in image coordinate
    mindepth, gpoints = min_ignore_zero(kgdepth * kgmask)
    print(mindepth, gpoints)
    # find grasp vector at grasp points
    gpoints = np.array([gpoints[0][0], gpoints[1][0]])
    robotgvec = normals[gpoints[0], gpoints[1]]
    robotgvec = np.array([robotgvec[0], robotgvec[1], robotgvec[2]])
    if robotgvec[2] < 0:
        robotgvec = -robotgvec
    if np.abs(robotgvec[2]) > 0:
        robotgvec = np.array([0., 0., -1.])
    # grasp rec angle
    gp1 = np.flip(np.mean(kgrec[0:2, :], 0), 0)
    gp2 = np.flip(np.mean(kgrec[2:4, :], 0), 0)
    gdiff = gp2 - gp1
    robotgang = np.arctan2(gdiff[1], -gdiff[0])
    # robot grasp point
    gcent = (gp1 + gp2) / 2
    robotgpoint = np.dot(transmat, np.array([gcent[0], gcent[1], mindepth, 1])).squeeze()
    print(robotgpoint)
    robotori = GetGraspOri(robotgvec, -robotgang)
    robotgvec = np.array(robotgvec)
    return robotgpoint, robotori, robotgvec


def ik_solve(limb, pos, orient):
    # ~ rospy.init_node("rsdk_ik_service_client")
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    print "iksvc: ", iksvc
    print "ikreq: ", ikreq
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        str(limb): PoseStamped(header=hdr,
                               pose=Pose(position=pos, orientation=orient))}

    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    if (resp.isValid[0]):
        print("SUCCESS - Valid Joint Solution Found:")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print limb_joints
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")
    return -1


def move_baxter_to_grasp(robotgpoint, robotori, robotgvec, baxter_limb):
    # pre-grasp position
    robotprepoint = robotgpoint - 0.2 * robotgvec
    # baxter movement
    robotgpoint = robotgpoint - np.array([0, 0, h_offset])
    move_limb_to_point(robotprepoint, robotori, limb=baxter_limb, vel=1.0)
    move_limb_to_point(robotgpoint, robotori, limb=baxter_limb, vel=1.0)


def move_limb_to_point(point, ori=None, limb='left', vel=1.0):
    limbhandle = baxter_interface.Limb(limb)
    limbhandle.set_joint_position_speed(vel)
    loc = Point(point[0], point[1], point[2])
    if ori is None:
        ori = [0, 1, 0, 0];
    ang = Quaternion(ori[0], ori[1], ori[2], ori[3])
    limbjoints = ik_solve(limb, loc, ang)
    limbhandle.move_to_joint_positions(limbjoints)


def move_limb_to_initial(initial=None, limb='left', vel=1.0):
    '''
    poselist: ['e0','e1','s0','s1','w0','w1','w2']
    default initial pose:
    [-0.9587379924283836, 1.8453788878261528, 0.3351748021529629, -1.373679795551388,
    0.14841264122791378, 1.1685098651717138, 0.03259709174256504]
    '''
    if initial is None:
        initial = {
            'left_e0': -0.9587379924283836,
            'left_e1': 1.8453788878261528,
            'left_s0': 0.3351748021529629,
            'left_s1': -1.373679795551388,
            'left_w0': 0.14841264122791378,
            'left_w1': 1.1685098651717138,
            'left_w2': 0.03259709174256504
        }

    limbhandle = baxter_interface.Limb(limb)
    limbhandle.set_joint_position_speed(vel)
    limbhandle.move_to_joint_positions(initial)


def move_limb_to_neutral(limb='left', vel=1.0):
    limbhandle = baxter_interface.Limb(limb)
    limbhandle.set_joint_position_speed(vel)
    limbhandle.move_to_neutral()


def grasp_and_put_thing_down(putposition, limb='left', vel=1.0):
    left_arm = baxter_interface.Limb('left')
    endpoints = left_arm.endpoint_pose()
    curpos = endpoints['position']
    curpos = np.array([curpos.x, curpos.y, curpos.z])
    curpos = curpos + np.array([0, 0, 0.3])
    gripper = baxter_interface.Gripper(limb, CHECK_VERSION)
    rospy.sleep(0.5)
    gripper.close()
    rospy.sleep(0.5)
    preputpos = putposition + np.array([0, 0, 0.2])
    move_limb_to_point(point=curpos, limb=limb, vel=vel)
    move_limb_to_point(point=preputpos, limb=limb, vel=vel)
    # move_limb_to_point(point = putposition, limb=limb, vel = vel)
    gripper.open()
    rospy.sleep(0.5)
    move_limb_to_point(point=preputpos, limb=limb, vel=vel)


def get_image_blob(im, size=600):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])

    target_size = size
    im_scale = float(target_size) / float(im_size_min)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


# mgn
def grasp_obj_mgn(cls, img, net):
    img = img[:, :, ::-1]
    img = img[250:700, 520:1120, :]
    imshow = img.copy()

    img, im_scale = get_image_blob(img)
    im_info_np = np.array([[img.shape[0], img.shape[1], im_scale]], dtype=np.float32)

    im_data_pt = torch.from_numpy(img)
    im_data_pt = im_data_pt.unsqueeze(0)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)

    gt = {
        'boxes': gt_boxes,
        'grasps': gt_grasps,
        'grasp_inds': gt_grasp_inds,
        'num_boxes': num_boxes,
        'num_grasps': num_grasps,
        'im_info': im_info
    }

    rois, cls_prob, bbox_pred, rpn_loss_cls, \
    rpn_loss_box, loss_cls, loss_bbox, rois_label, \
    grasp_loc, grasp_prob, grasp_bbox_loss, \
    grasp_cls_loss, grasp_conf_label, grasp_all_anchors \
        = net(im_data, gt)

    boxes = rois.data[:, :, 1:5]

    scores = cls_prob.data
    grasp_scores = grasp_prob.data

    if cfg.TEST.COMMON.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        grasp_box_deltas = grasp_loc.data
        grasp_box_deltas = grasp_box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
        grasp_box_deltas = grasp_box_deltas.view(grasp_all_anchors.size())
        # bs*N x K*A x 5
        grasp_pred = grasp_decode(grasp_box_deltas, grasp_all_anchors)
        # bs*N x K*A x 1
        rois_w = (rois[:, :, 3] - rois[:, :, 1]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 0:1])
        rois_h = (rois[:, :, 4] - rois[:, :, 2]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 1:2])
        keep_mask = (grasp_pred[:, :, 0:1] > 0) & (grasp_pred[:, :, 1:2] > 0) & \
                    (grasp_pred[:, :, 0:1] < rois_w) & (grasp_pred[:, :, 1:2] < rois_h)
        grasp_scores = (grasp_scores).contiguous(). \
            view(rois.size(0), rois.size(1), -1, 2)
        # bs*N x 1 x 1
        xleft = rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
        ytop = rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
        # rois offset
        grasp_pred[:, :, 0:1] = grasp_pred[:, :, 0:1] + xleft
        grasp_pred[:, :, 1:2] = grasp_pred[:, :, 1:2] + ytop
        # bs x N x K*A x 8
        grasp_pred_boxes = labels2points(grasp_pred).contiguous().view(rois.size(0), rois.size(1), -1, 8)
        # bs x N x K*A
        grasp_pos_scores = grasp_scores[:, :, :, 1]
        # bs x N x K*A
        _, grasp_score_idx = torch.sort(grasp_pos_scores, dim=2, descending=True)
        _, grasp_idx_rank = torch.sort(grasp_score_idx, dim=2)
        # bs x N x K*A mask
        topn_grasp = 1
        grasp_maxscore_mask = (grasp_idx_rank < topn_grasp)
        # bs x N x topN
        grasp_maxscores = grasp_scores[:, :, :, 1][grasp_maxscore_mask].contiguous(). \
            view(rois.size()[:2] + (topn_grasp,))
        # scores = scores * grasp_maxscores[:, :, 0:1]
        # bs x N x topN x 8
        grasp_pred_boxes = grasp_pred_boxes[grasp_maxscore_mask].view(rois.size()[:2] + (topn_grasp, 8))
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
            cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * 32)
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    pred_boxes[:, 0::4] /= im_scale
    pred_boxes[:, 1::4] /= im_scale
    pred_boxes[:, 2::4] /= im_scale
    pred_boxes[:, 3::4] /= im_scale
    grasp_pred_boxes = grasp_pred_boxes.squeeze()
    grasp_scores = grasp_scores.squeeze()
    if grasp_pred_boxes.dim() == 2:
        grasp_pred_boxes[:, 0::4] /= im_scale
        grasp_pred_boxes[:, 1::4] /= im_scale
        grasp_pred_boxes[:, 2::4] /= im_scale
        grasp_pred_boxes[:, 3::4] /= im_scale
    elif grasp_pred_boxes.dim() == 3:
        grasp_pred_boxes[:, :, 0::4] /= im_scale
        grasp_pred_boxes[:, :, 1::4] /= im_scale
        grasp_pred_boxes[:, :, 2::4] /= im_scale
        grasp_pred_boxes[:, :, 3::4] /= im_scale
    elif grasp_pred_boxes.dim() == 4:
        grasp_pred_boxes[:, :, :, 0::4] /= im_scale
        grasp_pred_boxes[:, :, :, 1::4] /= im_scale
        grasp_pred_boxes[:, :, :, 2::4] /= im_scale
        grasp_pred_boxes[:, :, :, 3::4] /= im_scale

    thresh = 0.05

    obj = None
    grasp = None
    inds = torch.nonzero(scores[:, cls] > thresh).view(-1)
    # if there is det
    if inds.numel() > 0:
        cls_scores = scores[:, cls][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = pred_boxes[inds][:, cls * 4:(cls + 1) * 4]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        cur_grasp = grasp_pred_boxes[inds, :]
        cur_grasp = cur_grasp[order]

        keep = nms(cls_dets, cfg.TEST.COMMON.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        cur_grasp = cur_grasp[keep.view(-1).long()]
        cls_dets = cls_dets.cpu().numpy()
        cur_grasp = cur_grasp.cpu().numpy()

        obj = cls_dets[0]
        grasp = cur_grasp[0]

        imshow = draw_single_bbox(imshow, obj.astype(np.int32), '%s:%.3f' % (clslist[cls], obj[-1]),
                                  color_dict[clslist[cls]])
        imshow = draw_single_grasp(imshow, grasp.astype(np.int32))
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite('output/rob_result/result/' + now_time + '.jpg', imshow)

        obj[:4] += (520, 250, 520, 250)
        grasp += (520, 250, 520, 250, 520, 250, 520, 250)

    return cls, grasp


def load_mgn(name, vmrd_classes):
    input_dir = 'output' + "/" + 'res101' + "/" + 'vmrdcompv1'
    load_name = os.path.join(input_dir, name)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    Network = MGN.resnet(vmrd_classes, 101, pretrained=True, class_agnostic=False)
    Network.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    Network.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    return Network


def current_target(mani_tree, targets):
    childs = []
    for i in range(targets.size(0)):
        if len(mani_tree[targets[i].item()]['child']) == 0:
            return targets[i].item()
        else:
            childs += mani_tree[targets[i].item()]['child']
    # delete repeated elements
    childs = list(set(childs))
    return current_target(mani_tree, torch.LongTensor(childs).type_as(targets))


# vmrn + fcgn:
def grasp_obj_vmrn_fcgn(cls, img, depth, vmrn, fcgn):
    img = img[:, :, ::-1]
    img = img[250:700, 600:1050, :]
    depth = depth[250:700, 600:1050, :]
    imshow = img.copy()

    #################### VMRN #########################
    img_vmrn = img.copy()
    img_vmrn, im_scale_vmrn = get_image_blob(img_vmrn, size=300)
    im_info_np_vmrn = np.array([[img_vmrn.shape[0], img_vmrn.shape[1], im_scale_vmrn, im_scale_vmrn]], dtype=np.float32)

    im_data_pt_vmrn = torch.from_numpy(img_vmrn)
    im_data_pt_vmrn = im_data_pt_vmrn.unsqueeze(0)
    im_data_pt_vmrn = im_data_pt_vmrn.permute(0, 3, 1, 2)
    im_info_pt_vmrn = torch.from_numpy(im_info_np_vmrn)

    im_data_vmrn = im_data.clone()
    im_info_vmrn = im_info.clone()
    im_data_vmrn.data.resize_(im_data_pt_vmrn.size()).copy_(im_data_pt_vmrn)
    im_info_vmrn.data.resize_(im_info_pt_vmrn.size()).copy_(im_info_pt_vmrn)

    bbox_pred, cls_prob, rel_result, \
    net_loss_bbox, net_loss_cls, rel_loss_cls = vmrn(im_data_vmrn, im_info_vmrn, gt_boxes, num_boxes, rel_mat)
    boxes = vmrn.priors.type_as(bbox_pred)
    scores = cls_prob.data
    box_deltas = bbox_pred.data
    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                 + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
    box_deltas = box_deltas.view(1, -1, 4)
    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info_vmrn.data, 1)
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    pred_boxes[:, 0::4] /= im_scale_vmrn
    pred_boxes[:, 1::4] /= im_scale_vmrn
    pred_boxes[:, 2::4] /= im_scale_vmrn
    pred_boxes[:, 3::4] /= im_scale_vmrn

    ############################ FCGN #############################

    img_fcgn = img.copy()
    img_fcgn[:, :, 2:3] = (depth.astype(np.float32) / 16).astype(np.int32)
    img_fcgn = img_fcgn[:, :, ::-1]

    img_fcgn, im_scale_fcgn = get_image_blob(img_fcgn, size=320)
    im_info_np_fcgn = np.array([[img_fcgn.shape[0], img_fcgn.shape[1], im_scale_fcgn, im_scale_fcgn]], dtype=np.float32)

    im_data_pt_fcgn = torch.from_numpy(img_fcgn)
    im_data_pt_fcgn = im_data_pt_fcgn.unsqueeze(0)
    im_data_pt_fcgn = im_data_pt_fcgn.permute(0, 3, 1, 2)
    im_info_pt_fcgn = torch.from_numpy(im_info_np_fcgn)

    im_data_fcgn = im_data.clone()
    im_info_fcgn = im_info.clone()
    im_data_fcgn.data.resize_(im_data_pt_fcgn.size()).copy_(im_data_pt_fcgn)
    im_info_fcgn.data.resize_(im_info_pt_fcgn.size()).copy_(im_info_pt_fcgn)

    bbox_pred_fcgn, cls_prob_fcgn, loss_bbox_fcgn, \
    loss_cls_fcgn, rois_label_fcgn, boxes_fcgn = \
        fcgn(im_data_fcgn, im_info_fcgn, gt_grasps, num_boxes)

    scores_fcgn = cls_prob_fcgn.data
    if cfg.TEST.COMMON.BBOX_REG:
        box_deltas_fcgn = bbox_pred_fcgn.data
        box_deltas_fcgn = box_deltas_fcgn.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                          + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
        box_deltas_fcgn = box_deltas_fcgn.view(1, -1, 5)
        pred_label_fcgn = grasp_decode(box_deltas_fcgn, boxes_fcgn)
        pred_boxes_fcgn = labels2points(pred_label_fcgn)
        imshape_fcgn = np.tile(np.array([320, 320])
                               , (int(pred_boxes_fcgn.size(1)), int(pred_boxes_fcgn.size(2) / 2)))
        imshape_fcgn = torch.from_numpy(imshape_fcgn).type_as(pred_boxes_fcgn)
        keep = (((pred_boxes_fcgn > imshape_fcgn) | (pred_boxes_fcgn < 0)).sum(2) == 0)
        pred_boxes_fcgn = pred_boxes_fcgn[keep]
        scores_fcgn = scores_fcgn[keep]

    scores_fcgn = scores_fcgn.squeeze()
    pred_boxes_fcgn = pred_boxes_fcgn.squeeze()
    pred_boxes_fcgn[:, 0::4] /= im_scale_fcgn
    pred_boxes_fcgn[:, 1::4] /= im_scale_fcgn
    pred_boxes_fcgn[:, 2::4] /= im_scale_fcgn
    pred_boxes_fcgn[:, 3::4] /= im_scale_fcgn

    thresh_fcgn = 0.2
    inds = torch.nonzero(scores_fcgn[:, 1] > thresh_fcgn).view(-1)
    if inds.numel() > 0:
        cls_scores_fcgn = scores_fcgn[:, 1][inds]
        _, order = torch.sort(cls_scores_fcgn, 0, True)
        cls_boxes_fcgn = pred_boxes_fcgn[inds, :]
        cls_dets_fcgn = torch.cat((cls_boxes_fcgn, cls_scores_fcgn.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets_fcgn = cls_dets_fcgn[order]
        # cls_dets_fcgn = cls_dets_fcgn[0][:8]
        cls_dets_fcgn = cls_dets_fcgn[:, :8]
        cls_dets_fcgn = cls_dets_fcgn.cpu().numpy()

    ########################## Reasoning ###############################

    obj_rois, obj_label, rels = rel_result
    if rels.numel() > 0:
        _, rels = torch.max(rels, dim=1)
        rels += 1

    # initialize relationship tree
    rel_num = 0
    mani_tree = {}
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    relfile = open('output/rob_result/result/' + now_time + '.txt', 'w')
    for i in range(obj_rois.size(0)):
        if i not in mani_tree.keys():
            mani_tree[i] = {}
            mani_tree[i]['child'] = []
            mani_tree[i]['parent'] = []
            mani_tree[i]['name'] = clslist[int(obj_label[i].item())]
            mani_tree[i]['bbox'] = obj_rois[i].cpu().numpy()
            mani_tree[i]['cls'] = int(obj_label[i].item())
        for ii in range(i + 1, obj_rois.size(0)):
            if ii not in mani_tree.keys():
                mani_tree[ii] = {}
                mani_tree[ii]['child'] = []
                mani_tree[ii]['parent'] = []
                mani_tree[ii]['name'] = clslist[int(obj_label[ii].item())]
                mani_tree[ii]['bbox'] = obj_rois[ii].cpu().numpy()
                mani_tree[ii]['cls'] = int(obj_label[ii].item())
            rel = rels[rel_num]
            if rel == cfg.VMRN.FATHER:
                # FATHER
                print(clslist[int(obj_label[ii].item())] + '---->' + clslist[int(obj_label[i].item())])
                relfile.write(clslist[int(obj_label[ii].item())] + '---->' + clslist[int(obj_label[i].item())] + '\n')
                mani_tree[i]['child'].append(ii)
                mani_tree[ii]['parent'].append(i)
            elif rel == cfg.VMRN.CHILD:
                # CHILD
                print(clslist[int(obj_label[i].item())] + '---->' + clslist[int(obj_label[ii].item())])
                relfile.write(clslist[int(obj_label[i].item())] + '---->' + clslist[int(obj_label[ii].item())] + '\n')
                mani_tree[i]['parent'].append(ii)
                mani_tree[ii]['child'].append(i)
            else:
                print(clslist[int(obj_label[ii].item())] + ' and ' +
                      clslist[int(obj_label[i].item())] + ' have no relationship.')
                relfile.write(clslist[int(obj_label[ii].item())] + ' and ' +
                              clslist[int(obj_label[i].item())] + ' have no relationship.' + '\n')
            rel_num += 1
    relfile.close()

    def find_grasp(obj_bbox, all_grasps):
        obj_xc = (obj_bbox[0] + obj_bbox[2]) / 2
        obj_yc = (obj_bbox[1] + obj_bbox[3]) / 2
        grasp_xc = (all_grasps[:, 0] + all_grasps[:, 4]) / 2
        grasp_yc = (all_grasps[:, 1] + all_grasps[:, 5]) / 2
        dists2 = np.power(grasp_xc - obj_xc, 2) + np.power(grasp_yc - obj_yc, 2)
        minind = np.argmin(dists2, 0)
        grasp = all_grasps[minind]
        return grasp

    obj = None
    grasp = None
    # for ele in mani_tree.values():
    #     imshow = vis_detections(imshow, ele['cls'], np.expand_dims(ele['bbox'], 0),
    #                          cfg.TEST.COMMON.OBJ_DET_THRESHOLD, np.expand_dims(ele['grasp'],0))
    if torch.sum(obj_label == cls).item() > 0:
        # the robot can see the target
        targets = torch.nonzero((obj_label == cls))
        target_ = mani_tree[current_target(mani_tree, targets)]
        grasp = find_grasp(target_['bbox'], cls_dets_fcgn)
        curcls = target_['cls']
        if curcls != cls:
            for i in range(targets.size(0)):
                target_bbox = mani_tree[targets[i].item()]['bbox']
                imshow = draw_single_bbox(imshow, target_bbox.astype(np.int32), 'tgt:%s' % (clslist[cls],),
                                          (0, 0, 0))
        obj = target_['bbox']
    else:
        # the robot cannot see the target
        for target_ in mani_tree.values():
            if len(target_['child']) == 0:
                grasp = find_grasp(target_['bbox'], cls_dets_fcgn)
                curcls = target_['cls']
                obj = target_['bbox']
                break

    if curcls != cls:
        print(curcls)
        imshow = draw_single_bbox(imshow, obj.astype(np.int32), 'crt:%s' % (clslist[curcls],),
                                  color_dict[clslist[curcls]])
    else:
        imshow = draw_single_bbox(imshow, obj.astype(np.int32), 'tgt:%s' % (clslist[curcls],),
                                  (0, 0, 0))
    imshow = draw_single_grasp(imshow, grasp.astype(np.int32))
    cv2.imwrite('output/rob_result/result/' + now_time + '.jpg', imshow)

    grasp += (600, 250, 600, 250, 600, 250, 600, 250)
    return curcls, grasp


def load_vmrn(name, vmrd_classes):
    input_dir = 'output' + "/" + 'vgg16' + "/" + 'vmrdcompv1'
    load_name = os.path.join(input_dir, name)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    Network = SSD_VMRN.vgg16(vmrd_classes, pretrained=True)
    Network.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    Network.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    return Network


# all in one
def grasp_obj_allinone(cls, img, net):
    img = img[:, :, ::-1]
    img = img[250:700, 520:1120, :]
    imshow = img.copy()

    img, im_scale = get_image_blob(img, size=800)
    im_info_np = np.array([[img.shape[0], img.shape[1], im_scale, im_scale]], dtype=np.float32)

    im_data_pt = torch.from_numpy(img)
    im_data_pt = im_data_pt.unsqueeze(0)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)

    gt = {
        'boxes': gt_boxes,
        'grasps': gt_grasps,
        'grasp_inds': gt_grasp_inds,
        'num_boxes': num_boxes,
        'num_grasps': num_grasps,
        'im_info': im_info,
        'rel_mat': rel_mat
    }

    rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_box, \
    loss_cls, loss_bbox, rel_loss_cls, rois_label, \
    grasp_loc, grasp_prob, grasp_bbox_loss, \
    grasp_cls_loss, grasp_conf_label, grasp_all_anchors \
        = net(im_data, gt)

    boxes = rois.data[:, :, 1:5]

    scores = cls_prob.data
    grasp_scores = grasp_prob.data

    if cfg.TEST.COMMON.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        grasp_box_deltas = grasp_loc.data
        grasp_box_deltas = grasp_box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
        grasp_box_deltas = grasp_box_deltas.view(grasp_all_anchors.size())
        # bs*N x K*A x 5
        grasp_pred = grasp_decode(grasp_box_deltas, grasp_all_anchors)
        # bs*N x K*A x 1
        rois_w = (rois[:, :, 3] - rois[:, :, 1]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 0:1])
        rois_h = (rois[:, :, 4] - rois[:, :, 2]).data.view(-1). \
            unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 1:2])
        keep_mask = (grasp_pred[:, :, 0:1] > 0) & (grasp_pred[:, :, 1:2] > 0) & \
                    (grasp_pred[:, :, 0:1] < rois_w) & (grasp_pred[:, :, 1:2] < rois_h)
        grasp_scores = (grasp_scores).contiguous(). \
            view(rois.size(0), rois.size(1), -1, 2)
        # bs*N x 1 x 1
        xleft = rois[:, :, 1].data.view(-1).unsqueeze(1).unsqueeze(2)
        ytop = rois[:, :, 2].data.view(-1).unsqueeze(1).unsqueeze(2)
        # rois offset
        grasp_pred[:, :, 0:1] = grasp_pred[:, :, 0:1] + xleft
        grasp_pred[:, :, 1:2] = grasp_pred[:, :, 1:2] + ytop
        # bs x N x K*A x 8
        grasp_pred_boxes = labels2points(grasp_pred).contiguous().view(rois.size(0), rois.size(1), -1, 8)
        # bs x N x K*A
        grasp_pos_scores = grasp_scores[:, :, :, 1]
        # bs x N x K*A
        _, grasp_score_idx = torch.sort(grasp_pos_scores, dim=2, descending=True)
        _, grasp_idx_rank = torch.sort(grasp_score_idx, dim=2)
        # bs x N x K*A mask
        grasp_maxscore_mask = (grasp_idx_rank < topn_grasp)
        # bs x N x topN
        grasp_maxscores = grasp_scores[:, :, :, 1][grasp_maxscore_mask].contiguous(). \
            view(rois.size()[:2] + (topn_grasp,))
        # scores = scores * grasp_maxscores[:, :, 0:1]
        # bs x N x topN x 8
        grasp_pred_boxes = grasp_pred_boxes[grasp_maxscore_mask].view(rois.size()[:2] + (topn_grasp, 8))
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
            cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * 32)
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    pred_boxes[:, 0::4] /= im_scale
    pred_boxes[:, 1::4] /= im_scale
    pred_boxes[:, 2::4] /= im_scale
    pred_boxes[:, 3::4] /= im_scale
    grasp_pred_boxes = grasp_pred_boxes.squeeze()
    grasp_scores = grasp_scores.squeeze()
    if grasp_pred_boxes.dim() == 2:
        grasp_pred_boxes[:, 0::4] /= im_scale
        grasp_pred_boxes[:, 1::4] /= im_scale
        grasp_pred_boxes[:, 2::4] /= im_scale
        grasp_pred_boxes[:, 3::4] /= im_scale
    elif grasp_pred_boxes.dim() == 3:
        grasp_pred_boxes[:, :, 0::4] /= im_scale
        grasp_pred_boxes[:, :, 1::4] /= im_scale
        grasp_pred_boxes[:, :, 2::4] /= im_scale
        grasp_pred_boxes[:, :, 3::4] /= im_scale
    elif grasp_pred_boxes.dim() == 4:
        grasp_pred_boxes[:, :, :, 0::4] /= im_scale
        grasp_pred_boxes[:, :, :, 1::4] /= im_scale
        grasp_pred_boxes[:, :, :, 2::4] /= im_scale
        grasp_pred_boxes[:, :, :, 3::4] /= im_scale

    thresh = 0.05
    all_grasps = [[]]
    for j in range(1, 32):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            cur_grasp = grasp_pred_boxes[inds]
            cur_grasp = cur_grasp[order]

            keep = nms(cls_dets, cfg.TEST.COMMON.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            cur_grasp = cur_grasp[keep.view(-1).long()]
            # cls_dets = cls_dets.cpu().numpy()
            cur_grasp = cur_grasp.cpu().numpy()
            all_grasps.append((cls_dets, cur_grasp))
            # imshow = vis_detections(imshow, clslist[j], cls_dets.cpu().numpy(),
            #                      cfg.TEST.COMMON.OBJ_DET_THRESHOLD, cur_grasp)
        else:
            all_grasps.append([])

    obj_rois, obj_label, rels = rel_result
    if rels.numel() > 0:
        _, rels = torch.max(rels, dim=1)
        rels += 1

    # initialize relationship tree
    rel_num = 0
    mani_tree = {}
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    relfile = open('output/rob_result/result/' + now_time + '.txt', 'w')
    for i in range(obj_rois.size(0)):
        if i not in mani_tree.keys():
            mani_tree[i] = {}
            mani_tree[i]['child'] = []
            mani_tree[i]['parent'] = []
            mani_tree[i]['name'] = clslist[int(obj_label[i].item())]
            mani_tree[i]['bbox'] = obj_rois[i].cpu().numpy()
            mani_tree[i]['cls'] = int(obj_label[i].item())
            ovs = bbox_overlaps(all_grasps[mani_tree[i]['cls']][0][:, :4], obj_rois[i:i + 1])
            _, argmax = torch.max(ovs, dim=0)
            argmax = argmax.item()
            mani_tree[i]['grasp'] = all_grasps[mani_tree[i]['cls']][1][argmax]

        for ii in range(i + 1, obj_rois.size(0)):
            if ii not in mani_tree.keys():
                mani_tree[ii] = {}
                mani_tree[ii]['child'] = []
                mani_tree[ii]['parent'] = []
                mani_tree[ii]['name'] = clslist[int(obj_label[ii].item())]
                mani_tree[ii]['bbox'] = obj_rois[ii].cpu().numpy()
                mani_tree[ii]['cls'] = int(obj_label[ii].item())
                ovs = bbox_overlaps(all_grasps[mani_tree[ii]['cls']][0][:, :4], obj_rois[ii:ii + 1])
                _, argmax = torch.max(ovs, dim=0)
                argmax = argmax.item()
                mani_tree[ii]['grasp'] = all_grasps[mani_tree[ii]['cls']][1][argmax]

            rel = rels[rel_num]
            if rel == cfg.VMRN.FATHER:
                # FATHER
                print(clslist[int(obj_label[ii].item())] + '---->' + clslist[int(obj_label[i].item())])
                relfile.write(clslist[int(obj_label[ii].item())] + '---->' + clslist[int(obj_label[i].item())] + '\n')
                mani_tree[i]['child'].append(ii)
                mani_tree[ii]['parent'].append(i)

            elif rel == cfg.VMRN.CHILD:
                # CHILD
                print(clslist[int(obj_label[i].item())] + '---->' + clslist[int(obj_label[ii].item())])
                relfile.write(clslist[int(obj_label[i].item())] + '---->' + clslist[int(obj_label[ii].item())] + '\n')
                mani_tree[i]['parent'].append(ii)
                mani_tree[ii]['child'].append(i)

            else:
                print(clslist[int(obj_label[ii].item())] + ' and ' +
                      clslist[int(obj_label[i].item())] + ' have no relationship.')
                relfile.write(clslist[int(obj_label[ii].item())] + ' and ' +
                              clslist[int(obj_label[i].item())] + ' have no relationship.' + '\n')

            rel_num += 1
    relfile.close()

    obj = None
    grasp = None
    # for ele in mani_tree.values():
    #     imshow = vis_detections(imshow, ele['cls'], np.expand_dims(ele['bbox'], 0),
    #                          cfg.TEST.COMMON.OBJ_DET_THRESHOLD, np.expand_dims(ele['grasp'],0))

    if torch.sum(obj_label == cls).item() > 0:
        # the robot can see the target
        targets = torch.nonzero((obj_label == cls))
        target_ = mani_tree[current_target(mani_tree, targets)]
        grasp = target_['grasp']
        if len(grasp.shape) == 2:
            obj_xc = (target_['bbox'][0] + target_['bbox'][2]) / 2
            obj_yc = (target_['bbox'][1] + target_['bbox'][3]) / 2
            grasp_xc = (target_['grasp'][:, 0] + target_['grasp'][:, 4]) / 2
            grasp_yc = (target_['grasp'][:, 1] + target_['grasp'][:, 5]) / 2
            dists2 = np.power(grasp_xc - obj_xc, 2) + np.power(grasp_yc - obj_yc, 2)
            minind = np.argmin(dists2, 0)
            grasp = grasp[minind]
        curcls = target_['cls']
        if curcls != cls:
            for i in range(targets.size(0)):
                target_bbox = mani_tree[targets[i].item()]['bbox']
                imshow = draw_single_bbox(imshow, target_bbox.astype(np.int32), 'tgt:%s' % (clslist[cls],),
                                          (0, 0, 0))
        obj = target_['bbox']
    else:
        # the robot cannot see the target
        for target_ in mani_tree.values():
            if len(target_['child']) == 0:
                grasp = target_['grasp']
                if len(grasp.shape) == 2:
                    obj_xc = (target_['bbox'][0] + target_['bbox'][2]) / 2
                    obj_yc = (target_['bbox'][1] + target_['bbox'][3]) / 2
                    grasp_xc = (target_['grasp'][:, 0] + target_['grasp'][:, 4]) / 2
                    grasp_yc = (target_['grasp'][:, 1] + target_['grasp'][:, 5]) / 2
                    dists2 = np.power(grasp_xc - obj_xc, 2) + np.power(grasp_yc - obj_yc, 2)
                    minind = np.argmin(dists2, 0)
                    grasp = grasp[minind]
                curcls = target_['cls']
                obj = target_['bbox']
                break

    if curcls != cls:
        print(curcls)
        imshow = draw_single_bbox(imshow, obj.astype(np.int32), 'crt:%s' % (clslist[curcls],),
                                  color_dict[clslist[curcls]])
    else:
        imshow = draw_single_bbox(imshow, obj.astype(np.int32), 'tgt:%s' % (clslist[curcls],),
                                  (0, 0, 0))
    imshow = draw_single_grasp(imshow, grasp.astype(np.int32))
    cv2.imwrite('output/rob_result/result/' + now_time + '.jpg', imshow)

    grasp += (520, 250, 520, 250, 520, 250, 520, 250)

    return curcls, grasp


def load_allinone(name, vmrd_classes):
    input_dir = 'output' + "/" + 'res101' + "/" + 'vmrdcompv1'
    load_name = os.path.join(input_dir, name)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    Network = ALL_IN_ONE.resnet(vmrd_classes, 101, pretrained=True, class_agnostic=False)
    Network.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    Network.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    return Network


def load_fcgn(name):
    input_dir = 'output' + "/" + 'res50' + "/" + 'cornell_rgd_iw_2'
    load_name = os.path.join(input_dir, name)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    Network = FCGN.resnet(num_layers=50, pretrained=True)
    Network.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    Network.load_state_dict(checkpoint['model'])
    print('load model successfully!')
    return Network


def grasp_obj_fcgn(img, depth, net):
    t2_b = time.time()
    # img[:,:,2] = (depth.astype(np.float32) / 16).astype(np.int32)
    img = img[:, :, ::-1]
    img = img[250:700, 600:1050, :]
    imshow = img.copy()

    img, im_scale = get_image_blob(img, size=320)
    im_info_np = np.array([[img.shape[0], img.shape[1], im_scale, im_scale]], dtype=np.float32)

    im_data_pt = torch.from_numpy(img)
    im_data_pt = im_data_pt.unsqueeze(0)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    t2_e = time.time()
    f.write("time of image preprocessing is:")
    f.write(str(t2_e - t2_b) + '\n')

    bbox_pred, cls_prob, loss_bbox, \
    loss_cls, rois_label, boxes = \
        net(im_data, im_info, gt_grasps, num_boxes)

    scores = cls_prob.data
    if cfg.TEST.COMMON.BBOX_REG:
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 5) * torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.FCGN.BBOX_NORMALIZE_STDS).cuda()
        box_deltas = box_deltas.view(1, -1, 5)
        pred_label = grasp_decode(box_deltas, boxes)
        pred_boxes = labels2points(pred_label)
        imshape = np.tile(np.array([cfg.TRAIN.COMMON.INPUT_SIZE, cfg.TRAIN.COMMON.INPUT_SIZE])
                          , (int(pred_boxes.size(1)), int(pred_boxes.size(2) / 2)))
        imshape = torch.from_numpy(imshape).type_as(pred_boxes)
        keep = (((pred_boxes > imshape) | (pred_boxes < 0)).sum(2) == 0)
        pred_boxes = pred_boxes[keep]
        scores = scores[keep]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    pred_boxes[:, 0::4] /= im_scale
    pred_boxes[:, 1::4] /= im_scale
    pred_boxes[:, 2::4] /= im_scale
    pred_boxes[:, 3::4] /= im_scale

    thresh = 0.05
    inds = torch.nonzero(scores[:, 1] > thresh).view(-1)
    if inds.numel() > 0:
        cls_scores = scores[:, 1][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = pred_boxes[inds, :]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        cls_dets = cls_dets[0][:8]
        cls_dets = cls_dets.cpu().numpy()

    imshow = draw_single_grasp(imshow, cls_dets.astype(np.int32))
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.imwrite('output/rob_result/result/' + now_time + '.jpg', imshow)

    cls_dets += (600, 250, 600, 250, 600, 250, 600, 250)

    return cls_dets


if __name__ == '__main__':

    f = open("time_delay", 'w')
    # target1
    target = 'knife'
    # mgn or
    frame = 'allinone'
    # vmrn_fcgn

    '''toothpaste pliers stapler screwdriver wrist developer tape knife glasses'''
    # class list
    clslist = ('__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch')

    color_pool = [
        (255, 0, 0),
        (255, 102, 0),
        (255, 153, 0),
        (255, 204, 0),
        (255, 255, 0),
        (204, 255, 0),
        (153, 255, 0),
        (0, 255, 51),
        (0, 255, 153),
        (0, 255, 204),
        (0, 255, 255),
        (0, 204, 255),
        (0, 153, 255),
        (0, 102, 255),
        (102, 0, 255),
        (153, 0, 255),
        (204, 0, 255),
        (255, 0, 204),
        (187, 68, 68),
        (187, 116, 68),
        (187, 140, 68),
        (187, 163, 68),
        (187, 187, 68),
        (163, 187, 68),
        (140, 187, 68),
        (68, 187, 92),
        (68, 187, 140),
        (68, 187, 163),
        (68, 187, 187),
        (68, 163, 187),
        (68, 140, 187),
        (68, 116, 187),
        (116, 68, 187),
        (140, 68, 187),
        (163, 68, 187),
        (187, 68, 163),
        (255, 119, 119),
        (255, 207, 136),
        (119, 255, 146),
        (153, 214, 255)
    ]
    np.random.shuffle(color_pool)
    color_dict = {}
    for i, clsname in enumerate(clslist):
        color_dict[clsname] = color_pool[i]

    name2cls = {}
    for i, v in enumerate(clslist):
        name2cls[v] = i

    # whether to calibrate
    is_cali = False

    # init kinect camera
    camera_cfg = {'imgtype': 'hd'}
    rospy.init_node('baxter_VMRN', anonymous=True)
    kinect1 = kinect_reader(camera_cfg)
    rospy.sleep(2)

    # init robot interface
    limb = 'left'
    rs = baxter_interface.RobotEnable()
    rs.enable()
    putposition = np.array([0.5425595053903344, 0.4993226181469783, -0.065858719706178637])

    # get calibration info
    if is_cali:
        robot_coordinate = np.array([0.39529879445498084, 0.1147669605233396, -0.07500805629102376,
                                     0.4000419325866116, -0.016605280925929253, -0.07675070149413621,
                                     0.5281275737990325, 0.115334767795245, -0.07483814997405426,
                                     0.51951092386028, -0.006144350470089932, 0.0548140176904994])
        calibrate_kinect(robot_coordinate)
        transmat = np.loadtxt('output/rob_result/trans_mat.txt')
    else:
        transmat = np.loadtxt('output/rob_result/trans_mat.txt')

    if frame == 'mgn':
        cfg_file = 'cfgs/vmrdcompv1_mgn_res101_DEMO.yml'
        cfg_from_file(cfg_file)
        net_name = 'mgn_1_16_3319_3.pth'
        network = load_mgn(net_name, clslist)
    elif frame == 'allinone':
        cfg_file = 'cfgs/vmrdcompv1_all_in_one_res101_DEMO.yml'
        cfg_from_file(cfg_file)
        net_name = 'all_in_one_1_13_1407_3.pth'
        network = load_allinone(net_name, clslist)
    elif frame == 'fcgn':
        cfg_file = 'cfgs/cornell_fcgn_res50_DA.yml'
        cfg_from_file(cfg_file)
        net_name = 'fcgn_6_22_591_0.pth'
        network = load_fcgn(net_name)
    elif frame == 'vmrn_fcgn':
        cfg_file = 'cfgs/vmrdcompv1_ssd_vmrn_fcgn_vgg16_DEMO.yml'
        cfg_from_file(cfg_file)
        net_name = 'fcgn_1_608_83.pth'
        fcgn = load_fcgn(net_name)
        net_name = 'ssd_vmrn_1_10_100_0.pth'
        vmrn = load_vmrn(net_name, clslist)

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    num_grasps = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    gt_grasps = torch.FloatTensor(1)
    # visual manipulation relationship matrix
    rel_mat = torch.FloatTensor(1)
    gt_grasp_inds = torch.LongTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    num_grasps = num_grasps.cuda()
    gt_boxes = gt_boxes.cuda()
    gt_grasps = gt_grasps.cuda()
    rel_mat = rel_mat.cuda()
    gt_grasp_inds = gt_grasp_inds.cuda()

    # make variable
    im_data = Variable(im_data, requires_grad=False)
    im_info = Variable(im_info, requires_grad=False)
    num_grasps = Variable(num_grasps, requires_grad=False)
    num_boxes = Variable(num_boxes, requires_grad=False)
    gt_boxes = Variable(gt_boxes, requires_grad=False)
    gt_grasps = Variable(gt_grasps, requires_grad=False)
    rel_mat = Variable(rel_mat, requires_grad=False)
    gt_grasp_inds = Variable(gt_grasp_inds, requires_grad=False)

    cfg.CUDA = True
    if frame == 'vmrn_fcgn':
        vmrn.cuda()
        vmrn.eval()
        fcgn.cuda()
        fcgn.eval()
    else:
        network.cuda()
        network.eval()

    # grasp loop
    while (True):
        t1_b = time.time()
        img, depth = kinect1.get_image()

        depth = depth % 4096
        t1_e = time.time()
        f.write("time of image acquisition:")
        f.write(str(t1_e - t1_b) + '\n')

        t2_b = time.time()
        if frame == 'mgn':
            obj, kgrec = grasp_obj_mgn(name2cls[target], img, network)
        elif frame == 'allinone':
            obj, kgrec = grasp_obj_allinone(name2cls[target], img, network)
        elif frame == 'fcgn':
            obj = -1
            kgrec = grasp_obj_fcgn(img, depth, network)
        elif frame == 'vmrn_fcgn':
            obj, kgrec = grasp_obj_vmrn_fcgn(name2cls[target], img, depth, vmrn, fcgn)

        t3_b = time.time()
        kgrec = np.reshape(kgrec, (-1, 2))
        # exchange x and y coord
        kgrec = kgrec[:, ::-1]
        kgrec = np.concatenate((kgrec[0:1], kgrec[1:][::-1, :]), axis=0)
        robotgpoint, robotori, robotgvec = image_grasp_to_robot_v2(kgrec, transmat, kinect1.image_depth)
        t3_e = time.time()
        f.write("time of grasp point and grasp vector computation:")
        f.write(str(t3_e - t2_b) + '\n')

        move_baxter_to_grasp(robotgpoint, robotori, robotgvec, limb)
        grasp_and_put_thing_down(putposition)
        move_limb_to_initial()
        if obj == name2cls[target]:
            print("target grasped.")
            time.sleep(5)
            break
