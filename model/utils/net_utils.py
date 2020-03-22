import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

import numpy as np
import torchvision.models as models
from model.utils.config import cfg
from model.roi_crop.functions.roi_crop import RoICropFunction
import cv2
import pdb
import random
import time

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.fcgn.bbox_transform_grasp import labels2points, grasp_decode
from model.nms.nms_wrapper import nms

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
                m.bias.data.zero_()

def weights_xavier_init(model):
    def xavier(param):
        init.xavier_uniform(param)

    if isinstance(model, list):
        for m in model:
            weights_xavier_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm.item())

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def save_checkpoint(state, filename):
    torch.save(state, filename)

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from 
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1       ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.RCNN_COMMON.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE)))
      bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W)\
                                                                .contiguous().view(-1, D, H, W)
      crops = F.grid_sample(bottom, grid)
    
    return crops, grid

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([\
      (y2 - y1) / (height - 1),
      zero,
      (y1 + y2 - height + 1) / (height - 1),
      zero,
      (x2 - x1) / (width - 1),
      (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta

def compare_grid_sample():
    # do gradcheck
    N = random.randint(1, 8)
    C = 2 # random.randint(1, 8)
    H = 5 # random.randint(1, 8)
    W = 4 # random.randint(1, 8)
    input = Variable(torch.randn(N, C, H, W).cuda(), requires_grad=True)
    input_p = input.clone().data.contiguous()
   
    grid = Variable(torch.randn(N, H, W, 2).cuda(), requires_grad=True)
    grid_clone = grid.clone().contiguous()

    out_offcial = F.grid_sample(input, grid)    
    grad_outputs = Variable(torch.rand(out_offcial.size()).cuda())
    grad_outputs_clone = grad_outputs.clone().contiguous()
    grad_inputs = torch.autograd.grad(out_offcial, (input, grid), grad_outputs.contiguous())
    grad_input_off = grad_inputs[0]


    crf = RoICropFunction()
    grid_yx = torch.stack([grid_clone.data[:,:,:,1], grid_clone.data[:,:,:,0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()

def box_unnorm_torch(box, normalizer, d_box = 4, class_agnostic=True, n_cls = None):
    mean = normalizer['mean']
    std = normalizer['std']
    assert len(mean) == len(std) == d_box
    if box.dim() == 2:
        if class_agnostic:
            box = box * torch.FloatTensor(std).type_as(box) + torch.FloatTensor(mean).type_as(box)
        else:
            box = box.view(-1, d_box) * torch.FloatTensor(std).type_as(box) + torch.FloatTensor(mean).type_as(box)
            box = box.view(-1, d_box * n_cls)
    elif box.dim() == 3:
        batch_size = box.size(0)
        if class_agnostic:
            box = box.view(-1, d_box) * torch.FloatTensor(std).type_as(box) + torch.FloatTensor(mean).type_as(box)
            box = box.view(batch_size, -1, d_box)
        else:
            box = box.view(-1, d_box) * torch.FloatTensor(std).type_as(box) + torch.FloatTensor(mean).type_as(box)
            box = box.view(batch_size, -1, d_box * n_cls)
    return box

def box_recover_scale_torch(box, x_scaler, y_scaler):
    if box.dim() == 2:
        box[:, 0::2] /= x_scaler
        box[:, 1::2] /= y_scaler
    elif box.dim() == 3:
        box[:, :, 0::2] /= x_scaler
        box[:, :, 1::2] /= y_scaler
    elif box.dim() == 4:
        box[:, :, :, 0::2] /= x_scaler
        box[:, :, :, 1::2] /= y_scaler
    return box

def box_filter(box, box_scores, thresh, use_nms = True):
    """
    :param box: N x d_box
    :param box_scores: N scores
    :param thresh:
    :param use_nms:
    :return:
    """
    d_box = box.size(-1)
    inds = torch.nonzero(box_scores > thresh).view(-1)
    if inds.numel() > 0:
        cls_scores = box_scores[inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = box[inds, :]
        if use_nms:
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.COMMON.NMS)
            cls_scores = cls_dets[keep.view(-1).long()][:, -1]
            cls_dets = cls_dets[keep.view(-1).long()][:, :-1]
            order = order[keep.view(-1).long()]
        else:
            cls_scores = cls_scores[order]
            cls_dets = cls_boxes[order]
        cls_dets = cls_dets.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        order = order.cpu().numpy()
    else:
        cls_scores = np.zeros(shape=(0,), dtype=np.float32)
        cls_dets = np.zeros(shape=(0, d_box), dtype=np.float32)
        order = np.array([], dtype=np.int32)
    return cls_dets, cls_scores, (inds.cpu().numpy())[order]

def objdet_inference(cls_prob, box_output, im_info, box_prior = None, class_agnostic = True, n_classes = None, for_vis = False):
    assert box_output.dim() == 2, "Multi-instance batch inference has not been implemented."

    if for_vis:
        thresh = cfg.TEST.COMMON.OBJ_DET_THRESHOLD
    else:
        thresh = 0.

    scores = cls_prob

    # TODO: Inference for anchor free algorithms has not been implemented.
    if box_prior is None:
        raise NotImplementedError("Inference for anchor free algorithms has not been implemented.")
    
    if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        normalizer = {'mean': cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS, 'std': cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS}
        box_output = box_unnorm_torch(box_output, normalizer, 4, class_agnostic, n_classes)
    else:
        raise RuntimeError("BBOX_NORMALIZE_TARGETS_PRECOMPUTED is forced to be True in our version.")

    pred_boxes = bbox_transform_inv(box_prior, box_output, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info, 1)

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    pred_boxes = box_recover_scale_torch(pred_boxes, im_info[3], im_info[2])

    all_box = [[]]
    for j in xrange(1, n_classes):
        if class_agnostic:
            cls_boxes = pred_boxes
        else:
            cls_boxes = pred_boxes[:, j * 4:(j + 1) * 4]
        cls_dets, cls_scores, _ = box_filter(cls_boxes, scores[:, j], thresh, use_nms = True)
        cls_dets = np.concatenate((cls_dets, np.expand_dims(cls_scores, -1)), axis = -1)
        if for_vis:
            cls_dets[:, -1] = j
        all_box.append(cls_dets)
    if for_vis:
        return np.concatenate(all_box[1:], axis = 0)
    return all_box

def grasp_inference(cls_prob, box_output, im_info, box_prior = None, topN = False):
    assert box_output.dim() == 2, "Multi-instance batch inference has not been implemented."
    if not topN:
        thresh = 0.5
    else:
        thresh = 0

    # TODO: Inference for anchor free algorithms has not been implemented.
    if box_prior is None:
        raise NotImplementedError("Inference for anchor free algorithms has not been implemented.")

    scores = cls_prob
    normalizer = {'mean': cfg.FCGN.BBOX_NORMALIZE_MEANS, 'std': cfg.FCGN.BBOX_NORMALIZE_STDS}
    box_output = box_unnorm_torch(box_output, normalizer, d_box=5, class_agnostic=True, n_cls=None)

    pred_label = grasp_decode(box_output, box_prior)
    pred_boxes = labels2points(pred_label)

    imshape = np.tile(np.array([im_info[1], im_info[0]]),
                      pred_boxes.shape[:-2] + (int(pred_boxes.size(-2)), int(pred_boxes.size(-1) / 2)))
    imshape = torch.from_numpy(imshape).type_as(pred_boxes)
    keep = (((pred_boxes > imshape) | (pred_boxes < 0)).sum(-1) == 0)
    pred_boxes = pred_boxes[keep]
    scores = scores[keep]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    pred_boxes = box_recover_scale_torch(pred_boxes, im_info[3], im_info[2])

    grasps, scores, _ = box_filter(pred_boxes, scores[:, 1], thresh, use_nms = False)
    grasps = np.concatenate((grasps, np.expand_dims(scores, -1)), axis=-1)
    if topN:
        grasps = grasps[:topN]
    return grasps

def objgrasp_inference(o_cls_prob, o_box_output, g_cls_prob, g_box_output, im_info, rois = None,
                       class_agnostic = True, n_classes = None, g_box_prior = None, for_vis = False, topN_g = False):
    """
    :param o_cls_prob: N x N_cls tensor
    :param o_box_output: N x 4 tensor
    :param g_cls_prob: N x K*A x 2 tensor
    :param g_box_output: N x K*A x 5 tensor
    :param im_info: size 4 tensor
    :param rois: N x 4 tensor
    :param g_box_prior: N x K*A * 5 tensor
    :return:

    Note:
    1 This function simultaneously supports ROI-GN with or without object branch. If no object branch, o_cls_prob
    and o_box_output will be none, and object detection results are shown in the form of ROIs.
    2 This function can only detect one image per invoking.
    """
    o_scores = o_cls_prob
    rois = rois[:, 1:5]

    g_scores = g_cls_prob

    if for_vis:
        o_thresh = 0.5
    else:
        o_thresh = 0.
        topN_g = 1

    if not topN_g:
        g_thresh = 0.5
    else:
        g_thresh = 0.

    if rois is None:
        raise RuntimeError("You must specify rois for ROI-GN.")

    if g_box_prior is None:
        raise NotImplementedError("Inference for anchor free algorithms has not been implemented.")

    # infer grasp boxes
    normalizer = {'mean': cfg.FCGN.BBOX_NORMALIZE_MEANS, 'std': cfg.FCGN.BBOX_NORMALIZE_STDS}
    g_box_output = box_unnorm_torch(g_box_output, normalizer, d_box=5, class_agnostic=True, n_cls=None)
    g_box_output = g_box_output.view(g_box_prior.size())
    # N x K*A x 5
    grasp_pred = grasp_decode(g_box_output, g_box_prior)

    # N x K*A x 1
    rois_w = (rois[:, 2] - rois[:, 0]).view(-1).unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 0:1])
    rois_h = (rois[:, 3] - rois[:, 1]).view(-1).unsqueeze(1).unsqueeze(2).expand_as(grasp_pred[:, :, 1:2])
    keep_mask = (grasp_pred[:, :, 0:1] > 0) & (grasp_pred[:, :, 1:2] > 0) & \
                (grasp_pred[:, :, 0:1] < rois_w) & (grasp_pred[:, :, 1:2] < rois_h)
    grasp_scores = g_scores.contiguous().view(rois.size(0), -1, 2)
    # N x 1 x 1
    xleft = rois[:, 0].view(-1).unsqueeze(1).unsqueeze(2)
    ytop = rois[:, 1].view(-1).unsqueeze(1).unsqueeze(2)
    # rois offset
    grasp_pred[:, :, 0:1] = grasp_pred[:, :, 0:1] + xleft
    grasp_pred[:, :, 1:2] = grasp_pred[:, :, 1:2] + ytop
    # N x K*A x 8
    grasp_pred_boxes = labels2points(grasp_pred).contiguous().view(rois.size(0), -1, 8)
    # N x K*A
    grasp_pos_scores = grasp_scores[:, :, 1]
    if topN_g:
        # N x K*A
        _, grasp_score_idx = torch.sort(grasp_pos_scores, dim=-1, descending=True)
        _, grasp_idx_rank = torch.sort(grasp_score_idx, dim=-1)
        # N x K*A mask
        topn_grasp = topN_g
        grasp_maxscore_mask = (grasp_idx_rank < topn_grasp)
        # N x topN
        grasp_maxscores = grasp_pos_scores[grasp_maxscore_mask].contiguous().view(rois.size()[:1] + (topn_grasp,))
        # N x topN x 8
        grasp_pred_boxes = grasp_pred_boxes[grasp_maxscore_mask].view(rois.size()[:1] + (topn_grasp, 8))
    else:
        raise NotImplementedError("Now ROI-GN only supports top-N grasp detection for each object.")

    # infer object boxes
    if cfg.TRAIN.COMMON.BBOX_REG:
        if cfg.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            normalizer = {'mean': cfg.TRAIN.COMMON.BBOX_NORMALIZE_MEANS, 'std': cfg.TRAIN.COMMON.BBOX_NORMALIZE_STDS}
            box_output = box_unnorm_torch(o_box_output, normalizer, 4, class_agnostic, n_classes)
            pred_boxes = bbox_transform_inv(rois, box_output, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info, 1)
    else:
        pred_boxes = rois.clone()

    pred_boxes = box_recover_scale_torch(pred_boxes, im_info[3], im_info[2])
    grasp_pred_boxes = box_recover_scale_torch(grasp_pred_boxes, im_info[3], im_info[2])

    all_box = [[]]
    all_grasp = [[]]
    for j in xrange(1, n_classes):
        if class_agnostic or not cfg.TRAIN.COMMON.BBOX_REG:
            cls_boxes = pred_boxes
        else:
            cls_boxes = pred_boxes[:, j * 4:(j + 1) * 4]
        cls_dets, cls_scores, box_keep_inds = box_filter(cls_boxes, o_scores[:, j], o_thresh, use_nms=True)
        cls_dets = np.concatenate((cls_dets, np.expand_dims(cls_scores, -1)), axis=-1)
        grasps = (grasp_pred_boxes.cpu().numpy())[box_keep_inds]

        if for_vis:
            cls_dets[:, -1] = j
        else:
            grasps = np.squeeze(grasps, axis = 1)
        all_box.append(cls_dets)
        all_grasp.append(grasps)

    if for_vis:
        all_box = np.concatenate(all_box[1:], axis = 0)
        all_grasp = np.concatenate(all_grasp[1:], axis = 0)

    return all_box, all_grasp

