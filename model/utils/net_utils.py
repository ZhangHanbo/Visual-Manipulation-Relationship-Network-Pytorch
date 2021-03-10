import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from model.utils.config import cfg

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.fcgn.bbox_transform_grasp import labels2points, grasp_decode
from model.roi_layers import nms
import time
import copy

import networkx as nx

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

def weights_normal_init(module, dev=0.01, bias = 0):
    if isinstance(module, list):
        for m in module:
            weights_normal_init(m, dev)
    else:
        for m in module.modules():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 0.0, dev)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias)

def weights_xavier_init(module, gain=1, bias=0, distribution='normal'):
    if isinstance(module, list):
        for m in module:
            weights_xavier_init(m)
    else:
        assert distribution in ['uniform', 'normal']
        for m in module.modules():
            if hasattr(m, 'weight'):
                if distribution == 'uniform':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias)

def weights_uniform_init(module, a=0, b=1, bias=0):
    if isinstance(module, list):
        for m in module:
            weights_uniform_init(m, a, b)
    else:
        for m in module.modules():
            if hasattr(m, 'weight'):
                nn.init.uniform_(m.weight, a, b)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias)

def weight_kaiming_init(module, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    if isinstance(module, list):
        for m in module:
            weight_kaiming_init(m, mode, nonlinearity, bias, distribution)
    else:
        assert distribution in ['uniform', 'normal']
        for m in module.modules():
            if hasattr(m, 'weight') and len(m.weight.shape) >= 2:
                if distribution == 'uniform':
                    nn.init.kaiming_uniform_(
                        m.weight, mode=mode, nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode=mode, nonlinearity=nonlinearity)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias)

def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

def set_bn_unfix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=True

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def gradient_norm(model):
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm.item())
    return totalnorm

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = gradient_norm(model)
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

def _focal_loss(cls_prob, labels, alpha = 0.25, gamma = 2):

    labels = labels.view(-1)
    final_prob = torch.gather(cls_prob.view(-1, cls_prob.size(-1)), 1, labels.unsqueeze(1))
    loss_cls = - torch.log(final_prob)

    # setting focal weights
    focal_weights = torch.pow((1. - final_prob), gamma)

    # setting the coefficient to balance pos and neg samples.
    alphas = torch.Tensor(focal_weights.shape).zero_().type_as(focal_weights)
    alphas[labels == 0] = 1. - alpha
    alphas[labels > 0] = alpha

    loss_cls = (loss_cls * focal_weights * alphas).sum() / torch.clamp(torch.sum(labels > 0).float(), min = 1.0)
    # loss_cls = (loss_cls * focal_weights * alphas).mean()

    return loss_cls

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
            keep = nms(cls_dets[:, :4], cls_dets[:, 4], cfg.TEST.COMMON.NMS)
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

def objdet_inference(cls_prob, box_output, im_info, box_prior = None, class_agnostic = True,
                     for_vis = False, recover_imscale = True, with_cls_score = False):
    """
    :param cls_prob: predicted class info
    :param box_output: predicted bounding boxes (for anchor-based detection, it indicates deltas of boxes).
    :param im_info: image scale information, for recovering the original bounding box scale before image resizing.
    :param box_prior: anchors, RoIs, e.g.
    :param class_agnostic: whether the boxes are class-agnostic. For faster RCNN, it is class-specific by default.
    :param n_classes: number of object classes
    :param for_vis: the results are for visualization or validation of VMRN.
    :param recover_imscale: whether the predicted bounding boxes are recovered to the original scale.
    :param with_cls_score: if for_vis and with_cls_score are both true, the class confidence score will be attached.
    :return: a list of bounding boxes, one class corresponding to one element. If for_vis, they will be concatenated.
    """
    assert box_output.dim() == 2, "Multi-instance batch inference has not been implemented."
    n_classes = cls_prob.shape[1]

    if for_vis:
        thresh = cfg.TEST.COMMON.OBJ_DET_THRESHOLD
    else:
        thresh = 0.01

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
    if recover_imscale:
        pred_boxes = box_recover_scale_torch(pred_boxes, im_info[3], im_info[2])

    all_box = [[]]
    if for_vis:
        cls = []
    for j in xrange(1, n_classes):
        if class_agnostic:
            cls_boxes = pred_boxes
        else:
            cls_boxes = pred_boxes[:, j * 4:(j + 1) * 4]
        cls_dets, cls_scores, _ = box_filter(cls_boxes, scores[:, j], thresh, use_nms = True)
        cls_dets = np.concatenate((cls_dets, np.expand_dims(cls_scores, -1)), axis = -1)
        if for_vis:
            cls.append(j * np.ones((cls_dets.shape[0], 1)))
        all_box.append(cls_dets)
    if for_vis:
        cls = np.concatenate(cls, axis=0)
        all_box = np.concatenate(all_box[1:], axis=0)
        if with_cls_score:
            all_box = np.concatenate([all_box, cls], axis = 1)
        else:
            all_box[:, -1:] = cls
    return all_box

def grasp_inference(cls_prob, box_output, im_info, box_prior = None, topN = False, recover_imscale = True):
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
    if recover_imscale:
        pred_boxes = box_recover_scale_torch(pred_boxes, im_info[3], im_info[2])

    grasps, scores, _ = box_filter(pred_boxes, scores[:, 1], thresh, use_nms = False)
    grasps = np.concatenate((grasps, np.expand_dims(scores, -1)), axis=-1)
    if topN:
        grasps = grasps[:topN]
    return grasps

def objgrasp_inference(o_cls_prob, o_box_output, g_cls_prob, g_box_output, im_info, rois = None,
                       class_agnostic = True, g_box_prior = None, for_vis = False, topN_g = False,
                       recover_imscale = True):
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
    n_classes = o_cls_prob.shape[1]

    g_scores = g_cls_prob

    if for_vis:
        o_thresh = cfg.TEST.COMMON.OBJ_DET_THRESHOLD
    else:
        o_thresh = 0.01
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

    if recover_imscale:
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

def rel_prob_to_mat(rel_cls_prob, num_obj):
    """
    :param rel_cls_prob: N x 3 relationship class score
    :param num_obj: an int indicating the number of objects
    :return: a N_obj x N_obj relationship matrix. element(i, j) indicates the relationship between i and j,
                i.e., i  -- rel --> j

    The input is Tensors and the output is np.array.
    """

    rel_cls_prob_cpu = rel_cls_prob.cpu()
    if num_obj == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    elif num_obj == 1:
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.float32)

    rel_score, rel = torch.max(rel_cls_prob_cpu, dim = 1)
    rel += 1 # to make the label match the macro, i.e., cfg.VMRN.CHILD, cfg.VMRN.FATHER, cfg.VMRN.NO_REL
    rel[rel >=3] = 3

    rel_mat = np.zeros((num_obj, num_obj), dtype=np.int32)
    rel_score_mat = np.zeros((3, num_obj, num_obj), dtype=np.float32)
    counter = 0
    for o1 in range(num_obj):
        for o2 in range(o1 + 1, num_obj):
            rel_mat[o1, o2] = rel[counter]
            rel_score_mat[:, o1, o2] = rel_cls_prob_cpu[counter]
            counter += 1
    for o1 in range(num_obj):
        for o2 in range(o1):
            if rel_mat[o2, o1] == 3:
                rel_mat[o1, o2] = rel_mat[o2, o1]
            elif rel_mat[o2, o1] == 1 or rel_mat[o2, o1] == 2:
                rel_mat[o1, o2] = 3 - rel_mat[o2, o1]
            else:
                raise RuntimeError
            rel_score_mat[:, o1, o2] = rel_score_mat[:, o2, o1]
    return rel_mat, rel_score_mat

def relscores_to_visscores(rel_score_mat):
    return np.max(rel_score_mat, axis=0)

def create_mrt(rel_mat, class_names=None, rel_score=None):
    # using relationship matrix to create manipulation relationship tree
    mrt = nx.DiGraph()

    if rel_mat.size == 0:
        # No object is detected
        return mrt
    elif (rel_mat > 0).sum() == 0:
        # No relation is detected, meaning that there is only one object in the scene
        class_names = class_names or [0]
        mrt.add_node(class_names[0])
        return mrt

    node_num = np.max(np.where(rel_mat > 0)[0]) + 1
    if class_names is None:
        # no other node information
        class_names = list(range(node_num))
    elif isinstance(class_names[0], float):
        # normalized confidence score
        class_names = ["{:d}\n{:.2f}".format(i, cls) for i, cls in enumerate(class_names)]
    else:
        # class name
        class_names = ["{:s}{:d}".format(cls, i) for i, cls in enumerate(class_names)]

    if rel_score is None:
        rel_score = np.zeros(rel_mat.shape, dtype=np.float32)

    for obj1 in xrange(node_num):
        mrt.add_node(class_names[obj1])
        for obj2 in xrange(obj1):
            if rel_mat[obj1, obj2].item() == cfg.VMRN.FATHER:
                # OBJ1 is the father of OBJ2
                weight = rel_score[..., obj1, obj2].max()
                mrt.add_edge(class_names[obj2], class_names[obj1],
                             weight=np.round(weight.item(), decimals=2))

            if rel_mat[obj1, obj2].item() == cfg.VMRN.CHILD:
                # OBJ1 is the father of OBJ2
                weight = rel_score[..., obj1, obj2].max()
                mrt.add_edge(class_names[obj1], class_names[obj2],
                             weight=np.round(weight.item(), decimals=2))
    return mrt

def find_all_paths(mrt, t_node = 0):
    """
    :param mrt: a manipulation relationship tree
    :param t_node: the index of the target node
    :return: paths: a list of all possible paths
    """
    # depth-first search
    assert t_node in mrt.nodes, "The target node is not found in the given manipulation relationship tree."
    paths = []
    for e in mrt.edges:
        if t_node == e[1]:
            # find all sub paths from current target node
            paths += find_all_paths(mrt, e[0])
    # attach current target node in front of all sub paths
    for i in xrange(len(paths)):
        paths[i] += [t_node, ]
    if len(paths) == 0:
        return [[t_node, ]]
    else:
        return paths

def find_shortest_path(mrt, t_node = 0):
    paths = find_all_paths(mrt, t_node)
    p_lenth = np.inf
    best_path = None
    for p in paths:
        if len(p) < p_lenth:
            best_path = p
    return best_path

def find_all_leaves(mrt, t_node = 0):
    """
    :param mrt: a manipulation relationship tree
    :param t_node: the index of the target node
    :return: paths: a list of all possible paths
    NOTE: this function cannot deal with graph including cycles.
    """
    # depth-first search
    assert t_node in mrt.nodes, "The target node is not found in the given manipulation relationship tree."
    path = [t_node,]

    for e in mrt.edges:
        if t_node == e[1]:
            # find all sub paths from current target node
            path.append(e[0])

    for leaf in path[1:]:
        sub_leaves = find_all_leaves(mrt, leaf)
        exist_leaf_inds = []
        for leaf in sub_leaves[1:]:
            if leaf not in path:
                path.append(leaf)
            else:
                exist_leaf_inds.append(path.index(leaf))

        # for existing nodes, we need move them all at the end of the path, maintaining the current order
        exist_leaves = [path[ind] for ind in np.sort(exist_leaf_inds)]

        for leaf in exist_leaves:
            path.remove(leaf)
            path.append(leaf)

    return path

def leaf_and_desc_estimate(rel_prob_mat, sample_num = 1000, with_virtual_node=False, removed=None):
    # TODO: Numpy may support a faster implementation.
    def sample_trees(rel_prob, sample_num=1):
        return torch.multinomial(rel_prob, sample_num, replacement=True)

    cuda_data = False
    if rel_prob_mat.is_cuda:
        # this function runs much faster on CPU.
        cuda_data = True
        rel_prob_mat = rel_prob_mat.cpu()

    # add virtual node, with uniform relation priors
    num_obj = rel_prob_mat.shape[-1]
    if with_virtual_node:
        if removed is None:
            removed = []
        removed = torch.tensor(removed).long()
        v_row = torch.zeros((3, 1, num_obj + 1)).type_as(rel_prob_mat)
        v_column = torch.zeros((3, num_obj, 1)).type_as(rel_prob_mat)
        # no other objects can be the parent node of the virtual node,
        # i.e., we assume that the virtual node must be a root node
        # 1) if the virtual node is the target, its parents can be ignored
        # 2) if the virtual node is not the target, such a setting will
        # not affect the relationships among other nodes
        v_column[0] = 0
        v_column[1] = 1./3.
        v_column[2] = 2./3.
        v_column[1, removed] = 0.
        v_column[2, removed] = 1.
        rel_prob_mat = torch.cat(
            [torch.cat([rel_prob_mat, v_column], dim=2),
             v_row], dim=1)
    else:
        # initialize the virtual node to have no relationship with other objects
        v_row = torch.zeros((3, 1, num_obj + 1)).type_as(rel_prob_mat)
        v_column = torch.zeros((3, num_obj, 1)).type_as(rel_prob_mat)
        v_column[2, :, 0] = 1
        rel_prob_mat = torch.cat(
            [torch.cat([rel_prob_mat, v_column], dim=2),
             v_row], dim=1)

    rel_prob_mat = rel_prob_mat.permute((1, 2, 0))
    mrt_shape = rel_prob_mat.shape[:2]
    rel_prob = rel_prob_mat.view(-1, 3)
    rel_valid_ind = rel_prob.sum(-1) > 0

    # sample mrts
    samples = sample_trees(rel_prob[rel_valid_ind], sample_num) + 1
    mrts = torch.zeros((sample_num,) + mrt_shape).type_as(samples)
    mrts = mrts.view(sample_num, -1)
    mrts[:, rel_valid_ind] = samples.permute((1,0))
    mrts = mrts.view((sample_num,) + mrt_shape)
    p_mats = (mrts == 1)
    c_mats = (mrts == 2)
    adj_mats = p_mats + c_mats.transpose(1,2)
    
    def v_node_is_leaf(adj_mat):
        return adj_mat[-1].sum() == 0

    def no_cycle(adj_mat):
        keep_ind = (adj_mat.sum(0) > 0)
        if keep_ind.sum() == 0:
            return True
        elif keep_ind.sum() == adj_mat.shape[0]:
            return False
        adj_mat = adj_mat[keep_ind][:, keep_ind]
        return no_cycle(adj_mat)

    def descendants(adj_mat):
        def find_children(node, adj_mat):
            return torch.nonzero(adj_mat[node]).view(-1).tolist()

        def find_descendant(node, adj_mat, visited, desc_mat):
            if node in visited:
                return visited, desc_mat
            else:
                desc_mat[node][node] = 1
                for child in find_children(node, adj_mat):
                    visited, desc_mat = find_descendant(child, adj_mat, visited, desc_mat)
                    desc_mat[node] = (desc_mat[node] | desc_mat[child])
                visited.append(node)
            return visited, desc_mat

        roots = torch.nonzero(adj_mat.sum(0) == 0).view(-1).tolist()
        visited = []
        desc_mat = torch.zeros(mrt_shape).type_as(adj_mat).long()
        for root in roots:
            visited, desc_list = find_descendant(root, adj_mat, visited, desc_mat)
        return desc_mat.transpose(0,1)

    leaf_desc_prob = torch.zeros(mrt_shape).type_as(rel_prob_mat)
    desc_prob = torch.zeros(mrt_shape).type_as(rel_prob_mat)
    desc_num = torch.zeros(mrt_shape[0]).type_as(rel_prob_mat)
    ance_num = torch.zeros(mrt_shape[0]).type_as(rel_prob_mat)

    if with_virtual_node:
        v_desc_num_after_q2 = torch.zeros(mrt_shape[0]).type_as(rel_prob_mat)

    count = 0
    for adj_mat in adj_mats:
        if removed is None and with_virtual_node and v_node_is_leaf(adj_mat):
            continue
        if not no_cycle(adj_mat):
            continue
        else:
            desc_mat = descendants(adj_mat)
            desc_num += desc_mat.sum(0)
            ance_num += desc_mat.sum(1) - 1  # ancestors don't include the object itself
            leaf_desc_mat = desc_mat * (adj_mat.sum(1, keepdim=True) == 0)
            desc_prob += desc_mat
            leaf_desc_prob += leaf_desc_mat
            count += 1

    desc_num /= count
    ance_num /= count
    leaf_desc_prob /= count
    desc_prob /= count
    leaf_prob = leaf_desc_prob.diag()
    if cuda_data:
        leaf_desc_prob = leaf_desc_prob.cuda()
        leaf_prob = leaf_prob.cuda()
        desc_prob = desc_prob.cuda()
        ance_num = ance_num.cuda()
        desc_num = desc_num.cuda()

    return leaf_desc_prob, desc_prob, leaf_prob, desc_num, ance_num

def leaf_prob_comp(rel_prob_mat):
    # TODO: this function does not exclude the situations in which the MRT includes cycles.
    parent_prob_mat = rel_prob_mat[cfg.VMRN.FATHER - 1]
    child_prob_mat = rel_prob_mat[cfg.VMRN.CHILD - 1]
    parent_prob_mat += child_prob_mat.transpose(0, 1)
    return torch.cumprod(1 - parent_prob_mat, dim = -1)[:, -1]

def inner_loop_planning(belief, planning_depth=3):
    num_obj = belief["ground_prob"].shape[0] - 1 # exclude the virtual node
    penalty_for_asking = -3
    # ACTIONS: Do you mean ... ? (num_obj) + Where is the target ? (1) + grasp object (num_obj)
    def grasp_reward_estimate(belief):
        # reward of grasping the corresponding object
        # return is a 1-D tensor including num_obj elements, indicating the reward of grasping the corresponding object.
        ground_prob = belief["ground_prob"]
        leaf_desc_tgt_prob = (belief["leaf_desc_prob"] * ground_prob.unsqueeze(0)).sum(-1)
        leaf_prob = torch.diag(belief["leaf_desc_prob"])
        not_leaf_prob = 1. - leaf_prob
        target_prob = ground_prob
        leaf_tgt_prob = leaf_prob * target_prob
        leaf_desc_prob = leaf_desc_tgt_prob - leaf_tgt_prob
        leaf_but_not_desc_tgt_prob = leaf_prob - leaf_desc_tgt_prob

        # grasp and the end
        r_1 = not_leaf_prob * (-10) + leaf_but_not_desc_tgt_prob * (-10) + leaf_desc_prob * (-10)\
                  + leaf_tgt_prob * (0)
        r_1 = r_1[:-1] # exclude the virtual node

        # grasp and not the end
        r_2 = not_leaf_prob * (-10) + leaf_but_not_desc_tgt_prob * (-5) + leaf_desc_prob * (0)\
                  + leaf_tgt_prob * (-10)
        r_2 = r_2[:-1]  # exclude the virtual node
        return torch.cat([r_1, r_2], dim=0)

    def belief_update(belief):
        I = torch.eye(belief["ground_prob"].shape[0]).type_as(belief["ground_prob"])
        updated_beliefs = []
        # Do you mean ... ?
        # Answer No
        beliefs_no = belief["ground_prob"].unsqueeze(0).repeat(num_obj + 1, 1)
        beliefs_no *= (1. - I)
        beliefs_no /= torch.clamp(torch.sum(beliefs_no, dim = -1, keepdim=True), min=1e-10)
        # Answer Yes
        beliefs_yes = I.clone()
        for i in range(beliefs_no.shape[0] - 1):
            updated_beliefs.append([beliefs_no[i], beliefs_yes[i]])

        # Is the target detected? Where is it?
        updated_beliefs.append([beliefs_no[-1], I[-1],])
        return updated_beliefs

    def is_onehot(vec, epsilon = 1e-2):
        return (torch.abs(vec - 1) < epsilon).sum().item() > 0

    def estimate_q_vec(belief, current_d):
        if current_d == planning_depth - 1:
            q_vec = grasp_reward_estimate(belief)
            return q_vec
        else:
            # branches of grasping
            q_vec = grasp_reward_estimate(belief).tolist()
            ground_prob = belief["ground_prob"]
            new_beliefs = belief_update(belief)
            new_belief_dict = copy.deepcopy(belief)

            # Q1
            for i, new_belief in enumerate(new_beliefs[:-1]):
                q = 0
                for j, b in enumerate(new_belief):
                    new_belief_dict["ground_prob"] = b
                    # branches of asking questions
                    if is_onehot(b):
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                    else:
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                    if j == 0:
                        # Answer is No
                        q += t_q * (1 - ground_prob[i])
                    else:
                        # Answer is Yes
                        q += t_q * ground_prob[i]
                q_vec.append(q.item())

            # Q2
            q = 0
            new_belief = new_beliefs[-1]
            for j, b in enumerate(new_belief):
                new_belief_dict["ground_prob"] = b
                if j == 0:
                    # target has been detected
                    if is_onehot(b):
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                    else:
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                    q += t_q * (1 - ground_prob[-1])
                else:
                    new_belief_dict["leaf_desc_prob"][:, -1] = new_belief_dict["leaf_desc_prob"][:, :-1].sum(-1) / num_obj
                    t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                    q += t_q * ground_prob[-1]
            q_vec.append(q.item())
            return torch.Tensor(q_vec).type_as(belief["ground_prob"])

    with torch.no_grad():
        belief["leaf_desc_prob"], belief["desc_prob"], belief["leaf_prob"], belief["desc_num"], belief["ance_num"] = \
            leaf_and_desc_estimate(prob_rel_mat, sample_num=1000, with_virtual_node=True)

    q_vec = estimate_q_vec(belief, 0)
    print("Q Value for Each Action: ")
    print(q_vec.tolist()[:num_obj])
    print(q_vec.tolist()[num_obj:2*num_obj])
    print(q_vec.tolist()[2*num_obj:3*num_obj])
    print(q_vec.tolist()[3*num_obj])
    return torch.argmax(q_vec).item()


def planning_with_macro(belief, planning_depth=3):
    """
    :param belief: including "leaf_desc_prob", "desc_num", and "ground_prob"
    :param planning_depth:
    :return:
    """
    num_obj = belief["ground_prob"].shape[0] - 1  # exclude the virtual node

    # ALL ACTIONS INCLUDE:
    # Do you mean ... ? (num_obj) + Where is the target ? (1) + grasping macro (1)
    penalty_for_asking = -2
    penalty_for_fail = -10

    def gen_grasp_macro(belief):

        grasp_macros = {i: None for i in range(num_obj)}
        belief_infos = belief["infos"]

        cache_leaf_desc_prob = {}
        cache_leaf_prob = {}
        for i in range(num_obj + 1):
            grasp_macro = {"seq": [], "leaf_prob": []}
            grasp_macro["seq"].append(torch.argmax(belief_infos["leaf_desc_prob"][:, i]).item())
            grasp_macro["leaf_prob"].append(belief_infos["leaf_prob"][grasp_macro["seq"][0]].item())

            rel_mat = belief["relation_prob"].clone()
            while (grasp_macro["seq"][-1] != i):
                removed = torch.tensor(grasp_macro["seq"]).type_as(rel_mat).long()
                indice = ''.join([str(o) for o in np.sort(grasp_macro["seq"]).tolist()])
                if indice in cache_leaf_desc_prob:
                    leaf_desc_prob = cache_leaf_desc_prob[indice]
                    leaf_prob = cache_leaf_prob[indice]
                else:
                    rel_mat[0:2, removed, :] = 0.
                    rel_mat[0:2, :, removed] = 0.
                    rel_mat[2, removed, :] = 1.
                    rel_mat[2, :, removed] = 1.
                    triu_mask = torch.triu(torch.ones(rel_mat[0].shape), diagonal=1)
                    triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
                    rel_mat *= triu_mask

                    leaf_desc_prob, _, leaf_prob, _, _ = \
                        leaf_and_desc_estimate(rel_mat, removed=grasp_macro["seq"], with_virtual_node=True)

                    cache_leaf_desc_prob[indice] = leaf_desc_prob
                    cache_leaf_prob[indice] = leaf_prob

                grasp_macro["seq"].append(torch.argmax(leaf_desc_prob[:, i]).item())
                grasp_macro["leaf_prob"].append(leaf_prob[grasp_macro["seq"][-1]].item())

            grasp_macro["seq"] = torch.tensor(grasp_macro["seq"]).type_as(rel_mat).long()
            grasp_macro["leaf_prob"] = torch.tensor(grasp_macro["leaf_prob"]).type_as(rel_mat)
            grasp_macros[i] = grasp_macro

        return grasp_macros

    def grasp_reward_estimate(belief):
        # reward of grasping macro, equal to: desc_num * reward_of_each_grasp_step
        # POLICY: treat the object with the highest conf score as the target
        ground_prob = belief["ground_prob"]
        target = torch.argmax(ground_prob).item()
        grasp_macros = belief["grasp_macros"][target]
        leaf_prob = grasp_macros["leaf_prob"]

        # p_remove_target = ground_prob[grasp_macros["seq"][:-1]]
        # if p_remove_target.numel() > 0:
        #     p_remove_target = torch.sum(p_remove_target).item()
        # else:
        #     p_remove_target = 0.
        p_not_remove_non_leaf = torch.cumprod(leaf_prob, dim=0)[-1].item()
        p_fail = 1. - ground_prob[target].item() * p_not_remove_non_leaf

        return penalty_for_fail * p_fail

    def belief_update(belief):
        I = torch.eye(belief["ground_prob"].shape[0]).type_as(belief["ground_prob"])
        updated_beliefs = []
        # Do you mean ... ?
        # Answer No
        beliefs_no = belief["ground_prob"].unsqueeze(0).repeat(num_obj + 1, 1)
        beliefs_no *= (1. - I)
        beliefs_no /= torch.clamp(torch.sum(beliefs_no, dim=-1, keepdim=True), min=1e-10)
        # Answer Yes
        beliefs_yes = I.clone()
        for i in range(beliefs_no.shape[0] - 1):
            updated_beliefs.append([beliefs_no[i], beliefs_yes[i]])

        return updated_beliefs

    def is_onehot(vec, epsilon=1e-2):
        return (torch.abs(vec - 1) < epsilon).sum().item() > 0

    def estimate_q_vec(belief, current_d):
        if current_d == planning_depth - 1:
            return torch.tensor([grasp_reward_estimate(belief)])
        else:
            # branches of grasping
            q_vec = [grasp_reward_estimate(belief)]
            ground_prob = belief["ground_prob"]
            new_beliefs = belief_update(belief)
            new_belief_dict = copy.deepcopy(belief)

            # q-value for asking Q1
            for i, new_belief in enumerate(new_beliefs):
                q = 0
                for j, b in enumerate(new_belief):
                    new_belief_dict["ground_prob"] = b
                    # branches of asking questions
                    if is_onehot(b):
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, planning_depth - 1).max())
                    else:
                        t_q = (penalty_for_asking + estimate_q_vec(new_belief_dict, current_d + 1).max())
                    if j == 0:
                        # Answer is No
                        q += t_q * (1 - ground_prob[i])
                    else:
                        # Answer is Yes
                        q += t_q * ground_prob[i]
                q_vec.append(q.item())

            return torch.Tensor(q_vec).type_as(belief["ground_prob"])

    infos = {}
    with torch.no_grad():
        infos["leaf_desc_prob"], infos["desc_prob"], infos["leaf_prob"], infos["desc_num"], infos["ance_num"] = \
            leaf_and_desc_estimate(belief["relation_prob"], sample_num=1000, with_virtual_node=True)
    belief["infos"] = infos
    belief["grasp_macros"] = gen_grasp_macro(belief)

    q_vec = estimate_q_vec(belief, 0)
    print("Q Value for Each Action: ")
    print("Grasping:{:.3f}".format(q_vec.tolist()[0]))
    print("Asking Q1:{:s}".format(q_vec.tolist()[1:num_obj + 1]))
    # print("Asking Q2:{:.3f}".format(q_vec.tolist()[num_obj+1]))
    return torch.argmax(q_vec).item()

if __name__ == '__main__':

    # rel_mat = [[0,1,1,3,3],
    #             [2,0,3,1,3],
    #             [2,3,0,1,1],
    #             [3,2,2,0,3],
    #             [3,3,2,3,0]]
    # rel_mat = [[0,3,1,1,2],
    #             [3,0,2,2,3],
    #             [2,1,0,2,2],
    #             [2,1,1,0,3],
    #             [1,3,1,3,0]]
    # rel_mat = [ [0,3,3,1,2,3,3,2],
    #             [3,0,2,3,3,1,2,3],
    #             [3,1,0,1,3,1,2,1],
    #             [2,3,2,0,3,3,3,2],
    #             [1,3,3,3,0,1,3,2],
    #             [3,2,2,3,2,0,3,2],
    #             [3,1,1,3,3,3,0,3],
    #             [1,3,2,1,1,1,3,0] ]
    # rel_mat = torch.Tensor(rel_mat)
    # mrt = create_mrt(rel_mat)
    # path = find_all_leaves(mrt, 6)

    prob_rel_mat = [[
        [0, 0.9, 0.8, 0.2, 0.1],
        [0, 0., 0.1, 0.7, 0.1],
        [0, 0., 0., 0.2, 0.9],
        [0, 0., 0., 0, 0.1],
        [0, 0, 0, 0, 0, ]
    ],[
        [0, 0.1, 0.1, 0.1, 0.1],
        [0, 0, 0.1, 0.1, 0.1],
        [0, 0., 0, 0.1, 0.05],
        [0, 0, 0, 0, 0.1],
        [0, 0, 0, 0, 0, ]
    ],[
        [0, 0., 0.1, 0.7, 0.8],
        [0, 0, 0.8, 0.2, 0.8],
        [0, 0., 0, 0.7, 0.05],
        [0, 0, 0, 0, 0.8],
        [0, 0, 0, 0, 0, ]
    ]]
    prob_rel_mat = np.array(prob_rel_mat)
    prob_rel_mat = torch.from_numpy(prob_rel_mat)
    # prob_rel_mat = prob_rel_mat.cuda()

    t_b = time.time()

    belief = {}
    belief["relation_prob"] = prob_rel_mat
    belief["ground_prob"] = torch.Tensor([0.0, 0.0, 0.0, 0.25, 0.75, 0.0])

    planning_with_macro(belief)
    print("cost: {:.2f}s".format(time.time() - t_b))