import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from model.utils.config import cfg
from model.roi_crop.functions.roi_crop import RoICropFunction

import cv2
import pdb
import random

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


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


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


def vis_detections(im_obj, im_grasp, class_name, dets, thresh=0.8, grasps=None, color_dict=None, index=None):
    """Visual debugging of detections."""
    if color_dict is not None:
        bbox_color = color_dict[class_name]
    else:
        bbox_color = (0, 0, 0)
    # if no index, zero is used
    if index is None:
        index = np.zeros(dets.shape[0])

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            if grasps is not None:
                if len(grasps.shape) == 2:
                    current_gr = grasps[i]
                    im_grasp = draw_single_grasp(im_grasp, current_gr, '%s ind:%d' % (class_name, index[i]), bbox_color)
                elif len(grasps.shape) == 3:
                    current_gr = grasps[i]
                    num_grasp = current_gr.shape[0]
                    for k in range(num_grasp):
                        im_grasp = draw_single_grasp(im_grasp, current_gr[k], '%s ind:%d' % (class_name, index[i]),
                                                     bbox_color)
            im_obj = draw_single_bbox(im_obj, bbox, '%s: %.3f ind: %d' % (class_name, score, index[i]), bbox_color)

    return im_obj, im_grasp


def draw_single_bbox(img, bbox, text_str="", bbox_color=(255, 0, 0)):
    bbox = tuple(bbox)
    text_rd = (bbox[2], bbox[1] + 25)
    cv2.rectangle(img, bbox[0:2], bbox[2:4], bbox_color, 2)
    cv2.rectangle(img, bbox[0:2], text_rd, bbox_color, -1)
    cv2.putText(img, text_str, (bbox[0], bbox[1] + 20),
                cv2.FONT_HERSHEY_PLAIN,
                2, (255, 255, 255), thickness=2)
    return img


def draw_single_grasp(img, grasp, test_str="", bbox_color=(255, 0, 0)):
    text_len = len(test_str)
    gr_c = (int((grasp[0] + grasp[4]) / 2), int((grasp[1] + grasp[5]) / 2))
    for j in range(4):
        if j % 2 == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        p1 = (int(grasp[2 * j]), int(grasp[2 * j + 1]))
        p2 = (int(grasp[(2 * j + 2) % 8]), int(grasp[(2 * j + 3) % 8]))
        cv2.line(img, p1, p2, color, 2)
    text_w = 17 * text_len
    gtestpos = (gr_c[0] - text_w / 2, gr_c[1] + 20)
    gtest_lu = (gr_c[0] - text_w / 2, gr_c[1])
    gtest_rd = (gr_c[0] + text_w / 2, gr_c[1] + 25)
    cv2.rectangle(img, gtest_lu, gtest_rd, bbox_color, -1)
    cv2.putText(img, test_str, gtestpos,
                cv2.FONT_HERSHEY_PLAIN,
                2, (255, 255, 255), thickness=2)
    return img


def draw_grasp(im, dets):
    """
    :param im: original image
    :param dets: detections. N x 8
    :return: im
    """
    # make memory contiguous
    im = np.ascontiguousarray(im)
    num_grasp = dets.shape[0]
    for i in range(num_grasp):
        for j in range(4):
            if j % 2 == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            p1 = (dets[i][2 * j], dets[i][2 * j + 1])
            p2 = (dets[i][(2 * j + 2) % 8], dets[i][(2 * j + 3) % 8])
            cv2.line(im, p1, p2, color, 2)
    return im


def draw_rect(im, dets):
    """
    :param im: original image
    :param dets: detections. N x 8
    :return: im
    """
    # make memory contiguous
    im = np.ascontiguousarray(im)
    num_grasp = dets.shape[0]
    dets = np.concatenate([
        dets[:, 0:2],
        dets[:, 0:1],
        dets[:, 3:4],
        dets[:, 2:4],
        dets[:, 2:3],
        dets[:, 1:2]

    ], axis=1)
    for i in range(num_grasp):
        for j in range(4):
            color = (0, 255, 0)
            p1 = (dets[i][2 * j], dets[i][2 * j + 1])
            p2 = (dets[i][(2 * j + 2) % 8], dets[i][(2 * j + 3) % 8])
            cv2.line(im, p1, p2, color, 2)
    return im


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
    theta = torch.cat([ \
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
        pre_pool_size = cfg.RCNN_COMMON.POOLING_SIZE * 2
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)
        crops = F.max_pool2d(crops, 2, 2)
    else:
        grid = F.affine_grid(theta,
                             torch.Size((rois.size(0), 1, cfg.RCNN_COMMON.POOLING_SIZE, cfg.RCNN_COMMON.POOLING_SIZE)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
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
    theta = torch.cat([ \
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

    theta = torch.cat([ \
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
    C = 2  # random.randint(1, 8)
    H = 5  # random.randint(1, 8)
    W = 4  # random.randint(1, 8)
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
    grid_yx = torch.stack([grid_clone.data[:, :, :, 1], grid_clone.data[:, :, :, 0]], 3).contiguous().cuda()
    out_stn = crf.forward(input_p, grid_yx)
    grad_inputs = crf.backward(grad_outputs_clone.data)
    grad_input_stn = grad_inputs[0]
    pdb.set_trace()

    delta = (grad_input_off.data - grad_input_stn).sum()


def rel_prob_to_mat(rel_cls_prob, num_obj):
    """
    :param rel_cls_prob: N x 3 relationship class score
    :param num_obj: an int indicating the number of objects
    :return: a N_obj x N_obj relationship matrix. element(i, j) indicates the relationship between i and j,
                i.e., i  -- rel --> j

    The input is Tensors and the output is np.array.
    """

    if num_obj == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    elif num_obj == 1:
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.float32)

    _, rel = torch.max(rel_cls_prob, dim=1)
    rel += 1  # to make the label match the macro, i.e., cfg.VMRN.CHILD, cfg.VMRN.FATHER, cfg.VMRN.NO_REL
    rel[rel >= 3] = 3

    rel_mat = np.zeros((num_obj, num_obj), dtype=np.int32)
    rel_score_mat = np.zeros((3, num_obj, num_obj), dtype=np.float32)
    counter = 0
    for o1 in range(num_obj):
        for o2 in range(o1 + 1, num_obj):
            rel_mat[o1, o2] = rel[counter]
            rel_score_mat[:, o1, o2] = rel_cls_prob[counter]
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
        class_names = list(range(node_num))
    else:
        class_names = [cls + str(i) for i, cls in enumerate(class_names)]

    if rel_score is None:
        rel_score = np.zeros(rel_mat.shape, dtype=np.float32)

    for obj1 in xrange(node_num):
        mrt.add_node(class_names[obj1])
        for obj2 in xrange(obj1):
            if rel_mat[obj1, obj2].item() == cfg.VMRN.FATHER:
                # OBJ1 is the father of OBJ2
                mrt.add_edge(class_names[obj2], class_names[obj1],
                             weight=np.round(rel_score[obj1, obj2].item(), decimals=2))

            if rel_mat[obj1, obj2].item() == cfg.VMRN.CHILD:
                # OBJ1 is the father of OBJ2
                mrt.add_edge(class_names[obj1], class_names[obj2],
                             weight=np.round(rel_score[obj1, obj2].item(), decimals=2))
    return mrt


def find_all_paths(mrt, t_node=0):
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


def find_shortest_path(mrt, t_node=0):
    paths = find_all_paths(mrt, t_node)
    p_lenth = np.inf
    best_path = None
    for p in paths:
        if len(p) < p_lenth:
            best_path = p
    return best_path


def find_all_leaves(mrt, t_node=0):
    """
    :param mrt: a manipulation relationship tree
    :param t_node: the index of the target node
    :return: paths: a list of all possible paths
    NOTE: this function cannot deal with graph including cycles.
    """
    # depth-first search
    assert t_node in mrt.nodes, "The target node is not found in the given manipulation relationship tree."
    path = [t_node, ]

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

def leaf_and_descendant_stats(rel_prob_mat, sample_num = 1000):
    # TODO: Numpy may support a faster implementation.
    def sample_trees(rel_prob, sample_num=1):
        return torch.multinomial(rel_prob, sample_num, replacement=True)

    for i in range(3):
        rel_prob_mat[i] = torch.triu(rel_prob_mat[i], diagonal=1)

    cuda_data = False
    if rel_prob_mat.is_cuda:
        # this function runs much faster on CPU.
        cuda_data = True
        rel_prob_mat = rel_prob_mat.cpu()

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
    f_mats = (mrts == 1)
    c_mats = (mrts == 2)
    adj_mats = f_mats + c_mats.transpose(1,2)

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
    count = 0
    for adj_mat in adj_mats:
        if no_cycle(adj_mat):
            desc_mat = descendants(adj_mat)
            leaf_desc_mat = desc_mat.float() * (adj_mat.sum(1, keepdim=True) == 0).float()
            leaf_desc_prob += leaf_desc_mat
            count += 1
    leaf_desc_prob = leaf_desc_prob / count

    leaf_desc_prob_with_v_node = torch.eye(mrt_shape[0] + 1).type_as(rel_prob_mat)
    leaf_desc_prob_with_v_node[:mrt_shape[0], :mrt_shape[1]] = leaf_desc_prob
    if cuda_data:
        leaf_desc_prob_with_v_node = leaf_desc_prob_with_v_node.cuda()
    return leaf_desc_prob_with_v_node

def leaf_prob(rel_prob_mat):
    for i in range(3):
        rel_prob_mat[i] = torch.triu(rel_prob_mat[i], diagonal=1)
    # TODO: this function does not exclude the situations in which the MRT includes cycles.
    parent_prob_mat = rel_prob_mat[cfg.VMRN.FATHER - 1]
    child_prob_mat = rel_prob_mat[cfg.VMRN.CHILD - 1]
    parent_prob_mat += child_prob_mat.transpose(0, 1)
    return torch.cumprod(1 - parent_prob_mat, dim = -1)[:, -1]