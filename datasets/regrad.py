# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import os
import os.path as osp
import time
import sys
import json
import numpy as np
import pickle as pkl
import xml.etree.ElementTree as ET
import pdb
import torch

from torch.utils.data import Dataset
from .imdb import imdb
from model.utils.config import cfg

class regrad(imdb):
    def __init__(self, image_set, version = 'super_mini', devkit_path=None):
        imdb.__init__(self, 'regrad_' + version + "_" + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, image_set)
        self._cache_path = os.path.join(cfg.DATA_DIR, "cache")
        self._image_index = self._load_image_set_index(image_set, version)
        self._roidb_handler = self.gt_roidb
        self._classes = open(os.path.join(self._devkit_path, "classes.txt"), "r").readlines()
        self._classes.insert(0, "__background__")

        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

        assert os.path.exists(self._data_path), \
            'REGRAD path does not exist: {}'.format(self._data_path)

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'REGRAD')

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        return self._image_index[i]

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, self._image_set,
                                  index[:-1], index[-1], "rgb.jpg")
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, split, version):
        data_list_file = os.path.join(self._devkit_path, "{}_{}.txt".format(split, version))
        assert os.path.exists(data_list_file), data_list_file + " does not exist"
        return ["".join(s.strip().split("/")) for s in open(data_list_file, "r").readlines()]

    def gt_roidb(self, split, version):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self._cache_path, "regrad_{}_{}_gt.pkl".format(version, split))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                all_annos = pkl.load(fid)
            print('regrad gt loaded from {}'.format(cache_file))
            return all_annos

        gt_roidb = self._load_all_annos(split, version)
        with open(cache_file, 'wb') as fid:
            pkl.dump(gt_roidb, fid, pkl.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_single_anno(self, data_path):
        scene_path = "/".join(data_path.split("/")[:-1])

        anno = {}
        # loading mrt label
        mrt_anno_path = osp.join(scene_path, "mrt.json")
        with open(mrt_anno_path, "r") as f:
            mrt_anno = json.load(f)
        # key is the object id, and value is the parent node list
        mrt_anno = {int(k.split("-")[1]): [int(obj.split("-")[1]) for obj in v] for k, v in mrt_anno.items()}
        num_obj = len(mrt_anno)

        childs = {}
        id_to_ind = dict(zip(mrt_anno.keys(), range(num_obj)))
        ind_to_id = {v: k for k, v in id_to_ind.items()}
        for k in mrt_anno:
            for p in mrt_anno[k]:
                # k is the child of p
                # rel_mat[id_to_ind[k], id_to_ind[p]] = 2
                if p not in childs:
                    childs[p] = []
                childs[p].append(k)

        anno["node_inds"] = []
        anno["child_lists"] = []
        anno["parent_lists"] = []
        for ind in range(num_obj):
            obj_id = ind_to_id[ind]
            anno["node_inds"].append(obj_id)
            anno["child_lists"].append(childs[obj_id] if obj_id in childs else [])
            anno["parent_lists"].append(mrt_anno[obj_id])

        label_file = osp.join(data_path, "info.json")
        with open(label_file, "r") as f:
            other_annos = json.load(f)

        obj_ids = [a['obj_id'] // 10 for a in other_annos]
        id_to_bbox = {a['obj_id'] // 10: a['bbox'] for a in other_annos}

        bboxes = np.zeros((num_obj, 4))
        for i in obj_ids:
            bboxes[id_to_ind[i]] = id_to_bbox[i]
        anno["boxes"] = bboxes

        valid_box_ind = bboxes.sum(-1) > 0
        anno["boxes"] = anno["boxes"][valid_box_ind]
        anno["parent_lists"] = [anno["parent_lists"][ind] for ind, isvalid in enumerate(valid_box_ind) if isvalid]
        anno["child_lists"] = [anno["child_lists"][ind] for ind, isvalid in enumerate(valid_box_ind) if isvalid]
        anno["img_path"] = data_path

        return anno


    def _load_all_annos(self, split, version):
        data_list = self._image_index

        print("Total: {:d}".format(len(data_list)))
        count = 0
        t_b = time.time()
        all_annos = []
        for d in data_list:
            d = d[:-1] + "/" + d[-1]
            data_path = osp.join(self._data_path, d)
            if osp.isdir(data_path):
                all_annos.append(self._load_single_anno(data_path))

            # for visualization
            count += 1
            time_consumed = time.time() - t_b
            avg_time = time_consumed / float(count)
            estimated_finish_time = avg_time * (len(data_list) - count)
            sys.stdout.write(
                "Finished: {:d}/{:d}, Remaining time: {:.1f}s, Consumed time: {:.1f}, Avg time/img: {:.1f}\r".
                    format(count, len(data_list), estimated_finish_time, time_consumed, avg_time))
            sys.stdout.flush()

        return all_annos

    def evaluate_relationships(self, all_rel):
        all_tp = 0
        all_fp = 0
        all_gt = 0

        img_ntp = 0
        img_ntp_dif_objnum = {2:0,3:0,4:0,5:0}
        img_num_dif_objnum = {2:0,3:0,4:0,5:0}
        image_ind_to_roidb_ind = dict(zip(self.image_index, list(range(len(self.image_index)))))

        for im_ind, index in enumerate(self.image_index):
            det_result = all_rel[im_ind]
            anno = self.roidb[image_ind_to_roidb_ind[index]]
            img_num_dif_objnum[anno['boxes'].shape[0]] += 1

            ntp, nfp, ngt = self.do_rel_single_image_eval(det_result, anno)

            all_tp += ntp
            all_fp += nfp
            all_gt += ngt

            if nfp == 0 and ntp == ngt:
                img_ntp += 1
                img_ntp_dif_objnum[anno['boxes'].shape[0]] += 1

        o_rec = float(all_tp) / float(all_gt)
        if all_tp + all_fp > 0:
            o_prec = float(all_tp) / float(all_tp + all_fp)
        else:
            o_prec = 0
        img_prec = float(img_ntp) / len(self.image_index)

        img_prec_dif_objnum = []
        for i in range(2,6):
            img_prec_dif_objnum.append(str(img_ntp_dif_objnum[i]) + '/' + str(img_num_dif_objnum[i]))

        return o_rec, o_prec, img_prec, img_prec_dif_objnum

    def do_rel_single_image_eval(self,det_result, anno):
        gt_bboxes = anno["boxes"]
        gt_classes = anno["gt_classes"]
        num_gt = gt_bboxes.shape[0]
        rel_mat_gt = np.zeros([num_gt, num_gt])
        for o1 in range(num_gt):
            for o2 in range(num_gt):
                ind_o1 = anno['node_inds'][o1]
                ind_o2 = anno['node_inds'][o2]
                if ind_o2 == ind_o1 or rel_mat_gt[o1, o2].item() != 0:
                    continue
                o1_children = anno['child_lists'][o1]
                o1_parents = anno['parent_lists'][o1]
                if ind_o2 in o1_children:
                    # o1 is o2's parent
                    rel_mat_gt[o1, o2] = cfg.VMRN.FATHER
                elif ind_o2 in o1_parents:
                    # o1 is o2's child
                    rel_mat_gt[o1, o2] = cfg.VMRN.CHILD
                else:
                    # o1 and o2 has no relationship
                    rel_mat_gt[o1, o2] = cfg.VMRN.NOREL

        det_bboxes = det_result[0].cpu().numpy()
        det_labels = det_result[1].cpu().numpy()
        det_rel_prob = det_result[2].cpu().numpy()

        # no detected rel, tp and fp is all 0
        if not det_rel_prob.shape[0]:
            return 0, 0, num_gt * (num_gt - 1) /2

        det_rel = np.argmax(det_rel_prob, 1) + 1
        n_det_rel = det_rel_prob.shape[0]

        xmin = np.maximum(gt_bboxes[:, 0:1].T, det_bboxes[:, 0:1])
        ymin = np.maximum(gt_bboxes[:, 1:2].T, det_bboxes[:, 1:2])
        xmax = np.minimum(gt_bboxes[:, 2:3].T, det_bboxes[:, 2:3])
        ymax = np.minimum(gt_bboxes[:, 3:4].T, det_bboxes[:, 3:4])
        w = np.maximum(xmax - xmin + 1., 0.)
        h = np.maximum(ymax - ymin + 1., 0.)
        inters = w * h

        # union
        uni = ((det_bboxes[:, 2:3] - det_bboxes[:, 0:1] + 1.) * (det_bboxes[:, 3:4] - det_bboxes[:, 1:2] + 1.) +
               ((gt_bboxes[:, 2:3] - gt_bboxes[:, 0:1] + 1.)
               * (gt_bboxes[:, 3:4] - gt_bboxes[:, 1:2] + 1.)).T - inters)
        overlaps = inters / uni

        # match bbox ground truth and detections
        match_mat = np.zeros([det_bboxes.shape[0], gt_bboxes.shape[0]])
        for i in range(det_bboxes.shape[0]):
            match_cand_inds = (det_labels[i] == gt_classes)
            match_cand_overlap = overlaps[i] * match_cand_inds
            # decending sort
            ovs = np.sort(match_cand_overlap, 0)
            ovs = ovs[::-1]
            inds = np.argsort(match_cand_overlap, 0)
            inds = inds[::-1]
            for ii, ov in enumerate(ovs):
                if ov > 0.5 and np.sum(match_mat[:,inds[ii]]) == 0:
                    match_mat[i, inds[ii]] = 1
                    break
                elif ov < 0.5:
                    break
        # total number of relationships
        ngt_rel = num_gt * (num_gt - 1) /2
        # true positive and false positive
        tp = 0
        fp = 0
        rel_ind = 0
        for b1 in range(det_bboxes.shape[0]):
            for b2 in range(b1+1, det_bboxes.shape[0]):
                if np.sum(match_mat[b1]) > 0 and np.sum(match_mat[b2])> 0:
                    b1_gt = np.argmax(match_mat[b1])
                    b2_gt = np.argmax(match_mat[b2])
                    rel_gt = rel_mat_gt[b1_gt, b2_gt]
                    if rel_gt == det_rel[rel_ind]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fp += 1
                rel_ind += 1
        return tp, fp, ngt_rel