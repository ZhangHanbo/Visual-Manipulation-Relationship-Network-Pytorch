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
import pickle

from torch.utils.data import Dataset
from .imdb import imdb
from model.utils.config import cfg

class regrad(imdb):
    def __init__(self, image_set, version = 'supermini', devkit_path=None):
        imdb.__init__(self, 'regrad_' + version + "_" + image_set)
        self._image_set = image_set
        self._version = version
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, image_set)
        self._cache_path = os.path.join(cfg.DATA_DIR, "cache")
        self._image_index = self._load_image_set_index(image_set, version)
        self._roidb_handler = self.gt_roidb
        self._classes = open(os.path.join(self._devkit_path, "classes.txt"), "r").readlines()
        self._classes.insert(0, "__background__")
        for i in range(len(self._classes)):
            self._classes[i] = self._classes[i].strip()
        self._classes_to_clsid = {v: i for i, v in enumerate(self._classes)}

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
        image_path = os.path.join(self._data_path, index[:-1], index[-1], "rgb.jpg")
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, split, version):
        data_list_file = os.path.join(self._devkit_path, "{}_{}.txt".format(split, version))
        assert os.path.exists(data_list_file), data_list_file + " does not exist"
        return ["".join(s.strip().split("/")) for s in open(data_list_file, "r").readlines()]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self._cache_path, "{}_gt.pkl".format(self._name))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                all_annos = pkl.load(fid)
            print('regrad gt loaded from {}'.format(cache_file))
            return all_annos

        gt_roidb = self._load_all_annos()
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

        label_file = osp.join(data_path, "info.json")
        with open(label_file, "r") as f:
            other_annos = json.load(f)

        obj_ids = [a['obj_id'] // 10 for a in other_annos]
        id_to_bbox = {a['obj_id'] // 10: a['bbox'] for a in other_annos}
        id_to_name = {a['obj_id'] // 10: a['model_name'] for a in other_annos}

        bboxes = np.zeros((num_obj, 4))
        for i in obj_ids:
            bboxes[id_to_ind[i]] = id_to_bbox[i]

        gt_classes = np.zeros((num_obj), dtype=np.int32)
        for i in obj_ids:
            gt_classes[id_to_ind[i]] = self._classes_to_clsid[id_to_name[i]]

        valid_box_mask = bboxes.sum(-1) > 0
        valid_box_inds = valid_box_mask.nonzero()[0]
        valid_box_ids = [ind_to_id[ind] for ind in valid_box_inds]
        anno["boxes"] = bboxes[valid_box_mask]
        anno["gt_classes"] = gt_classes[valid_box_mask]
        anno["img_path"] = data_path
        anno["flipped"] = False
        anno["rotated"] = 0

        anno["nodeinds"] = []
        anno["fathers"] = []
        anno["children"] = []
        for ind in range(num_obj):
            obj_id = ind_to_id[ind]
            if obj_id in valid_box_ids:
                anno["nodeinds"].append(obj_id)
                anno["children"].append([c for c in childs[obj_id] if c in valid_box_ids] if obj_id in childs else [])
                anno["fathers"].append([p for p in mrt_anno[obj_id] if p in valid_box_ids])

        anno["nodeinds"] = np.array(anno["nodeinds"], dtype=np.uint16)
        anno["fathers"] = [np.array(arr, dtype=np.uint16) for arr in anno["fathers"]]
        anno["children"] = [np.array(arr, dtype=np.uint16) for arr in anno["children"]]

        return anno


    def _load_all_annos(self):
        data_list = self._image_index

        print("Total: {:d}".format(len(data_list)))
        all_annos = []
        for i, d in enumerate(data_list):
            sys.stdout.write("Processed/Total: {}/{}\r".format(i, len(data_list)))
            sys.stdout.flush()
            d = d[:-1] + "/" + d[-1]
            data_path = osp.join(self._data_path, d)
            if osp.isdir(data_path):
                all_annos.append(self._load_single_anno(data_path))

        return all_annos


    def parse_objdet_anno(self, file_name):
        objects = []
        with open(file_name, "r") as f:
            other_annos = json.load(f)

        obj_ids = [a['obj_id'] // 10 for a in other_annos]
        id_to_bbox = {a['obj_id'] // 10: a['bbox'] for a in other_annos}
        id_to_name = {a['obj_id'] // 10: a['model_name'] for a in other_annos}

        for obj_id in obj_ids:
            if id_to_bbox[obj_id] is not None:
                obj_struct = {}
                obj_struct["name"] = id_to_name[obj_id]
                obj_struct["bbox"] = id_to_bbox[obj_id]
                objects.append(obj_struct)
        return objects


    def _get_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._cache_path, 'results', 'REGRAD', self._image_set)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} REGRAD results file'.format(cls))
            filename = self._get_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k , 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _ap(self, rec, prec, use_07_metric=False):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _regrad_objdet_eval(self,
                            detpath,
                            annopath,
                            imagesetfile,
                            classname,
                            cachedir,
                            ovthresh=0.5,
                            use_07_metric=False
                            ):

        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, '%s_annots.pkl' %
                                 os.path.join(imagesetfile.split("/")[-1][:-4]))
        print(cachefile)
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile):
            # load annotations
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_objdet_anno(annopath.format(imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                try:
                    recs = pickle.load(f)
                except:
                    recs = pickle.load(f, encoding='bytes')

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            npos = npos + len(R)
            class_recs[imagename.replace("/", "")] = {'bbox': bbox,
                                     'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self._ap(rec, prec, use_07_metric)

        return rec, prec, ap

    def _do_python_eval(self):
        annopath = os.path.join(
            self._devkit_path,
            self._image_set,
            '{:s}',
            'info.json')
        imagesetfile = os.path.join(
            self._devkit_path,
            self._image_set + '_' + self._version + '.txt')
        cachedir = os.path.join(self._cache_path, 'annotations_cache', 'REGRAD')
        outdir = os.path.join(self._cache_path, 'results', 'REGRAD')
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            rec, prec, ap = self._regrad_objdet_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=True)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(outdir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        return np.mean(aps)

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        map = self._do_python_eval()
        return map

    def evaluate_relationships(self, all_rel):
        all_tp = 0
        all_fp = 0
        all_gt = 0

        f_info = {"fp": 0, "tp": 0, "gt": 0}
        c_info = {"fp": 0, "tp": 0, "gt": 0}
        n_info = {"fp": 0, "tp": 0, "gt": 0}

        img_ntp = 0
        # img_ntp_dif_objnum = {2:0,3:0,4:0,5:0}
        img_ntp_dif_objnum = {i : 0 for i in range(2, 20)}
        # img_num_dif_objnum = {2:0,3:0,4:0,5:0}
        img_num_dif_objnum = {i: 0 for i in range(2, 20)}
        image_ind_to_roidb_ind = dict(zip(self.image_index, list(range(len(self.image_index)))))

        for im_ind, index in enumerate(self.image_index):
            det_result = all_rel[im_ind]
            anno = self.roidb[image_ind_to_roidb_ind[index]]
            img_num_dif_objnum[anno['boxes'].shape[0]] += 1

            ntp, nfp, ngt, verbals = self.do_rel_single_image_eval(det_result, anno)

            all_tp += ntp
            all_fp += nfp
            all_gt += ngt

            f_info["tp"] += verbals[0][0]
            f_info["fp"] += verbals[0][1]
            f_info["gt"] += verbals[0][2]
            c_info["tp"] += verbals[1][0]
            c_info["fp"] += verbals[1][1]
            c_info["gt"] += verbals[1][2]
            n_info["tp"] += verbals[2][0]
            n_info["fp"] += verbals[2][1]
            n_info["gt"] += verbals[2][2]

            if nfp == 0 and ntp == ngt:
                img_ntp += 1
                img_ntp_dif_objnum[anno['boxes'].shape[0]] += 1

        print("Parent Recall and Precision: {}; {}".format(float(f_info["tp"]) / float(f_info["gt"]),
                                    float(f_info["tp"]) / (float(f_info["tp"]) + float(f_info["fp"]))))
        print("Child Recall and Precision: {}; {}".format(float(c_info["tp"]) / float(c_info["gt"]),
                                    float(c_info["tp"]) / (float(c_info["tp"]) + float(c_info["fp"]))))
        print("None Recall and Precision: {}; {}".format(float(n_info["tp"]) / float(n_info["gt"]),
                                    float(n_info["tp"]) / (float(n_info["tp"]) + float(n_info["fp"]))))

        o_rec = float(all_tp) / float(all_gt)
        if all_tp + all_fp > 0:
            o_prec = float(all_tp) / float(all_tp + all_fp)
        else:
            o_prec = 0
        img_prec = float(img_ntp) / len(self.image_index)

        img_prec_dif_objnum = []
        for i in range(2,20):
            img_prec_dif_objnum.append(str(img_ntp_dif_objnum[i]) + '/' + str(img_num_dif_objnum[i]))

        return o_rec, o_prec, img_prec, img_prec_dif_objnum

    def do_rel_single_image_eval(self,det_result, anno):
        gt_bboxes = anno["boxes"]
        gt_classes = anno["gt_classes"]
        num_gt = gt_bboxes.shape[0]
        rel_mat_gt = np.zeros([num_gt, num_gt])
        for o1 in range(num_gt):
            for o2 in range(num_gt):
                ind_o1 = anno['nodeinds'][o1]
                ind_o2 = anno['nodeinds'][o2]
                if ind_o2 == ind_o1 or rel_mat_gt[o1, o2].item() != 0:
                    continue
                o1_children = anno['children'][o1]
                o1_parents = anno['fathers'][o1]
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

        tp_f = 0
        fp_f = 0
        gt_f = 0

        tp_c = 0
        fp_c = 0
        gt_c = 0

        tp_n = 0
        fp_n = 0
        gt_n = 0

        for b1 in range(det_bboxes.shape[0]):
            for b2 in range(b1+1, det_bboxes.shape[0]):
                if np.sum(match_mat[b1]) > 0 and np.sum(match_mat[b2])> 0:
                    b1_gt = np.argmax(match_mat[b1])
                    b2_gt = np.argmax(match_mat[b2])
                    rel_gt = rel_mat_gt[b1_gt, b2_gt]

                    if rel_gt == 1:
                        gt_f += 1
                    elif rel_gt == 2:
                        gt_c += 1
                    else:
                        gt_n += 1

                    if rel_gt == det_rel[rel_ind]:
                        tp += 1
                        if rel_gt == 1:
                            tp_f += 1
                        elif rel_gt == 2:
                            tp_c += 1
                        else:
                            tp_n += 1
                    else:
                        fp += 1
                        if det_rel[rel_ind] == 1:
                            fp_f += 1
                        elif det_rel[rel_ind] == 2:
                            fp_c += 1
                        else:
                            fp_n += 1
                else:
                    fp += 1
                rel_ind += 1
        return tp, fp, ngt_rel, ((tp_f, fp_f, gt_f), (tp_c, fp_c, gt_c), (tp_n, fp_n, gt_n))