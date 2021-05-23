# --------------------------------------------------------
# Copyright (c) 2018 Xi'an Jiaotong University
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import uuid
import pickle
from .imdb import imdb
from .pascal_voc import pascal_voc
from .voc_eval import voc_eval
import xml.etree.ElementTree as ET
import scipy
import cv2
import pdb

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_overlaps

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete

class vmrd(pascal_voc):
    def __init__(self, image_set, version = 'compv1', use07metric = True, devkit_path=None):
        imdb.__init__(self, 'vmrd_' + version + image_set)
        self._image_set = image_set
        self._version = version
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'vmrd' + self._version)
        self._classes = ('__background__',  # always index 0
                     'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                     'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                     'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                     'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                     'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}
        self._use07metric = use07metric

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _get_default_path(self):
        """
        Return the default path where Visual Manipulation Realtionship Dataset is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VMRD')

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VMRD/results/vmrdcompv1/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'vmrd' + self._version, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

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

        gt_roidb = [dict(self._load_vmrd_annotation(index).items() +
                         self._load_grasp_annotation(index).items())
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_grasp_annotation(self,index):
        filename = os.path.join(self._data_path, 'Grasps', index + '.txt')
        assert os.path.exists(filename), \
            'Path does not exist: {}'.format(filename)
        with open(filename) as f:
            grasps = [x.strip() for x in f.readlines()]
        ind = np.array([grasp.split(' ')[8] for grasp in grasps], dtype=np.float32)
        grasp_mat = np.array([grasp.split(' ')[:8] for grasp in grasps], dtype=np.float32)
        return {
            'grasps': grasp_mat,
            'grasp_inds':ind
        }

    def _load_vmrd_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        nodeinds = np.zeros(num_objs, dtype=np.uint16)
        father_list = []
        child_list = []
        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            nodeind = int(obj.find('index').text)
            fathernodes = obj.find('father').findall('num')
            fathers = [int(f.text) for f in fathernodes]
            childnodes = obj.find('children').findall('num')
            children = [int(f.text) for f in childnodes]
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            if x1>= x2 or y1 >= y2:
                print(filename)
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            nodeinds[ix] = nodeind
            father_list.append(np.array(fathers, dtype=np.uint16))
            child_list.append(np.array(children, dtype=np.uint16))

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'nodeinds': nodeinds,
                'fathers': father_list,
                'children': child_list,
                'rotated': 0}

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VMRD results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'vmrd' + self._version,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'vmrd' + self._version,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._use07metric) else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
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

    def evaluate_relationships(self, all_rel):
        all_tp = 0
        all_fp = 0
        all_gt = 0

        img_ntp = 0
        img_ntp_dif_objnum = {2:0,3:0,4:0,5:0}
        img_num_dif_objnum = {2:0,3:0,4:0,5:0}

        infos = []
        for im_ind, index in enumerate(self.image_index):
            det_result = all_rel[im_ind]
            anno = self._load_vmrd_annotation(index)
            img_num_dif_objnum[anno['boxes'].shape[0]] += 1

            ntp, nfp, ngt, info = self.do_rel_single_image_eval(det_result, anno)
            infos += info

            all_tp += ntp
            all_fp += nfp
            all_gt += ngt

            if nfp == 0 and ntp == ngt:
                img_ntp += 1
                img_ntp_dif_objnum[anno['boxes'].shape[0]] += 1

        o_rec = float(all_tp) / float(all_gt)
        o_prec = float(all_tp) / float(all_tp + all_fp)
        img_prec = float(img_ntp) / len(self.image_index)

        img_prec_dif_objnum = []
        for i in range(2,6):
            img_prec_dif_objnum.append(str(img_ntp_dif_objnum[i]) + '/' + str(img_num_dif_objnum[i]))
        import pickle as pkl
        with open("rel_density_estimation.pkl", "wb") as f:
            pkl.dump(infos, f)
        return o_rec, o_prec, img_prec, img_prec_dif_objnum

    def do_rel_single_image_eval(self,det_result, anno):
        # info is for collecting all detection data and the corresponding ground truth
        info = []
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
                o1_fathers = anno['fathers'][o1]
                if ind_o2 in o1_children:
                    # o1 is o2's father
                    rel_mat_gt[o1, o2] = cfg.VMRN.FATHER
                elif ind_o2 in o1_fathers:
                    # o1 is o2's child
                    rel_mat_gt[o1, o2] = cfg.VMRN.CHILD
                else:
                    # o1 and o2 has no relationship
                    rel_mat_gt[o1, o2] = cfg.VMRN.NOREL

        det_bboxes = np.array(det_result[0])
        det_labels = np.array(det_result[1])
        det_rel_prob = np.array(det_result[2])

        # no detected rel, tp and fp is all 0
        if not det_rel_prob.shape[0]:
            return 0, 0, num_gt * (num_gt - 1) /2, []

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
                    info.append({"gt": rel_gt, "det_score": det_rel_prob[rel_ind]})
                    if rel_gt == det_rel[rel_ind]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fp += 1
                rel_ind += 1
        return tp, fp, ngt_rel, info

    def evaluate_multigrasp_detections(self, all_boxes):
        print('-----------------------------------------------------')
        print('Computing results of Multi-Grasp Detection.')
        print('-----------------------------------------------------')
        print('Evaluating MR-FPPI...')
        # 100 points MissRate-FPPI
        grasp_MRFPPI, APs = self.evaluate_multigrasp_MRFPPI(all_boxes)
        print('Evaluating Completed...')
        print('Log-Average Miss Rate Results...')
        mean_grasp_MRFPPI = []
        points = -np.arange(9).astype(np.float32) / 4 # 9 points in [-2, 0]
        keypoints = np.array([-1, 0])
        key_point_MRFPPI = [[] for i in range(keypoints.size)]

        for i, MF in enumerate(grasp_MRFPPI):
            assert MF[1, :][-1] >= 0, "box threshold is too high."
            cur_mean = 0.

            for p in points:
                miss_rate_ind = np.cumsum(MF[1, :] < p).max() - 1
                if miss_rate_ind == -1:
                    cur_mean += 1
                else:
                    cur_mean += MF[0, :][miss_rate_ind]
            mean_grasp_MRFPPI.append(cur_mean / len(points))

            for j,p in enumerate(keypoints):
                miss_rate_ind = np.cumsum(MF[1, :] < p).max() - 1
                key_point_MRFPPI[j].append(MF[0, :][miss_rate_ind])

            print("Log-Average Miss Rate for All Objects: %.4f" % (cur_mean/len(points)))

        key_point_MRFPPI = [np.mean(np.array(a)) for a in key_point_MRFPPI]

        for i,a in enumerate(key_point_MRFPPI):
            print("Miss Rate for All Objects (FPPI = %.1f): %.4f" % (keypoints[i],a))

        return grasp_MRFPPI, mean_grasp_MRFPPI, key_point_MRFPPI

    def evaluate_multigrasp_MRFPPI(self, all_boxes):
        MRFPPI = []
        AP = []
        boxthresh = 0.5
        gr_jacth = 0.25
        gr_angth = 30
        cls_dets_all = []
        GTall = 0.
        for cls in range(1, len(self.classes)):
            GT = 0.
            # NUM_IMG = 0.
            # all detection results across all the test images.
            for im_ind, index in enumerate(self.image_index):
                if len(all_boxes[cls][im_ind]):
                    boxanno = self._load_vmrd_annotation(index)
                    all_boxes[cls][im_ind][0] = np.concatenate([all_boxes[cls][im_ind][0],
                                                                np.zeros((all_boxes[cls][im_ind][0].shape[0], 1))
                                                                ],axis=1)
                    if cls not in boxanno['gt_classes']:
                        continue
                    else:
                        # NUM_IMG += 1
                        boxannoindex = boxanno['nodeinds'][boxanno['gt_classes'] == cls]
                        boxanno = boxanno['boxes'][boxanno['gt_classes'] == cls]
                        GT += boxanno.shape[0]

                        graspanno = self._load_grasp_annotation(index)
                        gt_grasp = self.points2label(graspanno['grasps'])
                        gt_grasp_inds = graspanno['grasp_inds']
                        boxdets = all_boxes[cls][im_ind][0]

                        sort_inds = np.argsort(boxdets[:, 4])[::-1]
                        boxdets = boxdets[sort_inds]
                        graspdets = self.points2label(all_boxes[cls][im_ind][1])
                        graspdets = graspdets[sort_inds]
                        if len(graspdets.shape)!= 2:
                            assert 0, "only support top1 grasp evaluation."

                        # N_gt x 1 - 1 x N_det = N_gt x N_det
                        ixmin = np.maximum(boxanno[:, 0:1], np.expand_dims(boxdets[:, 0],0))
                        iymin = np.maximum(boxanno[:, 1:2], np.expand_dims(boxdets[:, 1],0))
                        ixmax = np.minimum(boxanno[:, 2:3], np.expand_dims(boxdets[:, 2],0))
                        iymax = np.minimum(boxanno[:, 3:4], np.expand_dims(boxdets[:, 3],0))
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih
                        # union
                        uni = ((np.expand_dims(boxdets[:, 2] - boxdets[:, 0], 0) + 1.) *
                               (np.expand_dims(boxdets[:, 3] - boxdets[:, 1], 0) + 1.) +
                               (boxanno[:, 2:3] - boxanno[:, 0:1] + 1.) *
                               (boxanno[:, 3:4] - boxanno[:, 1:2] + 1.) - inters)
                        IoUs = inters / uni

                        for i in range(boxanno.shape[0]):
                            flag_assign = False
                            gt_index = boxannoindex[i]
                            current_gtgrasp = gt_grasp[gt_grasp_inds == gt_index]
                            for j in range(boxdets.shape[0]):
                                # this detected box os already assigned to a ground truth
                                if all_boxes[cls][im_ind][0][j, -1] == 1:
                                    continue
                                if IoUs[i][j] > boxthresh:
                                    current_detgrasp = graspdets[j]
                                    for gtgr in range(current_gtgrasp.shape[0]):
                                        gr_ov = self._jaccard_overlap(current_detgrasp, current_gtgrasp[gtgr])
                                        angdiff = np.abs(current_detgrasp[4] - current_gtgrasp[gtgr][4]) % 180

                                        if gr_ov > gr_jacth and ((angdiff < gr_angth) or (angdiff > 180 - gr_angth) > 0):
                                            all_boxes[cls][im_ind][0][j, -1] = 1
                                            flag_assign = True
                                            break
                                    if flag_assign:
                                        break

            cls_dets = []
            for i in range(len(all_boxes[cls])):
                if len(all_boxes[cls][i]):
                    cls_dets.append(all_boxes[cls][i][0])
            cls_dets = np.concatenate(cls_dets, axis = 0)
            cls_dets_all.append(cls_dets)
            GTall += GT
            sort_inds = np.argsort(cls_dets[:, -2])
            cls_dets = cls_dets[sort_inds[::-1]]
            TP = cls_dets[:, -1]
            FP = np.cumsum(1 - TP)
            TP = np.cumsum(TP)
            Miss = GT - TP
            rec = TP / GT
            prec = TP / (TP + FP)
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
            print("AP with grasp detection for %s: %.4f" % (self._classes[cls], ap))
            AP.append(ap)

            #MRFPPI.append(np.concatenate([
            #    np.expand_dims(Miss / GT, 0),
            #    np.log(np.expand_dims(FP / len(self.image_index), 0))
            #], axis = 0))

        AP = np.array(AP)
        print("mAP with grasp: %.4f" % (np.mean(AP[np.nonzero(1-np.isnan(AP))])))

        cls_dets_all = np.concatenate(cls_dets_all, axis = 0)
        sort_inds = np.argsort(cls_dets_all[:, -2])
        cls_dets_all = cls_dets_all[sort_inds[::-1]]
        TP = cls_dets_all[:, -1]
        FP = np.cumsum(1 - TP)
        TP = np.cumsum(TP)
        Miss = GTall - TP
        MRFPPI.append(np.concatenate([
            np.expand_dims(Miss / GTall, 0),
            np.log(np.expand_dims(FP / len(self.image_index), 0)) / np.log(10.)
        ], axis=0))

        return MRFPPI, AP

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
