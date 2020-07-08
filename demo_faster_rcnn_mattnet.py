import numpy as np
import os
import torch
from torch.autograd import Variable
import argparse
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from model.FasterRCNN import fasterRCNN
from model.utils.config import read_cfgs, cfg
from model.utils.blob import prepare_data_batch_from_cvimage
from model.utils.net_utils import rel_prob_to_mat, find_all_paths, create_mrt, objdet_inference
from roi_data_layer.roidb import combined_roidb
from model.utils.data_viewer import dataViewer
from model.rpn.bbox_transform import bbox_xy_to_xywh
from model.MattNet import MattNetV2

# import sys
# print(sys.path)
# import model.mattnet.tools._init_paths
# print(sys.path)

# from mrcn import inference
# from model.nms_wrapper import nms

import cv2
from torchsummary import summary

# ------------- Static Functions --------------
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


class fasterRCNNMattNetDemo(object):
    def __init__(self, args, model_dir):
        # init RCNN
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        # load trained model
        load_name = os.path.join(model_dir, args.frame + '_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                                args.checkpoint))
        print(load_name)
        trained_model = torch.load(load_name)
        # init VMRN
        _,_,_,_,cls_list = combined_roidb(args.imdbval_name, training=False)
        self.RCNN = fasterRCNN(len(cls_list), class_agnostic=trained_model['class_agnostic'], feat_name=args.net,
                    feat_list=('conv' + conv_num,), pretrained=True)
        self.RCNN.create_architecture()
        self.RCNN.load_state_dict(trained_model['model'])

        # print(self.RCNN)

        if args.cuda:
            self.RCNN.cuda()
            self.cuda = True
        self.RCNN.eval()
        # init classes
        self.classes = cls_list
        self.class_to_ind = dict(zip(self.classes, xrange(len(cls_list))))
        self.ind_to_class = dict(zip(xrange(len(cls_list)), self.classes))

        # init data viewer
        self.data_viewer = dataViewer(self.classes)

        # init Mattnet
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
        parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
        parser.add_argument('--model_id', type=str, default='mrcn_cmr_with_st', help='model id name')
        args = parser.parse_args('')
        self.mattnet = MattNetV2(args)

        # !!! For testing only
        # load mrcn 
        # tic = time.time()
        # args.imdb_name = self.model_opt['imdb_name']
        # args.net_name = self.model_opt['net_name']
        # args.tag = self.model_opt['tag']
        # args.iters = self.model_opt['iters']
        # self.mrcn = inference.Inference(args)
        # self.imdb = self.mrcn.imdb
        # print('Mask R-CNN: imdb[%s], tag[%s], id[%s_mask_rcnn_iter_%s] loaded in %.2f seconds.' % \
        #     (args.imdb_name, args.tag, args.net_name, args.iters, time.time()-tic))

    """
    def cls_to_detections(self, scores, boxes, nms_thresh, conf_thresh):
        # run nms and threshold for each class detection
        cls_to_dets = {}
        num_dets = 0
        for cls_ind, class_name in enumerate(self.imdb.classes[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind+1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                                cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(torch.from_numpy(dets), nms_thresh)
            dets = dets[keep.numpy(), :]
            inds = np.where(dets[:, -1] >= conf_thresh)[0]
            dets = dets[inds, :]
            cls_to_dets[class_name] = dets
            num_dets += dets.shape[0]
        return cls_to_dets, num_dets

    def mrcn_forward_image(self, img_path, nms_thresh=.3, conf_thresh=.65, true_bboxes=None):
        '''
        Arguments:
        - img_path   : path to image
        - nms_thresh : nms threshold 
        - conf_thresh: confidence threshold [0,1]
        Return "data" is a dict of
        - det_ids: list of det_ids, order consistent with dets and masks
        - dets   : [{det_id, box, category_name, category_id, score}], box is [xywh] and category_id is coco_cat_id
        - masks  : ndarray (n, im_h, im_w) uint8 [0,1]
        - Feats  :
        - pool5     : Variable cuda (n, 1024, 7, 7)
        - fc7       : Variable cuda (n, 2048, 7, 7)
        - lfeats    : Variable cuda (n, 5)
        - dif_lfeats: Variable cuda (n, 5*topK)
        - cxt_fc7   : Variable cuda (n, topK, 2048)
        - cxt_lfeats: Variable cuda (n, topK, 5)
        - cxt_det_ids : list of [surrounding_det_ids] for each det_id
        '''
        # read image
        im = imread(img_path)

        # 1st step: detect objects
        scores, boxes = self.mrcn.predict(img_path)
        print(boxes)
        boxes = true_bboxes # !!! For testing only. replace boxes. 

        # get head feats, i.e., net_conv 
        net_conv = self.mrcn.net._predictions['net_conv']  # Variable cuda (1, 1024, h, w)
        im_info = self.mrcn.net._im_info  # [[H, W, im_scale]]

        # get cls_to_dets, class_name -> [xyxys] which is (n, 5)
        cls_to_dets, num_dets = self.cls_to_detections(scores, boxes, nms_thresh, conf_thresh)
        # make sure num_dets > 0
        thresh = conf_thresh
        while num_dets == 0:
            thresh -= 0.1
            cls_to_dets, num_dets = self.cls_to_detections(scores, boxes, nms_thresh, thresh)

        # add to dets
        dets = []
        det_id = 0
        for category_name, detections in cls_to_dets.items():
            # detections: list of (n, 5), [xyxyc]
            for detection in detections:
                x1, y1, x2, y2, sc = detection
                det = {'det_id': det_id, 
                    'box': [x1, y1, x2-x1+1, y2-y1+1],
                    'category_name': category_name,
                    'category_id': self.imdb._class_to_coco_cat_id[category_name],
                    'score': sc}
                dets += [det]
                det_id += 1
        Dets = {det['det_id']: det for det in dets}
        det_ids = [det['det_id'] for det in dets]

        # 2nd step: get masks
        boxes = xywh_to_xyxy(np.array([det['box'] for det in dets]))  # xyxy (n, 4) ndarray
        labels = np.array([self.imdb._class_to_ind[det['category_name']] for det in dets])
        # mask_prob = self.mrcn.net._predict_masks_from_boxes_and_labels(net_conv, boxes*im_info[0][2], labels)
        # mask_prob = mask_prob.data.cpu().numpy()
        # masks = recover_masks(mask_prob, boxes, im.shape[0], im.shape[1])  # (N, ih, iw) uint8 [0-255]
        # masks = (masks > 122.).astype(np.uint8)  # (N, ih, iw) uint8 [0,1]

        # 3rd step: compute features
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)  # (n, 1024, 7, 7), (n, 2048, 7, 7)
        lfeats = self.compute_lfeats(det_ids, Dets, im)
        dif_lfeats = self.compute_dif_lfeats(det_ids, Dets)
        cxt_fc7, cxt_lfeats, cxt_det_ids = self.fetch_cxt_feats(det_ids, Dets, fc7, self.model_opt)

        # move to Variable cuda
        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())

        # return
        data = {}
        data['det_ids'] = det_ids
        data['dets'] = dets
        # data['masks'] = masks
        data['cxt_det_ids'] = cxt_det_ids
        data['Feats'] = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats, 
                        'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats}
        return data
    """

    def fasterRCNN_forward_image(self, image, save_res=False, id=""):
        data_batch = prepare_data_batch_from_cvimage(image, is_cuda=True)
        # result = (rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label)
        result = self.RCNN(data_batch)
        rois = result[0][0][:, 1:5].data
        cls_prob = result[1][0].data
        bbox_pred = result[2][0].data
        obj_boxes = objdet_inference(cls_prob, bbox_pred, data_batch[1][0], rois,
                                     class_agnostic=False, for_vis=True, recover_imscale=True)
        obj_classes = obj_boxes[:, -1]
        obj_boxes = obj_boxes[:, :-1]
        num_box = obj_boxes.shape[0]
        obj_cls_name = []
        for cls in obj_classes:
            obj_cls_name.append(self.ind_to_class[cls])

        if save_res:
            obj_det_img = self.data_viewer.draw_objdet(image.copy(),
                np.concatenate((obj_boxes, np.expand_dims(obj_classes, 1)), axis=1), o_inds=list(range(num_box)))
            cv2.imwrite("images/" + id + "object_det.png", obj_det_img)

        # add to dets
        dets = []
        det_id = 0
        for idx, obj_class in enumerate(obj_classes):
            # detections: list of (n, 5), [xyxyc]
            x1, y1, x2, y2 = obj_boxes[idx]  # TODO check xywh or xyxy
            det = {'det_id': det_id,
                   'box': [x1, y1, x2-x1+1, y2-y1+1],
                   # 'category_name': category_name,
                   'category_id': obj_class,
                   # 'score': sc
                   }
            dets += [det]
            det_id += 1
        Dets = {det['det_id']: det for det in dets}
        det_ids = [det['det_id'] for det in dets]

        # Compute features
        # (n, 1024, 7, 7), (n, 2048, 7, 7) TODO
        print(obj_boxes)
        obj_boxes = torch.from_numpy(obj_boxes).cuda()
        obj_boxes = obj_boxes.unsqueeze(0)
        print(obj_boxes)
        pool5, fc7 = self.RCNN.box_to_spatial_fc7(obj_boxes)
        print('pool5 shape {}'.format(pool5.shape))
        print('fc7 shape {}'.format(fc7.shape))
        lfeats = self.compute_lfeats(det_ids, Dets, image) # location feature against the image
        dif_lfeats = self.compute_dif_lfeats(det_ids, Dets) # location feature against five objects of the same category
        cxt_fc7, cxt_lfeats, cxt_det_ids = self.fetch_cxt_feats(det_ids, Dets, fc7)  # relational feature

        # move to Variable cuda
        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())

        # return
        data = {}
        data['det_ids'] = det_ids
        data['dets'] = dets
        # data['masks'] = masks
        data['cxt_det_ids'] = cxt_det_ids
        data['Feats'] = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                         'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats}
        return data

    def forward_process(self, image, expr, save_res=False, id="", img_path=""):
        img_data = self.fasterRCNN_forward_image(image, save_res, id)
        # obj_boxes = self.fasterRCNN_forward_image(image, save_res, id)
        # img_data = self.mrcn_forward_image(img_path, true_bboxes=obj_boxes)
 
        entry = self.mattnet.comprehend(img_data, expr)
        print('overall_score: {}'.format(entry['overall_scores']))
        print('module_score: {}'.format(entry['module_scores']))
        tokens = expr.split()
        print('sub(%.2f):' % entry['weights'][0], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['sub_attn'])]))
        print('loc(%.2f):' % entry['weights'][1], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['loc_attn'])]))
        print('rel(%.2f):' % entry['weights'][2], ''.join(['(%s,%.2f)'% (tokens[i], s) for i, s in enumerate(entry['rel_attn'])]))
        # predict attribute on the predicted object
        print(entry['pred_atts'])
        fig = plt.figure()
        self.show_boxes(img_path, xywh_to_xyxy(np.vstack([entry['pred_box']])), ['blue'], texts=None)
        plt.show()

    def show_boxes(self, img_path, boxes, colors, texts=None, masks=None):
        # boxes [[xyxy]]
        img = imread(img_path)
        plt.imshow(img)
        ax = plt.gca()
        for k in range(boxes.shape[0]):
            box = boxes[k]
            xmin, ymin, xmax, ymax = list(box)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[k]
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            if texts is not None:
                ax.text(xmin, ymin, texts[k], bbox={'facecolor':color, 'alpha':0.5})
        # show mask
        if masks is not None:
            for k in range(len(masks)):
                mask = masks[k]
                m = np.zeros( (mask.shape[0], mask.shape[1], 3))
                m[:,:,0] = 0; m[:,:,1] = 0; m[:,:,2] = 1.
                ax.imshow(np.dstack([m*255, mask*255*0.4]).astype(np.uint8)) 

    def compute_lfeats(self, det_ids, Dets, im):
        '''
        object's location in image
        '''
        # Compute (n, 5) lfeats for given det_ids
        lfeats = np.empty((len(det_ids), 5), dtype=np.float32)
        for ix, det_id in enumerate(det_ids):
            det = Dets[det_id]
            x, y, w, h = det['box']
            ih, iw = im.shape[0], im.shape[1]
            lfeats[ix] = np.array(
                [[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32)
        return lfeats

    def fetch_neighbour_ids(self, ref_det_id, Dets):
        '''
        For a given ref_det_id, we return
        - st_det_ids: same-type neighbouring det_ids (not including itself)
        - dt_det_ids: different-type neighbouring det_ids
        Ordered by distance to the input det_id
        '''
        ref_det = Dets[ref_det_id]
        x, y, w, h = ref_det['box']
        rx, ry = x+w/2, y+h/2

        def compare(det_id0, det_id1):
            x, y, w, h = Dets[det_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x, y, w, h = Dets[det_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer --> former
            if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
                return -1
            else:
                return 1

        det_ids = list(Dets.keys())  # copy in case the raw list is changed
        det_ids = sorted(det_ids, cmp=compare)
        st_det_ids, dt_det_ids = [], []
        for det_id in det_ids:
            if det_id != ref_det_id:
                if Dets[det_id]['category_id'] == ref_det['category_id']:
                    st_det_ids += [det_id]
                else:
                    dt_det_ids += [det_id]
        return st_det_ids, dt_det_ids

    def compute_dif_lfeats(self, det_ids, Dets, topK=5):
        '''
        object's location wrt to other objects of the same category
        '''
        # return ndarray float32 (#det_ids, 5*topK)
        dif_lfeats = np.zeros((len(det_ids), 5*topK), dtype=np.float32)
        for i, ref_det_id in enumerate(det_ids):
            # reference box
            rbox = Dets[ref_det_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2] / \
                2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            # candidate boxes
            st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id, Dets)
            for j, cand_det_id in enumerate(st_det_ids[:topK]):
                cbox = Dets[cand_det_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = \
                    np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx) / \
                            rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats

    def fetch_cxt_feats(self, det_ids, Dets, spatial_fc7, topK=5, with_st=1):
        '''
        object's location wrt to other objects of the different category

        Arguments:
        - det_ids    : list of det_ids
        - Dets       : each det is {det_id, box, category_id, category_name}
        - spatial_fc7: (#det_ids, 2048, 7, 7) Variable cuda
        Return
        - cxt_feats  : Variable cuda (#det_ids, topK, feat_dim)
        - cxt_lfeats : ndarray (#det_ids, topK, 5)
        - cxt_det_ids: [[det_id]] of size (#det_ids, topK), padded with -1
        Note we use neighbouring objects for computing context objects, zeros padded.
        '''
        fc7 = spatial_fc7.mean(3).mean(2)  # (n, 2048)
        cxt_feats = Variable(spatial_fc7.data.new(
            len(det_ids), topK, 2048).zero_())
        cxt_lfeats = np.zeros((len(det_ids), topK, 5), dtype=np.float32)
        cxt_det_ids = -np.ones((len(det_ids), topK),
                               dtype=np.int32)  # (#det_ids, topK)
        for i, ref_det_id in enumerate(det_ids):
            # reference box
            rbox = Dets[ref_det_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2] / \
                2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            # candidate boxes
            st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id, Dets)
            if with_st > 0:
                cand_det_ids = dt_det_ids + st_det_ids
            else:
                cand_det_ids = dt_det_ids
            cand_det_ids = cand_det_ids[:topK]
            for j, cand_det_id in enumerate(cand_det_ids):
                cand_det = Dets[cand_det_id]
                cbox = cand_det['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                cxt_lfeats[i, j, :] = np.array(
                    [(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
                cxt_feats[i, j, :] = fc7[det_ids.index(cand_det_id)]
                cxt_det_ids[i, j] = cand_det_id
        cxt_det_ids = cxt_det_ids.tolist()
        return cxt_feats, cxt_lfeats, cxt_det_ids

if __name__ == '__main__':
    # we need to read configs of VMRN that were used in training and also need to be used in this demo
    args = read_cfgs()
    demo = fasterRCNNMattNetDemo(args, os.path.join(args.save_dir + "/" + args.dataset + "/" + args.net))
    while True:
        image_id = raw_input('Image ID: ').lower()
        if image_id == 'break':
            break
        expr = raw_input('Expression:').lower()
        if expr == 'break':
            break
        # read cv image
        test_img_path = os.path.join('images', image_id + ".jpg")
        cv_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        # VMRN forward process
        demo.forward_process(cv_img, expr, save_res=True, id = image_id, img_path=test_img_path)

