import numpy as np
import os
import torch

from model.FasterRCNN import fasterRCNN
from model.utils.config import read_cfgs, cfg
from model.utils.blob import prepare_data_batch_from_cvimage
from model.utils.net_utils import rel_prob_to_mat, find_all_paths, create_mrt, objdet_inference
from roi_data_layer.roidb import combined_roidb
from model.utils.data_viewer import dataViewer
from model.rpn.bbox_transform import bbox_xy_to_xywh

import cv2
from torchsummary import summary

class fasterRCNNDemo(object):
    def __init__(self, args, model_dir):
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

    def fasterRCNN_forward_process(self, image, save_res = False, id = ""):
        data_batch = prepare_data_batch_from_cvimage(image, is_cuda = True)
        result = self.RCNN(data_batch)
        rois = result[0][0][:,1:5].data
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

        print('obj_classes: {}'.format(obj_classes))
        print('obj_boxes: {}'.format(obj_boxes))
        print('obj_cls_name: {}'.format(obj_cls_name))

        if save_res:
            obj_det_img = self.data_viewer.draw_objdet(image.copy(),
                np.concatenate((obj_boxes, np.expand_dims(obj_classes, 1)), axis = 1), o_inds=list(range(num_box)))
            cv2.imwrite("images/" + id + "object_det.png", obj_det_img)

        return obj_boxes, obj_cls_name

if __name__ == '__main__':
    # we need to read configs of VMRN that were used in training and also need to be used in this demo
    args = read_cfgs()
    fasterrcnn_demo = fasterRCNNDemo(args, os.path.join(args.save_dir + "/" + args.dataset + "/" + args.net))
    while True:
        image_id = raw_input('Image ID: ').lower()
        if image_id == 'break':
            break
        # read cv image
        test_img_path = os.path.join('images', image_id + ".jpg")
        cv_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        # VMRN forward process
        obj_box, obj_cls = fasterrcnn_demo.fasterRCNN_forward_process(cv_img, save_res=True, id = image_id)

