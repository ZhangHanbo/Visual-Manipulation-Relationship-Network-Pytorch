import numpy as np
import os
import torch

from model.FasterRCNN_VMRN import fasterRCNN_VMRN
from model.utils.config import read_cfgs, cfg
from model.utils.blob import prepare_data_batch_from_cvimage
from model.utils.net_utils import rel_prob_to_mat, find_all_paths, create_mrt
from roi_data_layer.roidb import get_imdb, combined_roidb
from model.utils.data_viewer import dataViewer
from model.rpn.bbox_transform import bbox_xy_to_xywh

import cv2

class VMRNDemo(object):
    def __init__(self, args, model_dir):
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        # load trained model
        load_name = os.path.join(model_dir, args.frame + '_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                                args.checkpoint))
        trained_model = torch.load(load_name)
        # init VMRN
        _, _, _, _, cls_list = combined_roidb(args.imdbval_name, training=False)
        self.VMRN = fasterRCNN_VMRN(len(cls_list), class_agnostic=trained_model['class_agnostic'], feat_name=args.net,
                    feat_list=('conv' + conv_num,), pretrained=True)
        self.VMRN.create_architecture(cfg.TRAIN.VMRN.OBJ_MODEL_PATH)
        self.VMRN.load_state_dict(trained_model['model'])
        if args.cuda:
            self.VMRN.cuda()
            self.cuda = True
        self.VMRN.eval()
        # init classes
        self.classes = cls_list
        self.class_to_ind = dict(zip(self.classes, xrange(len(cls_list))))
        self.ind_to_class = dict(zip(xrange(len(cls_list)), self.classes))

        # init data viewer
        self.data_viewer = dataViewer(self.classes)

    def VMRN_forward_process(self, image, save_res = False, id = ""):
        data_batch = prepare_data_batch_from_cvimage(image, is_cuda = True)
        rel_result = self.VMRN(data_batch)
        rel_result = rel_result[3]
        obj_bboxes = rel_result[0].cpu().numpy()
        obj_classes = rel_result[1].cpu().numpy()
        num_box = obj_bboxes.shape[0]
        rel_prob = rel_result[2]
        rel_mat, rel_score = rel_prob_to_mat(rel_prob, num_box)
        obj_cls_name = []
        for cls in obj_classes:
            obj_cls_name.append(self.ind_to_class[cls])
        mrt = create_mrt(rel_mat, rel_score=rel_score)

        if save_res:
            obj_det_img = self.data_viewer.draw_objdet(image.copy(),
                np.concatenate((obj_bboxes, np.expand_dims(obj_classes, 1)), axis = 1), o_inds=list(range(num_box)))
            cv2.imwrite("images/" + id + "object_det.png", obj_det_img)
            rel_det_img = self.data_viewer.draw_mrt(image.copy(), rel_mat, class_names=obj_cls_name, rel_score=rel_score)
            cv2.imwrite("images/" + id + "relation_det.png", rel_det_img)

        return obj_bboxes, obj_cls_name, mrt

if __name__ == '__main__':
    # we need to read configs of VMRN that were used in training and also need to be used in this demo
    args = read_cfgs()
    vmrn_demo = VMRNDemo(args, os.path.join(args.save_dir + "/" + args.dataset + "/" + args.net))
    while True:
        image_id = raw_input('Image ID: ').lower()
        if image_id == 'break':
            break
        # read cv image
        test_img_path = os.path.join('images', image_id + ".jpg")
        cv_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        # VMRN forward process
        obj_box, obj_cls, mrt = vmrn_demo.VMRN_forward_process(cv_img, save_res=True, id = image_id)

