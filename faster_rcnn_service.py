import rospy
from faster_rcnn_detector.srv import ObjectDetection, ObjectDetectionResponse

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
from cv_bridge import CvBridge
br = CvBridge()

class fasterRCNNService(object):
    def __init__(self, args, model_path):
        rospy.init_node('faster_rcnn_server')
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        # load trained model
        trained_model = torch.load(model_path)
        # init VMRN
        _,_,_,_,cls_list = combined_roidb(args.imdbval_name, training=False)
        self.RCNN = fasterRCNN(len(cls_list), class_agnostic=trained_model['class_agnostic'], feat_name=args.net,
                    feat_list=('conv' + conv_num,), pretrained=True)
        self.RCNN.create_architecture()
        self.RCNN.load_state_dict(trained_model['model'])
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
        s = rospy.Service('faster_rcnn_server', ObjectDetection, self.det_serv_callback)
        print("Ready to detect object.")
        rospy.spin()

    def det_serv_callback(self, req):
        img_msg = req.img
        img = br.imgmsg_to_cv2(img_msg)
        dets = self.fasterRCNN_forward_process(img, save_res=True)
        obj_box = dets[0]
        obj_cls = dets[1]
        num_obj = dets[0].shape[0]
        res = ObjectDetectionResponse()
        res.num_box = int(num_obj)
        res.bbox = obj_box.astype(np.float64).reshape(-1).tolist()
        res.cls = obj_cls.astype(np.int32).reshape(-1).tolist()
        return res

    def fasterRCNN_forward_process(self, image, save_res = False, id = ""):
        data_batch = prepare_data_batch_from_cvimage(image, is_cuda = True)
        result  = self.RCNN(data_batch)
        rois = result[0][0][:,1:5].data
        cls_prob = result[1][0].data
        bbox_pred = result[2][0].data
        obj_boxes = objdet_inference(cls_prob, bbox_pred, data_batch[1][0], rois,
                                     class_agnostic=False, for_vis=True, recover_imscale=True)
        if save_res:
            np.save("images/output/" + id + "_bbox.npy", obj_boxes)
        obj_classes = obj_boxes[:, -1]
        obj_boxes = obj_boxes[:, :-1]
        num_box = obj_boxes.shape[0]
        obj_cls_name = []
        for cls in obj_classes:
            obj_cls_name.append(self.ind_to_class[cls])

        if save_res:
            obj_det_img = self.data_viewer.draw_objdet(image.copy(),
                np.concatenate((obj_boxes, np.expand_dims(obj_classes, 1)), axis = 1), o_inds=list(range(num_box)))
            cv2.imwrite("images/output/" + id + "object_det.png", obj_det_img)

        return obj_boxes, obj_classes, obj_cls_name

if __name__ == '__main__':
    # we need to read configs of VMRN that were used in training and also need to be used in this demo
    args = read_cfgs()
    fasterrcnn_service = fasterRCNNService(args, os.path.join(args.save_dir , args.dataset , args.net, "faster_rcnn_1_7_9151.pth"))



