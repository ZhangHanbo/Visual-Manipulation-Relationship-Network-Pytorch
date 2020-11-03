import torch
import torch.nn as nn
from utils.config import cfg
from utils.caffe_utils import load_caffemodel

from FasterRCNN import fasterRCNN
from utils.net_utils import set_bn_eval, set_bn_fix

class UNITER_pytorch(fasterRCNN):
    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained=True):
        super(UNITER_pytorch, self).__init__(classes, class_agnostic, feat_name, feat_list, pretrained)
        self._fix_fasterRCNN = cfg.TRAIN.UNITER.FIX_OBJDET
        if self._fix_fasterRCNN:
            self._fixed_keys = []

class UNITER__caffe(nn.Module):
    def __init__(self, classes, class_agnostic, feat_name, feat_list=('conv4',), pretrained=True):
        super(UNITER__caffe, self).__init__()
        model_path = "data/pretrained_model/resnet101_faster_rcnn_final.caffemodel"
        prototxt_path = "data/pretrained_model/test_gt.prototxt"
        print("Loading Caffe Model: " + model_path)
        self.caffe_model = load_caffemodel(prototxt_path, model_path, gpu_id=0)

    def load_pretrained_uniter(self):
        pass

    def forward(self, data_batch):
        im_data = data_batch[0]
        im_info = data_batch[1]
        gt_boxes = data_batch[2]
        num_boxes = data_batch[3]
