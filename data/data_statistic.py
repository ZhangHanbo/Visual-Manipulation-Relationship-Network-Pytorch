from __future__ import print_function
from __future__ import absolute_import

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import uuid
import pickle
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

def load_vmrd_annotation(index):
    filename = os.path.join('Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    gt_classes = []
    father_list = []
    child_list = []

    # Load object bounding boxes into a data frame.

    for ix, obj in enumerate(objs):
        cls = obj.find('name').text.lower().strip()
        gt_classes += cls

        fathernodes = obj.find('father').findall('num')
        fathers = [int(f.text) for f in fathernodes]
        childnodes = obj.find('children').findall('num')
        children = [int(f.text) for f in childnodes]

        father_list.append(np.array(fathers, dtype=np.uint16))
        child_list.append(np.array(children, dtype=np.uint16))

    return {
            'gt_classes': gt_classes,
            'obj_num': num_objs,
            }

def load_index(trainval_list_path, test_list_path):
    trainval_file = open(trainval_list_path)
    test_file = open(test_list_path)
    trainval_list = trainval_file.readlines()
    test_list = test_file.readlines()
    return trainval_list, test_list

trainval_file = 'VMRD/vmrdcompv1/ImageSets/Main/trainval.txt'
test_file = 'VMRD/vmrdcompv1/ImageSets/Main/test.txt'
trainval_list, test_list = load_index(trainval_file, test_file)
cls_list = (
         'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
         'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
         'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
         'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
         'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch'
        )
cls_counter = dict(zip(cls_list, list(np.zeros(len(cls_list)))))
for line in trainval_list:
    index = line[:-1]
    anno = load_vmrd_annotation(index)
    for cls in anno['gt_classes']:
        cls_counter[cls] += 1





