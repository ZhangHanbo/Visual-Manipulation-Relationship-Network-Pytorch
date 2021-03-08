from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import argparse
import pprint


__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()
__C.TRAIN.COMMON = edict()
__C.TRAIN.RCNN_COMMON = edict()
__C.TRAIN.FASTER_RCNN = edict()
__C.TRAIN.FPN = edict()
__C.TRAIN.SSD = edict()
__C.TRAIN.VMRN = edict()
__C.TRAIN.FCGN = edict()

# Common hyperparams
__C.TRAIN.COMMON.LEARNING_RATE = 0.001 # Initial learning rate
__C.TRAIN.COMMON.MOMENTUM = 0.9 # Momentum
__C.TRAIN.COMMON.WEIGHT_DECAY = 0.0005 # Weight decay, for regularization
__C.TRAIN.COMMON.MAX_EPOCH = 20
__C.TRAIN.COMMON.GAMMA = 0.1 # Factor for reducing the learning rate
# Step size for reducing the learning rate, currently only support one step. (unit: epoch)
__C.TRAIN.COMMON.LR_DECAY_STEPSIZE = [80000]
__C.TRAIN.COMMON.DOUBLE_BIAS = True # Whether to double the learning rate for bias
__C.TRAIN.COMMON.TRUNCATED = False # Whether to initialize the weights with truncated normal distribution
__C.TRAIN.COMMON.BIAS_DECAY = False # Whether to have weight decay on bias as well
# User interaction
__C.TRAIN.COMMON.DISPLAY = 10 # Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.COMMON.SNAPSHOT_KEPT = 3 # The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.COMMON.SNAPSHOT_AFTER_TEST = False
__C.TRAIN.COMMON.SUMMARY_INTERVAL = 180 # The time interval for saving tensorflow summaries
# Images to use per minibatch
__C.TRAIN.COMMON.IMS_PER_BATCH = 1

__C.TRAIN.COMMON.MAX_SIZE = 1000 # Max pixel size of the longest side of a scaled input image
# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.COMMON.BBOX_THRESH = 0.5
__C.TRAIN.COMMON.SNAPSHOT_ITERS = 5000 # Iterations between snapshots

# __C.TRAIN.COMMON.SNAPSHOT_INFIX = ''
# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS = True
__C.TRAIN.COMMON.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.COMMON.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.COMMON.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.COMMON.USE_ALL_GT = True
# Whether to tune the batch normalization parameters during training
__C.TRAIN.COMMON.BN_TRAIN = False
# Whether to use augmentation
__C.TRAIN.COMMON.AUGMENTATION = False
# Train using these proposals
__C.TRAIN.COMMON.PROPOSAL_METHOD = 'gt'
# FOCAL LOSS
__C.TRAIN.COMMON.USE_FOCAL_LOSS = False
__C.TRAIN.COMMON.FOCAL_LOSS_GAMMA = 2
__C.TRAIN.COMMON.FOCAL_LOSS_ALPHA = 0.25
__C.TRAIN.COMMON.BBOX_REG = True # Train bounding-box regressors
__C.TRAIN.COMMON.USE_ODLOSS = True

# RCNN params
__C.TRAIN.RCNN_COMMON.USE_GT = False # Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.RCNN_COMMON.ASPECT_GROUPING = False # Whether to use aspect-ratio grouping of training images, introduced merely for saving GPU memory
# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
# Trim size for input images to create minibatch
__C.TRAIN.RCNN_COMMON.TRIM_HEIGHT = 600
__C.TRAIN.RCNN_COMMON.TRIM_WIDTH = 600
__C.TRAIN.RCNN_COMMON.BATCH_SIZE = 128 # Minibatch size (number of regions of interest [ROIs])
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.RCNN_COMMON.FG_THRESH = 0.5 # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.RCNN_COMMON.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.RCNN_COMMON.BG_THRESH_HI = 0.5
__C.TRAIN.RCNN_COMMON.BG_THRESH_LO = 0.1
# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# __C.TRAIN.RCNN_COMMON.USE_PREFETCH = False
# Use RPN to detect objects
__C.TRAIN.RCNN_COMMON.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RCNN_COMMON.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RCNN_COMMON.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RCNN_COMMON.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RCNN_COMMON.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RCNN_COMMON.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RCNN_COMMON.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RCNN_COMMON.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RCNN_COMMON.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RCNN_COMMON.RPN_MIN_SIZE = 8
# Deprecated (outside weights)
__C.TRAIN.RCNN_COMMON.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RCNN_COMMON.RPN_POSITIVE_WEIGHT = -1.0
# Deprecated (inside weights)
__C.TRAIN.RCNN_COMMON.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.TRAIN.RCNN_COMMON.RPN_USE_FOCAL_LOSS = False


# SSD params
__C.TRAIN.SSD.NEG_POS_RATIO = 3

__C.TRAIN.FCGN.NEG_POS_RATIO = 3
__C.TRAIN.FCGN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.TRAIN.FCGN.BBOX_POSITIVE_WEIGHTS = -1.0
__C.TRAIN.FCGN.ANGLE_MATCH = True
# if ANGLE_MATCH is True:
__C.TRAIN.FCGN.ANGLE_THRESH = 15.0
# else if ANGLE MATCH is False, use Jaccard Index match
__C.TRAIN.FCGN.JACCARD_THRESH = 0.5

# VMRN params
__C.TRAIN.VMRN.ONLINEDATA_BEGIN_ITER = 10000
__C.TRAIN.VMRN.TOP_N_ROIS_FOR_OBJ_DET = 300
__C.TRAIN.VMRN.TRAINING_DATA = 'all'
__C.TRAIN.VMRN.USE_REL_GRADIENTS = True # the gradient from vmrn
# (o1,r,o2) and (o2,r',o1)
__C.TRAIN.VMRN.ISEX = True
__C.TRAIN.VMRN.USE_REL_CLS_GRADIENTS = True # the gradient from vmrn classifier

# If true, during training, in each batch and each image, only one relation datum is selected to compute gradient.
__C.TRAIN.VMRN.ONE_DATA_PER_IMG = False

# True means the object detector is fixed during training, and a model path must be
# specified when this option is enabled.
__C.TRAIN.VMRN.FIX_OBJDET = False
__C.TRAIN.VMRN.OBJ_MODEL_PATH = 'output/coco+vmrd/res101/faster_rcnn_1_9_25724.pth'

#
# Testing options
#
__C.TEST = edict()
__C.TEST.COMMON = edict()
__C.TEST.RCNN_COMMON = edict()
__C.TEST.FASTER_RCNN = edict()
__C.TEST.FPN = edict()
__C.TEST.SSD = edict()
__C.TEST.VMRN = edict()
__C.TEST.FCGN = edict()


# Max pixel size of the longest side of a scaled input image
__C.TEST.COMMON.MAX_SIZE = 1000
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.COMMON.NMS = 0.3
# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.COMMON.MODE = 'nms'
# Object detection final threshold
__C.TEST.COMMON.OBJ_DET_THRESHOLD = 0.5
# Test using these proposals
__C.TEST.COMMON.PROPOSAL_METHOD = 'gt'
# Test using bounding-box regressors
__C.TEST.COMMON.BBOX_REG = True
# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.RCNN_COMMON.SVM = False
# Propose boxes
__C.TEST.RCNN_COMMON.HAS_RPN = False
## NMS threshold used on RPN proposals
__C.TEST.RCNN_COMMON.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RCNN_COMMON.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RCNN_COMMON.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RCNN_COMMON.RPN_MIN_SIZE = 16
# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RCNN_COMMON.RPN_TOP_N = 5000


__C.TEST.VMRN.ISEX = True

__C.TEST.FCGN.JACCARD_OVERLAP_THRESH = 0.25

#
# ResNet options
#
__C.RESNET = edict()
# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False
# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

__C.VGG = edict()
__C.VGG.FIXED_BLOCKS = 1

#
# MobileNet options
#
__C.MOBILENET = edict()
# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False
# Number of fixed layers during training, by default the first of all 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5
# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004
# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1. / 16.
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])
__C.PIXEL_MEANS_CAFFE = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.PIXEL_STDS = np.array([[[0.229, 0.224, 0.225]]])

__C.PRETRAIN_TYPE = "caffe"
# For reproducibility
__C.RNG_SEED = 3
# A small number that's used many times
__C.EPS = 1e-14
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'
# Place outputs under an experiments directory
__C.EXP_DIR = 'default'
# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True
# Default GPU device id
__C.GPU_ID = 0
# Maximal number of gt rois in an image during Training
__C.MAX_NUM_GT_BOXES = 20
__C.MAX_NUM_GT_GRASPS = 100
__C.CUDA = True
__C.CLASS_AGNOSTIC = True

__C.SCALES = (600,)
# For SSD, the FIXED_INPUT_SIZE need to be true
__C.FIXED_INPUT_SIZE = False

__C.RCNN_COMMON = edict()
__C.RCNN_COMMON.POOLING_MODE = 'crop'
# Size of the pooled region after RoI pooling
__C.RCNN_COMMON.POOLING_SIZE = 7
# Anchor scales for RPN
__C.RCNN_COMMON.ANCHOR_SCALES = [8,16,32]
# Anchor ratios for RPN
__C.RCNN_COMMON.ANCHOR_RATIOS = [0.5,1,2]
# Feature stride for RPN
__C.RCNN_COMMON.FEAT_STRIDE = [16, ]
__C.RCNN_COMMON.CROP_RESIZE_WITH_MAX_POOL = True
# The output layer of faster RCNN is determined by the dataset. If you want to train the model using some combined
# datasets (e.g. coco+pascal_voc) but test the model using a single dataset (e.g. pascal_voc), you need to specify
# this parameter to the combined one (e.g. coco+pascal_voc) so that the output layer matches the one you used in
# training.
# __C.RCNN_COMMON.OUT_LAYER = ''

__C.VMRN = edict()
__C.VMRN.OP2L_POOLING_MODE = 'pool'
__C.VMRN.OP2L_POOLING_SIZE = 7
# visual manipulation relationship types
__C.VMRN.FATHER = 1
__C.VMRN.CHILD = 2
__C.VMRN.NOREL = 3
# use shared weights in relationship network
__C.VMRN.SHARE_WEIGHTS = False
__C.VMRN.RELATION_CLASSIFIER = "vmrn"
__C.VMRN.UVTRANSE_REGULARIZATION = 1.0
__C.VMRN.USE_CRF = False
__C.VMRN.SCORE_POSTPROC=False

__C.SSD = edict()
__C.SSD.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
__C.SSD.PRIOR_MIN_SIZE = [30, 60, 111, 162, 213, 264]
__C.SSD.PRIOR_MAX_SIZE = [60, 111, 162, 213, 264, 315]
__C.SSD.PRIOR_STEP = [8, 16, 32, 64, 100, 300]
__C.SSD.PRIOR_ASPECT_RATIO = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
__C.SSD.PRIOR_CLIP = True

__C.FPN = edict()
__C.FPN.SHARE_HEADER = True
__C.FPN.SHARE_RPN = True
# k is in {0,1,2,3,4}
__C.FPN.K = 3
# add a convolutional layer after each unsampling layer
__C.FPN.UPSAMPLE_CONV = False

__C.FCGN = edict()
__C.FCGN.ANCHOR_SCALES = [54]
__C.FCGN.ANCHOR_RATIOS = [1]
__C.FCGN.ANCHOR_ANGLES = [-75, -45, -15, 15, 45, 75]
__C.FCGN.FEAT_STRIDE = [32]
__C.FCGN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2, 0.2)
__C.FCGN.BBOX_NORMALIZE_MEANS = (0., 0., 0., 0., 0.)

__C.MGN = edict()
__C.MGN.USE_ADAPTIVE_ANCHOR = False
__C.MGN.OBJECT_GRASP_BALANCE = 1.0
__C.MGN.USE_FIXED_SIZE_ROI = False
__C.MGN.FIX_OBJDET = False
__C.MGN.OBJ_MODEL_PATH = ''

import pdb
def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(imdb, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--frame', dest='frame',
                    help='faster_rcnn, fpn, ssd, faster_rcnn_vmrn, ssd_vmrn, fcgn, mgn, allinone',
                    default='faster_rcnn', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=0, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=0, type=int)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="output",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='the GPUs you do want to use for training',
                      default='', type=str)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=0, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      default=False, type=bool)
  parser.add_argument('--test', dest='test',
                      help='whether to perform test',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=None, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=None, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=None, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      action='store_true')
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      action='store_true')
  parser.add_argument('--vis', dest='vis',
                      help='whether to visualize training data',
                      action='store_true')

  args = parser.parse_args()
  return args

def dataset_name_to_cfg(name):
    data_config = {}
    if name == "pascal_voc":
        data_config['train'] = "voc_2007_trainval"
        data_config['val'] = "voc_2007_test"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name == "pascal_voc_0712":
        data_config['train'] = "voc_2007_trainval+voc_2012_trainval"
        data_config['val'] = "voc_2007_test"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name == "coco":
        data_config['train'] = "coco_2017_train"
        data_config['val'] = "coco_2017_val"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '50']
    elif name == "imagenet":
        data_config['train'] = "imagenet_train"
        data_config['val'] = "imagenet_val"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '30']
    elif name == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        data_config['train'] = "vg_1600-400-20_train"
        data_config['val'] = "vg_1600-400-20_val"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '50']
    elif name == 'vmrdcompv1':
        data_config['train'] = "vmrd_compv1_trainval"
        data_config['val'] = "vmrd_compv1_test"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name == 'regrad_super_mini':
        data_config['train'] = "regrad_super_mini_train"
        data_config['val'] = "regrad_super_mini_unseenval"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name == 'regrad_mini':
        data_config['train'] = "regrad_mini_train"
        data_config['val'] = "regrad_super_mini_unseenval"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name == 'regrad':
        data_config['train'] = "regrad_v1_train"
        data_config['val'] = "regrad_v1_unseenval"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name == 'vmrdext':
        data_config['train'] = "vmrd_ext_trainval"
        data_config['val'] = "vmrd_compv1_test"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '50']
    elif name == 'coco+vmrd':
        data_config['train'] = "coco_2017_train+vmrd_compv1_trainval"
        data_config['val'] = "coco_2017_val+vmrd_compv1_test"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '50']
    elif name == 'refcoco':
        data_config['train'] = "refcocog_umd_train"
        data_config['val'] = "refcocog_umd_val"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '50']
    elif name == 'bdds':
        data_config['train'] = "bdds_trainval"
        data_config['val'] = "bdds_test"
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '20']
    elif name[:7] == 'cornell':
        cornell = name.split('_')
        data_config['train'] = 'cornell_{}_{}_trainval_{}'.format(cornell[1],cornell[2],cornell[3])
        data_config['val'] = 'cornell_{}_{}_test_{}'.format(cornell[1],cornell[2],cornell[3])
        data_config['cfgs'] = ['MAX_NUM_GT_BOXES', '50']
    elif name[:8] == 'jacquard':
        jacquard = name.split('_')
        data_config['train'] = 'jacquard_{}_trainval_{}'.format(jacquard[1], jacquard[2])
        data_config['val'] = 'jacquard_{}_test_{}'.format(jacquard[1], jacquard[2])
        data_config['cfgs'] = ['MAX_NUM_GT_GRASPS', '1000']
    else:
        raise RuntimeError("The training set combination is not supported.")
    return data_config

def read_cfgs():
    args = parse_args()
    print('Called with args:')
    print(args)
    dataset_cfg = dataset_name_to_cfg(args.dataset)
    args.imdb_name, args.imdbval_name, args.set_cfgs = dataset_cfg['train'], dataset_cfg['val'], dataset_cfg['cfgs']
    if args.dataset[:7] == 'cornell':
        args.cfg_file = "cfgs/cornell_{}_{}_ls.yml".format(args.frame, args.net) if args.large_scale \
        else "cfgs/cornell_{}_{}.yml".format(args.frame, args.net)
    elif args.dataset[:8] == 'jacquard':
        args.cfg_file = "cfgs/jacquard_{}_{}_ls.yml".format(args.frame, args.net) if args.large_scale \
        else "cfgs/jacquard_{}_{}.yml".format(args.frame, args.net)
    else:
        args.cfg_file = "cfgs/{}_{}_{}_ls.yml".format(args.dataset, args.frame, args.net) if args.large_scale \
        else "cfgs/{}_{}_{}.yml".format(args.dataset, args.frame, args.net)
    print("Using cfg file: " + args.cfg_file)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if not args.disp_interval:
        args.disp_interval = cfg.TRAIN.COMMON.DISPLAY
    if not args.batch_size:
        args.batch_size = cfg.TRAIN.COMMON.IMS_PER_BATCH
    if not args.lr_decay_step:
        args.lr_decay_step = cfg.TRAIN.COMMON.LR_DECAY_STEPSIZE[0]
    if not args.lr:
        args.lr = cfg.TRAIN.COMMON.LEARNING_RATE
    if not args.lr_decay_gamma:
        args.lr_decay_gamma = cfg.TRAIN.COMMON.GAMMA
    if not args.max_epochs:
        args.max_epochs = cfg.TRAIN.COMMON.MAX_EPOCH
    print('Using config:')
    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.COMMON.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    pprint.pprint(cfg)
    if args.cuda:
        cfg.CUDA = True

    return args