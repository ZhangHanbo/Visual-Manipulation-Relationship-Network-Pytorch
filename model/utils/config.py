from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

from model.utils.augmentations import Augmentation, Augmentation_Grasp, \
  Augmentation_Grasp_Test, Augmentation_VMRD, Augmentation_Grasp_Roign_Cornell

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
# Use horizontally-flipped images during training?
__C.TRAIN.COMMON.USE_FLIPPED = True
__C.TRAIN.COMMON.USE_VERTICAL_ROTATED = False
__C.TRAIN.COMMON.MAX_SIZE = 1000 # Max pixel size of the longest side of a scaled input image
# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.COMMON.BBOX_THRESH = 0.5
__C.TRAIN.COMMON.SNAPSHOT_ITERS = 5000 # Iterations between snapshots
# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.COMMON.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
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
# Fix input size. SSD must be True
__C.TRAIN.COMMON.FIXED_INPUT_SIZE = False
__C.TRAIN.COMMON.INPUT_SIZE = 300
# Whether to use augmentation
__C.TRAIN.COMMON.AUGMENTATION = False
__C.TRAIN.COMMON.AUGMENTER = Augmentation()
# Train using these proposals
__C.TRAIN.COMMON.PROPOSAL_METHOD = 'gt'
# FOCAL LOSS
__C.TRAIN.COMMON.USE_FOCAL_LOSS = False
__C.TRAIN.COMMON.FOCAL_LOSS_GAMMA = 2
__C.TRAIN.COMMON.FOCAL_LOSS_ALPHA = 0.25
__C.TRAIN.COMMON.BBOX_REG = True # Train bounding-box regressors

# RCNN params
__C.TRAIN.RCNN_COMMON.USE_GT = False # Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.RCNN_COMMON.ASPECT_GROUPING = False # Whether to use aspect-ratio grouping of training images, introduced merely for saving GPU memory
# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.RCNN_COMMON.SCALES = (600,) # random scale.
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
# (o1,r,o2) and (o2,r',o1)
__C.TRAIN.VMRN.ISEX = True

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
__C.TEST.COMMON.OBJ_DET_THRESHOLD = 0.3
# Test using these proposals
__C.TEST.COMMON.PROPOSAL_METHOD = 'gt'
__C.TEST.COMMON.FIXED_INPUT_SIZE = False
__C.TEST.COMMON.INPUT_SIZE = 300
__C.TEST.COMMON.AUGMENTATION = False
__C.TEST.COMMON.AUGMENTER = Augmentation_Grasp_Test()
# Test using bounding-box regressors
__C.TEST.COMMON.BBOX_REG = True


# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.RCNN_COMMON.SCALES = (600,)
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


__C.TEST.VMRN.ISEX = False

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
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
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
__C.CUDA = False

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

__C.VMRN = edict()
__C.VMRN.OP2L_POOLING_MODE = 'pool'
__C.VMRN.OP2L_POOLING_SIZE = 7
# visual manipulation relationship types
__C.VMRN.FATHER = 1
__C.VMRN.CHILD = 2
__C.VMRN.NOREL = 3
# use shared weights in relationship network
__C.VMRN.SHARE_WEIGHTS = False

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
__C.MGN.USE_POOLED_FEATS = True
__C.MGN.USE_ADAPTIVE_ANCHOR = False
__C.MGN.OBJECT_GRASP_BALANCE = 1.0
__C.MGN.USE_FIXED_SIZE_ROI = False

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
      if k == 'AUGMENTER':
        augmenter_list = ("Augmentation()",
                          "Augmentation_Grasp()",
                          "Augmentation_Grasp_Roign_Cornell()",
                          "Augmentation_Grasp_Test()",
                          "Augmentation_VMRD()")
        if v in augmenter_list:
          v = eval(v)
        else:
          raise ValueError("Augmenter not defined.")
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
