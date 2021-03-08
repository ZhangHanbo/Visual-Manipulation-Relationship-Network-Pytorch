# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.refcoco import refcoco
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.vmrd import vmrd
from datasets.bdds import bdds
from datasets.cornell import cornell
from datasets.jacquard import jacquard
from datasets.regrad import regrad

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_<2014 or 2017>_<split>
for year in ['2014', '2017']:
  if year == '2014':
      for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))
  else:
      for split in ['train', 'val', 'capval', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

for split in ['google', 'unc', 'umd']:
    for version in ['', '+', 'g']:
        for imageset in ['train', 'val', 'test']:
            name = 'refcoco{}_{}_{}'.format(version, split, imageset)
            __sets[name] = (lambda imageset=imageset, split=split, version=version: refcoco(imageset, split, version))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

for version in ['v1', 'compv1', 'ext']:
    for split in ['trainval', 'test']:
        name = 'vmrd_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vmrd(split, version))

for version in ['super_mini', 'mini', 'v1']:
    for split in ['train', 'seenval', 'seentest', 'unseenval', 'unseentest']:
        name = 'regrad_{}_{}'.format(version, split)
        __sets[name] = (lambda split=split, version=version: regrad(split, version))

for split in ['trainval', 'test']:
    name = 'bdds_{}'.format(split)
    __sets[name] = (lambda split=split:bdds(split))

for version in ['origin','rgd']:
    for splitway in ['iw', 'ow']:
        for split in ['trainval', 'test']:
            for testfold in range(1,6):
                name = 'cornell_{}_{}_{}_{}'.format(version,splitway, split, testfold)
                __sets[name] = (lambda split=split, version=version, splitway=splitway,
                                       testfold=testfold: cornell(split, version, splitway, testfold))

for split in ['trainval', 'test']:
    for testfold in range(1,6):
        for version in ['rgb', 'rgd', 'depth']:
            name = 'jacquard_{}_{}_{}'.format(version, split, testfold)
            __sets[name] = (lambda split=split, version=version, testfold=testfold:
                                jacquard(split, version, testfold))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
