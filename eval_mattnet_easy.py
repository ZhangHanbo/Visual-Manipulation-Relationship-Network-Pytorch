from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse

from model.utils.config import read_cfgs, cfg

# model
from gt_rcnn_loader import GtRCNNLoader
import model.mattnet.tools._init_paths
from layers.joint_match import JointMatching
import models.eval_easy_utils as eval_utils

# torch
import torch
import torch.nn as nn

this_dir = osp.dirname(__file__)
MATTNET_DIR = osp.join(this_dir, 'model/mattnet')

def load_model(checkpoint_path, opt):
  tic = time.time()
  model = JointMatching(opt)
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'].state_dict())
  model.eval()
  model.cuda()
  print('model loaded in %.2f seconds' % (time.time()-tic))
  return model

def evaluate(args):

  opt = vars(args)
  # make other options
  opt['dataset_splitBy'] = opt['ref_dataset'] + '_' + opt['splitBy']
  opt['split'] = 'testA' # 'split: testAB or val, etc')
  opt['verbose'] = 1 # help='if we want to print the testing progress')

  # set up loader
  data_json = osp.join(MATTNET_DIR, 'cache/prepro', opt['dataset_splitBy'], 'data.json')
  data_h5 = osp.join(MATTNET_DIR, 'cache/prepro', opt['dataset_splitBy'], 'data.h5')
  loader = GtRCNNLoader(data_h5=data_h5, data_json=data_json)

  # load mode info
  model_prefix = osp.join(MATTNET_DIR, 'output', opt['dataset_splitBy'], opt['id'])
  infos = json.load(open(model_prefix+'.json'))
  model_opt = infos['opt']
  model_path = model_prefix + '.pth'
  model = load_model(model_path, model_opt)

  # loader's feats
  feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['ref_imdb_name'], model_opt['tag'])
  # args.imdb_name = model_opt['ref_imdb_name']
  # args.net_name = model_opt['net_name']
  # args.tag = model_opt['tag']
  # args.iters = model_opt['iters']
  loader.prepare_rcnn(head_feats_dir=osp.join(MATTNET_DIR, 'cache/feats/', model_opt['dataset_splitBy'], 'rcnn', feats_dir), 
                      args=args) 
  ann_feats = osp.join(MATTNET_DIR, 'cache/feats', model_opt['dataset_splitBy'], 'rcnn', 
                       '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['ref_imdb_name'], model_opt['tag']))
  loader.loadFeats({'ann': ann_feats})

  # check model_info and params
  assert model_opt['dataset'] == opt['dataset']
  assert model_opt['splitBy'] == opt['splitBy']

  # evaluate on the split, 
  # predictions = [{sent_id, sent, gd_ann_id, pred_ann_id, pred_score, sub_attn, loc_attn, weights}]
  split = opt['split']
  model_opt['num_sents'] = opt['num_sents']
  model_opt['verbose'] = opt['verbose']
  crit = None
  val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, crit, split, model_opt)
  print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
        (opt['dataset_splitBy'], opt['split'], len(predictions), acc*100.)) 
  print('attribute precision : %.2f%%' % (overall['precision']*100.0))
  print('attribute recall    : %.2f%%' % (overall['recall']*100.0))
  print('attribute f1        : %.2f%%' % (overall['f1']*100.0))       

  # save
  out_dir = osp.join('cache', 'results', opt['dataset_splitBy'], 'easy')
  if not osp.isdir(out_dir):
    os.makedirs(out_dir)
  out_file = osp.join(out_dir, opt['id']+'_'+opt['split']+'.json')
  with open(out_file, 'w') as of:
    json.dump({'predictions': predictions, 'acc': acc, 'overall': overall}, of)

  # write to results.txt
  f = open(MATTNET_DIR, 'experiments/easy_results.txt', 'a')
  f.write('[%s][%s], id[%s]\'s acc is %.2f%%\n' % \
          (opt['dataset_splitBy'], opt['split'], opt['id'], acc*100.0))


if __name__ == '__main__':
    
  args = read_cfgs()
  evaluate(args)


