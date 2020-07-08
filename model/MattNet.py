# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# add matplotlib before cv2, otherwise bug
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rcParams['image.interpolation'] = 'nearest'

from scipy.misc import imread, imresize
import scipy.ndimage
import numpy as np
import argparse
import json
import os
import os.path as osp
import time
from pprint import pprint

import mattnet.tools._init_paths
from layers.joint_match import JointMatching

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# root directory
MATTNET_ROOT_DIR = osp.join(osp.dirname(__file__), 'mattnet')

class MattNetV2(object):
    def __init__(self, args):
        # load model
        model_prefix = osp.join(MATTNET_ROOT_DIR, 'output', args.dataset+'_'+args.splitBy, args.model_id)
        tic = time.time()
        infos = json.load(open(model_prefix+'.json'))
        model_path = model_prefix + '.pth'
        self.dataset = args.dataset
        self.model_opt = infos['opt']
        self.word_to_ix = infos['word_to_ix']
        self.ix_to_att = {ix: att for att, ix in infos['att_to_ix'].items()}
        self.model = self.load_mattnet_model(model_path, self.model_opt)
        print('MattNet [%s_%s\'s %s] loaded in %.2f seconds.' %
            (args.dataset, args.splitBy, args.model_id, time.time()-tic))

    def load_mattnet_model(self, checkpoint_path, opt):
        # load MatNet model from pre-trained checkpoint_path
        tic = time.time()
        model = JointMatching(opt)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
        model.eval()
        model.cuda()
        return model

    def comprehend(self, img_data, expr):
        """
        Arguments:
        - img_data: computed from self.forward_image()
        - expr    : expression in string format
        Return entry is a dict of:
        - tokens     : list of words
        - pred_det_id: predicted det_id
        - pred_box   : pred_det's box [xywh]
        - rel_det_id : relative det_id
        - rel_box    : relative box [xywh]
        - sub_grid_attn: list of 49 attn
        - sub_attn   : list of seq_len attn
        - loc_attn   : list of seq_len attn
        - rel_attn   : list of seq_len attn
        - weights    : list of 3 module weights
        - pred_atts  : top 5 attributes, list of (att_wd, score)
        """
        # image data
        det_ids = img_data['det_ids']
        cxt_det_ids = img_data['cxt_det_ids']
        Dets = {det['det_id']: det for det in img_data['dets']}
        # masks = img_data['masks']
        Feats = img_data['Feats']

        # encode labels
        expr = expr.lower().strip()
        labels = self.encode_labels([expr], self.word_to_ix)  # (1, sent_length)
        labels = Variable(torch.from_numpy(labels).long().cuda())
        expanded_labels = labels.expand(
            len(det_ids), labels.size(1))  # (n, sent_length)
        scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores, module_scores = \
            self.model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                    Feats['cxt_fc7'], Feats['cxt_lfeats'],
                    expanded_labels)

        # move to numpy
        scores = scores.data.cpu().numpy()
        module_scores = module_scores.data.cpu().numpy()
        pred_ix = np.argmax(scores)
        pred_det_id = det_ids[pred_ix]
        att_scores = F.sigmoid(att_scores)  # (n, num_atts)
        rel_ixs = rel_ixs.data.cpu().numpy().tolist()  # (n, )
        rel_ix = rel_ixs[pred_ix]

        # get everything
        entry = {}
        entry['tokens'] = expr.split()
        entry['pred_det_id'] = det_ids[pred_ix]
        entry['pred_box'] = Dets[pred_det_id]['box']
        # entry['pred_mask'] = masks[pred_ix]
        # relative det_id
        entry['rel_det_id'] = cxt_det_ids[pred_ix][rel_ix]
        entry['rel_box'] = Dets[entry['rel_det_id']
                                ]['box'] if entry['rel_det_id'] > 0 else [0, 0, 0, 0]
        # attention
        entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist()  # list of 49 attn
        entry['sub_attn'] = sub_attn[pred_ix].data.cpu().numpy().tolist()  # list of seq_len attn
        entry['loc_attn'] = loc_attn[pred_ix].data.cpu().numpy().tolist()  # list of seq_len attn
        entry['rel_attn'] = rel_attn[pred_ix].data.cpu().numpy().tolist()  # list of seq_len attn
        entry['weights'] = weights[pred_ix].data.cpu().numpy().tolist()   # list of 3 weights
        entry['overall_scores'] = scores
        entry['module_scores'] = module_scores
        # attributes
        pred_atts = []  # list of (att_wd, score)
        pred_att_scores = att_scores[pred_ix].data.cpu().numpy()
        top_att_ixs = pred_att_scores.argsort(
        )[::-1][:5]  # check top 5 attributes
        for k in top_att_ixs:
            pred_atts.append((self.ix_to_att[k], float(pred_att_scores[k])))
        entry['pred_atts'] = pred_atts

        return entry

    def encode_labels(self, sent_str_list, word_to_ix):
        """
        Arguments:
        - sent_str_list: list of n sents in string format
        return:
        - labels: int32 (n, sent_length)
        """
        num_sents = len(sent_str_list)
        max_len = max([len(sent_str.split()) for sent_str in sent_str_list])
        L = np.zeros((num_sents, max_len), dtype=np.int32)
        for i, sent_str in enumerate(sent_str_list):
            tokens = sent_str.split()
            for j, w in enumerate(tokens):
                L[i,j] = word_to_ix[w] if w in word_to_ix else word_to_ix['<UNK>']
        return L