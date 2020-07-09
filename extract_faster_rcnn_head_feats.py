from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import json
import time
import numpy as np
import h5py
import pprint
from scipy.misc import imread, imresize
import cv2
import torch

# faster rcnn
from model.utils.config import read_cfgs, cfg
from model.FasterRCNN import fasterRCNN
from roi_data_layer.roidb import combined_roidb
from model.utils.blob import prepare_data_batch_from_cvimage

this_dir = osp.dirname(__file__)
MATTNET_DIR = osp.join(this_dir, 'model/mattnet')
sys.path.insert(0, osp.join(MATTNET_DIR, 'lib/loaders'))

# dataloader
from loader import Loader

def main(args):
    dataset_splitBy = args.ref_dataset + '_' + args.splitBy
    if not osp.isdir(osp.join('cache/feats/', dataset_splitBy)):
        os.makedirs(osp.join('cache/feats/', dataset_splitBy))

    # Image Directory
    if 'coco' in dataset_splitBy:
        IMAGE_DIR = 'model/mattnet/data/images/mscoco/images/train2014'
    elif 'clef' in dataset_splitBy:
        IMAGE_DIR = 'model/mattnet/data/images/saiapr_tc-12'
    else:
        print('No image directory prepared for ', args.ref_dataset)
        sys.exit(0)

    # load dataset
    data_json = osp.join(MATTNET_DIR, 'cache/prepro', dataset_splitBy, 'data.json')
    data_h5 = osp.join(MATTNET_DIR, 'cache/prepro', dataset_splitBy, 'data.h5')
    loader = Loader(data_json, data_h5)
    images = loader.images

    # load RCNN model
    conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
    model_dir = os.path.join(args.save_dir + "/" + args.dataset + "/" + args.net)
    load_name=os.path.join(model_dir, args.frame + '_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                               args.checkpoint))
    print(load_name)
    trained_model=torch.load(load_name)

    _, _, _, _, cls_list=combined_roidb(args.imdbval_name, training=False)
    RCNN=fasterRCNN(len(cls_list), class_agnostic=trained_model['class_agnostic'], feat_name=args.net,
                        feat_list=('conv' + conv_num,), pretrained=True)
    RCNN.create_architecture()
    RCNN.load_state_dict(trained_model['model'])
    if args.cuda:
        RCNN.cuda()
    RCNN.eval()

    # feats_h5
    feats_dir=osp.join(MATTNET_DIR, 'cache/feats', dataset_splitBy, 'rcnn',
                        '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag))
    if not osp.isdir(feats_dir):
        os.makedirs(feats_dir)

    # extract
    for i, image in enumerate(images):
        file_name = image['file_name']
        img_path = osp.join(IMAGE_DIR, file_name)
        print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        data_batch = prepare_data_batch_from_cvimage(image, is_cuda=True)

        # !!! Here img_scale[x] should be equal to img_scale[y]
        # TODO, to add assert
        feat = RCNN.get_base_feat(data_batch)  # (1, 1024, x, y)
        feat = feat.data.cpu().numpy()
        im_info = data_batch[1][:, :3].cpu().numpy()  # (H, W, img_scale)
        print('feat {}'.format(feat.shape))
        print('im_info {}'.format(im_info.shape))

        # write
        feat_h5 = osp.join(feats_dir, str(image['image_id'])+'.h5')
        f = h5py.File(feat_h5, 'w')
        f.create_dataset('head', dtype=np.float32, data=feat)
        f.create_dataset('im_info', dtype=np.float32, data=im_info)
        f.close()
        if i % 10 == 0:
            print('%s/%s image_id[%s] size[%s] im_scale[%.2f] writen.' %
                  (i+1, len(images), image['image_id'], feat.shape, im_info[0][2]))

    print('Done.')


if __name__ == '__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--imdb_name', default='coco_minus_refer',
    #                     help='image databased trained on.')
    # parser.add_argument('--net_name', default='res101')
    # parser.add_argument('--iters', default=1250000, type=int)
    # parser.add_argument('--tag', default='notime')

    # parser.add_argument('--ref_dataset', type=str, default='refcoco',
    #                     help='dataset name: refclef, refcoco, refcoco+, refcocog')
    # parser.add_argument('--splitBy', type=str, default='unc',
    #                     help='splitBy: unc, google, berkeley')

    # args=parser.parse_args()

    args=read_cfgs()
    main(args)
