"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
from datasets.joint_od import joint_od
from model.utils.config import dataset_name_to_cfg
import pdb
import pickle


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
    roidb = imdb.roidb
    if not (imdb.name.startswith('coco') or isinstance(imdb, joint_od)):
        widths = imdb.widths
        heights = imdb.heights

    # xmax,ymax,xmin,ymin = (0,0,300,300)

    for i in range(len(imdb.image_index)):
        # if (np.max(roidb[i]['boxes'][:,::2]) > xmax):
        # xmax = np.max(roidb[i]['boxes'][:,::2])
        # if (np.max(roidb[i]['boxes'][:,1::2]) > ymax):
        # ymax = np.max(roidb[i]['boxes'][:,1::2])
        # if (np.min(roidb[i]['boxes'][:,::2]) < xmin):
        # xmin = np.min(roidb[i]['boxes'][:,::2])
        # if (np.min(roidb[i]['boxes'][:,1::2]) < ymin):
        # ymin = np.min(roidb[i]['boxes'][:,1::2])

        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco') or isinstance(imdb, joint_od)):
            roidb[i]['width'] = widths[i]
            roidb[i]['height'] = heights[i]

        # TODO: There may be replicated img_id for different images. Deal with them!
        if roidb[i]['img_id'].startswith("coco"):
            roidb[i]['img_id'] = roidb[i]['img_id'].split("_")[1]
        elif roidb[i]['img_id'].startswith("vg"):
            roidb[i]['img_id'] = roidb[i]['img_id'].split("_")[1]

        # need gt_overlaps as a dense array for argmax
        if 'gt_overlaps' in roidb[i]:
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = -1

    while i < len(roidb) - 1:
        i += 1
        if 'boxes' in roidb[i]:
            if len(roidb[i]['boxes']) == 0:
                del roidb[i]
                i -= 1
                continue

        if 'grasps' in roidb[i]:
            if len(roidb[i]['grasps']) == 0:
                del roidb[i]
                i -= 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb


def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """

    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        print('Preparing training data...')
        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print('done')
        return imdb.roidb

    # generate original roidb
    imdb_list = imdb_names.split('+')
    imdb_list.sort()

    print('Set proposal method: {:s}'.format(cfg.TRAIN.COMMON.PROPOSAL_METHOD))
    imdbs = [get_imdb(s) for s in imdb_list]
    imdb_dict = dict(zip(imdb_list, imdbs))

    # modify roidbs to the combined versions
    if len(imdbs) > 1:
        imdb = joint_od(imdbs)
    else:
        imdb = imdbs[0]
    cls_list_to_build_net = imdb.classes
    roidb = get_training_roidb(imdb)

    # if cfg.RCNN_COMMON.OUT_LAYER != '':
    #   assert len(imdbs) == 1, "Now the specified OUT_LAYER only support non-combined dataset."
    #   dataset_cfg = dataset_name_to_cfg(cfg.RCNN_COMMON.OUT_LAYER)
    #   if training:
    #     out_layer_cfg = dataset_cfg['train']
    #   else:
    #     out_layer_cfg = dataset_cfg['val']
    #   if imdb_names != out_layer_cfg:
    #     assert len(out_layer_cfg.split('+')) > 1, "When the categories in testing set is a subset of those of" \
    #       " the training, you also need to use '+' to concatenate them (e.g. coco+pascal_voc)"
    #     temp_imdb_list = out_layer_cfg.split('+')
    #     temp_imdb_list.sort()
    #     temp_imdbs = []
    #     # to make sure that the test dataset can be modified to match the one used in training
    #     # WE ASSUME that the memory of the test dataset can be shared.
    #     for s in temp_imdb_list:
    #       if s in imdb_dict:
    #         temp_imdbs.append(imdb_dict[s])
    #       else:
    #         temp_imdbs.append(get_imdb(s))
    #
    #     temp_imdb = joint_od(temp_imdbs)
    #     for db in imdbs:
    #       db._update_roidb()
    #     cls_list_to_build_net = temp_imdb.classes

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index, cls_list_to_build_net
