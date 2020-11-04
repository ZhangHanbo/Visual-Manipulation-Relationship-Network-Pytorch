# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo
# based on code from Jiasen Lu, Jianwei Yang, Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pprint
import time
import torch.nn as nn
import pickle

from torch.utils.data.sampler import Sampler
import matplotlib
matplotlib.use('Agg')

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import *
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, parse_args, read_cfgs
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.data_viewer import dataViewer
from model.utils.blob import image_unnormalize

from model.FasterRCNN import fasterRCNN
from model.FPN import FPN
from model.SSD import SSD
from model.FasterRCNN_VMRN import fasterRCNN_VMRN
from model.FCGN import FCGN
from model.SSD_VMRN import SSD_VMRN
from model.MGN import MGN
from model.AllinOne import All_in_One
from model.EfficientDet import EfficientDet
import model.VAM as VAM

from model.utils.net_utils import objdet_inference, grasp_inference, objgrasp_inference, rel_prob_to_mat
from datasets.factory import get_imdb

import warnings

torch.set_default_tensor_type(torch.FloatTensor)

# implemented-algorithm list
LEGAL_FRAMES = {"faster_rcnn", "ssd", "fpn", "faster_rcnn_vmrn", "ssd_vmrn", "all_in_one", "fcgn", "mgn", "vam",
                "efc_det"}

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def makeCudaData(data_list):
    for i, data in enumerate(data_list):
        data_list[i] = data.cuda()
    return data_list

def init_network(args, n_cls):
    """
    :param args: define hyperparameters
    :param n_cls: number of object classes for initializing network output layers
    :return:
    """
    # initilize the network here.'
    if args.frame == 'faster_rcnn':
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        Network = fasterRCNN(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                             feat_list=('conv' + conv_num,), pretrained = True)
    elif args.frame == 'faster_rcnn_vmrn':
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        Network = fasterRCNN_VMRN(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                             feat_list=('conv' + conv_num,), pretrained=True)
    elif args.frame == 'fpn':
        Network = FPN(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                             feat_list=('conv2', 'conv3', 'conv4', 'conv5'), pretrained=True)
    elif args.frame == 'fcgn':
        conv_num = str(int(np.log2(cfg.FCGN.FEAT_STRIDE[0])))
        Network = FCGN(feat_name=args.net, feat_list=('conv' + conv_num,), pretrained=True)
    elif args.frame == 'mgn':
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        Network = MGN(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                                  feat_list=('conv' + conv_num,), pretrained=True)
    elif args.frame == 'all_in_one':
        conv_num = str(int(np.log2(cfg.RCNN_COMMON.FEAT_STRIDE[0])))
        Network = All_in_One(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                      feat_list=('conv' + conv_num,), pretrained=True)
    elif args.frame == 'ssd':
        Network = SSD(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                      feat_list=('conv3', 'conv4'), pretrained=True)
    elif args.frame == 'ssd_vmrn':
        Network = SSD_VMRN(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                      feat_list=('conv3', 'conv4'), pretrained=True)

    elif args.frame == 'efc_det':
        Network = EfficientDet(n_cls, class_agnostic=args.class_agnostic, feat_name=args.net,
                               feat_list=('conv3', 'conv4', 'conv5', 'conv6', 'conv7'), pretrained=True)
    elif args.frame == 'vam':
        if args.net == 'vgg16':
            Network = VAM.vgg16(n_cls, pretrained=True)
        elif args.net == 'res50':
            Network = VAM.resnet(n_cls, layer_num=50, pretrained=True)
        elif args.net == 'res101':
            Network = VAM.resnet(n_cls, layer_num=101, pretrained=True)
        else:
            print("network is not defined")
            pdb.set_trace()
    else:
        print("frame is not defined")
        pdb.set_trace()

    if args.frame in {'ssd_vmrn', 'faster_rcnn_vmrn'} and cfg.TRAIN.VMRN.FIX_OBJDET:
        Network.create_architecture(cfg.TRAIN.VMRN.OBJ_MODEL_PATH)
    elif args.frame in {'mgn', 'all_in_one'} and cfg.MGN.FIX_OBJDET:
        Network.create_architecture(cfg.MGN.OBJ_MODEL_PATH)
    else:
        Network.create_architecture()

    lr = args.lr
    # tr_momentum = cfg.TRAIN.COMMON.MOMENTUM
    # tr_momentum = args.momentum

    args.start_epoch = 1
    if args.resume:
        output_dir = args.save_dir + "/" + args.dataset + "/" + args.net
        load_name = os.path.join(output_dir,
                                 args.frame + '_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                        args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        Network.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.RCNN_COMMON.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        if args.iter_per_epoch is not None:
            Network.iter_counter = (args.checkepoch - 1) * args.iter_per_epoch + args.checkpoint
        print("start iteration:", Network.iter_counter)

    if args.cuda:
        Network.cuda()

    if len(args.mGPUs) > 0:
        gpus = [int(i) for i in args.mGPUs.split('')]
        Network = nn.DataParallel(Network, gpus)

    params = []
    for key, value in dict(Network.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.COMMON.DOUBLE_BIAS + 1),
                          'weight_decay': cfg.TRAIN.COMMON.BIAS_DECAY and cfg.TRAIN.COMMON.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.COMMON.WEIGHT_DECAY}]

    # init optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.COMMON.MOMENTUM)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return Network, optimizer

def detection_filter(all_boxes, all_grasp = None, max_per_image = 100):
    # Limit to max_per_image detections *over all classes*
    image_scores = np.hstack([all_boxes[j][:, -1]
                              for j in xrange(1, len(all_boxes))])
    if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in xrange(1, len(all_boxes)):
            keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
            all_boxes[j] = all_boxes[j][keep, :]
            if all_grasp is not None:
                all_grasp[j] = all_grasp[j][keep, :]
    if all_grasp is not None:
        return all_boxes, all_grasp
    else:
        return all_boxes

def vis_gt(data_list, visualizer, frame, train_mode = False):
    im_vis = image_unnormalize(data_list[0].permute(1, 2, 0).cpu().numpy())
    # whether to visualize training data
    if not train_mode:
        im_vis = cv2.resize(im_vis, None, None, fx=1. / data_list[1][3].item(), fy=1. / data_list[1][2].item(),
                        interpolation=cv2.INTER_LINEAR)
    if frame in {"fpn", "faster_rcnn", "ssd"}:
        im_vis = visualizer.draw_objdet(im_vis, data_list[2].cpu().numpy())
    elif frame in {"ssd_vmrn", "vam", "faster_rcnn_vmrn"}:
        im_vis = visualizer.draw_objdet(im_vis, data_list[2].cpu().numpy(), o_inds = np.arange(data_list[3].item()))
        im_vis = visualizer.draw_mrt(im_vis, data_list[4].cpu().numpy(), rel_score=data_list[5])
    elif frame in {"fcgn"}:
        im_vis = visualizer.draw_graspdet(im_vis, data_list[2].cpu().numpy())
    elif frame in {"all_in_one"}:
        # TODO: visualize manipulation relationship tree
        im_vis = visualizer.draw_graspdet_with_owner(im_vis, data_list[2].cpu().numpy(),
                                                     data_list[3].cpu().numpy(), data_list[7].cpu().numpy())
        im_vis = visualizer.draw_mrt(im_vis, data_list[6].cpu().numpy())
    elif frame in {"mgn"}:
        im_vis = visualizer.draw_graspdet_with_owner(im_vis, data_list[2].cpu().numpy(),
                                                     data_list[3].cpu().numpy(), data_list[6].cpu().numpy())
    else:
        raise RuntimeError
    return im_vis

def evalute_model(Network, namedb, args):
    max_per_image = 100

    # load test dataset
    imdb, roidb, ratio_list, ratio_index, cls_list = combined_roidb(namedb, False)
    if args.frame in {"fpn", "faster_rcnn", "efc_det"}:
        dataset = fasterrcnnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    elif args.frame in {"ssd"}:
        dataset = ssdbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    elif args.frame in {"ssd_vmrn", "vam"}:
        dataset = svmrnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    elif args.frame in {"faster_rcnn_vmrn"}:
        dataset = fvmrnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    elif args.frame in {"fcgn"}:
        dataset = fcgnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    elif args.frame in {"all_in_one"}:
        dataset = fallinonebatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    elif args.frame in {"mgn"}:
        dataset = roignbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=False, cls_list=cls_list, augmentation=False)
    else:
        raise RuntimeError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)
    num_images = len(roidb)

    output_dir = args.save_dir + "/" + args.dataset + "/" + args.net

    if args.vis:
        visualizer = dataViewer(cls_list)
        data_vis_dir = os.path.join(args.save_dir, args.dataset, 'data_vis', 'test')
        if not os.path.exists(data_vis_dir):
            os.makedirs(data_vis_dir)
        id_number_to_name = {}
        for r in roidb:
            id_number_to_name[r["img_id"]] = r["image"]

    start = time.time()

    # init variables
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(len(cls_list))]
    all_rel = []
    all_grasp = [[[] for _ in xrange(num_images)]
                 for _ in xrange(len(cls_list))]

    Network.eval()
    empty_array= np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i in range(num_images):

        data_batch = next(data_iter)
        if args.cuda:
            data_batch = makeCudaData(data_batch)

        det_tic = time.time()
        # forward process
        if args.frame == 'faster_rcnn' or args.frame == 'fpn':
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, net_loss_cls, net_loss_bbox, rois_label = Network(data_batch)
            boxes = rois[:, :, 1:5]
        elif args.frame in {'ssd', 'efc_det'}:
            bbox_pred, cls_prob, net_loss_bbox, net_loss_cls = Network(data_batch)
            boxes = Network.priors.type_as(bbox_pred).unsqueeze(0)
        elif args.frame == 'faster_rcnn_vmrn':
            rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_rel_loss_cls, reg_loss, rois_label = Network(data_batch)
            boxes = rois[:, :, 1:5]
            all_rel.append(rel_result)
        elif args.frame == 'ssd_vmrn' or args.frame == 'vam':
            bbox_pred, cls_prob, rel_result, loss_bbox, loss_cls, rel_loss_cls ,reg_loss= Network(data_batch)
            boxes = Network.priors.type_as(bbox_pred)
            all_rel.append(rel_result)
        elif args.frame == 'fcgn':
            bbox_pred, cls_prob, loss_bbox, loss_cls, rois_label, boxes = Network(data_batch)
        elif args.frame == 'mgn':
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rois_label, grasp_loc, grasp_prob, \
                grasp_bbox_loss, grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
            boxes = rois[:, :, 1:5]
        elif args.frame == 'all_in_one':
            rois, cls_prob, bbox_pred, rel_result, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rel_loss_cls, reg_loss, rois_label, \
            grasp_loc, grasp_prob, grasp_bbox_loss, grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
            boxes = rois[:, :, 1:5]
            all_rel.append(rel_result)

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        # collect results
        if args.frame in {'ssd', 'fpn', 'faster_rcnn', 'faster_rcnn_vmrn', 'ssd_vmrn', 'vam', 'efc_det'}:
            # detected_box is a list of boxes. len(list) = num_classes
            det_box = objdet_inference(cls_prob[0].data, bbox_pred[0].data, data_batch[1][0].data,
                        box_prior=boxes[0].data, class_agnostic=args.class_agnostic, for_vis=False)
            if args.vis:
                if args.frame not in {'faster_rcnn_vmrn', 'ssd_vmrn', 'vam'}:
                    # for object detection algorithms
                    vis_boxes = objdet_inference(cls_prob[0].data, bbox_pred[0].data, data_batch[1][0].data,
                                     box_prior=boxes[0].data, class_agnostic=args.class_agnostic,
                                     for_vis=True)
                    data_list = [data_batch[0][0], data_batch[1][0], torch.Tensor(vis_boxes)]
                else:
                    # for visual manipulation relationship detection algorithms
                    det_res = all_rel[-1]
                    if det_res[0].shape[0] > 0:
                        vis_boxes = torch.cat([det_res[0], det_res[1].unsqueeze(1)], dim = 1)
                    else:
                        vis_boxes = torch.Tensor([])
                    rel_mat, rel_score = rel_prob_to_mat(det_res[2], vis_boxes.size(0))
                    data_list = [data_batch[0][0], data_batch[1][0], vis_boxes,
                                 torch.Tensor([vis_boxes.size(0)]), torch.Tensor(rel_mat), torch.Tensor(rel_score)]
            if max_per_image > 0:
                det_box = detection_filter(det_box, None, max_per_image)
            for j in xrange(1, len(det_box)):
                all_boxes[j][i] = det_box[j]

        elif args.frame in {'mgn'}:
            det_box, det_grasps = objgrasp_inference(cls_prob[0].data,
                        bbox_pred[0].data if bbox_pred is not None else bbox_pred,
                        grasp_prob.data, grasp_loc.data, data_batch[1][0].data, boxes[0].data,
                        class_agnostic=args.class_agnostic,
                        g_box_prior=grasp_all_anchors.data, for_vis=False, topN_g = 1)
            if args.vis:
                vis_boxes, vis_grasps = objgrasp_inference(cls_prob[0].data,
                             bbox_pred[0].data if bbox_pred is not None else bbox_pred,
                             grasp_prob.data, grasp_loc.data, data_batch[1][0].data, boxes[0].data,
                             class_agnostic=args.class_agnostic,
                             g_box_prior=grasp_all_anchors.data, for_vis=True, topN_g=3)
                if vis_boxes.shape[0] > 0:
                    g_inds = torch.Tensor(np.arange(vis_boxes.shape[0])).unsqueeze(1).repeat(1, vis_grasps.shape[1])
                else:
                    g_inds = torch.Tensor([])
                data_list = [data_batch[0][0], data_batch[1][0], torch.Tensor(vis_boxes),
                             torch.Tensor(vis_grasps).view(-1, vis_grasps.shape[-1]), None, None,
                             g_inds.long().view(-1)]

            if max_per_image > 0:
                det_box, det_grasps = detection_filter(det_box, det_grasps, max_per_image)
            for j in xrange(1, len(det_box)):
                all_boxes[j][i] = det_box[j]
                all_grasp[j][i] = det_grasps[j]

        elif args.frame in {'all_in_one'}:
            # detected_box is a list of boxes. len(list) = num_classes
            det_box, det_grasps = objgrasp_inference(cls_prob[0].data,
                                                     bbox_pred[0].data if bbox_pred is not None else bbox_pred,
                                                     grasp_prob.data, grasp_loc.data, data_batch[1][0].data,
                                                     boxes[0].data,
                                                     class_agnostic=args.class_agnostic,
                                                     g_box_prior=grasp_all_anchors.data, for_vis=False, topN_g=1)
            if args.vis:
                # for visual manipulation relationship detection algorithms
                det_res = all_rel[-1]
                vis_boxes, vis_grasps = objgrasp_inference(cls_prob[0].data,
                             bbox_pred[0].data if bbox_pred is not None else bbox_pred,
                             grasp_prob.data, grasp_loc.data, data_batch[1][0].data, boxes[0].data,
                             class_agnostic=args.class_agnostic,
                             g_box_prior=grasp_all_anchors.data, for_vis=True, topN_g=3)
                if vis_boxes.shape[0] > 0:
                    g_inds = torch.Tensor(np.arange(vis_boxes.shape[0])).unsqueeze(1).repeat(1, vis_grasps.shape[1])
                else:
                    g_inds = torch.Tensor([])
                rel_mat, rel_score = rel_prob_to_mat(det_res[2], vis_boxes.shape[0])
                data_list = [data_batch[0][0], data_batch[1][0], torch.Tensor(vis_boxes),
                             torch.Tensor(vis_grasps).view(-1, vis_grasps.shape[-1]),
                             torch.Tensor([vis_boxes.shape[0]]), None,
                             torch.Tensor(rel_mat), g_inds.long().view(-1), torch.Tensor(rel_score)]
            if max_per_image > 0:
                det_box, det_grasps = detection_filter(det_box, det_grasps, max_per_image)
            for j in xrange(1, len(det_box)):
                all_boxes[j][i] = det_box[j]
                all_grasp[j][i] = det_grasps[j]
        elif args.frame in {'fcgn'}:
            det_grasps = grasp_inference(cls_prob[0].data, bbox_pred[0].data, data_batch[1][0].data, box_prior = boxes[0].data, topN = 1)
            all_grasp[1][i] = det_grasps
            if args.vis:
                data_list = [data_batch[0][0], data_batch[1][0], torch.Tensor(det_grasps)]
        else:
            raise RuntimeError("Illegal algorithm.")

        if args.vis:
            im_vis = vis_gt(data_list, visualizer, args.frame)
            # img_name = id_number_to_name[data_batch[1][0][4].item()].split("/")[-1]
            img_name = str(int(data_batch[1][0][4].item())) + ".jpg"
            # When using cv2.imwrite, channel order should be BGR
            cv2.imwrite(os.path.join(data_vis_dir, img_name), im_vis[:, :, ::-1])

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                     .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    print('Evaluating detections')
    result = {}
    if args.frame in {'fcgn'} or 'cornell' in args.dataset or 'jacquard' in args.dataset:
        result["Acc_grasp"] = imdb.evaluate_detections(all_grasp, output_dir)
        result["Main_Metric"] = result["Acc_grasp"]
    else:
        with open("det_res.pkl", "wb") as f:
            pickle.dump(all_boxes, f)
        result["mAP"] = imdb.evaluate_detections(all_boxes, output_dir)
        if args.frame in {'ssd', 'faster_rcnn', 'fpn', 'efc_det'}:
            result["Main_Metric"] = result["mAP"]

    if args.frame in {'mgn', "all_in_one"}:
        # when using mgn in single-object grasp dataset, we only use accuracy to measure the performance instead of mAP.
        if 'cornell' in args.dataset or 'jacquard' in args.dataset:
            pass
        else:
            print('Evaluating grasp detection results')
            oag = False if Network.use_objdet_branch else True
            grasp_MRFPPI, mean_MRFPPI, key_point_MRFPPI, mAPgrasp = \
                        imdb.evaluate_multigrasp_detections(all_boxes, all_grasp, object_class_agnostic = oag)
            print('Mean Log-Average Miss Rate: %.4f' % np.mean(np.array(mean_MRFPPI)))
            result["mAP_grasp"] = mAPgrasp
            if args.frame == 'mgn':
                result["Main_Metric"] = mAPgrasp

    if args.frame in {"faster_rcnn_vmrn", "ssd_vmrn", "all_in_one"}:
        print('Evaluating relationships')
        orec, oprec, imgprec, imgprec_difobjnum = imdb.evaluate_relationships(all_rel)
        print("object recall:   \t%.4f" % orec)
        print("object precision:\t%.4f" % oprec)
        print("image acc:       \t%.4f" % imgprec)
        print("image acc for images with different object numbers (2,3,4,5):")
        print("%s\t%s\t%s\t%s\t" % tuple(imgprec_difobjnum))
        result["Obj_Recall_Rel"] = orec
        result["Obj_Precision_Rel"] = oprec
        result["Img_Acc_Rel"] = imgprec
        result["Main_Metric"] = imgprec

    # TODO: implement all_in_one's metric for evaluation

    end = time.time()
    print("test time: %0.4fs" % (end - start))
    return result

def train():
    # check cuda devices
    if not torch.cuda.is_available():
        assert RuntimeError("Training can only be done by GPU. Please use --cuda to enable training.")
    if torch.cuda.is_available() and not args.cuda:
        assert RuntimeError("You have a CUDA device, so you should probably run with --cuda")

    # init random seed
    np.random.seed(cfg.RNG_SEED)

    # init logger
    # TODO: RESUME LOGGER
    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        current_t = time.strftime("%Y_%m_%d") + "_" + time.strftime("%H:%M:%S")
        logger = Logger(os.path.join('.', 'logs', current_t + "_" + args.frame + "_" + args.dataset + "_" + args.net))

    # init dataset
    imdb, roidb, ratio_list, ratio_index, cls_list = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sampler_batch = sampler(train_size, args.batch_size)
    iters_per_epoch = int(train_size / args.batch_size)
    if args.frame in {"fpn", "faster_rcnn", "efc_det"}:
        dataset = fasterrcnnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"ssd"}:
        dataset = ssdbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"ssd_vmrn", "vam"}:
        dataset = svmrnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"faster_rcnn_vmrn"}:
        dataset = fvmrnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"fcgn"}:
        dataset = fcgnbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"all_in_one"}:
        dataset = fallinonebatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    elif args.frame in {"mgn"}:
        dataset = roignbatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           len(cls_list), training=True, cls_list=cls_list, augmentation=cfg.TRAIN.COMMON.AUGMENTATION)
    else:
        raise RuntimeError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    args.iter_per_epoch = int(len(roidb) / args.batch_size)

    # init output directory for model saving
    output_dir = args.save_dir + "/" + args.dataset + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.vis:
        visualizer = dataViewer(cls_list)
        data_vis_dir = os.path.join(args.save_dir, args.dataset, 'data_vis', 'train')
        if not os.path.exists(data_vis_dir):
            os.makedirs(data_vis_dir)
        id_number_to_name = {}
        for r in roidb:
            id_number_to_name[r["img_id"]] = r["image"]

    # init network
    Network, optimizer = init_network(args, len(cls_list))

    # init variables
    current_result, best_result, loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rel_pred, \
        loss_grasp_box, loss_grasp_cls, fg_cnt, bg_cnt, fg_grasp_cnt, bg_grasp_cnt = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
    save_flag, rois, rpn_loss_cls, rpn_loss_box, rel_loss_cls, cls_prob, bbox_pred, rel_cls_prob, loss_bbox, loss_cls, \
        rois_label, grasp_cls_loss, grasp_bbox_loss, grasp_conf_label = \
            False, None,None,None,None,None,None,None,None,None,None,None,None,None

    # initialize step counter
    if args.resume:
        step = args.checkpoint
    else:
        step = 0
        
    for epoch in range(args.checkepoch, args.max_epochs + 1):
        # setting to train mode
        Network.train()

        start_epoch_time = time.time()
        start = time.time()

        data_iter = iter(dataloader)
        while(True):

            if step >= iters_per_epoch:
                break

            # get data batch
            data_batch = next(data_iter)
            if args.vis:
                for i in range(data_batch[0].size(0)):
                    data_list = [data_batch[d][i] for d in range(len(data_batch))]
                    im_vis = vis_gt(data_list, visualizer, args.frame, train_mode=True)
                    # img_name = id_number_to_name[data_batch[1][i][4].item()].split("/")[-1]
                    img_name = str(int(data_batch[1][i][4].item())) + ".jpg"
                    # When using cv2.imwrite, channel order should be BGR
                    cv2.imwrite(os.path.join(data_vis_dir, img_name), im_vis[:, :, ::-1])
            # ship to cuda
            if args.cuda:
                data_batch = makeCudaData(data_batch)

            # setting gradients to zeros
            Network.zero_grad()
            optimizer.zero_grad()

            # forward process
            if args.frame == 'faster_rcnn_vmrn':
                rois, cls_prob, bbox_pred, rel_cls_prob, rpn_loss_cls, rpn_loss_box, loss_cls, \
                            loss_bbox, rel_loss_cls, reg_loss, rois_label = Network(data_batch)
                loss = (rpn_loss_cls + rpn_loss_box + loss_cls + loss_bbox + reg_loss + rel_loss_cls).mean()
            elif args.frame == 'faster_rcnn' or args.frame == 'fpn':
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, \
                            rois_label = Network(data_batch)
                loss = (rpn_loss_cls + rpn_loss_box + loss_cls + loss_bbox).mean()
            elif args.frame == 'fcgn':
                bbox_pred, cls_prob, loss_bbox, loss_cls, rois_label,rois = Network(data_batch)
                loss = (loss_bbox + loss_cls).mean()
            elif args.frame == 'mgn':
                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rois_label, grasp_loc, \
                grasp_prob, grasp_bbox_loss, grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
                loss = rpn_loss_box.mean() + rpn_loss_cls.mean() + loss_cls.mean() + loss_bbox.mean() + \
                       cfg.MGN.OBJECT_GRASP_BALANCE * (grasp_bbox_loss.mean() + grasp_cls_loss.mean())
            elif args.frame == 'all_in_one':
                rois, cls_prob, bbox_pred, rel_cls_prob, rpn_loss_cls, rpn_loss_box, loss_cls, loss_bbox, rel_loss_cls, reg_loss, rois_label, \
                grasp_loc, grasp_prob, grasp_bbox_loss,grasp_cls_loss, grasp_conf_label, grasp_all_anchors = Network(data_batch)
                loss = (rpn_loss_box + rpn_loss_cls + loss_cls + loss_bbox + rel_loss_cls + reg_loss \
                       + cfg.MGN.OBJECT_GRASP_BALANCE * grasp_bbox_loss + grasp_cls_loss).mean()

            elif args.frame in {'ssd', 'efc_det'}:
                bbox_pred, cls_prob, loss_bbox, loss_cls = Network(data_batch)
                loss = loss_bbox.mean() + loss_cls.mean()
            elif args.frame == 'ssd_vmrn' or args.frame == 'vam':
                bbox_pred, cls_prob, rel_result, loss_bbox, loss_cls, rel_loss_cls, reg_loss = Network(data_batch)
                loss = (loss_cls + loss_bbox + rel_loss_cls + reg_loss).mean()
            loss_temp += loss.data.item()

            # backward process
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(Network, 10.)
            optimizer.step()
            step += 1

            # record training information
            if len(args.mGPUs) > 0:
                if rpn_loss_cls is not None and isinstance(rpn_loss_cls, torch.Tensor):
                    loss_rpn_cls += rpn_loss_cls.mean().data[0].item()
                if rpn_loss_box is not None and isinstance(rpn_loss_box, torch.Tensor):
                    loss_rpn_box += rpn_loss_box.mean().data[0].item()
                if loss_cls is not None and isinstance(loss_cls, torch.Tensor):
                    loss_rcnn_cls += loss_cls.mean().data[0].item()
                if loss_bbox is not None and isinstance(loss_bbox, torch.Tensor):
                    loss_rcnn_box += loss_bbox.mean().data[0].item()
                if rel_loss_cls is not None and isinstance(rel_loss_cls, torch.Tensor):
                    loss_rel_pred += rel_loss_cls.mean().data[0].item()
                if grasp_cls_loss is not None and isinstance(grasp_cls_loss, torch.Tensor):
                    loss_grasp_cls += grasp_cls_loss.mean().data[0].item()
                if grasp_bbox_loss is not None and isinstance(grasp_bbox_loss, torch.Tensor):
                    loss_grasp_box += grasp_bbox_loss.mean().data[0].item()
                if rois_label is not None and isinstance(rois_label, torch.Tensor):
                    tempfg = torch.sum(rois_label.data.ne(0))
                    fg_cnt += tempfg
                    bg_cnt += (rois_label.data.numel() - tempfg)
                if grasp_conf_label is not None and isinstance(grasp_conf_label, torch.Tensor):
                    tempfg = torch.sum(grasp_conf_label.data.ne(0))
                    fg_grasp_cnt += tempfg
                    bg_grasp_cnt += (grasp_conf_label.data.numel() - tempfg)
            else:
                if rpn_loss_cls is not None and isinstance(rpn_loss_cls, torch.Tensor):
                    loss_rpn_cls += rpn_loss_cls.item()
                if rpn_loss_cls is not None and isinstance(rpn_loss_cls, torch.Tensor):
                    loss_rpn_box += rpn_loss_box.item()
                if loss_cls is not None and isinstance(loss_cls, torch.Tensor):
                    loss_rcnn_cls += loss_cls.item()
                if loss_bbox is not None and isinstance(loss_bbox, torch.Tensor):
                    loss_rcnn_box += loss_bbox.item()
                if rel_loss_cls is not None and isinstance(rel_loss_cls, torch.Tensor):
                    loss_rel_pred += rel_loss_cls.item()
                if grasp_cls_loss is not None and isinstance(grasp_cls_loss, torch.Tensor):
                    loss_grasp_cls += grasp_cls_loss.item()
                if grasp_bbox_loss is not None and isinstance(grasp_bbox_loss, torch.Tensor):
                    loss_grasp_box += grasp_bbox_loss.item()
                if rois_label is not None and isinstance(rois_label, torch.Tensor):
                    tempfg = torch.sum(rois_label.data.ne(0))
                    fg_cnt += tempfg
                    bg_cnt += (rois_label.data.numel() - tempfg)
                if grasp_conf_label is not None and isinstance(grasp_conf_label, torch.Tensor):
                    tempfg = torch.sum(grasp_conf_label.data.ne(0))
                    fg_grasp_cnt += tempfg
                    bg_grasp_cnt += (grasp_conf_label.data.numel() - tempfg)

            if Network.iter_counter % args.disp_interval == 0:
                end = time.time()
                loss_temp /= args.disp_interval
                loss_rpn_cls /= args.disp_interval
                loss_rpn_box /= args.disp_interval
                loss_rcnn_cls /= args.disp_interval
                loss_rcnn_box /= args.disp_interval
                loss_rel_pred /= args.disp_interval
                loss_grasp_cls /= args.disp_interval
                loss_grasp_box /= args.disp_interval

                print("[session %d][epoch %2d][iter %4d/%4d] \n\t\t\tloss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, optimizer.param_groups[0]['lr']))
                print('\t\t\ttime cost: %f' % (end - start,))
                if rois_label is not None:
                    print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt, bg_cnt))
                if grasp_conf_label is not None:
                    print("\t\t\tgrasp_fg/grasp_bg=(%d/%d)" % (fg_grasp_cnt, bg_grasp_cnt))
                if rpn_loss_box is not None and rpn_loss_cls is not None:
                    print("\t\t\trpn_cls: %.4f\n\t\t\trpn_box: %.4f\n\t\t\trcnn_cls: %.4f\n\t\t\trcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                else:
                    print("\t\t\trcnn_cls: %.4f\n\t\t\trcnn_box %.4f" \
                          % (loss_rcnn_cls, loss_rcnn_box))
                if rel_loss_cls is not None:
                    print("\t\t\trel_loss %.4f" \
                          % (loss_rel_pred,))
                if grasp_cls_loss is not None and grasp_bbox_loss is not None:
                    print("\t\t\tgrasp_cls: %.4f\n\t\t\tgrasp_box %.4f" \
                          % (loss_grasp_cls, loss_grasp_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                    }
                    if rpn_loss_cls:
                        info['loss_rpn_cls'] = loss_rpn_cls
                    if rpn_loss_box:
                        info['loss_rpn_box'] = loss_rpn_box
                    if rel_loss_cls:
                        info['loss_rel_pred'] = loss_rel_pred
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, Network.iter_counter)

                loss_temp = 0.
                loss_rpn_cls = 0.
                loss_rpn_box = 0.
                loss_rcnn_cls = 0.
                loss_rcnn_box = 0.
                loss_rel_pred = 0.
                loss_grasp_box = 0.
                loss_grasp_cls = 0.
                fg_cnt = 0.
                bg_cnt = 0.
                fg_grasp_cnt = 0.
                bg_grasp_cnt = 0.
                start = time.time()

            # adjust learning rate
            if args.lr_decay_step == 0:
                # clr = lr / (1 + decay * n) -> lr_n / lr_n+1 = (1 + decay * (n+1)) / (1 + decay * n)
                decay = (1 + args.lr_decay_gamma * Network.iter_counter) / (1 + args.lr_decay_gamma * (Network.iter_counter + 1))
                adjust_learning_rate(optimizer, decay)
            elif Network.iter_counter % (args.lr_decay_step) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)

            # test and save
            if (Network.iter_counter - 1) % cfg.TRAIN.COMMON.SNAPSHOT_ITERS == 0:
                # test network and record results

                if cfg.TRAIN.COMMON.SNAPSHOT_AFTER_TEST:
                    Network.eval()
                    with torch.no_grad():
                        current_result = evalute_model(Network, args.imdbval_name, args)
                    torch.cuda.empty_cache()
                    if args.use_tfboard:
                        for key in current_result.keys():
                            logger.scalar_summary(key, current_result[key], Network.iter_counter)
                    Network.train()
                    if current_result["Main_Metric"] > best_result:
                        best_result = current_result["Main_Metric"]
                        save_flag = True
                else:
                    save_flag = True

                if save_flag:
                    save_name = os.path.join(output_dir, args.frame + '_{}_{}_{}.pth'.format(args.session, epoch, step))
                    save_checkpoint({
                        'session': args.session,
                        'model': Network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'pooling_mode': cfg.RCNN_COMMON.POOLING_MODE,
                        'class_agnostic': args.class_agnostic,
                    }, save_name)
                    print('save model: {}'.format(save_name))
                    save_flag = False

        end_epoch_time = time.time()
        print("Epoch finished. Time costing: ", end_epoch_time - start_epoch_time, "s")
        step = 0 # reset step counter

def test():

    # check cuda devices
    if not torch.cuda.is_available():
        assert RuntimeError("Training can only be done by GPU. Please use --cuda to enable training.")
    if torch.cuda.is_available() and not args.cuda:
        assert RuntimeError("You have a CUDA device, so you should probably run with --cuda")

    # init output directory for model saving
    output_dir = args.save_dir + "/" + args.dataset + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _, _, _, _, cls_list = combined_roidb(args.imdbval_name, training = False)

    args.iter_per_epoch = None
    # init network
    Network, optimizer = init_network(args, len(cls_list))
    Network.eval()
    evalute_model(Network, args.imdbval_name, args)

# def test_testcode():
#     with open('det_res.pkl', 'rb') as f:
#         all_boxes = pickle.load(f)
#     imdb, roidb, ratio_list, ratio_index = combined_roidb('coco_2017_val+vmrd_compv1_test', False)
#     imdb.evaluate_detections(all_boxes, 'output/coco+vmrd/res101')

if __name__ == '__main__':

    # init arguments
    args = read_cfgs()
    assert args.frame in LEGAL_FRAMES, "Illegal algorithm name."

    if args.test:
        args.resume = True
        test()
    else:
        train()