import sys
import rospy
from model.utils.data_viewer import dataViewer
from model.utils.net_utils import leaf_and_descendant_stats, inner_loop_planning, relscores_to_visscores

from faster_rcnn_detector.srv import ObjectDetection
from vmrn_old.srv import VmrDetection
from ingress_msgs.srv import MAttNetGrounding

import cv2
from cv_bridge import CvBridge
br = CvBridge()

import torch
import numpy as np
import os
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_overlaps

import time
import datetime

BG_SCORE = 0.25

def faster_rcnn_client(img):
    rospy.wait_for_service('faster_rcnn_server')
    try:
        obj_det = rospy.ServiceProxy('faster_rcnn_server', ObjectDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = obj_det(img_msg)
        return res.num_box, res.bbox, res.cls
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def vmrn_client(img, bbox):
    rospy.wait_for_service('vmrn_server')
    try:
        vmr_det = rospy.ServiceProxy('vmrn_server', VmrDetection)
        img_msg = br.cv2_to_imgmsg(img)
        res = vmr_det(img_msg, bbox)
        return res.rel_mat, res.rel_score_mat
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def mattnet_client(img, bbox, cls, expr):
    rospy.wait_for_service('mattnet_server')
    try:
        grounding = rospy.ServiceProxy('mattnet_server', MAttNetGrounding)
        img_msg = br.cv2_to_imgmsg(img)
        res = grounding(img_msg, bbox, cls, expr)
        return res.ground_prob
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def vis_action(action_str, shape):
    im = 255. * np.ones(shape)
    cv2.putText(im, action_str, (0, im.shape[0] / 2),
                cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 0), thickness=2)
    return im

def score_to_prob(score):
    print("Grounding Score: ")
    print(score.tolist())
    prob = torch.nn.functional.softmax(10 * torch.from_numpy(score), dim=0)
    return prob.numpy()

def bbox_filter(bbox, cls):
    # apply NMS
    keep = nms(torch.from_numpy(bbox[:, :-1]), torch.from_numpy(bbox[:, -1]), 0.7)
    keep = keep.view(-1).numpy().tolist()
    for i in range(bbox.shape[0]):
        if i not in keep and bbox[i][-1] > 0.8:
            keep.append(i)
    bbox = bbox[keep]
    cls = cls[keep]
    return bbox, cls

def bbox_match(bbox, prev_bbox):
    # TODO: apply Hungarian algorithm to match boxes
    # match bboxes between two steps.
    ovs = bbox_overlaps(torch.from_numpy(bbox[:, :4]), torch.from_numpy(prev_bbox[:, :4])).numpy()
    cls_mask = np.zeros(ovs.shape, dtype=np.uint8)
    for i, cls in enumerate(bbox[:, -1]):
        cls_mask[i][prev_bbox[:, -1] == cls] = 1
    ovs *= cls_mask
    mapping = np.argmax(ovs, axis=-1)
    matched = (np.max(ovs, axis=-1) > 0.5)
    ind_match_dict = {i: mapping[i] for i in range(mapping.shape[0]) if matched[i]}
    return ind_match_dict

def update_belief(belief, a, ans, data):
    ground_belief = belief["ground_prob"].cpu().numpy()
    leaf_desc_belief = belief["leaf_desc_prob"].cpu().numpy()
    num_box = ground_belief.shape[0] - 1
    ans = ans.lower()

    if a < 2 * num_box:
        return belief
    # Q1
    elif a < 3 * num_box:
        if ans in {"yes", "yeah", "yep", "sure"}:
            ground_belief[:] = 0
            ground_belief[a - 2*num_box] = 1
        elif ans in {"no", "nope", "nah"}:
            ground_belief[a - 2*num_box] = 0
            ground_belief /= np.sum(ground_belief)
    # Q2
    else:
        if ans in {"yes", "yeah", "yep", "sure"}:
            ground_belief[-1] = 0
            ground_belief /= np.sum(ground_belief)
        else:
            ground_belief[:] = 0
            ground_belief[-1] = 1
            img = data["img"]
            bbox = data["bbox"]
            cls = data["cls"]
            expr = ans
            t_ground = mattnet_client(img, bbox, cls, expr)
            bg_score = BG_SCORE
            t_ground += (bg_score,)
            t_ground = score_to_prob(np.array(t_ground))
            t_ground = np.expand_dims(t_ground, 0)
            leaf_desc_belief[:, -1] = (t_ground * leaf_desc_belief[:, :-1]).sum(-1)

    belief["ground_prob"] = torch.from_numpy(ground_belief)
    belief["leaf_desc_prob"] = torch.from_numpy(leaf_desc_belief)
    return belief

def single_step_perception(img, prevs=None):
    tb = time.time()
    obj_result = faster_rcnn_client(img)
    bboxes = np.array(obj_result[1]).reshape(-1, 5)
    cls = np.array(obj_result[2]).reshape(-1, 1)
    bboxes, cls = bbox_filter(bboxes, cls)

    scores = bboxes[:, -1].reshape(-1, 1)
    bboxes = bboxes[:, :4]
    bboxes = np.concatenate([bboxes, cls], axis=-1)

    ind_match_dict = {}
    if prevs is not None:
        ind_match_dict = bbox_match(bboxes, prevs["bbox"])
        not_matched = set(range(bboxes.shape[0])) - set(ind_match_dict.keys())
        ignored = set(range(prevs["bbox"].shape[0])) - set(ind_match_dict.values())
        # ignored = list(ignored - {prevs["actions"]})
        # bboxes = np.concatenate([bboxes, prevs["bbox"][ignored]], axis=0)
        # cls = np.concatenate([cls, prevs["cls"][ignored]], axis=0)
        prevs["qa_his"] = qa_his_mapping(qa_his, ind_match_dict)

    num_box = bboxes.shape[0]

    rel_result = vmrn_client(img, bboxes[:, :4].reshape(-1).tolist())
    rel_mat = np.array(rel_result[0]).reshape((num_box, num_box))
    rel_score_mat = np.array(rel_result[1]).reshape((3, num_box, num_box))
    if prevs is not None:
        # modify the relationship probability according to the new observation
        rel_score_mat[:, ind_match_dict.keys()][:, :, ind_match_dict.keys()] += \
            prevs["rel_score_mat"][:, ind_match_dict.values()][:, :, ind_match_dict.values()]
        rel_score_mat[:, ind_match_dict.keys()][:, :, ind_match_dict.keys()] /= 2

    ground_score = mattnet_client(img, bboxes[:, :4].reshape(-1).tolist(), cls.reshape(-1).tolist(), expr)
    bg_score = BG_SCORE
    ground_score += (bg_score,)
    ground_score = np.array(ground_score)
    if prevs is not None:
        ground_score[ind_match_dict.keys()] += prevs["ground_score"][ind_match_dict.values()]
        ground_score[ind_match_dict.keys()] /= 2
    ground_result = score_to_prob(ground_score)

    # utilize the answered questions to correct grounding results.
    if prevs is not None:
        for k, v in prevs["qa_his"].items():
            if v == 0:
                if k == "bg":
                    # target has already been detected in the last step
                    for i in not_matched:
                        ground_result[i] = 0
                    ground_result[-1] = 0
                else:
                    ground_result[k] = v
            elif v == 1:
                if k == "bg":
                    # target has not been detected in the last step
                    for i in ind_match_dict.keys():
                        ground_result[i] = 0
                else:
                    ground_result[:] = 0
                    ground_result[k] = v
        ground_result /= ground_result.sum()

    print("Perception Time Consuming: " + str(time.time() - tb) + "s")

    if prevs is not None:
        return bboxes, scores, rel_mat, rel_score_mat, ground_score, ground_result, prevs["qa_his"]
    else:
        return bboxes, scores, rel_mat, rel_score_mat, ground_score, ground_result, {}

def execute_action(a):
    pass

def qa_his_mapping(qa_his, ind_match_dict):
    new_qa_his = {}
    for key, v in ind_match_dict.items():
        if v in qa_his.keys():
            new_qa_his[key] = qa_his[v]
    if "bg" in qa_his.keys(): new_qa_his["bg"] = qa_his["bg"]
    if "clue" in qa_his.keys(): new_qa_his["clue"] = qa_his["clue"]
    return new_qa_his

def save_visualization(img, bboxes, rel_mat, rel_score_mat, expr, ground_prob, a, im_id = None):
    ############ visualize
    # resize img for visualization
    scalar = 500. / min(img.shape[:2])
    img_show = cv2.resize(img, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
    vis_bboxes = bboxes * scalar
    vis_bboxes[:, -1] = bboxes[:, -1]
    num_box = bboxes.shape[0]

    # object detection
    cls = bboxes[:, -1]
    object_det_img = data_viewer.draw_objdet(img_show.copy(), vis_bboxes, list(range(cls.shape[0])))

    # relationship detection
    rel_det_img = data_viewer.draw_mrt(img_show.copy(), rel_mat, rel_score=rel_score_mat)
    rel_det_img = cv2.resize(rel_det_img, (img_show.shape[1], img_show.shape[0]))

    # grounding
    print("Grounding Probability: ")
    print(ground_prob.tolist())
    ground_img = data_viewer.draw_grounding_probs(img_show.copy(), expr, vis_bboxes, ground_prob[:-1])
    cv2.imwrite("ground.png", ground_img)

    print("Optimal Action:")
    if a < num_box:
        action_str = "Grasping object " + str(a) + " and ending the program"
    elif a < 2 * num_box:
        action_str = "Grasping object " + str(a - num_box) + " and continuing"
    elif a < 3 * num_box:
        action_str = "Asking Q1 for " + str(a - 2 * num_box) + "th object"
    else:
        action_str = "Asking Q2"
    print(action_str)

    action_img = vis_action(action_str, img_show.shape)
    final_img = np.concatenate([
        np.concatenate([object_det_img, rel_det_img], axis=1),
        np.concatenate([ground_img, action_img], axis=1),
    ], axis=0)

    # save result
    out_dir = "images/output"
    if im_id is None:
        save_name = str(datetime.datetime.now()) + "_result.png"
    else:
        save_name = im_id.split(".")[0] + "_result.png"
    save_path = os.path.join(out_dir, save_name)
    i = 1
    while (os.path.exists(save_path)):
        i += 1
        save_name = im_id.split(".")[0] + "_result_{:d}.png".format(i)
        save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(save_path, final_img)

if __name__ == "__main__":
    classes = ['__background__',  # always index 0
               'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
               'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
               'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
               'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
               'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']
    data_viewer = dataViewer(classes)

    im_id = "36.png"
    expr = "left apple"
    img = cv2.imread("images/" + im_id)

    # first iteration
    bboxes, scores, rel_mat, rel_score_mat, ground_score, ground_result, qa_his = single_step_perception(img)
    num_box = bboxes.shape[0]

    # dummy action for initialization
    a = 3 * num_box + 1
    # outer-loop planning: in each step, grasp the leaf-descendant node.
    while (True):
        vis_rel_score_mat = relscores_to_visscores(rel_score_mat)

        belief = {}
        with torch.no_grad():
            triu_mask = torch.triu(torch.ones(num_box, num_box), diagonal=1)
            triu_mask = triu_mask.unsqueeze(0).repeat(3, 1, 1)
            rel_score_mat = torch.from_numpy(rel_score_mat)
            rel_score_mat *= triu_mask
            belief["leaf_desc_prob"] = leaf_and_descendant_stats(rel_score_mat)
        belief["ground_prob"] = torch.from_numpy(ground_result)

        # inner-loop planning, with a sequence of questions and a last grasping.
        while (True):
            a = inner_loop_planning(belief)
            save_visualization(img, bboxes, rel_mat, vis_rel_score_mat, expr, ground_result, a, im_id=im_id)
            if a < 2 * num_box:
                break
            else:
                ans = raw_input("Your answer: ")
                data = {"img": img,
                        "bbox": bboxes[:, :4].reshape(-1).tolist(),
                        "cls": bboxes[:, 4].reshape(-1).tolist()}

                if a < 3 * num_box:
                    # we use binary variables to encode the answer of q1 questions.
                    qa_his[a - 2 * num_box] = 1 if ans in {"yes", "yeah", "yep", "sure"} else 0
                else:
                    if ans in {"yes", "yeah", "yep", "sure"}:
                        qa_his["bg"] = 0
                    else:
                        # TODO: using Standord Core NLP library to parse the constituency of the sentence.
                        ans = ans[9:]
                        qa_his["bg"] = 1
                        qa_his["clue"] = ans

                belief = update_belief(belief, a, ans, data)

        # execute grasping action
        execute_action(a)
        if a < num_box:
            break
        else:
            im_id = "37.png"
            img = cv2.imread("images/" + im_id)
            prevs = {
                "bbox": bboxes,
                "score": scores,
                "rel_mat": rel_mat,
                "rel_score_mat":rel_score_mat.numpy(),
                "ground_score": ground_score,
                "ground_prob":ground_result,
                "qa_history": qa_his,
                "last_action": a - num_box # the last grasping should not be grasping and ending
            }
            bboxes, scores, rel_mat, rel_score_mat, ground_score, ground_result, qa_his = single_step_perception(img, prevs)
            num_box = bboxes.shape[0]
