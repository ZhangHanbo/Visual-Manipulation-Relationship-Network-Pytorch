#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg
import copy
import numpy as np
OPENCV_IMSHOW = False  # set this to True, if you want a GUI display for results


def ground():
    rospy.init_node('InteractiveGrounding', anonymous=True)

    # [INPUT] image
    path = 'images/table.png'

    # wait for action servers to show up
    # if you are stuck here, that means the servers are not ready
    # or your network connection is broken
    load_client = actionlib.SimpleActionClient('dense_refexp_load', action_controller.msg.DenseRefexpLoadAction)
    rospy.loginfo("1. Waiting for dense_refexp_load action server ...")
    load_client.wait_for_server()

    query_client = actionlib.SimpleActionClient('dense_refexp_query', action_controller.msg.DenseRefexpQueryAction)
    rospy.loginfo("2. Waiting for dense_refexp_query action server ...")
    query_client.wait_for_server()

    rospy.loginfo("Ground servers found!")

    # load image, extract and store feature vectors for objects
    # this can be done once per image. everytime the scene changes, you have to reload the image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    msg_frame = CvBridge().cv2_to_imgmsg(img, "rgb8")
    goal = action_controller.msg.DenseRefexpLoadGoal(msg_frame)
    load_client.send_goal(goal)
    load_client.wait_for_result()
    load_result = load_client.get_result()

    # load results: bounding boxes and self-ref captions for all objects (before grounding)
    boxes = np.reshape(load_result.boxes, (-1, 4))
    captions = np.array(load_result.captions)

    incorrect_idxs = []
    if OPENCV_IMSHOW:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', img.shape[1], img.shape[0])

    # once the image has been loaded up, the grounding server can be repeatedly queried with expressions
    while True:

        # [INPUT] referring expression
        query = raw_input('Search Query: ').lower()

        # send expression for grounding
        goal = action_controller.msg.DenseRefexpQueryGoal(query, incorrect_idxs)
        query_client.send_goal(goal)
        query_client.wait_for_result()
        query_result = query_client.get_result()

        # grounding results: indexes of mostly bounding boxes
        top_idx = query_result.top_box_idx
        context_boxes_idxs = list(query_result.context_boxes_idxs)
        context_boxes_idxs.append(top_idx)

        # self referrential and relational captions for asking questions
        self_captions = [captions[idx] for idx in context_boxes_idxs]
        rel_captions = query_result.predicted_captions

        # ------------------------------------------------
        # visualization
        draw_img = img.copy()
        for (count, idx) in enumerate(context_boxes_idxs):

            x1 = int(boxes[idx][0])
            y1 = int(boxes[idx][1])
            x2 = int(boxes[idx][0] + boxes[idx][2])
            y2 = int(boxes[idx][1] + boxes[idx][3])

            if count == len(context_boxes_idxs) - 1:
                # top result
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 12)
            else:
                # context boxes
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if OPENCV_IMSHOW:
            cv2.imshow('result', draw_img)
            k = cv2.waitKey(0)
        else:
            cv2.imwrite('./grounding_result.png', draw_img)

        # ------------------------------------------------
        # print captions
        print ""
        rospy.loginfo("Self Referential Captions: ")
        print self_captions
        print ""

        if len(rel_captions) > 0:
            rospy.loginfo("Relational Captions: ")
            print rel_captions
            print ""
        else:
            rospy.logwarn("lib/comprehension_test.py was started without --disambiguate mode!")

    return True

if __name__ == '__main__':
    try:
        ground()

    except rospy.ROSInterruptException:
        pass
