import rospy
import numpy as np
import os

from rosapi.ingress_client import IngressClient
from demo_vmrn import VMRNDemo
from model.utils.config import read_cfgs, cfg
from model.utils.net_utils import find_all_leaves
from model.rpn.bbox_transform import bbox_xy_to_xywh

import cv2

class IngressVMRNDemo(VMRNDemo):
    def __init__(self, args, model_dir):
        super(IngressVMRNDemo, self).__init__(args, model_dir)
        self.INGRESS = IngressClient()

if __name__ == '__main__':
    # we need to read configs of VMRN that were used in training and also need to be used in this demo
    args = read_cfgs()

    rospy.init_node('InteractiveGrounding', anonymous=True)
    try:
        ingress_client = IngressVMRNDemo(args, os.path.join(args.save_dir + "/" + args.dataset + "/" + args.net))

        while True:
            image_id = raw_input('Image ID: ').lower()
            if image_id == 'break':
                break
            query = raw_input('Search Query: ').lower()
            # read cv image
            test_img_path = os.path.join('images', image_id + ".jpg")
            cv_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
            # VMRN forward process
            obj_box, obj_cls, mrt = ingress_client.VMRN_forward_process(cv_img, save_res=True, id = image_id)
            # INGRESS forward process
            bboxes, top_idx, context_idxs, captions = \
                ingress_client.INGRESS.ground_img_with_bbox(cv_img, bbox_xy_to_xywh(obj_box.copy()).astype(np.int32),
                                                            query, obj_cls)
            # Find all possible paths
            path = find_all_leaves(mrt, top_idx)
            if captions is None:
                # the target has not been detected.
                print("The target has not been detected. Randomly grasping...")
                print(obj_cls[path[-1]])
            else:
                print("target object: " + obj_cls[top_idx])
                print([obj_cls[id] + ", " for id in path])
            # save results
            resim_save_path = os.path.join("images", image_id + "_" + query + "_res.jpg")
            rescap_save_path = os.path.join("images", image_id + "_" + query + "_pol.txt")
            cv_img = ingress_client.data_viewer.draw_single_bbox(cv_img, obj_box[top_idx].astype(np.int32), text_str="Target")
            cv2.imwrite(resim_save_path, cv_img)
            with open(rescap_save_path, "w") as f:
                if captions is None:
                    f.writelines(obj_cls[path[-1]])
                else:
                    f.writelines([obj_cls[id] + ", " for id in path])
            f.close()

            if captions is not None and len(path) == 1:
                # the target is grasped
                break

    except rospy.ROSInterruptException:
        pass
