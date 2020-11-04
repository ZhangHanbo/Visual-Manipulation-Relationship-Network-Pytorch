from pycocotools.coco import COCO
import os.path as osp
import json

coco_anno = osp.join("/data1/zhb/coco", 'annotations',
                    'instances' + '_' + "train" + "2017" + '.json')
coco_train = COCO(coco_anno)
cls_id_to_name = dict(zip([cat["id"] for cat in coco_train.dataset["categories"]],
                          [cat["name"] for cat in coco_train.dataset["categories"]]))

vmrd_classes = ['__background__',  # always index 0
                     'box', 'banana', 'notebook', 'screwdriver', 'toothpaste', 'apple',
                     'stapler', 'mobile phone', 'bottle', 'pen', 'mouse', 'umbrella',
                     'remote controller', 'cans', 'tape', 'knife', 'wrench', 'cup', 'charger',
                     'badminton', 'wallet', 'wrist developer', 'glasses', 'pliers', 'headset',
                     'toothbrush', 'card', 'paper', 'towel', 'shaver', 'watch']
desired_cats = ["umbrella", "bottle", "cup", "knife", "banana", "apple",
                "remote", "mouse", "cell phone", "book", "toothbrush"]
vmrd_cat_to_id = dict(zip(vmrd_classes, list(range(len(vmrd_classes)))))
vmrd_cat_to_id['remote'] = vmrd_cat_to_id['remote controller']
vmrd_cat_to_id["cell phone"] = vmrd_cat_to_id["mobile phone"]
vmrd_cat_to_id["book"] = vmrd_cat_to_id["notebook"]

labels = coco_train.dataset

selected = {u'images': [], u'annotations': [], u'categories': desired_cats}
for ann in coco_train.dataset["annotations"]:
    if cls_id_to_name[ann['category_id']] in desired_cats:
        new_cat_id = vmrd_cat_to_id[cls_id_to_name[ann['category_id']]]
        ann["category_id"] = new_cat_id
        selected["annotations"].append(ann)
        selected["images"].append(ann["image_id"])

im_ind_to_im = dict(zip([im["id"] for im in coco_train.dataset["images"]], coco_train.dataset["images"]))
selected["images"] = [im_ind_to_im[id] for id in list(set(selected["images"]))]

with open("objects_coco.json", "wb") as f:
    json.dump(selected, f)

pass