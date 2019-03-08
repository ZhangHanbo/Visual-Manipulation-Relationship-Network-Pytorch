import cv2
import numpy as np
import os

'''
bbox_file = open("/Users/hanbo/Downloads/processedData/bbox.txt", "w")

bg_map_file = open("/Users/hanbo/Downloads/processedData/backgroundMapping.txt", "r")
bg_map_list = bg_map_file.readlines()
bg_map = {}
for i in bg_map_list:
    a = i.split(" ")
    bg_map[a[0]] = a[1][:-1]

begin = 100
cropsize = 320
img_dir = "/Users/hanbo/Downloads/processedData/Images"
bg_dir = "/Users/hanbo/Downloads/processedData/Background"
bg_list = os.listdir(bg_dir)
out_dir = "/Users/hanbo/Downloads/processedData/Output"
thresh = 6000
img_list = os.listdir(img_dir)
for img_name in img_list:
    bg_name = bg_map[img_name][:4] + bg_map[img_name][5:]
    img = cv2.imread(os.path.join(img_dir, img_name))
    img2show = img.copy()
    img = img[begin+1:begin+cropsize, begin+1:begin+cropsize, :]
    img = img.astype(np.float32)
    bg = cv2.imread(os.path.join(bg_dir, bg_name))
    bg = bg.astype(np.float32)
    bg = bg[begin+1:begin+cropsize, begin+1:begin+cropsize, :]

    if int(img_name[4:7]) >= 816 and int(img_name[4:7]) <= 875:
        img[324-begin:324-begin+10, 318-begin:318-begin+10] = bg[324-begin:324-begin+10, 318-begin:318-begin+10]

    dif = img - bg
    dif2 = np.sum(np.power(dif, 2), axis=2)

    obj_seg = np.where(dif2 > thresh)
    xmin = np.min(obj_seg[1]) + begin
    xmax = np.max(obj_seg[1]) + begin
    ymin = np.min(obj_seg[0]) + begin
    ymax = np.max(obj_seg[0]) + begin
    bbox = (xmin, ymin, xmax, ymax)
    # cv2.rectangle(img2show, bbox[0:2], bbox[2:4], (255,0,0),2)
    # cv2.imwrite(os.path.join(out_dir, img_name), img2show)
    bbox_file.write(img_name + ' ' + str(xmin) +' ' + str(ymin) +' ' + str(xmax) +' ' + str(ymax)+'\n')
'''

'''
bbox_file = open("/Users/hanbo/Downloads/processedData/bboxnew3.txt", "r")
bbox_file_modified = open("/Users/hanbo/Downloads/processedData/bbox.txt", "r")
bbox_file_new = open("/Users/hanbo/Downloads/processedData/bboxnew4.txt", "w")
mod_list_file = open("/Users/hanbo/Downloads/processedData/modlistbig.txt", "r")
bbox = bbox_file.readlines()
bbox_modified = bbox_file_modified.readlines()
mod_list = mod_list_file.readlines()

bbox_dict = {}
for box in bbox:
    bbox_dict[box[:12]] = box[12:]

bbox_modified_dict = {}
for box in bbox_modified:
    bbox_modified_dict[box[:12]] = box[12:]

for mod_name in mod_list:
    mod_name = mod_name[:-1]
    bbox_dict['pcd' + mod_name + 'r.png'] = bbox_modified_dict['pcd' + mod_name + 'r.png']

for key in bbox_dict.keys():
    bbox_file_new.write(key + bbox_dict[key])
'''


img_dir = "/Users/hanbo/Downloads/processedData/Images"
out_dir = "/Users/hanbo/Downloads/processedData/Output"
bbox_file = open("/Users/hanbo/Downloads/processedData/bboxnew4.txt", "r")
bbox = bbox_file.readlines()
for box in bbox:
    box_ = box[:-1].split(' ')
    img_name = box_[0]
    img = cv2.imread(os.path.join(img_dir, img_name))
    box_ = box_[1:]
    for i,v in enumerate(box_):
        box_[i] = int(v)
    box_ = tuple(box_)
    cv2.rectangle(img, box_[0:2], box_[2:4], (255,0,0),2)
    cv2.imwrite(os.path.join(out_dir, img_name), img)

