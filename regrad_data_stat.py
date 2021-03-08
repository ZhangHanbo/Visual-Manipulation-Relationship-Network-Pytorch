import os
import json
import sys

def generate_split():

    regrad_path = 'data/REGRAD'
    split = 'train'
    scene_list = os.listdir(os.path.join(regrad_path, split))

    split_file = 'train_super_mini.txt'
    f = open(split_file, "w")

    count = 0
    for i in scene_list:
        scene_dir = os.path.join(regrad_path, split, i)
        if os.path.isdir(scene_dir):
            for j in range(1, 10):
                img_dir = os.path.join(scene_dir, str(j))
                if os.path.exists(img_dir):
                    data_list = os.listdir(img_dir)
                    if len(data_list) == 10:
                        f.write("/".join(img_dir.split("/")[3:]) + "\n")
                        count += 1
        if split_file == "train_super_mini.txt" and count > 10000:
            break

def generate_class_list():
    regrad_path = 'data/REGRAD'
    split = 'train'
    scene_list = os.listdir(os.path.join(regrad_path, split))

    classes_file = 'classes.txt'
    f = open(classes_file, "w")

    count = 0
    cats = []
    scene_num = len(scene_list)
    for n, i in enumerate(scene_list):
        print("Finished: {:d}/{:d}\r".format(n, scene_num))

        scene_dir = os.path.join(regrad_path, split, i)
        if os.path.isdir(scene_dir):
            for j in range(1, 10):
                img_dir = os.path.join(scene_dir, str(j))
                if os.path.exists(img_dir):
                    label_file = os.path.join(img_dir, "info.json")
                    with open(label_file, "r") as lf:
                        other_annos = json.load(lf)
                    cats += [a['model_name'] for a in other_annos]

    cats = set(cats)
    for cat in cats:
        f.write(cat + "\n")

generate_class_list()