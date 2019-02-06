import os
import random
import PIL
from PIL import Image
import numpy as np
import copy

def normalize_depth(depth):
    d_max = np.max(depth[np.where(depth>0)])
    d_min = np.min(depth[np.where(depth>0)])
    depth = (depth - d_min) * 255 / (d_max - d_min)
    depth[np.where(depth <= 0)] = 0
    return depth.astype(np.uint8)

def read_file_paths(jacquard_dir):
    sub_sets = os.listdir(jacquard_dir)
    temp = copy.deepcopy(sub_sets)
    for sub_set in temp:
        if os.path.isfile(os.path.join(jacquard_dir, sub_set)):
            sub_sets.remove(sub_set)
    # generate file list
    file_list = []
    for sub_set in sub_sets:
        sub_set_path = os.path.join(jacquard_dir, sub_set)
        dirs = os.listdir(sub_set_path)
        for dir in dirs:
            dir_path = os.path.join(sub_set_path, dir)
            files = os.listdir(dir_path)
            for file in files:
                if file.split('.')[-1] == 'txt':
                    basename = file[:-len('_grasps.txt')]
                    filepath = os.path.join(cwd, dir_path, basename)
                    file_list += [filepath]
    return file_list

def generate_depth_png(file_list):
    counter = 0
    for file in file_list:
        counter += 1
        print('current: ' + str(counter) + ' totally: ' + str(len(file_list)) + ' current: ' + file)
        depth = Image.open(file + '_stereo_depth.tiff')
        depth_array = np.array(depth)
        depth_array = normalize_depth(depth_array)
        depth_array = np.repeat(np.expand_dims(depth_array, axis = 2), 3, axis = 2)
        D = Image.fromarray(depth_array).convert('RGB')
        D.save(file + '_Depth.png')

def generate_rgd_png(file_list):
    counter = 0
    for file in file_list:
        counter += 1
        print('current: '+ str(counter) + ' totally: ' + str(len(file_list)) + ' current: '+ file)
        depth = Image.open(file + '_stereo_depth.tiff')
        RGB = Image.open(file + '_RGB.png')
        depth_array = np.array(depth)
        depth_array = normalize_depth(depth_array)
        rgd_array = np.array(RGB)
        rgd_array[:,:,2] = depth_array
        RGD = Image.fromarray(rgd_array).convert('RGB')
        RGD.save(file + '_RGD.png')

def genenrate_file_list(jacquard_dir, n_fold):

    file_list = []
    sub_sets = os.listdir(jacquard_dir)
    # exclude files.
    temp = copy.deepcopy(sub_sets)
    for sub_set in temp:
        if os.path.isfile(os.path.join(jacquard_dir, sub_set)):
            sub_sets.remove(sub_set)

    n_labels = 0
    for sub_set in sub_sets:
        sub_set_path = jacquard_dir + '/' + sub_set
        dirs = os.listdir(sub_set_path)
        for dir in dirs:
            dir_path = sub_set_path + '/' + dir
            files = os.listdir(dir_path)
            for file in files:
                if file.split('.')[-1] == 'txt':
                    basename = file[:-len('_grasps.txt')]
                    filepath = cwd + '/' + dir_path + '/' + basename
                    file_list += [filepath]
                    temp_file = open(filepath + '_grasps.txt', 'r')
                    labels = temp_file.readlines()
                    if max(n_labels, len(labels)) > n_labels:
                        print(temp_file)
                    n_labels = max(n_labels, len(labels))

    print("max label number: "+ str(n_labels))

    # create new trainval.txt
    train_file_list = []
    for i in range(1, 1 + n_fold):
        train_file_list += [open('data/Jacquard/trainval_' + str(i) + '.txt', 'w')]

    random.shuffle(file_list)
    n_file = len(file_list)
    n_file_per_fold = int(n_file / n_fold)

    begin_file = 0
    for fold in range(n_fold):
        for i in range(begin_file, min(begin_file + n_file_per_fold, n_file)):
            train_file_list[fold].write(file_list[i] + '\n')
        begin_file = begin_file + n_file_per_fold
        if begin_file >= n_file:
            break


n_fold = 5
cwd = os.getcwd()
jacquard_dir = 'data/Jacquard'
jacquard_dir = os.path.join(cwd, jacquard_dir)
file_list = read_file_paths(jacquard_dir)
# generate_rgd_png(file_list)
# genenrate_file_list(jacquard_dir, n_fold)
generate_depth_png(file_list)