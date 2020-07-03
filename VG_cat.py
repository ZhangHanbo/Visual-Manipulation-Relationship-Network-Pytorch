import nltk
import json
import copy
import numpy as np
from model.utils.data_viewer import dataViewer
import os
import cv2
from nltk.corpus import wordnet as wn

data_path = '/data1/zhb/vg'

def search_alias(name, alias_sets):
    for alias_set in alias_sets:
        if name in alias_set:
            return alias_set
    return None

def load_obj_labels():
    obj_label_path = os.path.join(data_path, 'objects.json')
    with open(obj_label_path, 'rb') as f:
        objects = json.load(f)
    return objects

def load_obj_synsets():
    obj_syn_file_path = os.path.join(data_path, 'object_synsets.json')
    with open(obj_syn_file_path, 'rb') as f:
        object_synsets = json.load(f)
    return object_synsets

def load_alias_sets():
    obj_alias_file_path = os.path.join(data_path, 'object_alias.txt')
    with open(obj_alias_file_path, 'rb') as f:
        object_alias = f.readlines()

    alias_sets = []
    for alias_set in object_alias:
        alias_set = alias_set[:-1].split(',')
        alias_sets.append(alias_set)

    return alias_sets

def generate_alias_sets(desired_classes):
    object_alias = load_alias_sets()
    desired_classes = copy.deepcopy(desired_classes)
    for i, cls in enumerate(desired_classes):
        if isinstance(cls, list):
            aliases = copy.deepcopy(cls)
            for c in cls:
                sub_alias_set = search_alias(c, object_alias)
                aliases.extend(sub_alias_set if sub_alias_set is not None else [])
            aliases = list(set(aliases))
            desired_classes[i] = aliases
        else:
            aliases = search_alias(cls, object_alias)
            if aliases is not None:
                desired_classes[i] = aliases
            else:
                desired_classes[i] = [cls]
    return desired_classes

def generate_all_hyponyms(synset):
    sub_synset = synset.hyponyms()

    all_hypos = []
    for syn in sub_synset:
        all_hypos.extend(generate_all_hyponyms(syn))

    all_hypos.extend(sub_synset)
    return list(set(all_hypos))

def generate_synsets(desired_classes):
    """
    WARNING: this function can only work with the desired_classes in this code because the wrong labels and synsets in
            VG will cause unpredictable errors if you do not check every class carefully.
    """
    desired_classes = copy.deepcopy(desired_classes)
    names_to_synsets = load_obj_synsets()
    vg_synsets = set(names_to_synsets.values())

    synsets_to_names = dict(zip(vg_synsets, [[] for _ in vg_synsets]))
    for name, syn in names_to_synsets.items():
        synsets_to_names[syn].append(name)

    synsets = []
    for i, clss in enumerate(desired_classes):
        synset = []

        print(clss)
        for cls in clss:
            if cls in names_to_synsets.keys():
                synset.append(names_to_synsets[cls])
                print(cls, names_to_synsets[cls])
                print(wn.synset(names_to_synsets[cls]).definition())

        synset = list(set(synset))

        for syn in copy.deepcopy(synset):
            synset.extend([s.name() for s in generate_all_hyponyms(wn.synset(syn))])
        synsets.append(synset)

        desired_classes[i] = []
        for syn in synset:
            if syn in synsets_to_names.keys():
                desired_classes[i].extend(synsets_to_names[syn])

    return synsets, desired_classes

def count_samples(all_cls, synsets, object_labels):
    """
    :param classes:
    :param object_labels:
    :return:
    """
    classes = all_cls["desired_classes"]
    vmrd_classes = all_cls["vmrd_classes"]
    n_samples = dict(zip(vmrd_classes, [0 for _ in range(len(classes))]))
    obj_num_per_img = np.zeros(shape=len(object_labels))
    for i, l_img in enumerate(object_labels):
        if i % 1000 == 0:
            print("Current: {:d}".format(i) + ". Totally: {:d}".format(len(object_labels)))
        for l_obj in l_img['objects']:
            cls_name = l_obj['names'][0]
            for j, alias in enumerate(classes):
                if cls_name in alias:
                    n_samples[vmrd_classes[j]] += 1
                    obj_num_per_img[i] += 1

    return n_samples, obj_num_per_img

def im_to_dir():
    img_to_dir = {}
    img_set1 = os.listdir(os.path.join(data_path, 'VG_100K'))
    img_set2 = os.listdir(os.path.join(data_path, 'VG_100K_2'))
    for im in img_set1:
        if im[-3:] == 'jpg':
            img_to_dir[im] = os.path.join(data_path, 'VG_100K')
    for im in img_set2:
        if im[-3:] == 'jpg':
            img_to_dir[im] = os.path.join(data_path, 'VG_100K_2')
    return img_to_dir

def filter_data(all_cls, synsets, object_labels, save_path = data_path):
    classes = all_cls["desired_classes"]
    vmrd_classes = all_cls["vmrd_classes"]

    selected_data = []
    for i, l_img in enumerate(object_labels):
        if i % 1000 == 0:
            print("Current: {:d}".format(i) + ". Totally: {:d}".format(len(object_labels)))
        objs = []

        for l_obj in l_img['objects']:
            cls_name = l_obj['names'][0]
            for j, alias in enumerate(classes):
                if cls_name in alias:
                    l_obj['names'][0] = vmrd_classes[j]
                    objs.append(l_obj)
                    break

        if len(objs) > 0:
            selected_data.append({u"image_id": l_img['image_id'], u'objects': objs})

    if save_path:
        with open(os.path.join(save_path, "objects_vmrd.json"), "wb") as f:
            json.dump(selected_data, f)

    return selected_data

def do_data_statistics(desired_classes):
    vmrd_classes = [cls if isinstance(cls, str) else cls[0] for cls in desired_classes]
    desired_classes = generate_alias_sets(desired_classes)
    synsets, desired_classes = generate_synsets(desired_classes)
    desired_classes = generate_alias_sets(desired_classes)
    desired_classes = {
        "desired_classes" : desired_classes,
        "vmrd_classes" : vmrd_classes
    }
    print("Loading labels...")
    object_labels = load_obj_labels()
    print("Do data counting...")
    n_samples, obj_nums = count_samples(desired_classes, synsets, object_labels)
    print(n_samples)
    max_obj_num = np.max(obj_nums)
    for i in range(int(max_obj_num) + 1):
        print("Number of Images Containing {:d} objects: {:d}".format(i, (obj_nums == i).sum()))

def do_data_filtering(desired_classes):
    vmrd_classes = [cls if isinstance(cls, str) else cls[0] for cls in desired_classes]
    desired_classes = generate_alias_sets(desired_classes)
    synsets, desired_classes = generate_synsets(desired_classes)
    desired_classes = generate_alias_sets(desired_classes)
    desired_classes = {
        "desired_classes": desired_classes,
        "vmrd_classes": vmrd_classes
    }
    print("Loading labels...")
    object_labels = load_obj_labels()
    selected = filter_data(desired_classes, synsets, object_labels, save_path='/data1/zhb/vg')
    return selected

def vis_gt(labels, cls_list, vis_dir = 'vis_gt'):
    vis_data_path = os.path.join(data_path, vis_dir)
    if not os.path.exists(vis_data_path):
        os.makedirs(vis_data_path)

    data_viewer = dataViewer([cls if isinstance(cls, str) else cls[0] for cls in cls_list])
    cls_to_ind = dict(zip(cls_list, list(range(len(cls_list)))))
    im_dir = im_to_dir()

    for l in labels:
        num_obj = len(l['objects'])
        if num_obj > 0:
            im_name = str(l['image_id']) + '.jpg'
            im_path = os.path.join(im_dir[im_name], im_name)
            im = cv2.imread(im_path)

            bboxes = np.zeros(shape=(num_obj, 5), dtype=np.float32)
            for i, obj in enumerate(l['objects']):
                x1, y1, x2, y2 = obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']
                cls_name = obj['names'][0]
                if cls_name in cls_list:
                    cls_ind = cls_to_ind[cls_name]
                    bboxes[i] = [x1, y1, x2, y2, cls_ind]

            if (bboxes[:, 0] > 0).sum() > 0:
                im = data_viewer.draw_objdet(im, bboxes)
                cv2.imwrite(os.path.join(vis_data_path, im_name), im)

def interactive_vis(labels, vis_dir = 'vis_gt'):
    vis_data_path = os.path.join(data_path, vis_dir)
    if not os.path.exists(vis_data_path):
        os.makedirs(vis_data_path)

    cls_list = []
    for img in labels:
        cls_list.extend([obj['names'][0] for obj in img['objects']])
    cls_list = list(set(cls_list))
    cls_to_ind = dict(zip(cls_list, list(range(len(cls_list)))))

    data_viewer = dataViewer(cls_list)

    im_dir = im_to_dir()
    ind_to_label = dict(zip([im["image_id"] for im in labels], labels))

    while(True):
        im_id = raw_input("Image_ID: ")
        l = ind_to_label[int(im_id)]
        num_obj = len(l['objects'])
        im_name = str(l['image_id']) + '.jpg'
        im_path = os.path.join(im_dir[im_name], im_name)
        im = cv2.imread(im_path)

        bboxes = np.zeros(shape=(num_obj, 5), dtype=np.float32)
        for i, obj in enumerate(l['objects']):
            x1, y1, x2, y2 = obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']
            cls_name = obj['names'][0]
            print(x1, y1, x2, y2, cls_name)
            if cls_name in cls_list:
                cls_ind = cls_to_ind[cls_name]
                bboxes[i] = [x1, y1, x2, y2, cls_ind]

        if (bboxes[:, 0] > 0).sum() > 0:
            im = data_viewer.draw_objdet(im, bboxes)
            cv2.imwrite(os.path.join(vis_data_path, im_name), im)


if __name__ == '__main__':
    desired_classes = ['box', 'banana', ['notebook', 'book'], 'screwdriver', 'toothpaste', 'apple',
                     'stapler', ['mobile phone', 'cell phone', 'cellphone', 'phone'], 'bottle', 'pen', 'mouse', 'umbrella',
                     ['remotes', 'remote controller'], 'cans', 'tape', 'knife', ['wrench', 'spanner'], 'cup', ['charger', 'battery charger'],
                     ['badminton', 'shuttlecock', 'shuttle'], 'wallet',
                     ['eyeglass', 'spectacles', 'eyewear', 'eye glasses', 'eye glass'], 'pliers', ['headset', 'headphone'],
                     'toothbrush', 'card', 'toilet paper', 'towel', 'razor', 'watch']
    desired_classes = [cls if isinstance(cls, list) else [cls] for cls in desired_classes]

    # do_data_statistics(desired_classes)
    # do_data_filtering(desired_classes)

    # with open('/data1/zhb/vg/objects_vmrd.json', "rb") as f:
    #     labels = json.load(f)
    # vis_gt(labels, ['pliers'], 'vis_gt')
    interactive_vis(load_obj_labels())




