import pickle
import cv2

with open("roidb.pkl", "rb") as f:
    roidb = pickle.load(f)

roidb, ratio_list, ratio_index = roidb["roidb"], roidb["ratio_list"], roidb["ratio_index"]

count = 0
for i, r in enumerate(roidb):
    img = cv2.imread(r["image"])
    if r["rotated"] not in {1,3}:
        w = r["width"]
        h = r["height"]
    else:
        w = r["height"]
        h = r["width"]
    if img.shape[1] != w or img.shape[0] != h:
        print(i)
        count +=1

print (count)