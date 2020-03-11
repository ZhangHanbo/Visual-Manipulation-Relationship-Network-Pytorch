"""
borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import pdb
import warnings

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A \cap B / A \cup B = A \cap B / (area(A) + area(B) - A \cap B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):

        for t in self.transforms:
            data = None
            while(data is None):
                data = t(img, boxes, labels, grasps, boxes_keep, grasps_keep)
            img, boxes, labels, grasps, boxes_keep, grasps_keep = data

        return img, boxes, labels, grasps, boxes_keep, grasps_keep


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        return self.lambd(img, boxes, labels, grasps, boxes_keep, grasps_keep)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        return image.astype(np.float32), boxes, labels, grasps, boxes_keep, grasps_keep


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, grasps, boxes_keep, grasps_keep


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        height, width, channels = image.shape

        if boxes is not None:
            boxes[:, 0::2] *= width
            boxes[:, 1::2] *= height

        if grasps is not None:
            grasps[:, 0::2] *= width
            grasps[:, 1::2] *= height

        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        height, width, channels = image.shape

        if boxes is not None:
            boxes[:, 0::2] /= width
            boxes[:, 1::2] /= height

        if grasps is not None:
            grasps[:, 0::2] /= width
            grasps[:, 1::2] /= height

        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class Resize(object):
    def __init__(self, w=300, h=None):
        self.w = w
        if h is None:
            self.h = w
        else:
            self.h = h

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep=None, grasps_keep=None):
        scaler = np.array([self.h, self.w], dtype=np.float32) / image.shape[:2]
        image = cv2.resize(image, (self.w, self.h))
        # [h,w] -> [w,h]
        scaler = scaler[::-1]

        if boxes is not None:
            ori_shape = boxes.shape
            boxes = (boxes.reshape(-1,2) * scaler).reshape(ori_shape)

        if grasps is not None:
            ori_shape = grasps.shape
            grasps = (grasps.reshape(-1, 2) * scaler).reshape(ori_shape)
            
        if scaler[0] != scaler[1]:
            warnings.warn("Resize scalers on x-axis and y-axis are not equal. Grasping rects is unusable.")

        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep=None, grasps_keep=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep = None, grasps_keep = None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep=None, grasps_keep=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, grasps, boxes_keep,grasps_keep


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, grasps = None,
                 boxes_keep=None, grasps_keep=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None, grasps = None,
                 boxes_keep=None, grasps_keep=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), \
               boxes, labels, grasps, boxes_keep, grasps_keep


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None, grasps = None,
                 boxes_keep=None, grasps_keep=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), \
               boxes, labels, grasps, boxes_keep, grasps_keep

# WARNING: When using this crop method, the image's height-width ratio will change and may cause some problems for
#          training grasp detector.
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, im, bs = None, ls=None, gr = None, bk = None, gk = None):
        height, width, _ = im.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return im, bs, ls, gr, bk, gk

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                image = im.copy()

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                # cut the crop from the image
                image = image[rect[1]:rect[3], rect[0]:rect[2],:]

                boxes = None
                labels = None
                boxes_keep = None
                if bs is not None:
                    boxes = bs.copy()
                    labels = ls.copy()
                    boxes_keep = bk.copy()
                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    overlap = jaccard_numpy(boxes, rect)
                    # is min and max overlap constraint satisfied? if not try again
                    if overlap.min() < min_iou and max_iou < overlap.max():
                        continue
                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                    # mask in that both m1 and m2 are true
                    mask = m1 * m2
                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue
                    # take only matching gt boxes
                    boxes = boxes[mask, :]
                    # take only matching gt labels
                    labels = labels[mask]
                    boxes_keep = boxes_keep[mask]
                    # should we use the box left and top corner or the crop's
                    boxes[:, :2] = np.maximum(boxes[:, :2],
                                                  rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    boxes[:, :2] -= rect[:2]
                    boxes[:, 2:] = np.minimum(boxes[:, 2:],
                                                      rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    boxes[:, 2:] -= rect[:2]

                grasps = None
                grasps_keep = None
                if gr is not None:
                    grasps = gr.copy()
                    grasps_keep = gk.copy()
                    ori_shape = grasps.shape
                    img_bound = np.tile((w, h), (ori_shape[0], ori_shape[1] / 2))
                    grasps = (grasps.reshape(-1, 2) - rect[:2]).reshape(ori_shape)
                    keep_grasp = (np.sum((grasps < 0) | (grasps > img_bound), 1) == 0)
                    if keep_grasp.sum() == 0:
                        continue
                    else:
                        grasps = grasps[keep_grasp]
                        grasps_keep = grasps_keep[keep_grasp]

                return image, boxes, labels, grasps, boxes_keep, grasps_keep

# WARNING: When using this crop method, the image's height-width ratio will change and may cause some problems for
#          training grasp detector.
class RandomCropKeepBoxes(object):
    def __call__(self, im, bs=None, ls=None, gr=None, bk=None, gk=None):
        height, width, _ = im.shape
        # Get the minimum boundary of all bboxes in dim x and y
        xmin = np.min(bs[:, 0])
        ymin = np.min(bs[:, 1])
        # Get the maximum boundary of all bboxes in dim x and y
        xmax = np.max(bs[:, 2])
        ymax = np.max(bs[:, 3])

        # Get top left corner's coordinate of the crop box
        x_start = int(random.uniform(0, xmin))
        y_start = int(random.uniform(0, ymin))
        # Get lower right corner corner's coordinate of the crop box
        x_end = int(random.uniform(xmax, width))
        y_end = int(random.uniform(ymax, height))

        # Crop the image
        im = im[y_start:y_end, x_start:x_end]

        # Adjust the bboxes to fit the cropped image
        bs[:, 0::2] -= x_start
        bs[:, 1::2] -= y_start

        # Adjust the grasp boxes to fit the cropped image
        gr[:, 0::2] -= x_start
        gr[:, 1::2] -= y_start

        return im, bs, ls, gr, bk, gk


class FixedSizeCrop(object):
    def __init__(self, min_x, min_y, max_x, max_y, cropsize_x, cropsize_y):
        self.minx = min_x
        self.miny = min_y
        self.maxx = max_x
        self.maxy = max_y
        self.cpx = cropsize_x
        self.cpy = cropsize_y

    def __call__(self, im, bs=None, ls=None, gr=None, bk=None, gk=None):

        assert self.minx < im.shape[1] and self.minx > 0 \
               and self.maxx < im.shape[1] and self.maxx > 0 \
               and self.miny < im.shape[0] and self.miny > 0 \
               and self.maxy < im.shape[0] and self.maxy > 0 \
               and self.minx < self.maxx and self.miny < self.maxy,  \
            "size out of bound"

        while (True):

            x = random.randint(self.minx, self.maxx)
            y = random.randint(self.miny, self.maxy)
            image = im.copy()
            image = image[y:y+self.cpy, x:x+self.cpx,:]

            boxes = None
            labels = None
            boxes_keep = None
            if bs is not None:
                boxes = bs.copy()
                labels = ls.copy()
                boxes_keep = bk.copy()
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (x < centers[:, 0]) * (y < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (x+self.cpx > centers[:, 0]) * (y+self.cpy > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                # take only matching gt boxes
                boxes = boxes[mask, :]
                # take only matching gt labels
                labels = labels[mask]
                boxes_keep = boxes_keep[mask]
                # should we use the box left and top corner or the crop's
                boxes[:, :2] = np.maximum(boxes[:, :2],(x,y))
                # adjust to crop (by substracting crop's left,top)
                boxes[:, :2] -= (x,y)
                boxes[:, 2:] = np.minimum(boxes[:, 2:],(x+self.cpx, y+self.cpy))
                # adjust to crop (by substracting crop's left,top)
                boxes[:, 2:] -= (x,y)

            grasps = None
            grasps_keep = None
            if gr is not None:
                grasps = gr.copy()
                grasps_keep = gk.copy()
                ori_shape = grasps.shape
                grasps = (grasps.reshape((-1, 2)) - (x,y)).reshape(ori_shape)

                # check crop
                img_bound = np.tile((self.cpx, self.cpy), (ori_shape[0], ori_shape[1] / 2))
                keep = (np.sum((grasps < 0) | (grasps > img_bound), 1) == 0)
                if np.sum(keep) == 0:
                    continue
                else:
                    grasps = grasps[keep]
                    grasps_keep = grasps_keep[keep]
                    break
            else:
                break

        return image, boxes, labels, grasps, boxes_keep, grasps_keep

# only for oriented bounding box
class RandomRotate(object):
    def __init__(self, max_r = 30, rand_center = False):
        self.max_r = max_r
        self.rand_center = rand_center

    def __call__(self, im, bs= None, ls= None, gr= None, bk=None, gk=None):

        assert bs is None, "Boxes should be None when using RandomRotate"

        if random.randint(2):
            return im, bs, ls, gr, bk, gk

        while(True):
            image = im.copy()
            height, width, depth = image.shape
            if self.rand_center:
                center = (random.randint(width), random.randint(height))
            else:
                center = (width / 2, height / 2)
            r = random.uniform(-self.max_r, self.max_r)
            M = cv2.getRotationMatrix2D(center, r, scale=1)
            image = cv2.warpAffine(image, M, (width, height))

            grasps = None
            if gr is not None:
                grasps = gr.copy()
                grasps_keep = gk.copy()

                ori_shape = grasps.shape
                grasps = np.reshape(grasps, (-1, 2))
                grasps = np.hstack((grasps, np.ones((grasps.shape[0], 1))))
                grasps = M.dot(grasps.T)

                grasps = np.round(np.reshape(grasps.T, ori_shape)).astype(int)

                # check rotation results
                img_bound = np.tile((width, height), (ori_shape[0], ori_shape[1] / 2))
                keep = (np.sum((grasps < 0) | (grasps > img_bound), 1) == 0)
                if np.sum(keep) == 0:
                    continue
                else:
                    grasps = grasps[keep]
                    grasps_keep = grasps_keep[keep]
                    break
            else:
                break

        return image, bs, ls, grasps, bk, grasps_keep

class RandomVerticalRotate(object):
    def __call__(self, image, boxes= None, classes= None, grasps= None,
                 boxes_keep=None, grasps_keep=None):

        # 0: no rotation
        # 1: ccw 90 degrees
        # 2: ccw 180 degrees
        # 3: ccw 270 degrees
        r = random.randint(4)
        h,w,_ = image.shape
        if not r:
            return image, boxes, classes, grasps, boxes_keep, grasps_keep
        else:
            image = np.rot90(image, r, axes=(0,1))

            def rotcoords(coords, rot, isbbox=False):
                new_coords = np.zeros(coords.shape)
                # (y, w-x)
                if rot == 1:
                    new_coords[:, 0::2] = coords[:, 1::2]
                    new_coords[:, 1::2] = w - coords[:, 0::2] - 1
                # (w-x, h-y)
                elif rot == 2:
                    new_coords[:, 0::2] = w - coords[:, 0::2] - 1
                    new_coords[:, 1::2] = h - coords[:, 1::2] - 1
                # (h-y,x)
                elif rot == 3:
                    new_coords[:, 0::2] = h - coords[:, 1::2] - 1
                    new_coords[:, 1::2] = coords[:, 0::2]
                if isbbox:
                    new_coords = np.concatenate(
                        (np.minimum(new_coords[:, 0:1], new_coords[:, 2:3]),
                         np.minimum(new_coords[:, 1:2], new_coords[:, 3:4]),
                         np.maximum(new_coords[:, 0:1], new_coords[:, 2:3]),
                         np.maximum(new_coords[:, 1:2], new_coords[:, 3:4]))
                        ,axis=1)
                return new_coords

            if boxes is not None:
                boxes = rotcoords(boxes, r, True)

            if grasps is not None:
                grasps = rotcoords(grasps, r, False)

        return image, boxes, classes, grasps, boxes_keep, grasps_keep

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes= None, labels= None, grasps= None,
                 boxes_keep=None, grasps_keep=None):
        if random.randint(2):
            return image, boxes, labels, grasps, boxes_keep, grasps_keep

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        if boxes is not None:
            boxes = boxes.copy()
            ori_shape = boxes.shape
            boxes = (boxes.reshape((-1,2)) + (int(left), int(top))).reshape(ori_shape)

        if grasps is not None:
            grasps = grasps.copy()
            ori_shape = grasps.shape
            grasps = (grasps.reshape((-1, 2)) + (int(left), int(top))).reshape(ori_shape)

        return image, boxes, labels, grasps, boxes_keep, grasps_keep


class RandomMirror(object):
    def __call__(self, image, boxes= None, classes= None, grasps= None,
                 boxes_keep=None, grasps_keep=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]

            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 0::2][:,::-1] - 1

            if grasps is not None:
                grasps = grasps.copy()
                grasps[:, 0::2] = width - grasps[:, 0::2] - 1

        return image, boxes, classes, grasps, boxes_keep, grasps_keep


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes=None, labels=None, grasps=None, boxes_keep=None, grasps_keep=None):
        im = image.copy()
        im, boxes, labels, grasps, boxes_keep, grasps_keep = \
            self.rand_brightness(im, boxes, labels, grasps, boxes_keep,grasps_keep)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, grasps, boxes_keep, grasps_keep = \
            distort(im, boxes, labels, grasps, boxes_keep, grasps_keep)
        return self.rand_light_noise(im, boxes, labels, grasps, boxes_keep, grasps_keep)


class Augmentation(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
        ])

    def __call__(self, img, boxes=None, labels=None, grasps=None,
                 boxes_keep=None, grasps_keep=None):
        return self.augment(img, boxes, labels, grasps, boxes_keep, grasps_keep)

class Augmentation_VMRD(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean
        self.augment = Compose([
            ConvertFromInts(),
            RandomCropKeepBoxes(),
            PhotometricDistort(),
            Expand(self.mean),
        ])

    def __call__(self, img, boxes=None, labels=None, grasps=None,
                 boxes_keep=None, grasps_keep=None):
        resizer = Resize(img.shape[1],img.shape[0])
        img, boxes, labels, grasps, boxes_keep, grasps_keep =  \
            self.augment(img, boxes, labels, grasps, boxes_keep, grasps_keep)
        # return img, boxes, labels, grasps, boxes_keep, grasps_keep
        return resizer(img, boxes, labels, grasps, boxes_keep, grasps_keep)
        

class Augmentation_Grasp(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean
        self.augment = Compose([
            FixedSizeCrop(100, 50, 200, 150, 320, 320),
            RandomRotate(),
        ])

    def __call__(self, img, boxes=None, labels=None , grasps=None,
                 boxes_keep=None, grasps_keep=None):
        return self.augment(img, boxes, labels , grasps, boxes_keep, grasps_keep)

class Augmentation_Grasp_Roign_Cornell(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean
        self.augment = Compose([
            FixedSizeCrop(100, 50, 200, 150, 320, 320),
            PhotometricDistort(),
        ])

    def __call__(self, img, boxes=None, labels=None , grasps=None,
                 boxes_keep=None, grasps_keep=None):
        return self.augment(img, boxes, labels , grasps, boxes_keep, grasps_keep)

class Augmentation_Grasp_Test(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean
        self.augment = Compose([
            FixedSizeCrop(100, 100, 101, 101, 320, 320),
        ])

    def __call__(self, img, boxes=None, labels=None , grasps=None,
                 boxes_keep=None, grasps_keep=None):
        return self.augment(img, boxes, labels, grasps, boxes_keep, grasps_keep)

class ResizeAndNormalize(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels, grasps,
                 boxes_keep=None, grasps_keep=None):
        return self.augment(img, boxes, labels, grasps, b)
