# --------------------------------------------------------
# Visual Detection: State-of-the-Art
# Copyright: Hanbo Zhang
# Licensed under The MIT License [see LICENSE for details]
# Written by Hanbo Zhang
# --------------------------------------------------------

import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt

from model.utils.config import cfg
from model.utils.net_utils import create_mrt
from sklearn.manifold import TSNE

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

# Numpy data viewer to demonstrate detection results or ground truth.
class dataViewer(object):
    def __init__(self, classes):
        self.color_pool = [(255, 207, 136), (68, 187, 92), (153, 255, 0), (68, 187, 187), (0, 153, 255), (187, 68, 163),
                           (255, 119, 119), (116, 68, 187), (68, 187, 163), (163, 187, 68), (0, 204, 255), (68, 187, 140),
                           (204, 0, 255), (255, 204, 0), (102, 0, 255), (255, 0, 0), (68, 140, 187), (187, 187, 68),
                           (0, 255, 153), (119, 255, 146), (187, 163, 68), (187, 140, 68), (255, 153, 0), (255, 255, 0),
                           (153, 0, 255), (0, 255, 204), (68, 116, 187), (0, 255, 51), (187, 68, 68), (140, 187, 68),
                           (68, 163, 187), (187, 116, 68), (163, 68, 187), (204, 255, 0), (255, 0, 204), (0, 255, 255),
                           (140, 68, 187), (0, 102, 255), (153, 214, 255), (255, 102, 0)]
        self.classes = classes
        self.num_classes = len(self.classes)
        # Extend color_pool so that it is longer than classes
        self.color_pool = (self.num_classes / len(self.color_pool) + 1) * self.color_pool
        self.class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self.ind_to_class = dict(zip(xrange(self.num_classes), self.classes))
        self.color_dict = dict(zip(self.classes, self.color_pool[:self.num_classes]))

        self.TSNE = TSNE(n_components=2, init='pca')

    def draw_single_bbox(self, img, bbox, bbox_color=(163, 68, 187), text_str="", test_bg_color = None):
        if test_bg_color is None:
            test_bg_color = bbox_color
        bbox = tuple(bbox)
        text_rd = (bbox[2], bbox[1] + 25)
        cv2.rectangle(img, bbox[0:2], bbox[2:4], bbox_color, 2)
        cv2.rectangle(img, bbox[0:2], text_rd, test_bg_color, -1)
        cv2.putText(img, text_str, (bbox[0], bbox[1] + 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        return img

    def draw_single_grasp(self, img, grasp, test_str=None, text_bg_color=(255, 0, 0)):
        gr_c = (int((grasp[0] + grasp[4]) / 2), int((grasp[1] + grasp[5]) / 2))
        for j in range(4):
            if j % 2 == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            p1 = (int(grasp[2 * j]), int(grasp[2 * j + 1]))
            p2 = (int(grasp[(2 * j + 2) % 8]), int(grasp[(2 * j + 3) % 8]))
            cv2.line(img, p1, p2, color, 2)

        # put text
        if test_str is not None:
            text_len = len(test_str)
            text_w = 17 * text_len
            gtextpos = (gr_c[0] - text_w / 2, gr_c[1] + 20)
            gtext_lu = (gr_c[0] - text_w / 2, gr_c[1])
            gtext_rd = (gr_c[0] + text_w / 2, gr_c[1] + 25)
            cv2.rectangle(img, gtext_lu, gtext_rd, text_bg_color, -1)
            cv2.putText(img, test_str, gtextpos,
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), thickness=2)
        return img

    def draw_graspdet(self, im, dets, g_inds=None):
        """
        :param im: original image numpy array
        :param dets: detections. size N x 8 numpy array
        :param g_inds: size N numpy array
        :return: im
        """
        # make memory contiguous
        im = np.ascontiguousarray(im)
        if dets.shape[0] == 0:
            return im

        dets = dets[(dets[:,:8].sum(-1)) > 0].astype(np.int)
        num_grasp = dets.shape[0]
        for i in range(num_grasp):
            im = self.draw_single_grasp(im, dets[i], str(g_inds[i]) if g_inds is not None else None)
        return im

    def draw_objdet(self, im, dets, o_inds = None):
        """
        :param im: original image
        :param dets: detections. size N x 5 with 4-d bbox and 1-d class
        :return: im
        """
        # make memory contiguous
        im = np.ascontiguousarray(im)
        if dets.shape[0] == 0:
            return im

        dets = dets[(dets[:,:4].sum(-1)) > 0].astype(np.int)
        num_box = dets.shape[0]

        for i in range(num_box):
            cls = self.ind_to_class[dets[i, -1]]
            if o_inds is None:
                im = self.draw_single_bbox(im, dets[i][:4], self.color_dict[cls], cls)
            else:
                im = self.draw_single_bbox(im, dets[i][:4], self.color_dict[cls], '%s%d' % (cls, o_inds[i]))
        return im

    def draw_graspdet_with_owner(self, im, o_dets, g_dets, g_inds):
        """
        :param im: original image numpy array
        :param o_dets: object detections. size N x 5 with 4-d bbox and 1-d class
        :param g_dets: grasp detections. size N x 8 numpy array
        :param g_inds: grasp indice. size N numpy array
        :return:
        """
        im = np.ascontiguousarray(im)
        if o_dets.shape[0] > 0:
            o_inds = np.arange(o_dets.shape[0])
            im = self.draw_objdet(im, o_dets, o_inds)
            im = self.draw_graspdet(im, g_dets, g_inds)
        return im

    def draw_mrt(self, img, rel_mat, class_names = None, rel_score = None):
        if rel_mat.shape[0] == 0:
            return img

        mrt = create_mrt(rel_mat, class_names, rel_score)
        # for e in mrt.edges():
        #     print(e)

        fig = plt.figure(0, figsize=(3, 3))
        pos = nx.circular_layout(mrt)
        nx.draw(mrt, pos, with_labels=True, font_size=16,
                node_color='#FFF68F', node_shape='s', node_size=300, labels={node:node for node in mrt.nodes()})
        edge_labels = nx.get_edge_attributes(mrt, 'weight')
        nx.draw_networkx_edge_labels(mrt, pos, edge_labels=edge_labels)
        # grab the pixel buffer and dump it into a numpy array
        rel_img = fig2data(fig)

        rel_img = cv2.resize(rel_img[:,:,:3], (300, 300), interpolation=cv2.INTER_LINEAR)
        # img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        if min(img.shape[:2]) < 300:
            scalar = 300. / min(img.shape[:2])
            img = cv2.resize(img, None, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_LINEAR)
        img[:300, :300] = rel_img
        plt.close(0)

        return img

    def draw_caption(self, im, dets, captions):
        im = np.ascontiguousarray(im)
        if dets.shape[0] == 0:
            return im

        dets = dets[(dets[:,:4].sum(-1)) > 0].astype(np.int)
        num_box = dets.shape[0]

        for i in range(num_box):
            cls = self.ind_to_class[dets[i, -1]]
            im = self.draw_single_bbox(im, dets[i][:4], self.color_dict[cls], '{}'.format(captions[i]))
        return im

    def draw_image_caption(self, im, caption, test_bg_color=(0,0,0)):
        text_rd = (im.shape[1], 25)
        cv2.rectangle(im, (0, 0), text_rd, test_bg_color, -1)
        cv2.putText(im, caption, (0, 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), thickness=2)
        return im

    def draw_grounding_probs(self, im, expr, dets, ground_probs):
        im = np.ascontiguousarray(im)
        self.draw_image_caption(im, expr)
        if dets.shape[0] == 0:
            return im
        dets = dets[(dets[:,:4].sum(-1)) > 0].astype(np.int)
        assert dets.shape[0] == ground_probs.shape[0]
        num_box = dets.shape[0]

        for i in range(num_box):
            prob = '{:.2f}'.format(ground_probs[i])
            im = self.draw_single_bbox(im, dets[i][:4], text_str=prob)
        return im