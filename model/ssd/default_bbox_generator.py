"""
Borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/prior_box.py
"""

from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']

        if 'max_sizes' in cfg.keys():
            self.max_sizes = cfg['max_sizes']
        else:
            self.max_sizes = None

        if 'angles' in cfg.keys():
            self.angles = cfg['angles']
        else:
            self.angles = None

        self.steps = cfg['steps']

        if 'aspect_ratios' in cfg.keys():
            self.aspect_ratios = cfg['aspect_ratios']
        else:
            self.aspect_ratios = None

        self.clip = cfg['clip']

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # grasp priors
                if self.angles:
                    s_k = self.min_sizes[k] / self.image_size
                    for angle in self.angles:
                        mean += [cx, cy, s_k, s_k, angle]
                # object priors
                else:
                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    mean += [cx, cy, s_k, s_k]

                if self.max_sizes:
                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                if self.aspect_ratios:
                    for ar in self.aspect_ratios[k]:
                        # grasp priors
                        if self.angles:
                            for angle in self.angles:
                                mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar), angle]
                        # object priors
                        else:
                            mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                            mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def backward(self):
        pass