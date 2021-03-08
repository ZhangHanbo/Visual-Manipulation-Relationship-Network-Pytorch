# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torchvision.ops import roi_align
from torch.nn.functional import avg_pool2d, max_pool2d

class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

class RoIAlignAvg(ROIAlign):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(RoIAlignAvg, self).__init__(output_size, spatial_scale, sampling_ratio)

    def forward(self, features, rois):
        x = roi_align(
            features, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoIAlignMax(ROIAlign):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(RoIAlignMax, self).__init__(output_size, spatial_scale, sampling_ratio)

    def forward(self, features, rois):
        x = roi_align(
            features, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )
        return max_pool2d(x, kernel_size=2, stride=1)
