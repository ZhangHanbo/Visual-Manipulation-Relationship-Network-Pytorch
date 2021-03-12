from .nms import nms

import torch
if torch.__version__.split(".")[0] == "1":
    from .roi_align import ROIAlign, RoIAlignAvg, RoIAlignMax
    from .roi_pool import ROIPool

elif torch.__version__ == "0.4.0":
    import model.roi_align.modules.roi_align as _ROIAlign
    import model.roi_pooling.modules.roi_pool as _ROIPool

    class ROIAlign(_ROIAlign.RoIAlign):
        def __init__(self, output_size, spatial_scale, sampling_ratio=None):
            super(ROIAlign, self).__init__(output_size[0], output_size[1], spatial_scale)

    class RoIAlignAvg(_ROIAlign.RoIAlignAvg):
        def __init__(self, output_size, spatial_scale, sampling_ratio=None):
            super(RoIAlignAvg, self).__init__(output_size[0], output_size[1], spatial_scale)

    class RoIAlignMax(_ROIAlign.RoIAlignMax):
        def __init__(self, output_size, spatial_scale, sampling_ratio=None):
            super(RoIAlignMax, self).__init__(output_size[0], output_size[1], spatial_scale)

    class ROIPool(_ROIPool._RoIPooling):
        def __init__(self, output_size, spatial_scale):
            super(ROIPool, self).__init__(output_size[0], output_size[1], spatial_scale)

else:
    raise RuntimeError("unsupported torch version. Supported: 0.4.0 (recommended) and 1.x")