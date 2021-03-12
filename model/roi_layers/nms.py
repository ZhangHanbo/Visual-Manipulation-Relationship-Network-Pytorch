# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C

import torch
if torch.__version__.split(".")[0] == "1":
    from torchvision.ops import nms
elif torch.__version__ == "0.4.0":
    from model.nms.nms_wrapper import nms
else:
    raise RuntimeError("unsupported torch version. Supported: 0.4.0 (recommended) and 1.x")

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
