import torch
from torch import nn
from torch.autograd import Variable
from rois_pair_expanding_layer import _RoisPairExpandingLayer
from object_pairing_layer import _ObjPairLayer
from op2l import _OP2L

bbox = torch.Tensor([
    [0, 100., 200., 300., 400.],
    [0, 200., 100., 400., 300.],
    [0, 58.3, 62., 125., 321.],
    [0, 200., 287., 412., 322.],
    [1, 50., 120., 420., 380.],
    [0, 100., 200., 300., 400.],
    [0, 200., 100., 400., 300.],
    [0, 58.3, 62., 125., 321.],
])

feats = Variable(torch.rand(2,512,50,50))
feats.cuda()
feats.requires_grad = True
batch_size = 2
obj_num = torch.Tensor([4,1,3])

"""
roipair = _RoisPairExpandingLayer()
pairbox = roipair(bbox, batch_size, obj_num )
"""

"""
objpair = _ObjPairLayer(isex = True)
pairfeat = objpair(feats, batch_size, obj_num)
print(pairfeat.size())
loss = pairfeat.sum()
loss.backward()
print(111)
"""


op2l = _OP2L(isex=True)
opfeats = op2l(feats, bbox, batch_size, obj_num)
loss = opfeats.sum()
loss.backward()
