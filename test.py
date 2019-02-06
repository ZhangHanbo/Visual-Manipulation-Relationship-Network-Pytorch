import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as f

bbox = Variable(0.5 * torch.rand([10,2]))
maxoffset = torch.min(1 - bbox)
bbox = torch.cat([bbox, bbox + maxoffset * Variable(torch.rand([10,2]))],1)
bbox.detach()

imagesize = 32.
featmap = Variable(torch.rand(128, imagesize, imagesize))
featmap.requires_grad = True

bboxnum = bbox.size()[0]
for t in range(0,bboxnum):
    for tt in range(t,bboxnum):
        print(t,tt)
        box1 = bbox[t]
        box2 = bbox[tt]
        tmax = torch.max(box1,box2)
        tmin = torch.min(box1,box2)
        minmask = Variable(torch.ByteTensor([1,1,0,0]))
        maxmask = Variable(torch.ByteTensor([0,0,1,1]))
        unionbox = torch.cat([torch.masked_select(tmin,minmask),torch.masked_select(tmax,maxmask)],0)
        unionbox = torch.reshape(unionbox,(-1,4))
        bbox = torch.cat([bbox,unionbox],0)

featbox = Variable((torch.round(imagesize * bbox)).long())
AdapMaxPool = f.adaptive_max_pool2d
ROIout = Variable(torch.Tensor([]))

for t in range(bbox.size()[0]):
    x1 = featbox[t, 0]
    y1 = featbox[t, 1]
    x2 = featbox[t, 2]
    y2 = featbox[t, 3]
    if x1 == x2:
        x2 += 1
    if y1 == y2:
        y2 += 1
    ROIout = torch.cat([ROIout, AdapMaxPool(featmap[:, x1:x2, y1:y2],[7,7])],0)

out = ROIout.sum()
out.backward()
print(featmap.grad[0])

