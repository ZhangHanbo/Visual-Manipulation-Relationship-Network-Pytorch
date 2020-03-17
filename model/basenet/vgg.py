'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

local_model_paths = {
    'vgg16': 'data/pretrained_model/vgg16_caffe.pth',
}


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=1000, feat_list = ("conv4",),
                 pretrained_model_path = None):
        super(VGG, self).__init__()
        self.features = features[0]
        self.pool_loc = features[1]
        assert len(self.pool_loc) == 6
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

         # Initialize weights
        if pretrained_model_path is not None:
            print("loading pretrained model: ", pretrained_model_path)
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict({k: v for k, v in state_dict.items() if k in self.state_dict()})
        else:
            self._initialize_weights()

        self.feat_list = feat_list
        # init feat layer
        self.feat_layer["conv1"] = nn.Sequential(self.features[self.pool_loc[0] : self.pool_loc[1]])
        self.feat_layer["conv2"] = nn.Sequential(self.features[self.pool_loc[1] : self.pool_loc[2]])
        self.feat_layer["conv3"] = nn.Sequential(self.features[self.pool_loc[2] : self.pool_loc[3]])
        self.feat_layer["conv4"] = nn.Sequential(self.features[self.pool_loc[3] : self.pool_loc[4]])
        self.feat_layer["conv5"] = nn.Sequential(self.features[self.pool_loc[4] : self.pool_loc[5]])
        self.feat_layer["fc"] = self.classifier[:-1]
        self.feat_layer["cscore"] = self.classifier[-1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = []
        for conv_key, conv_layer in self.feat_layer.items():
            if conv_key == 'fc':
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            x = conv_layer(x)
            if conv_key in self.feat_list:
                feats.append(x)
            if conv_key == self.feat_list[-1]:
                break
        if len(self.feat_list) == 1:
            feats = feats[0]
        return feats

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    max_pool_layer_nums = []

    layer_counter = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            max_pool_layer_nums.append(layer_counter)
            layer_counter += 1
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            max_pool_layer_nums.append(layer_counter)
            layer_counter += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layer_counter += 1
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layer_counter += 1
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                layer_counter += 1
            in_channels = v
    max_pool_layer_nums.append(layer_counter)
    max_pool_layer_nums[0] = 0
    return nn.Sequential(*layers), max_pool_layer_nums

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg_initializer(name, feat_list, pretrained = False):
    cfg_dict = {
        "vgg11": {"cfg": "A", "bn": False},
        "vgg13": {"block": "B", "bn": False},
        "vgg16": {"block": "D", "bn": False},
        "vgg19": {"block": "E", "bn": False},
        "vgg11_bn": {"block": "A", "bn": True},
        "vgg13_bn": {"block": "B", "bn": True},
        "vgg16_bn": {"block": "D", "bn": True},
        "vgg19_bn": {"block": "E", "bn": True},
    }
    model = VGG(make_layers(cfgs[cfg_dict[name]["cfg"]], batch_norm=cfg_dict[name]["bn"]), feat_list,
                local_model_paths[name] if pretrained else None)
    return model