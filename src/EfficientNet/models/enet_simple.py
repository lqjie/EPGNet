import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from .enet_pytorch import EfficientNet
from .utils import MemoryEfficientSwish,round_filters,get_same_padding_conv2d
import torch

def get_model(model_name):
    print(model_name)
    if model_name == 'enet_b0':
        model = enet_b0()
    elif model_name == 'sca_enet_b0':
        model = sca_enet_b0()
    elif model_name == 'epg_enet_b0':
        model = epg_enet_b0()
    
    return model

class enet_b0(nn.Module):
    def __init__(self):
        super(enet_b0, self).__init__()
        # Original function 'from_pretrained' has been slightly modified to remove the first layer's pooling 
        # model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False) #stem's stride = 1
        self.enet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1, num_classes=2)#one-channel

    def forward(self, input):

        output = self.enet(input)
        return output

class sca_enet_b0(nn.Module):
    def __init__(self):
        super(sca_enet_b0, self).__init__()

        self.enet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1, num_classes=2)

    def forward(self, input):
        bs = input.size(0)
        img, beta = torch.split(input, 1, dim=1)
        x = self.enet._swish(self.enet._conv_stem(img))
        beta = beta+1e-6 # Plus a small value to avoid NaN gradient
        beta = F.conv2d(beta, torch.abs(self.enet._conv_stem.weight), stride=1, padding=1)
        beta = torch.sqrt(beta)
        x = x + beta
        x = self.enet.extract_features_wo_stem(x)

        # Pooling and final linear layer
        x = self.enet._avg_pooling(x)
        x = x.view(bs, -1)
        x = self.enet._dropout(x)
        x = self.enet._fc(x)
        return x

class BN_init_zero(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BN_init_zero, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                 track_running_stats=track_running_stats)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
class EPGM(nn.Module):
    def __init__(self, in_channel,stride):
        super(EPGM, self).__init__()
        
        self.conv_beta = nn.Sequential(
            nn.Conv2d(1, in_channel, kernel_size=3, padding=1,bias=False),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0,bias=False),
            BN_init_zero(in_channel,eps=1e-3, momentum=0.01),
        )
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, img, beta):
        beta = self.pool(beta)
        beta = self.conv_beta(beta)
        output_w = img * beta
        output = self.fusion(output_w) + img
        return output

class epg_enet_b0(nn.Module):
    def __init__(self):
        super(epg_enet_b0, self).__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1, num_classes=2)

        self.epgms = nn.ModuleDict({ '4':EPGM(40,4), '10':EPGM(112,8)})
        self.epgm_stem = EPGM(32,1)
        bn_mom = 1 - self.enet._global_params.batch_norm_momentum
        bn_eps = self.enet._global_params.batch_norm_epsilon
        Conv2d = get_same_padding_conv2d()
        self.att1 = nn.Sequential(Conv2d(32, 40, kernel_size=4, stride=4, bias=False),
                                nn.BatchNorm2d(40, momentum=bn_mom, eps=bn_eps),
                                MemoryEfficientSwish(),
        )
        self.att2 = nn.Sequential(Conv2d(80, 112, kernel_size=2, stride=2,bias=False),
                                nn.BatchNorm2d(112, momentum=bn_mom, eps=bn_eps),
                                MemoryEfficientSwish(),
        )
        self.att3 = nn.Sequential(Conv2d(224, 320, kernel_size=2, stride=2,bias=False),
                                nn.BatchNorm2d(320, momentum=bn_mom, eps=bn_eps),
                                MemoryEfficientSwish(),
        )
        self._conv_head = nn.Conv2d(640, 1280, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=1280, momentum=bn_mom, eps=bn_eps)
        self._swish= MemoryEfficientSwish()
    def forward(self, input):
        bs = input.size(0)
        img, beta = torch.split(input, 1, dim=1)
        x = self.enet.fea_stem(img)
        x = self.epgm_stem(x,beta)
        fusion=[]
        fusion.append(x)
        for idx, block in enumerate(self.enet._blocks):
            drop_connect_rate = self.enet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.enet._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if str(idx) in self.epgms:
                x = self.epgms[str(idx)](x, beta)
                fusion.append(x)
        att1 = self.att1(fusion[0])
        att2 = torch.cat([fusion[1],att1], dim=1)
        att2 = self.att2(att2)
        att3 = torch.cat([fusion[2],att2], dim=1)
        att3 = self.att3(att3)
        x = torch.cat([x,att3], dim=1)
        x = self._swish(self._bn1(self._conv_head(x)))
        # Pooling and final linear layer
        x = self.enet._avg_pooling(x)
        x = x.view(bs, -1)
        x = self.enet._dropout(x)
        x = self.enet._fc(x)
        return x
