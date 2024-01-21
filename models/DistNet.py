from __future__ import absolute_import, division, print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import *


import numpy as np
from collections import OrderedDict

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU() # by jiahan NOTE 这里如果设置 inplace 会发生错误，这是因为 features 是从外部传进来的 tensor

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class DistHeader(nn.Module):
    def __init__(self) -> None:
        super(DistHeader, self).__init__()
        self.convs = []
        self.convs.append(ConvBlock(512,128))
        self.convs.append(ConvBlock(128,32))
        self.convs.append(ConvBlock(32,2))

        self.convs = nn.ModuleList(self.convs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = x
        for i in range(len(self.convs)):
            out = self.convs[i](out)
            if i != len(self.convs)-1:
                out = self.relu(out)
        
        out = out.mean(3).mean(2)
        out = self.sigmoid(out)
        
        out = out.view(-1,2)
        ab = out.clone()
        # ab[:,1] = -(2*out[:,1]-1) # beta 用于修正深度图中的最小值
        
        return ab
        

class DistNet(nn.Module):

    def __init__(self):
        super(DistNet, self).__init__()
        self.header = DistHeader()

    def init_weights(self):
        pass

    def forward(self, x):
        ab = self.header(x)
        
        return ab
