import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time
from torch.nn import Linear, Conv1d, BatchNorm1d, Conv3d, InstanceNorm3d, AdaptiveAvgPool1d, ModuleList
import math
import numpy as np
from itertools import combinations
from typing import Optional

def voxel_shuffle(input, upscale_factor):
    if type(upscale_factor) == int:
        upscale_factor = [upscale_factor] * 3
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= np.prod(upscale_factor)

    out_height = in_height * upscale_factor[0]
    out_width = in_width * upscale_factor[1]
    out_depth = in_depth * upscale_factor[2]

    input_view = input.reshape(batch_size, channels, upscale_factor[0], upscale_factor[1], upscale_factor[2], in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class VoxelShuffle(nn.Module):
    def __init__(self,inchannels,outchannels,upscale_factor=2):
        super(VoxelShuffle,self).__init__()
        if type(upscale_factor) == int:
            upscale_factor = [upscale_factor] * 3
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(inchannels,outchannels*(np.prod(upscale_factor)),3,1,1)

    def forward(self,x):
        x = voxel_shuffle(self.conv(x),self.upscale_factor)
        return x

class Upsample(nn.Module):
    def __init__(self,inchannels,outchannels,factor=2.0):
        super(Upsample,self).__init__()
        self.conv = nn.Conv3d(inchannels,outchannels,kernel_size=3,stride=1,padding=1)
        self.factor = factor

    def forward(self,x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode="trilinear",align_corners=False)
        x = self.conv(x)
        return x

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResidualLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels//4,kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels//4, out_channels//4,kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(out_channels//4, out_channels,kernel_size=3, padding=1, bias=False)
        self.ac = nn.SiLU()

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels,kernel_size=3, padding=1, bias=False)

    def forward(self, x) :

        h = self.conv1(x)

        h = self.ac(h)

        h = self.conv2(h)

        h = self.ac(h)

        h = self.conv3(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6/ self.in_features)/30, np.sqrt(6/ self.in_features)/30)  
        
    def forward(self, input):
        return self.linear(input)

class ParamerEmedding(nn.Module):
    def __init__(self, in_dim, ch):
        super().__init__()

        self.in_dim = in_dim

        self.ensemble_embeder = []

        self.ensemble_embeder.append(nn.Linear(self.in_dim, ch))
        self.ensemble_embeder.append(nn.SiLU())
        self.ensemble_embeder.append(nn.Linear(ch, ch * 4))
        self.ensemble_embeder.append(nn.SiLU())
        self.ensemble_embeder.append(nn.Linear(ch*4, ch))

        self.ensemble_embeder = nn.Sequential(*self.ensemble_embeder)

    def forward(self, params):
        ensemble_embedding = self.ensemble_embeder(params)
        return ensemble_embedding

class ConvDecoder(nn.Module):
    def __init__(self, out_dim, ch, dataset='Nyx'):
        super().__init__()

        self.ac = nn.SiLU()

        self.decoder = []

        self.dataset = dataset

        self.conv_in = nn.Conv3d(ch, 256,kernel_size=3,stride=1,padding=1)

        self.ch = 256

        if self.dataset in ['Nyx','Castro','ColverLeaf']:
            self.scale = int(np.log2(128))
        elif self.dataset in ['MPAS-Ocean']:
            self.scale = int(np.log2(64))

            self.upscaler = []
            self.upscaler.append(ResidualLayer(self.ch, self.ch))
            self.upscaler.append(VoxelShuffle(self.ch, self.ch//2, upscale_factor=[2,1,1]))
            self.upscaler.append(nn.SiLU())

            self.ch = self.ch // 2

            self.upscaler.append(ResidualLayer(self.ch, self.ch))
            self.upscaler.append(VoxelShuffle(self.ch, self.ch, upscale_factor=[2,2,1]))
            self.upscaler.append(nn.SiLU())

            self.upscaler = nn.Sequential(*self.upscaler)


        for i in range(0,self.scale):
            self.decoder.append(ResidualLayer(self.ch, self.ch))
            self.decoder.append(VoxelShuffle(self.ch, self.ch//2))
            self.decoder.append(nn.SiLU())
            self.ch = self.ch // 2

        self.decoder = nn.Sequential(*self.decoder)

        self.conv_out = nn.Conv3d(self.ch,out_dim,kernel_size=3,stride=1,padding=1)

    def forward(self, feats):
        feats = feats.view(feats.size(0),-1,1,1,1)

        v = self.conv_in(feats)

        v = self.ac(v)

        if self.dataset in ['MPAS-Ocean']:
            v = self.upscaler(v)

        v = self.decoder(v)

        return self.conv_out(v)



class EnsembleNet(nn.Module):
    def __init__(self,in_dim=3,out_dim=1,ch=128,dataset='Nyx'):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ch = ch

        self.param_embeder = ParamerEmedding(in_dim, ch)

        self.decoder = ConvDecoder(out_dim, ch, dataset)

    def forward(self,params):
        e = self.param_embeder(params)

        return self.decoder(e)

    def get_ensmeble_embedding(self, ensemble_params):
        e = self.param_embeder(ensemble_params)
        return e