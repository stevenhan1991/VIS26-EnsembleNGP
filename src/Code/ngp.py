"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from termcolor import colored
import resfields
import torch.nn as nn
import util_misc

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

from math import sqrt

class modulator(nn.Module):

    def __init__(self,
            in_dim,
            out_dim,
            use_bias=True,
            w0=30.0
                 ):
        super().__init__()  
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.act_fn = Sine(w0=w0)
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(self, x):
        return self.act_fn(self.linear(x))

class MertricEmbedding(torch.nn.Module):
    def __init__(self, in_channels, N_freqs=0, logscale=True):

        super(MertricEmbedding, self).__init__()
        # original p.e.
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin] # , torch.cos
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    

    def forward(self, x):
        #out = [(x+1.0)/2.0] # range[0,1] for different storage lengths
        out = [x]
        sin_1 = torch.sin(torch.tensor(1.0))
        out += [(torch.sin(x)-sin_1)/(2.0 * sin_1)] # range[-1,0]
        out += [(torch.arcsin(x) - (3.0*torch.pi / 2.0)) / (2.0 * torch.pi)] # range[-1,-0.5]
        return torch.stack(out, dim=0)

class Net(torch.nn.Module):
    def __init__(self, n_output_dims, in_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        # Decoder
        self.w0 = 30.0
        self.use_bias = True
        self.omegas = 30

        layers = []

        self.num_layers = num_layers
        self.dim_hidden = hidden_dim
        self.in_dim = in_dim

        for ind in range(self.num_layers - 1):
            is_first = ind == 0
            layer_in_dim = n_output_dims if is_first else self.dim_hidden
            layers.append(
                SirenLayer(
                    in_dim=layer_in_dim,
                    out_dim=self.dim_hidden,
                    w0=self.w0,
                    use_bias=self.use_bias,
                is_first=is_first,
                )
            )
        self.net = nn.Sequential(*layers)

class Modulators(torch.nn.Module):
    def __init__(self, n_output_dims, in_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        # Decoder
        self.w0 = 30.0
        self.use_bias = True
        self.omegas = 30

        layers = []

        self.num_layers = num_layers
        self.dim_hidden = hidden_dim
        self.in_dim = in_dim

        self.modulators = nn.ModuleList(
            [modulator(in_dim=self.in_dim if i==0 else n_output_dims,
                                   out_dim=self.dim_hidden, w0=self.omegas)
             for i in range(self.num_layers-1)])

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)

        w_std = (1 / in_dim) if self.is_first else (sqrt(c / in_dim) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if not self.is_last:
            out = self.activation(out)
        return out

class ParameterEncoder(nn.Module):
    def __init__(self, grid_shape, num_feat_1d):
        super().__init__()
        self.line_dims = grid_shape
        self.line_dimid = list(range(0,len(grid_shape)))
        self.lines = []

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=(0.01)**(1/len(self.line_dimid)), b=(0.02)**(1/len(self.line_dimid)))
            self.lines.append(line)
        self.lines = torch.nn.ParameterList(self.lines)

    def forward(self, coords):
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = coords[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            #f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.long)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.long)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        return param_feats.T

class NGPM(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(self,
                 encoding="hashgrid", 
                 num_layers=3, 
                 hidden_dim=64, 
                 spatial_dim=2, 
                 ensemble_dim=3,
                 ensemble_embed_dim = 32,
                 out_dim=1,
                 num_levels=13, 
                 level_dim=2, 
                 base_resolution=16, 
                 log2_hashmap_size=24, 
                 desired_resolution=256
                 ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.ensemble_dim = ensemble_dim
        self.out_dim = out_dim
        self.base_resolution = base_resolution
        self.max_resolution = desired_resolution
        self.n_levels = num_levels
        self.F = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.num_layers = num_layers
        self.dim_hidden = hidden_dim

        self.w0 = 30.0
        self.use_bias = True
        self.omegas = 30
        
        per_level_scale = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1)
        ).tolist()

        self.spatial_embedding = MertricEmbedding(self.spatial_dim)

        print(
            f'hash INFO: base_reso={self.base_resolution} '
            f'max_reso={self.max_resolution} up_sacle={per_level_scale:5f} '
            f'per_channels={2} hash_lengh=2^{self.log2_hashmap_size} '
            f'levels={self.n_levels} '
        )

        self.encoder_s = tcnn.Encoding(
                n_input_dims=self.spatial_dim,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": self.n_levels,
                    "n_features_per_level": self.F,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},dtype=torch.float)


        self.ensemble_feature_dim = ensemble_embed_dim

        self.net = Net(self.encoder_s.n_output_dims+self.ensemble_feature_dim, self.spatial_dim, self.dim_hidden, self.num_layers)
        self.modulator = Modulators(self.encoder_s.n_output_dims+self.ensemble_feature_dim, self.spatial_dim, self.dim_hidden, self.num_layers)
        
        self.last_layer = SirenLayer(in_dim=self.dim_hidden, out_dim=self.out_dim, w0=self.w0, use_bias=self.use_bias, is_last=True)

    def get_optparam_groups(self, lr_init_grid = 1e-3, lr_init_network = 1e-4):
        grad_vars = []
        grad_vars += [{'params': self.encoder_s.parameters(), 'lr': lr_init_grid}]
        grad_vars += [{'params': self.net.parameters(), 'lr': lr_init_network}]
        grad_vars += [{'params': self.modulator.parameters(), 'lr':lr_init_network}]
        grad_vars += [{'params': self.last_layer.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def forward(self, coords, encoded_outputs_e):

        xyz_coords = coords[:,0:self.spatial_dim]

        xyz_flatten = xyz_coords.view(-1, self.spatial_dim)

        ### Process XYZ 

        coord_xyz = self.spatial_embedding(xyz_coords)

        encoded_outputs_s = torch.stack([self.encoder_s(coord_xyz[i]) for i in range(coord_xyz.size(0))], dim=1)

        ### Merge XYZ and E 

        encoded_outputs = torch.stack([torch.cat((encoded_outputs_s[:,i,:,].unsqueeze(1), encoded_outputs_e.unsqueeze(1)), dim=-1) for i in range(encoded_outputs_s.size(1))], dim=1).squeeze()

        xyze = torch.mean(encoded_outputs, dim=1).float()

        encoded_outputs = encoded_outputs.permute(1, 0, 2).contiguous() # [num_grid, b, F*num_levels]

        for (i, layer) in enumerate(self.net.net):
            if i==0:
                modulate = self.modulator.modulators[i](xyz_flatten)
            elif encoded_outputs.shape[0]>=i:
                modulate = self.modulator.modulators[i](encoded_outputs[i-1].float())
            else:
                modulate = self.modulator.modulators[i](encoded_outputs[-1].float())

            backbone = layer(xyze)

            modulate = modulate*modulate

            xyze = modulate*backbone

        out = self.last_layer(xyze)
        return out 

    @staticmethod
    @torch.no_grad()
    def sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
    
    @staticmethod
    @torch.no_grad()
    def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
