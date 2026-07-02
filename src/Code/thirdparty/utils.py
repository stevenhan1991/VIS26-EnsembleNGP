import os
import numpy as np
from skimage.transform import resize
import sys
import tqdm
import time
import pdb
import copy
import configparser
from omegaconf import OmegaConf
import argparse
import yaml
import numpy as np
from scipy import io
from scipy.stats import entropy
from math import e
import torch
from torch.nn import Parameter
import struct
from model import *

def get_3d_dim():
    dim1d = 16
    param_fdim = 16
    out_features = 1

    for spatial_fdim in range(64,2,-1):
        for dim3d in range(32,1,-1):
            #for dim2d in range(128,1,-1):
            dim2d = 4*dim3d
            feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*dim3d, np.ones(3, dtype=np.int32)*dim2d, np.ones(3, dtype=np.int32)*dim1d))
            inr_fg = INR_FG(feature_grid_shape, spatial_fdim, spatial_fdim, param_fdim, out_features)
            print('Model Size is {:f} MB under 3D Dim {:d}, 2D Dim {:d}, and Spatial Dim {:d}'.format(count_parameters(inr_fg)*4/1024/1024, dim3d, dim2d, spatial_fdim))

def imp_func(data, data_size, minval, maxval, bw, maxidx, num_bins):
        freq = None
        nBlocks = 2
        block_size = data_size // nBlocks
        for bidx in range(nBlocks):
            block_freq = torch.histc(data[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=minval, max=maxval).type(torch.long)
            if freq is None:
                freq = block_freq
            else:
                freq += block_freq
        freq = freq.type(torch.double)
        importance = 1. / freq
        importance_idx = torch.clamp((data - minval) / bw, min=0.0, max=maxidx).type(torch.long)
        return importance, importance_idx


def get_n_features(opt):
    #cr_target = [3488,6473,10099,16949,35913,152356]
    #idx = 0

    if opt.model.model == 'fVSRN':
        for j in range(180,1,-1):
            opt.model.nodes_per_layer = j
            model = fVSRN(opt)
            num_parms = count_parameters(model)
            cr = opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms
            if np.abs(cr-opt.model.cr)/cr <= 0.05:
                    print('CR is {:4f} with {:d} Features'.format(cr,j))
                    break
    elif opt.model.model == 'APMGSRN':
        for j in range(180,1,-1):
            opt.model.nodes_per_layer = j
            model = APMGSRN(opt)
            num_parms = count_parameters(model)

            cr = opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms
            if np.abs(cr-opt.model.cr)/cr <= 0.05:
                    print('CR is {:4f} with {:d} Features'.format(cr,j))
                    break
    elif opt.model.model == 'NGP':
        for j in range(180,1,-1):
            opt.model.hash_log2_size = j
            for k in range(180,1,-1):
                opt.model.nodes_per_layer = k
                model = NGP(opt)

                num_parms = count_parameters(model)

                cr = opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms

                if np.abs(cr-opt.model.cr)/cr <= 0.05:
                    print('CR is {:4f} with {:d} Hash Size {:d} Features'.format(cr,opt.model.hash_log2_size,opt.model.nodes_per_layer))
                    break
                    break
    


def forward_maxpoints(model, coords, out_dim=1, max_points=100000, 
                      data_device="cuda", device="cuda"):
    output_shape = list(coords.shape)
    output_shape[-1] = out_dim
    output = torch.empty(output_shape, 
        dtype=torch.float32, 
        device=data_device)
    
    for start in range(0, coords.shape[0], max_points):
        with torch.cuda.amp.autocast():
            output[start:min(start+max_points, coords.shape[0])] = \
            model(coords[start:min(start+max_points, coords.shape[0])].to(device)).to(data_device)
    return output


def make_coord_grid(shape, device, flatten=True, align_corners=False, use_half=False):
    """ 
    Make coordinates at grid centers.
    return (shape.prod, 3) matrix with (z,y,x) coordinate
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        left = -1.0
        right = 1.0
        if(align_corners):
            r = (right - left) / (n-1)
            seq = left + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()

        else:
            r = (right - left) / (n+1)
            seq :torch.Tensor = left + r + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()
            
        if(use_half):
                seq = seq.half()
        coord_seqs.append(seq)

    ret = torch.meshgrid(*coord_seqs, indexing="ij")
    ret = torch.stack(ret, dim=-1)
    if(flatten):
        ret = ret.view(-1, ret.shape[-1])
    return ret.flip(-1)

def save_yaml(filepath, info):
    with open(filepath+'',"w") as f:
        yaml.dump(info,f)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config



def get_mgrid(sidelen, dim=2, s=1,t=0,ghost_cell=None):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    
    if ghost_cell:
        pixel_coords = pixel_coords[:,ghost_cell[4]:sidelen[2]-ghost_cell[5],ghost_cell[2]:sidelen[1]-ghost_cell[3],ghost_cell[0]:sidelen[0]-ghost_cell[1],:,]
    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    return pixel_coords

def vorticity_3d(dx,dy,dz,u,v,w):

    dFx_dy = np.gradient(u, dy, axis = 1)
    dFx_dz = np.gradient(u, dz, axis = 2)
    dFy_dx = np.gradient(v, dx, axis = 0)
    dFy_dz = np.gradient(v, dz, axis = 2)
    dFz_dx = np.gradient(w, dx, axis = 0)
    dFz_dy = np.gradient(w, dy, axis = 1)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy


    vorticity = [rot_x,rot_y,rot_z]

    av = np.sqrt(rot_x**2+rot_y**2+rot_z**2)

    return av


def acceleration(u,v,w,dx,dy,dz,dt):
    ### ref: https://www.youtube.com/watch?v=PpY9k6QEqo0
    dFx_dx = np.gradient(u, dx, axis = 1)
    dFx_dy = np.gradient(u, dy, axis = 2)
    dFx_dz = np.gradient(u, dz, axis = 3)
    dFx_dt = np.gradient(u, dt, axis = 0)

    dFy_dx = np.gradient(v, dx, axis = 1)
    dFy_dy = np.gradient(v, dy, axis = 2)
    dFy_dz = np.gradient(v, dz, axis = 3)
    dFy_dt = np.gradient(v, dt, axis = 0)

    dFz_dx = np.gradient(w, dx, axis = 1)
    dFz_dy = np.gradient(w, dy, axis = 2)
    dFz_dz = np.gradient(w, dz, axis = 3)
    dFz_dt = np.gradient(w, dt, axis = 0)

    acc_x = u*dFx_dx+v*dFx_dy+w*dFx_dz+dFx_dt
    acc_y = u*dFy_dx+v*dFy_dy+w*dFy_dz+dFy_dt
    acc_z = u*dFz_dx+v*dFz_dy+w*dFz_dz+dFz_dt

    av = np.sqrt(np.power(acc_x,2.0) + np.power(acc_y,2.0) + np.power(acc_z,2.0))

    return  av

class UnfoldNd(torch.nn.Module):
    '''
        Unfolding operator for 3D tensors with disjoint folding. The operator
        is a drop in replacement for unfoldNd.UnfoldNd
        
        Inputs:
            kernel_size: Folding kernel size
            stride: Folding stride. This is just for compatibility, as the 
                stride is fixed to be the same as kernel_size
            tensor: (N, C, H, W, T) sized tensor
                
        Outputs:
            unfolded_tensor: (N, kernel_size**3, -1) sized unfolded tensor
    '''
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, tensor):
        N, C, H, W, T = tensor.shape
        
        fold1 = tensor.unfold(2, self.ksize[2], self.ksize[2])
        fold2 = fold1.unfold(3, self.ksize[1], self.ksize[1])
        fold3 = fold2.unfold(4, self.ksize[0], self.ksize[0])
        
        collapse1 = fold3.reshape(N, -1, self.ksize[2], self.ksize[1], self.ksize[0])
        collapse2 = collapse1.reshape(N, -1, np.prod(self.ksize))
        
        # To maintain compatibility, swap the two axes
        collapse3 = collapse2.permute(0, 2, 1)
        
        return collapse3
    
class FoldNd(torch.nn.Module):
    '''
        Folding operator for 3D tensors with disjoint folding. The operator
        is a drop in replacement for unfoldNd.FoldNd
        
        Inputs:
            output_size: (H, W, T) -- Three-tuple with output size
            kernel_size: Folding kernel size
            stride: Folding stride. This is just for compatibility, as the 
                stride is fixed to be the same as kernel_size
                
        Outputs:
            folded_tensor: (N, C, H, W, T) sized folded tensor
    '''
    def __init__(self, output_size, kernel_size, stride=None):
        super().__init__()
        
        self.output_size = output_size
        self.ksize = kernel_size
        self.stride = stride
        
        # Create a folded set of indices here
        self.numel = output_size[0]*output_size[1]*output_size[2]
        idx = torch.arange(self.numel, dtype=torch.int64)
        idx_cube = idx.reshape(1, 1, *output_size)
        
        self.unfolding = UnfoldNd(self.ksize)
        self.folded_idx = self.unfolding(idx_cube)

    def forward(self, tensor, output=None):
        if output is None:
            output = torch.zeros(self.numel, device=tensor.device)
        else:
            output = output.flatten()
            
        output[self.folded_idx] = tensor
        output_cube = output.reshape(1, 1, *self.output_size)
        
        return output_cube


def get_coords(imsize, ksize, unfold):
    '''
        Generate coordinates for MINER training
        
        Inputs:
            imsize: (H, W) image size
            ksize: Kernel size
            coordstype: 'global' or 'local'
            unfold: Unfold operator
    '''
    H, W, L = imsize    
    nchunks = int(H*W*L/(np.prod(ksize)))
    X, Y, Z = torch.meshgrid(torch.linspace(0, H-1, H),
                                 torch.linspace(0, W-1, W),
                                 torch.linspace(0, L-1, L))
    coords = torch.cat((X[None, None, ...],
                        Y[None, None, ...],
                        Z[None, None, ...]), 0)
    coords_chunked = unfold(coords).permute(2, 1, 0)
    return coords_chunked

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.inference_mode()
def field_from_net(dataset, net, tiled_res=32):
    target_res = dataset.vol_res
    net = net.cuda()
    full_vol = torch.zeros(target_res)
    for xdx in np.arange(0,target_res[0],tiled_res):
        x_begin = xdx
        x_end = xdx+tiled_res if xdx+tiled_res <= target_res[0] else target_res[0]
        for ydx in np.arange(0,target_res[1],tiled_res):
            y_begin = ydx
            y_end = ydx+tiled_res if ydx+tiled_res <= target_res[1] else target_res[1]
            for zdx in np.arange(0,target_res[2],tiled_res):
                z_begin = zdx
                z_end = zdx+tiled_res if zdx+tiled_res <= target_res[2] else target_res[2]
                tile_resolution = torch.tensor([x_end-x_begin,y_end-y_begin,z_end-z_begin],dtype=torch.int)
                min_alpha_bb = torch.tensor([x_begin/(target_res[0]-1),y_begin/(target_res[1]-1),z_begin/(target_res[2]-1)],dtype=torch.float)
                max_alpha_bb = torch.tensor([(x_end-1)/(target_res[0]-1),(y_end-1)/(target_res[1]-1),(z_end-1)/(target_res[2]-1)],dtype=torch.float)
                min_bounds = dataset.min_bb + min_alpha_bb*(dataset.max_bb-dataset.min_bb)
                max_bounds = dataset.min_bb + max_alpha_bb*(dataset.max_bb-dataset.min_bb)
                with torch.no_grad():
                    tile_positions = dataset.scales.view(1,1,1,3)*dataset.tile_sampling(min_bounds,max_bounds,tile_resolution)
                    tile_positions = tile_positions.unsqueeze(0).cuda()
                    tile_vol = net(tile_positions.unsqueeze(0)).squeeze(0).squeeze(-1)
                    full_vol[x_begin:x_end,y_begin:y_end,z_begin:z_end] = tile_vol.cpu()
    return full_vol

def PSNR(vol,preds):
    mse = np.mean((vol-preds)**2)
    psnr = 20*np.log10(vol.max()-vol.min())-10*np.log10(mse)
    return psnr

def compute_entropy(data):
    values, counts = np.unique(data,return_counts=True)
    return entropy(counts,base=e)


def get_param_groups(models):
    param_groups = {}
    for model in models:
        for name, p in model.named_parameters():
            if p.requires_grad:
                if name.startswith('backbone'):
                    k = 'dec'
                elif name.startswith('encoder'):
                    k = 'hg'
                elif name.startswith('hg0'):
                    k = 'hg0'
                elif name.startswith('lc0'):
                    k = 'lc0'
                elif name.startswith('lcb0'):
                    k = 'lcb0'
                elif name.startswith('kc0'):
                    k = 'kc0'
                elif name.startswith('ks0'):
                    k = 'ks0'
                elif name.startswith('net'):
                    k = 'dec'
                elif name.startswith('transformer'):
                    k = 'dec'
                elif name.startswith('last_layer'):
                    k = 'dec'
                elif name.startswith('modulator'):
                    k = 'dec'
                elif name.startswith('mlp_head'):
                    k = 'dec'
                elif name.startswith('ensemble'):
                    k = 'dec'
                elif name.startswith('decoder'):
                    pass
                elif name.startswith('conv'):
                    pass
                else:
                    raise NotImplementedError
                if k not in param_groups:
                    param_groups[k] = []
                param_groups[k].append(p)
                #print(k, name, p.shape)
    return param_groups

def configure_optimizers(param_groups, hparams):
    optims = {}
    for k in param_groups.keys():
        hparams_k = hparams[k]
        optim_type = hparams_k['type']
        optim_fn = parse_optim_type(optim_type)
        if optim_type == 'SGD':
            optims[k] = optim_fn(param_groups[k], lr=hparams_k['lr'], weight_decay=hparams_k['wd'])
        elif optim_type == 'SparseAdam':
            optims[k] = optim_fn(param_groups[k], lr=hparams_k['lr'], betas=hparams_k['betas'], eps=hparams_k['eps'])
        else:
            optims[k] = optim_fn(param_groups[k], lr=hparams_k['lr'], betas=hparams_k['betas'], eps=hparams_k['eps'], weight_decay=hparams_k['wd'])
    return optims

def parse_optim_type(optim_type):
    if optim_type == 'SparseAdam':
        return torch.optim.SparseAdam
    elif optim_type == 'Adam':
        return torch.optim.Adam
    elif optim_type == 'RAdam':
        return torch.optim.RAdam
    elif optim_type == 'SGD':
        return torch.optim.SGD
    else:
        raise NotImplementedError

def parse_lr_sch_type(lr_sch_type):
    if lr_sch_type == 'ngp':
        return lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.33**max(0, step//10000))
    elif lr_sch_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    elif lr_sch_type == 'linear':
        return torch.optim.lr_scheduler.MultiStepLR
    else:
        raise NotImplementedError


def configure_lr_schedulers(optims, hparams):
    lr_schs = {}
    for k in optims.keys():
        hparams_k = hparams[k]
        lr_sch_type = hparams_k['type']
        lr_sch_fn = parse_lr_sch_type(lr_sch_type)
        if lr_sch_type == 'ngp':
            lr_schs[k] = lr_sch_fn(optims[k])
        elif lr_sch_type == 'cosine':
            lr_schs[k] = lr_sch_fn(optims[k], T_max=hparams_k['T_max'], eta_min=optims[k].param_groups[0]['lr']/hparams_k['gamma'])
        elif lr_sch_type == 'linear':
            lr_schs[k] = lr_sch_fn(optims[k],milestones=[500,750,900,950])
    return lr_schs


