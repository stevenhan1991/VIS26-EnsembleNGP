import os
import numpy as np
from skimage.transform import resize
import sys
from tqdm import tqdm
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
import csv
from drr import *

def extract_embedding(config, device):
    ensemble_parms = []
    if config.data.dataset == 'Nyx':
      with open('../Params/nyx.csv', newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            OmB = float(row[0])
            OmM = float(row[1])
            h = float(row[2])
            OmB = 2.0 * (OmB - 0.0215) / (0.0235 - 0.0215) - 1.0
            OmM = 2.0 * (OmM - 0.12000) / (0.1550 - 0.1200) - 1.0
            h = 2.0 * (h - 0.5500) / (0.8500 - 0.5500) - 1.0
            ensemble_parms.append(torch.FloatTensor([OmB,OmM,h]).float())
    elif config.data.dataset == 'Castro':
      with open('../Params/castro.csv', newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
          p1 = float(row[0])
          p2 = float(row[1])
          p1 = 2.0 * (p1 - 0.8) / (0.95 - 0.8) - 1.0
          p2 = 2.0 * (p2 - 0.8) / (0.95 - 0.8) - 1.0
          ensemble_parms.append(torch.FloatTensor([p1,p2]).float())
    elif config.data.dataset == 'Colver':
        with open('../Params/colverleaf3d.csv', newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                d1 = float(row[0])
                d2 = float(row[1])
                d3 = float(row[2])
                e1 = float(row[3])
                e2 = float(row[4])
                e3 = float(row[5])
                d1 = 2.0 * (d1 - 0.505) / (1.0 - 0.505) - 1.0
                d2 = 2.0 * (d2 - 1.0) / (2.0 - 1.0) - 1.0
                d3 = 2.0 * (d3 - 1.5) / (3.0 - 1.5) - 1.0
                e1 = 2.0 * (e1 - 0.75) / (2.0 - 0.75) - 1.0
                e2 = 2.0 * (e2 - 1.5) / (3.5 - 1.5) - 1.0
                e3 = 2.0 * (e3 - 4.0) / (7.0 - 4.0) - 1.0
                ensemble_parms.append(torch.FloatTensor([d1, d2, d3, e1, e2, e3]).float())
    elif config.data.dataset == 'MPAS-Ocean':
        with open('../Params/MPAS-Ocean.csv', newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                BwsA = float(row[0])
                CbrN = float(row[1])
                GM = float(row[2])
                HV = float(row[3])
                BwsA = 2.0 * (BwsA - 0.0) / (5.0 - 0.0) - 1.0
                CbrN = 2.0 * (CbrN - 0.25) / (1.0 - 0.25) - 1.0
                GM = 2.0 * (GM - 600) / (1500.0 - 600) - 1.0
                HV = 2.0 * (HV - 100) / (300.0 - 100) - 1.0
                ensemble_parms.append(torch.FloatTensor([BwsA, CbrN, GM, HV]).float())
    

    selected_idx = []
      
    ensemble_parms = torch.stack(ensemble_parms)

    model = EnsembleNet(**config.model.ensemblenet).to(device)
    model.load_state_dict(torch.load(config.model.model_path+'ensemble_{:d}_best_model.pth'.format(config.model.ensemblenet.ch),map_location=device))
    min_v = []
    max_v = []
    for idx in tqdm(range(0,len(ensemble_parms))):
      if idx not in selected_idx:
        for t in range(0,config.data.time_steps):
          '''
          d = np.fromfile('/root/autodl-tmp/Data/{:s}_reduced/{:04d}/{:04d}.dat'.format(config.data.dataset, idx, t),dtype='<f')
          #d = d.reshape((config.data.res[2], config.data.res[1], config.data.res[0])).transpose()
          d = d.reshape((128,128,128)).transpose()
          d = (d-d.min())/(d.max()-d.min())
          '''

          ensemble_param = ensemble_parms[idx][None,...]

          #print(ensemble_param)

          time_chunks = torch.zeros((ensemble_param.size(0),1))
          time = t/(config.data.time_steps-1)
          time -= 0.5
          time *= 2.0
          time_chunks.fill_(time)

          params = torch.cat((ensemble_param, time_chunks),dim=-1).float().to(device)

          params = (params + 1.0) / 2.0

          embeding = model.get_ensmeble_embedding(params).detach().cpu().numpy()
        
          #print('Min and Max is {:f} and {:f} at Ensemble {:d} with Time Step {:d}'.format(embeding.min(), embeding.max(), idx, t))
          '''
          pred = model(params).detach().cpu().numpy()[0][0]
          p = PSNR(d, pred)
          print('PSNR at Ensemble {:d} Time Step {:d} is {:f}'.format(idx, t+1, p))
          '''
          min_v.append(embeding.min())
          max_v.append(embeding.max())

    min_v = min(min_v)
    max_v = max(max_v)

    print(min_v)
    print(max_v)

    print('Min and Max is {:f} and {:f}'.format(min_v, max_v))
    v = max([-min_v, max_v])
    print('The Stored Value is {:f}'.format(v))
    values = np.asarray([v], dtype='<f')
    values.tofile(config.model.result_path+'{:d}.dat'.format(config.model.ensemblenet.ch),format='<f')


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


def PSNR(vol,preds):
    mse = np.mean((vol-preds)**2)
    psnr = 20*np.log10(vol.max()-vol.min())-10*np.log10(mse)
    return psnr

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
                print(k, name, p.shape)
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


