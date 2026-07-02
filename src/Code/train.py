import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.optim as optim
import time
from model import *
from utils import *
import yaml
import math
from torch.utils.data import DataLoader
from ngp import *
import csv
from copy import deepcopy

def train_decoder(config,volume,device):

    criterion = nn.MSELoss()

    train_dl = DataLoader(volume,batch_size=1,shuffle=True,num_workers=0, pin_memory=False)

    learning_rate = 2e-5

    model = EnsembleNet(**config.model.ensemblenet).to(device)

    num_parms = count_parameters(model)

    print('Num of Params = '+str(num_parms))
    print('Model Size is {:f} MB'.format(num_parms*4/1024/1024))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.999))

    loss_ = 1e5

    info = {'Loss':[],'Time':0}

    training_time = 0

    for epoch in range(1,200+1):

        loss_mse = 0

        epoch_loss = {'loss':0}

        x = time.time()
        
        for batch_idx, data in enumerate(train_dl):

            params = data['params'].cuda()
            v = data['vols'].cuda()

            params = (params + 1.0) / 2.0
            v = (v + 1.0) / 2.0

            optimizer.zero_grad()

            if len(params.shape) == 3:
                params = params.squeeze(1)

            if len(v.shape) == 4:
                v = v[None,...]

            v_pred = model(params)
            mse = criterion(v_pred,v)

            mse.backward()
            optimizer.step()

            loss_mse += mse.mean().item()
            epoch_loss['loss'] += mse.item()
        
        y = time.time()

        info['Time'] += y-x

        info['Loss'].append(epoch_loss['loss'])
        print('Loss at epoch {:d} is {:f}'.format(epoch,info['Loss'][-1]))
        save_yaml(config.model.model_path+'embedder_{:d}_loss.yaml'.format(config.model.ensemblenet.ch),info)

        if loss_ > info['Loss'][-1]:
            loss_ = info['Loss'][-1]
            torch.save(model.param_embeder.state_dict(),config.model.model_path+'embedder_{:d}_best_model.pth'.format(config.model.ensemblenet.ch))
            torch.save(model.state_dict(),config.model.model_path+'ensemble_{:d}_best_model.pth'.format(config.model.ensemblenet.ch))


def train_ngp(config,volume,device):

    criterion = nn.MSELoss()

    train_dl = DataLoader(volume,batch_size=1,shuffle=True, num_workers=0, pin_memory=False)
    model = NGPM(**config.model.ngp).to(device)

    embedder = ParamerEmedding(config.model.ensemblenet.in_dim, config.model.ensemblenet.ch).to(device)
    embedder.load_state_dict(torch.load(config.model.model_path+'embedder_{:d}_best_model.pth'.format(config.model.ensemblenet.ch),map_location=device))

    num_parms = count_parameters(model) + count_parameters(embeder)

    print('Num of Total Params = '+str(num_parms))
    print('Model Size is {:f} MB'.format(num_parms*2/1024/1024))

    eps = 1e-15

    optims = {}
    optims["dec"] = {'type': 'Adam', 'lr': 2e-5, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}
    optims["hg"] = {'type': 'Adam', 'lr': 2e-5, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}

    # lr schedulers
    T_max = 400
    lr_gamma = 100
    lr_schs = {}
    lr_schs["dec"] = {'type': 'linear', 'T_max': T_max, 'gamma': lr_gamma}
    lr_schs["hg"] = {'type': 'linear', 'T_max': T_max, 'gamma': lr_gamma}

    param_groups = get_param_groups([model])
    optims = configure_optimizers(param_groups, optims)
    #lr_schs = configure_lr_schedulers(optims, lr_schs)

    loss_ = 1e5

    val = np.fromfile(config.model.result_path+'{:d}.dat'.format(config.model.ensemblenet.ch),dtype='<f')
    val = torch.from_numpy(val).to(device)

    info = {'MSE Loss':[],'Time':0}

    training_time = 0

    lossfn = nn.MSELoss()

    epoch = 1

    while epoch <= T_max:

        x = time.time()

        loss_mse = 0

        for idx, data in enumerate(train_dl):

            coords = data['coords'].squeeze()
            values = data['values']

            coords = coords.cuda()
            values = values.cuda()

            coords = (coords + 1.0) / 2.0

            values = (values + 1.0) / 2.0

            
            for k, v in optims.items():
                v.zero_grad() 

            ensemble_features = embedder(coords[:,3:])

            if val > 1:
                ensemble_features /= val

            ensemble_features = (ensemble_features + 1.0) / 2.0

            v_pred = model(coords[:,0:3], ensemble_features)

            mse = lossfn(v_pred.squeeze(),values.squeeze())

            mse.backward()

            loss_mse += mse.mean().item()

        
            for k, v in optims.items():
                v.step()

        y = time.time()
        
        info['Time'] += y-x

        info['MSE Loss'].append(float(loss_mse))

        print('Loss at epoch {:d} is {:f}'.format(epoch, info['MSE Loss'][-1]))
        
        save_yaml(config.model.model_path+'{:d}_{:s}_{:d}_loss.yaml'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size),info)
    
        if loss_ > info['MSE Loss'][-1]:
            loss_ = info['MSE Loss'][-1]
            model_half = deepcopy(model).half()
            torch.save(model_half.state_dict(),config.model.model_path+'{:d}_{:s}_{:d}_best_model.pth'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size))

        epoch += 1
        embedder_half = deepcopy(embeder).half()
        torch.save(embedder_half.state_dict(),config.model.model_path+'embedder_{:d}_best_model.pth'.format(config.model.ensemblenet.ch))
