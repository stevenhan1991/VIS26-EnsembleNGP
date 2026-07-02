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

def train_INR(config,volume,device):

    criterion = nn.MSELoss()

    train_dl = DataLoader(volume,batch_size=1,shuffle=True,num_workers=0, pin_memory=False)

    if config.model.model == 'fVSRN':
        learning_rate = 5e-4
        model = fVSRN(config).to(device)
    else:
        learning_rate = 2e-5
        model = CoordNet(**config.model.coordnet).to(device)

    num_parms = count_parameters(model)

    model_size = num_parms*4/1024/1024

    print('Model Size is {:f} MB'.format(model_size))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.999))

    loss_ = 1e5

    info = {'MSE Loss':[],'Time':0}

    training_time = 0

    for epoch in range(1,500+1):

        loss_mse = 0

        epoch_loss = 0

        x = time.time()
        
        for batch_idx, data in enumerate(train_dl):

            coords = data['coords'].squeeze()
            values = data['values']

            coords = coords.cuda()
            values = values.cuda()

            coords = (coords + 1.0) / 2.0

            values = (values + 1.0) / 2.0

            optimizer.zero_grad()

            v_pred = model(coords)
            mse = criterion(v_pred.view(-1),values.view(-1))

            mse.backward()
            optimizer.step()

            loss_mse += mse.mean().item()
            epoch_loss += mse.item()
        
        y = time.time()

        info['Time'] += y-x

        info['MSE Loss'].append(epoch_loss)
        print('Loss at epoch {:d} is {:f}'.format(epoch,info['MSE Loss'][-1]))
        save_yaml(config.model.model_path+'{:s}_{:.2f}_loss.yaml'.format(config.model.model,model_size),info)
    
        if loss_ > info['MSE Loss'][-1]:
            loss_ = info['MSE Loss'][-1]
            torch.save(model.state_dict(),config.model.model_path+'{:s}_{:.2f}_best_model.pth'.format(config.model.model, model_size))

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

    for epoch in range(1,400+1):

        loss_mse = 0

        epoch_loss = {'loss':0}

        x = time.time()
        
        for batch_idx, data in enumerate(train_dl):

            params = data['params'].cuda()
            v = data['vols'].cuda()

            params = (params + 1.0) / 2.0
            v = (v + 1.0) / 2.0

            optimizer.zero_grad()

            if len(params.shape) == 1:
                params = params[None,...]

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
        save_yaml(config.model.model_path+'ensemble_{:d}_{:f}_loss.yaml'.format(config.model.ensemblenet.ch, config.data.sample_interval),info)

        if loss_ > info['Loss'][-1]:
            loss_ = info['Loss'][-1]
            torch.save(model.state_dict(),config.model.model_path+'ensemble_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.data.sample_interval))


def train_ngp(config,volume,device):

    criterion = nn.MSELoss()

    train_dl = DataLoader(volume,batch_size=1,shuffle=True, num_workers=0, pin_memory=False)

    if config.model.model == 'NGP':
        model = NGP(**config.model.ngp).to(device)
    elif config.model.model == 'NGPM':
        model = NGPM(**config.model.ngp).to(device)

    decoder = EnsembleNet(**config.model.ensemblenet).to(device)
    decoder.load_state_dict(torch.load(config.model.model_path+'ensemble_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.data.sample_interval),map_location=device))

    num_parms = count_parameters(model) + count_parameters(decoder.ensemble_embeder)

    print('Num of Total Params = '+str(num_parms))
    print('Model Size is {:f} MB'.format(num_parms*2/1024/1024))

    eps = 1e-15

    optims = {}
    optims["dec"] = {'type': 'Adam', 'lr': 2e-5, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}
    optims["hg"] = {'type': 'Adam', 'lr': 2e-5, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}

    # lr schedulers
    T_max = 500
    lr_gamma = 100
    lr_schs = {}
    lr_schs["dec"] = {'type': 'linear', 'T_max': T_max, 'gamma': lr_gamma}
    lr_schs["hg"] = {'type': 'linear', 'T_max': T_max, 'gamma': lr_gamma}

    param_groups = get_param_groups([model])
    optims = configure_optimizers(param_groups, optims)
    #lr_schs = configure_lr_schedulers(optims, lr_schs)

    loss_ = 1e5

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

            ensemble_features = decoder.get_ensmeble_embedding(coords[:,3:])

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
        save_yaml(config.model.model_path+'{:d}_{:s}_{:d}_{:f}_loss.yaml'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size, config.data.sample_interval),info)
    
        if loss_ > info['MSE Loss'][-1]:
            loss_ = info['MSE Loss'][-1]
            torch.save(model.state_dict(),config.model.model_path+'{:d}_{:s}_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size, config.data.sample_interval))
        epoch += 1


def train_exploreINR(config, dim3d, device):
    # reference: https://github.com/YiTangChen/ExplorableINR/tree/main

    out_features = 1
    nEnsemble = config.model.ensemblenet.in_dim
    data_size = np.prod(config.data.res)

    sf_sr = 0.05
    if config.data.dataset == 'Nyx':
        batch_size = 2**17
    else:
        batch_size = 2**18

    sp_sr = 0.3
    dim2d = 4*dim3d
    dim1d = 16
    spatial_fdim = 64
    param_fdim = 16

    num_sf_batches = math.ceil(nEnsemble * data_size * sf_sr / batch_size)
    num_sp_sampling = math.ceil(70 * sp_sr)

    feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*dim3d, np.ones(3, dtype=np.int32)*dim2d, np.ones(config.model.ensemblenet.in_dim, dtype=np.int32)*dim1d))

    inr_fg = INR_FG(feature_grid_shape, spatial_fdim, spatial_fdim, param_fdim, out_features)

    model_size = count_parameters(inr_fg)*4/1024/1024

    print('Model Size is {:f} MB'.format(model_size))
    
    inr_fg.to(device)

    ensemble_parms = []

    coords = get_mgrid(config.data.res, dim=3)
    coords_torch = torch.from_numpy(coords).to(device)

    if config.data.dataset == 'Nyx':
        with open('../Params/nyx.csv', newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
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
            next(reader)
            for row in reader:
                p1 = float(row[0])
                p2 = float(row[1])
                p1 = 2.0 * (p1 - 0.8) / (0.95 - 0.8) - 1.0
                p2 = 2.0 * (p2 - 0.8) / (0.95 - 0.8) - 1.0
                ensemble_parms.append(torch.FloatTensor([p1,p2]).float())
    elif config.data.dataset == 'Colver':
        with open('../Params/colverleaf.csv', newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                d1 = float(row[0])
                d2 = float(row[1])
                d3 = float(row[2])
                e1 = float(row[3])
                e2 = float(row[4])
                e3 = float(row[5])
                d1 = 2.0 * (d1 - 0.01) / (1.0 - 0.01) - 1.0
                d2 = 2.0 * (d2 - 0.5) / (2.0 - 0.5) - 1.0
                d3 = 2.0 * (d3 - 1.5) / (3.0 - 1.5) - 1.0
                e1 = 2.0 * (e1 - 0.75) / (2.0 - 0.75) - 1.0
                e2 = 2.0 * (e2 - 1.5) / (3.5 - 1.5) - 1.0
                e3 = 2.0 * (e3 - 4.0) / (7.0 - 4.0) - 1.0

                ensemble_parms.append(torch.FloatTensor([d1, d2, d3, e1, e2, e3]).float())
        
    ensemble_parms = torch.stack(ensemble_parms)
        
    selected_idx = [int(i) for i in np.arange(0, config.data.num_ensemble, config.data.sample_interval)] + [config.data.num_ensemble-1]

    data_dicts = torch.zeros(len(selected_idx), data_size)
    selected_ensemble_params = torch.zeros(len(selected_idx), config.model.ensemblenet.in_dim)

    for i in range(len(selected_idx)):
        idx = selected_idx[i]
        curr_scalar_field = np.fromfile(config.data.path+'{:04d}.dat'.format(idx+1),dtype='<f')
        curr_scalar_field = 2.0 * (curr_scalar_field-curr_scalar_field.min()) / (curr_scalar_field.max()-curr_scalar_field.min()) - 1.0
        curr_scalar_field = torch.from_numpy(curr_scalar_field)
        data_dicts[i] = curr_scalar_field
        selected_ensemble_params[i] = ensemble_parms[idx]

    data_dicts = data_dicts.to(device)
    selected_ensemble_params = selected_ensemble_params.to(device)

    print('Number of Traning Ensembles are {:d}'.format(len(data_dicts)))

    num_bins = 10
    bin_width = 1.0 / num_bins
    max_binidx_f = float(num_bins-1)
    batch_size_per_field = batch_size 
    nEnsembleGroups_per_epoch = (len(data_dicts)+nEnsemble-1) // nEnsemble

    #####################################################################################

    optimizer = torch.optim.Adam(inr_fg.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    info = {'MSE Loss':[],'Time':0}

    loss_ = 1e5

    if os.path.exists('../Importance/{:s}.npy'.format(config.data.dataset)):
        sfimps_np = np.load('../Importance/{:s}.npy'.format(config.data.dataset))
    else:
        nBlocks = 2
        block_size = data_size // nBlocks
        all_freq = None
        for tdidx, sf in enumerate(data_dicts):
            curr_freq = None
            for bidx in range(nBlocks):
                block_freq = torch.histc(sf[bidx*block_size:(bidx+1)*block_size], bins=num_bins, min=0.0, max=1.0).type(torch.long)
                if curr_freq is None:
                    curr_freq = block_freq
                else:
                    curr_freq += block_freq
            if all_freq is None:
                all_freq = curr_freq
            else:
                all_freq += curr_freq
                print('tdidx: ', tdidx, '  ', torch.sum(all_freq) / data_size)
        all_freq = all_freq.type(torch.double)
        importance = 1. / all_freq
        sfimps = torch.zeros(len(data_dicts))

        for tdidx, sf in enumerate(data_dicts):
            curr_impidx = torch.clamp(sf / bin_width, min=0.0, max=max_binidx_f).type(torch.long)
            curr_sfimp = importance[curr_impidx].sum()
            sfimps[tdidx] = curr_sfimp
            print('tdidx: ', tdidx, '  ', curr_sfimp)
        sfimps = sfimps / sfimps.sum()
        sfimps_np = sfimps.cpu().numpy()
        np.save('../Importance/{:s}.npy'.format(config.data.dataset), sfimps_np)

    sfimps = torch.from_numpy(sfimps_np)

    #####################################################################################

    for epoch in range(1, 501):
        print('epoch {0}'.format(epoch))
        total_loss = 0
        tstart = time.time()
        e_rndidx = torch.multinomial(sfimps, nEnsembleGroups_per_epoch * nEnsemble, replacement=True)
        #print(e_rndidx)
        for egidx in range(nEnsembleGroups_per_epoch):
            scalar_fields = []
            sample_weights_arr = []
            params_batch = None
            errsum = 0

            for eidx in range(nEnsemble):
                curr_scalar_field = data_dicts[e_rndidx[egidx*nEnsemble + eidx]]
                curr_params = selected_ensemble_params[e_rndidx[egidx*nEnsemble + eidx]].reshape(1,nEnsemble)
                curr_params_batch = curr_params.repeat(batch_size_per_field, 1)
                if params_batch is None:
                    params_batch = curr_params_batch
                else:
                    params_batch = torch.cat((params_batch, curr_params_batch), 0)
                curr_imp, curr_impidx = imp_func(curr_scalar_field, data_size, 0.0, 1.0, bin_width, max_binidx_f, num_bins)
                curr_sample_weights = curr_imp[curr_impidx]
                
                scalar_fields.append(curr_scalar_field)
                sample_weights_arr.append(curr_sample_weights)

            #params_batch = curr_params_batch.to(device)

            scalar_fields = torch.stack(scalar_fields)

            for field_idx in range(num_sf_batches):
                coord_batch = None
                value_batch = None
                for eidx in range(nEnsemble):
                    #####
                    rnd_idx = torch.multinomial(sample_weights_arr[eidx].to(device), batch_size_per_field, replacement=True)
                    ######
                    if coord_batch is None:
                        coord_batch = coords_torch[rnd_idx]
                        value_batch = scalar_fields[eidx][rnd_idx]
                    else:
                        coord_batch, value_batch = torch.cat((coord_batch, coords_torch[rnd_idx]), 0), torch.cat((value_batch, scalar_fields[eidx][rnd_idx]), 0)
                # model outputs are float32 but mpaso values are float64
                value_batch = value_batch.reshape(len(value_batch), 1).type(torch.float32)
                #coord_batch = coord_batch.to(device)
                #value_batch = value_batch.to(device)

                coord_batch = (coord_batch + 1.0) / 2.0
                params_batch = (params_batch + 1.0) / 2.0
                value_batch = (value_batch + 1.0) / 2.0

                # ===================forward=====================
                model_output = inr_fg(torch.cat((coord_batch, params_batch), 1))
                loss = criterion(model_output, value_batch)

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        tend = time.time()
        info['MSE Loss'].append(float(total_loss))
        info['Time'] += tend - tstart
        print('Loss at epoch {:d} is {:f}'.format(epoch, info['MSE Loss'][-1]))
        save_yaml(config.model.model_path+'{:s}_{:.2f}_loss.yaml'.format(config.model.model,model_size),info)
    
        if loss_ > info['MSE Loss'][-1]:
            loss_ = info['MSE Loss'][-1]
            torch.save(inr_fg.state_dict(),config.model.model_path+'{:s}_{:.2f}_best_model.pth'.format(config.model.model, model_size))

def train_fVSRN(model,dataset,opt,device):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    model.train(True)

    loss_ = 1e5

    num_parms = count_parameters(model)
    info = {'Loss':[],'Time':0,'Params':num_parms}
    optimizer = optim.Adam(model.parameters(), lr=opt.model.lr,betas=[0.9,0.999]) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[opt.model.iterations*(2/5), opt.model.iterations*(3/5), opt.model.iterations*(4/5)],gamma=0.33)
    early_stopping_data = (False,torch.zeros([opt.model.iterations], dtype=torch.float32, device=device))

    print('Num of Params = '+str(num_parms))

    print('Compression Ratio is {:4f}'.format(opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms))


    for (iteration, batch) in enumerate(dataloader):
        #if info['Time'] <= opt.model.time:

        start_time = time.time()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = x.squeeze(0)
        y = y.squeeze(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            model_output = model(x)

        loss = F.mse_loss(model_output, y, reduction='none')
        loss.mean().backward()                   

        optimizer.step()
        scheduler.step() 

        info['Loss'].append(loss.mean().item())

        end_time = time.time()
        sec_passed = end_time-start_time
        info['Time'] += sec_passed

        if (iteration+1) % 500 == 0:

            save_yaml(opt.model.result_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'-loss-{:04d}.yaml'.format(t),info)
            #torch.save(model.state_dict(),opt.model.model_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'_best_model_{:04d}.pth'.format(t))
        #else:
            #break
        if loss_ > info['Loss'][-1]:
            loss_ = info['Loss'][-1]
            torch.save(model.state_dict(),opt.model.model_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'_best_model_{:04d}.pth'.format(t))

    save_yaml(opt.model.result_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'-loss-{:04d}.yaml'.format(t),info)
    #torch.save(model.state_dict(),opt.model.model_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'_best_model_{:04d}.pth'.format(t))

def train_NGP(model,dataset,opt,device, t):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    model.train(True)

    loss_ = 1e5

    num_parms = count_parameters(model)
    info = {'Loss':[],'Time':0,'Params':num_parms}
    optimizer = optim.Adam(model.parameters(), lr=opt.model.lr,betas=[0.9,0.999]) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[opt.model.iterations*(2/5), opt.model.iterations*(3/5), opt.model.iterations*(4/5)],gamma=0.33)
    early_stopping_data = (False,torch.zeros([opt.model.iterations], dtype=torch.float32, device=device))

    print('Num of Params = '+str(num_parms))

    print('Compression Ratio is {:4f}'.format(opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms))


    for (iteration, batch) in enumerate(dataloader):
        if info['Time'] <= opt.model.time:

            start_time = time.time()

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            x = x.squeeze(0)
            y = y.squeeze(0)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                model_output = model(x)

            loss = F.mse_loss(model_output, y, reduction='none')
            loss.mean().backward()                   

            optimizer.step()
            scheduler.step() 

            info['Loss'].append(loss.mean().item())

            end_time = time.time()
            sec_passed = end_time-start_time
            info['Time'] += sec_passed

            if (iteration+1) % 500 == 0:

                save_yaml(opt.model.result_path+str(opt.model.hash_log2_size)+'/'+opt.model.model+'-loss-{:04d}.yaml'.format(t),info)
            
            if loss_ > info['Loss'][-1]:
                loss_ = info['Loss'][-1]
                torch.save(model.state_dict(),opt.model.model_path+str(opt.model.hash_log2_size)+'/'+opt.model.model+'_best_model_{:04d}.pth'.format(t))
        else:
            break
    save_yaml(opt.model.result_path+str(opt.model.hash_log2_size)+'/'+opt.model.model+'-loss-{:04d}.yaml'.format(t),info)

def train_step_APMGSRN(opt, device, iteration, batch, dataset, model, optimizer, scheduler, info, 
                      early_stop_reconstruction,early_stop_grid,early_stopping_reconstruction_losses,early_stopping_grid_losses):

    if(early_stop_reconstruction and early_stop_grid):
        return early_stop_reconstruction, early_stop_grid, early_stopping_reconstruction_losses, early_stopping_grid_losses, info

    optimizer[0].zero_grad()                  

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    x = x.squeeze(0)
    y = y.squeeze(0)
    
    transformed_x = model.transform(x)    
    model_output = model.forward_pre_transformed(transformed_x)
    
    loss = F.mse_loss(model_output, y, reduction='none')
    loss = loss.sum(dim=1, keepdim=True)
    
    loss.mean().backward()
    early_stopping_reconstruction_losses[iteration] = loss.mean().detach()
    early_stop_reconstruction = optimizer[0].param_groups[0]['lr'] < opt.model.lr * 1e-2

    #print('Loss is {:04f}'.format(loss.mean().item()))
    info['Loss'].append(loss.mean().item())

    if(iteration > 500 and  # let the network learn a bit first
        iteration < opt.model.iterations*0.8 and  # stop the grid moving to adequately learn at the end
        not early_stop_grid):
        optimizer[1].zero_grad() 
        
        density = model.feature_density_pre_transformed(transformed_x) 
        
        density /= density.sum().detach()
        target = torch.exp(torch.log(density+1e-16) * \
            (loss.mean()/(loss+1e-16)))
        target /= target.sum()
        
        density_loss = F.kl_div(
           torch.log(density+1e-16), 
           torch.log(target.detach()+1e-16), reduction='none', 
            log_target=True)
        
        density_loss.mean().backward()

        info['Density'].append(density_loss.mean().item())
        
        optimizer[1].step()
        scheduler[1].step()   

        early_stopping_grid_losses[iteration] = density_loss.mean().detach()
        if(iteration >= 2500):
            prev_avg = early_stopping_grid_losses[iteration-2000:iteration-1000].mean()
            current_avg = early_stopping_grid_losses[iteration-1000:iteration].mean()
            
            thresh = prev_avg * 1e-4
            momentum_needed = 1
            
            # See if the slope is under the threshold
            thresh_met = prev_avg - current_avg < thresh
            
            # a let the momentum of the grids finish for 1k more iterations
            if(thresh_met):
                early_stopping_grid_losses[-1] += 1
            else:
                early_stopping_grid_losses[-1] = 0
                
            early_stop_grid = thresh_met and early_stopping_grid_losses[-1] > momentum_needed 
            if(early_stop_grid):
                print(f"Grid has converged. Setting early stopping flag.")
    
    optimizer[0].step()

    if(early_stop_grid):
        scheduler[0].step(early_stopping_reconstruction_losses[iteration-1000:iteration].mean())   
    
    return early_stop_reconstruction, early_stop_grid, early_stopping_reconstruction_losses, early_stopping_grid_losses, info

def train_APMGSRN(model,dataset,opt,device, t):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    model.train(True)

    num_parms = count_parameters(model)
    info = {'Loss':[],'Density':[],'Time':0,'Params':num_parms}
    optimizer = [optim.Adam(model.get_model_parameters(), lr=opt.model.lr,betas=[0.9,0.999], eps = 10e-15),
                 optim.Adam(model.get_transform_parameters(), lr=opt.model.lr * 0.05, betas=[0.9,0.999], eps = 10e-15)]        
    scheduler = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0],mode="min", patience=500, threshold=1e-4, threshold_mode="rel",cooldown=250,factor=0.1,verbose=True),
                 torch.optim.lr_scheduler.LinearLR(optimizer[1], start_factor=1, end_factor=0.5)]      
    early_stopping_data = (False, False,torch.zeros([opt.model.iterations], dtype=torch.float32, device=device),
                                        torch.zeros([opt.model.iterations], dtype=torch.float32, device=device))

    print('Num of Params = '+str(num_parms))

    print('Compression Ratio is {:4f}'.format(opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms))

    early_stop_reconstruction = False
    early_stop_grid = False
    early_stopping_reconstruction_losses = torch.zeros([opt.model.iterations], dtype=torch.float32, device=device)
    early_stopping_grid_losses = torch.zeros([opt.model.iterations], dtype=torch.float32, device=device)

    for (iteration, batch) in enumerate(dataloader):
        #if info['Time'] <= opt.model.time:
        start_time = time.time()
        early_stop_reconstruction,early_stop_grid,early_stopping_reconstruction_losses,early_stopping_grid_losses,info = train_step_APMGSRN(opt,device,iteration,batch,dataset,model,optimizer,scheduler,info,early_stop_reconstruction,early_stop_grid,early_stopping_reconstruction_losses,early_stopping_grid_losses)
        end_time = time.time()
        sec_passed = end_time-start_time
        info['Time'] += sec_passed

        if (iteration+1) % 500 == 0:
            save_yaml(opt.model.result_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'-loss-{:04d}.yaml'.format(t),info)
            torch.save(model.state_dict(),opt.model.model_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'_best_model_{:04d}.pth'.format(t))
        '''
        else:
            break
        '''

    save_yaml(opt.model.result_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'-loss-{:04d}.yaml'.format(t),info)
    torch.save(model.state_dict(),opt.model.model_path+str(opt.model.nodes_per_layer)+'/'+opt.model.model+'_best_model_{:04d}.pth'.format(t))






