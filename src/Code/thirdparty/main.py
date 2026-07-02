from dataio import *
import sys
import os
from torch.utils.data import DataLoader
import torch
import random
import copy
from utils import *
from train import *
from omegaconf import OmegaConf
import time as clock
import yaml
from model import * 
from copy import deepcopy
from skimage.transform import rescale, resize


p = argparse.ArgumentParser()
p.add_argument('--config_file', type=str, default='Nyx.yaml')
p.add_argument('--device', type=int, default=0)
p.add_argument('--ch', type=int, default=32)
p.add_argument('--ratio', type=float, default=0.5)
p.add_argument('--mode', type=str, default='decoder')
p.add_argument('--model', type=str, default='NGP')
p.add_argument('--log2_hashmap_size',type=int,default=18)

opt = p.parse_args()

def getPath(dataset, model):
  path = None
  res = None
  if dataset == 'Nyx':
    path = '/root/autodl-tmp/Data/Nyx/'
    res = [128,128,128]
    if model in ['NGP', "NGPM"]:
      batch_size = 2**17
    else:
      batch_size = 64000
  elif dataset == 'Castro':
    path = '/root/autodl-tmp/Data/Castro/'
    res = [256,256,256]
    if model in ['NGP', "NGPM"]:
      batch_size = 2**17
    else:
      batch_size = 32000
  elif dataset == 'Colver':
    path = '/root/autodl-tmp/Data/ColverLeaf/'
    res = [192,192,192]
    if model in ['NGP', "NGPM"]:
      batch_size = 2**17
    else:
      batch_size = 64000
  else:
    raise NotImplementedError('Not Implemented for the '+str(dataset)+' Data Set!')

  return path, res, batch_size


def main():

  with open(os.path.join("configs", opt.config_file), "r") as f:
    config = load_config(f,True)

  if not os.path.exists(config.model.model_path):
    os.mkdir(config.model.model_path)

  if not os.path.exists(config.model.result_path):
    os.mkdir(config.model.result_path)


  path, res, batch_size = getPath(config.data.dataset, opt.model)

  config.data.res = res
  config.data.path = path
  config.data.batch_size = batch_size

  config.data.sample_interval = 1/opt.ratio

  config.device = opt.device

  config.model.ngp.log2_hashmap_size = opt.log2_hashmap_size

  config.model.model = opt.model
  config.model.ensemblenet.ch = opt.ch
  config.model.ngp.ensemble_embed_dim = opt.ch

  torch.cuda.set_device(config.device)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if opt.mode == 'train':
    if config.GPU_Sampling:
      sampling_device = 'cuda'
    else:
      sampling_device = 'cpu'
    if opt.model in ['NGP','NGPM']:
      vol = EnsembleData(config, sampling_device)
      train_ngp(config,vol, device)
    elif opt.model in ['CoodNet','fVSRN']:
      vol = EnsembleData(config, sampling_device)
      train(config,vol, device)
    elif opt.model == 'ExploreableINR':
      train_explorINR(config, device)

  elif opt.mode == 'decoder':
    vol = Ensemble(config, device)
    train_decoder(config, vol, device)
  elif opt.mode == 'count':
    if opt.model == 'ExploreableINR':
      get_3d_dim()
  elif opt.mode == 'decode-inf':
    ensemble_parms = []
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

    model = EnsembleNet(**config.model.ensemblenet).to(device)
    model.load_state_dict(torch.load(config.model.model_path+'ensemble_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.data.sample_interval),map_location=device))

    for idx in range(0,config.data.num_ensemble):
      if idx not in selected_idx:
        d = np.fromfile('/root/autodl-tmp/Data/{:s}/{:04d}.dat'.format(config.data.dataset, idx+1),dtype='<f')
        d = d.reshape((config.data.res[2], config.data.res[1], config.data.res[0])).transpose()
        if config.data.dataset != 'Nyx':
          d = resize(d, (128,128,128), order=3)

        d = (d-d.min())/(d.max()-d.min())

        e = ensemble_parms[idx].float()[None,...].to(device)
        pred = model(e)[0][0].detach().cpu().numpy()
        pred = np.clip(pred, 0, 1)

        pred = 2.0 * pred - 1.0
        d = 2.0 * d - 1.0

        p = PSNR(d, pred)

        print('PSNR is {:f} under Ensemble ID {:d}'.format(p, idx))
  elif opt.mode == 'inf':
    if opt.model == 'NGP':
      model = NGP(**config.model.ngp).to(device)
      model.load_state_dict(torch.load(config.model.model_path+'{:d}_{:s}_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size, config.data.sample_interval),map_location=device))
    elif opt.model == 'NGPM':
      model = NGPM(**config.model.ngp).to(device)
      model.load_state_dict(torch.load(config.model.model_path+'{:d}_{:s}_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size, config.data.sample_interval),map_location=device))

      model.half()
      model.float()
    elif opt.model == 'ExploreableINR':
      out_features = 1
      nEnsemble = 300
      data_size = 256*256*256

      sf_sr = 0.05
      batch_size = 1
      sp_sr = 0.3
      dim3d = 64
      dim2d = 256
      dim1d = 16
      spatial_fdim = 64
      param_fdim = 16

      num_sf_batches = math.ceil(nEnsemble * data_size * sf_sr / batch_size)
      num_sp_sampling = math.ceil(70 * sp_sr)
      network_str = 'mpaso_' + str(dim3d) + '_' + str(dim2d) + '_' + str(dim1d) + '_' + str(spatial_fdim) + '_' + str(param_fdim)

      #####################################################################################

      feature_grid_shape = np.concatenate((np.ones(3, dtype=np.int32)*dim3d, np.ones(3, dtype=np.int32)*dim2d, np.ones(3, dtype=np.int32)*dim1d))

      model = INR_FG(feature_grid_shape, spatial_fdim, spatial_fdim, param_fdim, out_features).to(device)
      model.load_state_dict(torch.load(config.model.model_path+'ensemble_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.data.sample_interval),map_location=device))

    coords = get_mgrid(config.data.res, dim=3)
    coords = torch.from_numpy(coords).float()

    ensemble_parms = []

    info = {'PSNR':[],'Avg':0}

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


    decoder = EnsembleNet(**config.model.ensemblenet).to(device)
    decoder.load_state_dict(torch.load(config.model.model_path+'ensemble_{:d}_{:f}_best_model.pth'.format(config.model.ensemblenet.ch, config.data.sample_interval),map_location=device))

    decoder.half()
    decoder.float()
    
    num = 0
    for e in range(0,config.data.num_ensemble):
      if e not in selected_idx:

        ensemble_param_ = ensemble_parms[e][None,...].float()

        ensemble_param_ = ensemble_param_.expand(coords.size(0),-1)

        xyze_coords = torch.cat((coords, ensemble_param_),dim=-1)

        xyze_coords = (xyze_coords + 1.0) / 2.0

        train_dl = DataLoader(xyze_coords,batch_size=xyze_coords.size(0)//4,shuffle=False, num_workers=0, pin_memory=True)

        v = []

        for batch_idx, coords_ in enumerate(train_dl):
          coords_ = coords_.cuda()
          with torch.no_grad():
            ensemble_features = decoder.get_ensmeble_embedding(coords_[:,3:])
            ensemble_features = (ensemble_features + 1.0) / 2.0
            v_pred = model(coords_[:,0:3], ensemble_features).squeeze().detach().cpu().numpy()
            v += list(v_pred)

        v = np.asarray(v, dtype='<f')
        v = np.clip(v, 0.0, 1.0)
        v = 2.0 * v - 1.0

        gt = np.fromfile('/root/autodl-tmp/Data/{:s}/{:04d}.dat'.format(config.data.dataset, e+1),dtype='<f')

        gt = 2.0 * (gt - gt.min()) / (gt.max() - gt.min()) - 1.0

        p = PSNR(gt, v)

        info['PSNR'].append('PSNR is {:f} under Ensemble ID {:04d}'.format(p, e))

        num += 1

        info['Avg'] += float(p)
        print('PSNR is {:f} under Ensemble ID {:04d}'.format(p, e))
        save_yaml(config.model.result_path+'{:d}_{:s}_{:d}_{:f}_PSNR.yaml'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size, config.data.sample_interval),info)

    info['Avg'] = info['Avg'] / num
    save_yaml(config.model.result_path+'{:d}_{:s}_{:d}_{:f}_PSNR.yaml'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size, config.data.sample_interval),info)


if __name__== "__main__":
  main()
