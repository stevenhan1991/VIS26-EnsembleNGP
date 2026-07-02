from dataio import *
import sys
import os
from torch.utils.data import DataLoader
import torch
import random
import copy
import time as timer
from utils import *
from train import *
from omegaconf import OmegaConf
import yaml
from model import * 
from drr import *
from copy import deepcopy
from skimage.transform import rescale, resize
from tqdm import tqdm


p = argparse.ArgumentParser()
p.add_argument('--config_file', type=str, default='Nyx.yaml')
p.add_argument('--device', type=int, default=0)
p.add_argument('--ch', type=int, default=32)
p.add_argument('--mode', type=str, default='train')
p.add_argument('--idx', type=int, default=1)
p.add_argument('--log2_hashmap_size',type=int,default=18)


opt = p.parse_args()

def getPath(dataset, model):
  path = None
  res = None
  if dataset == 'Nyx':
    path = '/root/autodl-tmp/Data/Nyx/'
    res = [128,128,128]
    batch_size = 2**16
  elif dataset == 'Castro':
    path = '/root/autodl-tmp/Data/Castro/'
    res = [256,256,256]
    batch_size = 2**16
  elif dataset == 'Colver':
    path = '/root/autodl-tmp/Data/ColverLeaf/'
    res = [192,192,192]
    batch_size = 2**16
  elif dataset == 'MPAS-Ocean':
    path = '/root/autodl-tmp/Data/MPAS-Ocean/'
    res = [256,128,64]
    batch_size = 2**16
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

  config.device = opt.device

  config.model.ngp.log2_hashmap_size = opt.log2_hashmap_size

  config.model.model = opt.model
  config.model.ensemblenet.ch = opt.ch
  config.model.ensemblenet.dataset = config.data.dataset
  config.model.ngp.ensemble_embed_dim = opt.ch

  torch.cuda.set_device(config.device)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if opt.mode == 'train':
    if not os.path.exists(config.model.model_path+'embedder_{:d}_best_model.pth'.format(config.model.ensemblenet.ch)):
      vol = EnsembleT(config, device)
      train_decoder(config, vol, device)
      extract_embedding(config,device)
    vol = EnsembleTData(config, device)
    train_ngp(config,vol, device)
  elif opt.mode == 'inf-decoder':
    extract_embedding(config, device)
  elif opt.mode == 'inf':
    os.makedirs(config.model.result_path+'{:d}/'.format(config.model.ensemblenet.ch,), exist_ok=True)
    os.makedirs(config.model.result_path+'{:d}/{:d}/'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size), exist_ok=True)
    os.makedirs(config.model.result_path+'{:d}/{:d}/{:s}/'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model), exist_ok=True)

    model = NGPM(**config.model.ngp).to(device)
    model.load_state_dict(torch.load(config.model.model_path+'{:d}_{:s}_{:d}_best_model.pth'.format(config.model.ensemblenet.ch, config.model.model, config.model.ngp.log2_hashmap_size),map_location=device))
    model.float()

    if config.data.dataset in ['MPAS-Ocean']:
      coords = np.load('/root/autodl-tmp/Data/{:s}/coord.npy'.format(config.data.dataset))
      coords = coords.reshape(-1,3)
      num_points = coords.shape[0]
      coords[:,0] = 2.0 * (coords[:,0] - coords[:,0].min()) / (coords[:,0].max() - coords[:,0].min()) - 1.0
      coords[:,1] = 2.0 * (coords[:,1] - coords[:,1].min()) / (coords[:,1].max() - coords[:,1].min()) - 1.0
      coords[:,2] = 2.0 * (coords[:,2] - coords[:,2].min()) / (coords[:,2].max() - coords[:,2].min()) - 1.0
      d = np.fromfile('/root/autodl-tmp/Data/{:s}/{:04d}/{:04d}.dat'.format(config.data.dataset, 0, 0),dtype='<f')
      index_ocean = np.where(d!=-1e34)
      index_land = np.where(d==-1e34)
      coords = coords[index_ocean]
    else:
      coords = get_mgrid(config.data.res, dim=3)
    
    coords = torch.from_numpy(coords).float()

    ensemble_parms = []

    info = {'Model Size':0, 'PSNR':[],'Avg':0, 'Time':0}

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
        
    ensemble_parms = torch.stack(ensemble_parms)

    embeder = ParamerEmedding(config.model.ensemblenet.in_dim, config.model.ensemblenet.ch).to(device)
    embeder.load_state_dict(torch.load(config.model.model_path+'embedder_{:d}_best_model.pth'.format(config.model.ensemblenet.ch),map_location=device))

    embeder.float()

    selected_idx = [int(i) for i in np.arange(0, config.data.num_ensemble_training)]

    num_parms = count_parameters(model) + count_parameters(embeder)

    val = np.fromfile(config.model.result_path+'{:d}.dat'.format(config.model.ensemblenet.ch),dtype='<f')
    val = torch.from_numpy(val).to(device)

    info['Model Size'] = float(num_parms * 2 /1024 / 1024)
    
    num = 0

    max_points = 2**22

    for e in range(len(selected_idx),len(ensemble_parms)):
      if e not in selected_idx:
        if config.data.dataset in ['MPAS-Ocean']:
          d = np.zeros((num_points))
          d.fill(-1e34)
        x = timer.time()
        os.makedirs(config.model.result_path+'{:d}/{:d}/{:s}/{:04d}'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model, e), exist_ok=True)
        for t in range(0,config.data.time_steps):

          ensemble_param = ensemble_parms[e][None,...]

          time_chunks = torch.zeros((ensemble_param.size(0),1))
          time = t/(config.data.time_steps-1)
          time -= 0.5
          time *= 2.0
          time_chunks.fill_(time)

          params = torch.cat((ensemble_param,time_chunks),dim=-1).float()

          params = params.expand(coords.size(0),-1)

          xyzet_coords = torch.cat((coords, params),dim=-1)

          xyzet_coords = (xyzet_coords + 1.0 ) / 2.0

          if os.path.exists(config.model.result_path+'{:d}/{:d}/{:s}/{:04d}/{:04d}.dat'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model, e, t)):
            v = np.fromfile(config.model.result_path+'{:d}/{:d}/{:s}/{:04d}/{:04d}.dat'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model, e, t),dtype='<f')
          else:

            x = timer.time()

            v = []

            for start in range(0, xyzet_coords.shape[0], max_points):
              with torch.no_grad():
                ensemble_features = embeder(xyzet_coords[start:min(start+max_points, coords.shape[0]),3:].to(device))
                if val > 1:
                  ensemble_features /= val

                ensemble_features = (ensemble_features + 1.0) / 2.0

                v_pred = model(xyzet_coords[start:min(start+max_points, coords.shape[0]),0:3].to(device), ensemble_features).squeeze().detach().cpu().numpy()
                v += list(v_pred)

            v = np.asarray(v, dtype='<f')
            v = np.clip(v, 0.0, 1.0)
            v = 2.0 * v - 1.0

            if config.data.dataset in ['MPAS-Ocean']:
              d[index_ocean] = v
              v = d
              v = np.asarray(v, dtype='<f')

            y = timer.time()

            info['Time'] += y-x

            v.tofile(config.model.result_path+'{:d}/{:d}/{:s}/{:04d}/{:04d}.dat'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model, e, t), format='<f')

          gt = np.fromfile('/root/autodl-tmp/Data/{:s}/{:04d}/{:04d}.dat'.format(config.data.dataset, e, t),dtype='<f')

          if config.data.dataset in ['MPAS-Ocean']:
            min_v = gt[index_ocean].min()
            max_v = gt[index_ocean].max()

            gt = 2.0 * (gt - min_v) / (max_v - min_v) - 1.0

          else:
            gt = 2.0 * (gt - gt.min()) / (gt.max() - gt.min()) - 1.0

          if config.data.dataset in ['MPAS-Ocean']:
            p = PSNR(gt[index_ocean],v[index_ocean])
          else:
            p = PSNR(gt, v)

          info['PSNR'].append('PSNR is {:f} under Ensemble ID {:04d} at Time Step {:04d}'.format(p, e, t))

          num += 1

          info['Avg'] += float(p)
          print('PSNR is {:f} under Ensemble ID {:04d} at Time Step {:04d}'.format(p, e, t))
          save_yaml(config.model.result_path+'{:d}/{:d}/{:s}/PSNR.yaml'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model),info)

    info['Avg'] = info['Avg'] / num
    info['Time'] = info['Time'] / num
    save_yaml(config.model.result_path+'{:d}/{:d}/{:s}/PSNR.yaml'.format(config.model.ensemblenet.ch, config.model.ngp.log2_hashmap_size, opt.model),info)
  
if __name__== "__main__":
  main()