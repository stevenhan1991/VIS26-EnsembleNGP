import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.data import DataLoader, Dataset
from math import e
from utils import *
import time as clock
import torch.nn.functional as F
from tqdm import tqdm
from ngp import *
from itertools import product
import csv
import gc

class EnsembleTData(Dataset):
    def __init__(self, cfg, device):
        
        self.cfg = cfg

        self.path = self.cfg.data.path

        self.res = self.cfg.data.res

        self.device = device

        self.data = []

        self.ensemble_parms = []

        self.factor = self.cfg.data.factor

        if self.cfg.data.dataset == 'Nyx':
            self.sampling_ratio = 0.2
            with open('../Params/nyx.csv', newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    OmB = float(row[0])
                    OmM = float(row[1])
                    h = float(row[2])
                    OmB = 2.0 * (OmB - 0.0215) / (0.0235 - 0.0215) - 1.0
                    OmM = 2.0 * (OmM - 0.12000) / (0.1550 - 0.1200) - 1.0
                    h = 2.0 * (h - 0.5500) / (0.8500 - 0.5500) - 1.0
                    self.ensemble_parms.append(torch.FloatTensor([OmB,OmM,h]).float())
    
        elif self.cfg.data.dataset == 'Castro':
            self.sampling_ratio = 0.2
            with open('../Params/castro.csv', newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    p1 = float(row[0])
                    p2 = float(row[1])
                    p1 = 2.0 * (p1 - 0.8) / (0.95 - 0.8) - 1.0
                    p2 = 2.0 * (p2 - 0.8) / (0.95 - 0.8) - 1.0
                    self.ensemble_parms.append(torch.FloatTensor([p1,p2]).float())
            
        elif self.cfg.data.dataset == 'Colver':
            self.sampling_ratio = 0.12
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
                    self.ensemble_parms.append(torch.FloatTensor([d1, d2, d3, e1, e2, e3]).float())
        elif self.cfg.data.dataset == 'MPAS-Ocean':
            self.sampling_ratio = 0.15
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

                    self.ensemble_parms.append(torch.FloatTensor([BwsA, CbrN, GM, HV]).float())
        
        
        self.selected_idx = [int(i) for i in np.arange(0, self.cfg.data.num_ensemble_training)] 
        
        self.ensemble_parms = torch.stack(self.ensemble_parms).to(self.device)

        self.time_steps = self.cfg.data.time_steps
        
        print('Total Number of Training Ensembles are {:d}'.format(len(self.selected_idx)))
        print(self.selected_idx)
        print('Total Training Samples are {:d}'.format(int(self.factor*len(self.selected_idx)*(int(self.sampling_ratio*self.time_steps)))))

        if self.cfg.data.dataset in ['Nyx','Castro','Colver']:
            self.coords = get_mgrid(self.res, dim=3)
            self.coords = torch.from_numpy(self.coords).float().to(self.device)
            self.samples = np.prod(self.res)
        else:
            self.coords = np.load('/root/autodl-tmp/Data/{:s}/coord.npy'.format(self.cfg.data.dataset))

            self.coords = self.coords.reshape(-1,3)
            self.coords[:,0] = 2.0 * (self.coords[:,0] - self.coords[:,0].min()) / (self.coords[:,0].max() - self.coords[:,0].min()) - 1.0
            self.coords[:,1] = 2.0 * (self.coords[:,1] - self.coords[:,1].min()) / (self.coords[:,1].max() - self.coords[:,1].min()) - 1.0
            self.coords[:,2] = 2.0 * (self.coords[:,2] - self.coords[:,2].min()) / (self.coords[:,2].max() - self.coords[:,2].min()) - 1.0


            self.coords = torch.from_numpy(self.coords).float()
            self.samples = 11845146
            d = np.fromfile('/root/autodl-tmp/Data/{:s}/{:04d}/{:04d}.dat'.format(self.cfg.data.dataset, 0, 0),dtype='<f')
            self.index = np.where(d!=-1e34)
            self.coords = self.coords[self.index].to(self.device)

        self.batch_size = self.cfg.data.batch_size

        self.samples_per_vol = 2**15

        self.probe = 0

        self.num = int(self.factor*len(self.selected_idx)*(int(self.sampling_ratio*self.time_steps)*self.samples_per_vol))//self.batch_size
        print(self.num)

    def get_random_points(self):
        if self.probe == 0:

            if hasattr(self, "selected_coords"):
                del self.selected_coords
            if hasattr(self, "vol"):
                del self.vol

            torch.cuda.empty_cache()

            selected_coords_ = []
            vol_ = []
            for i in range(0,len(self.selected_idx)):
                selected_t = list(torch.randperm(self.time_steps-2)[:int(self.sampling_ratio*self.time_steps)-2].numpy() + [1]) + [0] + [self.time_steps-1]
                for t in selected_t:
                    idx = self.selected_idx[i]
                    d = np.memmap('/root/autodl-tmp/Data/{:s}/{:04d}/{:04d}.dat'.format(self.cfg.data.dataset, idx, t),dtype='<f', mode='r')
                    if self.cfg.data.dataset in ['MPAS-Ocean']:
                        d = d[self.index]
                    d = 2*(d-d.min())/(d.max()-d.min())-1

                    vol = torch.from_numpy(d).float().to(self.device)

                    idx = torch.randperm(self.samples, device=self.device)[:int(self.factor*self.samples_per_vol)]

                    coords = self.coords[idx,...]
                    vol = vol[idx]

                    ensemble_param = self.ensemble_parms[i][None,...].float()

                    time_chunks = torch.zeros((ensemble_param.size(0),1),device=self.device)
                    time = t/(self.time_steps-1)
                    time -= 0.5
                    time *= 2.0
                    time_chunks.fill_(time)

                    params = torch.cat((ensemble_param,time_chunks),dim=-1)

                    params = params.expand(coords.size(0),-1)

                    selected_coords = torch.cat((coords, params),dim=-1)

                    selected_coords_.append(selected_coords)

                    vol_.append(vol)

                    del selected_coords, vol, idx

                torch.cuda.synchronize()

            self.selected_coords = torch.cat(selected_coords_,dim=0)
            self.vol = torch.cat(vol_,dim=0)

            idx = torch.randperm(self.vol.size(0), device=self.device)
            self.selected_coords = self.selected_coords[idx]
            self.vol = self.vol[idx]

            del selected_coords_, vol_, idx
            torch.cuda.empty_cache()

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.get_random_points()

        selected_coords = self.selected_coords[idx*self.batch_size:(idx+1)*self.batch_size:,]
        vol = self.vol[idx*self.batch_size:(idx+1)*self.batch_size:,]
        self.probe += 1
        if self.probe == self.num:
            self.probe = 0
        return {'coords': selected_coords,'values':vol}

class EnsembleT(Dataset):
    def __init__(self, cfg, device):
        
        self.cfg = cfg

        self.path = cfg.data.path

        self.res = cfg.data.res

        self.device = device

        self.time_steps = self.cfg.data.time_steps

        self.ensemble_parms = []

        self.sampling_ratio = 0.1

        if self.cfg.data.dataset == 'Nyx':
            with open('../Params/nyx.csv', newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    OmB = float(row[0])
                    OmM = float(row[1])
                    h = float(row[2])
                    OmB = 2.0 * (OmB - 0.0215) / (0.0235 - 0.0215) - 1.0
                    OmM = 2.0 * (OmM - 0.12000) / (0.1550 - 0.1200) - 1.0
                    h = 2.0 * (h - 0.5500) / (0.8500 - 0.5500) - 1.0
                    self.ensemble_parms.append(torch.FloatTensor([OmB,OmM,h]).float())
    
        elif self.cfg.data.dataset == 'Castro':
            with open('../Params/castro.csv', newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    p1 = float(row[0])
                    p2 = float(row[1])
                    p1 = 2.0 * (p1 - 0.8) / (0.95 - 0.8) - 1.0
                    p2 = 2.0 * (p2 - 0.8) / (0.95 - 0.8) - 1.0
                    self.ensemble_parms.append(torch.FloatTensor([p1,p2]).float())

        elif self.cfg.data.dataset == 'Colver':
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

                    self.ensemble_parms.append(torch.FloatTensor([d1, d2, d3, e1, e2, e3]).float())
        elif self.cfg.data.dataset == 'MPAS-Ocean':
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

                    self.ensemble_parms.append(torch.FloatTensor([BwsA, CbrN, GM, HV]).float())
        
        self.selected_idx = [int(i) for i in np.arange(0, self.cfg.data.num_ensemble_training)]
        
        self.ensemble_parms = torch.stack(self.ensemble_parms)
        
        print('Total Training Ensembles are {:d}'.format(len(self.selected_idx)))
        print(self.ensemble_parms.shape)

        self.probe = 0

    def get_samples(self):
        if self.probe == 0:
            self.data = []
            self.params = []
            for i in tqdm(range(0, len(self.selected_idx))):
                selected_t = list(torch.randperm(self.time_steps-2)[:int(self.sampling_ratio*self.time_steps)-2].numpy() + [1]) + [0] + [self.time_steps-1]
                for t in selected_t:
                    idx = self.selected_idx[i]
                    if self.cfg.data.dataset == 'Nyx':
                        d = np.memmap('/root/autodl-tmp/Data/{:s}/{:04d}/{:04d}.dat'.format(self.cfg.data.dataset, idx, t),dtype='<f', mode='r')
                        d = d.reshape((self.res[2], self.res[1], self.res[0])).transpose()
                    elif self.cfg.data.dataset in ['Castro','Colver']:
                        d = np.memmap('/root/autodl-tmp/Data/{:s}_reduced/{:04d}/{:04d}.dat'.format(self.cfg.data.dataset, idx, t),dtype='<f', mode='r')
                        d = d.reshape((128,128,128)).transpose()
                    elif self.cfg.data.dataset in ['MPAS-Ocean']:
                        d = np.fromfile('/root/autodl-tmp/Data/{:s}_reduced/{:04d}/{:04d}.dat'.format(self.cfg.data.dataset, idx, t),dtype='<f')
                        d = d.reshape((64,128,256)).transpose()
                        d[d==-10000] = -3
                    d = 2*(d-d.min())/(d.max()-d.min())-1
                    d = torch.from_numpy(d).float()[None,...]
                    self.data.append(d)

                    ensemble_param = self.ensemble_parms[i][None,...]

                    time_chunks = torch.zeros((ensemble_param.size(0),1))
                    time = t/(self.time_steps-1)
                    time -= 0.5
                    time *= 2.0
                    time_chunks.fill_(time)

                    params = torch.cat((ensemble_param,time_chunks),dim=-1)


                    self.params.append(params)

    def __len__(self):
        return len(self.selected_idx)*(int(self.sampling_ratio*self.time_steps))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.get_samples()
        params = self.params[idx].float()
        vol = self.data[idx]

        self.probe += 1
        if self.probe == len(self.selected_idx)*(int(self.sampling_ratio*self.time_steps)):
            self.probe = 0

        return {'params': params,'vols':vol}


