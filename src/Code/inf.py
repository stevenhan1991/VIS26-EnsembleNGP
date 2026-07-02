from model import *
from utils import *
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from os.path import exists
from sklearn.cluster import DBSCAN
from copy import deepcopy
from model import *
from srn import *
import gc

def model_reconstruction_chunked_single(opt,device,model_type,hidden_features,round_):
	
	chunk_size = 512
	full_shape = [opt.data.res[2],opt.data.res[1],opt.data.res[0]]
	
	psnr_info = {'PSNR':[], 'Avg':0, 'CR':0}
	with torch.no_grad():
		opt.model.nodes_per_layer = hidden_features
		for t in range(opt.data.total_steps,opt.data.total_steps+1):
			output = torch.empty(full_shape, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			
			gt = np.fromfile(opt.data.path+'{:04d}'.format(t)+'.dat',dtype='<f')
			gt = (gt-gt.min())/(gt.max()-gt.min())
			gt -= 0.5
			gt *= 2.0
			if model_type == 'fVSRN':
				model = fVSRN(opt).to(device)
			elif model_type == 'APMGSRN':
				model = APMGSRN(opt).to(device)
			elif model_type == 'RMDSRN':
				model = RMDSRN(opt).to(device)

			model.load_state_dict(torch.load(opt.model.model_path+'{:s}-{:d}-best-model-{:d}-{:d}.pth'.format(model_type,opt.model.nodes_per_layer,t,round_),map_location=device))
			if t == 1:
				num_parms = count_parameters(model)
				psnr_info['CR'] = opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms

			for z_ind in range(0, full_shape[0], chunk_size):
				z_ind_end = min(full_shape[0], z_ind+chunk_size)
				z_range = z_ind_end-z_ind
				for y_ind in range(0, full_shape[1], chunk_size):
					y_ind_end = min(full_shape[1], y_ind+chunk_size)
					y_range = y_ind_end-y_ind            
					for x_ind in range(0, full_shape[2], chunk_size):
						x_ind_end = min(full_shape[2], x_ind+chunk_size)
						x_range = x_ind_end-x_ind
						
						opt['extents'] = f"{z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}"
																	
						grid = [z_range, y_range, x_range]
						coord_grid = make_coord_grid(grid, 'cpu', flatten=True,align_corners=True,use_half=False)
						
						coord_grid += 1.0
						coord_grid /= 2.0
						
						coord_grid[:,0] *= (x_range-1) / (full_shape[2]-1)
						coord_grid[:,1] *= (y_range-1) / (full_shape[1]-1)
						coord_grid[:,2] *= (z_range-1) / (full_shape[0]-1)
						
						coord_grid[:,0] += x_ind / (full_shape[2]-1)
						coord_grid[:,1] += y_ind / (full_shape[1]-1)
						coord_grid[:,2] += z_ind / (full_shape[0]-1)
						
						coord_grid *= 2.0
						coord_grid -= 1.0
						
						out_tmp = forward_maxpoints(model, 
													coord_grid, max_points=2**20, 
													data_device='cpu',
													device=device)
						out_tmp = out_tmp.permute(1,0)
						out_tmp = out_tmp.view([out_tmp.shape[0]] + grid)
						output[0,:,z_ind:z_ind_end,y_ind:y_ind_end,x_ind:x_ind_end] = out_tmp

						
			output = output.cpu().detach().numpy()
			output = output[0][0]
			output = output.transpose()
			output = np.asarray(output,dtype='<f')
			output = output.flatten('F')
			output = np.clip(output,-1.0,1.0)
			output.tofile(opt.model.result_path+'/{:s}/{:d}/{:04d}'.format(model_type,hidden_features,round_)+'.dat',format='<f')
			print(output.min())
			print(output.max())
			'''
			p = PSNR(gt,output)
			psnr_info['PSNR'].append(float(p))
			psnr_info['Avg'] += float(p)
			'''
			#print('[Time Step {:02d}] PSNR : {:.4f}'.format(t, p))
			'''
			save_yaml(opt.model.result_path+'/{:s}/{:d}/PSNR.yaml'.format(model_type,hidden_features), psnr_info)
			'''

		'''
		psnr_info['Avg'] /= opt.data.total_steps
		print('Average PSNR : {:.4f}'.format(psnr_info['Avg']))
		save_yaml(opt.model.result_path+'/{:s}/{:d}/PSNR.yaml'.format(model_type,hidden_features), psnr_info)
		'''

def model_reconstruction_chunked(opt,device,model_type,hidden_features):
	
	chunk_size = 512
	full_shape = [opt.data.res[0],opt.data.res[1],opt.data.res[2]]
	
	psnr_info = {'PSNR':[], 'Avg':0, 'CR':0}
	with torch.no_grad():
		if model_type == 'CoordNet':
			model = CoordNet_Han(opt.model.params.in_ch,opt.model.params.out_ch,hidden_features).to(device)
			model.load_state_dict(torch.load(opt.model.model_path+'CoordNet-{:d}-best-model.pth'.format(hidden_features),map_location=device))
		elif model_type == 'SIREN':
			model = SIREN(opt.model.params.in_ch,opt.model.params.out_ch,hidden_features).to(device)
			model.load_state_dict(torch.load(opt.model.model_path+'SIREN-{:d}-best-model.pth'.format(hidden_features),map_location=device))

		for t in range(1,opt.data.total_steps+1):

			output = torch.empty(full_shape, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
			
			gt = np.fromfile(opt.data.path+'{:04d}'.format(t)+'.dat',dtype='<f')
			gt = (gt-gt.min())/(gt.max()-gt.min())
			gt -= 0.5
			gt *= 2.0
			
			if t == 1:
				num_parms = count_parameters(model)
				psnr_info['CR'] = opt.data.total_steps*opt.data.res[0]*opt.data.res[1]*opt.data.res[2]/num_parms

			for z_ind in range(0, full_shape[0], chunk_size):
				z_ind_end = min(full_shape[0], z_ind+chunk_size)
				z_range = z_ind_end-z_ind
				for y_ind in range(0, full_shape[1], chunk_size):
					y_ind_end = min(full_shape[1], y_ind+chunk_size)
					y_range = y_ind_end-y_ind            
					for x_ind in range(0, full_shape[2], chunk_size):
						x_ind_end = min(full_shape[2], x_ind+chunk_size)
						x_range = x_ind_end-x_ind
						
						opt['extents'] = f"{z_ind},{z_ind_end},{y_ind},{y_ind_end},{x_ind},{x_ind_end}"
																	
						grid = [z_range, y_range, x_range]
						coord_grid = make_coord_grid(grid, 'cpu', flatten=True,align_corners=True,use_half=False)
						
						coord_grid += 1.0
						coord_grid /= 2.0
						
						coord_grid[:,0] *= (x_range-1) / (full_shape[2]-1)
						coord_grid[:,1] *= (y_range-1) / (full_shape[1]-1)
						coord_grid[:,2] *= (z_range-1) / (full_shape[0]-1)
						
						coord_grid[:,0] += x_ind / (full_shape[2]-1)
						coord_grid[:,1] += y_ind / (full_shape[1]-1)
						coord_grid[:,2] += z_ind / (full_shape[0]-1)
						
						coord_grid *= 2.0
						coord_grid -= 1.0

						time = torch.zeros(coord_grid.size(0),1)
						t_ = (t-1)/(opt.data.total_steps-1)
						t_ -= 0.5
						t_ *= 2.0
						time.fill_(t_)

						coord_grid = torch.cat((time,coord_grid),dim=-1)
						
						out_tmp = forward_maxpoints(model, 
													coord_grid, max_points=2**20, 
													data_device='cpu',
													device=device)
						out_tmp = out_tmp.permute(1,0)
						out_tmp = out_tmp.view([out_tmp.shape[0]] + grid)
						output[0,:,z_ind:z_ind_end,y_ind:y_ind_end,x_ind:x_ind_end] = out_tmp

			output = output.cpu().detach().numpy()
			output = output[0][0]
			output = np.asarray(output,dtype='<f')
			output = output.flatten('F')
			output = np.clip(output,-1.0,1.0)
			output.tofile(opt.model.result_path+'/{:s}/{:d}/{:04d}'.format(model_type,hidden_features,t)+'.dat',format='<f')
			p = PSNR(gt,output)
			psnr_info['PSNR'].append(float(p))
			psnr_info['Avg'] += float(p)
			#print('[Time Step {:02d}] PSNR : {:.4f}'.format(t, p))
			save_yaml(opt.model.result_path+'/{:s}/{:d}/PSNR.yaml'.format(model_type,hidden_features), psnr_info)
				
		psnr_info['Avg'] /= opt.data.total_steps
		print('Average PSNR : {:.4f}'.format(psnr_info['Avg']))
		save_yaml(opt.model.result_path+'/{:s}/{:d}/PSNR.yaml'.format(model_type,hidden_features), psnr_info)