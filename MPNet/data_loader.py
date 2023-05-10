import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math

# Environment Encoder

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(2800, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
			
	def forward(self, x):
		x = self.encoder(x)
		return x

class Encoder_JM(nn.Module):
	def __init__(self):
		super(Encoder_JM, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(6000, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
	
	def forward(self, x):
		x = self.encoder(x)
		return x

#N=number of environments; NP=Number of Paths
def load_dataset(N=100,NP=4000):

	Q = Encoder()
	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()

		
	obs_rep=np.zeros((N,28),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile('../dataset/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(int(len(temp)/2),2)
		obstacles=np.zeros((1,2800),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[i]=output.numpy()



	
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../dataset/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(int(len(path)/2),2)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			

	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../dataset/e'+str(i)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(int(len(path)/2),2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					

	dataset=[]
	targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(32,dtype=np.float32)
					for k in range(0,28):
						data[k]=obs_rep[i][k]
					data[28]=paths[i][j][m][0]
					data[29]=paths[i][j][m][1]
					data[30]=paths[i][j][path_lengths[i][j]-1][0]
					data[31]=paths[i][j][path_lengths[i][j]-1][1]
						
					targets.append(paths[i][j][m+1])
					dataset.append(data)
			
	data=list(zip(dataset,targets))
	random.shuffle(data)	
	dataset,targets=list(zip(*data))
	return 	np.asarray(dataset),np.asarray(targets) 

#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000):

	obc=np.zeros((N,7,2),dtype=np.float32)
	temp=np.fromfile('../dataset/obs.dat')
	obs=temp.reshape(len(temp)/2,2)

	temp=np.fromfile('../dataset/obs_perm2.dat',np.int32)
	perm=temp.reshape(77520,7)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,7):
			for k in range(0,2):
				obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
	Q = Encoder()
	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()
	
	obs_rep=np.zeros((N,28),dtype=np.float32)	
	k=0
	for i in range(s,s+N):
		temp=np.fromfile('../dataset/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(int(len(temp)/2),2)
		obstacles=np.zeros((1,2800),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[k]=output.numpy()
		k=k+1
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../dataset/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(int(len(path)/2),2)
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			

	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../dataset/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(int(len(path)/2),2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	
					



	return 	obc,obs_rep,paths,path_lengths
	

def load_dataset_EndtoEnd_JM(N=1000, NP=10):
	import open3d as o3d
	
	import pickle

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../trajectory_data/env_{:06d}/path_{:d}.p'.format(i+1,j)
			if os.path.isfile(fname):
				path=pickle.load(open(fname,'rb'), encoding='latin1')['path']
				path_lengths[i][j]=path.shape[0]	
				if path.shape[0]> max_length:
					max_length=path.shape[0]
	paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths

	normalization_data = pickle.load(open('../trajectory_data/env_1000_range.pkl','rb'))
	ranges = normalization_data['range']
	origins = normalization_data['mid']

	PC = []   ## Point Clouds
	PCid = []  ## Point Clouds ID for each planning context
	Context = []   ## Planning Context, current state, goal state
	Targets = []   ## Targets for behavior cloning

	for i in range(0,N):
		#load obstacle point cloud
		temp=o3d.io.read_point_cloud('../trajectory_data/obs_cloud/map_'+str(i+1)+'.ply')
		temp=np.asarray(temp.points)
		temp.astype('float32')
		# downsample to 2000 points, choose randomly
		temp_ind = np.random.choice(temp.shape[0], 2000, replace=False)
		temp = temp[temp_ind]
		temp = temp.flatten()
		PC.append(temp)
		for j in range(0,NP):
			fname='../trajectory_data/env_{:06d}/path_{:d}.p'.format(i+1,j)
			if os.path.isfile(fname):
				path=pickle.load(open(fname,'rb'), encoding='latin1')['path']
				for k in range(0,len(path)):
					paths[i][j][k]=(path[k] - origins)/ranges # normalize the path
		for j in range(0,NP):
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):
					PCid.append(i)
					context = np.zeros((14), dtype=np.float32)
					context[0:7] = paths[i][j][m]
					context[7:14] = paths[i][j][path_lengths[i][j]-1]
					Context.append(context)
					Targets.append(paths[i][j][m+1])
			
	data=list(zip(PCid, Context,Targets))
	random.shuffle(data)	
	PCid, Context, Targets=list(zip(*data))
	return 	np.asarray(PC), np.asarray(PCid), np.asarray(Context), np.asarray(Targets) 

def load_dataset_JM(N=1000, NP=10):
	import open3d as o3d
	Q = Encoder_JM()
	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()
	
	obs_rep=np.zeros((N,28),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloud
		temp=o3d.io.read_point_cloud('../trajectory_data/obs_cloud/map_'+str(i+1)+'.ply')
		temp=np.asarray(temp.points)
		# downsample to 2000 points, choose randomly
		temp_ind = np.random.choice(temp.shape[0], 2000, replace=False)
		temp = temp[temp_ind]
		obstacles=np.zeros((1,6000),dtype=np.float32)
		obstacles[0]=temp.flatten()
		inp=torch.from_numpy(obstacles)
		inp=Variable(inp).cuda()
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[i]=output.numpy()



	import pickle
	normalization_data = pickle.load(open('../trajectory_data/env_1000_range.pkl','rb'))
	ranges = normalization_data['range']
	origins = normalization_data['mid']
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../trajectory_data/env_{:06d}/path_{:d}.p'.format(i+1,j)
			if os.path.isfile(fname):
				path=pickle.load(open(fname,'rb'), encoding='latin1')['path']
				path_lengths[i][j]=path.shape[0]	
				if path.shape[0]> max_length:
					max_length=path.shape[0]
	print(max_length)
			

	paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../trajectory_data/env_{:06d}/path_{:d}.p'.format(i+1,j)
			if os.path.isfile(fname):
				path=pickle.load(open(fname,'rb'), encoding='latin1')['path']
				for k in range(0,len(path)):
					paths[i][j][k]=(path[k] - origins)/ranges # normalize the path
	
					

	dataset=[]
	targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(42,dtype=np.float32)
					for k in range(0,28):
						data[k]=obs_rep[i][k]
					data[28:35]=paths[i][j][m][0:7]
					data[35:42]=paths[i][j][path_lengths[i][j]-1][0:7]
						
					targets.append(paths[i][j][m+1])
					dataset.append(data)
			
	data=list(zip(dataset,targets))
	random.shuffle(data)	
	dataset,targets=list(zip(*data))
	return 	np.asarray(dataset),np.asarray(targets) 
