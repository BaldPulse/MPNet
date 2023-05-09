import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import os.path
import random
import open3d as o3d


def load_dataset(N=30000,NP=1800):

	obstacles=np.zeros((N,2800),dtype=np.float32)
	for i in range(0,N):
		temp=np.fromfile('../dataset/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(int(len(temp)/2),2)
		obstacles[i]=temp.flatten()

	
	return 	obstacles

def load_dataset_JM(N=1000):

	obstacles=np.zeros((N,6000),dtype=np.float32)
	for i in range(0,N):
		temp=o3d.io.read_point_cloud('../trajectory_data/obs_cloud/map_'+str(i+1)+'.ply')
		temp=np.asarray(temp.points)
		# downsample to 2000 points, choose randomly
		temp_ind = np.random.choice(temp.shape[0], 2000, replace=False)
		temp = temp[temp_ind]
		obstacles[i]=temp.flatten()

	
	return 	obstacles	
