import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_dataset_EndtoEnd_JM
from model import EMLP 
from torch.autograd import Variable 
import math
import time


def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,PC, PCid, Context, Targets, bs):

	if i+bs<len(PCid):
		bpc=PC[PCid[i:i+bs]]
		bc=Context[i:i+bs]
		bt=Targets[i:i+bs]
	else:
		bpc=PC[PCid[i:]]
		bc=Context[i:]
		bt=Targets[i:]
		
	return torch.from_numpy(bpc),torch.from_numpy(bc),torch.from_numpy(bt)

# Load trained model for path generation
emlp = EMLP(42, 7) 
emlp.load_state_dict(torch.load('models/emlp_100_4000_PReLU_ae_dd_final.pkl'))

if torch.cuda.is_available():
    emlp.cuda()

#load test dataset
PC, PCid, Context, Targets= load_dataset_EndtoEnd_JM(N=900)
criterion = nn.MSELoss()
# run 10 different tests
bpc_np, bc_np, bt_np = get_input(i,PC, PCid, Context, Targets, 10)
bpc = to_var(bpc_np).to(torch.float32)
bc = to_var(bc_np).to(torch.float32)
bo = emlp(bpc, bc)
for i in range(10):
	print("target: ", bt_np[i])
	print("output: ", bo[i])

# import open3d as o3d

# for i in range(10):
#     input = dataset[i]
#     # randomly change the last 14 dimensions of the input
#     input[28:] = np.random.rand(14)
#     input = torch.from_numpy(input)
#     input = input.cuda()
#     input = Variable(input)
#     output = mlp(input).cpu().data.numpy()
#     print("output: ", output)
    
