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


    
def main(args):
	# Create model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
    
    
	# Build data loader
	PC, PCid, Context, Targets= load_dataset_EndtoEnd_JM(N=900)
	
	# Build the models
	emlp = EMLP(args.input_size, args.output_size)
    
	if torch.cuda.is_available():
		emlp.cuda()

	# Loss and Optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adagrad(emlp.parameters()) 
    
	import time
	# Train the Models
	total_loss=[]
	print (len(PC))
	print (len(Context))
	print (len(Targets))
	sm=100 # start saving models after 100 epochs
	last_epoch_time = time.time()
	for epoch in range(args.num_epochs):
		print ("epoch" + str(epoch))
		avg_loss=0
		for i in range (0,len(PCid),args.batch_size):
			# Forward, Backward and Optimize
			emlp.zero_grad()			
			bpc_np, bc_np, bt_np = get_input(i,PC, PCid, Context, Targets, args.batch_size)
			bpc = to_var(bpc_np).to(torch.float32)
			bc = to_var(bc_np).to(torch.float32)
			bt = to_var(bt_np).to(torch.float32)
			bo = emlp(bpc, bc)
			loss = criterion(bo,bt)
			avg_loss=avg_loss+loss.data.item()
			loss.backward()
			optimizer.step()
		print ("--average loss:")
		print (avg_loss/(len(PC)/args.batch_size))
		total_loss.append(avg_loss/(len(PC)/args.batch_size))
		print ("--time last epoch:" + str(time.time()-last_epoch_time))
		print("estimated time left: {:.2f} hours".format((args.num_epochs-epoch)*(time.time()-last_epoch_time)/3600))
		last_epoch_time = time.time()
		# Save the models
		if epoch==sm:
			model_path='emlp_100_4000_PReLU_ae_dd'+str(sm)+'.pkl'
			torch.save(emlp.state_dict(),os.path.join(args.model_path,model_path))
			sm=sm+50 # save model after every 50 epochs from 100 epoch ownwards
	torch.save(total_loss,'total_loss.dat')
	model_path='emlp_100_4000_PReLU_ae_dd_final.pkl'
	torch.save(emlp.state_dict(),os.path.join(args.model_path,model_path))
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
	parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
	parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')

	# Model parameters
	parser.add_argument('--input_size', type=int , default=42, help='dimension of the input vector')
	parser.add_argument('--output_size', type=int , default=7, help='dimension of the output vector')
	parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')

	parser.add_argument('--num_epochs', type=int, default=500)
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	args = parser.parse_args()
	print(args)
	main(args)



