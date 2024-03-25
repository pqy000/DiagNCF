import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import config
import evaluate
import data_utils
import wandb
import random
min_value = 20

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=2048,
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=8,
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=3,
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=8,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=2,
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99,
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="7",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
run = wandb.init(mode="disabled")
# run = wandb.init(project="mimic3_exp", name=config.model+"_ng_"+str(args.test_num_ng+1)+"_K_"+str(args.top_k)
# 											+"_factor_"+str(args.factor_num)+"_min_"+str(min_value))

############################## PREPARE DATASET ##########################
train_data, test_data, val_data, user_num ,item_num, train_mat = data_utils.load_all()

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
val_dataset = data_utils.NCFData(
		val_data, item_num, train_mat, 0, False)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = data.DataLoader(val_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

########################### TRAINING #####################################
count, best_hr = 0, 0
best_test_hr, best_test_ndcg = 0, 0
train_loss = []
val_HR, val_NDCG = [], []
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()
	loss_avg = []
	for user, item, label in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		wandb.log({"train_loss":  loss.item()})
		loss_avg.append(loss.item())
		count += 1
	wandb.log({"avg_train_loss": np.mean(loss_avg)})


	model.eval()
	HR, NDCG = evaluate.metrics(model, val_loader, args.top_k)
	train_loss.append(np.mean(loss_avg))
	val_HR.append(np.mean(HR))
	val_NDCG.append(np.mean(NDCG))

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
	wandb.log({"HR": np.mean(HR), "NDCG": np.mean(NDCG)})

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		test_hr, test_ndcg = evaluate.metrics(model, test_loader, args.top_k)
		wandb.log({"t_hr": test_hr, "t_ndcg": test_ndcg})
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, '{}{}.pth'.format(config.model_path, config.model))

print("End. Best Validation epoch {:03d}: HR = {:.3f}, "
	  "NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
print("End. Test Performance HR = {:.3f}, NDCG = {:.3f}".format(test_hr, test_ndcg))
print("#" * 100)

file = open("items"+ config.model +".txt",'w')
file.write("Traing loss ")
for elem in train_loss:
	file.write(str(elem)+",")
file.write("\n")
file.write("HR loss ")
for elem in val_HR:
    file.write(str(elem)+",")
file.write("\n")
file.write("NDCG loss ")
for elem in val_NDCG:
	file.write(str(elem) + ",")
file.write("\n")


###########################TESTING#######################################
model_weight = torch.load('{}{}.pth'.format(config.model_path, config.model))
model.eval()
HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
# print("End. Test Performance HR = {:.3f}, NDCG = {:.3f}".format(HR, NDCG))


