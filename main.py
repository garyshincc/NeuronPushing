import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import copy

import numpy as np

from torchvision import datasets, transforms
from model import *

torch.set_printoptions(profile="default")


#load the data
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
					 transform=transforms.Compose([
						 transforms.ToTensor(),
						 transforms.Normalize((0.1307,), (0.3081,))
					 ])),
	batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
						 transforms.ToTensor(),
						 transforms.Normalize((0.1307,), (0.3081,))
					 ])),
	batch_size=1000, shuffle=True)



##### Initialize the convolutional network
conv_model = ConvNetwork()
conv_model2 = ConvNetwork()
optimizer = optim.Adam(conv_model2.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

for param in conv_model.fc3.parameters():
	print(param)
	mean_weight = param.mean()
	print(f"mean_weight: {mean_weight}")
	num_over = (param > mean_weight).sum()
	num_under = (param <= mean_weight).sum()
	print(f"num_over: {num_over}")
	print(f"num_under: {num_under}")
	tanh_param = F.tanh(param * 1.01)
	print(f"tanh_param: {tanh_param}")


for i, (data, target) in enumerate(train_loader):
	data, target = data, target
	output, latent = conv_model(data)
	mean_latent = latent.mean()
	activated = latent.detach()
	activated = latent[0] > mean_latent
	for param in conv_model.fc2.parameters():
		old_param = copy.deepcopy(param)
		for neuron_idx, weight in enumerate(param):
		if activated[neuron_idx]:
			weight += (weight * 0.001)
		else:
			weight -= (weight * 0.001)
	if i % 200 == 0:
		print(target)
		print(output)
#		 for param in conv_model.fc2.parameters():
#		 print(param)
	if i > 1000:
		break
		
params1 = conv_model.named_parameters()
params2 = conv_model2.named_parameters()

dict_params2 = dict(params2)

for name1, param1 in params1:
	if name1 in dict_params2:
		dict_params2[name1].data.copy_(param1.data)

# ##### train the network
device = 'cpu'
conv_model2.train()
for param in conv_model2.parameters():
	print(param.shape)
	param.requires_grad = False
for param in conv_model2.fc3.parameters():
	print(param.shape)
	param.requires_grad = True


for i, (data, target) in enumerate(train_loader):
	optimizer.zero_grad()
	output, latent = conv_model2(data)
	output = output.squeeze(1)
	output = F.softmax(output)
#	 print(f"output: {output}")
#	 print(f"target: {target}")
#	 break
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	if i % 100 == 0:
		print(loss)

conv_model2.eval()
for i, (data, target) in enumerate(test_loader):
	data, target = data.to(device), target.to(device)
	
	output, latent = conv_model2(data)
	output = output.squeeze(1)
	output = F.softmax(output)
	val, index = output.max(1)
	correct = (index == target)
	print(float(correct.sum()) / float(len(correct)))
	break
# conv network results better even without hyperparameter tuning

num_examples = 1000
avg_dicts = dict.fromkeys([i for i in range(10)])
avg_dicts[0] = list()
avg_dicts[1] = list()
avg_dicts[2] = list()
avg_dicts[3] = list()
avg_dicts[4] = list()
avg_dicts[5] = list()
avg_dicts[6] = list()
avg_dicts[7] = list()
avg_dicts[8] = list()
avg_dicts[9] = list()
for i, (data, target) in enumerate(train_loader):
	data, target = data, target
	output, latent = conv_model(data)
	probs = F.softmax(output)
	arr = np.round(np.array(probs[0].detach()), 3)
	avg_dicts[target.item()].append(arr)
#	 print(target.item())
#	 print(f"avg_dicts: {avg_dicts}")
#	 print(f"target: {target}")
	if i > num_examples:
		break

avg_1_probs = np.mean(avg_dicts[1], 0)
avg_2_probs = np.mean(avg_dicts[2], 0)
avg_3_probs = np.mean(avg_dicts[3], 0)
avg_4_probs = np.mean(avg_dicts[4], 0)
avg_5_probs = np.mean(avg_dicts[5], 0)
avg_6_probs = np.mean(avg_dicts[6], 0)
avg_7_probs = np.mean(avg_dicts[7], 0)
avg_8_probs = np.mean(avg_dicts[8], 0)
avg_9_probs = np.mean(avg_dicts[9], 0)

# print(np.array(avg_dicts[1]))
print(f"avg_1_probs: {avg_1_probs}")
print(f"avg_2_probs: {avg_2_probs}")
print(f"avg_3_probs: {avg_3_probs}")
print(f"avg_4_probs: {avg_4_probs}")
print(f"avg_5_probs: {avg_5_probs}")
print(f"avg_6_probs: {avg_6_probs}")
print(f"avg_7_probs: {avg_7_probs}")
print(f"avg_8_probs: {avg_8_probs}")
print(f"avg_9_probs: {avg_9_probs}")

delta1_7 = avg_1_probs - avg_7_probs
delta1_2 = avg_1_probs - avg_2_probs
print(f"delta1_7: {delta1_7}")
print(f"delta1_2: {delta1_2}")

