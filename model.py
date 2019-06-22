import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import copy

import numpy as np

from torchvision import datasets, transforms

torch.set_printoptions(profile="default")

class LinearNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 1000, bias=False)
		self.fc2 = nn.Linear(1000, 1000, bias=False)
		self.fc3 = nn.Linear(1000, 500, bias=False)
		self.fc4 = nn.Linear(500, 200, bias=False)
		self.fc5 = nn.Linear(200, 10, bias=False)
		
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = self.fc5(x)
		return x
	
class ConvNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 5, 1,)
		self.conv2 = nn.Conv2d(32, 32, 5, 1)
		self.fc1 = nn.Linear(4*4*32, 500, bias=False)
		self.fc2 = nn.Linear(500, 200, bias=False)
		self.fc3 = nn.Linear(200, 10, bias=False)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*32)
		x = F.relu(self.fc1(x))
		latent = F.relu(self.fc2(x))
		logit = self.fc3(latent)
		return logit, latent