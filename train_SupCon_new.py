import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from resnet_big import *
from loss_functions import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('Arguments for Training')
parser.add_argument('--dataset_path', type = str, default = '/home/pulkit/Desktop/MTP/person_reid_contrastive_representation_learning/', help = 'The dataset path')
parser.add_argument('--batch_size', type = int, default = 16, help = 'The training batch size')
parser.add_argument('--epochs', type = int, default = 1000, help = 'The number of training epochs')
parser.add_argument('--checkpoint_directory', type = str, default = 'checkpoints_cifar-10_SupCon', help = 'The path to store the model checkpoints')
parser.add_argument('--save_frequency', type = int, default = 10, help = 'Specify the number of epochs after which the model will be saved')
parser.add_argument('--lr', type = float, default = 0.5, help = 'The learning rate')
parser.add_argument('--lr_decay_rate', type = float, default = 0.1, help = 'The learning rate decay rate')
parser.add_argument('--cosine', type = int, default = 1, help = 'Whether to use learning rate cosine annealing')
args = parser.parse_args()

writer = SummaryWriter('logs_cifar10_SupCon')
os.makedirs(args.checkpoint_directory, exist_ok = True)

class TwoCropImageTransform():

	def __init__(self):

		self.mean = (0.4914, 0.4822, 0.4465) # For CIFAR-10
		self.std = (0.2023, 0.1994, 0.2010) # For CIFAR-10
		# For simple dataset like CIFAR-10
		self.transforms = transforms.Compose(
			[transforms.RandomHorizontalFlip(p = 0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean = self.mean, std = self.std)]
			)

		# For complicated datasets like "ImageNet", "MARS", "Market-1501"

		# self.transforms = transforms.Compose(
		# 	[transforms.RandomResizedCrop(size = 32, scale = (0.2, 1)),
		# 	transforms.RandomHorizontalFlip(p = 0.5),
		# 	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p = 0.8),
		# 	transforms.RandomGrayscale(p = 0.2),
		# 	transforms.ToTensor(),
		# 	transforms.Normalize(mean = self.mean, std = self.std)]
		# 	)

	def __call__(self, x):

		img1 = self.transforms(x)
		img2 = self.transforms(x)
		return [img1, img2]

class TwoCropTargetTransform():

	def __init__(self):
		pass

	def __call__(self, x):
		
		return [x, x]

train_dataset = torchvision.datasets.CIFAR10(root = args.dataset_path, train = True, 
	transform = TwoCropImageTransform(), target_transform = TwoCropTargetTransform(), download = True)

train_loader = DataLoader(train_dataset, shuffle = True, num_workers = 0, batch_size = args.batch_size)
no_of_training_batches = len(train_dataset)/args.batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SupConResNet()
criterion = SupConLoss(temp = 0.1)
model.to(device)
criterion.to(device)
cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-4)

if args.cosine:
	eta_min = args.lr * (args.lr_decay_rate ** 3)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = eta_min)

def run_epoch(data_loader, model, optimizer, epoch_count = 0):

	model.to(device)
	model.train()

	running_loss = 0.0

	for batch_id, (imgs, labels) in enumerate(data_loader):

		iter_count = epoch_count * len(data_loader) + batch_id
		images = torch.cat((imgs[0], imgs[1]), dim = 0)
		targets = torch.cat((labels[0], labels[1]), dim = 0)		
		images = images.to(device)
		targets = targets.to(device)
		embeddings = model(images)
		sup_loss = criterion(embeddings, targets)

		optimizer.zero_grad()
		sup_loss.backward()
		optimizer.step()

		running_loss = running_loss + sup_loss.item()

		# Adding the logs in Tensorboard
		writer.add_scalar('Supervised Contrastive Loss', sup_loss.item(), iter_count)

	return running_loss

def fit(data_loader, model, optimizer, scheduler, n_epochs):

	print('Training Started\n')
	
	for param_group in optimizer.param_groups:
		writer.add_scalar('Learning Rate', param_group['lr'], 0)
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(data_loader, model, optimizer, epoch_count = epoch)
		loss = loss/no_of_training_batches

		scheduler.step()

		for param_group in optimizer.param_groups:
			writer.add_scalar('Learning Rate', param_group['lr'], (epoch + 1))
		
		if (((epoch + 1) % args.save_frequency) == 0):
			print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
			torch.save({'state_dict': model.cpu().state_dict()}, args.checkpoint_directory + '/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, model, optimizer = optimizer, scheduler = scheduler, n_epochs = args.epochs)