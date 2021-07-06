import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from resnet_big import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('Arguments for Training')
parser.add_argument('--dataset_path', type = str, default = '/home/pulkit/Desktop/MTP/person_reid_contrastive_representation_learning/', help = 'The path of the training set images')
parser.add_argument('--batch_size', type = int, default = 128, help = 'The training batch size')
parser.add_argument('--epochs', type = int, default = 100, help = 'The number of training epochs')
parser.add_argument('--classifier_checkpoint_directory', type = str, default = 'checkpoints_cifar-10_classifier', help = 'The path to store the model checkpoints')
parser.add_argument('--SupCon_checkpoint_directory', type = str, default = 'checkpoints_cifar-10_SupCon', help = 'The path where SupCon checkpoints are stored')
parser.add_argument('--save_frequency', type = int, default = 5, help = 'Specify the number of epochs after which the model will be saved')
parser.add_argument('--lr', type = float, default = 0.1, help = 'The learning rate')
parser.add_argument('--lr_decay_rate', type = float, default = 0.1, help = 'The learning rate decay rate')
parser.add_argument('--cosine', type = int, default = 1, help = 'Whether to use learning rate cosine annealing')

args = parser.parse_args()

writer = SummaryWriter('logs_cifar10_classifier')
os.makedirs(args.classifier_checkpoint_directory, exist_ok = True)

mean = (0.4914, 0.4822, 0.4465) # For CIFAR-10
std = (0.2023, 0.1994, 0.2010) # For CIFAR-10

train_transform = transforms.Compose(
	[transforms.RandomResizedCrop(size = 32, scale = (0.2, 1)),
	transforms.RandomHorizontalFlip(p = 0.5),
	transforms.ToTensor(),
	transforms.Normalize(mean = mean, std = std)]
	)

test_transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize(mean = mean, std = std)]
	)

train_dataset = torchvision.datasets.CIFAR10(root = args.dataset_path, train = True, transform = train_transform, download = True)
test_dataset = torchvision.datasets.CIFAR10(root = args.dataset_path, train = False, transform = test_transform, download = True)
train_data_loader = DataLoader(train_dataset, shuffle = True, num_workers = 16, batch_size = args.batch_size)
test_data_loader = DataLoader(test_dataset, shuffle = True, num_workers = 16, batch_size = args.batch_size)

no_of_training_batches = len(train_dataset)/args.batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_embeddings = SupConResNet()
checkpoint = torch.load(args.SupCon_checkpoint_directory + '/model_epoch_150.pth')
model_parameters = checkpoint['state_dict']
model_embeddings.load_state_dict(model_parameters)
model_embeddings.to(device)
model_embeddings.eval()

model_classifier = LinearClassifier()
criterion = nn.CrossEntropyLoss(reduction = 'mean')
activation = nn.Softmax(dim = 1)

model_classifier.to(device)
criterion.to(device)
cudnn.benchmark = True

optimizer = optim.SGD(model_classifier.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-4)
# optimizer = optim.Adam(model_classifier.parameters(), lr = args.lr, betas = (0.9, 0.999))

if args.cosine:
	eta_min = args.lr * (args.lr_decay_rate ** 3)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = eta_min)

def run_epoch(train_data_loader, model_embeddings, model_classifier, optimizer, epoch_count = 0):

	model_classifier.to(device)
	model_classifier.train()

	running_loss = 0.0

	for batch_id, (imgs, labels) in enumerate(train_data_loader):

		iter_count = epoch_count * len(train_data_loader) + batch_id
		imgs = imgs.to(device)
		labels = labels.to(device)
		
		with torch.no_grad():
			embeddings = model_embeddings.encoder(imgs)
		
		pre_activations = model_classifier(embeddings)
		cross_entropy_loss = criterion(pre_activations, labels)
		
		optimizer.zero_grad()
		cross_entropy_loss.backward()
		optimizer.step()

		running_loss = running_loss + cross_entropy_loss.item()

		# Adding the logs in Tensorboard
		writer.add_scalar('Cross Entropy Loss', cross_entropy_loss.item(), iter_count)
		
	return running_loss

def fit(train_data_loader, test_data_loader, model_embeddings, model_classifier, optimizer, n_epochs):

	print('Training Started\n')

	for param_group in optimizer.param_groups:
		writer.add_scalar('Learning Rate', param_group['lr'], 0)
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(train_data_loader, model_embeddings, model_classifier, optimizer, epoch_count = epoch)
		loss = loss/no_of_training_batches

		scheduler.step()

		for param_group in optimizer.param_groups:
			writer.add_scalar('Learning Rate', param_group['lr'], (epoch + 1))

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		
		if (((epoch + 1) % args.save_frequency) == 0):
			torch.save({'state_dict': model_classifier.cpu().state_dict()}, args.classifier_checkpoint_directory + '/model_epoch_' + str(epoch + 1) + '.pth')

		model_classifier.eval()
		model_classifier.to(device)

		train_running_matches = 0
		test_running_matches = 0

		################ Code for calculating classification accuracy on train dataset ################
		for batch_id, (imgs, labels) in enumerate(train_data_loader):
			
			imgs = imgs.to(device)
			
			with torch.no_grad():
				embeddings = model_embeddings.encoder(imgs)
				pre_activations = model_classifier(embeddings)
			
			probabilities = activation(pre_activations)
			classes_predicted = torch.argmax(probabilities, dim = 1, keepdim = False)
			num_classes_matched = torch.sum(torch.eq(classes_predicted.cpu(), labels).float())
			train_running_matches = train_running_matches + num_classes_matched			
		###############################################################################################

		train_accuracy = (train_running_matches/len(train_dataset)) * 100

		################ Code for calculating classification accuracy on test dataset ################
		for batch_id, (imgs, labels) in enumerate(test_data_loader):

			imgs = imgs.to(device)
			
			with torch.no_grad():
				embeddings = model_embeddings.encoder(imgs)
				pre_activations = model_classifier(embeddings)
			
			probabilities = activation(pre_activations)
			classes_predicted = torch.argmax(probabilities, dim = 1, keepdim = False)
			num_classes_matched = torch.sum(torch.eq(classes_predicted.cpu(), labels).float())
			test_running_matches = test_running_matches + num_classes_matched
		###############################################################################################

		test_accuracy = (test_running_matches/len(test_dataset)) * 100

		print('Train set classification accuracy after epoch ' + str(epoch + 1) + ' is:', train_accuracy)
		print('Test set classification accuracy after epoch ' + str(epoch + 1) + ' is:', test_accuracy)
		# Adding the logs in Tensorboard
		writer.add_scalar('Train set classification accuracy', train_accuracy, epoch)
		writer.add_scalar('Test set classification accuracy', test_accuracy, epoch)

fit(train_data_loader, test_data_loader, model_embeddings = model_embeddings, model_classifier = model_classifier, optimizer = optimizer, n_epochs = args.epochs)