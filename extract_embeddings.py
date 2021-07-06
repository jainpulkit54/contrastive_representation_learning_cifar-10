import os
import h5py
import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from resnet_big import *

parser = argparse.ArgumentParser('Arguments for Extracting Embeddings')
parser.add_argument('--dataset_path', type = str, default = '/home/pulkit/Desktop/MTP/person_reid_contrastive_representation_learning/', help = 'The dataset path')
parser.add_argument('--batch_size', type = int, default = 128, help = 'The batch size')
parser.add_argument('--checkpoint_directory', type = str, default = 'checkpoints_cifar-10_SupCon', help = 'The path where the model checkpoints are stored')
parser.add_argument('--embeddings_folder', type = str, default = 'cifar-10_test_set_embeddings_SupCon', help = 'The path where the embeddings will be stored')
parser.add_argument('--use_test_set', type = int, default = 1, help = 'Set this to 1 if want to use test set else 0 for train set')
args = parser.parse_args()

os.makedirs(args.embeddings_folder, exist_ok = True)
images_embeddings = open(args.embeddings_folder + '/images_embeddings.npy', 'wb')
images_class = open(args.embeddings_folder + '/images_class.npy', 'wb')

if args.use_test_set:
	flag = False
else:
	flag = True

dataset = datasets.CIFAR10(root = args.dataset_path, train = flag, transform = transforms.ToTensor(), download = True)
data_loader = DataLoader(dataset, shuffle = True, num_workers = 0, batch_size = args.batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SupConResNet()
checkpoint = torch.load(args.checkpoint_directory + '/model_epoch_350.pth')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.to(device)
model.eval()

def extract_embeddings(data_loader):

	print('Extracting Embeddings:')
	emb = []
	class_info = []

	for batch_id, (imgs, clss) in enumerate(data_loader):

		imgs = imgs.to(device)
		with torch.no_grad():
			embeddings = model(imgs)
		embeddings = embeddings.cpu().numpy()
		emb.extend(embeddings)
		class_info.extend(clss)

	return emb, class_info

embeddings, class_info = extract_embeddings(data_loader)
embeddings = np.array(embeddings)
class_info = np.array(class_info)

np.save(images_embeddings, embeddings)
np.save(images_class, class_info)

print(embeddings.shape)
print(class_info.shape)
