import argparse
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser('Arguments for tsne embeddings settings')
parser.add_argument('--embeddings_folder', type = str, default = 'cifar-10_train_set_embeddings_SimCLR', help = 'The path where the embeddings are stored')
parser.add_argument('--image_file_name', type = str, default = 'cifar-10_train_set_embeddings_SimCLR_', help = 'The image file')
args = parser.parse_args()

class_num = np.load(args.embeddings_folder + '/images_class.npy')
image_embeddings = np.load(args.embeddings_folder + '/images_embeddings.npy')

print('Files successfully loaded')
print(image_embeddings.shape)
print(class_num.shape)

tsne = TSNE(n_components = 2)
x = tsne.fit_transform(image_embeddings)
np.random.seed(0)

def plot_embeddings_3d(embeddings, targets):

	fig = plt.figure()
	ax = Axes3D(fig)
	### For CIFAR-10
	legend = ['deer', 'automobile', 'bird', 'horse', 'airplane', 'frog', 'truck', 'cat', 'dog', 'ship']
	colors = np.random.rand(10, 3)
	for i in range(10):

		legend.append(str(i))
		inds = np.where(targets == i)[0]
		x = embeddings[inds, 0]
		y = embeddings[inds, 1]
		z = embeddings[inds, 2]
		ax.scatter(x, y, z, alpha = 1, color = colors[i, :])

	plt.legend(legend)
	plt.savefig(args.image_file_name + 'tsne3d.png')
	plt.show()

def plot_embeddings_2d(embeddings, targets):

	fig, ax = plt.subplots()
	### For CIFAR-10
	legend = ['deer', 'automobile', 'bird', 'horse', 'airplane', 'frog', 'truck', 'cat', 'dog', 'ship']
	colors = np.random.rand(10, 3)
	for i in range(10):

		legend.append(str(i))
		inds = np.where(targets == i)[0]
		x = embeddings[inds, 0]
		y = embeddings[inds, 1]
		ax.scatter(x, y, alpha = 1, color = colors[i, :])

	plt.legend(legend)
	plt.savefig(args.image_file_name + 'tsne2d.png')
	plt.show()

# plot_embeddings_3d(x, class_num)
plot_embeddings_2d(x, class_num)