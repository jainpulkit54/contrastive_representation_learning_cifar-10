import torch
import torch.nn as nn
import numpy as np

class SupConLoss(nn.Module):

	'''
	Args:
		batch_size: embeddings.shape[0]/2
		embeddings: The embeddings of shape [batch_size * n_views, embedding_dimension], E.g., [8, 128]
		targets: The labels of the corresponding image's embedding. The shape is [batch_size * n_views]

		where, (n_views) represent the number of multiviewed samples of each image taken (which is 2 for SupCon)
	
	Returns:
		A scalar loss value
	'''

	def __init__(self, temp = 0.1):
		
		super(SupConLoss, self).__init__()
		self.temp = temp
	
	def forward(self, embeddings, targets):
	
		multiviewed_batch_size = embeddings.shape[0]
		zi_za_matrix = torch.div(torch.matmul(embeddings, embeddings.T), self.temp)

		######### For the purpose of Numerical Stability ###############
		zi_za_matrix_max, _ = torch.max(zi_za_matrix, dim = 1, keepdim = True)
		zi_za_matrix = zi_za_matrix - zi_za_matrix_max
		#######################################################################
		
		zi_za_exp_matrix = torch.exp(zi_za_matrix)
		denominator_mask = torch.ones(multiviewed_batch_size, multiviewed_batch_size).cuda() - torch.eye(multiviewed_batch_size, multiviewed_batch_size).cuda()
		denominator_term = torch.mul(zi_za_exp_matrix, denominator_mask)
		
		targets = targets.contiguous().view(-1, 1)
		positive_mask = torch.eq(targets, targets.T).float().cuda() - torch.eye(multiviewed_batch_size, multiviewed_batch_size).cuda()
		cardinality_p = torch.sum(positive_mask, dim = 1, keepdim = True)

		per_image = torch.log(torch.div(zi_za_exp_matrix, torch.sum(denominator_term, dim = 1, keepdim = True)))
		per_image = torch.mul(per_image, positive_mask)
		per_image = -1 * torch.div(torch.sum(per_image, dim = 1, keepdim = True), cardinality_p)
		loss = torch.mean(per_image)
		
		return loss

class SimCLR(nn.Module):

	'''
	Args:
		batch_size: embeddings.shape[0]/2
		embeddings: The embeddings of shape [batch_size * n_views, embedding_dimension], E.g., [8, 128]

		where, (n_views) represent the number of multiviewed samples of each image taken (which is 2 for SimCLR)
	
	Returns:
		A scalar loss value
	'''

	def __init__(self, temp = 0.1):
		
		super(SimCLR, self).__init__()
		self.temp = temp
	
	def forward(self, embeddings):
	
		multiviewed_batch_size = embeddings.shape[0]
		batch_size = int(embeddings.shape[0]/2)
		zi_za_matrix = torch.div(torch.matmul(embeddings, embeddings.T), self.temp)

		######### For the purpose of Numerical Stability ###############
		zi_za_matrix_max, _ = torch.max(zi_za_matrix, dim = 1, keepdim = True)
		zi_za_matrix = zi_za_matrix - zi_za_matrix_max
		#######################################################################
		
		zi_za_exp_matrix = torch.exp(zi_za_matrix)
		denominator_mask = torch.ones(multiviewed_batch_size, multiviewed_batch_size).cuda() - torch.eye(multiviewed_batch_size, multiviewed_batch_size).cuda()
		denominator_term = torch.mul(zi_za_exp_matrix, denominator_mask)
		
		positive_mask = torch.eye(batch_size, batch_size).repeat(2, 2).cuda() - torch.eye(multiviewed_batch_size, multiviewed_batch_size).cuda()

		per_image = torch.log(torch.div(zi_za_exp_matrix, torch.sum(denominator_term, dim = 1, keepdim = True)))
		per_image = torch.mul(per_image, positive_mask)
		per_image = -1 * torch.sum(per_image, dim = 1, keepdim = True)
		loss = torch.mean(per_image)
		
		return loss	