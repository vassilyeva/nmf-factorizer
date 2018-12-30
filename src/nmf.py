import os.path
import dataprocessing as dp

import numpy as np
from scipy import sparse

import tables as tb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda as tc



method = 'tables'   
use_memmap = True
use_tables = False



# TODO: 
'''
1. Remove ugly global parameters - decide what to use for large datafile
2. Check wordnet similarity functions - do they compute the correct result?

'''


''' MODEL '''

class NMF(torch.nn.Module):
	def __init__(self, n_entities, n_words, n_features, params):
		super().__init__()
		W_data = Variable(torch.FloatTensor(n_features, n_words), requires_grad = True).to(params.device)
		E_data = Variable(torch.FloatTensor(n_entities, n_features), requires_grad = True).to(params.device)
		W_data.normal_(std = 1).abs_()
		E_data.normal_(std = 1).abs_()
		self.W = nn.Parameter(W_data)   
		self.E = nn.Parameter(E_data)
		self.device = params.device
		self.type = params.type


	def forward(self, i, js):
		return self.E[i, :].matmul(self.W[:, js])

	def positivise(self):
		''' Make W and H non-negative '''
		self.W.data = torch.clamp(self.W, min = 0.)
		self.E.data = torch.clamp(self.E, min = 0.)

	def print_params(self):
		print("W: ")
		for row in self.W:
			print(row.data)
		print()
		print('E')
		for row in self.E:
			print(row.data)

class CustomLoss(torch.nn.Module):
	# TODO: add extra term to the loss function
	def __init__(self, device):
		super(CustomLoss, self).__init__()
		self.device = device

	def penalize(self, A):
		tensor_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor
		return (A>0).type(tensor_type)*torch.clamp(A, max=0.)

	def similarity_penalty_Sw(self, P, S, j_indices, i_indices, device):
		''' P - for parameter matrix (E or W)
			S - for similarity matrix (Se or Sw)
		'''
		penalty = 0
		for j, i_inds in zip(j_indices, i_indices):
			Swj_whole = torch.Tensor(S.getrow(j).toarray()[0])
			Swj = torch.gather(Swj_whole, 0, i_inds).to(device)
			W_sample_i = P.index_select(1, i_inds.to(device))
			W_sample_j = P.index_select(1, j.to(device))
			diff = (W_sample_j - W_sample_i).norm(2)
			penalty += (diff * Swj).sum()

		return penalty

	def similarity_penalty_Se(self, P, S, i, j_indices, device):
		''' P - for parameter matrix (E or W)
			S - for similarity matrix (Se or Sw)
		'''
		penalty = 0
		Sei = torch.Tensor(S.getrow(i).toarray()[0])
		Se_sample = torch.gather(Sei, 0, j_indices).to(device)
		Ei = P.index_select(0, torch.LongTensor([i]).to(device))
		Ej = P.index_select(0, j_indices.to(device))

		diff = (Ei - Ej).norm(2)
		penalty = (diff*Se_sample).sum()
		return penalty

	def forward(self, actual, prediction, row_ind, word_i_indices, entity_j_indices, sample_j_indices, lamb, model, Sw, Se):
		tensor_type = model.type
		penalty = 0
		for p in model.parameters():
			penalty += ((p>0).type(tensor_type)*torch.clamp(p, min=0.)).norm(2)      # norm of positive values only

		#penalty = ((model.W > 0).type(tensor_type)*torch.clamp(model.W, min = 0.)).norm(2)
		#penalty += ((model.E > 0).type(tensor_type)*torch.clamp(model.E, min = 0.)).norm(2)

		similarity_penalty = self.similarity_penalty_Sw(model.W, Sw, sample_j_indices, word_i_indices, model.device)
		print("received similarity pernalty value: ", similarity_penalty)
		
		similarity_penalty += self.similarity_penalty_Se(model.E, Se, row_ind, entity_j_indices, model.device)
		print('received similarity penalty values : ', similarity_penalty.item())


		return (actual.to(model.device) - prediction).norm(2) + lamb*penalty + similarity_penalty.type(tensor_type)

def get_Sw_sample(Sw, row_inds, col_inds):
	''' For each index in row_inds, collect all elements from that col '''
	print(row_ind)
	print()
	print(col_inds)
	exit()

def get_batch_indices(V, Sw, Se, row_ind, device, n_entities):
	n = 20   	# number positive samples to pick
	nonzeros = V.indices[V.indptr[row_ind] : V.indptr[row_ind + 1]]
	zeros = np.setxor1d(nonzeros, np.arange(n_entities))
	nonzero_sample = torch.LongTensor(np.random.choice(nonzeros, size = min(n, len(nonzeros)), replace = False))
	zero_sample = torch.LongTensor(np.random.choice(zeros, size = min(5*n, len(zeros)), replace = False))
	V_samples_indices = torch.cat((nonzero_sample, zero_sample), 0)
	#V_sample = torch.gather(V[row_ind], 0, V_samples_indices)

	# get indices for Sw and Se
	# for word to word similarity matrix, we keep the same j'th (V_sample_indices), and we find all the words they are similar to - i.e. all nonzero values for each j
	word_i_indices = [Sw.indices[Sw.indptr[j]  : Sw.indptr[j+1]] for j in V_samples_indices.data]

	entity_j_indices = Se.indices[Se.indptr[row_ind] : Se.indptr[row_ind + 1]]
	return V_samples_indices, torch.LongTensor(word_i_indices), torch.LongTensor(entity_j_indices)


def custom_nmf(V, Sw, Se, params):
	# Construct data
	print(params.dataset_text)
	# TFIDF matrix
	

	assert type(V) == sparse.csr_matrix, 'Wrong matrix type for V'
	assert type(Sw) == sparse.csr_matrix, 'Wrong matrix type for Sw'
	assert type(Se) == sparse.csr_matrix, 'Wrong matrix type for Se'

	n_words = Sw.shape[0]
	n_entities = Se.shape[0]

	losses = []

	# define model

	n_features = 10				# <-- For the love of god, change that when done testing!!!!

	model = NMF(n_entities, n_words, n_features, params)
	model.to(params.device)

	loss = CustomLoss(params.device)
	optimizer = torch.optim.SGD(model.parameters(), lr = params.lr) 
	n_epochs = params.n_epochs
	lam = 10 # TODO: clean this up
	

	# construct tensors
	V_tensor = torch.Tensor(V.todense().astype(np.float32))
	V_var = Variable(V_tensor, requires_grad = False)

	model.print_params()
	print("\n\n After learning")

	for epoch in range(n_epochs):
		# construct batch = one row, n positive samples, 3n negative samples
		row_permutation = np.random.permutation(n_entities) 
		for row_ind in row_permutation:
			# construct sample
			sample_indices, word_i_indices, entity_j_indices = get_batch_indices(V, Sw, Se, row_ind, params.device, n_entities)
			#sample_values, sample_columns, word_i_indices, entity_j_indices = generate_batch(V, Sw, Se, row, n_entities)

			target = torch.gather(V_var[row_ind], 0, sample_indices)

			optimizer.zero_grad()
			prediction = model(row_ind, sample_indices)

			l = loss(target, prediction, row_ind, word_i_indices, entity_j_indices, sample_indices, lam, model, Sw, Se)
			losses.append(l.item())
			model.print_params()
			l.backward(retain_graph = True)

			optimizer.step()
			model.positivise()

			model.print_params()

			exit()
			



" QUESTIONS "
'''


4. Keep similarity threshold for entities as well?


'''


