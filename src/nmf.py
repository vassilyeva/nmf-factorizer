import os.path
import dataprocessing as dp

import numpy as np
from scipy import sparse

import tables as tb
import torch
from torch.autograd import Variable
import torch.nn as nn



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
		W_data = Variable(torch.FloatTensor(n_words, n_features), requires_grad = True).to(params.device)
		E_data = Variable(torch.FloatTensor(n_features, n_entities), requires_grad = True).to(params.device)
		W_data.normal_(std = 1).abs_()
		E_data.normal_(std = 1).abs_()
		self.W = nn.Parameter(W_data)   
		self.E = nn.Parameter(E_data)
		self.device = params.device


	def forward(self, i, js):
		out = 0.
		for j in js:
			prediction_j = (self.E[i, :] * self.W[:, j]).sum(1)
			out += prediction_j.data
		return torch.FloatTensor(out).to(self.device)

	def positivise(self):
		''' Make W and H non-negative '''
		self.W.data = torch.clamp(self.W, min = 0.)
		self.E.data = torch.clamp(self.E, min = 0.)

class CustomLoss(torch.nn.Module):
	# TODO: add extra term to the loss function
	def __init__(self):
		super(CustomLoss, self).__init__()

	def penalize(self, A):
		tensor_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor
		return (A>0).type(tensor_type)*torch.clamp(A, max=0.)

	def parameter_penalty(self, P, S, i_indices, j_indices):
		''' P - for parameter matrix (E or W)
			S - for similarity matrix (Se or Sw)
		'''
		penalty = 0
		for i in i_indices:
			for j in j_indices:
				penalty += (P[i, j] * (S[i, :] - S[j, :]).norm(2)).item()
		print('penalty in penalty computing function - ', penalty )
		return penalty

	def forward(self, actual, prediction, lamb, parameters):
		penalty, parameter_penalty = 0, 0
		tensor_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor

		for p in parameters:
			penalty += ((p>0).type(tensor_type)*torch.clamp(p, min=0.)).norm(2).item()      # norm of positive values only
			parameter_penalty += self.parameter_penalty(torch.clamp(p, min = 0.) )
		penalty_tensor = torch.tensor(penalty).type(tensor_type)

		return (actual - prediction).norm(2) + lamb*penalty_tensor

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
	word_i_indices = [[Sw.indices[Sw.indptr[j]  : Sw.indptr[j+1]] for j in V_samples_indices.data]]

	# NOTE: why aren't indices for Sw and Se not the same length?? Should be the same length - 20. Definitely not more than 20!
	# TODO: check later
	entity_j_indices = Se.indices[Se.indptr[row_ind] : Se.indptr[row_ind + 1]]
	return V_samples_indices, word_i_indices, entity_j_indices
	#return V_samples_indices, torch.LongTensor(word_i_indices), torch.LongTensor(entity_j_indices)




def custom_nmf(params):
	# Construct data
	print(params.dataset_text)
	# TFIDF matrix
	V, sentences, words = dp.compute_tfidf_matrix(params)
	print('TFIDF Shape - ', V.shape)

	if params.similarity == 'cos':
		V, Sw = dp.compute_Sw_cosine(params, V, sentences, words)
	else: 
		Sw = dp.compute_wordnet_similarity_words(params, words)
	print('Constructed Sw matrix, shape - {}'.format(Sw.shape))

	#S_matrix = np.memmap(params.similarity_matrix_filename, dtype = 'float32', mode = 'r')

	Se = dp.compute_Se_cosine(params)
	print('Constructed Se matrix, shape - {}'.format(Se.shape))

	assert type(V) == sparse.csr_matrix, 'Wrong matrix type for V'
	assert type(Sw) == sparse.csr_matrix, 'Wrong matrix type for Sw'
	assert type(Se) == sparse.csr_matrix, 'Wrong matrix type for Se'

	n_words = Sw.shape[0]
	n_entities = Se.shape[0]

	# define model

	n_features = 10				# <-- For the love of god, change that when done testing!!!!

	model = NMF(n_entities, n_words, n_features, params)
	model.to(params.device)

	loss = CustomLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = params.lr) 
	n_epochs = params.n_epochs

	

	# construct tensors
	#V_tensor = torch.Tensor(V.astype(np.float32)).to(params.device)
	#V_variable = Variable(V_tensor, requires_grad = False).to(params.device)

	for epoch in range(n_epochs):
		# construct batch = one row, n positive samples, 3n negative samples
		row_permutation = np.random.permutation(n_entities) 
		for row in row_permutation:
			# construct sample
			sample_indices, word_i_indices, entity_j_indices = get_batch_indices(V, Sw, Se, row, params.device, n_entities)
			#sample_values, sample_columns, word_i_indices, entity_j_indices = generate_batch(V, Sw, Se, row, n_entities)

			optimizer.zero_grad()
			prediction = model(row, sample_indices)
			l = loss(sample_values, prediction, word_i_indices, entity_j_indices, lam, model.parameters())
			losses.append(l.item())
			differences.append((V_var - prediction).norm(2).item())
			l.backward(retain_graph = True)

			print(i, l.item(), (V_var - prediction).norm(2).item())

			optimiser.step()
			model.positivise()
			



" QUESTIONS "
'''


4. Keep similarity threshold for entities as well?


'''


