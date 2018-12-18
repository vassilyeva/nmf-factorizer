import os.path
import dataprocessing as dp

import numpy as np
from scipy import sparse

import tables as tb
import torch

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
	def __init__(self, n_words, n_entities, n_features, params):
		super().__init__()
	
		W_data = Variable(torch.FloatTensor(n_words, n_features), requires_grad = True).to(params.device)
		H_data = Variable(torch.FloatTensor(n_features, n_entities), requires_grad = True).to(params.device)
		W_data.normal_(std = 1).abs_()
		H_data.normal_(std = 1).abs_()
		self.W = nn.Parameter(W_data)
		self.H = nn.Parameter(H_data)


	def forward(self):
		out = self.W.mm(self.H)
		return out

	def positivise(self):
		''' Make W and H non-negative '''
		self.W.data = torch.clamp(self.W, min = 0.)
		self.H.data = torch.clamp(self.H, min = 0.)

class CustomLoss(torch.nn.Module):
	def __init__(self):
		super(CustomLoss, self).__init__()

	def penalize(self, A):
		tensor_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor
		return (A>0).type(tensor_type)*torch.clamp(A, max=0.)

	def forward(self, actual, prediction, lamb, parameters):
		penalty = 0
		tensor_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor

		for p in parameters:
			penalty += ((p>0).type(tensor_type)*torch.clamp(p, min=0.)).norm(2).item()      # norm of positive values only
		penalty_tensor = torch.tensor(penalty).type(tensor_type)

		return (actual - prediction).norm(2) + lamb*penalty_tensor


def custom_nmf(params):
	print(params.dataset_text)
	# TFIDF matrix
	V, sentences, words = dp.compute_tfidf_matrix(params)
	print('TFIDF Shape - ', V.shape)

	if params.similarity == 'cos' and (not os.path.isfile(params.h5omega) and method == 'tables'):
		V = dp.compute_cosine_similarity_words(params, V, sentences, words)
	elif method == 'tables' and not os.path.isfile(params.h5omega): 
		dp.compute_wordnet_similarity_words(params, words)

	#S_matrix = np.memmap(params.similarity_matrix_filename, dtype = 'float32', mode = 'r')

	S = dp.compute_entity_matrix(params)

	

	# the learning 
	

" QUESTIONS "
'''


4. Keep similarity threshold for entities as well?


'''


