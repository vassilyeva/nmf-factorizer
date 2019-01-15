import os.path
import dataprocessing as dp

import numpy as np
from scipy import sparse

import tables as tb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda as tc

from sklearn.utils import shuffle
import timeit
import matplotlib.pyplot as plt



print_time = False
print_time_similarity = False


# TODO: 
'''
1. Remove ugly global parameters - decide what to use for large datafile
2. Check wordnet similarity functions - do they compute the correct result?

'''


''' MODEL '''

class NMF(torch.nn.Module):
	def __init__(self, n_entities, n_words, n_features, params):
		super().__init__()
		W_data = Variable(torch.FloatTensor(n_features, n_words))
		E_data = Variable(torch.FloatTensor(n_entities, n_features))
		W_data.normal_(std = 1).abs_()
		E_data.normal_(std = 1).abs_()
		self.W = nn.Parameter(W_data)   
		self.E = nn.Parameter(E_data)
		self.device = params.device
		self.floatType = params.floatType
		self.longType = params.longType


	def forward(self, batch):
		'''   For every (i,j) in batch, compute E[i, :]W[:, j]   '''
		row_indices = torch.LongTensor([b[0] for b in batch]).to(self.device)
		col_indices = torch.LongTensor([b[1] for b in batch]).to(self.device)
		E_sample = self.E.to(self.device).index_select(0, row_indices)
		W_sample = self.W.to(self.device).index_select(1, col_indices)

		result = (E_sample * W_sample.transpose(0,1)).sum(1).cpu()
		return result


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

	'''
	def similarity_penalty_Sw(self, P, S, j_indices, i_indices, device):
			P - for parameter matrix (E or W)
			S - for similarity matrix (Se or Sw)
		
		penalty = 0
		for j, i_inds in zip(j_indices, i_indices):
			Swj_whole = torch.Tensor(S.getrow(j).toarray()[0])
			Swj = torch.gather(Swj_whole, 0, i_inds).to(device)
			W_sample_i = P.index_select(1, i_inds.to(device))
			W_sample_j = P.index_select(1, j.to(device))
			diff = (W_sample_j - W_sample_i).norm(2)
			penalty += (diff * Swj).sum()

		return penalty
	'''
	'''
	def similarity_penalty_Se(self, P, S, i, j_indices, device):
		 P - for parameter matrix (E or W)
			S - for similarity matrix (Se or Sw)
		
		penalty = 0
		Sei = torch.Tensor(S.getrow(i).toarray()[0])
		Se_sample = torch.gather(Sei, 0, j_indices).to(device)
		Ei = P.index_select(0, torch.LongTensor([i]).to(device))
		Ej = P.index_select(0, j_indices.to(device))

		diff = (Ei - Ej).norm(2)
		penalty = (diff*Se_sample).sum()
		return penalty
		'''


	def entity_penalty2(self, P, S, batch, device, entity = True):
		''' For entity term: 
			For every i (= item[0]), find set of js (= nonzero values in row i in Se)
				For every j: 
					total += Se(i,j)*(E(i) - E(j)).norm(2)
		'''
		# For every entity in the batch - find a set of entities that are similar to it
		ind = 0 if entity else 1
		S_sample_indices = torch.LongTensor([S.getrow(item[ind]).nonzero()[1] for item in batch]).to(device)
		penalty = 0

		for i, js in zip([item[ind] for item in batch], S_sample_indices):
			Pi = P.index_select(ind, torch.LongTensor([i])).to(device)
			Pj = P.to(device).index_select(ind, js)

			if not entity:
				Pi = Pi.transpose(0,1)
				Pj = Pj.transpose(0,1)

			print('Pi shape ', Pi.shape)
			print('Pj shape ', Pj.shape)

			norms = torch.norm(Pi - Pj, p = 2, dim = 1)
			print('norms shape ', norms.shape)
			S_row_i = torch.FloatTensor(S.getrow(i).toarray()[0]).to(device)
			S_sample = torch.gather(S_row_i, 0, js).to(device)
			print('S_sample shape is ', S_sample.shape)
			penalty += (norms * S_sample).sum()
			print('Penalty shape is ', penalty.shape)
			print('penalty is ', penalty.item())

		return penalty

	def similarity_matrix_penalty(self, P, S, batch, device, entity = True):
		''' For entity term: 
			For every i (= item[0]), find set of js (= nonzero values in row i in Se)
				For every j: 
					total += Se(i,j)*(E(i) - E(j)).norm(2)
		'''
		# For every entity in the batch - find a set of entities that are similar to it
		ind = 0 if entity else 1
		i_indices_numpy = np.array([item[ind] for item in batch])
		if print_time_similarity: 
			t_start = timeit.default_timer()
		S_sample_indices = S.tocsc()[i_indices_numpy, :].nonzero()[1]

		'''
		S_sample_indices = torch.LongTensor([S.getrow(item[ind]).nonzero()[1] for item in batch]).to(device)
		'''

		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to compute S_sample_indices - ", t_diff)

		
		#i_indices = torch.LongTensor(i_indices_list).to(device)
		i_indices = torch.from_numpy(i_indices_numpy).to(device)
		n_repeat = len(S_sample_indices) // len(batch)
		
		if print_time_similarity: 
			t_start = timeit.default_timer()
		Pi = P.to(device).index_select(ind, i_indices)
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to index select Pi - ", t_diff)
		#print('batchsize = ', len(batch))
		if print_time_similarity: 
			t_start = timeit.default_timer()
		j_indices_long = torch.LongTensor(S_sample_indices).to(device)
		Pj = P.to(device).index_select(ind, j_indices_long)
		#Pj = P.to(device).index_select(ind, torch.flatten(S_sample_indices))
		
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to index select Pj - ", t_diff)
		if not entity:
			Pi = Pi.transpose(0,1)
			Pj = Pj.transpose(0,1)
		Pi = Pi.repeat(1, n_repeat).view(-1, Pj.shape[1])

		if print_time_similarity: 
			t_start = timeit.default_timer()
		norms = torch.norm(Pi - Pj, p = 2, dim = 1)
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to compute norm - ", t_diff)

		if print_time_similarity: 
			t_start = timeit.default_timer()
		rows = i_indices_numpy.repeat(n_repeat)

		if print_time_similarity: 
			t_start = timeit.default_timer()
		S_sample = torch.FloatTensor(S[rows, S_sample_indices]).flatten().to(device)
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to gather S_sample - ", t_diff)
		penalty = (norms * S_sample).sum()
		return penalty
		'''
		n_repeat = S_sample_indices.shape[1]
		columns = S_sample_indices.data.cpu().numpy().flatten()
		rows = np.array(i_indices_list).repeat(n_repeat)
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to collect indices for the S matrices - ", t_diff)
		if print_time_similarity: 
			t_start = timeit.default_timer()
		S_sample = torch.FloatTensor(S[rows, columns]).flatten().to(device)
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to gather S_sample - ", t_diff)
		if print_time_similarity: 
			t_start = timeit.default_timer()
		penalty = (norms * S_sample).sum()
		if print_time_similarity:
			t_diff = (timeit.default_timer() - t_start)*100
			print("time to compute penalty - ", t_diff)

		return penalty
		'''

	def forward(self, target, prediction, lamb, batch, model, Sw, Se):
		tensor_type = model.type
		tensor_type = tc.FloatTensor
		penalty = tensor_type([0])

		if print_time:
			t_start = timeit.default_timer()
		for p in model.parameters():
			penalty += ((p > 0).type(tc.FloatTensor)*torch.clamp(p, min = 0.).type(tc.FloatTensor)).norm(2)
		if print_time:
			t_diff = (timeit.default_timer() - t_start)*100
			print('time to compute parameter penalty - ', t_diff)
		
		if print_time:
			t = timeit.default_timer()
		entity_penalty = self.similarity_matrix_penalty(model.E, Se, batch, model.device, True) 
		if print_time:
			t_diff = (timeit.default_timer() - t_start)*100
			print('time to compute similarity matrices penalty - ', t_diff)
		word_penalty = self.similarity_matrix_penalty(model.W, Sw, batch, model.device, False)

		similarity_penalty = word_penalty + entity_penalty
		
			


		#similarity_penalty += self.entity_penalty3()
		return (target.to(model.device) - prediction.to(model.device)).norm(2) + lamb*penalty.to(model.device) \
				+ similarity_penalty

	'''
	def forward(self, actual, prediction, row_ind, word_i_indices, entity_j_indices, sample_j_indices, lamb, model, Sw, Se):
		tensor_type = model.type
		penalty = 0
		for p in model.parameters():
			penalty += ((p>0).type(tensor_type)*torch.clamp(p, min=0.)).norm(2)      # norm of positive values only

		#penalty = ((model.W > 0).type(tensor_type)*torch.clamp(model.W, min = 0.)).norm(2)
		#penalty += ((model.E > 0).type(tensor_type)*torch.clamp(model.E, min = 0.)).norm(2)

		similarity_penalty = self.similarity_penalty_Sw(model.W, Sw, sample_j_indices, word_i_indices, model.device)
		
		similarity_penalty += self.similarity_penalty_Se(model.E, Se, row_ind, entity_j_indices, model.device)


		return (actual.to(model.device) - prediction).norm(2) + lamb*penalty + similarity_penalty.type(tensor_type)
	'''
def rejection_sample(item, nonzeros, total):
	negative_sample = []
	while len(negative_sample) < 5:
		randoms = np.random.randint(low = 0, high = total, size = 5 - len(negative_sample))
		negative_sample.extend([(item[0], j) for j in randoms if (item[0], j) not in nonzeros])
	return negative_sample

def rejection_sampling(n, nonzeros, already_sampled, n_rows, n_cols):
	negative_sample = []
	while len(negative_sample) < n:
		diff = n - len(negative_sample)
		random_sample = [(np.random.randint(low = 0, high = n_rows), np.random.randint(low = 0, high = n_cols)) for _ in range(diff)]
		negative_sample.extend([item for item in random_sample if (item not in nonzeros and item not in already_sampled)])
	return negative_sample


def construct_batches(n_batches, n_negative, n_words, V):
	row_inds, col_inds = V.nonzero()

	batchsize = np.int_(np.ceil(len(row_inds) / n_batches))

	row_inds, col_inds = shuffle(row_inds, col_inds)  # reshuffle both arrays

	nonzeros = [(i, j) for i, j in zip(row_inds, col_inds)]
	
	# construct positive samples
	batches = [nonzeros[i : i+batchsize] for i in range(0, len(row_inds), batchsize)]
	assert len(batches) == n_batches, "Wrong number of batches computed!"

	print('Constructed positive samples')

	# add negative samples
	already_sampled = set()

	nonzeros = set(nonzeros)
	for batch in batches:
		negative_samples = rejection_sampling(len(batch) * n_negative, nonzeros, already_sampled, V.shape[0], n_words)
		batch.extend(negative_samples)
		already_sampled.update(negative_samples)
	print('Constructed negative samples')
	return batches


def get_Sw_sample(Sw, row_inds, col_inds):
	''' For each index in row_inds, collect all elements from that col '''
	print(row_ind)
	print()
	print(col_inds)

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

def get_target_values(batch, V):
	row_indices = [i[0] for i in batch]
	col_indices = [i[1] for i in batch]
	return torch.FloatTensor(V[row_indices, col_indices])
	#return torch.cat([V[i].unsqueeze(0) for i in batch])



def custom_nmf(V, Sw, Se, params):

	n_words = Sw.shape[0]
	n_entities = Se.shape[0]

	print('V shape: ', V.shape)
	print('n entities - ', n_entities)
	print('n words - ', n_words)

	losses = []

	# define model


	model = NMF(n_entities, n_words, params.n_features, params)

	loss = CustomLoss(params.device)
	optimizer = torch.optim.SGD(model.parameters(), lr = params.lr) 
	n_epochs = params.n_epochs
	lam = 10 # TODO: clean this up
	

	# construct tensors
	V_tensor = torch.Tensor(V.todense().astype(np.float32))
	V_var = Variable(V_tensor, requires_grad = False)

	losses = []
	differences = []
	for epoch in range(n_epochs):
		# construct batch = one row, n positive samples, 3n negative samples
		row_permutation = np.random.permutation(n_entities)
		total_loss = 0 
		batches = construct_batches(params.n_batches, params.n_negative, n_words, V)
		bcount = -1
		for batch in batches:
			bcount += 1
			model.zero_grad()

			#print('On batch ', bcount)
			if print_time:
				# construct sample
				t_start = timeit.default_timer()
			target = get_target_values(batch, V)
			
			if print_time: 
				t_diff = (timeit.default_timer() - t_start)*100
				print('Time to collect target - ', t_diff)


			if print_time: t_start = timeit.default_timer()
			prediction = model(batch)
			
			if print_time: 
				t_diff = (timeit.default_timer() - t_start)*100
				print('Time to compute prediction - ', t_diff)

			if print_time: t_start = timeit.default_timer()
			l = loss(target, prediction, lam, batch, model, Sw, Se)
			
			if print_time: 
				t_diff = (timeit.default_timer() - t_start)*100
				print('Time to compute loss - ', t_diff)



			'''
			sample_indices, word_i_indices, entity_j_indices = get_batch_indices(V, Sw, Se, row_ind, params.device, n_entities)
			#sample_values, sample_columns, word_i_indices, entity_j_indices = generate_batch(V, Sw, Se, row, n_entities)

			target = torch.gather(V_var[row_ind], 0, sample_indices)

			optimizer.zero_grad()
			prediction = model(row_ind, sample_indices)
	

			l = loss(target, prediction, row_ind, word_i_indices, entity_j_indices, sample_indices, lam, model, Sw, Se)
			'''
			losses.append(l.item())
			l.backward(retain_graph = True)

			optimizer.step()
			model.positivise()
			total_loss += l.item()
			torch.cuda.empty_cache()

		# test the model after each epoch
		if epoch % 50 == 0:
			t_start = timeit.default_timer()
			V_prediction = torch.mm(model.E, model.W)   
			t_diff = timeit.default_timer() - t_start
			print('time to compute difference - ', t_diff * 100)
			difference = (V_tensor - V_prediction).norm(2).item()
			differences.append(difference)
			print('Epoch - {}, error - {}'.format(epoch, difference))

		total_loss /= params.n_batches   
		losses.append(total_loss)
		print('Epoch {}: current loss - {}'.format(epoch, total_loss))
		print()

	# plot loss and difference
	plt.plot(differences)
	plt.xlabel("Epoch"); plt.ylabel("Error")
	plt.title("Difference between actual matrix and computed matrix")
	plt.show()

	plt.plot(losses)
	plt.xlabel("Epoch"); plt.ylabel("Loss")
	plt.title("Loss function")
	plt.show()



			



" QUESTIONS "
'''


4. Keep similarity threshold for entities as well?


'''


