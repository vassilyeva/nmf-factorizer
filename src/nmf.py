from collections import Counter
import dataprocessing as dp
import imp
from sklearn import decomposition
from dataloader import Dataloader

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


''' MODEL '''

class NMF(torch.nn.Module):
	def __init__(self, n_entities, n_words, n_features, initE, initW, params):
		super().__init__()
		#W_data = Variable(torch.FloatTensor(n_features, n_words))
		#E_data = Variable(torch.FloatTensor(n_entities, n_features))
		W_data = Variable(torch.from_numpy(initW).type(torch.FloatTensor))
		E_data = Variable(torch.FloatTensor(initE))

		#W_data.uniform_(0, 1).abs_()
		#E_data.uniform_(0, 1).abs_()
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
		self.W.data = torch.clamp(self.W, min = 0.001)
		self.E.data = torch.clamp(self.E, min = 0.001)

	def print_params(self):
		print("W: ")
		for row in self.W:
			print(row.data)
		print()
		print('E')
		for row in self.E:
			print(row.data)

class CustomLoss(torch.nn.Module):
	def __init__(self, device):
		super(CustomLoss, self).__init__()
		self.device = device
		self.float_type = tc.FloatTensor if torch.cuda.is_available else torch.FloatTensor

	def penalize(self, A):
		return (A>0).type(float_type)*torch.clamp(A, max=0.)



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
		i_indices = torch.LongTensor(i_indices_numpy).to(device)
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


	def forward(self, target, prediction, reg, batch, model, Sw, Se):
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

		#similarity_penalty = regword_penalty + entity_penalty
		difference = (target.to(model.device) - prediction.to(model.device)).norm(2)
		loss = difference + \
			    reg[0]*entity_penalty + reg[1]*word_penalty + reg[2]*penalty.to(model.device)
		return loss




		#similarity_penalty += self.entity_penalty3()
		return (target.to(model.device) - prediction.to(model.device)).norm(2) + lamb*penalty.to(model.device) \
				+ similarity_penalty



def plot_results(losses, differences, n_epochs, model, title):
	# plot loss and difference
	plt.plot(list(range(0, n_epochs, 50)) + [n_epochs - 1], differences, label = '|V - EW|')
	plt.xlabel("Epoch"); plt.ylabel("Error")
	plt.title(title[:-4])

	plt.plot(losses, label = 'Loss')
	plt.xlabel("Epoch")
	plt.legend()
	plt.show()

	plt.clf()
	plt.plot(losses)
	plt.title('Loss')
	plt.xlabel("Epoch"); plt.ylabel('Loss')
	plt.show()

	plt.clf()
	plt.plot(list(range(0, n_epochs, 50)) + [n_epochs - 1], differences)
	plt.title('Difference ||V - EW||')
	plt.xlabel("Epoch"); plt.ylabel('Difference')
	plt.show()



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

def check_densities(V):
	rows, cols = V.nonzero()
	row_counter, col_counter = Counter(rows), Counter(cols)
	row_max = max(list(row_counter.values()))
	col_max = max(list(col_counter.values()))

	return True if row_max/V.shape[1] <= col_max/V.shape[0] else False

def init_values(V, n_features):
	mdl = decomposition.NMF(n_components = n_features)
	initE = mdl.fit_transform(V)
	initW = mdl.components_
	return initE, initW

def custom_nmf(V, Sw, Se, params, hyperparams, title, initE, initW):
	print('Without transfer learning')
	print('with momentum and lr scheduler')

	# decide where to draw negative samples from - entities (= for every (i,j) positive, find (i, *) neg) or words
	'''density_column = np.max(V.sum(0).A1)/V.shape[0]
	density_row = np.max(V.sum(1).A1)/V.shape[1]
	from_entities = True if density_row <= density_column else False
	'''

	from_entities = check_densities(V)
	V, n_negatives, n_batches, from_entities, valid
	dl = Dataloader(V, hyperparams.n_negatives, hyperparams.n_batches, from_entities, hyperparams.valid)

	n_words = Sw.shape[0]
	n_entities = Se.shape[0]

	losses = []

	# define model
	'''
	initE, initW = init_values(V, hyperparams.n_features)
	np.save('initE.npy', initE)
	np.save('initW.npy', initW)
	'''
	initW = np.load('initW.npy')
	initE = np.load('initE.npy')

	initW = np.abs(initW)
	initE = np.abs(initE)
	print(initW.shape)
	'''
	initW = np.random.uniform(0, 10, size = (n_words, hyperparams.n_features))
	initE = np.random.uniform(0, 10, size = (n_entities, hyperparams.n_features))
	'''
	model = NMF(n_entities, n_words, hyperparams.n_features, initE, initW, params)
	initW = None; initE = None
	print('Created NMF model')

	loss = CustomLoss(params.device)
	optimizer = hyperparams.optim(model.parameters(), **hyperparams.optim_settings)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = .1)


	# construct tensors
	V_tensor = torch.Tensor(V.todense().astype(np.float32))
	V_var = Variable(V_tensor, requires_grad = False)

	losses = []
	differences = []
	for epoch in range(hyperparams.n_epochs):
		# construct batch = one row, n positive samples, 3n negative samples
		total_loss = 0
		batches = construct_batches(hyperparams.n_batches, hyperparams.n_negatives, V, from_entities)
		bcount = -1
		for batch in batches:
			bcount += 1
			optimizer.zero_grad()

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
			l = loss(target, prediction, hyperparams.lambdas, batch, model, Sw, Se)

			if print_time:
				t_diff = (timeit.default_timer() - t_start)*100
				print('Time to compute loss - ', t_diff)


			l.backward(retain_graph = True)

			optimizer.step()
			model.positivise()
			total_loss += l.item()

		# test the model after each epoch
		#total_loss /= hyperparams.n_batches
		losses.append(total_loss)
		print('Epoch - {}: loss - {}'.format(epoch, total_loss))
		if epoch % 50 == 0 or epoch == hyperparams.n_epochs - 1:
			t_start = timeit.default_timer()
			V_prediction = torch.mm(model.E, model.W)
			t_diff = timeit.default_timer() - t_start
			#print('time to compute difference - ', t_diff * 100)
			diff = (V_tensor - V_prediction).norm(2).item()
			differences.append(diff)
			print('\t\t \t \t difference - {}'.format(diff))

		scheduler.step()
	plot_results(losses, differences, hyperparams.n_epochs, model, title)

	return model.E, model.W





" QUESTIONS "
'''


4. Keep similarity threshold for entities as well?


'''


