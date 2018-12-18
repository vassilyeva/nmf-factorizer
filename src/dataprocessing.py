import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec

import tables as tb
import torch


''' which datastruct to use for word similarity matrix:
choices = memmap, tables, sparse (for sparse csr matrix)
'''
method = 'tables'   
use_memmap = True
use_tables = False
without_pytorch = True
 
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def dropcols_fancy(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(np.arange(M.shape[1]), idx_to_drop, assume_unique=True)
    return M[:, np.where(keep)[0]]

""" COMPUTE INPUT MATRICES """
# TF Matrix
def compute_tfidf_matrix(params):

	vocab = {}
	with open(params.dataset_vocab) as vocab_file:
		vocab = {word : int(idx) for line in vocab_file for (word, idx) in (line.strip().split(), ) }
		
	corpus = pd.read_csv(params.dataset_text, sep = '\t', header = None)

	sentences = corpus.loc[:, 2]
	
	vectorizer = TfidfVectorizer(use_idf = params.use_idf, vocabulary = vocab)
	V = vectorizer.fit_transform(corpus.loc[:, 2])
	print('Finished constructing TFIDF matrix')

	words = vectorizer.get_feature_names()


	return V.transpose().tocsr(), sentences, words
	


# compute cosine similarity between vectors using wordnet functions
def compute_similarity(params, synset_pair):
	synsets1, synsets2 = synset_pair
	similarities_vector = [params.similarity_function(syn1, syn2) for syn1 in synsets1 for syn2 in synsets2]
	try: 
		max_sim = max(similarities_vector)
	except:
		max_sim = 0
	return 0 if math.isnan(max_sim) or max_sim <= params.similarity_threshold  else max_sim

	

def compute_entity_matrix(params):
	# load embedding vectors
	entity_embeddings = []
	with open(params.dataset_transe, 'r') as fin:
		for line in fin:
			embedding = line.split('\t')
			assert embedding[-1] == '\n', 'ONE LINE DOES NOT HAVE ENDOFLINE'
			entity_embeddings.append([float(elem) for elem in embedding[:-1]])
	entity_embeddings = np.array(entity_embeddings)
	size = len(entity_embeddings)

	nzeros = 0 # rm
	if not params.use_pytorch_entities:
		S = cosine_similarity(entity_embeddings)
		S[S <= params.similarity_threshold] = 0
		nzeros += (S == 0).sum() # rm
		S_sparse = sparse.lil_matrix(S)
	else: 
		embeddings = torch.tensor(entity_embeddings).to(params.device)
		embeddings_norm = embeddings / embeddings.norm(dim = 1)[:, None]
		S_sparse = sparse.lil_matrix((size, size))
		for i in range(0, size, params.incr):
			print('On row ', i)
			end = min(i+params.incr, size)
			S_part = torch.mm(embeddings_norm, embeddings_norm[i:end].transpose(0,1))
			S_part[S_part <= params.similarity_threshold] = 0
			nzeros += (S_part == 0).sum()  # rm
			# TODO; QUESTION: should I sparsify this as well by checking against threshold??
			try:
				S_sparse[:, i:end] = S_part.data.numpy()
			except: 
				S_sparse[:, i:end] = S_part.cpu().data.numpy()

		print('N zeros in entity to entity - ', nzeros)
		''' without for loop 
		S = torch.mm(embeddings_norm, embeddings_norm.transpose(0, 1))

		S[S <= params.similarity_threshold] = 0
		try: 
			S_sparse = sparse.csr_matrix(S.data.numpy())
		except:
			S_sparse = sparse.csr_matrix(S.cpu().data.numpy())
		'''
	return S_sparse

def construct_Sw(params, entity_embeddings):
	embedding_len = len(entity_embeddings)
	h5file = tb.open_file(params.h5filename, 'w')
	S = h5file.create_carray(h5file.root, 'similarity', tb.Float32Atom(), shape = (embedding_len, embedding_len))
	#S = torch.zeros(len(entity_embeddings), len(entity_embeddings))
	embeddings = torch.tensor(entity_embeddings).to(params.device)
	embeddings_norm = embeddings / embeddings.norm(dim = 1)[:, None]

	nzeros = 0

	if without_pytorch:
		for i in range(0, embeddings.shape[0], params.incr):
			print('On row ', i)
			end = min(i+params.incr, embeddings.shape[0])
			S_part = cosine_similarity(embeddings, embeddings[i:end])
			S[:, i:end] = S_part

	for i in range(0, embeddings.shape[0], params.incr):
		print('On row ', i)
		end = min(i+params.incr, embeddings.shape[0])
		S_part = torch.mm(embeddings_norm, embeddings_norm[i:end].transpose(0,1))
		S_part[S_part <= params.similarity_threshold] = 0   # sparsify
		nzeros += (S_part == 0).sum().data 
		try:
			S[:, i:end] = S_part.data.numpy()
		except: 
			S[:, i:end] = S_part.cpu().data.numpy()
	print('Sw matrix number zeros - {}; proportion - {}'.format(nzeros, nzeros/embedding_len**2))
	S.flush()
	h5file.close()

def construct_Omega(params, size):
	''' computes L - Sw '''
	#Omega = tensor.zeros(size, size).to(params.device)
	h5omega = tb.open_file(params.h5omega, 'w')
	Omega = h5omega.create_carray(h5omega.root, 'omega', tb.Float32Atom(), shape = (size, size))

	h5sim = tb.open_file(params.h5filename, 'r')   # Sw matrix
	nzeros = 0

	for i in range(size):
		L = np.zeros(size)
		L[i] = h5sim.root.similarity[i].sum()
		Omega[i] = L - h5sim.root.similarity[i]
		nzeros += (Omega[i] == 0).sum()
	print('Omega matrix number zeros - {}; proportion = {}'.format(nzeros, nzeros/size**2))
	h5omega.close()
	h5sim.close()






def compute_cosine_similarity_words(params, tfidf_matrix, documents, words):
	pytorch = 'with pytorch' if without_pytorch == False else 'without pytorch'
	print("Computing "+pytorch + ' using '+method + 'increment - ', params.incr)
	model = Word2Vec([[word for word in sentence.split()] for sentence in documents], size = 100, min_count = 0)

	embedding_len = len(list(model.wv.vocab.values()))
	vocab = list(model.wv.vocab.keys())

	missing_words = set(words) - set(vocab)

	to_drop = [i for i, j in enumerate(words) if j in missing_words]
	words = [w for w in words if w not in missing_words]   # remove missing words from list of words
	print('Total number words: ', len(words))

	#tfidf_matrix = dropcols_fancy(tfidf_matrix, to_drop)
	tfidf_matrix = delete_rows_csr(tfidf_matrix, to_drop)
	print('Removed empty rows from TFIDF matrix')

	# compute similarity matrix
	embeddings = [model.wv[word] for word in words]
	print('Constructed embeddings matrix')

	n_rows = len(embeddings)
	# use memmap
	if method == 'memmap':
		filename = params.similarity_matrix_filename
		mem = np.memmap(filename, dtype = np.float32, mode = 'w+', shape = (n_rows, n_rows))

		for i, elem in enumerate(embeddings):
			if i % 1000 == 0:
				print("Finished {}'th row".format(i))
			sim_row = cosine_similarity(embeddings, elem.reshape(1, -1)).reshape(1, -1)
			mem[i, :] = sim_row
	if method == 'tables':
		construct_Sw(params, embeddings)
		print('Word-to-word similarity matrix - constructed')
		construct_Omega(params, embedding_len)
		print('Omega matrix constructed')
		
		'''
		h5file = tb.open_file('similarity_matrix.h5', 'w')
		data = h5file.create_carray(h5file.root, 'similarity', tb.Float32Atom(), shape = (embedding_len, embedding_len))
		for i, elem in enumerate(embeddings):
			if i % 1000 ==0: 
				print('Finished {}th row'.format(i))
			sim_row = cosine_similarity(embeddings, elem.reshape(1, -1)).reshape(1, -1)
			sim_row[sim_row <= params.similarity_threshold] = 0
			data[i] = sim_row
		'''

	if method == 'sparse':
		zeros_old, zeros_new = 0, 0
		nonzeros_new = 0
		sim_matrix = sparse.lil_matrix((n_rows, n_rows))
		for i, elem in enumerate(embeddings):
			sim_row = cosine_similarity(embeddings, elem.reshape(1, -1)).reshape(1, -1)
			
			zeros_old += (sim_row == 0).sum()
			sim_row[sim_row <= params.similarity_threshold] = 0
			if i % 1000 == 0:
				print('finished {}th row; zeros before - {}, zeros now - {}, nonzeros - {}'.format(i, zeros_old, zeros_new, nonzeros_new))
				print('nonzeros in this row: ', (sim_row != 0).sum())
			zeros_new += (sim_row == 0).sum()
			nonzeros_new += (sim_row != 0).sum()
			sim_matrix[i] = sim_row
		print('zeros before: ', zeros_old, '; zeros now: ', zeros_new)

		print('Finished mem matrix construction')
		del mem
	return tfidf_matrix


def compute_wordnet_similarity_words(params, words):
	all_synsets = [wn.synsets(word) for word in words]
	synset_combinations = itertools.combinations(all_synsets, 2)
	similarities_vector = [compute_similarity(params, word_pair) for word_pair in synset_combinations]
	return sparse.csr_matrix(similarities_vector)