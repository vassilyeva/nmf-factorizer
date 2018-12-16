import os.path
import dataprocessing as dp

import numpy as np
from scipy import sparse

import tables as tb
import torch


def custom_nmf(params):
	print(params.dataset_text)
	# TFIDF matrix
	V, sentences, words = dp.compute_tfidf_matrix(params)

	if params.similarity == 'cos' and not os.path.isfile(params.h5omega):
		V = dp.compute_cosine_similarity_words(params, V, sentences, words)
	elif not os.path.isfile(params.h5omega): 
		dp.compute_wordnet_similarity_words(params, words)

	#S_matrix = np.memmap(params.similarity_matrix_filename, dtype = 'float32', mode = 'r')

	S = dp.compute_entity_matrix(params)

	# similarity matrix

" QUESTIONS "
'''
1. Some of the combination words are not included in wordnet. What is more important - keeping them together
   and computing similarity between embeddings or using wordnet similarity?

2. Some words do not make it into model.vocabulary for some odd reason


3. Some words are in vocabulary list, but are not actually in the sentence; if i try setting their vector to all zeros, 
(for embedding), cosine similarity breaks for some reason
So, I am removing them from matrix V

4. Keep similarity threshold for entities as well?


5. BIG ONE: should I use TransE output (from Train_TransE) or should I use the pretrained model for generating embeddings again? 
   Where is the damn model?
'''


