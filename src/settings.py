""" Project settings """
from nltk.corpus import wordnet as wn

from tempfile import mkdtemp
import os.path as path
 

class Settings:
	def __init__(self):
		# Dataset
		self.dataset_text 		= '../datasets/Freebase15k/entityWords.txt'
		#self.dataset_entities 	= '../datasets/Freebase15k/train.txt'
		self.dataset_vocab 		= '../datasets/Freebase15k/word2id.txt'

		# Word similarity matrix
		self.use_idf        = True
		self.similarity     = 'cos'
		self.similarity_threshold = 0.8

		self.similarity_matrix_filename = path.join(mkdtemp(), 'similarity_matrix.dat')

		self.similarity_function = wn.path_similarity

		self.h5filename = 'similarity_matrix_word2vec.h5'
		self.h5omega	= 'omega_matrix_word2vec.h5'

		# entity embeddings settings
		self.transe_method = 'bern'   # bern or unif
		self.dataset_transe = '../datasets/entity2vec.' + self.transe_method

		# PYTORCH settings
		self.device = None

		# TEMPORARY SETTINGS:
		self.use_pytorch_entities = True
		self.test = False


		# NMF learning parameters
		self.lr = .0001
		self.n_epochs = 1000