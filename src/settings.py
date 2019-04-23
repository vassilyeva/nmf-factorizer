""" Project settings """
from nltk.corpus import wordnet as wn
from gensim.models import FastText
from gensim.models import Word2Vec

from tempfile import mkdtemp
import os.path as path
import torch.cuda as tc
import torch




class Settings:
	def __init__(self):
		self.test = False


	def set(self, args):
		# set up input filenames
		if args.dataset == 'test':
			self.dataset_text 		= '../datasets/Test/entityWords.txt'
			self.dataset_vocab 		= '../datasets/Test/word2id.txt'
			self.dataset_transe 	= '../datasets/Test/entity2vec.' + args.transe_method
			self.test 			= True
			#self.n_batches = 10
		elif args.dataset == 'freebase':
			self.dataset_text 	= '../datasets/Freebase15k/entityWords.txt'
			self.dataset_vocab 		= '../datasets/Freebase15k/word2id.txt'
			self.dataset_entities = '../datasets/Freebase15k/train.txt'
			self.dataset_transe 	= '../datasets/Freebase15k/entity2vec.' + args.transe_method

		# set up device
		use_gpu = True if (args.device == 'default' and torch.cuda.is_available \
							or args.device == 'gpu') else False

		if use_gpu:
			self.device = torch.device('cuda:0')
			self.floatType = torch.cuda.FloatTensor
			self.longType = torch.cuda.LongTensor
		else:
			self.device = torch.device('cpu')
			self.floatType = torch.FloatTensor
			self.longType = torch.LongTensor

		# set up word embedding model
		if args.word_model == 'word2vec':
			self.emb_model = Word2Vec
		else:
			self.emb_model = FastText

		self.incr = args.incr
		self.similarity = args.similarity
		self.transe_method = args.transe_method
		self.n_similar = args.n_similar
		self.use_idf        = args.use_idf


'''
Hyperparams settings for NMF
'''
import torch

class Hyperparameters:
	def __init__(self):
		self.lambdas = [.01, .01, .2]  # regularizer params for Se, Sw, and model parameters
		self.momentum = .4

	def set(self, args):
		self.n_epochs = args.n_epochs
		self.lr = args.lr
		self.n_batches = args.n_batches
		self.n_features = args.n_features
		self.n_negatives = args.n_negatives
		self.valid = args.valid


		if args.optim == 'sgd':
			self.optim = torch.optim.SGD
			self.optim_settings = {'lr': self.lr, 'momentum': self.momentum}

		elif args.optim == 'adam':
			self.optim = torch.optim.Adam
			wd = .001
			self.optim_settings.update({'weight_decay:', wd})
		else: 		# adagrad
			self.optim = torch.optim.Adagrad
			lr_decay = .001
			self.optim_settings.update({'lr_decay': lr_decay})

	def print_hp(self):
		settings = ("Optimizer: {}, l_rate: {}, # epochs: {}," \
			" # batches: {}".format(self.optim.__name__, \
				self.lr, self.n_epochs, self.n_batches ))
		print(settings)

	def print_lambdas(self):
		lambdas_string = ('Regularization params: Se - {}, ' \
						  'Sw - {}, P - {}'.format(*self.lambdas))
		print(lambdas_string)



