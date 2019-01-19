
import argparse
from settings import Settings
from hyperparams import Hyperparameters
from nmf import custom_nmf
from nltk.corpus import wordnet as wn

import torch

import subprocess
import os

import os.path
import dataprocessing as dp

import numpy as np
from scipy import sparse




def parse_args():
	parser = argparse.ArgumentParser(description = "NMF Factorization parameters")

	parser.add_argument('--similarity', default = 'cos', choices = ['wn-path', 'wn-wup', 'cos'], help = 'type of similarity between words - either Wordnet based or cosine')
	parser.add_argument('--transe_method', default = 'bern', choices = ['bern', 'unif'])
	parser.add_argument('--device', default = 'default', choices = ['default', 'gpu', 'cpu'])
	parser.add_argument('--use_idf', default = True, type = bool)
	parser.add_argument('--incr', default = 1000, type = int)
	parser.add_argument('--sim-matrix', default = 'compute', choices = ['compute', 'fromfile'], help = 'compute similarity matrices or load them from file')
	parser.add_argument('--n-similar', type = int, default = 20, help = 'Number of top similar words/entities to consider for Sw/Se matrices')
	parser.add_argument('--word-model', default = 'word2vec', choices = ['word2vec', 'fasttext'])

	# NMF parameters
	parser.add_argument('--lr', default = .001, type = float, help = 'learning rate')
	parser.add_argument('--optim', default = 'sgd', choices = ['sgd', 'adam', 'adagrad'])
	parser.add_argument('--n-epochs', type = int, default = 250, help = 'Number of epochs for optimizer')
	parser.add_argument('--n-batches', type = int, default = 100, help = 'Number of batches')
	parser.add_argument('--n-features', type = int, default = 100, help = 'Number of features in embedding')
	parser.add_argument('--n-negatives', type = int, default = 5, help = 'Number negative samples for each positive one')


	# TEST ARGUMENTS
	parser.add_argument('--dataset', default = 'freebase', choices = ['freebase', 'test'])

	args = parser.parse_args()
	return args


def generate_tfidf_matrix(filename):
	tfidf_transformer = TfidfTransformer()

	corpus = pd.read_csv(dataset_file, sep = '\t', header = None)
	remove_digits = str.maketrans('', '', digits)
	corpus = [c.translate(remove_digits) for c in corpus.loc[:, 1]]
	corpus = [c[:-3] if c[-3:] == '@en' else c for c in corpus]
	vectorizer = CountVectorizer()
	TF_matrix = vectorizer.fit_transform(corpus[:4])
	words = vectorizer.get_feature_names()

def set_params(params, args):
	if args.dataset == 'test':
		params.dataset_text 	= '../datasets/Test/entityWords.txt'
		params.dataset_vocab 		= '../datasets/Test/word2id.txt'
		params.dataset_transe = '../datasets/Test/entity2vec.' + args.transe_method	
		params.test = True	
		params.n_batches = 10
	elif args.dataset == 'freebase':
		params.dataset_text 		= '../datasets/Freebase15k/entityWords.txt'
		params.dataset_entities 	= '../datasets/Freebase15k/train.txt'
		params.dataset_transe = '../datasets/Freebase15k/entity2vec.' + args.transe_method
	params.use_idf = args.use_idf
	params.similarity = args.similarity

	if args.similarity == 'wn-path':
		params.similarity_function = wn.path_similarity
		params.h5filename 		   = 'similarity_wordnet_path_' + args.dataset + '.h5'
		params.h5omega			   = 'omega_wordnet_path_' + args.dataset + '.h5'
	elif args.similarity == 'wn-wup':
		params.similarity_function = wn.wup_similarity
		params.h5filename		   = 'similarity_wordnet_wup_' + args.dataset + '.h5'
		params.h5omega 			   = 'omega_wordnet_wup_' + args.dataset + '.h5'
	else: 
		params.h5filename		   = 'similarity_word2vec_' + args.dataset + '.h5'
		params.h5omega			   = 'omega_word2vec_' + args.dataset + '.h5'

	if  args.device == 'default':
		has_cuda = True if  torch.cuda.is_available() else False
		if has_cuda:
			params.device = torch.device("cuda:0")
			params.floatType = torch.cuda.FloatTensor
			params.longType = torch.cuda.LongTensor
		else:
			params.device = torch.device("cpu")
			params.floatType = torch.FloatTensor
			params.longType = torch.LongTensor
	elif args.device == 'gpu':
		params.device = torch.device('cuda:0')
		params.floatType = torch.cuda.FloatTensor
		params.longType = torch.cuda.LongTensor
	else:
		params.device = torch.device('cpu')
		params.floatType = torch.FloatTensor
		params.longType = torch.LongTensor


	if args.n_epochs:
		params.n_epochs = args.n_epochs
	if args.n_batches:
		params.n_batches = args.n_batches
	params.incr = args.incr
	params.opt = args.optimizer
	params.lr = args.lr




if __name__ == "__main__":
	print('Now without dividing total loss by n_batches')
	params = Settings()   # get model parameters
	args = parse_args()
	params.set(args)
	print('Embedding model - ' + args.word_model)


	hp = Hyperparameters()
	hp.set(args)
	#set_params(params, args)

	

	if args.sim_matrix == 'fromfile':
		V = sparse.load_npz('V_'+args.dataset+'.npz')
		print('Loaded TFIDF Matrix')
		Sw = sparse.load_npz('Sw_'+args.dataset+'.npz')
		print('Loaded Sw matrix')
		Se = sparse.load_npz('Se_' + args.dataset + '.npz')
		print('Loaded Se matrix')

	else:
		V, sentences, words = dp.compute_tfidf_matrix(params)
		print('TFIDF Shape - ', V.shape)
		if params.similarity == 'cos':
			V, Sw = dp.compute_Sw_cosine(params, V, sentences, words, args.dataset)
		else: 
			Sw = dp.compute_wordnet_similarity_words(params, words)
		print('Constructed Sw matrix, shape - {}'.format(Sw.shape))

		#S_matrix = np.memmap(params.similarity_matrix_filename, dtype = 'float32', mode = 'r')
		Se = dp.compute_Se_cosine(params, args.dataset)
		print('Constructed Se matrix')
	'''
	hp.print_hp()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	hp.optim.__name__ + str(hp.lr)[2:]
	#print('Learning rate - {}, n batches - {}, n epochs - {}'.format(hp.lr, hp.n_batches, hp.n_epochs))
	#density_column = np.max(V.sum(0).A1)/V.shape[0]
	#density_row = np.max(V.sum(1).A1)/V.shape[1]
	W, E = custom_nmf(V, Sw, Se,  params, hp, title)   # computes the custom nmf params

	print()
	'''
	# reset hyperparameters
	hp.lr = .01
	hp.optim = torch.optim.Adagrad
	hp.optim_settings = {'lr': hp.lr, 'lr_decay': .001}
	hp.print_hp()
	print('lr decay - ', .001)
	hp.print_lambdas()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	W, E = custom_nmf(V, Sw, Se, params, hp, title)
	print()

	hp.lambdas = [.3, .3, .9]
	hp.print_hp()
	print('lr decay - ', .001)
	hp.print_lambdas()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	W, E = custom_nmf(V, Sw, Se, params, hp, title)
	print()

	hp.lambdas = [5, 10, 7]
	hp.print_hp()
	print('lr decay - ', .001)
	hp.print_lambdas()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	W, E = custom_nmf(V, Sw, Se, params, hp, title)
	print()

	hp.lambdas = [.5, .5, .5]
	hp.lr = .01
	hp.optim = torch.optim.Adagrad
	hp.optim_settings = {'lr': hp.lr, 'lr_decay': .01}
	hp.print_hp()
	print('lr decay - ', .01)
	hp.print_lambdas()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	W, E = custom_nmf(V, Sw, Se, params, hp, title)

	print()
	hp.lr = .001
	hp.optim = torch.optim.Adam 
	hp.optim_settings = {'lr': hp.lr}
	hp.print_hp()
	hp.print_lambdas()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	W, E = custom_nmf(V, Sw, Se, params.hp, title)


	print()
	hp.lambdas = [3, 3, .5]
	hp.print_hp()
	hp.print_lambdas()
	title = '_'.join([hp.optim.__name__, str(hp.lr)[2:], *[str(l) for l in hp.lambdas]]) + '.png'
	W, E = custom_nmf(V, Sw, Se, params.hp, title)





