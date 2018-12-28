
import argparse
from settings import Settings
from nmf import custom_nmf
from nltk.corpus import wordnet as wn

import torch

import subprocess
import os


def parse_args():
	parser = argparse.ArgumentParser(description = "NMF Factorization parameters")
	parser.add_argument('--similarity', default = 'cos', choices = ['wn-path', 'wn-wup', 'cos'], help = 'type of similarity between words - either Wordnet based or cosine')
	parser.add_argument('--data_text')     # default is freebase
	parser.add_argument('--data_entities')
	parser.add_argument('--transe_method', default = 'bern', choices = ['bern', 'unif'])
	parser.add_argument('--device', default = 'default', choices = ['default', 'gpu', 'cpu'])
	parser.add_argument('--use_idf', default = True, type = bool)
	parser.add_argument('--incr', default = 1000, type = int)

	# NN parameters
	parser.add_argument('--lambda', default = 10, type = float)
	parser.add_argument('--nepochs', default = 1000, type = int)

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
		params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		params.torch_type = torch.FloatTensor
	elif args.device == 'gpu':
		params.device = torch.device('cuda:0')
		params.torch_type = torch.cuda.FloatTensor
	else:
		params.device = torch.device('cpu')
	params.incr = args.incr




if __name__ == "__main__":
	params = Settings()   # get model parameters
	args = parse_args()
	set_params(params, args)

	'''
	params.transe_method = args.transe_method
	params.transe_dataset = '../datasets/entity2vec.' + args.transe_method
	if not os.path.isfile(params.transe_dataset):     # vector embeddings for entities does not exist
		# TODO: RUN THE C++ EXECUTABLE   ???
	'''


	W, H = custom_nmf(params)   # computes the custom nmf params


