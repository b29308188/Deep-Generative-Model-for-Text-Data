import logging
from gensim.models import word2vec
import re


def read_data(data_path):
	with open(data_path, "r") as f:
		sentences = []
		for i, line in enumerate(f):
			sentences.append(line.strip().split())
	return sentences

if __name__ == "__main__":
	sentences = read_data("../data/IMDB.train.corpus")
	
	#logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 	
	#train word vectors
	model = word2vec.Word2Vec(sentences, size=32, min_count = 3, workers = 24, window = 10, sg  = 0) # CBOW
    
	#store the model
	model.save('../models/word2vec.mod')

