import sys
import re
import numpy as np
from scipy import spatial
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.optimizers import RMSprop
#from seq2seq.models import SimpleSeq2Seq
from gensim import models, matutils

#w2v_model = None
#w2v_model download path: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

def normalize(word):
	word  = re.sub(r'[^\w\s]' , "", word.decode("utf-8"), re.UNICODE)
	word = word.lower()
	return word

def read_data(file_path, max_len = 30, dimension = 300):
	X = []
	sentences = []
	with open(file_path, "r") as f:
		for line in f:
			x = np.zeros((max_len, dimension))
			sentence = []
			tokens = line.strip().split()
			index = 0
			for token in map(normalize, tokens):
				if token in w2v_model:
					x[index] = w2v_model[token]
					index += 1
					sentence.append(token)
				if index >= max_len:
					break
			X.append(x)
			sentences.append(sentence)
	return np.array(X), dimension, sentences

def build_generative_model(input_dim, max_len = 30):
	G = Sequential()
	G.add(LSTM(128, input_dim = input_dim, input_length = max_len, return_sequences = True))
	G.add(Dropout(0.25))
	G.add(TimeDistributed(Dense(input_dim)))###Shoud be sequence to sequence?
	G.add(Activation("relu"))
	#print G.summary()
	return G

def build_discriminative_model(input_dim, max_len = 30):
	D = Sequential()
	D.add(LSTM(128, input_dim = input_dim, input_length = max_len, return_sequences = False))
	D.add(Dropout(0.25))
	D.add((Dense(30)))
	D.add(Activation("relu"))
	D.add(Dropout(0.25))
	D.add(Dense(1))
	D.add(Activation("relu"))
	#print G.summary()
	return D

def build_GAN(G, D):
	model = Sequential()
	model.add(G)
	D.trainable = False
	model.add(D)
	return model

def generate_noise(X):
	#noise = np.random.normal(0, 1, size = X.shape)
	#return X+noise
	#return np.random.uniform(size = X.shape)
	return X + np.random.normal(0, np.std(X), size = X.shape)

def find_nearest_vocab(sentences, gen_X):
	print "# sentences: ", len(sentences)
	gen_sentences = []

	for sentence, gen_x in zip(sentences, gen_X):
		gen_tokens = []
		for token, token_vec in zip(sentence, gen_x):
			nearest = w2v_model.most_similar(positive=[token], topn=10000)
			index = -1
			min_dist = 100000
			for i, nbr in enumerate(nearest):
				try:
					nbr[0].encode('ascii')
				except UnicodeEncodeError:
					continue
				d = spatial.distance.cosine(token_vec, w2v_model[nbr[0]])
				if d < min_dist:
					index = i
					min_dist = d
			if index == -1:
				gen_tokens.append('')
			else:
				gen_tokens.append(nearest[index][0])
			# distances = [spatial.distance.cosine(token_vec, w2v_model[nbr]) for nbr, foo in nearest]
			# gen_tokens.append(nearest[distances.index(min(distances))][0])
		print ' '.join(gen_tokens)
		gen_sentences.append(' '.join(gen_tokens))

	return gen_sentences
		

if __name__ == "__main__":
	print "Loading word vectors ..."
	#w2v_model = models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
	
	print "Reading text data ..."
	X, dimension, sentences = read_data("../data/brown.txt")
	print "Building model ..."
	G = build_generative_model(dimension)
	D = build_discriminative_model(dimension)
	model = build_GAN(G, D)
	G.compile(loss = "kld", optimizer = RMSprop(lr=0.0001))
	model.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001))
	D.trainable = True
	D.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001))
	print "Begin training ..."
	batch_size = 128
	for epoch in range(10):
		print "==========Epoch %d===========" % (epoch + 1)
		for index in range(0, len(X), batch_size):
			batch = X[index:index+batch_size]
			Y = [1]*len(batch) + [0]*len(batch) 
	   		noise = generate_noise(batch)
			gen_batch = G.predict(noise)
			combined_batch = np.concatenate((batch, gen_batch))
			d_loss = D.train_on_batch(combined_batch, Y)
			D.trainable = False
			g_loss = model.train_on_batch(noise, [1]*len(batch))
			D.trainable = True
			print "d_loss = %f, g_loss = %f" %(d_loss, g_loss)
			gen_sentences = find_nearest_vocab(sentences[index:index+1], gen_batch[:1])
