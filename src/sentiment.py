import sys
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Activation, LSTM
from keras.optimizers import RMSprop
#from seq2seq.models import SimpleSeq2Seq
from gensim import models, matutils

import theano
theano.config.openmp = True
#OMP_NUM_THREADS=24 python sentiment.py

def read_data(file_path, w2v_model, max_len = 300, hidden_dim = 32):
	X = []
	with open(file_path, "r") as f:
		for line in f:
			x = np.zeros((max_len, hidden_dim))
			tokens = line.strip().split()
			index = 0
			for token in map(normalize, tokens):
				if token in w2v_model:
					x[index] = w2v_model[token]
					index += 1
				if index >= max_len:
					break
			X.append(x)
	return np.array(X)

def normalize(word):
	word  = re.sub(r'[^\w\s]' , "", word.decode("utf-8"), re.UNICODE)
	word = word.lower()
	return word

class Generator:
	def __init__(self, max_len = 300, hidden_dim = 32):
		self.max_len = max_len
		self.hidden_dim  = hidden_dim
		self.G = self.build_generative_model()
		self.D = self.build_discriminative_model()
		self.GAN = self.build_GAN(self.G, self.D)
		self.G.compile(loss = "kld", optimizer = RMSprop(lr=0.0001))
		self.GAN.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001))
		self.D.trainable = True
		self.D.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001))

	def build_generative_model(self):
		G = Sequential()
		G.add(LSTM(128, input_dim = self.hidden_dim, input_length = self.max_len, return_sequences = True))
		G.add(Dropout(0.25))
		G.add(TimeDistributed(Dense(self.hidden_dim)))###Shoud be sequence to sequence?
		G.add(Activation("relu"))
		return G

	def build_discriminative_model(self):
		D = Sequential()
		D.add(LSTM(128, input_dim = self.hidden_dim, input_length = self.max_len, return_sequences = False))
		D.add(Dropout(0.25))
		D.add((Dense(30)))
		D.add(Activation("relu"))
		D.add(Dropout(0.25))
		D.add(Dense(1))
		D.add(Activation("relu"))
		return D

	def build_GAN(self, G, D):
		GAN = Sequential()
		GAN.add(G)
		GAN.trainable = False
		GAN.add(D)
		return GAN

	def generate_noise(self, X):
		#noise = np.random.normal(0, 1, size = X.shape)
		#return X+noise
		return np.random.uniform(size = X.shape)
		#return X + np.random.normal(0, np.std(X), size = X.shape)
	
	def train(self, X, batch_size = 128):
		for index in range(0, len(X), batch_size):
			batch = X[index:index+batch_size]
			Y = [1]*len(batch) + [0]*len(batch) 
			noise = self.generate_noise(batch)
			gen_batch = self.G.predict(noise)
			combined_batch = np.concatenate((batch, gen_batch))
			d_loss = self.D.train_on_batch(combined_batch, Y)
			self.D.trainable = False
			g_loss = self.GAN.train_on_batch(noise, [1]*len(batch))
			self.D.trainable = True
			print "d_loss = %f, g_loss = %f" %(d_loss, g_loss)
	
	def inference(self, X):
		gen_X = []
		batch_size = 128
		for index in range(0, len(X), batch_size):
			batch = X[index:index+batch_size]
			noise = self.generate_noise(batch)
			gen_batch = self.G.predict(noise)
			#x = np.mean(gen_batch, axis = 1)
			gen_X.append(gen_batch)
		gen_X = np.concatenate(gen_X)
		return gen_X
	
	def save(self, file_path):
		self.G.save_weights(file_path+".G.h5")
		self.D.save_weights(file_path+ ".D.h5")
		self.GAN.save_weights(file_path+ ".GAN.h5")

	def load(self, file_path):
		self.G.load_weights(file_path+".G.h5")
		self.D.load_weights(file_path+ ".D.h5")
		self.GAN.load_weights(file_path+ ".GAN.h5")

class Classifier:
	def __init__(self, max_len = 300, hidden_dim = 32):
		self.max_len = max_len
		self.hidden_dim  = hidden_dim
		self.CLF = self.build_CLF()
	
	def build_CLF(self):
		CLF = Sequential()
		CLF.add(LSTM(128, input_dim = self.hidden_dim, input_length = self.max_len, return_sequences = False))
		CLF.add(Dropout(0.25))
		CLF.add(Dense(1))
		CLF.add(Activation("relu"))
		CLF.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001), metrics = ["accuracy"])
		return CLF

	def train(self, X, Y):
		self.CLF.fit(X, Y, nb_epoch = 3, batch_size = 128)

	def evaluate(self, testX, testY):
		print "Accuracy on testing data :", self.CLF.evaluate(testX, testY)
	
	def save(self, file_path):
		self.CLF.save_weights(file_path+".h5")

	def load(self, file_path):
		self.CLF.load_weights(file_path+".h5")

if __name__ == "__main__":
	
	print "Loading word vectors ..."
	w2v_model = models.Word2Vec.load("../models/word2vec.mod")
	print "Reading text data ..."
	"""
	trainX_pos = read_data("../data/train-pos.txt", w2v_model)
	trainX_neg = read_data("../data/train-neg.txt", w2v_model)
	testX_pos = read_data("../data/test-pos.txt", w2v_model)
	testX_neg = read_data("../data/test-neg.txt", w2v_model)
	"""
	trainX_pos = read_data("../data/toy.txt", w2v_model)
	trainX_neg = read_data("../data/toy.txt", w2v_model)
	testX_pos = read_data("../data/toy.txt", w2v_model)
	testX_neg = read_data("../data/toy.txt", w2v_model)
	trainX = np.vstack((trainX_pos, trainX_neg))
	trainY  = [1]*len(trainX_pos)+ [0]*len(trainX_neg)
	testX = np.vstack((testX_pos, testX_neg))
	testY  = [1]*len(testX_pos)+ [0]*len(testX_neg)
	
	print "Building the basic classifier ..."
	clf = Classifier()
	print "Training the basic classifier ..."
	clf.train(trainX, trainY)
	clf.evaluate(testX, testY)
	clf.save("../models/clf")
	clf.load("../models/clf")

	print "Building the generative model for positive examples..."
	pos_gan = Generator()	
	print "Training ..."
	for epoch in range(1, 4):
		print "==========Epoch %d===========" % (epoch)
		pos_gan.train(trainX)
		pos_gan.save("../models/pos-%d" % (epoch))
		pos_gan.load("../models/pos-%d" % (epoch))
	
	print "Building the generative model for positive examples..."
	neg_gan = Generator()	
	print "Training ..."
	for epoch in range(1, 4):
		print "==========Epoch %d===========" % (epoch)
		neg_gan.train(trainX)
		neg_gan.save("../models/neg-%d" % (epoch))
		neg_gan.load("../models/neg-%d" % (epoch))

	print "Generating samples ..."
	posX = pos_gan.inference(trainX_pos)
	negX = neg_gan.inference(trainX_neg)
	trainX = np.concatenate((trainX, posX, negX))
	trainY = trainY + [1]*len(posX) + [0]*len(negX)
	
	print "Building the augmented classifier ..."
	aug_clf = Classifier()
	print "Training the augmented classifier ..."
	aug_clf.train(trainX, trainY)
	aug_clf.evaluate(testX, testY)
	aug_clf.save("../models/clf")
	aug_clf.load("../models/clf")
