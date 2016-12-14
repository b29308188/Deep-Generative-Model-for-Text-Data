import sys
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Activation, LSTM, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
#from seq2seq.models import SimpleSeq2Seq
from gensim import models, matutils
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import theano
theano.config.openmp = True
from sklearn.linear_model import LogisticRegression
#OMP_NUM_THREADS=24 python sentiment.py

def read_data(file_path, w2v_model, max_len = 300, hidden_dim = 32):
	X = []
	with open(file_path, "r") as f:
		for line in f:
			x = np.zeros(hidden_dim)
			tokens = line.strip().split()
			index = 0
			for token in map(normalize, tokens):
				if token in w2v_model:
					x += w2v_model[token]
					index += 1
				if index >= max_len:
					break
			x /= index
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
		self.G.compile(loss = "mse", optimizer = Adam(lr=0.0001))
		self.GAN.compile(loss = "mse", optimizer = Adam(lr=0.0001))
		self.D.trainable = True
		self.D.compile(loss = "mse", optimizer = Adam(lr=0.0001))
		self.f = open("../to_plot.txt", "w")
	def build_generative_model(self):
		G = Sequential()
		G.add(Dense(300, input_dim = self.hidden_dim))
		#G.add(BatchNormalization())
		G.add(Activation("tanh"))
		G.add(Dense(300, activation = "tanh"))
		G.add(Dense(300, activation = "tanh"))
		G.add(Dense(self.hidden_dim, activation = "tanh"))
		return G

	def build_discriminative_model(self):
		D = Sequential()
		D.add(Dense(300, input_dim = self.hidden_dim, activation = "tanh"))
		D.add(Dense(300, activation = "tanh"))
		D.add(Dense(300, activation = "tanh"))
		D.add(Dense(1, activation = "sigmoid"))
		return D

	def build_GAN(self, G, D):
		GAN = Sequential()
		GAN.add(G)
		D.trainable = False
		GAN.add(D)
		return GAN

	def generate_noise(self, shape):
		#noise = np.random.normal(np.mean(X), np.std(X), size = X.shape)
		#return noise
		#return X
		return np.random.uniform(-1, 1, size = shape)

		#return X + np.random.normal(0, np.std(X), size = X.shape)
		#return np.ones(X.shape)
	
	def train(self, X, batch_size = 128):
		G_loss = []
		D_loss = []
		self.D.trainable = True
		for index in range(0, len(X), batch_size):
			batch = X[index:index+batch_size]
			Y = [1]*len(batch) + [0]*len(batch)
			noise = self.generate_noise(batch.shape)
			gen_batch = self.G.predict(noise)
			#print gen_batch
			combined_batch = np.concatenate((batch, gen_batch))
			d_loss = self.D.train_on_batch(combined_batch, Y)
			self.D.trainable = False
			noise = self.generate_noise(batch.shape)
			g_loss = self.GAN.train_on_batch(noise, [1]*len(batch))
			self.D.trainable = True
			#while d_loss > g_loss:
				#d_loss = self.D.train_on_batch(combined_batch, Y)
				#print "D:", d_loss, g_loss
			#self.D.trainable = False
			#while g_loss > d_loss:
				#g_loss = self.GAN.train_on_batch(noise, [1]*len(batch))
				#print "G:", d_loss, g_loss
			#self.D.trainable = True
			G_loss.append(g_loss)
			D_loss.append(d_loss)
			self.f.write("%f %f\n" %(d_loss, g_loss))
		print "d_loss = %f, g_loss = %f" %(np.mean(D_loss), np.mean(G_loss))
	
	
	def generate(self, shape):
		noise = self.generate_noise(shape)
		gen_X = self.G.predict(noise)
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
		self.CLF = LogisticRegression()
		#self.max_len = max_len
		#self.hidden_dim  = hidden_dim
		#self.CLF = self.build_CLF()
	
	def build_CLF(self):
		CLF = Sequential()
		CLF.add(Dense(128, input_dim = self.hidden_dim, activation = "tanh"))
		CLF.add(Dropout(0.25))
		CLF.add(Dense(64, activation = "tanh"))
		CLF.add(Dropout(0.25))
		CLF.add(Dense(2, activation = "softmax"))
		CLF.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 0.001), metrics = ["accuracy"])
		return CLF

	def train(self, X, Y):
		X, Y = shuffle(X, Y)
		#Y = np_utils.to_categorical(Y, 2)
		self.CLF.fit(X, Y)
	
	def evaluate(self, testX, testY):
		#testY = np_utils.to_categorical(testY, 2)
		print "Accuracy on testing data :", self.CLF.score(testX, testY)
	
	def save(self, file_path):
		#self.CLF.save_weights(file_path+".h5")
		pass
	
	def load(self, file_path):
		#self.CLF.load_weights(file_path+".h5")
		pass

if __name__ == "__main__":
	#add noise 
	#RNN
	np.random.seed(0)
	"""
	print "Loading word vectors ..."
	w2v_model = models.Word2Vec.load("../models/word2vec.mod")
	print "Reading text data ..."
	trainX_pos = read_data("../data/train-pos.txt", w2v_model)
	trainX_neg = read_data("../data/train-neg.txt", w2v_model)
	testX_pos = read_data("../data/test-pos.txt", w2v_model)
	testX_neg = read_data("../data/test-neg.txt", w2v_model)
	
	trainX_pos = np.random.normal(1, 1, size = (100, 32))
	trainX_neg = np.random.normal(-1, 1, size = (100, 32))
	testX_pos = np.random.normal(1, 1, size = (100, 32))
	testX_neg = np.random.normal(-1, 1, size = (100, 32))
	"""
	tmpX_pos = np.ones((100, 32))/2
	tmpX_neg = -np.ones((100, 32))
	trainX = np.vstack((trainX_pos, trainX_neg))
	trainY  = [1]*len(trainX_pos)+ [0]*len(trainX_neg)
	testX = np.vstack((testX_pos, testX_neg))
	testY  = [1]*len(testX_pos)+ [0]*len(testX_neg)
	
	scaler = MinMaxScaler(feature_range = (-1, 1))
	trainX = scaler.fit_transform(trainX)
	testX = scaler.transform(testX)
	X_pos = scaler.transform(trainX_pos)
	X_neg = scaler.transform(trainX_neg)
	
	print "Building the pos generative model..."
	pos_gan = Generator()	
	print "Training ..."
	for epoch in range(1, 100):
		print "==========Epoch %d===========" % (epoch)
		pos_gan.train(X_pos[:200])
	posX = pos_gan.generate((100, 32))	
	#print posX	
	pos_gan.f.close()
	print "Building the neg generative model..."
	neg_gan = Generator()	
	print "Training ..."
	for epoch in range(1, 100):
		print "==========Epoch %d===========" % (epoch)
		neg_gan.train(X_neg[:200])
	negX = neg_gan.generate((100, 32))	
	sample_trainX = np.vstack((posX, negX))
	sample_trainY = [1]*len(posX) + [0]*len(negX)
		
	print "Building the basic classifier ..."
	clf = Classifier()
	print "Training the basic classifier ..."
	clf.train(np.vstack((trainX[:50], trainX[-50:])), [1]*50+[0]*50)
	clf.evaluate(testX, testY)
	
	print "Building the augmented classifier ..."
	aug_clf = Classifier()
	print "Training the augmented classifier ..."
	aug_clf.train(sample_trainX, sample_trainY)
	aug_clf.evaluate(testX, testY)
