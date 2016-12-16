import sys
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Activation, LSTM, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
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
        self.GAN.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=0.0001))
        self.D.trainable = True
        self.D.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=0.0001))
    def build_generative_model(self):
        G = Sequential()
        G.add(Dense(300, input_dim = self.hidden_dim, activation = "relu"))
        #G.add(BatchNormalization())
        G.add(Dense(300, activation = "relu"))
        G.add(Dense(300, activation = "relu"))
        G.add(Dense(300, activation = "relu"))
        G.add(Dense(self.hidden_dim))
        return G

    def build_discriminative_model(self):
        D = Sequential()
        D.add(Dense(300, input_dim = self.hidden_dim, activation ="relu"))
        D.add(Dense(300, activation = "relu"))
        D.add(Dense(300, activation = "relu"))
        D.add(Dense(2, activation = "softmax"))
        return D

    def build_GAN(self, G, D):
        GAN = Sequential()
        GAN.add(G)
        D.trainable = False
        GAN.add(D)
        return GAN

    def generate_noise(self, shape):
        return np.random.uniform(-1, 1, size = shape)

    def pre_trainG(self, X, batch_size = 128):
        print "Pre-train G ..."
        L = []
        for index in range(0, len(X), batch_size):
            batch = X[index:index+batch_size]
            noise = self.generate_noise(batch.shape)
            loss = self.G.train_on_batch(noise, batch)
            L.append(loss)
        print "loss = %f" % np.mean(loss)
    
    def pre_trainD(self, X, batch_size = 128):
        print "Pre-train D"
        L = []
        for index in range(0, len(X), batch_size):
            batch = X[index:index+batch_size]
            noise = self.generate_noise(batch.shape)
            gen_batch = self.G.predict(noise)
            Y = [1]*len(batch) + [0]*len(batch)
	    Y = np_utils.to_categorical(Y, nb_classes = 2)
            combined_batch = np.concatenate((batch, gen_batch))
            loss = self.D.train_on_batch(combined_batch, Y)
            L.append(loss)
        print "loss = %f" % np.mean(loss)

    def train(self, X, batch_size = 128):
        G_loss = []
        D_loss = []
        for index in range(0, len(X), batch_size):
            batch = X[index:index+batch_size]
            Y = [1]*len(batch) + [0]*len(batch)
	    Y = np_utils.to_categorical(Y, nb_classes = 2)
            noise = self.generate_noise(batch.shape)
            gen_batch = self.G.predict(noise)
            combined_batch = np.concatenate((batch, gen_batch))

            d_loss = self.D.train_on_batch(combined_batch, Y)
            noise = self.generate_noise(batch.shape)
            g_loss = self.GAN.train_on_batch(noise, np_utils.to_categorical([1]*len(batch), nb_classes = 2))
            G_loss.append(g_loss)
            D_loss.append(d_loss)
        print "d_loss = %f, gan_loss = %f" %(np.mean(D_loss), np.mean(G_loss))
    
    
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
    
    def train(self, X, Y):
        #X, Y = shuffle(X, Y)
        self.CLF.fit(X, Y)
        #print X
        #print self.CLF.score(X, Y)
        #print self.CLF.coef_
    def evaluate(self, testX, testY):
        #print testX
        #print self.CLF.predict(testX)
        print "Accuracy on testing data :", self.CLF.score(testX, testY)
    
    def save(self, file_path):
        pass   
    def load(self, file_path):
        pass

if __name__ == "__main__":
    #RNN
    np.random.seed(0)
    
    print "Loading word vectors ..."
    w2v_model = models.Word2Vec.load("../models/word2vec.mod")
    print "Reading text data ..."
    
    trainX_pos = read_data("../data/train-pos.small", w2v_model)
    trainX_neg = read_data("../data/train-neg.small", w2v_model)
    testX_pos = read_data("../data/test-pos.small", w2v_model)
    testX_neg = read_data("../data/test-neg.small", w2v_model)
    
    #scaler = MinMaxScaler(feature_range = (0, 1))
    #trainX_pos = np.ones((500,32))*1.5
    #trainX_neg = -np.ones((500,32))*1.5
    #testX_pos = np.ones((500,32))*1.5
    #testX_neg = -np.ones((500,32))*1.5
    """
    X = np.concatenate((trainX_pos, trainX_neg, testX_pos, testX_neg))
    scaler.fit(X)
    trainX_pos = scaler.transform(trainX_pos)
    trainX_neg = scaler.transform(trainX_neg)
    testX_pos = scaler.transform(testX_pos)
    testX_neg = scaler.transform(testX_neg)
    """
    trainX = np.vstack((trainX_pos, trainX_neg))
    trainY  = [1]*len(trainX_pos)+ [0]*len(trainX_neg)
    testX = np.vstack((testX_pos, testX_neg))
    testY  = [1]*len(testX_pos)+ [0]*len(testX_neg)
    print len(testX)
    print "Building the pos generative model..."
    pos_gan = Generator()    
    """
    for epoch in range(1, 30):
        print "==========Epoch %d===========" % (epoch)
        pos_gan.pre_trainG(trainX_pos)
    for epoch in range(1, 30):
        print "==========Epoch %d===========" % (epoch)
        pos_gan.pre_trainD(trainX_pos)
    """
    #print "Training ..."
    for epoch in range(1, 30):
        print "==========Epoch %d===========" % (epoch)
        pos_gan.train(trainX_pos)
   
    posX = pos_gan.generate((50, 32))    
    pos_gan.save("../models/pos_basic_32")
    pos_gan.load("../models/pos_basic_32")
    print "Building the neg generative model..."
    neg_gan = Generator()    
    """
    for epoch in range(1, 30):
        print "==========Epoch %d===========" % (epoch)
        neg_gan.pre_trainG(trainX_neg)
    for epoch in range(1, 30):
        print "==========Epoch %d===========" % (epoch)
        neg_gan.pre_trainD(trainX_neg)
    """
    print "Training ..."
    for epoch in range(1, 30):
        print "==========Epoch %d===========" % (epoch)
        neg_gan.train(trainX_neg)
    
    negX = neg_gan.generate((50, 32))    
    neg_gan.save("../models/neg_basic_32")
    neg_gan.load("../models/neg_basic_32")
    sample_trainX = np.vstack((posX, negX))
    sample_trainX.dump("../sampleX.np")
    sample_trainY = [1]*len(posX) + [0]*len(negX)
    print "Building the basic classifier ..."
    clf = Classifier()
    print "Training the basic classifier ..."
    clf.train(np.vstack((trainX[:10], trainX[-10:])), [1]*10+[0]*10)
    #clf.train(trainX, trainY)
    clf.evaluate(testX, testY)
    
    print "Building the sampled classifier ..."
    aug_clf = Classifier()
    print "Training the sampled classifier ..."
    aug_clf.train(sample_trainX, sample_trainY)
    aug_clf.evaluate(testX, testY)
