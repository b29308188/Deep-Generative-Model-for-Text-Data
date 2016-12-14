import os
import sys
import re
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
#from seq2seq.models import SimpleSeq2Seq
from gensim import models, matutils
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def sentence_to_vector(line, w2v_model, max_len=300, hidden_dim=32):
    x = np.zeros((max_len, hidden_dim))
    tokens = tokenize(line)
    index = 0
    for token in tokens:
        if token in w2v_model:
            x[index] = w2v_model[token]
            index += 1
            if index >= max_len:
                break
    return x

def read_data_dir(dir_path, w2v_model, max_len = 300, hidden_dim = 32):
    X = []
    for fname in os.listdir(dir_path):
        for line in open(os.path.join(dir_path, fname)):
            X.append(sentence_to_vector(line, w2v_model, max_len, hidden_dim))
    return np.array(X)

def read_data_file(file_path, w2v_model, max_len = 300, hidden_dim = 32):
    X = []
    for line in open(file_path):
        X.append(sentence_to_vector(line, w2v_model, max_len, hidden_dim))
    return np.array(X)

def tokenize(word):
    word  = re.sub(r'[^\w\s]' , "", word.decode("utf-8"), re.UNICODE)
    word = word.lower()
    return [lemmatizer.lemmatize(token) for token in word.split()]

def choose_answers(Y):
    with open('output.txt', 'w') as out:
        out.write('\n'.join([str(y[0]) for y in Y]))

    answers = []
    for i in range(0, len(Y), 5):
        ans = 0
        max_prob = Y[i][0]
        for j in range(1, 5):
            if Y[i + j][0] > max_prob:
                ans = j
                max_prob = Y[i + j]

        answers.append(ans)
    return answers

def read_answers():
    answers = []
    for line in open('../data/ans.txt'):
        answers.append(int(line.strip()))
    return answers

def calc_accuracy(output, answers):
    correct = [1 if a1 == a2 else 0 for a1, a2 in zip(output, answers)]
    return float(correct.count(1)) / float(len(answers))

class Generator:
    def __init__(self, max_len = 300, hidden_dim = 32, batch_size = 128):
        self.max_len = max_len
        self.hidden_dim  = hidden_dim
        self.batch_size = batch_size
        self.G = self.build_generative_model()
        self.D = self.build_discriminative_model()
        self.GAN = self.build_GAN(self.G, self.D)
        self.G.compile(loss = "kld", optimizer = Adam(lr=0.0001))
        self.GAN.compile(loss = "mse", optimizer = Adam(lr=0.00001))
        self.D.trainable = True
        self.D.compile(loss = "mse", optimizer = Adam(lr=0.00001))

    def build_generative_model(self):
        G = Sequential()
        G.add(LSTM(self.batch_size, input_dim = self.hidden_dim, input_length = self.max_len, return_sequences = True))
        G.add(BatchNormalization())
        G.add(Dropout(0.25))
        G.add(TimeDistributed(Dense(self.hidden_dim)))###Shoud be sequence to sequence?
        G.add(Activation("relu"))
        return G

    def build_discriminative_model(self):
        D = Sequential()
        D.add(LSTM(self.batch_size, input_dim = self.hidden_dim, input_length = self.max_len, return_sequences = False))
        # D.add(BatchNormalization(mode=2))
        D.add(Dropout(0.25))
        D.add(Dense(30))
        D.add(LeakyReLU(0.2))
        # D.add(Activation("relu"))
        D.add(Dropout(0.25))
        D.add(Dense(1))
        D.add(Activation("sigmoid"))
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
        return X + np.random.uniform(-1, 1, size = X.shape)
        # return X + np.random.normal(0, np.std(X), size = X.shape)
    
    def train(self, X):
        batch_size = self.batch_size
        for index in range(0, len(X), batch_size):
            batch = X[index:index+batch_size]
            Y = [1] * (len(batch) / 2) + [0] * (len(batch) * 2)
            noise = self.generate_noise(batch)
            gen_batch = self.G.predict(noise)
            noise2 = self.generate_noise(batch)
            gen_batch2 = self.G.predict(noise2)
            combined_batch = np.concatenate((batch[:len(batch)/2], gen_batch, gen_batch2))
            d_loss = self.D.train_on_batch(combined_batch, Y)
            print "d_loss = %f" % d_loss
            if index / batch_size == 10:
                self.D.trainable = False
                g_loss = self.GAN.train_on_batch(noise, [1]*len(batch))
                print "g_loss = %f" % g_loss
            self.D.trainable = True
            print self.D.predict(batch)[0], self.GAN.predict(noise)[0]
    
    def inference(self, X):
        gen_X = []
        batch_size = self.batch_size
        for index in range(0, len(X), batch_size):
            batch = X[index:index+batch_size]
            # noise = self.generate_noise(batch)
            gen_batch = self.D.predict(batch)
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

if __name__ == "__main__":
    
    # gan_mode = False
    gan_mode = True

    print "Loading word vectors ..."
    dimension = 300
    max_len = 100
    w2v_model = models.Word2Vec.load("../models/holmes_lem_300.model")

    print "Building the generative model..."
    gan = Generator(hidden_dim=dimension, max_len=max_len)  
 
    testX = read_data_file("../data/msr_sentence_completion_questions.txt", w2v_model, hidden_dim=dimension, max_len=max_len)
    answers = read_answers()
    accuracy = []

    if gan_mode:
        print "Reading train data from novel ..."
        trainX = read_data_dir("../data/holmes_train_data", w2v_model, hidden_dim=dimension, max_len=max_len)

        print "Training ..."
        print "Input size: %d" % len(trainX)
        for epoch in range(15):
            print "==========Epoch %d===========" % (epoch + 1)
            gan.train(trainX)
            gan.save("../models/gan-holmes-%d" % (epoch))
            Y = gan.inference(testX)
            print Y[:10]
            acc = calc_accuracy(choose_answers(Y), answers)
            print 'Accuracy: ', acc
            accuracy.append(acc)
    else:
        gan.load("../models/gan-holmes-2")

    print accuracy
