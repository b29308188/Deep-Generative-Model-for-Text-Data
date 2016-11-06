import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.optimizers import RMSprop
#from seq2seq.models import SimpleSeq2Seq

def read_data(file_path, max_len = 20):
    vocabs = set()
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                vocabs.add(token)
    voc_index = {voc:i for i,voc in enumerate(list(vocabs))}
    X = []
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            x = np.zeros((max_len, len(voc_index))) #one hot encoding -> replaced by word vectors?
            for i, token in enumerate(tokens):
                if i >= max_len:
                    break
                x[i,voc_index[token]] = 1
            X.append(x)
    return np.array(X), voc_index

def build_generative_model(input_dim, max_len = 20):
    G = Sequential()
    #G.add(Embedding(voc_size, embedding_size))
    G.add(LSTM(128, input_dim = input_dim, input_length = max_len, return_sequences = True))
    G.add(Dropout(0.25))
    G.add(TimeDistributed(Dense(input_dim)))###Shoud be sequence to sequence?
    G.add(Activation("relu"))
    #print G.summary()
    return G

def build_discriminative_model(input_dim, max_len = 20):
    D = Sequential()
    D.add(LSTM(128, input_dim = input_dim, input_length = max_len, return_sequences = False))
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
    #print G.summary()
    return model

def generate_noise(shape):
    X = []
    for i in range(shape[0]):
        x = np.zeros((shape[1], shape[2]))
        for j in range(shape[1]):
           x[j, np.random.randint(shape[2])] = 1
        X.append(x)
    return np.array(X)
    #return np.random.uniform(size = shape)
    # 1. Should only have only one 1 in each row?
    # 2. Should use Gussian noise?

if __name__ == "__main__":
    X, voc_index = read_data(sys.argv[1])
    G = build_generative_model(len(voc_index))
    D = build_discriminative_model(len(voc_index))
    model = build_GAN(G, D)
    G.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 0.0001))
    model.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr = 0.0001))
    D.trainable = True
    D.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr = 0.0001))
    for epoch in range(100):
        print "==========Epoch %d===========" % epoch
        Y = [1]*len(X) + [0]*len(X) 
        noise = generate_noise(X.shape)
        gen_X = G.predict(noise)
        combined_X = np.concatenate((X, gen_X))
        d_loss = D.train_on_batch(combined_X, Y)
        D.trainable = False
        g_loss = model.train_on_batch(noise, [1]*len(X))
        D.trainable = True
        print "d_loss = %f, g_loss = %f" %(d_loss, g_loss)
