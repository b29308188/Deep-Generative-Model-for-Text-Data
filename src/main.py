import sys
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

def read_data(file_path, max_len=20):
    X = []
    sentences = []
    with open(file_path, "r") as f:
        for line in f:
            # word2vec dimension = 300
            x = np.zeros((max_len, 300))
            sentence = []
            tokens = line.strip().split()
            index = 0
            for token in tokens:
                try:
                    token.encode('ascii')
                except UnicodeEncodeError:
                    continue

                if token in w2v_model:
                    x[index] = w2v_model[token]
                    index += 1
                    sentence.append(token)
                if index >= 20:
                    break
            X.append(x)
            sentences.append(sentence)
    return np.array(X), 300, sentences

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
    return np.random.uniform(size = shape)
    # 1. Should only have only one 1 in each row?
    # 2. Should use Gussian noise?

def find_nearest_vocab(sentences, gen_X):
    print "Shape(gen_X): ", gen_X.shape
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
#                print d
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
    #w2v_model = models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    print "Finish loading word2vec model"
    X, dimension, sentences = read_data("data/data.txt")
    G = build_generative_model(dimension)
    D = build_discriminative_model(dimension)
    model = build_GAN(G, D)
    G.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr=0.0001))
    model.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001))
    D.trainable = True
    D.compile(loss = "binary_crossentropy", optimizer = RMSprop(lr=0.0001))
    for epoch in range(10):
        print "==========Epoch %d===========" % epoch
        Y = [1]*len(X) + [0]*len(X) 
        noise = generate_noise(X.shape)
        gen_X = G.predict(noise)
        
        gen_sentences = find_nearest_vocab(sentences[:2], gen_X[:2])
        #for gen_sentence in gen_sentences:
        #    print gen_sentence

        combined_X = np.concatenate((X, gen_X))
        d_loss = D.train_on_batch(combined_X, Y)
        D.trainable = False
        g_loss = model.train_on_batch(noise, [1]*len(X))
        D.trainable = True
        print "d_loss = %f, g_loss = %f" %(d_loss, g_loss)
