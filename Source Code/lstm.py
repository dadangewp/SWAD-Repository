# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:14:13 2020

@author: dadangewp
"""


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.models import Model
from keras.layers import LSTM, GRU, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
from sklearn.metrics import confusion_matrix

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess) 

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[2]
                label = line.split("\t")[1]
                if "Yes" in label :
                    misogyny = 1
                else :
                    misogyny = 0
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def RNN(X):
    inputs = Input(name='inputs',shape=(None,))
    layer = Embedding(vocab+1, 128, input_length=100)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(16,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

dataTrain, dataLabel = parse_training("SWAD-training.txt")
dataTest, labelTest = parse_training("SWAD-testing.txt")

Y_train = pd.get_dummies(dataLabel).values
Y_test = pd.get_dummies(labelTest).values
#print(Y_test)

dataTrain_cleaned = []
dataTest_cleaned = []

for tweet in dataTrain :
    tweet = tweet.replace("<b>","")
    tweet = tweet.replace("</b>","")
    dataTrain_cleaned.append(tweet)
    
for tweet in dataTest :
    tweet = tweet.replace("<b>","")
    tweet = tweet.replace("</b>","")
    dataTest_cleaned.append(tweet)

max_len = 100
max_words = 15000
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(dataTrain_cleaned)
vocab = len(tok.word_index)

sequences = tok.texts_to_sequences(dataTrain_cleaned)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(dataTest_cleaned)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


model = RNN(dataTrain)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])
model.fit(sequences_matrix,Y_train,batch_size=32,epochs=3)

accr = model.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob = model.predict(test_sequences_matrix) 
y_pred = np.argmax(y_prob, axis=1)
y_test = np.argmax(Y_test, axis=1)
#print(y_pred)
#print(y_test)
acc = metrics.accuracy_score(y_test, y_pred) 
score_pos = metrics.f1_score(y_test, y_pred, average="macro")
#score_neg = metrics.f1_score(Y_test[1], y_pred, pos_label=0)
prec = metrics.precision_score(y_test, y_pred, average="macro")
rec = metrics.recall_score(y_test, y_pred, average="macro")
#tn, fp, fn, tp = confusion_matrix(Y_test[1],y_pred).ravel()
#avg = metrics.f1_score(y_test, y_pred, average="macro")
print(acc)
print(score_pos)
#print(avg)
print(prec)
print(rec)
