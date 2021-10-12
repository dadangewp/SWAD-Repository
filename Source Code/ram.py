# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:29:55 2020

@author: dadangewp
"""

from custom_layers import Attention, RecurrentAttention, InteractiveAttention, ContentAttention

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import re
np.random.seed(1)
import codecs
from keras.models import Model
from keras.layers import LSTM, GRU, Lambda, RepeatVector, Activation, Dense, Dropout, Input, Embedding, multiply,Bidirectional, Flatten, concatenate
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
                tweet = line.split("\t")[1]
                label = line.split("\t")[2]
                if "Yes" in label :
                    misogyny = 1
                else :
                    misogyny = 0
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def getTarget(text):
    sents = text.lower()
    text = re.findall("<b>(.*?)</b>", sents)
    return text[0]

def RNN(X):
    n_hop = 3
    inputs = Input(name='inputs',shape=(None,))
    target = Input(name='target',shape=(None,))
    layer1 = Embedding(vocab+1, 128, input_length=100)(inputs)
    layer2 = Embedding(vocab+1, 16, input_length=100)(target)
    aspect_embed = Flatten()(layer2)
    hidden_out_1 = Bidirectional(LSTM(64, return_sequences=True))(layer1)
    memory = Bidirectional(LSTM(64, return_sequences=True))(hidden_out_1)
    final_attend = RecurrentAttention(units=64, n_hop=n_hop)([memory, aspect_embed])
    layer = Dense(16,name='FC1')(final_attend)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=[inputs,target],outputs=layer)
    return model

dataTrain, dataLabel = parse_training("SWAD-training-new.txt")
dataTest, labelTest = parse_training("SWAD-testing-new.txt")

Y_train = pd.get_dummies(dataLabel).values
Y_test = pd.get_dummies(labelTest).values
#print(Y_test)

dataTrain_cleaned = []
dataTest_cleaned = []

target_train = []
target_test = []

for tweet in dataTrain :
    target = getTarget(tweet)
    tweet = tweet.replace("<b>","")
    tweet = tweet.replace("</b>","")
    dataTrain_cleaned.append(tweet)
    target_train.append(target)
    
for tweet in dataTest :
    target = getTarget(tweet)
    tweet = tweet.replace("<b>","")
    tweet = tweet.replace("</b>","")
    dataTest_cleaned.append(tweet)
    target_test.append(target)

max_len = 100
max_words = 15000
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(dataTrain_cleaned)
vocab = len(tok.word_index)

sequences = tok.texts_to_sequences(dataTrain_cleaned)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

sequences_target = tok.texts_to_sequences(target_train)
sequences_matrix_target = sequence.pad_sequences(sequences_target,maxlen=max_len)

test_sequences = tok.texts_to_sequences(dataTest_cleaned)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

test_sequences_target = tok.texts_to_sequences(target_test)
test_sequences_matrix_target = sequence.pad_sequences(test_sequences_target,maxlen=max_len)


model = RNN(dataTrain)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])
model.fit([sequences_matrix,sequences_matrix_target],Y_train,batch_size=32,epochs=3)

accr = model.evaluate([test_sequences_matrix,test_sequences_matrix_target], Y_test)
print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob = model.predict([test_sequences_matrix,test_sequences_matrix_target]) 
y_pred = np.argmax(y_prob, axis=1)
y_test = np.argmax(Y_test, axis=1)


acc = metrics.accuracy_score(y_test, y_pred) 
score_pos = metrics.f1_score(y_test, y_pred, pos_label=1)
score_neg = metrics.f1_score(y_test, y_pred, pos_label=0)
prec_pos = metrics.precision_score(y_test, y_pred, pos_label=1)
prec_neg = metrics.precision_score(y_test, y_pred, pos_label=0)
rec_pos = metrics.recall_score(y_test, y_pred, pos_label=1)
rec_neg = metrics.recall_score(y_test, y_pred, pos_label=0)

print(prec_pos)
print(prec_neg)
print(rec_pos)
print(rec_neg)
print(score_pos)
print(score_neg)
print(acc)
