# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:39:12 2020

@author: dadangewp
"""

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.models import Model
from keras.layers import LSTM, GRU, RepeatVector, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten, concatenate
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
import re

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

def splitString(text):
    sents = text.lower()
    text = re.findall("<b>(.*?)</b>", sents)
    #print (text)
    words = sents.split(" ")
    index = 0
    for idx, word in enumerate(words):
        if word == "<b>"+text[0]+"</b>":
            index = idx
    str1 = ""
    str2 = ""

    str1 = " ".join(words[0:index])
    str2 = " ".join(words[(index+1):])
    
    str2 = text[0]+" "+str2
    str1 = str1+" "+text[0]
    
    return str1, str2, text[0]


    
def RNN(X):
    inp_left = Input(name='inp_left',shape=(None,))
    inp_right = Input(name='inp_right',shape=(None,))
    target_word = Input(name='target_word',shape=(None,))
    embed_left = Embedding(vocab+1, 128, input_length=100)(inp_left)
    embed_right = Embedding(vocab+1, 128, input_length=100)(inp_right)
    embed_target = Embedding(vocab+1, 16, input_length=100)(target_word)
    aspect_embed = Flatten()(embed_target)
    aspect_repeat_l = RepeatVector(max_len)(aspect_embed)
    aspect_repeat_r = RepeatVector(max_len)(aspect_embed)
    left = concatenate([aspect_repeat_l, embed_left], axis=-1)
    right = concatenate([aspect_repeat_r, embed_right], axis=-1)
    lstm_left = LSTM(64)(left)
    lstm_right = LSTM(64)(right)
    #dense_left = Dense(16,name='FC1')(lstm_left)
    #dense_left = Activation('relu')(dense_left)
    #dense_left = Dropout(0.2)(dense_left)
    #dense_right = Dense(16,name='FC2')(lstm_right)
    #dense_right = Activation('relu')(dense_right)
    #dense_right = Dropout(0.2)(dense_right)
    concat = concatenate([lstm_left, lstm_right], axis=-1)
    layer = Dense(16,name='FC1')(concat)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    #concat = concatenate([dense_left, dense_right], axis=-1)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=[inp_left,inp_right,target_word],outputs=layer)
    return model

dataTrain, dataLabel = parse_training("SWAD-training-new.txt")
dataTest, labelTest = parse_training("SWAD-testing-new.txt")

Y_train = pd.get_dummies(dataLabel).values
Y_test = pd.get_dummies(labelTest).values
#print(Y_test)

max_len = 100
max_words = 15000
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(dataTrain)
vocab = len(tok.word_index)

dataTrain_left = []
dataTrain_right = []

dataTest_left = []
dataTest_right = []

targetTrain = []
targetTest = []

for tweet in dataTrain:
    sent_left, sent_right, target = splitString(tweet)
    dataTrain_left.append(sent_left)
    dataTrain_right.append(sent_right)
    targetTrain.append(target)

for tweet in dataTest:
    sent_left, sent_right, target = splitString(tweet)
    dataTest_left.append(sent_left)
    dataTest_right.append(sent_right)
    targetTest.append(target)

sequences_left = tok.texts_to_sequences(dataTrain_left)
sequences_matrix_left = sequence.pad_sequences(sequences_left,maxlen=max_len)

sequences_right = tok.texts_to_sequences(dataTrain_right)
sequences_matrix_right = sequence.pad_sequences(sequences_right,maxlen=max_len)

sequences_target = tok.texts_to_sequences(targetTrain)
sequences_matrix_target = sequence.pad_sequences(sequences_target,maxlen=max_len)

test_sequences_left = tok.texts_to_sequences(dataTest_left)
test_sequences_matrix_left = sequence.pad_sequences(test_sequences_left,maxlen=max_len)

test_sequences_right = tok.texts_to_sequences(dataTest_right)
test_sequences_matrix_right = sequence.pad_sequences(test_sequences_right,maxlen=max_len)

test_sequences_target = tok.texts_to_sequences(targetTest)
test_sequences_matrix_target = sequence.pad_sequences(test_sequences_target,maxlen=max_len)


model = RNN(dataTrain)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])
model.fit([sequences_matrix_left,sequences_matrix_right,sequences_matrix_target],Y_train,batch_size=32,epochs=3)

accr = model.evaluate([test_sequences_matrix_left,test_sequences_matrix_right,test_sequences_matrix_target], Y_test)
print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob = model.predict([test_sequences_matrix_left,test_sequences_matrix_right,test_sequences_matrix_target]) 
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
