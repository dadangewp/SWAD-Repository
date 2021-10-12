# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:57:46 2020

@author: dadangewp
"""

from custom_layers import Attention, RecurrentAttention, InteractiveAttention, ContentAttention

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(1)
import codecs
from keras.models import Model
from keras.layers import LSTM, add, multiply, GRU, Lambda, TimeDistributed, RepeatVector, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten, concatenate
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

def get_mask(data_train, lefts, rights):
    word_mask = []
    for data, left, right in zip(data_train, lefts, rights):
        word_list = data.split(" ")
        start = len(left) - 1
        end = len(word_list) - (len(right) - 1)
        _word_mask = [1] * len(word_list)
        _word_mask[start:end] = [0.5] * (end - start)  # 1 for context, 0.5 for aspect
        #print(_word_mask)
        _word_mask = np.array(_word_mask)
        word_mask.append(_word_mask)
        
    return word_mask


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

def getTarget(text):
    sents = text.lower()
    text = re.findall("<b>(.*?)</b>", sents)
    return text[0]

def sequence_mask(sequence):
    return K.sign(K.max(K.abs(sequence), 2))

def sequence_length(sequence):
    return K.cast(K.sum(sequence_mask(sequence), 1), tf.int32)
    
def RNN(X):
    inp_text = Input(name='inp_text',shape=(None,))
    inp_left = Input(name='inp_left',shape=(None,))
    inp_right = Input(name='inp_right',shape=(None,))
    inp_mask = Input(name='inp_mask',shape=(None,))
    target_word = Input(name='target_word',shape=(None,))
    
    embed_text = Embedding(vocab+1, 128, input_length=100, mask_zero = True)(inp_text)
    embed_left = Embedding(vocab+1, 128, input_length=100, mask_zero = True)(inp_left)
    embed_right = Embedding(vocab+1, 128, input_length=100, mask_zero = True)(inp_right)
    embed_target = Embedding(vocab+1, 128, input_length=100)(target_word)
    aspect_embed = Flatten()(embed_target)

    hidden_l = GRU(64, go_backwards=True, return_sequences=True)(embed_left)
    hidden_r = GRU(64, return_sequences=True)(embed_right)
        
    context_attend_l = TimeDistributed(Dense(1, activation='sigmoid'))(hidden_l)

    context_attend_l = Lambda(lambda x: tf.reverse_sequence(x, sequence_length(x), 1, 0))(context_attend_l)
    context_attend_l = Lambda(lambda x: K.squeeze(x, -1))(context_attend_l)

    context_attend_r = TimeDistributed(Dense(1, activation='sigmoid'))(hidden_r)
    context_attend_r = Lambda(lambda x: K.squeeze(x, -1))(context_attend_r)
    
    
    context_attend = multiply([add([context_attend_l, context_attend_r]), inp_mask])

    context_attend_expand = Lambda(lambda x: K.expand_dims(x))(context_attend)
    memory = multiply([embed_text, context_attend_expand])

    sentence = Lambda(lambda x: K.mean(x, axis=1))(memory)
    final_output = ContentAttention()([memory, aspect_embed, sentence])
    
    layer = Dense(16,name='FC1')(final_output)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=[inp_text,inp_left,inp_right,inp_mask,target_word],outputs=layer)
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

dataTrain_cleaned = []
dataTest_cleaned = []

for tweet in dataTrain :
    target = getTarget(tweet)
    tweet = tweet.replace("<b>","")
    tweet = tweet.replace("</b>","")
    dataTrain_cleaned.append(tweet)
    
for tweet in dataTest :
    target = getTarget(tweet)
    tweet = tweet.replace("<b>","")
    tweet = tweet.replace("</b>","")
    dataTest_cleaned.append(tweet)

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

sequences = tok.texts_to_sequences(dataTrain_cleaned)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

sequences_left = tok.texts_to_sequences(dataTrain_left)
sequences_matrix_left = sequence.pad_sequences(sequences_left,maxlen=max_len)

sequences_right = tok.texts_to_sequences(dataTrain_right)
sequences_matrix_right = sequence.pad_sequences(sequences_right,maxlen=max_len)

sequences_target = tok.texts_to_sequences(targetTrain)
sequences_matrix_target = sequence.pad_sequences(sequences_target,maxlen=max_len)

test_sequences = tok.texts_to_sequences(dataTest_cleaned)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

test_sequences_left = tok.texts_to_sequences(dataTest_left)
test_sequences_matrix_left = sequence.pad_sequences(test_sequences_left,maxlen=max_len)

test_sequences_right = tok.texts_to_sequences(dataTest_right)
test_sequences_matrix_right = sequence.pad_sequences(test_sequences_right,maxlen=max_len)

test_sequences_target = tok.texts_to_sequences(targetTest)
test_sequences_matrix_target = sequence.pad_sequences(test_sequences_target,maxlen=max_len)

input_mask_train = get_mask(dataTrain_cleaned, dataTrain_left, dataTrain_right)
input_mask_train = sequence.pad_sequences(input_mask_train, maxlen=max_len)
input_mask_test = get_mask(dataTest_cleaned, dataTest_left, dataTest_right)
input_mask_test = sequence.pad_sequences(input_mask_test, maxlen=max_len)

model = RNN(dataTrain)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])
model.fit([sequences_matrix,sequences_matrix_left,sequences_matrix_right,input_mask_train,sequences_matrix_target],Y_train,batch_size=32,epochs=3)

accr = model.evaluate([test_sequences_matrix,test_sequences_matrix_left,test_sequences_matrix_right,input_mask_test,test_sequences_matrix_target], Y_test)
print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob = model.predict([test_sequences_matrix,test_sequences_matrix_left,test_sequences_matrix_right,input_mask_test,test_sequences_matrix_target]) 
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
