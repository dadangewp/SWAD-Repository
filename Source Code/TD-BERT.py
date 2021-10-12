!pip install bert-tensorflow==1.0.1
%tensorflow_version 1.x

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import keras as k
from sklearn import metrics
import codecs
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from bert.tokenization import FullTokenizer
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_score, recall_score
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

max_seq_length = 100

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

dataTrain, dataLabel = parse_training("SWAD-training-new.txt")
dataTest, labelTest = parse_training("SWAD-testing-new.txt")  

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

# Convert data to InputExample format

dataTrain_left = []
dataTrain_right = []

dataTest_left = []
dataTest_right = []

for tweet in dataTrain:
    sent_left, sent_right, target_tr = splitString(tweet)
    dataTrain_left.append(sent_left)
    dataTrain_right.append(sent_right)

for tweet in dataTest:
    sent_left, sent_right, target_ts = splitString(tweet)
    dataTest_left.append(sent_left)
    dataTest_right.append(sent_right)

# Create datasets (Only take up to max_seq_length words for memory)
train_text_left = [' '.join(t.split()[0:max_seq_length]) for t in dataTrain_left]
train_text_left = np.array(train_text_left, dtype=object)[:, np.newaxis]

train_text_right = [' '.join(t.split()[0:max_seq_length]) for t in dataTrain_right]
train_text_right = np.array(train_text_right, dtype=object)[:, np.newaxis]
#train_label = train_df['polarity'].tolist()

test_text_left = [' '.join(t.split()[0:max_seq_length]) for t in dataTest_left]
test_text_left = np.array(test_text_left, dtype=object)[:, np.newaxis]

test_text_right = [' '.join(t.split()[0:max_seq_length]) for t in dataTest_right]
test_text_right = np.array(test_text_right, dtype=object)[:, np.newaxis]
#test_label = test_df['polarity'].tolist()


train_examples_left = convert_text_to_examples(train_text_left, dataLabel)
test_examples_left = convert_text_to_examples(test_text_left, labelTest)

train_examples_right = convert_text_to_examples(train_text_right, dataLabel)
test_examples_right = convert_text_to_examples(test_text_right, labelTest)

# Convert to features
(train_input_ids_left, train_input_masks_left, train_segment_ids_left, train_labels) = convert_examples_to_features(tokenizer, train_examples_left, max_seq_length=max_seq_length)
(test_input_ids_left, test_input_masks_left, test_segment_ids_left, test_labels) = convert_examples_to_features(tokenizer, test_examples_left, max_seq_length=max_seq_length)

(train_input_ids_right, train_input_masks_right, train_segment_ids_right, train_labels) = convert_examples_to_features(tokenizer, train_examples_right, max_seq_length=max_seq_length)
(test_input_ids_right, test_input_masks_right, test_segment_ids_right, test_labels) = convert_examples_to_features(tokenizer, test_examples_right, max_seq_length=max_seq_length)

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        #bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = False
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

# Build model
def build_model(max_seq_length): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    in_id1 = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids1")
    in_mask1 = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks1")
    in_segment1 = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids1")
    bert_inputs1 = [in_id1, in_mask1, in_segment1]
    
    target_word = tf.keras.layers.Input(name='target_word',shape=(None,))
    embed_target = tf.keras.layers.Embedding(vocab+1, 16, input_length=100)(target_word)

    aspect_embed = tf.keras.layers.Flatten()(embed_target)
    
    aspect_repeat_l = tf.keras.layers.RepeatVector(max_len)(aspect_embed)
    aspect_repeat_l = tf.keras.layers.GlobalMaxPooling1D()(aspect_repeat_l)

    aspect_repeat_r = tf.keras.layers.RepeatVector(max_len)(aspect_embed)
    aspect_repeat_r = tf.keras.layers.GlobalMaxPooling1D()(aspect_repeat_r)

    bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)
    bert_output1 = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs1)

    left = tf.keras.layers.concatenate([aspect_repeat_l, bert_output], axis=-1)
    right = tf.keras.layers.concatenate([aspect_repeat_r, bert_output1], axis=-1)

    dense = tf.keras.layers.Dense(128, activation='relu')(left)
    dense = tf.keras.layers.Dropout(0.1)(dense)

    dense1 = tf.keras.layers.Dense(128, activation='relu')(right)
    dense1 = tf.keras.layers.Dropout(0.1)(dense1)

    concat = tf.keras.layers.concatenate([dense, dense1], axis=-1)

    pred = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
    
    model = tf.keras.models.Model(inputs=[bert_inputs,bert_inputs1,target], outputs=pred)
    myadam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=myadam, metrics=['accuracy'])
    model.summary()
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

max_len = 100
max_words = 15000
tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
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

model = build_model(max_seq_length)

# Instantiate variables
initialize_vars(sess)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.ckpt', monitor='val_loss',
                                                 save_weights_only=True,
                                                 verbose=1, save_best_only=True)

model.fit(
    [train_input_ids_left, train_input_masks_left, train_segment_ids_left, train_input_ids_right, train_input_masks_right, train_segment_ids_right,targetTrain], 
    train_labels,
#    validation_split=0.2,
#    verbose=True,
#    callbacks=[es, mc],
    epochs=3,
    batch_size=32
)

y_prob = model.predict([test_input_ids_left, test_input_masks_left, test_segment_ids_left, test_input_ids_right, test_input_masks_right, test_segment_ids_right,targetTest])
#print(y_prob)
y_pred = np.where(y_prob > 0.5, 1, 0)
#print(y_pred)

acc = accuracy_score(labelTest,y_pred)
p0 = precision_score(labelTest,y_pred, pos_label=0)
r0 = recall_score(labelTest,y_pred, pos_label=0)
f0 = f1_score(labelTest,y_pred, pos_label=0)
p1 = precision_score(labelTest,y_pred, pos_label=1)
r1 = recall_score(labelTest,y_pred, pos_label=1)
f1 = f1_score(labelTest,y_pred, pos_label=1)
p = precision_score(labelTest,y_pred, average="macro")
r = recall_score(labelTest,y_pred, average="macro")
f = f1_score(labelTest,y_pred, average="macro")

print(p0)
print(r0)
print(f0)
print(p1)
print(r1)
print(f1)
print(acc)
print(p)
print(r)
print(f)

CorpusFile = './prediction-bert-id.tsv'
CorpusFileOpen = codecs.open(CorpusFile, "w", "utf-8")
for label in y_pred:
    CorpusFileOpen.write(str(label[0])) 
    CorpusFileOpen.write("\n")   
CorpusFileOpen.close()