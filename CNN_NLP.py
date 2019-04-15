####################### NLP using CNN on toxic comment dataset ########################

## Importing libraries
import os
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score

####
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5

## Load in pretrained word vectors
print("Loading word vectors..... ")
word2vec = {}
with open(os.path.join('glove.6B/glove.6B.%sd.txt' %EMBEDDING_DIM)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype = 'float32')
        word2vec[word] = vec
        
print('Found %s word vectors' % len(word2vec))

## Prepare text samples and their labels
print("Loading in comments..... ")
train = pd.read_csv('train.csv')
sentences = train['comment_text'].fillna('DUMMY_VALUE').values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values



## convert the sentences into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
# print("sequences:", sequences); exit()

print('max sequence length: ', max(len(s) for s in sequences))
print('min sequence length: ', min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
print('median sequence length: ', s[len(s)//2])

# get word -> integer mappingpr
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)

## Building Model
print("Building model");
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128, 3, activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation = 'relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_labels), activation = 'sigmoid')(x)
model = Model(input_, output)
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Training model...... ")
r = model.fit(data, targets, validation_split = VALIDATION_SPLIT, epochs = EPOCHS, batch_size = BATCH_SIZE)

## Plotting 
plt.plot(r.history['acc'], label = 'acc')
plt.plot(r.history['val_acc'], label = 'val_acc')
plt.legend()
plt.show()

# loss
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

## plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))


## Test and submission
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')


test_sentences = test['comment_text'].fillna('DUMMY_VALUE').values
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
pred_test = model.predict(test_data, batch_size = 1024, verbose = 1)

sub[possible_labels] = pred_test
sub.to_csv('submission.csv', index=False)






