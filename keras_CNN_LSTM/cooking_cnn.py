from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append("../datasets")
import cooking

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from pandas import DataFrame

'''
    This example demonstrates the use of Convolution1D
    for text classification.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py

    Get to 0.8330 test accuracy after 3 epochs. 100s/epoch on K520 GPU.
'''

# set parameters:
maxlen = 65
batch_size = 50
embedding_dims = 300
nb_filter = 200
filter_length = 3
hidden_dims = 150
nb_epoch = 15

print("Loading data...")
(X_train, y_train), (X_test, y_test), (X_prediction, X_id), vocab, cuisines = cooking.load_data(train_path="../train.json", test_path="../test.json", test_split=0.0)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
max_features = len(vocab)
print("Pad sequences (samples x time)")
nb_classes = np.max(y_train)+1

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_prediction = sequence.pad_sequences(X_prediction, maxlen=maxlen)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# TODO: use word2vec model (one hot)

model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

model.add(Flatten())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
print('fitting model')
model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
          show_accuracy=True, validation_split=0.1, shuffle=True)
a = model.predict_classes(X_prediction)
print("Number of predictions %d" % len(a))
submision_complete = np.vstack([np.array(X_id), np.array(cuisines)[a]])
submision_prediction_df = DataFrame(np.transpose(submision_complete))
submision_prediction_df.to_csv(r'submision_final' + '.csv', header=False,
                                   index=False, sep=' ',
                                   mode='a')