'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

Get to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''

from __future__ import print_function
import numpy as np
import sys
sys.path.append("../datasets")
import cooking

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from datasets import cooking
from pandas import DataFrame
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU

# Embedding
max_features = 20000
maxlen = 65
embedding_size = 300

# Convolution
filter_length = 3
nb_filter = 150
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 50
nb_epoch = 8

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print("Loading data...")
(X_train, y_train), (X_test, y_test), (X_prediction, X_id), vocab, cuisines = cooking.load_data(train_path="../train.json",
                                                                                                test_path="../test_json", test_split=0.0)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
max_features = len(vocab)
print("Pad sequences (samples x time)")
nb_classes = np.max(y_train) + 1

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_prediction = sequence.pad_sequences(X_prediction, maxlen=maxlen)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')
print('Build model...')
model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

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
