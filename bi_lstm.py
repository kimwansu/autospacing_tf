# -*- coding: utf-8 -*-

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.optimizers import *

from multiprocessing import cpu_count, freeze_support
from multiprocessing.pool import Pool

from make_data import make_data
from make_data import norm_many

def bi_LSTM():
    # Network building
    net = input_data(shape=[None, 440])
    net = embedding(net, input_dim=20000, output_dim=128)
    net = dropout(net, 0.9)

    net = bidirectional_rnn(net,
                            BasicLSTMCell(128, forget_bias=1.),
                            BasicLSTMCell(128, forget_bias=1.))
    net = dropout(net, 0.7)
    net = fully_connected(net, 2, activation='softmax')

    net = regression(net,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

    return net

def train(trainX, trainY, model_file):
    # Data preprocessing
    trainX = pad_sequences(trainX, maxlen=440, value=0.)
    trainY = to_categorical(trainY, nb_classes=2)

    net = bi_LSTM()

    # Training
    '''
    tensorboard_verbose:
    0: Loss, Accuracy (Best Speed)
    1: Loss, Accuracy + Gradients
    2: Loss, Accuracy, Gradients, Weights
    3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity (Best Visualization)
    '''
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128)

    # Save model
    model.save(model_file)

def interference(testX, testY, model_file):
    # Data preprocessing
    testX = pad_sequences(testX, maxlen=440, value=0.)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = bi_LSTM()

    # Load model
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
    model.load(model_file)

    # Predict
    pred = model.predict(testX)

def main():
    pool = Pool(processes=cpu_count())
    X, Y = make_data(pool, 'ted_7_ErasePunc_FullKorean__train.txt')
    print('make data end.')
    X = norm_many(pool, X)
    print('norm_data end.')
    train(X, Y, 'model.tfl')

if __name__ == '__main__':
    freeze_support()
    main()
