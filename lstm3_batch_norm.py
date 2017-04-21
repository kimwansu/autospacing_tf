# -*- coding: utf-8 -*-

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

from make_data import make_data
from make_data import norm_many

def bi_LSTM():
    # Network building
    net = input_data(shape=[None, 200])
    net = embedding(net, input_dim=20000, output_dim=128)
    net = dropout(net, 0.8)

    #net = lstm(net, 128, return_seq=True)
    #net = batch_normalization(net)
    #net = lstm(net, 128, return_seq=True)
    net = lstm(net, 128)
    net = batch_normalization(net)
    #net = dropout(net, 0.5)

    net = fully_connected(net, 2, activation='softmax')
    net = regression(net,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.01)

    return net

def train(trainX, trainY, model_file):
    # Data preprocessing
    trainX = pad_sequences(trainX, maxlen=200, value=0.)
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
    model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=64)

    # Save model
    model.save(model_file)

def interference(testX, testY, model_file):
    # Data preprocessing
    testX = pad_sequences(testX, maxlen=200, value=0.)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = bi_LSTM()

    # Load model
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
    model.load(model_file)

    # Predict
    pred = model.predict(testX)

def main():
    X, Y = make_data('Etri_corpus_complete__train.txt')
    X = norm_many(X)
    train(X, Y, 'model.tfl')

if __name__ == '__main__':
    main()
