# -*- coding: utf-8 -*-

import codecs
import sys
import re

import h5py

import numpy as np

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell, GRUCell
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.optimizers import *

from multiprocessing import cpu_count, freeze_support
from multiprocessing.pool import Pool

from make_data import make_data, make_data_divided, norm_many
from util import read_text_lines, refine_line


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
    print('# Data preprocessing')
    trainX = pad_sequences(trainX, maxlen=440, value=0.)
    trainY = to_categorical(trainY, nb_classes=2)

    print('build network')
    net = bi_LSTM()

    print('# Training')
    '''
    tensorboard_verbose:
    0: Loss, Accuracy (Best Speed)
    1: Loss, Accuracy + Gradients
    2: Loss, Accuracy, Gradients, Weights
    3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity (Best Visualization)
    '''
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0,
                        checkpoint_path='./chkpoint_mdm001/',
                        best_checkpoint_path='./best_chkpoint_mdm001/',
                        best_val_accuracy=0.9)
    print('tfl.DNN end.')

    model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128,
              n_epoch=4, run_id='bilstm_170519b')
    print('model.fit end.')

    # Save model
    model.save(model_file)
    print('model save end.')


class Trainer():
    def __init__(self):
        print('train_diviced')
        print('# Network building')
        self.net = bi_LSTM()
        self.model = tflearn.DNN(self.net, clip_gradients=0., tensorboard_verbose=0,
                                 checkpoint_path='./chkpoint_b/',
                                 best_checkpoint_path='./best_chkpoint_b/',
                                 best_val_accuracy=0.9)
        print('tfl.DNN end.')
        self.i = 0

    def train(self, trainX, trainY):
        print('# Data preprocessing')
        trainX = pad_sequences(trainX, maxlen=440, value=0.)
        trainY = to_categorical(trainY, nb_classes=2)
        print('data preproc end.')

        self.model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128,
                       n_epoch=1, run_id='bilstm_170524mdm001')
        print('model.fit #{} end'.format(self.i))
        self.i += 1

    def save(self, model_file):
        self.model.save(model_file)
        print('model save end.')


def interference(testX, testY, model_file):
    print('interference')
    print('# Data preprocessing')
    testX = pad_sequences(testX, maxlen=440, value=0.)
    testY = to_categorical(testY, nb_classes=2)

    print('# Network building')
    net = bi_LSTM()

    print('# Load model')
    model = tflearn.DNN(net)
    model.load(model_file)
    if not model:
        print('model not loaded')
        sys.exit(1)
    else:
        print('model load.')

    print('# Predict')
    pred = model.predict(testX)
    new_y = np.argmax(pred, axis=1)
    result = new_y.astype(np.uint8)
    print('predict end.')
    result = str(result)
    print('pred to str.')
    with codecs.open('test_result.txt', 'w', encoding='utf-8') as wfh:
        wfh.write(result)

    print('end.')


class Tagger():
    def __init__(self, model_file):
        print('interference_divided')
        print('# Network building')
        self.net = bi_LSTM()

        print('# Load model')
        self.model = tflearn.DNN(self.net)
        self.model.load(model_file)
        if not self.model:
            print('model not loaded')
            sys.exit(1)
        else:
            print('model load.')

    def interference(self, testX):
        print('# Data preprocessing')
        testX = pad_sequences(testX, maxlen=440, value=0.)

        print('# Predict')
        pred = self.model.predict(testX)
        new_y = np.argmax(pred, axis=1)
        result = (int(y) for y in new_y.astype(np.uint8))

        return result


def run_train(train_file):
    print('train')
    pool = Pool(processes=cpu_count())
    X, Y = make_data(pool, train_file)
    print('make train data end.')
    X = norm_many(pool, X)
    print('norm_data end.')
    train(X, Y, 'model_MDM001.tfl')


def run_train_divided(train_file):
    print('train')
    pool = Pool(processes=cpu_count())
    trainer = Trainer()
    epoch = 4
    for i in range(epoch):
        for X, Y in make_data_divided(pool, train_file):
            print('epoch: {}'.format(epoch))
            trainer.train(X, Y)

    trainer.save('model_MDM001.tfl')


def run_test():
    print('test')
    pool = Pool(processes=cpu_count())
    X, Y = make_data(pool, 'ted_7_ErasePunc_FullKorean__test.txt')
    print('make test data end.')
    X = norm_many(pool, X)
    print('norm_data end.')
    interference(X, Y, 'model.tfl')


def run_test_divided(test_file):
    print('test')
    pool = Pool(processes=cpu_count())
    tagger = Tagger('model.tfl')
    for X, _ in make_data_divided(pool, test_file):
        y = (str(r) for r in tagger.interference(X))
        # y는 문장 구분 없이 한번에 다 들어오므로
        # X의 각 문장의 글자수 단위로 끊는다.
        # 그 다음에 y의 내용으로 원문을 복원한다.
        yield ''.join(y)


def main():
    if len(sys.argv) < 2:
        print('usage: bi_lstm.py (train|test|make)')
        sys.exit(1)

    if sys.argv[1] == 'train':
        train_file = 'MDM001_FullKorean__train.txt'
        #run_train(train_file)
        run_train_divided(train_file)
    elif sys.argv[1] == 'test':
        test_file = 'ted_7_ErasePunc_FullKorean__test.txt'
        lines = read_text_lines(test_file)
        lines = (refine_line(line) for line in lines)
        lines = [re.sub(r'[\ \n\r]+', '', line).strip() for line in lines]

        i = 0
        with codecs.open('ted_test_result.txt', 'w', encoding='utf-8') as wfh:
            for Y in run_test_divided(test_file):
                # Y의 길이와 lines의 길이를 확인해가면서 합치기
                # 아니면 Y가 10000줄 처리한 단위로 나오니까 10000줄씩 읽어서 대조해보기

                y_pos = 0
                buf = []
                while True:
                    '''
                    Y가 있는 만큼만 line을 진행시켜서 해보기
                    '''
                    line = lines[i]

                    result = ''
                    line_y = Y[y_pos:y_pos+len(line)]
                    for ch, y in zip(line, line_y):
                        if y == '1':
                            result += ' ' + ch
                        else:
                            result += ch

                    buf.append(result.strip())

                    y_pos += len(line)
                    i += 1
                    if y_pos >= len(Y):
                        break

                wfh.write('\n'.join(buf) + '\n')
    elif sys.argv[1] == 'make':
        make_file = 'MDM001_FullKorean__train.txt'
        lines = read_text_lines(make_file)
        lines = (refine_line(line) for line in lines)
        lines = [re.sub(r'[\ \n\r]+', '', line).strip() for line in lines]

        i = 0
        pool = Pool(processes=cpu_count())
        X = []
        Y = []
        for x, y in make_data_divided(pool, make_file):
            x = norm_many(pool, x)
            x = pad_sequences(x, maxlen=440, value=0.)
            if len(X) > 0:
                X = np.concatenate((X, x), axis=0)
            else:
                X = x

            print('{}) x'.format(i), end=', ')
            y = to_categorical(y, nb_classes=2)
            if len(Y) > 0:
                Y = np.concatenate((Y, y), axis=0)
            else:
                Y = y

            print('y')
            i += 1

        # TODO: 파일 이름, 데이터셋 이름 바꾸기
        #h5f = h5py.File('ted_train.h5', 'w')
        #h5f.create_dataset('ted7_X', data=X)
        #h5f.create_dataset('ted7_Y', data=Y)
        h5f = h5py.File('ted_MDM001.h5', 'w')
        h5f.create_dataset('MDM001_X', data=X)
        h5f.create_dataset('MDM001_Y', data=Y)
        h5f.close()
    else:
        print('usage: bi_lstm.py (train|test|make)')


if __name__ == '__main__':
    print('hello')
    print(sys.argv[1])
    #input()
    freeze_support()
    main()
