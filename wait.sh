#!/bin/bash

while [ 1 ] ; do
    sleep 30m
    if [ -f char2vec_MDM001_d40_1gram.txt ] ; then
    if [ -f char2vec_MDM001_d40_2gram.txt ] ; then
    if [ -f char2vec_MDM001_d40_3gram.txt ] ; then
    if [ `ls -1 char2vec_MDM001_d40_3gram.txt.*.npy | wc -l` == 2 ] ; then
    if [ `ps -e | grep 'python' | wc -l` == 0 ] ; then
        break
    fi
    fi
    fi
    fi
    fi
done

python bi_lstm.py train
