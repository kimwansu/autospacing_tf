# -*- coding: utf-8 --

import re
from multiprocessing import cpu_count, freeze_support
from multiprocessing.pool import Pool

from util import read_text_lines
from util import refine_line
from char2vec import load_model

B = 1
I = 0

'''
1. word2vec 모델 불러오기(from char2vec)
'''
def raw2corpus(raw_sentence):
    taggeds = []
    text = re.sub(r'(\ )+', ' ', raw_sentence).strip()
    for i in range(len(text)):
        if i == 0:
            taggeds.append('{}/B'.format(text[i]))
        elif text[i] != ' ':
            successor = text[i - 1]
            if successor == ' ':
                taggeds.append('{}/B'.format(text[i]))
            else:
                taggeds.append('{}/I'.format(text[i]))

    return ' '.join(taggeds)

def corpus2sent(line):
    sent = []
    tokens = line.split(' ')
    for token in tokens:
        if '/' not in token:
            continue

        word, tag = token.split('/')
        sent.append((word, tag))

    return sent

ngram2vec_models = []
for n in range(1, 4):
    ngram2vec_models.append(load_model(r'./char2vec_ted_d40_{}gram.txt'.format(n)))

def char2vec(ch):
    n = len(ch)
    return [float(f) for f in ngram2vec_models[n-1][ch]]

def generate_feature(args):
    line = ''.join([l[0] for l in args[0]])
    i = args[1]
    dim = 40

    feature = []

    # 1-gram
    feature += char2vec(line[i])
    feature += char2vec(line[i-1]) if i >= 1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i+1]) if i < len(line)-1 else [0.0 for a in range(dim)]

    # 2-gram
    feature += char2vec(line[i-2:i]) if i >= 2 else [0.0 for a in range(dim)]
    feature += char2vec(line[i-1:i+1]) if i >= 1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i:i+2]) if i < len(line)-1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i+1:i+3]) if i < len(line)-2 else [0.0 for a in range(dim)]

    # 3-gram
    feature += char2vec(line[i-2:i+1]) if i >= 2 else [0.0 for a in range(dim)]
    feature += char2vec(line[i-1:i+2]) if i >= 1 and i < len(line)-1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i:i+3]) if i < len(line)-2 else [0.0 for a in range(dim)]
    feature += char2vec(line[i+1:i+4]) if i < len(line)-3 else [0.0 for a in range(dim)]

    return feature

def make_data(pool, fname):
    lines = read_text_lines(fname)
    lines = [refine_line(line) for line in lines]
    corpus = [raw2corpus(line) for line in lines]
    sent = [corpus2sent(line) for line in corpus]

    X = []
    Y = []

    for line in sent:
        X += pool.map(generate_feature, [(line, i) for i in range(len(line))])
        Y += [ (B if y == 'B' else I) for _, y in line]

    return X, Y

# TODO: 무시한 특수문자로 인해 길이가 차이나는걸 고려하기
def restore_data(line, Y):
    s = []
    for l, y in zip(line, Y):
        if y:
            s.append(' ')

        s.append(l)

    return ''.join(s).strip()

def norm(arr):
    return [ round(x*1000, 0) + 10000 for x in arr ]

def norm_many(pool, X):
    return list(pool.map(norm, X))

def main():
    for i in range(1, 4):
        char2vec_model = load_model(r'./char2vec_ted_d40_{}gram.txt'.format(i))

        min_v = 0.0
        max_v = 0.0
        for k in char2vec_model.wv.vocab.keys():
            vec = char2vec_model.wv[k]
            for v in vec:
                if v < min_v:
                    min_v = v
                elif v > max_v:
                    max_v = v

        print('#{}: min={}, max={}'.format(i, min_v, max_v))

    pool = Pool(processes=cpu_count())
    X, Y = make_data(pool, r'./ted_7_ErasePunc_FullKorean__train.txt')
    print(X[:5])
    print(Y[:5])

if __name__ == '__main__':
    freeze_support()
    main()
