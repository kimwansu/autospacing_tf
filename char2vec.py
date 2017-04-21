# coding: utf-8

import re
from multiprocessing import cpu_count

import gensim

from util import read_text_lines
from util import refine_line

'''
1. 말뭉치 불러오기
2. 말뭉치 정제하기(특수문자 제거, 중복 빈칸, 앞뒤 빈칸 제거)
3. word2vec에 넣을 수 있는 형식으로 변환하기
4. word2vec학습 실행
5. 모델 저장
'''

def make_ngram2vec_corpus(lines, n):
    if n < 1 or n >= len(lines):
        return []
    else:
        return [[line[i:i+n] for i in range(len(line)-(n-1))] for line in lines]

def make_ngram2vec_model(lines, wv_dim):
    r = re.compile(r' ')
    # ngram을 학습시키도록 빈 칸을 없앤다.
    lines = [r.sub('', refine_line(line)) for line in lines]
    models = []
    for n in range(1, 4):
        corpus = make_ngram2vec_corpus(lines, n)
        model = gensim.models.Word2Vec(corpus,
                                       min_count=1,
                                       size=wv_dim,
                                       sg=1,
                                       window=5,
                                       batch_words=1000,
                                       iter=100,
                                       cbow_mean=0,
                                       workers=cpu_count())
        models.append(model)

    return models

def save_model(model, filename):
    model.Save(filename)

def load_model(filename):
    model = gensim.models.Word2Vec.load(filename)
    return model

def test(train_file, wv_dim):
    lines = read_text_lines(train_file)
    models = make_ngram2vec_model(lines, wv_dim)
    print('모델 생성 성공')
    for i, model in enumerate(models):
        model.save(r'char2vec_{}_d{}_{}gram.txt'.format(train_file.split('_')[0], wv_dim, i+1))

    print('모델 저장 성공')

if __name__ == '__main__':
    test('ted_7_ErasePunc_FullKorean.txt', 40)
