# -*- coding: utf-8 --

import re
import math
from multiprocessing import cpu_count, freeze_support
from multiprocessing.pool import Pool
import sys

from util import read_text_lines
from util import refine_line
from char2vec import load_model


B = 1
I = 0

'''
1. word2vec 모델 불러오기(from char2vec)
'''


def is_hangul(ch):
    codepoint = ord(ch) if isinstance(ch, str) else int(ch)
    return codepoint >= 0xac00 and codepoint <= 0xd7a3


def is_ascii(ch):
    codepoint = ord(ch) if isinstance(ch, str) else int(ch)
    return codepoint >= 0x20 and codepoint <= 0x7e


def ch2num(ch):
    codepoint = ord(ch) if isinstance(ch, str) else ch
    if is_hangul(ch):
        return codepoint - ord('가') + 256
    elif is_ascii(ch):
        return codepoint
    else:
        return None


def get_features(line_ch, i):
    X = [0 for i in range(6)]

    if i > 2:
        X[0] = ch2num(line_ch[i - 2])

    if i > 1:
        X[1] = ch2num(line_ch[i - 1])

    X[2] = ch2num(line_ch[i])

    if i < len(line_ch) - 1:
        X[3] = ch2num(line_ch[i + 1])

    if i < len(line_ch) - 2:
        X[4] = ch2num(line_ch[i + 2])

    # 문장의 시작 위치 기록
    if i == 0:
        X[5] = 1
    else:
        X[5] = 0

    return X


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

# 이건 사용 안함
char2vec_model = load_model(r'./char2vec_Etri_d30.txt')


ngram2vec_models = []
for n in range(1, 4):
    #ngram2vec_models.append(load_model(r'./char2vec_Etri_d30_{}gram.txt'.format(n)))
    #ngram2vec_models.append(load_model(r'./char2vec_ted_d40_{}gram.txt'.format(n)))
    ngram2vec_models.append(load_model(r'./char2vec_MDM001_d40_{}gram.txt'.format(n)))


def char2vec(ch):
    n = len(ch)
    return [float(f) for f in ngram2vec_models[n-1][ch]]


# 조화 평균
def hmean(values):
    top = float(len(values))
    bottom = 0.0
    for v in values:
        top *= v
        bottom += v

    return top / bottom


# 산술 평균
def amean(values):
    s = 0.0
    for v in values:
        s += v

    return v / len(values)


# 기하 평균
def gmean(values):
    m = 1
    for v in values:
        m *= v

    r = m ** (1.0/float(len(values)))
    return r


def index2feature(line, i, offsets):
    '''
    해당 offset에 위치한 글자의 word embedding 벡터를 가져온다.
    offset이 여러 개 있으면 중간값이나 평균(산술, 조화 등)값으로 합쳐서 실험해보기
    * 중간값: 다른 조합인데 같은 걸로 취급될 수 있는 경우가 있으므로 빼기
    * 실험1 --> 산술평균: (a + b + c) / 3
    * 실험2 --> 조화평균: 3*a*b*c / (a + b + c) --> 모두 양수일때만 의미있는 결과 나옴
    * 실험3 --> 기하평균: sqrt3(a * b * c)

    ※ 기하평균 --> 곱해야 하는 값의 평균 구할때 사용(예: 은행 n년간 평균 이자 계산 등)
    '''
    vec = []
    for off in offsets:
        if i + off < 0 or i + off >= len(line):
            return [0.0 for i in range(50)]

        ch, _ = line[i + off]
        vec.append(char2vec_model[ch])

    result = []
    for i in range(len(vec[0])):
        v = []
        for j in range(len(vec)):
            v.append(float(vec[j][i]))

        result.append(amean(v))

    return result


# 다른 논문 참고한 자질에서 2개이상 글자에 해당하는 임베딩은 각 글자의
# 임베딩 정보를 평균낸걸로 만든 자질
def generate_feature(args):
    line = args[0]
    i = args[1]

    feature = []

    feature += index2feature(line, i, [-1])
    feature += index2feature(line, i, [0])
    feature += index2feature(line, i, [1])
    feature += index2feature(line, i, [-2, -1])
    feature += index2feature(line, i, [-1, 0])
    feature += index2feature(line, i, [0, 1])
    feature += index2feature(line, i, [-2, -1, 0])
    feature += index2feature(line, i, [-1, 0, 1])
    feature += index2feature(line, i, [0, 1, 2])

    return feature


# 앞 2글자부터 뒤2글자까지 각 한글자씩의 임베딩 정보를 자질로 사용한 것
def generate_feature2(args):
    line = args[0]
    i = args[1]

    feature = []

    if i >= 2:
        ch, _ = line[i - 2]
        feature += char2vec(ch)
    else:
        feature += [0.0 for i in range(30)]

    if i >= 1:
        ch, _ = line[i - 1]
        feature += char2vec(ch)
    else:
        feature += [0.0 for i in range(30)]

    ch, _ = line[i]
    feature += char2vec(ch)

    if i < len(line) - 1:
        ch, _ = line[i + 1]
        feature += char2vec(ch)
    else:
        feature += [0.0 for i in range(30)]

    if i < len(line) - 2:
        ch, _ = line[i + 2]
        feature += char2vec(ch)
    else:
        feature += [0.0 for i in range(30)]

    return feature



# 타 논문 참고한 자질에서 여러 글자에 해당하는 임베딩 정보를 추가한 자질
def generate_feature3(args):
    line = ''.join([l[0] for l in args[0]])
    i = args[1]
    dim = 40

    feature = []

    # 1-gram
    feature += char2vec(line[i-1]) if i >= 1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i])
    feature += char2vec(line[i+1]) if i < len(line)-1 else [0.0 for a in range(dim)]

    # 2-gram
    feature += char2vec(line[i-2:i]) if i >= 2 else [0.0 for a in range(dim)]
    feature += char2vec(line[i-1:i+1]) if i >= 1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i:i+2]) if i < len(line)-1 else [0.0 for a in range(dim)]

    # 3-gram
    feature += char2vec(line[i-2:i+1]) if i >= 2 else [0.0 for a in range(dim)]
    feature += char2vec(line[i-1:i+2]) if i >= 1 and i < len(line)-1 else [0.0 for a in range(dim)]
    feature += char2vec(line[i:i+3]) if i < len(line)-2 else [0.0 for a in range(dim)]

    return feature


def generate_feature4(args):
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
    lines = (refine_line(line) for line in lines)
    corpus = (raw2corpus(line) for line in lines)
    sent = (corpus2sent(line) for line in corpus)

    X = []
    Y = []

    for line in sent:
        X += pool.map(generate_feature4, [(line, i) for i in range(len(line))])
        Y += [(1 if y == 'B' else 0) for _, y in line]

    return X, Y


def make_data_divided(pool, fname):
    lines = read_text_lines(fname)
    lines = (refine_line(line) for line in lines)
    corpus = (raw2corpus(line) for line in lines)
    sent = (corpus2sent(line) for line in corpus)

    line_cnt = 0
    X = []
    Y = []
    for line in sent:
        line_cnt += 1
        x = pool.map(generate_feature4, [(line, i) for i in range(len(line))])
        X += norm_many(pool, x)
        Y += ((1 if y == 'B' else 0) for _, y in line)
        if line_cnt == 100000:
            yield X, Y
            line_cnt = 0
            X = []
            Y = []

    yield X, Y


# todo: 여러 글자의 워드벡터를 더한 것을 고려해서 수정하기
def norm(arr):
    return [round(x*1000, 0) + 10000 for x in arr]


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
    #sys.exit(1)

    pool = Pool(processes=cpu_count())
    X, Y = make_data(pool, r'./ted_7_ErasePunc_FullKorean__train.txt')
    print(X[:5])
    print(Y[:5])


if __name__ == '__main__':
    freeze_support()
    main()
