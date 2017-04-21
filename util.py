# -*- coding: utf-8 -*-

import codecs
import re

def read_text_lines(fname, enc='utf-8'):
    with codecs.open(fname, 'r', encoding=enc) as rfh:
        return rfh.readlines()

'''
말뭉치를 정제한다.
* 특수 문자 제거
* 중복된 빈칸, 줄의 양 끝에 있는 빈칸 제거
* 한자 --> 한글 변환(TODO)
'''
def refine_line(line):
    line = line.strip()
    line = re.sub(r'[\ \t]+', ' ', line).strip()
    # 한글과 아스키 코드에서 출력 가능한 문자만 남긴다.
    # 아스키 코드에서 출력 가능한 문자는 0x20-0x7e까지이다.
    return re.sub(r'[^가-힣\u0020-\u007e]+', '', line)
