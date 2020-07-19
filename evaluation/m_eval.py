#! python
# -*- coding: utf-8 -*-

import sys
import argparse
import json
sys.path.append('..')
from inference.nlc_transformer import NLC
import nltk

def __normalize_text(input_text):
    output_text = output_text.lower()
    return output_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', help='mode (NLG(default) or NLU)', default='NLG')
    parser.add_argument('-p', help='parameter directory\'s name', default='../parameter/e2e')
    parser.add_argument('-data', help='evaluation test data', default='../corpus/e2e_test.tsv')
    parser.add_argument('-result', help='result file', default='result.tsv')
    parser.add_argument('-d', help='display attention map', action='store_true')
    args = parser.parse_args()

    print('** NLC evaluation **')
    print(' mode       : '+args.mode)
    print(' parameter  : '+args.p)
    print(' test data  : '+args.data)
    print(' result file: '+args.result)
    if args.mode.lower() == 'nlg':
        NLG = NLC(args.p, args.mode.upper())
    else:
        NLU = NLC(args.p, args.mode.upper())

    f_test = open(args.data, 'r', encoding='utf-8')
    a_input = f_test.readlines()
    f_test.close()

    fo = open(args.result, 'w', encoding='utf-8')
    fo.write('input\toutput(correct)\toutput(predict)\tresult\n')
    num_ok = 0
    count = 0
    total_score_bleu = 0.0
    total_score_meteor = 0.0
    total_score = [[0]*5]*8
    for i in range(len(a_input)):
        mr = a_input[i].rstrip('\n').split('\t')[0]
        text = a_input[i].rstrip('\n').split('\t')[1]

        flag = True
        if args.mode.lower() == 'nlg':
            nlg_text = NLG.convert(mr, args.mode.upper(), args.d)
            fo.write(mr+'\t'+text+'\t'+nlg_text+'\t')
            if text.lower() != nlg_text.lower():
                flag = False
            fo.write(str(flag)+'\t')
            score_bleu = nltk.translate.bleu_score.sentence_bleu([text.lower()], nlg_text.lower())
            fo.write(str(100*score_bleu)+'\t')
            total_score_bleu += score_bleu
            score_meteor = nltk.translate.meteor_score.single_meteor_score(text.lower(), nlg_text.lower())
            fo.write(str(100*score_meteor)+'\n')
            total_score_meteor += score_meteor

        else:
            nlu_mr = NLU.convert(text, args.mode.upper(), args.d)
            input_data = text.split(' ')
            fo.write(text.lower()+'\t'+mr.lower()+'\t'+nlu_mr.lower()+'\t')
            a_mr = mr.lower().split('|')
            a_nlu_mr = nlu_mr.lower().split('|')

            # precision
            if len(a_mr) != len(a_nlu_mr):
                flag = False
            else:
                for j in range(len(a_mr)):
                    if a_mr[j] != a_nlu_mr[j]:
                        flag = False
                        break
            fo.write(str(flag)+'\t')
            a_score = [[0]*5]*8

            len_mr = len(a_mr)
            if len_mr > len(a_nlu_mr):
                len_mr = len(a_nlu_mr)
            for j in range(len_mr):
                if a_mr[j] == 'no':
                    if a_nlu_mr[j] == 'no':
                        score = 0
                    else:
                        score = 1
                else:
                    if a_nlu_mr[j] == 'no':
                        score = 2
                    elif a_nlu_mr[j] == a_mr[j]:
                        score = 3
                    else:
                        score = 4
                a_score[j][score] = 1
                total_score[j][score] += 1
            for j in range(len_mr, 8):
                if a_mr[j] == 'no':
                    score = 1
                else:
                    score = 4
                a_score[j][score] = 1
                total_score[j][score] += 1
            for j in range(8):
                for k in range(5):
                    fo.write('\t'+str(a_score[j][k]))
            fo.write('\n')

        if flag is True:
            num_ok += 1
        count += 1

    print('** result **')
    print(' score = '+str(100.0*num_ok/count))
    fo.write(str(100.0*num_ok/count)+'\n')
    print('** done **')
    fo.close()
