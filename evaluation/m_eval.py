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
    parser.add_argument('-chain', help='NL chain', action='store_true')
    parser.add_argument('-d', help='display attention map', action='store_true')
    args = parser.parse_args()

    print('** NLC evaluation **')
    print(' mode       : '+args.mode)
    print(' parameter  : '+args.p)
    print(' test data  : '+args.data)
    print(' result file: '+args.result)
    print(' chain mode : '+str(args.chain))
    if args.mode.lower() == 'nlg':
        NLG = NLC(args.p, args.mode.upper(), args.chain)
    else:
        NLU = NLC(args.p, args.mode.upper(), args.chain)

    f_test = open(args.data, 'r', encoding='utf-8')
    a_input = f_test.readlines()
    f_test.close()

    fo = open(args.result, 'w', encoding='utf-8')
    fo.write('input\toutput(correct)\toutput(predict)\tresult\n')
    num_ok = 0
    count = 0
    '''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    '''
    for i in range(len(a_input)):
        mr = a_input[i].rstrip('\n').split('\t')[0]
        text = a_input[i].rstrip('\n').split('\t')[1]

        flag = True
        if args.mode.lower() == 'nlg':
            nlg_text = NLG.convert(mr, args.mode.upper(), args.d)
            fo.write(mr+'\t'+text+'\t'+nlg_text+'\t')
            if text.lower() != nlg_text.lower():
                flag = False
            fo.write(str(flag)+'\n')
        else:
            nlu_mr = NLU.convert(text, args.mode.upper(), args.d)
            input_data = text.split(' ')
            fo.write(text.lower()+'\t'+mr.lower()+'\t'+nlu_mr.lower()+'\t')
            a_mr = mr.lower().split('|')
            a_nlu_mr = nlu_mr.lower().split('|')

            length_min = min(len(a_mr), len(a_nlu_mr))
            length_max = max(len(a_mr), len(a_nlu_mr))

            if len(a_mr) != len(a_nlu_mr):
                flag = False
            for j in range(length_min):
                if a_mr[j] != a_nlu_mr[j]:
                    flag = False
                    break
            fo.write(str(flag)+'\n')
            '''
            if len(a_mr) == len(a_nlu_mr):
                mode = 0
            elif len(a_mr) < len(a_nlu_mr):
                mode = 1
            else:
                mode = 2
            for j in range(length_min):
                if a_mr[j] != '':
                    if a_mr[j] == a_nlu_mr[j]:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if a_mr[j] == a_nlu_mr[j]:
                        TN += 1
                    else:
                        FP += 1
            for j in range(length_min, length_max):
                if mode == 1:
                    FP += 1
                elif mode == 2:
                    FN += 1
            '''
        if flag is True:
            num_ok += 1
        count += 1

    '''
    if args.mode.lower() == 'nlu':
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if precision + recall > 0:
            f1score = (2.0 * precision * recall) / (precision + recall)
        else:
            f1score = 0
        if TP + FP + FN > 0:
            accuracy = TP / (TP + FP + FN)
        else:
            accuracy = 0
    '''
    print('** result **')
    print(' score(frame) = '+str(100.0*num_ok/count))
    '''
    if args.mode.lower() == 'nlu':
        print(' f1score = '+str(f1score))
        print(' precision = '+str(precision))
        print(' recall = '+str(recall))
        print(' accuracy = '+str(accuracy))
    '''
    print('** done **')
    fo.close()
