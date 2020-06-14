#! python
# -*- coding: utf-8 -*-

import sys
import argparse
import json
sys.path.append('..')
from inference.nlc_transformer import NLC

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
    for i in range(len(a_input)):
        mr = a_input[i].rstrip('\n').split('\t')[0]
        text = a_input[i].rstrip('\n').split('\t')[1]

        flag = True
        if args.mode.lower() == 'nlg':
            nlg_text = NLG.convert(mr, args.mode.upper(), args.d)
            fo.write(mr+'\t'+text+'\t'+nlg_text+'\t')
            if text.lower() != nlg_text.lower():
                flag = False
        else:
            nlu_mr = NLU.convert(text, args.mode.upper(), args.d)
            input_data = text.split(' ')
            fo.write(text+'\t'+mr+'\t'+nlu_mr+'\t')
            a_mr = mr.split('|')
            a_nlu_mr = nlu_mr.split('|')


            if len(a_mr) != len(a_nlu_mr):
                flag = False
            else:
                for j in range(len(a_mr)):
                    if a_mr[j] != a_nlu_mr[j]:
                        flag = False
                        break
        if flag is True:
            num_ok += 1
        count += 1
        fo.write(str(flag)+'\n')
    print('** result **')
    print(' score = '+str(100.0*num_ok/count))
    fo.write(str(100.0*num_ok/count)+'\n')
    print('** done **')
    fo.close()
