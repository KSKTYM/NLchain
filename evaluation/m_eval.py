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
    parser.add_argument('-d', help='display attention map', action='store_true')
    args = parser.parse_args()

    print('** NLC evaluation **')
    print(' mode      : '+args.mode)
    print(' parameter : '+args.p)
    print(' test data : '+args.data)
    if args.mode == 'NLG':
        NLG = NLC(args.p, args.mode)
    else:
        NLU = NLC(args.p, args.mode)

    f_test = open(args.data, 'r', encoding='utf-8')
    a_input = f_test.readlines()
    f_test.close()

    score = 0
    for i in range(len(a_input)):
        '''
        print('i: '+str(i))
        '''
        mr = a_input[i].rstrip('\n').split('\t')[0]
        text = a_input[i].rstrip('\n').split('\t')[1]

        if args.mode == 'NLG':
            nlg_text = NLG.convert(mr, args.mode, args.d)
            input_data = mr.split('|')
            org = text.split(' ')
            tgt = nlg_text.split(' ')
        else:
            nlu_mr = NLU.convert(text, args.mode, args.d)
            input_data = text.split(' ')
            org = mr.split('|')
            tgt = nlu_mr.split('|')
        '''
        print('input')
        print(input_data)
        print('org')
        print(org)
        print('tgt')
        print(tgt)
        '''
        if len(org) == len(tgt):
            count = 0
            for j in range(len(org)):
                if org[j].lower() == tgt[j].lower():
                    count += 1
            score += (count / len(org))
        '''
        print(str(score))
        '''
    print('** result **')
    print(' score = '+str(100.0*score/len(a_input)))
    print('** done **')
