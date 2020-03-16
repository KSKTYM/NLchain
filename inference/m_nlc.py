#! python
# -*- coding: utf-8 -*-

import sys
import argparse
import nlc_transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', help='mode (NLG(default) or NLU)', default='NLG')
    parser.add_argument('-p', help='parameter directory\'s name', default='../parameter/e2e')
    parser.add_argument('-corpus', help='corpus name', default='e2e')
    parser.add_argument('-d', help='display attention map', action='store_true')
    args = parser.parse_args()

    print('** NLconverter: NLG or NLU (trained by NLchain) ***')
    print(' mode          : '+str(args.mode))
    print(' parameter dir : '+str(args.p))
    print(' corpus name   : '+str(args.corpus))

    if args.mode == 'NLG':
        f_slot = open(args.p+'/'+args.corpus+'_slot', 'r', encoding='utf-8')
        a_slot = f_slot.readlines()
        f_slot.close()
        desc = ''
        for i in range(len(a_slot)):
            if i > 0:
                desc += '|'
            desc += a_slot[i].rstrip('\n')
        print(desc)

    NLC = nlc_transformer.NLC(args.p, args.mode)
    while True:
        input_data = sys.stdin.readline().rstrip('\n')
        output_data = NLC.convert(input_data, args.mode, args.d)
        print(output_data)
