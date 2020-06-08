#! python

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nlu', help='NLU', action='store_true')
    parser.add_argument('-nlg', help='NLG', action='store_true')
    parser.add_argument('-i', help='input file name')
    parser.add_argument('-o', help='output file name')
    args = parser.parse_args()

    fi = open(args.i, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    fo = open(args.o, 'w', encoding='utf-8')

    num_ok = 0
    count = 0
    if args.nlg is True:
        fo.write('result\t')
        fo.write(a_input[0])
        for i in range(1, len(a_input)):
            mr = a_input[i].rstrip('\n').split('\t')[0].lstrip('\"[').rstrip(']\"')
            sen_correct = a_input[i].rstrip('\n').split('\t')[1].lstrip('\"[').rstrip(']\"').split(', ')
            sen_predict = a_input[i].rstrip('\n').split('\t')[2].lstrip('\"[').rstrip(']\"').split(', ')
            len_correct = len(sen_correct)
            len_predict = len(sen_predict) - 1

            flag = True
            if len_correct != len_predict:
                flag = False
            else:
                for j in range(len_correct):
                    if sen_correct[j] != sen_predict[j]:
                        flag = False
                        break
            if flag is True:
                num_ok += 1
                fo.write('ok\t')
            else:
                fo.write('ng\t')
            fo.write(a_input[i])
            count += 1
        print(args.i)
        print('ok   : '+str(num_ok))
        print('count: '+str(count))
        if count > 0:
            print('score: '+str(100.0*num_ok/count))
            fo.write('score\t'+str(100.0*num_ok/count)+'\n')
        else:
            print('score: 0.0')
            fo.write('score\t0\n')
        fo.close()
    elif args.nlu is True:
        fo.write('result\t')
        fo.write(a_input[0])
        for i in range(1, len(a_input)):
            sen = a_input[i].rstrip('\n').split('\t')[0].lstrip('\"[').rstrip(']\"')
            mr_correct = a_input[i].rstrip('\n').split('\t')[1].lstrip('\"[').rstrip(']\"').split(', ')
            mr_predict = a_input[i].rstrip('\n').split('\t')[2].lstrip('\"[').rstrip(']\"').split(', ')
            len_correct = len(mr_correct)
            len_predict = len(mr_predict) - 1

            flag = True
            if len_correct != len_predict:
                flag = False
            else:
                for j in range(len_correct):
                    if mr_correct[j] != mr_predict[j]:
                        flag = False
                        break
            if flag is True:
                num_ok += 1
                fo.write('ok\t')
            else:
                fo.write('ng\t')
            fo.write(a_input[i])
            count += 1
        print(args.i)
        print('ok   : '+str(num_ok))
        print('count: '+str(count))
        if count > 0:
            print('score: '+str(100.0*num_ok/count))
            fo.write('score\t'+str(100.0*num_ok/count)+'\n')
        else:
            print('score: 0.0')
            fo.write('score\t0\n')
        fo.close()
