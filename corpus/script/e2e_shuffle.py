#! python

import numpy
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-itrain', help='[input] training dataset(csv)', default='../e2e-dataset/trainset_fix.csv')
    parser.add_argument('-ivalid', help='[input] validataion dataset(csv)', default='../e2e-dataset/devset.csv')
    parser.add_argument('-itest', help='[input] test dataset(csv)', default='../e2e-dataset/testset_w_refs.csv')
    parser.add_argument('-otrain', help='[output] training dataset(tsv)', default='../e2e_train.tsv')
    parser.add_argument('-ovalid', help='[output] validataion dataset(tsv)', default='../e2e_valid.tsv')
    parser.add_argument('-otest', help='[output] test dataset(tsv)', default='../e2e_test.tsv')
    args = parser.parse_args()

    print('** convert E2E dataset **')
    print('[input]')
    print(' training             :'+args.itrain)
    print(' validation           :'+args.ivalid)
    print(' test                 :'+args.itest)
    print('[output]')
    print(' training             :'+args.otrain)
    print(' validation           :'+args.ovalid)
    print(' test                 :'+args.otest)

    fi = open(args.itrain, 'r', encoding='utf-8')
    a_input0 = fi.readlines()
    fi.close()
    fi = open(args.ivalid, 'r', encoding='utf-8')
    a_input1 = fi.readlines()
    fi.close()
    fi = open(args.itest, 'r', encoding='utf-8')
    a_input2 = fi.readlines()
    fi.close()
    a_input = a_input0+a_input1+a_input2
    length = len(a_input)

    # 8:1:1
    length_split_A = int(length*0.80)
    length_split_B = int(length*0.90)
    '''
    # 4:1:1
    length_split_A = int(length*4.0/6.0)
    length_split_B = int(length*5.0/6.0)
    '''
    indices = numpy.arange(length)
    # 1234
    numpy.random.seed(1234)
    # 123
    #numpy.random.seed(123)

    numpy.random.shuffle(indices)

    fo = open(args.otrain, 'w', encoding='utf-8')
    fo_num = open(args.otrain+'.num', 'w', encoding='utf-8')
    for i in range(length_split_A):
        fo.write(a_input[indices[i]].rstrip('\n')+'\n')
        fo_num.write(str(indices[i])+'\n')
    fo.close()
    fo_num.close()

    fo = open(args.ovalid, 'w', encoding='utf-8')
    fo_num = open(args.ovalid+'.num', 'w', encoding='utf-8')
    for i in range(length_split_A, length_split_B):
        fo.write(a_input[indices[i]].rstrip('\n')+'\n')
        fo_num.write(str(indices[i])+'\n')
    fo.close()
    fo_num.close()

    fo = open(args.otest, 'w', encoding='utf-8')
    fo_num = open(args.otest+'.num', 'w', encoding='utf-8')
    for i in range(length_split_B, length):
        fo.write(a_input[indices[i]].rstrip('\n')+'\n')
        fo_num.write(str(indices[i])+'\n')
    fo.close()
    fo_num.close()
