#! python

import json
import copy
import argparse
#import sys
#sys.path.append('../..')
#from common.tokenizer import Tokenizer
import numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-itrain', help='[input] training dataset(csv)', default='../e2e-dataset/trainset_fix.csv')
    parser.add_argument('-ivalid', help='[input] validataion dataset(csv)', default='../e2e-dataset/devset.csv')
    parser.add_argument('-itest', help='[input] test dataset(csv)', default='../e2e-dataset/testset_w_refs.csv')
    parser.add_argument('-otrain', help='[output] training dataset(tsv)', default='../e2e_train.tsv')
    parser.add_argument('-ovalid', help='[output] validataion dataset(tsv)', default='../e2e_valid.tsv')
    parser.add_argument('-otest', help='[output] test dataset(tsv)', default='../e2e_test.tsv')
    parser.add_argument('-otrain_aug', help='[output] augmented training dataset(tsv)', default='../e2e_train_aug.tsv')
    parser.add_argument('-omr', help='[output] MR data(json)', default='../e2e_mr.json')
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
    print(' training(augumented) :'+args.otrain_aug)
    print(' MR                   :'+args.omr)

    #tokenizer = Tokenizer()
    ## training dataset
    # (train1) input file
    fi = open(args.itrain, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()

    # (train2) MR dataset
    # (train2-1) name list of attribute
    a_attribute_list = {}
    for i in range(1, len(a_input)):
        if len(a_input[i].rstrip('\n').split('",')) > 1:
            # mr
            mr_tmp = a_input[i].rstrip('\n').split('\",')[0]
            a_mr_tmp = mr_tmp.lstrip('\"').rstrip('\"').split('], ')
            for j in range(len(a_mr_tmp)):
                if len(a_mr_tmp[j].split('[')) > 1:
                    attribute = a_mr_tmp[j].split('[')[0]
                    value = a_mr_tmp[j].split('[')[1].rstrip(']')
                    if ((attribute in a_attribute_list) is False):
                        '''
                        #no
                        a_attribute_list[attribute] = {'order': -1, 'before': [], 'after': [], 'value': ['no']}
                        '''
                        #empty
                        a_attribute_list[attribute] = {'order': -1, 'before': [], 'after': [], 'value': []}
        else:
            print(str(i)+': '+a_input[i].rstrip('\n'))

    # (train2-2) order of attribute list
    for i in range(1, len(a_input)):
        if len(a_input[i].rstrip('\n').split('",')) > 1:
            # mr
            mr_tmp = a_input[i].rstrip('\n').split('\",')[0]
            a_mr_tmp = mr_tmp.lstrip('\"').rstrip('\"').split('], ')
            a_mr = []
            for j in range(len(a_mr_tmp)):
                if len(a_mr_tmp[j].split('[')) > 1:
                    attribute = a_mr_tmp[j].split('[')[0]
                    value = a_mr_tmp[j].split('[')[1].rstrip(']')
                    a_mr.append({'attribute': attribute, 'value': value})

            for j in range(len(a_mr)):
                flag = False
                attribute_j = a_mr[j]['attribute']
                for k in range(len(a_mr)):
                    attribute_k = a_mr[k]['attribute']
                    if attribute_j == a_mr[k]['attribute']:
                        value = a_mr[j]['value']
                        if ((value in a_attribute_list[attribute_j]['value']) is False):
                            a_attribute_list[attribute_j]['value'].append(value)
                        flag = True
                    else:
                        if flag is False:
                            if ((attribute_k in a_attribute_list[attribute_j]['before']) is False):
                                a_attribute_list[attribute_j]['before'].append(attribute_k)
                        else:
                            if ((attribute_k in a_attribute_list[attribute_j]['after']) is False):
                                a_attribute_list[attribute_j]['after'].append(attribute_k)

    for attribute in a_attribute_list:
        a_attribute_list[attribute]['order'] = len(a_attribute_list[attribute]['before'])

    #f = open('attribute_listC.json', 'w', encoding='utf-8')
    #json.dump(a_attribute_list, f, ensure_ascii=False, indent=4)
    #f.close()

    # (train2-3) MR data output
    a_mrdata = []
    num = 0
    while (num < len(a_attribute_list)):
        for attribute in a_attribute_list:
            if a_attribute_list[attribute]['order'] == num:
                a_mrdata.append({'attribute': attribute, 'value': a_attribute_list[attribute]['value']})
                num += 1
                break
    fo = open(args.omr, 'w', encoding='utf-8')
    json.dump(a_mrdata, fo, ensure_ascii=False, indent=4)    
    fo.close()

    # (train3) training dataset
    a_traindata = []
    for i in range(1, len(a_input)):
        traindata = {}
        if len(a_input[i].rstrip('\n').split('",')) > 1:
            # text
            text_tmp = a_input[i].rstrip('\n').split('\",')[1]
            text = text_tmp.lstrip('\"').rstrip('\"').replace('.', '. ').replace('  ', ' ').rstrip(' ')
            if (text.endswith('?') is False) and (text.endswith('.') is False):
                text += '.'
                #print(text)

            #traindata = {'text': text, 'mr': []}
            traindata['text'] = text
            traindata['mr'] = []
            # mr
            mr_tmp = a_input[i].rstrip('\n').split('\",')[0]
            a_mr_tmp = mr_tmp.lstrip('\"').rstrip('\"').split('], ')
            a_mr = []
            for j in range(len(a_mr_tmp)):
                if len(a_mr_tmp[j].split('[')) > 1:
                    attribute = a_mr_tmp[j].split('[')[0]
                    value = a_mr_tmp[j].split('[')[1].rstrip(']')
                    a_mr.append({'attribute': attribute, 'value': value})

            for k in range(len(a_mrdata)):
                '''
                #no
                traindata['mr'].append({'attribute': '', 'value': 'no', 'replace': False})
                '''
                #empty
                traindata['mr'].append({'attribute': '', 'value': '', 'replace': False})
                for j in range(len(a_mr)):
                    if a_mrdata[k]['attribute'] == a_mr[j]['attribute']:
                        traindata['mr'][k]['attribute'] = a_mr[j]['attribute']
                        traindata['mr'][k]['value'] = a_mr[j]['value']
                        if (((a_mr[j]['value'] != 'no') and (a_mr[j]['value'] != 'yes')) and \
                            (text.startswith(a_mr[j]['value']+' ') or \
                             ((' '+a_mr[j]['value']+'.') in text) or \
                             ((' '+a_mr[j]['value']+' ') in text))):
                            traindata['mr'][k]['replace'] = True
                        break
            a_traindata.append(traindata)
    del a_input
    count_train = 0
    fo = open(args.otrain, 'w', encoding='utf-8')
    a_text = {}
    for i in range(len(a_traindata)):
        if ((a_traindata[i]['text'] in a_text) is False):
            a_text[a_traindata[i]['text']] = True
            for j in range(len(a_traindata[i]['mr'])):
                if j > 0:
                    fo.write('|')
                fo.write(a_traindata[i]['mr'][j]['value'])
            fo.write('\t')
            fo.write(a_traindata[i]['text'])
            fo.write('\n')
            count_train += 1
    fo.close()

    # (train4) augmented dataset
    fo = open(args.otrain_aug, 'w', encoding='utf-8')
    '''
    # copy original training data
    fi_tmp = open(args.otrain, 'r', encoding='utf-8')
    a_input = fi_tmp.readlines()
    fi_tmp.close()
    for i in range(len(a_input)):
        fo.write(a_input[i])
    del a_input
    '''
    count_train_aug = 0
    for i in range(len(a_traindata)):
        for j in range(len(a_traindata[i]['mr'])):
            attribute = a_traindata[i]['mr'][j]['attribute']
            if (attribute == 'familyFriendly') or \
               (attribute == 'priceRange') or \
               (attribute == 'customer rating') or \
               (a_traindata[i]['mr'][j]['replace'] is False):
                continue

            value_old = a_traindata[i]['mr'][j]['value']
            traindata = copy.deepcopy(a_traindata[i])
            for k in range(len(a_attribute_list[attribute]['value'])):
                value_new = a_attribute_list[attribute]['value'][k]
                if (value_old != value_new) and \
                   (value_new != 'yes') and \
                   (value_new != 'no') and \
                   (a_traindata[i]['text'].startswith(value_old+' ') or \
                    ((' '+value_old+' ') in a_traindata[i]['text']) or \
                    ((' '+value_old+'.') in a_traindata[i]['text'])):
                    if a_traindata[i]['text'].count(value_old) > 1:
                        continue
                    traindata['mr'][j]['value'] = value_new
                    text_new = a_traindata[i]['text'].replace(value_old, value_new)
                    if ((text_new in a_text) is False):
                        a_text[text_new] = True
                        for l in range(len(traindata['mr'])):
                            if l > 0:
                                fo.write('|')
                            fo.write(traindata['mr'][l]['value'])
                        fo.write('\t')
                        fo.write(text_new)
                        fo.write('\n')
                        count_train_aug += 1
    fo.close()                        

    # small version
    indices = numpy.arange(count_train_aug)
    numpy.random.seed(1234)
    numpy.random.shuffle(indices)
    count_output = count_train * 4
    fi = open(args.otrain_aug, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    fname = args.otrain_aug.rstrip('.tsv')+'_small.tsv'
    fo = open(fname, 'w', encoding='utf-8')
    for i in range(count_output):
        fo.write(a_input[indices[i]].rstrip('\n')+'\n')
    fo.close()

    del a_text
    del a_traindata
    del a_attribute_list

    ## validation dataset
    fi = open(args.ivalid, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    a_validdata = []
    for i in range(1, len(a_input)):
        validdata = {}
        if len(a_input[i].rstrip('\n').split('",')) > 1:
            # text
            text_tmp = a_input[i].rstrip('\n').split('\",')[1]
            text = text_tmp.lstrip('\"').rstrip('\"').replace('.', '. ').replace('  ', ' ').rstrip(' ')
            if (text.endswith('?') is False) and (text.endswith('.') is False):
                text += '.'
                #print(text)
            validdata['text'] = text
            validdata['mr'] = []
            # mr
            mr_tmp = a_input[i].rstrip('\n').split('\",')[0]
            a_mr_tmp = mr_tmp.lstrip('\"').rstrip('\"').split('], ')
            a_mr = []
            for j in range(len(a_mr_tmp)):
                if len(a_mr_tmp[j].split('[')) > 1:
                    attribute = a_mr_tmp[j].split('[')[0]
                    value = a_mr_tmp[j].split('[')[1].rstrip(']')
                    a_mr.append({'attribute': attribute, 'value': value})

            for k in range(len(a_mrdata)):
                '''
                #no
                validdata['mr'].append({'attribute': '', 'value': 'no'})
                '''
                #empty
                validdata['mr'].append({'attribute': '', 'value': ''})
                for j in range(len(a_mr)):
                    if a_mrdata[k]['attribute'] == a_mr[j]['attribute']:
                        validdata['mr'][k]['attribute'] = a_mr[j]['attribute']
                        validdata['mr'][k]['value'] = a_mr[j]['value']
                        break
            a_validdata.append(validdata)
    del a_input
    fo = open(args.ovalid, 'w', encoding='utf-8')
    a_text = {}
    for i in range(len(a_validdata)):
        if ((a_validdata[i]['text'] in a_text) is False):
            a_text[a_validdata[i]['text']] = True
            for j in range(len(a_validdata[i]['mr'])):
                if j > 0:
                    fo.write('|')
                fo.write(a_validdata[i]['mr'][j]['value'])
            fo.write('\t')
            fo.write(a_validdata[i]['text'])
            fo.write('\n')
    fo.close()
    del a_text
    del a_validdata

    ## test dataset
    fi = open(args.itest, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    a_testdata = []
    for i in range(1, len(a_input)):
        testdata = {}
        if len(a_input[i].rstrip('\n').split('",')) > 1:
            # text
            text_tmp = a_input[i].rstrip('\n').split('\",')[1]
            text = text_tmp.lstrip('\"').rstrip('\"').replace('.', '. ').replace('  ', ' ').rstrip(' ')
            if (text.endswith('?') is False) and (text.endswith('.') is False):
                text += '.'
                #print(text)
            testdata['text'] = text
            testdata['mr'] = []
            # mr
            mr_tmp = a_input[i].rstrip('\n').split('\",')[0]
            a_mr_tmp = mr_tmp.lstrip('\"').rstrip('\"').split('], ')
            a_mr = []
            for j in range(len(a_mr_tmp)):
                if len(a_mr_tmp[j].split('[')) > 1:
                    attribute = a_mr_tmp[j].split('[')[0]
                    value = a_mr_tmp[j].split('[')[1].rstrip(']')
                    a_mr.append({'attribute': attribute, 'value': value})

            for k in range(len(a_mrdata)):
                '''
                #no
                testdata['mr'].append({'attribute': '', 'value': 'no'})
                '''
                #empty
                testdata['mr'].append({'attribute': '', 'value': ''})
                for j in range(len(a_mr)):
                    if a_mrdata[k]['attribute'] == a_mr[j]['attribute']:
                        testdata['mr'][k]['attribute'] = a_mr[j]['attribute']
                        testdata['mr'][k]['value'] = a_mr[j]['value']
                        break
            a_testdata.append(testdata)
    del a_input
    del a_mrdata
    fo = open(args.otest, 'w', encoding='utf-8')
    a_text = {}
    for i in range(len(a_testdata)):
        if ((a_testdata[i]['text'] in a_text) is False):
            a_text[a_testdata[i]['text']] = True
            for j in range(len(a_testdata[i]['mr'])):
                if j > 0:
                    fo.write('|')
                fo.write(a_testdata[i]['mr'][j]['value'])
            fo.write('\t')
            fo.write(a_testdata[i]['text'])
            fo.write('\n')
    fo.close()
    del a_text
    del a_testdata

    print('** done **')
