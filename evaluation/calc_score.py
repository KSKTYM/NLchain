#! python

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', help='mode (NLG(default) or NLU)', default='NLG')
    parser.add_argument('-data', help='name of dataset')
    parser.add_argument('-i', help='input file name')
    parser.add_argument('-o', help='output file name')
    args = parser.parse_args()

    fi = open(args.i, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    fo = open(args.o, 'w', encoding='utf-8')

    count = 0
    precision_ok = 0
    precision_ng = 0
    slot_drop_ok = 0
    slot_drop_ng = 0
    slot_hallucination_ok = 0
    slot_hallucination_ng = 0
    slot_replace_ok = 0
    slot_replace_ng = 0

    if args.data.lower() == 'e2e':
        idx = 0
    else:
        idx = 1

    if args.mode.lower() == 'nlg':
        fo.write('mr\t')
        fo.write('text(correct)\t')
        fo.write('text(predict)\t')
        fo.write('result\t')
        fo.write('result(slot drop)\t')
        fo.write('result(slot hallucination)\n')

        for i in range(1, len(a_input)-1):
            mr = a_input[i].rstrip('\n').split('\t')[0].lower()
            text_correct = a_input[i].rstrip('\n').split('\t')[1].lower()
            text_predict = a_input[i].rstrip('\n').split('\t')[2].lower()

            a_mr = mr.split('|')
            a_text_correct = text_correct.split(' ')
            a_text_predict = text_predict.split(' ')

            # precision
            precision_flag = True
            if text_correct != text_predict:
                precision_flag = False

            # slot drop
            slot_drop_flag = True
            for j in range(idx, len(a_mr)):
                if (a_mr[j] in text_predict) is False:
                    slot_drop_flag = False
                    break

            # slot hallucination (not precise)
            a_word = []
            for j in range(len(a_mr)):
                for k in range(len(a_mr[j].split(' '))):
                    word = a_mr[j].split(' ')[k]
                    if (word != '') and ((word in a_word) is False):
                        a_word.append(word)

            for j in range(len(a_text_correct)):
                word = a_text_correct[j]
                if (word != '') and ((word in a_word) is False):
                    a_word.append(word)
            for j in range(len(a_word)):
                a_word[j] = a_word[j].replace('.', '').replace(',', '')
            tmp_predict = text_predict.replace('.', '').replace(',', '')
            a_predict = tmp_predict.split(' ')

            slot_hallucination_flag = True
            for j in range(len(a_predict)):
                flag = True
                if a_predict[j] != '':
                    flag = False
                    for k in range(len(a_word)):
                        if a_word[k] == a_predict[j]:
                            flag = True
                            break
                if flag is False:
                    slot_hallucination_flag = False
                    break
            
            if precision_flag is True:
                precision_ok += 1
            else:
                precision_ng += 1
            if slot_drop_flag is True:
                slot_drop_ok += 1
            else:
                slot_drop_ng += 1
            if slot_hallucination_flag is True:
                slot_hallucination_ok += 1
            else:
                slot_hallucination_ng += 1
            count += 1
            fo.write(a_input[i].rstrip('\n').split('\t')[0]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[1]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[2]+'\t')
            fo.write(str(precision_flag)+'\t'+str(slot_drop_flag)+'\t'+str(slot_hallucination_flag)+'\n')

        print('precision           : '+str(100.0*precision_ok/count))
        print('slot (drop)         : '+str(100.0*slot_drop_ok/count))
        print('slot (hallucination): '+str(100.0*slot_hallucination_ok/count))
        fo.write(str(100.0*precision_ok/count)+'\t')
        fo.write(str(100.0*slot_drop_ok/count)+'\t')
        fo.write(str(100.0*slot_hallucination_ok/count)+'\n')

    if args.mode.lower() == 'nlu':
        fo.write('text\t')
        fo.write('mr(correct)\t')
        fo.write('mr(predict)\t')
        fo.write('result\t')
        fo.write('result(slot drop)\t')
        fo.write('result(slot hallucination)\t')
        fo.write('result(slot replace)\n')

        for i in range(1, len(a_input)-1):
            text = a_input[i].rstrip('\n').split('\t')[0].lower()
            mr_correct = a_input[i].rstrip('\n').split('\t')[1].lower()
            mr_predict = a_input[i].rstrip('\n').split('\t')[2].lower()

            a_text = text.split(' ')
            a_mr_correct = mr_correct.split('|')
            a_mr_predict = mr_predict.split('|')

            # precision
            precision_flag = True
            if len(a_mr_correct) != len(a_mr_predict):
                precision_flag = False
            else:
                for j in range(len(a_mr_correct)):
                    if a_mr_correct[j] != a_mr_predict[j]:
                        precision_flag = False
                        break

            # slot drop
            slot_drop_flag = True
            if len(a_mr_correct) > len(a_mr_predict):
                slot_drop_flag = False
            else:
                for j in range(idx, len(a_mr_correct)):
                    if (a_mr_correct[j] != '') and (a_mr_predict[j] == ''):
                        slot_drop_flag = False
                        break

            # slot hallucination
            slot_hallucination_flag = True
            if len(a_mr_correct) < len(a_mr_predict):
                slot_hallucination_flag = False
            else:
                for j in range(idx, len(a_mr_predict)):
                    if (a_mr_correct[j] == '') and (a_mr_predict[j] != ''):
                        slot_hallucination_flag = False
                        break

            # slot replace
            slot_replace_flag = True
            len_mr = min(len(a_mr_correct), len(a_mr_predict))
            for j in range(idx, len_mr):
                if (a_mr_correct[j] != '') and (a_mr_predict[j] != '') and (a_mr_correct[j] != a_mr_predict[j]):
                    slot_replace_flag = False
                    break

            if precision_flag is True:
                precision_ok += 1
            else:
                precision_ng += 1
            if slot_drop_flag is True:
                slot_drop_ok += 1
            else:
                slot_drop_ng += 1
            if slot_hallucination_flag is True:
                slot_hallucination_ok += 1
            else:
                slot_hallucination_ng += 1
            if slot_replace_flag is True:
                slot_replace_ok += 1
            else:
                slot_replace_ng += 1
            count += 1
            fo.write(a_input[i].rstrip('\n').split('\t')[0]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[1]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[2]+'\t')
            fo.write(str(precision_flag)+'\t'+str(slot_drop_flag)+'\t'+str(slot_hallucination_flag)+'\t'+str(slot_replace_flag)+'\n')

        print('precision           : '+str(100.0*precision_ok/count))
        print('slot (drop)         : '+str(100.0*slot_drop_ok/count))
        print('slot (hallucination): '+str(100.0*slot_hallucination_ok/count))
        print('slot (replace)      : '+str(100.0*slot_replace_ok/count))
        fo.write(str(100.0*precision_ok/count)+'\t')
        fo.write(str(100.0*slot_drop_ok/count)+'\t')
        fo.write(str(100.0*slot_hallucination_ok/count)+'\t')
        fo.write(str(100.0*slot_replace_ok/count)+'\n')

    fo.close()
