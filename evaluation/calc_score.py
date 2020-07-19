#! python

import argparse
import sys
import nltk
'''
# METEOR
# need nltk.download('wordnet') before using meteor_score()
nltk.translate.meteor_score.meteor_score([reference1, ...], hypothesis)
nltk.translate.meteor_score.single_meteor_score(reference, hypothesis)

# BLEU
nltk.translate.bleu_score.sentence_bleu([reference1, ...], hypothesis)
'''

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

    total_score_bleu = 0.0
    total_score_meteor = 0.0

    if args.data.lower() == 'e2e':
        idx = 0
    else:
        idx = 1

    if args.mode.lower() == 'nlg':
        fo.write('mr\t')
        fo.write('text(correct)\t')
        fo.write('text(predict)\t')
        fo.write('result(precision)\t')
        fo.write('result(BLEU)\t')
        fo.write('result(METEOR)\t')
        fo.write('(tmp)result(slot drop)\t')
        fo.write('(tmp)result(slot hallucination)\n')

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

            # bleu
            a_text_correct = []
            s = max(1, i-2)
            e = min(len(a_input)-1, i+3)
            for j in range(s, e):
                if mr == a_input[j].rstrip('\n').split('\t')[0].lower():
                    a_text_correct.append(a_input[j].rstrip('\n').split('\t')[1].lower())
            #score_bleu = nltk.translate.bleu_score.sentence_bleu([text_correct], text_predict)
            score_bleu = nltk.translate.bleu_score.sentence_bleu(a_text_correct, text_predict)
            total_score_bleu += score_bleu

            # meteor
            #score_meteor = nltk.translate.meteor_score.single_meteor_score(text_correct, text_predict)
            score_meteor = nltk.translate.meteor_score.meteor_score(a_text_correct, text_predict)
            total_score_meteor += score_meteor

            # slot drop (tmp)
            slot_drop_flag = True
            for j in range(idx, len(a_mr)):
                if (a_mr[j] in text_predict) is False:
                    slot_drop_flag = False
                    break

            # slot hallucination (tmp)
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
            fo.write(str(precision_flag)+'\t')
            fo.write(str(100.0*score_bleu)+'\t')
            fo.write(str(100.0*score_meteor)+'\t')
            fo.write(str(slot_drop_flag)+'\t')
            fo.write(str(slot_hallucination_flag)+'\n')

        print('precision           : '+str(100.0*precision_ok/count))
        print('BLEU                : '+str(100.0*total_score_bleu/count))
        print('METEOR              : '+str(100.0*total_score_meteor/count))
        print('slot (drop)         : '+str(100.0*slot_drop_ok/count))
        print('slot (hallucination): '+str(100.0*slot_hallucination_ok/count))
        fo.write('\t\t\t')
        fo.write(str(100.0*precision_ok/count)+'\t')
        fo.write(str(100.0*total_score_bleu/count)+'\t')
        fo.write(str(100.0*total_score_meteor/count)+'\t')
        fo.write(str(100.0*slot_drop_ok/count)+'\t')
        fo.write(str(100.0*slot_hallucination_ok/count)+'\n')

    if args.mode.lower() == 'nlu':
        fo.write('text\t')
        fo.write('mr(correct)\t')
        fo.write('mr(predict)\t')
        fo.write('result\t\t\t\t\t')
        fo.write('name\t\t\t\t\t')
        fo.write('eatType\t\t\t\t\t')
        fo.write('food\t\t\t\t\t')
        fo.write('priceRange\t\t\t\t\t')
        fo.write('customer rating\t\t\t\t\t')
        fo.write('area\t\t\t\t\t')
        fo.write('familyFriendly\t\t\t\t\t')
        fo.write('near\t\t\t\t\n')

        total_score = [[0] * 5] * 8
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

            a_score = [[0]*5]*8
            len_mr = len(a_mr_correct)
            if len_mr > len(a_mr_predict):
                len_mr = len(a_mr_predict)
            for j in range(len_mr):
                if a_mr_correct[j] == 'no':
                    if a_mr_predict[j] == 'no':
                        score = 0
                    else:
                        score = 1
                else:
                    if a_mr_predict[j] == 'no':
                        score = 2
                    elif a_mr_predict[j] == a_mr_correct[j]:
                        score = 3
                    else:
                        score = 4
                a_score[j][score] = 1
                total_score[j][score] += 1
            for j in range(len_mr, 8):
                if a_mr_correct[j] == 'no':
                    score = 1
                else:
                    score = 4
                a_score[j][score] = 1
                total_score[j][score] += 1

            if precision_flag is True:
                precision_ok += 1
            else:
                precision_ng += 1
            count += 1
            fo.write(a_input[i].rstrip('\n').split('\t')[0]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[1]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[2]+'\t')
            fo.write(str(precision_flag))
            for j in range(8):
                for k in range(5):
                    fo.write('\t'+str(a_score[j][k]))
            fo.write('\n')

        print('precision           : '+str(100.0*precision_ok/count))
        fo.write('\t\t\t')
        fo.write(str(100.0*precision_ok/count))
        for j in range(8):
            for k in range(5):
                fo.write('\t'+str(100.0*total_score[j][k]/count))
        fo.write('\n')
    fo.close()
