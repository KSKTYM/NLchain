#! python

import argparse
import sys
import nltk
sys.path.append('..')
from common.tokenizer import Tokenizer

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
    tokenizer = Tokenizer()

    fi = open(args.i, 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    fo = open(args.o, 'w', encoding='utf-8')
    print('*calc_score*')
    print(' mode  :'+str(args.mode))
    print(' data  :'+str(args.data))
    print(' input :'+str(args.i))
    print(' output:'+str(args.o))
    count = 0
    frame_ok = 0
    frame_ng = 0
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

        for i in range(1, len(a_input)):
            mr = a_input[i].rstrip('\n').split('\t')[0].lower()
            text_correct = a_input[i].rstrip('\n').split('\t')[1].lower()
            text_predict = a_input[i].rstrip('\n').split('\t')[2].lower()
            a_mr = mr.split('|')

            # precision
            frame_flag = True
            if text_correct != text_predict:
                frame_flag = False

            # bleu
            a_text_correct_bleu = []
            a_text_correct_meteor = []
            '''
            s = max(1, i-2)
            e = min(len(a_input), i+3)
            for j in range(s, e):
                if mr == a_input[j].rstrip('\n').split('\t')[0].lower():
                    a_text_correct_meteor.append(a_input[j].rstrip('\n').split('\t')[1].lower())
            for j in range(s, e):
                if mr == a_input[j].rstrip('\n').split('\t')[0].lower():
                    text_tmp = a_input[j].rstrip('\n').split('\t')[1].lower()
                    a_text_correct_bleu.append(tokenizer.text(text_tmp))
            '''
            s = 1
            e = len(a_input)
            for j in range(s, e):
                if mr == a_input[j].rstrip('\n').split('\t')[0].lower():
                    a_text_correct_meteor.append(a_input[j].rstrip('\n').split('\t')[1].lower())
                    text_tmp = a_input[j].rstrip('\n').split('\t')[1].lower()
                    a_text_correct_bleu.append(tokenizer.text(text_tmp))

            a_text_predict = tokenizer.text(text_predict)
            score_bleu = nltk.translate.bleu_score.sentence_bleu(a_text_correct_bleu, a_text_predict)
            total_score_bleu += score_bleu

            # meteor
            score_meteor = nltk.translate.meteor_score.meteor_score(a_text_correct_meteor, text_predict)
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

            a_text_correct = text_correct.split(' ')
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
            
            if frame_flag is True:
                frame_ok += 1
            else:
                frame_ng += 1
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
            fo.write(str(frame_flag)+'\t')
            fo.write(str(100.0*score_bleu)+'\t')
            fo.write(str(100.0*score_meteor)+'\t')
            fo.write(str(slot_drop_flag)+'\t')
            fo.write(str(slot_hallucination_flag)+'\n')

        print('accuracy(frame)     : '+str(100.0*frame_ok/count))
        print('BLEU                : '+str(100.0*total_score_bleu/count))
        print('METEOR              : '+str(100.0*total_score_meteor/count))
        print('slot (drop)         : '+str(100.0*slot_drop_ok/count))
        print('slot (hallucination): '+str(100.0*slot_hallucination_ok/count))
        fo.write('accuracy(frame)\t'+str(100.0*frame_ok/count)+'\n')
        fo.write('BLEU\t'+str(100.0*total_score_bleu/count)+'\n')
        fo.write('METEOR\t'+str(100.0*total_score_meteor/count)+'\n')
        fo.write('slot(drop)\t'+str(100.0*slot_drop_ok/count)+'\n')
        fo.write('slot(hallucination)\t'+str(100.0*slot_hallucination_ok/count)+'\n')

    if args.mode.lower() == 'nlu':
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        fo.write('text\tmr(correct)\tmr(predict)\tresult\n')
        for i in range(1, len(a_input)):
            text = a_input[i].rstrip('\n').split('\t')[0].lower()
            mr_correct = a_input[i].rstrip('\n').split('\t')[1].lower()
            mr_predict = a_input[i].rstrip('\n').split('\t')[2].lower()

            a_mr_correct = mr_correct.split('|')
            a_mr_predict = mr_predict.split('|')

            length_min = min(len(a_mr_correct), len(a_mr_predict))
            length_max = max(len(a_mr_correct), len(a_mr_predict))

            if len(a_mr_correct) == len(a_mr_predict):
                mode = 0
            elif len(a_mr_correct) < len(a_mr_predict):
                mode = 1
            else:
                mode = 2

            frame_flag = True
            if len(a_mr_correct) != len(a_mr_predict):
                frame_flag = False
            for j in range(length_min):
                if a_mr_correct[j] != a_mr_predict[j]:
                    frame_flag = False
                if a_mr_correct[j] != '':
                    if a_mr_correct[j] == a_mr_predict[j]:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if a_mr_correct[j] == a_mr_predict[j]:
                        TN += 1
                    else:
                        FP += 1
            for j in range(length_min, length_max):
                if mode == 1:
                    FP += 1
                elif mode == 2:
                    FN += 1

            if frame_flag is True:
                frame_ok += 1
            else:
                frame_ng += 1
            count += 1
            fo.write(a_input[i].rstrip('\n').split('\t')[0]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[1]+'\t')
            fo.write(a_input[i].rstrip('\n').split('\t')[2]+'\t')
            fo.write(str(frame_flag)+'\n')

        if count > 0:
            accuracy_frame = frame_ok / count
        else:
            accuracy_frame = 0
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
        print('accuracy(frame): '+str(accuracy_frame))
        print('f1score        : '+str(f1score))
        print('presicion      : '+str(precision))
        print('recall         : '+str(recall))
        print('accuracy       : '+str(accuracy))
        fo.write('accuracy(frame)\t'+str(accuracy_frame)+'\n')
        fo.write('f1score\t'+str(f1score)+'\n')
        fo.write('precision\t'+str(precision)+'\n')
        fo.write('recall\t'+str(recall)+'\n')
        fo.write('accuracy\t'+str(accuracy)+'\n')
    fo.close()
    print('*done*')
