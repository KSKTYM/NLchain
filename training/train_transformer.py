#! python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib

import random
import math
import time

import cloudpickle
#import dill
import json

sys.path.append('..')
from common.tokenizer import Tokenizer
from model.transformer import Encoder, Decoder, Seq2Seq
from inference.nlc_transformer import NLC
#from torchtext.data.metrics import bleu_score

## sub functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train_nlg(model_nlg, iterator, optimizer, criterion, clip, verbose_flag):
    model_nlg.train()
    train_loss = 0
    
    for i, batch in enumerate(iterator):
        mr = batch.mr
        sen = batch.sen
        if verbose_flag is True:
            print('***** i : '+str(i)+' *****')
            print('(1) sen: '+str(sen.size()))
            print(sen)
            print('(2) mr: '+str(mr.size()))
            print(mr)
            print('(3) sen[:,:-1]')
            print(sen[:,:-1])
            print('(4) mr[:,:-1]')
            print(mr[:,:-1])

        optimizer.zero_grad()
        output, _ = model_nlg(mr, sen[:,:-1])
        #output = [batch size, sen len - 1, output dim]
        if verbose_flag is True:
            print('(5) output_nlg: '+str(output.size()))
            print(output)

        #sen = [batch size, sen len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        if verbose_flag is True:
            print('(7) output_nlg: '+str(output.size()))
            print(output)

        sen = sen[:,1:].contiguous().view(-1)
        #output = [batch size * sen len - 1, output dim]
        #sen = [batch size * sen len - 1]
        if verbose_flag is True:
            print('(9) sen:'+str(sen.size()))
            print(sen)
            
        loss = criterion(output, sen)
        if verbose_flag is True:
            print('(11) loss_nlg: '+str(loss.size()))
            print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nlg.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(iterator)

def train_nlu(model_nlu, iterator, optimizer, criterion, clip, verbose_flag):
    model_nlu.train()
    train_loss = 0

    for i, batch in enumerate(iterator):
        sen = batch.sen
        mr = batch.mr
        if verbose_flag is True:
            print('***** i : '+str(i)+' *****')
            print('(1) sen: '+str(sen.size()))
            print(sen)
            print('(2) mr: '+str(mr.size()))
            print(mr)
            print('(3) sen[:,:-1]')
            print(sen[:,:-1])
            print('(4) mr[:,:-1]')
            print(mr[:,:-1])

        optimizer.zero_grad()
        output, _ = model_nlu(sen, mr[:,:-1])
        #output = [batch size, mr len - 1, output dim]
        if verbose_flag is True:
            print('(6) output_nlu: '+str(output.size()))
            print(output)

        #mr = [batch size, mr len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        if verbose_flag is True:
            print('(8) output_nlu: '+str(output.size()))
            print(output)

        mr = mr[:,1:].contiguous().view(-1)
        #output = [batch size * mr len - 1, output dim]
        #mr = [batch size * mr len - 1]
        if verbose_flag is True:
            print('(10) mr:'+str(mr.size()))
            print(mr)
            
        loss = criterion(output, mr)
        if verbose_flag is True:
            print('(12) loss_nlu: '+str(loss.size()))
            print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nlu.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(iterator)

def evaluate_nlg(model_nlg, iterator, criterion):
    model_nlg.eval()
    evaluate_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            mr = batch.mr
            sen = batch.sen
            output, _ = model_nlg(mr, sen[:,:-1])
            #output = [batch size, sen len - 1, output dim]
            #sen = [batch size, sen len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            sen = sen[:,1:].contiguous().view(-1)
            #output = [batch size * sen len - 1, output dim]
            #sen = [batch size * sen len - 1]
            loss = criterion(output, sen)
            evaluate_loss += loss.item()
        
    return evaluate_loss / len(iterator)

def evaluate_nlu(model_nlu, iterator, criterion):
    model_nlu.eval()
    evaluate_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            sen = batch.sen
            mr = batch.mr
            output, _ = model_nlu(sen, mr[:,:-1])
            #output = [batch size, mr len - 1, output dim]
            #mr = [batch size, mr len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            mr = mr[:,1:].contiguous().view(-1)
            #output = [batch size * mr len - 1, output dim]
            #mr = [batch size * mr len - 1]
            loss = criterion(output, mr)
            evaluate_loss += loss.item()

    return evaluate_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def calculate_valid(param_path, valid_data, SEN, MR, model_nlu, device, chain_flag):
    NLU = NLC(param_path, 'NLU', chain_flag)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    count = 0
    count_ok = 0
    for i in range(len(valid_data.examples)):
        sen = vars(valid_data.examples[i])['sen']
        mr = vars(valid_data.examples[i])['mr']
        translation, attension = NLU.translate_sentence(sen, SEN, MR, model_nlu, device)
        flag = True
        length_min = min(len(mr), len(translation)-1)
        length_max = max(len(mr), len(translation)-1)

        if len(mr) == len(translation)-1:
            mode = 0
        elif len(mr) < len(translation)-1:
            mode = 1
        else:
            mode = 2

        if len(mr) != len(translation)-1:
            flag = False
        for j in range(length_min):
            if mr[j] != translation[j]:
                flag = False
            if mr[j] != '':
                if mr[j] == translation[j]:
                    TP += 1
                else:
                    FN += 1
            else:
                if mr[j] == translation[j]:
                    TN += 1
                else:
                    FP += 1
        for j in range(length_min, length_max):
            if mode == 1:
                FP += 1
            elif mode == 2:
                FN += 1

        if flag is True:
            count_ok += 1
        count += 1

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
    if count > 0:
        correct = count_ok / count
    else:
        correct = 0

    print('precision: '+str(precision))
    print('recall   : '+str(recall))
    print('f1score  : '+str(f1score))
    print('accuracy : '+str(accuracy))
    print('correct  : '+str(correct)+'('+str(count_ok)+'/'+str(count)+')')

    return accuracy, precision, recall, f1score, correct

def train_chain(model_nlg, model_nlu, train_aug_iterator, optimizer_nlg, optimizer_nlu, criterion_nlg, criterion_nlu, CLIP, beta, verbose_flag):
    train_loss_nlg = 0
    train_loss_nlu = 0 
    for i, batch in enumerate(train_aug_iterator):
        sen = batch.sen
        mr = batch.mr#128-gyou,10-retsu(include <sos:2> and <eos:3>)

        # (1) generate temporary SEN' from MR for NLU
        model_nlg.eval()
        output_nlg, _ = model_nlg(mr, sen[:,:-1])
        output_nlg_dim = output_nlg.shape[-1]
        output_nlg = output_nlg.contiguous().view(-1, output_nlg_dim)
        a_output_nlg_token = output_nlg.argmax(1).tolist()
        len_sentence = int(len(a_output_nlg_token)/BATCH_SIZE)
        tmp_a = []
        num_batch = len(batch.mr)
        for j in range(num_batch):
            tmp_a.append(SEN.vocab.stoi['<sos>'])
            tmp_a.extend(a_output_nlg_token[j*len_sentence:(j+1)*len_sentence])
        tmp_b = torch.Tensor(tmp_a).long()
        generated_sentence = tmp_b.reshape(num_batch, len_sentence+1).to(device)
        del tmp_a, tmp_b

        # (2) generate temporary MR' from SEN for NLG
        model_nlu.eval()
        output_nlu, _ = model_nlu(sen, mr[:,:-1])
        output_nlu_dim = output_nlu.shape[-1]
        output_nlu = output_nlu.contiguous().view(-1, output_nlu_dim)
        a_output_nlu_token = output_nlu.argmax(1).tolist()
        len_mr = int(len(a_output_nlu_token)/BATCH_SIZE)
        tmp_a = []
        num_batch = len(batch.sen)
        for j in range(num_batch):
            tmp_a.append(MR.vocab.stoi['<sos>'])
            tmp_a.extend(a_output_nlu_token[j*len_mr:(j+1)*len_mr])
        tmp_b = torch.Tensor(tmp_a).long()
        generated_mr = tmp_b.reshape(num_batch, len_mr+1).to(device)
        del tmp_a, tmp_b
        '''
        print('*'+str(i)+'*')
        print('mr')
        print(mr[0])
        print('generated_mr')
        print(generated_mr[0])
        print('sen')
        print(sen[0])
        print('generated_sen')
        print(generated_sentence[0])
        '''
        # (3) NLG (MR' -> SEN'')
        model_nlg.train()
        optimizer_nlg.zero_grad()
        sen2 = batch.sen
        output_nlg2, _ = model_nlg(generated_mr, sen2[:, :-1])
        output_nlg2_dim = output_nlg2.shape[-1]
        output_nlg2 = output_nlg2.contiguous().view(-1, output_nlg2_dim)
        sen = batch.sen
        sen = sen[:,1:].contiguous().view(-1)
        loss_nlg = beta * criterion_nlg(output_nlg2, sen)
        loss_nlg.backward()
        torch.nn.utils.clip_grad_norm_(model_nlg.parameters(), CLIP)
        optimizer_nlg.step()
        train_loss_nlg += loss_nlg.item()

        # (4) NLU (SEN' -> MR'')
        model_nlu.train()
        optimizer_nlu.zero_grad()
        mr2 = batch.mr
        output_nlu2, _ = model_nlu(generated_sentence, mr2[:, :-1])
        output_nlu2_dim = output_nlu2.shape[-1]
        output_nlu2 = output_nlu2.contiguous().view(-1, output_nlu2_dim)
        mr = batch.mr
        mr = mr[:,1:].contiguous().view(-1)
        loss_nlu = beta * criterion_nlu(output_nlu2, mr)
        loss_nlu.backward()
        torch.nn.utils.clip_grad_norm_(model_nlu.parameters(), CLIP)
        optimizer_nlu.step()
        train_loss_nlu += loss_nlu.item()

    train_loss_nlg = train_loss_nlg / len(train_aug_iterator)
    train_loss_nlu = train_loss_nlu / len(train_aug_iterator)
    return train_loss_nlg, train_loss_nlu

def save_best_nlg(path, best_valid_loss_nlg, best_epoch_nlg, valid_loss_nlg, epoch, model_nlg, MR, SEN, chain_flag):
    if chain_flag is True:
        mode = 'chain_nlg'
    else:
        mode = 'nlg'
    if valid_loss_nlg < best_valid_loss_nlg:
        best_valid_loss_nlg = valid_loss_nlg
        best_epoch_nlg = epoch
        #torch.save(model_nlg.state_dict(), path+'/best_model_'+mode+'.pt')
        f = open(path+'/best_model_'+mode+'.pkl', 'wb')
        cloudpickle.dump(model_nlg, f)
        f.close()
        f = open(path+'/best_epoch_'+mode+'.txt', 'w')
        f.write(str(epoch))
        f.close()
        f = open(path+'/best_MR_'+mode+'.pkl', 'wb')
        cloudpickle.dump(MR, f)
        f.close()
        f = open(path+'/best_SEN_'+mode+'.pkl', 'wb')
        cloudpickle.dump(SEN, f)
        f.close()
    f = open(path+'/model_'+mode+'_'+str(epoch)+'.pkl', 'wb')
    cloudpickle.dump(model_nlg, f)
    f.close()

    return best_valid_loss_nlg, best_epoch_nlg

def save_best_nlu(path, best_valid_loss_nlu, best_epoch_nlu, valid_loss_nlu, epoch, model_nlu, MR, SEN, chain_flag):
    if chain_flag is True:
        mode = 'chain_nlu'
    else:
        mode = 'nlu'
    if valid_loss_nlu < best_valid_loss_nlu:
        best_valid_loss_nlu = valid_loss_nlu
        best_epoch_nlu = epoch
        #torch.save(model_nlu.state_dict(), path+'/best_model_'+mode+'.pt')
        f = open(path+'/best_model_'+mode+'.pkl', 'wb')
        cloudpickle.dump(model_nlu, f)
        f.close()
        f = open(path+'/best_epoch_'+mode+'.txt', 'w')
        f.write(str(epoch))
        f.close()
        f = open(path+'/best_MR_'+mode+'.pkl', 'wb')
        cloudpickle.dump(MR, f)
        f.close()
        f = open(path+'/best_SEN_'+mode+'.pkl', 'wb')
        cloudpickle.dump(SEN, f)
        f.close()
    f = open(path+'/model_'+mode+'_'+str(epoch)+'.pkl', 'wb')
    cloudpickle.dump(model_nlu, f)
    f.close()

    return best_valid_loss_nlu, best_epoch_nlu

## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='parameter directory\'s name', default='../parameter/')
    parser.add_argument('-path', help='training data directory\'s name', default='../corpus/')
    parser.add_argument('-file', help='training data file name', default='nlc_transformer')
    parser.add_argument('-mode', help='training mode{nlu|nlg|chain}', default='chain')
    parser.add_argument('-seed', help='seed number', type=int, default=1234)
    parser.add_argument('-epoch', help='epoch number', type=int, default=10)
    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-alpha', help='alpha parameter', type=float, default=0.5)
    parser.add_argument('-beta', help='beta parameter', type=float, default=1.0)
    parser.add_argument('-graph', help='show attention graph', action='store_true')
    parser.add_argument('-v', help='verbose(print debug)', action='store_true')
    parser.add_argument('-eval', help='evalation with test data', action='store_true')
    args = parser.parse_args()

    print('** NLchain **')
    print(' training mode           : '+str(args.mode))
    print(' training data directory : '+str(args.path))
    print(' training data file      : '+str(args.file))

    print(' parameter directory     : '+str(args.p))
    print(' random seed number      : '+str(args.seed))
    print(' epoch number            : '+str(args.epoch))
    print(' batch size              : '+str(args.batch))
    print(' alpha parameter         : '+str(args.alpha))
    print(' beta parameter          : '+str(args.beta))

    if args.graph is True:
        print(' show graph              : ON')
    if args.v is True:
        print(' verbose (print debug)   : ON')

    # output directory
    if not os.path.exists(args.p):
        os.mkdir(args.p)

    if args.mode.lower() == 'nlu':
        chain_flag = False
        nlu_flag = True
        nlg_flag = False
    elif args.mode.lower() == 'nlg':
        chain_flag = False
        nlu_flag = False
        nlg_flag = True
    #elif args.mode.lower() == 'chain':
    elif args.mode.lower().startswith('chain'):
        chain_flag = True
        nlu_flag = True
        nlg_flag = True

    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()

    # NLU: MR->SEN, NLG: SEN->MR
    MR = Field(tokenize = tokenizer.mr,
               init_token = '<sos>', 
               eos_token = '<eos>', 
               lower = True, 
               batch_first = True)
    SEN = Field(tokenize = tokenizer.text,
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    train_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
        path=args.path,
        train=args.file+'_train.tsv',
        validation=args.file+'_valid.tsv',
        test=args.file+'_test.tsv',
        format='tsv',
        fields=[('mr', MR), ('sen', SEN)])

    if chain_flag is True:
        train_aug_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
            path=args.path,
            train=args.file+'_train_aug_small.tsv',
            validation=args.file+'_valid.tsv',
            test=args.file+'_test.tsv',
            format='tsv',
            fields=[('mr', MR), ('sen', SEN)])

    MR.build_vocab(train_data, min_freq = 1)
    SEN.build_vocab(train_data, min_freq = 1)

    f = open('MR_'+args.mode+'.json', 'w', encoding='utf-8')
    json.dump(MR.vocab.itos, f, ensure_ascii=False, sort_keys=True)
    f.close()
    f = open('SEN_'+args.mode+'.json', 'w', encoding='utf-8')
    json.dump(SEN.vocab.itos, f, ensure_ascii=False, sort_keys=True)
    f.close()

    f = open(args.p+'/MR_tmp.pkl', 'wb')
    cloudpickle.dump(MR, f)
    f.close()
    f = open(args.p+'/SEN_tmp.pkl', 'wb')
    cloudpickle.dump(SEN, f)
    f.close()

    BATCH_SIZE = args.batch
    MAX_LENGTH = 100
    INPUT_DIM = len(MR.vocab)
    OUTPUT_DIM = len(SEN.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    LEARNING_RATE = 0.0005
    N_EPOCHS = args.epoch
    CLIP = 1

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        sort=False,
        device = device)
    if chain_flag is True:
        train_aug_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_aug_data, valid_data, test_data), 
            batch_size = BATCH_SIZE,
            sort=False,
            device = device)

    if nlg_flag is True:
        enc_nlg = Encoder(INPUT_DIM,
                          HID_DIM,
                          ENC_LAYERS,
                          ENC_HEADS,
                          ENC_PF_DIM,
                          ENC_DROPOUT,
                          device,
                          MAX_LENGTH)

        dec_nlg = Decoder(OUTPUT_DIM,
                          HID_DIM,
                          DEC_LAYERS,
                          DEC_HEADS,
                          DEC_PF_DIM,
                          DEC_DROPOUT,
                          device,
                          MAX_LENGTH)

    if nlu_flag is True:
        dec_nlu = Decoder(INPUT_DIM,
                          HID_DIM,
                          ENC_LAYERS,
                          ENC_HEADS,
                          ENC_PF_DIM,
                          ENC_DROPOUT,
                          device,
                          MAX_LENGTH)

        enc_nlu = Encoder(OUTPUT_DIM,
                          HID_DIM,
                          DEC_LAYERS,
                          DEC_HEADS,
                          DEC_PF_DIM,
                          DEC_DROPOUT,
                          device,
                          MAX_LENGTH)

    MR_PAD_IDX = MR.vocab.stoi[MR.pad_token]
    SEN_PAD_IDX = SEN.vocab.stoi[SEN.pad_token]

    if chain_flag is False:
        if nlg_flag is True:
            model_nlg = Seq2Seq(enc_nlg, dec_nlg, MR_PAD_IDX, SEN_PAD_IDX, device).to(device)
            print('The model (nlg) has {} trainable parameters'.format(count_parameters(model_nlg)))
            model_nlg.apply(initialize_weights);
            optimizer_nlg = torch.optim.Adam(model_nlg.parameters(), lr = LEARNING_RATE)
            criterion_nlg = nn.CrossEntropyLoss(ignore_index = SEN_PAD_IDX)
            a_performance = {'train_loss_nlg': [], 'valid_loss_nlg': []}
        if nlu_flag is True:
            model_nlu = Seq2Seq(enc_nlu, dec_nlu, SEN_PAD_IDX, MR_PAD_IDX, device).to(device)
            print('The model (nlu) has {} trainable parameters'.format(count_parameters(model_nlu)))
            model_nlu.apply(initialize_weights);
            optimizer_nlu = torch.optim.Adam(model_nlu.parameters(), lr = LEARNING_RATE)
            criterion_nlu = nn.CrossEntropyLoss(ignore_index = MR_PAD_IDX)
            a_performance = {'train_loss_nlu': [], 'valid_loss_nlu': [], 'accuracy_nlu': [], 'precision_nlu': [], 'recall_nlu': [], 'f1score_nlu': [], 'correct_nlu': []}
    else:
        if args.mode.lower() == 'chain_c':
            f = open(args.p+'/best_model_nlg.pkl', 'rb')
            model_nlg = cloudpickle.load(f)
            f.close()
            f = open(args.p+'/best_model_nlu.pkl', 'rb')
            model_nlu = cloudpickle.load(f)
            f.close()
        else:
            model_nlg = Seq2Seq(enc_nlg, dec_nlg, MR_PAD_IDX, SEN_PAD_IDX, device).to(device)
            model_nlu = Seq2Seq(enc_nlu, dec_nlu, SEN_PAD_IDX, MR_PAD_IDX, device).to(device)
            print('The model (nlg) has {} trainable parameters'.format(count_parameters(model_nlg)))
            print('The model (nlu) has {} trainable parameters'.format(count_parameters(model_nlu)))
            model_nlg.apply(initialize_weights);
            model_nlu.apply(initialize_weights);

        optimizer_nlg = torch.optim.Adam(model_nlg.parameters(), lr = LEARNING_RATE)
        optimizer_nlu = torch.optim.Adam(model_nlu.parameters(), lr = LEARNING_RATE)
        criterion_nlg = nn.CrossEntropyLoss(ignore_index = SEN_PAD_IDX)
        criterion_nlu = nn.CrossEntropyLoss(ignore_index = MR_PAD_IDX)
        a_performance = {
            'train_loss_chain_nlg': [],
            'valid_loss_chain_nlg': [],
            'train_loss_chain_nlu': [],
            'valid_loss_chain_nlu': [],
            'accuracy_chain_nlu': [],
            'precision_chain_nlu': [],
            'recall_chain_nlu': [],
            'f1score_chain_nlu': [],
            'correct_chain_nlu': [],
            'train_loss_nlg': [],
            'valid_loss_nlg': [],
            'train_loss_nlu': [],
            'valid_loss_nlu': [],
            'accuracy_nlu': [],
            'precision_nlu': [],
            'recall_nlu': [],
            'f1score_nlu': [],
            'correct_nlu': []
        }

    best_valid_loss_nlg = float('inf')
    best_valid_loss_nlu = float('inf')
    best_epoch_nlg = 0
    best_epoch_nlu = 0

    print('** NL training **')
    if chain_flag is False:
        if nlg_flag is True:
            # NLG only
            for epoch in range(N_EPOCHS):
                print('Epoch(NLG only): {} begin ...'.format(epoch))
                start_time = time.time()
                train_loss_nlg = train_nlg(model_nlg, train_iterator, optimizer_nlg, criterion_nlg, CLIP, args.v)
                valid_loss_nlg = evaluate_nlg(model_nlg, valid_iterator, criterion_nlg)
                best_valid_loss_nlg, best_epoch_nlg = save_best_nlg(args.p, best_valid_loss_nlg, best_epoch_nlg, valid_loss_nlg, epoch, model_nlg, MR, SEN, False)

                a_performance['train_loss_nlg'].append(train_loss_nlg)
                a_performance['valid_loss_nlg'].append(valid_loss_nlg)
                f = open(args.p+'/performance.json', 'w', encoding='utf-8')
                json.dump(a_performance, f, ensure_ascii=False, sort_keys=True)
                f.close()

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('Epoch:(NLG only) {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
                print('-NLG-')
                print('\tTrain Loss: {} | Train PPL: {}'.format(train_loss_nlg, math.exp(train_loss_nlg)))
                print('\t Val. Loss: {} |  Val. PPL: {}'.format(valid_loss_nlg, math.exp(valid_loss_nlg)))

        elif nlu_flag is True:
            # NLU only
            for epoch in range(N_EPOCHS):
                print('Epoch(NLU only): {} begin ...'.format(epoch))
                start_time = time.time()
                train_loss_nlu = train_nlu(model_nlu, train_iterator, optimizer_nlu, criterion_nlu, CLIP, args.v)
                valid_loss_nlu = evaluate_nlu(model_nlu, valid_iterator, criterion_nlu)
                best_valid_loss_nlu, best_epoch_nlu = save_best_nlu(args.p, best_valid_loss_nlu, best_epoch_nlu, valid_loss_nlu, epoch, model_nlu, MR, SEN, False)
                accuracy, precision, recall, f1score, correct = calculate_valid(args.p, valid_data, SEN, MR, model_nlu, device, chain_flag)

                a_performance['train_loss_nlu'].append(train_loss_nlu)
                a_performance['valid_loss_nlu'].append(valid_loss_nlu)
                a_performance['accuracy_nlu'].append(accuracy)
                a_performance['precision_nlu'].append(precision)
                a_performance['recall_nlu'].append(recall)
                a_performance['f1score_nlu'].append(f1score)
                a_performance['correct_nlu'].append(correct)
                f = open(args.p+'/performance.json', 'w', encoding='utf-8')
                json.dump(a_performance, f, ensure_ascii=False, sort_keys=True)
                f.close()

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('Epoch(NLU only): {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
                print('-NLU-')
                print('\tTrain Loss: {} | Train PPL: {}'.format(train_loss_nlu, math.exp(train_loss_nlu)))
                print('\t Val. Loss: {} |  Val. PPL: {}'.format(valid_loss_nlu, math.exp(valid_loss_nlu)))
    else:
        # NLchain
        if (args.mode.lower() == 'chain_a') or (args.mode.lower() == 'chain_c'):
            best_valid_loss_nlg = float('inf')
            best_valid_loss_nlu = float('inf')
            best_epoch_nlg = 0
            best_epoch_nlu = 0

            # (A) simultanous
            for epoch in range(N_EPOCHS):
                print('Epoch(NLU/NLG+NLchain): {} begin ...'.format(epoch))
                start_time = time.time()

                # supervised training w/ paired data
                train_loss_nlg = train_nlg(model_nlg, train_iterator, optimizer_nlg, criterion_nlg, CLIP, args.v)
                train_loss_nlu = train_nlu(model_nlu, train_iterator, optimizer_nlu, criterion_nlu, CLIP, args.v)

                # unsupervised training w/ unpaired data
                train_loss_chain_nlg, train_loss_chain_nlu = train_chain(model_nlg, model_nlu, train_aug_iterator, optimizer_nlg, optimizer_nlu, criterion_nlg, criterion_nlu, CLIP, args.beta, args.v)

                # validation
                valid_loss_nlg = evaluate_nlg(model_nlg, valid_iterator, criterion_nlg)
                valid_loss_nlu = evaluate_nlu(model_nlu, valid_iterator, criterion_nlu)
    
                best_valid_loss_nlg, best_epoch_nlg = save_best_nlg(args.p, best_valid_loss_nlg, best_epoch_nlg, valid_loss_nlg, epoch, model_nlg, MR, SEN, True)
                best_valid_loss_nlu, best_epoch_nlu = save_best_nlu(args.p, best_valid_loss_nlu, best_epoch_nlu, valid_loss_nlu, epoch, model_nlu, MR, SEN, True)
                accuracy, precision, recall, f1score, correct = calculate_valid(args.p, valid_data, SEN, MR, model_nlu, device, chain_flag)

                a_performance['train_loss_chain_nlu'].append(train_loss_chain_nlu)
                a_performance['train_loss_chain_nlg'].append(train_loss_chain_nlg)
                a_performance['train_loss_nlg'].append(train_loss_nlg)
                a_performance['valid_loss_nlg'].append(valid_loss_nlg)
                a_performance['train_loss_nlu'].append(train_loss_nlu)
                a_performance['valid_loss_nlu'].append(valid_loss_nlu)
                a_performance['accuracy_nlu'].append(accuracy)
                a_performance['precision_nlu'].append(precision)
                a_performance['recall_nlu'].append(recall)
                a_performance['f1score_nlu'].append(f1score)
                a_performance['correct_nlu'].append(correct)
                f = open(args.p+'/performance.json', 'w', encoding='utf-8')
                json.dump(a_performance, f, ensure_ascii=False, sort_keys=True)
                f.close()

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('Epoch(NLU/NLG+NLchain): {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
                print('-NLG-')
                print('\tTrain        Loss: {} | PPL: {}'.format(train_loss_nlg, math.exp(train_loss_nlg)))
                print('\tTrain(chain) Loss: {} | PPL: {}'.format(train_loss_chain_nlg, math.exp(train_loss_chain_nlg)))
                print('\tValidation   Loss: {} | PPL: {}'.format(valid_loss_nlg, math.exp(valid_loss_nlg)))
                print('-NLU-')
                print('\tTrain        Loss: {} | PPL: {}'.format(train_loss_nlu, math.exp(train_loss_nlu)))
                print('\tTrain(chain) Loss: {} | PPL: {}'.format(train_loss_chain_nlu, math.exp(train_loss_chain_nlu)))
                print('\tValidation   Loss: {} | PPL: {}'.format(valid_loss_nlu, math.exp(valid_loss_nlu)))

        if args.mode.lower() == 'chain_b':
            # (B) alternateness
            # supervised training w/ paired data
            best_valid_loss_nlg = float('inf')
            best_valid_loss_nlu = float('inf')
            best_epoch_nlg = 0
            best_epoch_nlu = 0
            for epoch in range(N_EPOCHS):
                print('Epoch(NLU/NLG): {} begin ...'.format(epoch))
                start_time = time.time()

                # supervised training
                train_loss_nlg = train_nlg(model_nlg, train_iterator, optimizer_nlg, criterion_nlg, CLIP, args.v)
                train_loss_nlu = train_nlu(model_nlu, train_iterator, optimizer_nlu, criterion_nlu, CLIP, args.v)
                # validation
                valid_loss_nlg = evaluate_nlg(model_nlg, valid_iterator, criterion_nlg)
                valid_loss_nlu = evaluate_nlu(model_nlu, valid_iterator, criterion_nlu)
    
                best_valid_loss_nlg, best_epoch_nlg = save_best_nlg(args.p, best_valid_loss_nlg, best_epoch_nlg, valid_loss_nlg, epoch, model_nlg, MR, SEN, False)
                best_valid_loss_nlu, best_epoch_nlu = save_best_nlu(args.p, best_valid_loss_nlu, best_epoch_nlu, valid_loss_nlu, epoch, model_nlu, MR, SEN, False)
                accuracy, precision, recall, f1score, correct = calculate_valid(args.p, valid_data, SEN, MR, model_nlu, device, False)

                a_performance['train_loss_nlg'].append(train_loss_nlg)
                a_performance['valid_loss_nlg'].append(valid_loss_nlg)
                a_performance['train_loss_nlu'].append(train_loss_nlu)
                a_performance['valid_loss_nlu'].append(valid_loss_nlu)
                a_performance['accuracy_nlu'].append(accuracy)
                a_performance['precision_nlu'].append(precision)
                a_performance['recall_nlu'].append(recall)
                a_performance['f1score_nlu'].append(f1score)
                a_performance['correct_nlu'].append(correct)
                f = open(args.p+'/performance.json', 'w', encoding='utf-8')
                json.dump(a_performance, f, ensure_ascii=False, sort_keys=True)
                f.close()

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('Epoch(NLU/NLG): {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
                print('-NLG-')
                print('\tTrain        Loss: {} | PPL: {}'.format(train_loss_nlg, math.exp(train_loss_nlg)))
                print('\tValidation   Loss: {} | PPL: {}'.format(valid_loss_nlg, math.exp(valid_loss_nlg)))
                print('-NLU-')
                print('\tTrain        Loss: {} | PPL: {}'.format(train_loss_nlu, math.exp(train_loss_nlu)))
                print('\tValidation   Loss: {} | PPL: {}'.format(valid_loss_nlu, math.exp(valid_loss_nlu)))

            '''
            # load best model
            f = open(args.p+'/best_model_nlg.pkl', 'rb')
            model_nlg = cloudpickle.load(f)
            f.close()
            f = open(args.p+'/best_model_nlu.pkl', 'rb')
            model_nlu = cloudpickle.load(f)
            f.close()
            '''

            # unsupervised training w/ unpaired data
            best_valid_loss_nlg = float('inf')
            best_valid_loss_nlu = float('inf')
            best_epoch_nlg = 0
            best_epoch_nlu = 0
            for epoch in range(N_EPOCHS):
                print('Epoch(NLchain): {} begin ...'.format(epoch))
                start_time = time.time()

                # unsupervised training
                train_loss_chain_nlg, train_loss_chain_nlu = train_chain(model_nlg, model_nlu, train_aug_iterator, optimizer_nlg, optimizer_nlu, criterion_nlg, criterion_nlu, CLIP, args.beta, args.v)

                # validation
                valid_loss_nlg = evaluate_nlg(model_nlg, valid_iterator, criterion_nlg)
                valid_loss_nlu = evaluate_nlu(model_nlu, valid_iterator, criterion_nlu)
    
                best_valid_loss_nlg, best_epoch_nlg = save_best_nlg(args.p, best_valid_loss_nlg, best_epoch_nlg, valid_loss_nlg, epoch, model_nlg, MR, SEN, True)
                best_valid_loss_nlu, best_epoch_nlu = save_best_nlu(args.p, best_valid_loss_nlu, best_epoch_nlu, valid_loss_nlu, epoch, model_nlu, MR, SEN, True)
                accuracy, precision, recall, f1score, correct = calculate_valid(args.p, valid_data, SEN, MR, model_nlu, device, chain_flag)

                a_performance['train_loss_chain_nlg'].append(train_loss_chain_nlg)
                a_performance['train_loss_chain_nlu'].append(train_loss_chain_nlu)
                a_performance['valid_loss_chain_nlg'].append(valid_loss_nlg)
                a_performance['valid_loss_chain_nlu'].append(valid_loss_nlu)
                a_performance['accuracy_chain_nlu'].append(accuracy)
                a_performance['precision_chain_nlu'].append(precision)
                a_performance['recall_chain_nlu'].append(recall)
                a_performance['f1score_chain_nlu'].append(f1score)
                a_performance['correct_chain_nlu'].append(correct)
                f = open(args.p+'/performance.json', 'w', encoding='utf-8')
                json.dump(a_performance, f, ensure_ascii=False, sort_keys=True)
                f.close()

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('Epoch(NLchain): {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
                print('-NLG-')
                print('\tTrain(chain) Loss: {} | PPL: {}'.format(train_loss_chain_nlg, math.exp(train_loss_chain_nlg)))
                print('\tValidation   Loss: {} | PPL: {}'.format(valid_loss_nlg, math.exp(valid_loss_nlg)))
                print('-NLU-')
                print('\tTrain(chain) Loss: {} | PPL: {}'.format(train_loss_chain_nlu, math.exp(train_loss_chain_nlu)))
                print('\tValidation   Loss: {} | PPL: {}'.format(valid_loss_nlu, math.exp(valid_loss_nlu)))

    # check training performance
    if nlg_flag is True:
        if chain_flag is False:
            f = open(args.p+'/best_model_nlg.pkl', 'rb')
            model_nlg = cloudpickle.load(f)
            f.close()
            #model_nlg.load_state_dict(torch.load(args.p+'/best_model_nlg.pt'))
        else:
            f = open(args.p+'/best_model_chain_nlg.pkl', 'rb')
            model_nlg = cloudpickle.load(f)
            f.close()
            #model_nlg.load_state_dict(torch.load(args.p+'/best_model_chain_nlg.pt'))
        test_loss_nlg = evaluate_nlg(model_nlg, test_iterator, criterion_nlg)
        print('-NLG-')
        print('| Test Loss: {} | Test PPL: {} | Epoch: {}'.format(test_loss_nlg, math.exp(test_loss_nlg), best_epoch_nlg))

    if nlu_flag is True:
        if chain_flag is False:
            f = open(args.p+'/best_model_nlu.pkl', 'rb')
            model_nlu = cloudpickle.load(f)
            f.close()
            #model_nlu.load_state_dict(torch.load(args.p+'/best_model_nlu.pt'))
        else:
            f = open(args.p+'/best_model_chain_nlu.pkl', 'rb')
            model_nlu = cloudpickle.load(f)
            f.close()
            #model_nlu.load_state_dict(torch.load(args.p+'/best_model_chain_nlu.pt'))
        test_loss_nlu = evaluate_nlu(model_nlu, test_iterator, criterion_nlu)
        print('-NLU-')
        print('| Test Loss: {} | Test PPL: {} | Epoch: {}'.format(test_loss_nlu, math.exp(test_loss_nlu), best_epoch_nlu))

    # check performance
    if nlg_flag is True:
        print('\n** check performance (NLG) **')
        NLG = NLC(args.p, 'NLG', chain_flag)

        # training data
        example_idx = 8
        print('** NLG: training data (idx: '+str(example_idx)+') *')
        mr = vars(train_data.examples[example_idx])['mr']
        sen = vars(train_data.examples[example_idx])['sen']
        print('mr = {}'.format(mr))
        print('sen(correct) = {}'.format(sen))
        translation, attention = NLG.translate_sentence(mr, MR, SEN, model_nlg, device)
        print('sen(predict) = {}'.format(translation))
        if args.graph is True:
            NLG.display_attention(mr, translation, attention)

        # validation data
        example_idx = 6
        print('** NLG: validation data (idx: '+str(example_idx)+') *')
        mr = vars(valid_data.examples[example_idx])['mr']
        sen = vars(valid_data.examples[example_idx])['sen']
        print('mr = {}'.format(mr))
        print('sen(correct) = {}'.format(sen))
        translation, attention = NLG.translate_sentence(mr, MR, SEN, model_nlg, device)
        print('sen(predict) = {}'.format(translation))
        if args.graph is True:
            NLG.display_attention(mr, translation, attention)

        # test data
        if args.eval is True:
            f = open(args.p+'/test_nlg.tsv', 'w', encoding='utf-8')
            f.write('mr\tsen(correct)\tsen(predict)\n')
            for example_idx in range(len(test_data.examples)):
                print('** NLG: test data (idx: '+str(example_idx)+') *')
                mr = vars(test_data.examples[example_idx])['mr']
                sen = vars(test_data.examples[example_idx])['sen']
                print('mr = {}'.format(mr))
                print('sen(correct) = {}'.format(sen))
                translation, attention = NLG.translate_sentence(mr, MR, SEN, model_nlg, device)
                print('sen(predict) = {}'.format(translation))
                f.write(str(mr)+'\t'+str(sen)+'\t'+str(translation)+'\n')
                if args.graph is True:
                    NLG.display_attention(mr, translation, attention)
            f.close()

    # check performance
    if nlu_flag is True:
        print('\n** check performance (NLU) **')
        NLU = NLC(args.p, 'NLU', chain_flag)

        # training data
        example_idx = 8
        print('** NLU: training data (idx: '+str(example_idx)+') *')
        sen = vars(train_data.examples[example_idx])['sen']
        mr = vars(train_data.examples[example_idx])['mr']
        print('sen = {}'.format(sen))
        print('mr(correct) = {}'.format(mr))
        translation, attention = NLU.translate_sentence(sen, SEN, MR, model_nlu, device)
        print('mr(predict) = {}'.format(translation[:-1]))
        if args.graph is True:
            NLU.display_attention(sen, translation, attention)

        # validation data
        example_idx = 6
        print('** NLU: validation data (idx: '+str(example_idx)+') *')
        sen = vars(valid_data.examples[example_idx])['sen']
        mr = vars(valid_data.examples[example_idx])['mr']
        print('sen = {}'.format(sen))
        print('mr(correct) = {}'.format(mr))
        translation, attention = NLU.translate_sentence(sen, SEN, MR, model_nlu, device)
        print('mr(predict) = {}'.format(translation[:-1]))
        if args.graph is True:
            NLU.display_attention(sen, translation, attention)

        # test data
        if args.eval is True:
            f = open(args.p+'/test_nlu.tsv', 'w', encoding='utf-8')
            f.write('sen\tmr(correct)\tmr(predict)\n')
            for example_idx in range(len(test_data.examples)):
                print('** NLU: test data (idx: '+str(example_idx)+') *')
                sen = vars(test_data.examples[example_idx])['sen']
                mr = vars(test_data.examples[example_idx])['mr']
                print('sen = {}'.format(sen))
                print('mr(correct) = {}'.format(mr))
                translation, attention = NLU.translate_sentence(sen, SEN, MR, model_nlu, device)
                print('mr(predict) = {}'.format(translation[:-1]))
                f.write(str(sen)+'\t'+str(mr)+'\t'+str(translation)+'\n')
                if args.graph is True:
                    NLU.display_attention(sen, translation, attention)
            f.close()
    print('** done **')
