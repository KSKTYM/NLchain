#! python
# -*- coding: utf-8 -*-

import argparse
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
import dill
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

def train_nlg(model_nlg, iterator, optimizer, criterion, clip):
    model_nlg.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        mr = batch.mr
        sen = batch.sen
        optimizer.zero_grad()
        output, _ = model_nlg(mr, sen[:,:-1])
        #output = [batch size, sen len - 1, output dim]
        #sen = [batch size, sen len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        sen = sen[:,1:].contiguous().view(-1)
        #output = [batch size * sen len - 1, output dim]
        #sen = [batch size * sen len - 1]
            
        loss = criterion(output, sen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nlg.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def train_nlu(model_nlu, iterator, optimizer, criterion, clip):
    model_nlu.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        sen = batch.sen
        mr = batch.mr
        optimizer.zero_grad()
        output, _ = model_nlu(sen, mr[:,:-1])
        #output = [batch size, mr len - 1, output dim]
        #mr = [batch size, mr len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        mr = mr[:,1:].contiguous().view(-1)
        #output = [batch size * mr len - 1, output dim]
        #mr = [batch size * mr len - 1]
            
        loss = criterion(output, mr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nlu.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate_nlg(model_nlg, iterator, criterion):
    model_nlg.eval()
    epoch_loss = 0

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
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate_nlu(model_nlu, iterator, criterion):
    model_nlu.eval()
    epoch_loss = 0

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
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

'''
def calculate_bleu(data, mr_field, sen_field, model, device, max_len = 50):
    sens = []
    pred_sens = []
    for datum in data:
        mr = vars(datum)['mr']
        sen = vars(datum)['sen']
        pred_sen, _ = NLC.translate_sentence(mr, mr_field, sen_field, model, device, max_len)
        #cut off <eos> token
        pred_sen = pred_sen[:-1]
        pred_sens.append(pred_sen)
        sens.append([sen])
        
    return bleu_score(pred_sens, sens)
'''

## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='parameter directory\'s name', default='../parameter/')
    parser.add_argument('-path', help='training data directory\'s name', default='../corpus/')
    parser.add_argument('-file', help='training data file name', default='nlc_transformer')
    parser.add_argument('-seed', help='seed number', type=int, default=1234)
    parser.add_argument('-epoch', help='epoch number', type=int, default=10)
    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-alpha', help='alpha parameter', type=float, default=0.5)
    parser.add_argument('-beta', help='beta parameter', type=float, default=1.0)
    parser.add_argument('-graph', help='show attention graph', action='store_true')
    parser.add_argument('-v', help='verbose(print debug)', action='store_true')
    args = parser.parse_args()

    print('** NLchain **')
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

    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()

    # NLU: TRG->SRC, NLG: SRC->TRG
    # SRC: MR, TRG: Sentence
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
    MR.build_vocab(train_data, min_freq = 1)
    SEN.build_vocab(train_data, min_freq = 1)

    if args.v is True:
        print(str(len(MR.vocab.itos)))#80
        print(MR.vocab.itos)
        print(str(len(MR.vocab.stoi)))
        print(MR.vocab.stoi)
        print(str(len(SEN.vocab.itos)))#2267
        print(SEN.vocab.itos)
        print(str(len(SEN.vocab.stoi)))
        print(SEN.vocab.stoi)

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

    model_nlg = Seq2Seq(enc_nlg, dec_nlg, MR_PAD_IDX, SEN_PAD_IDX, device).to(device)
    model_nlu = Seq2Seq(enc_nlu, dec_nlu, SEN_PAD_IDX, MR_PAD_IDX, device).to(device)
    print('The model has {} trainable parameters'.format(count_parameters(model_nlg)))
    model_nlg.apply(initialize_weights);
    model_nlu.apply(initialize_weights);

    optimizer_nlg = torch.optim.Adam(model_nlg.parameters(), lr = LEARNING_RATE)
    optimizer_nlu = torch.optim.Adam(model_nlu.parameters(), lr = LEARNING_RATE)
    criterion_nlg = nn.CrossEntropyLoss(ignore_index = SEN_PAD_IDX)
    criterion_nlu = nn.CrossEntropyLoss(ignore_index = MR_PAD_IDX)

    a_performance = {'train_loss_nlg': [], 'valid_loss_nlg': [], 'train_loss_nlu': [], 'valid_loss_nlu': []}

    best_valid_loss_nlg = float('inf')
    best_valid_loss_nlu = float('inf')
    best_epoch_nlg = 0
    best_epoch_nlu = 0

    print('** NL training **')
    for epoch in range(N_EPOCHS):
        print('Epoch: {} begin ...'.format(epoch))
        start_time = time.time()

        #train_loss_nlg = train_nlg(model_nlg, train_iterator, optimizer_nlg, criterion_nlg, CLIP)
        #train_loss_nlu = train_nlu(model_nlu, train_iterator, optimizer_nlu, criterion_nlu, CLIP)

        ## A. Supervised training with sentence-MR data pairs
        # NLU
        '''
        model_nlg.train()
        model_nlu.train()
        '''
        epoch_loss_nlg = 0
        epoch_loss_nlu = 0 

        for i, batch in enumerate(train_iterator):
            model_nlg.train()
            model_nlu.train()

            sen = batch.sen
            mr = batch.mr#128-gyou,10-retsu(include <sos:2> and <eos:3>)
            if args.v is True:
                print('***** i : '+str(i)+' *****')
                print('(1) sen: '+str(sen.size()))
                print(sen)
                print('(2) mr: '+str(mr.size()))
                print(mr)
                print('(3) sen[:,:-1]')
                print(sen[:,:-1])
                print('(4) mr[:,:-1]')
                print(mr[:,:-1])

            optimizer_nlg.zero_grad()
            optimizer_nlu.zero_grad()

            output_nlg, _ = model_nlg(mr, sen[:,:-1])
            output_nlu, _ = model_nlu(sen, mr[:,:-1])
            if args.v is True:
                print('(5) output_nlg: '+str(output_nlg.size()))
                print(output_nlg)
                print('(6) output_nlu: '+str(output_nlu.size()))
                print(output_nlu)

            output_nlg_dim = output_nlg.shape[-1]
            output_nlu_dim = output_nlu.shape[-1]

            output_nlg = output_nlg.contiguous().view(-1, output_nlg_dim)
            output_nlu = output_nlu.contiguous().view(-1, output_nlu_dim)
            if args.v is True:
                print('(7) output_nlg: '+str(output_nlg.size()))
                print(output_nlg)
                print('(8) output_nlu: '+str(output_nlu.size()))
                print(output_nlu)

            sen = sen[:,1:].contiguous().view(-1)
            mr = mr[:,1:].contiguous().view(-1)
            # means of 1: -> remove <sos> on top
            if args.v is True:
                print('(9) sen:'+str(sen.size()))
                print(sen)
                print('(10) mr:'+str(mr.size()))
                print(mr)

            loss_nlg = args.alpha * criterion_nlg(output_nlg, sen)
            loss_nlu = args.alpha * criterion_nlu(output_nlu, mr)
            if args.v is True:
                print('(11) loss_nlg: '+str(loss_nlg.size()))
                print(loss_nlg)
                print('(12) loss_nlu: '+str(loss_nlu.size()))
                print(loss_nlu)

            ### TEST (from here)
            ## (mr->)NLG->sentence->NLU->mr
            a_output_nlg_token = output_nlg.argmax(1).tolist()
            len_sentence = int(len(a_output_nlg_token)/BATCH_SIZE)
            if args.v is True:
                print('(A-1) output_nlg.argmax(1): '+str(output_nlg.argmax(1).size()))
                print(output_nlg.argmax(1))
                print('(A-2) output_nlg.argmax(1).tolist(): (len)'+str(len(output_nlg.argmax(1).tolist())))
                print(a_output_nlg_token)
                print('(A-3) len_sentence: '+str(len_sentence))

            tmp_a = []
            num_batch = len(batch.mr)
            for j in range(num_batch):
                tmp_a.append(SEN.vocab.stoi['<sos>'])
                tmp_a.extend(a_output_nlg_token[j*len_sentence:(j+1)*len_sentence])
            tmp_b = torch.Tensor(tmp_a).long()
            generated_sentence = tmp_b.reshape(num_batch, len_sentence+1).to(device)

            if args.v is True:
                print('(A-4) len(tmp_a): '+str(len(tmp_a)))
                print(tmp_a)
                print('(A-5) size(tmp_b): '+str(tmp_b.size()))
                print('(A-6) generated_sentence: '+str(generated_sentence.size()))
                print(generated_sentence)

            del tmp_a, tmp_b
            mr2 = batch.mr
            output_nlu2, _ = model_nlu(generated_sentence, mr2[:, :-1])

            if args.v is True:
                print('(A-7) mr2: '+str(mr2.size()))
                print(mr2)
                print('(A-8) mr2[:,:-1]')
                print(mr2[:,:-1])
                print('(A-9) output_nlu2: '+str(output_nlu2.size()))
                print(output_nlu2)

            output_nlu2_dim = output_nlu2.shape[-1]
            output_nlu2 = output_nlu2.contiguous().view(-1, output_nlu2_dim)
            '''
            a_output_nlu2_token = output_nlu2.argmax(1).tolist()
            tokens = [MR.vocab.itos[j] for j in a_output_nlu2_token]
            '''
            mr = batch.mr
            mr = mr[:,1:].contiguous().view(-1)
            loss_nlu += args.beta * criterion_nlu(output_nlu2, mr)
            if args.v is True:
                print('(A-10) loss_nlu: '+str(loss_nlu.size()))
                print(loss_nlu)

            ## (sentence->)NLU->mr->NLG->sentence
            a_output_nlu_token = output_nlu.argmax(1).tolist()
            len_mr = int(len(a_output_nlu_token)/BATCH_SIZE)
            if args.v is True:
                print('(B-1) output_nlu.argmax(1): '+str(output_nlu.argmax(1).size()))
                print(output_nlu.argmax(1))
                print('(B-2) output_nlu.argmax(1).tolist(): (len)'+str(len(output_nlu.argmax(1).tolist())))
                print(a_output_nlu_token)
                print('(B-3) len_mr: '+str(len_mr))

            tmp_a = []
            num_batch = len(batch.sen)
            for j in range(num_batch):
                tmp_a.append(MR.vocab.stoi['<sos>'])
                tmp_a.extend(a_output_nlu_token[j*len_mr:(j+1)*len_mr])
            tmp_b = torch.Tensor(tmp_a).long()
            generated_mr = tmp_b.reshape(num_batch, len_mr+1).to(device)
            if args.v is True:
                print('(B-4) len(tmp_a): '+str(len(tmp_a)))
                print(tmp_a)
                print('(B-5) size(tmp_b): '+str(tmp_b.size()))
                print('(B-6) generated_mr: '+str(generated_mr.size()))
                print(generated_mr)

            del tmp_a, tmp_b
            sen2 = batch.sen
            output_nlg2, _ = model_nlg(generated_mr, sen2[:, :-1])
            if args.v is True:
                print('(B-7) sen2: '+str(sen2.size()))
                print(sen2)
                print('(B-8) sen2[:,:-1]')
                print(sen2[:,:-1])
                print('(B-9) output_nlg2: '+str(output_nlg2.size()))
                print(output_nlg2)

            output_nlg2_dim = output_nlg2.shape[-1]
            output_nlg2 = output_nlg2.contiguous().view(-1, output_nlg2_dim)
            '''
            a_output_nlg2_token = output_nlg2.argmax(1).tolist()
            tokens = [SEN.vocab.itos[j] for j in a_output_nlg2_token]
            '''
            sen = batch.sen
            sen = sen[:,1:].contiguous().view(-1)
            loss_nlg += args.beta * criterion_nlg(output_nlg2, sen)
            if args.v is True:
                print('(B-10) loss_nlg: '+str(loss_nlg.size()))
                print(loss_nlg)
            ### TEST (to here)

            loss_nlg.backward()
            loss_nlu.backward()

            torch.nn.utils.clip_grad_norm_(model_nlg.parameters(), CLIP)
            torch.nn.utils.clip_grad_norm_(model_nlu.parameters(), CLIP)

            optimizer_nlg.step()
            optimizer_nlu.step()

            epoch_loss_nlg += loss_nlg.item()
            epoch_loss_nlu += loss_nlu.item()

            model_nlg.eval()
            model_nlu.eval()

        train_loss_nlg = epoch_loss_nlg / len(train_iterator)
        train_loss_nlu = epoch_loss_nlu / len(train_iterator)

        valid_loss_nlg = evaluate_nlg(model_nlg, valid_iterator, criterion_nlg)
        valid_loss_nlu = evaluate_nlu(model_nlu, valid_iterator, criterion_nlu)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss_nlg < best_valid_loss_nlg:
            best_valid_loss_nlg = valid_loss_nlg
            best_epoch_nlg = epoch
            torch.save(model_nlg.state_dict(), args.p+'/model_nlg.pt')

            f = open(args.p+'/model_nlg.pkl', 'wb')
            cloudpickle.dump(model_nlg, f)
            f.close()
            f = open(args.p+'/best_epoch_nlg.txt', 'w')
            f.write(str(epoch))
            f.close()

        if valid_loss_nlu < best_valid_loss_nlu:
            best_valid_loss_nlu = valid_loss_nlu
            best_epoch_nlu = epoch
            torch.save(model_nlu.state_dict(), args.p+'/model_nlu.pt')
    
            f = open(args.p+'/model_nlu.pkl', 'wb')
            cloudpickle.dump(model_nlu, f)
            f.close()
            f = open(args.p+'/best_epoch_nlu.txt', 'w')
            f.write(str(epoch))
            f.close()

        print('Epoch: {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
        print('-NLG-')
        print('\tTrain Loss: {} | Train PPL: {}'.format(train_loss_nlg, math.exp(train_loss_nlg)))
        print('\t Val. Loss: {} |  Val. PPL: {}'.format(valid_loss_nlg, math.exp(valid_loss_nlg)))
        a_performance['train_loss_nlg'].append(train_loss_nlg)
        a_performance['valid_loss_nlg'].append(valid_loss_nlg)
        print('-NLU-')
        print('\tTrain Loss: {} | Train PPL: {}'.format(train_loss_nlu, math.exp(train_loss_nlu)))
        print('\t Val. Loss: {} |  Val. PPL: {}'.format(valid_loss_nlu, math.exp(valid_loss_nlu)))
        a_performance['train_loss_nlu'].append(train_loss_nlu)
        a_performance['valid_loss_nlu'].append(valid_loss_nlu)

    '''
    f = open(args.p+'/model_nlg.pkl', 'wb')
    cloudpickle.dump(model_nlg, f)
    f.close()
    f = open(args.p+'/model_nlu.pkl', 'wb')
    cloudpickle.dump(model_nlu, f)
    f.close()
    '''
    f = open(args.p+'/MR.field', 'wb')
    dill.dump(MR, f)
    f.close()
    f = open(args.p+'/SEN.field', 'wb')
    dill.dump(SEN, f)
    f.close()
    f = open(args.p+'/performance.json', 'w', encoding='utf-8')
    json.dump(a_performance, f, ensure_ascii=False, sort_keys=True)
    f.close()

    # check training performance
    model_nlg.load_state_dict(torch.load(args.p+'/model_nlg.pt'))
    model_nlu.load_state_dict(torch.load(args.p+'/model_nlu.pt'))
    test_loss_nlg = evaluate_nlg(model_nlg, test_iterator, criterion_nlg)
    test_loss_nlu = evaluate_nlu(model_nlu, test_iterator, criterion_nlu)
    print('-NLG-')
    print('| Test Loss: {} | Test PPL: {} | Epoch: {}'.format(test_loss_nlg, math.exp(test_loss_nlg), best_epoch_nlg))
    print('-NLU-')
    print('| Test Loss: {} | Test PPL: {} | Epoch: {}'.format(test_loss_nlu, math.exp(test_loss_nlu), best_epoch_nlu))

    # check performance
    print('\n** check performance (NLG) **')
    NLG = NLC(args.p, 'NLG')

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
    '''
    # test data
    for example_idx in range(len(test_data.examples)):
        print('** NLG: test data (idx: '+str(example_idx)+') *')
        mr = vars(test_data.examples[example_idx])['mr']
        sen = vars(test_data.examples[example_idx])['sen']
        print('mr = {}'.format(mr))
        print('sen(correct) = {}'.format(sen))
        translation, attention = NLG.translate_sentence(mr, MR, SEN, model_nlg, device)
        print('sen(predict) = {}'.format(translation))
        if args.graph is True:
            NLG.display_attention(mr, translation, attention)
    '''

    '''
    # BLUE score (NLG)
    bleu_score = calculate_bleu(test_data, MR, SEN, model_nlg, device)
    print('BLEU score = {}'.format(bleu_score*100))
    '''
    # check performance
    print('\n** check performance (NLU) **')
    NLU = NLC(args.p, 'NLU')

    # training data
    example_idx = 8
    print('** NLU: training data (idx: '+str(example_idx)+') *')
    sen = vars(train_data.examples[example_idx])['sen']
    mr = vars(train_data.examples[example_idx])['mr']
    print('sen = {}'.format(sen))
    print('mr(correct) = {}'.format(mr))
    translation, attention = NLU.translate_sentence(sen, SEN, MR, model_nlu, device)
    print('mr(predict) = {}'.format(translation))
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
    print('mr(predict) = {}'.format(translation))
    if args.graph is True:
        NLU.display_attention(sen, translation, attention)
    '''
    # test data
    for example_idx in range(len(test_data.examples)):
        print('** NLU: test data (idx: '+str(example_idx)+') *')
        sen = vars(test_data.examples[example_idx])['sen']
        mr = vars(test_data.examples[example_idx])['mr']
        print('sen = {}'.format(sen))
        print('mr(correct) = {}'.format(mr))
        translation, attention = NLU.translate_sentence(sen, SEN, MR, model_nlu, device)
        print('mr(predict) = {}'.format(translation))
        if args.graph is True:
            NLU.display_attention(sen, translation, attention)
    '''

    print('** done **')
