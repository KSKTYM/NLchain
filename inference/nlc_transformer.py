#! python
# -*- coding: utf-8 -*-

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib

import sys
sys.path.append('..')
from common.tokenizer import Tokenizer

import cloudpickle
import dill
import re

class NLC():
    def __init__(self, param_dir, method_conv):
        self.tokenizer = Tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if method_conv == 'NLG':
            #f = open(param_dir+'/model_nlg.pkl', 'rb')
            f = open(param_dir+'/best_model_nlg.pkl', 'rb')
            self.model = cloudpickle.load(f)
            f.close()
            f = open(param_dir+'/MR.field', 'rb')
            self.SRC = dill.load(f)
            f.close()
            f = open(param_dir+'/SEN.field', 'rb')
            self.TRG = dill.load(f)
            f.close()
        else:
            #f = open(param_dir+'/model_nlu.pkl', 'rb')
            f = open(param_dir+'/best_model_nlu.pkl', 'rb')
            self.model = cloudpickle.load(f)
            f.close()
            f = open(param_dir+'/SEN.field', 'rb')
            self.SRC = dill.load(f)
            f.close()
            f = open(param_dir+'/MR.field', 'rb')
            self.TRG = dill.load(f)
            f.close()

    def translate_sentence(self, sentence, src_field, trg_field, model, device, max_len = 50):
        model.eval()
        
        tokens = [token.lower() for token in sentence]
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)
    
        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
        
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                '''
                print('#output#')
                print(str(output.size()))
                print(output)
                '''
            pred_token = output.argmax(2)[:,-1].item()
            '''
            print('#output.argmax(2)')
            print(output.argmax(2))
            print('#output.argmax(2)[:,-1]')
            print(output.argmax(2)[:,-1])
            print('#pred_token#')
            print(pred_token)
            '''
            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
        '''
        print('#len(trg_indexes): '+str(len(trg_indexes)))
        print('#trg_indexes')
        print(trg_indexes)
        '''
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        return trg_tokens[1:], attention

    def display_attention(self, sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
        assert n_rows * n_cols == n_heads
        fig = plt.figure(figsize=(15,25))
        for i in range(n_heads):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            _attention = attention.squeeze(0)[i].cpu().detach().numpy()
            cax = ax.matshow(_attention, cmap='bone')

            ax.tick_params(labelsize=12)
            ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                               rotation=45)
            ax.set_yticklabels(['']+translation)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
        plt.close()

    def convert(self, input_data, mode, display_flag=False):
        if mode == 'NLG':
            src = self.tokenizer.mr(input_data)
            src.append('<eos>')
        else:
            src = self.tokenizer.text(input_data)

        translation, attention = self.translate_sentence(src, self.SRC, self.TRG, self.model, self.device)

        output_data = ''
        if mode == 'NLG':
            for i in range(len(translation)):
                if translation[i] == '<eos>':
                    break
                if (i > 0) and (translation[i] != '.') and (translation[i] != ','):
                    output_data += ' '
                output_data += translation[i]
        else:
            for i in range(len(translation)):
                if translation[i] == '<eos>':
                    break
                if i > 0:
                    output_data += '|'
                output_data += translation[i]

        return output_data
