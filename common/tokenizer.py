#! python

import spacy
spacy_en = spacy.load('en')

class Tokenizer():
    def __init__(self):
        return

    def mr(self, input_text):
        a_token = []
        a_data = input_text.split('|')
        for i in range(len(a_data)):
            a_token.append(a_data[i])
        return a_token

    def text(self, input_text):
        return [tok.text for tok in spacy_en.tokenizer(input_text)]
