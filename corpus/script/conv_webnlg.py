#! python

import sys
import copy
import json
import argparse

s_data = {
    'text': [],
    'mr': {
        'category': '',
        'shape': '',
        'shape_type': '',
        'size': 1,
        'triple': []
    }
}

a_mr_param = ['category', 'eid', 'shape', 'shape_type', 'size']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-xml', help='input xml file')
    parser.add_argument('-json', help='json file')
    parser.add_argument('-old', help='old file')
    parser.add_argument('-new', help='new file')
    args = parser.parse_args()

    if args.xml is None:
        print('xml file should be set')
        sys.exit()
    if args.new is None:
        print('output file should be set')
        sys.exit()

    # input xml file
    f_xml = open(args.xml, 'r', encoding='utf-8')
    a_input = f_xml.readlines()
    f_xml.close()

    # parse xml file
    a_data = []
    for i in range(len(a_input)):
        input_line = a_input[i].lstrip(' ').rstrip('\n')
        if input_line.startswith('<entry'):
            data = copy.deepcopy(s_data)
            a_entry = input_line.lstrip('<entry ').rstrip('>').split('\" ')
            for j in range(len(a_entry)):
                attribute = a_entry[j].split('=')[0]
                value = a_entry[j].split('=')[1].replace('\"', '')
                for mr_param in a_mr_param:
                    if mr_param == attribute:
                        data['mr'][mr_param] = value.replace('\'\'', '').replace('\"', '').rstrip(' ')
                        if mr_param == 'size':
                            data['mr'][mr_param] = int(value)
        elif input_line.startswith('<modifiedtripleset>'):
            for j in range(i+1, i+1+data['mr']['size']):
                a_triple = a_input[j].rstrip('\n').lstrip(' ').replace('<mtriple>', '').replace('</mtriple>','').split(' | ')
                for k in range(len(a_triple)):
                    a_triple[k] = a_triple[k].replace('_', ' ').replace('\'\'', '').replace('\"', '').rstrip(' ')
                data['mr']['triple'].append(a_triple)
        elif input_line.startswith('<lex'):
            a_lex = input_line.split('\">')[0].split(' ')
            text = input_line.split('\">')[1].split('<')[0]

            comment = ''
            for j in range(len(a_lex)):
                if a_lex[j].startswith('comment'):
                    comment = a_lex[j].split('=')[1].replace('\"', '')
                    break
            if comment == 'good':
                data['text'].append(text)
        elif input_line.startswith('</entry>'):
            a_data.append(data)
            del data

    # dump json
    if args.json is not None:
        f_json = open(args.json, 'w', encoding='utf-8')
        json.dump(a_data, f_json, ensure_ascii=False, indent=4, sort_keys=True)
        f_json.close()

    f_new = open(args.new, 'w', encoding='utf-8')
    if args.old is not None:
        f_old = open(args.old, 'r', encoding='utf-8')
        a_old = f_old.readlines()
        f_old.close()
        for i in range(len(a_old)):
            f_new.write(a_old[i].rstrip('\n')+'\n')

    for data in a_data:
        out_mr = data['mr']['category']
        out_mr += '|'
        out_mr += data['mr']['shape']
        out_mr += '|'
        out_mr += data['mr']['shape_type']
        out_mr += '|'
        out_mr += str(data['mr']['size'])
        for j in range(len(data['mr']['triple'])):
            for k in range(len(data['mr']['triple'][j])):
                out_mr += '|'
                out_mr += data['mr']['triple'][j][k]
            #if j < len(data['mr']['triple']) - 1:
                #out_mr += '|<sep>'
        for j in range(len(data['text'])):
            f_new.write(out_mr+'\t'+data['text'][j]+'\n')
    f_new.close()
