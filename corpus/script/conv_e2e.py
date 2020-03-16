#! python

import sys
import json
import copy
argv = sys.argv

'''
sa_slot = {
    'area': '',
    'customer rating': '',
    'eatType': '',
    'familyFriendly': '',
    'food': '',
    'name': '',
    'near': '',
    'priceRange': ''
}
'''
if __name__ == '__main__':
    fi_slot = open(argv[1], 'r', encoding='utf-8')
    a_input_slot = fi_slot.readlines()
    fi_slot.close()
    sa_slot = {}
    for i in range(len(a_input_slot)):
        if (a_input_slot[i].rstrip('\n') in sa_slot) is False:
            sa_slot[a_input_slot[i].rstrip('\n')] = ''

    fi = open(argv[2], 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    fo = open(argv[3], 'w', encoding='utf-8')

    for i in range(1, len(a_input)):
        a_slot = copy.deepcopy(sa_slot)
        a_data = a_input[i].rstrip('\n').split('\",')
        mr = a_data[0].lstrip('\"').rstrip('\"')
        text = a_data[1].lstrip('\"').rstrip('\"')
        a_mr = mr.split(',')
        for j in range(len(a_mr)):
            if '[' in a_mr[j]:
                attribute = a_mr[j].split('[')[0].rstrip(' ').lstrip(' ')
                value = a_mr[j].split('[')[1].rstrip(']')
                a_slot[attribute] = value

        j = 0
        for attribute in a_slot:
            if j > 0:
                fo.write('|')
            fo.write(a_slot[attribute])
            j += 1
        fo.write('\t')
        fo.write(text+'\n')
        del a_slot
    fo.close()
