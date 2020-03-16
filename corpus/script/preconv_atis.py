#! python

import sys
import json
argv = sys.argv

if __name__ == '__main__':
    # intent (label)
    fi_intent = open(argv[1], 'r', encoding='utf-8')
    a_input = fi_intent.readlines()
    fi_intent.close()
    a_intent = []
    for i in range(len(a_input)):
        a_label = a_input[i].rstrip('\n').split('#')
        for j in range(len(a_label)):
            if (a_label[j] != '') and ((a_label[j] in a_intent) is False):
                a_intent.append(a_label[j])
    fo_intent = open(argv[2], 'w', encoding='utf-8')
    a_intent.sort()
    for i in range(len(a_intent)):
        fo_intent.write(a_intent[i]+'\n')
    fo_intent.close()

    # slot (seq.out)
    fi_slot = open(argv[3], 'r', encoding='utf-8')
    a_input = fi_slot.readlines()
    fi_slot.close()

    a_attribute = []
    for i in range(len(a_input)):
        a_data = a_input[i].rstrip('\n').split(' ')
        for j in range(len(a_data)):
            if a_data[j].startswith('B-'):
                attribute = a_data[j].lstrip('B-')
            elif a_data[j].startswith('I-'):
                attribute = a_data[j].lstrip('I-')
            else:
                attribute = ''
            if (attribute != '') and ((attribute in a_attribute) is False):
                a_attribute.append(attribute)

    a_attribute.sort()
    fo_slot = open(argv[4], 'w', encoding='utf-8')
    for i in range(len(a_attribute)):
        fo_slot.write(a_attribute[i]+'\n')
    fo_slot.close()
