#! python

import sys
argv = sys.argv

if __name__ == '__main__':
    fi = open(argv[1], 'r', encoding='utf-8')
    a_input = fi.readlines()
    fi.close()
    fo = open(argv[2], 'w', encoding='utf-8')

    a_attribute = []
    for i in range(1, len(a_input)):
        a_data = a_input[i].rstrip('\n').split('\",')
        mr = a_data[0].lstrip('\"').rstrip('\"')
        a_mr = mr.split(',')
        for j in range(len(a_mr)):
            if '[' in a_mr[j]:
                attribute = a_mr[j].split('[')[0].rstrip(' ').lstrip(' ')
                if (attribute != '') and ((attribute in a_attribute) is False):
                    a_attribute.append(attribute)
    a_attribute.sort()
    for i in range(len(a_attribute)):
        fo.write(a_attribute[i]+'\n')
    fo.close()
