#! python

import sys
import copy
argv = sys.argv

'''
sa_slot = {
    'album': '',
    'artist': '',
    'best_rating': '',
    'city': '',
    'condition_description': '',
    'condition_temperature': '',
    'country': '',
    'cuisine': '',
    'current_location': '',
    'entity_name': '',
    'facility': '',
    'genre': '',
    'geographic_poi': '',
    'location_name': '',
    'movie_name': '',
    'movie_type': '',
    'music_item': '',
    'object_location_type': '',
    'object_name': '',
    'object_part_of_series_type': '',
    'object_select': '',
    'object_type': '',
    'party_size_description': '',
    'party_size_number': '',
    'playlist': '',
    'playlist_owner': '',
    'poi': '',
    'rating_unit': '',
    'rating_value': '',
    'restaurant_name': '',
    'restaurant_type': '',
    'served_dish': '',
    'service': '',
    'sort': '',
    'spatial_relation': '',
    'state': '',
    'timeRange': '',
    'track': '',
    'year': ''
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

    fi_label = open(argv[2], 'r', encoding='utf-8')
    a_input_intent = fi_label.readlines()
    fi_label.close()

    fi_seqin = open(argv[3], 'r', encoding='utf-8')
    a_input_text = fi_seqin.readlines()
    fi_seqin.close()

    fi_seqout= open(argv[4], 'r', encoding='utf-8')
    a_input_mr = fi_seqout.readlines()
    fi_seqout.close()

    fo = open(argv[5], 'w', encoding='utf-8')

    for i in range(len(a_input_intent)):
        a_slot = copy.deepcopy(sa_slot)
        a_intent = a_input_intent[i].rstrip('\n').split('#')
        a_attribute = a_input_mr[i].rstrip('\n').split(' ')
        a_input_text[i] = a_input_text[i].rstrip(' ')
        a_input_text[i] = a_input_text[i].replace('  ', ' ')
        a_value = a_input_text[i].rstrip('\n').split(' ')
        for j in range(len(a_attribute)):
            attribute = ''
            if a_attribute[j].startswith('B-'):
                attribute = a_attribute[j].lstrip('B-')
                a_slot[attribute] = a_value[j]
            elif a_attribute[j].startswith('I-'):
                attribute = a_attribute[j].lstrip('I-')
                a_slot[attribute] += ' ' + a_value[j]

        for j in range(len(a_intent)):
            fo.write(a_intent[j])
            for attribute in a_slot:
                fo.write('|')
                fo.write(a_slot[attribute])
            fo.write('\t')
            fo.write(a_input_text[i].rstrip('\n')+'\n')
        del a_slot
    fo.close()
