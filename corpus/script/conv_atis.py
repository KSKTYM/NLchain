#! python

import sys
import copy
argv = sys.argv

'''
sa_slot = {
    'aircraft_code': '',
    'airline_code': '',
    'airline_name': '',
    'airport_code': '',
    'airport_name': '',
    'arrive_date.date_relative': '',
    'arrive_date.day_name': '',
    'arrive_date.day_number': '',
    'arrive_date.month_name': '',
    'arrive_date.today_relative': '',
    'arrive_time.end_time': '',
    'arrive_time.period_mod': '',
    'arrive_time.period_of_day': '',
    'arrive_time.start_time': '',
    'arrive_time.time': '',
    'arrive_time.time_relative': '',
    'city_name': '',
    'class_type': '',
    'connect': '',
    'cost_relative': '',
    'day_name': '',
    'day_number': '',
    'days_code': '',
    'depart_date.date_relative': '',
    'depart_date.day_name': '',
    'depart_date.day_number': '',
    'depart_date.month_name': '',
    'depart_date.today_relative': '',
    'depart_date.year': '',
    'depart_time.end_time': '',
    'depart_time.period_mod': '',
    'depart_time.period_of_day': '',
    'depart_time.start_time': '',
    'depart_time.time': '',
    'depart_time.time_relative': '',
    'economy': '',
    'fare_amount': '',
    'fare_basis_code': '',
    'flight_days': '',
    'flight_mod': '',
    'flight_number': '',
    'flight_stop': '',
    'flight_time': '',
    'fromloc.airport_code': '',
    'fromloc.airport_name': '',
    'fromloc.city_name': '',
    'fromloc.state_code': '',
    'fromloc.state_name': '',
    'meal': '',
    'meal_code': '',
    'meal_description': '',
    'mod': '',
    'month_name': '',
    'or': '',
    'period_of_day': '',
    'restriction_code': '',
    'return_date.date_relative': '',
    'return_date.day_name': '',
    'return_date.day_number': '',
    'return_date.month_name': '',
    'return_date.today_relative': '',
    'return_time.period_mod': '',
    'return_time.period_of_day': '',
    'round_trip': '',
    'state_code': '',
    'state_name': '',
    'stoploc.airport_name': '',
    'stoploc.city_name': '',
    'stoploc.state_code': '',
    'time': '',
    'time_relative': '',
    'today_relative': '',
    'toloc.airport_code': '',
    'toloc.airport_name': '',
    'toloc.city_name': '',
    'toloc.country_name': '',
    'toloc.state_code': '',
    'toloc.state_name': '',
    'transport_type': ''
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
