#! /bin/csh

/bin/cp ../corpus/atis_slot ../parameter/atis_nlu/
python3 train_transformer.py -nlu -p ../parameter/atis_nlu -path ../corpus/ -file atis -epoch 30 -batch 128 > RES_ATIS_NLU

/bin/cp ../corpus/atis_slot ../parameter/atis_nlg/
python3 train_transformer.py -nlg -p ../parameter/atis_nlg -path ../corpus/ -file atis -epoch 30 -batch 128 > RES_ATIS_NLG

/bin/cp ../corpus/atis_slot ../parameter/atis/
python3 train_transformer.py -p ../parameter/atis -path ../corpus/ -file atis -epoch 30 -batch 128 > RES_ATIS_CHAIN
