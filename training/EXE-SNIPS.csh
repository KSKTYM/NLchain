#! /bin/csh

/bin/cp ../corpus/snips_slot ../parameter/snips_nlu/
python3 train_transformer.py -nlu -p ../parameter/snips_nlu -path ../corpus/ -file snips -epoch 30 -batch 128 > RES_SNIPS_NLU

/bin/cp ../corpus/snips_slot ../parameter/snips_nlg/
python3 train_transformer.py -nlg -p ../parameter/snips_nlg -path ../corpus/ -file snips -epoch 30 -batch 128 > RES_SNIPS_NLG

/bin/cp ../corpus/snips_slot ../parameter/snips/
python3 train_transformer.py -p ../parameter/snips -path ../corpus/ -file snips -epoch 30 -batch 128 > RES_SNIPS_CHAIN

