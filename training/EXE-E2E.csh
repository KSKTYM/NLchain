#! /bin/csh

/bin/cp ../corpus/e2e_slot ../parameter/e2e_nlu/
python3 train_transformer.py -nlu -p ../parameter/e2e_nlu -path ../corpus/ -file e2e -epoch 30 -batch 128 > RES_E2E_NLU

/bin/cp ../corpus/e2e_slot ../parameter/e2e_nlg/
python3 train_transformer.py -nlg -p ../parameter/e2e_nlg -path ../corpus/ -file e2e -epoch 30 -batch 128 > RES_E2E_NLG

/bin/cp ../corpus/e2e_slot ../parameter/e2e/
python3 train_transformer.py -p ../parameter/e2e -path ../corpus/ -file e2e -epoch 30 -batch 128 > RES_E2E_CHAIN
