#! /bin/csh

## new (2020/7/14)

# NLU only
#python3 train_transformer.py -mode nlu -p ../parameter/e2e_nlu -path ../corpus/ -file e2e -epoch 20 -batch 128 -eval > RES_E2E_NLU
python3 train_transformer.py -mode nlu -p ../parameter/e2e_nlu -path ../corpus/ -file e2e -epoch 20 -batch 128 > RES_E2E_NLU

# NLG only
#python3 train_transformer.py -mode nlg -p ../parameter/e2e_nlg -path ../corpus/ -file e2e -epoch 20 -batch 128 -eval > RES_E2E_NLG
python3 train_transformer.py -mode nlg -p ../parameter/e2e_nlg -path ../corpus/ -file e2e -epoch 20 -batch 128 > RES_E2E_NLG

# NL chain
/bin/cp ../parameter/e2e_nlg/best_model_nlg.pkl ../parameter/e2e_chain/
/bin/cp ../parameter/e2e_nlu/best_model_nlu.pkl ../parameter/e2e_chain/
python3 train_transformer.py -mode chain -p ../parameter/e2e_chain -path ../corpus/ -file e2e -epoch 20 -batch 128 > RES_E2E_CHAIN
