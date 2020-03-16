#! /bin/csh

/bin/cp ../corpus/e2e_slot ../parameter/e2e/
python3 train_transformer.py -p ../parameter/e2e -path ../corpus/ -file e2e -epoch 30 -batch 128
