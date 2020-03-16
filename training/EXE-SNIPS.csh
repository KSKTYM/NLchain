#! /bin/csh

/bin/cp ../corpus/snips_slot ../parameter/snips/
python3 train_transformer.py -p ../parameter/snips -path ../corpus/ -file snips -epoch 30 -batch 128
