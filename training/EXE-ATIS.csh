#! /bin/csh

/bin/cp ../corpus/atis_slot ../parameter/atis/
python3 train_transformer.py -p ../parameter/atis -path ../corpus/ -file atis -epoch 30 -batch 128
