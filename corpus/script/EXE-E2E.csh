#! /bin/csh

python3 conv_e2e.py ../e2e_slot ../e2e-dataset/trainset_fix.csv ../e2e_train.tsv
python3 conv_e2e.py ../e2e_slot ../e2e-dataset/devset.csv ../e2e_valid.tsv
python3 conv_e2e.py ../e2e_slot ../e2e-dataset/testset_w_refs.csv ../e2e_test.tsv
