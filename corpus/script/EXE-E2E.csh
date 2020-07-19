#! /bin/csh

#python3 conv_e2e.py ../e2e_slot ../e2e-dataset/trainset_fix.csv ../e2e_train.tsv
#python3 conv_e2e.py ../e2e_slot ../e2e-dataset/devset.csv ../e2e_valid.tsv
#python3 conv_e2e.py ../e2e_slot ../e2e-dataset/testset_w_refs.csv ../e2e_test.tsv
python3 conv_e2e.py -itrain ../e2e-dataset/trainset_fix.csv -otrain ../e2e_train.tsv -ivalid ../e2e-dataset/devset.csv -ovalid ../e2e_valid.tsv -itest ../e2e-dataset/testset_w_refs.csv -otest ../e2e_test.tsv -otrain_aug ../e2e_train_aug.tsv -omr ../e2e_mr.json
