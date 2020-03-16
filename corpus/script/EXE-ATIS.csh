#! /bin/csh

python3 conv_atis.py ../atis_slot ../atis/train/label ../atis/train/seq.in ../atis/train/seq.out ../atis_train.tsv
python3 conv_atis.py ../atis_slot ../atis/valid/label ../atis/valid/seq.in ../atis/valid/seq.out ../atis_valid.tsv
python3 conv_atis.py ../atis_slot ../atis/test/label ../atis/test/seq.in ../atis/test/seq.out ../atis_test.tsv
