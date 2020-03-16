#! /bin/csh

python3 conv_snips.py ../snips_slot ../snips/train/label ../snips/train/seq.in ../snips/train/seq.out ../snips_train.tsv
python3 conv_snips.py ../snips_slot ../snips/valid/label ../snips/valid/seq.in ../snips/valid/seq.out ../snips_valid.tsv
python3 conv_snips.py ../snips_slot ../snips/test/label ../snips/test/seq.in ../snips/test/seq.out ../snips_test.tsv
