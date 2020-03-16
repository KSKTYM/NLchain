#! /bin/csh

# E2E
python3 m_eval -mode NLG -p ../parameter/e2e/ -data ../corpus/e2e_test.tsv > RES-E2E-NLG.txt
python3 m_eval -mode NLU -p ../parameter/e2e/ -data ../corpus/e2e_test.tsv > RES-E2E-NLU.txt

# ATIS
python3 m_eval -mode NLG -p ../parameter/atis/ -data ../corpus/atis_test.tsv > RES-ATIS-NLG.txt
python3 m_eval -mode NLU -p ../parameter/atis/ -data ../corpus/atis_test.tsv > RES-ATIS-NLU.txt

# SNIPS
python3 m_eval -mode NLG -p ../parameter/snips/ -data ../corpus/snips_test.tsv > RES-SNIPS-NLG.txt
python3 m_eval -mode NLU -p ../parameter/snips/ -data ../corpus/snips_test.tsv > RES-SNIPS-NLU.txt
