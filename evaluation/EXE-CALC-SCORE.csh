#! /bin/csh

set dir="atis e2e snips"
foreach d ($dir)
    python3 calc_score.py -nlu -i ../parameter/"$d"/test_nlu.tsv -o result_"$d"_chain_nlu.tsv
    python3 calc_score.py -nlg -i ../parameter/"$d"/test_nlg.tsv -o result_"$d"_chain_nlg.tsv

    python3 calc_score.py -nlu -i ../parameter/"$d"_nlu/test_nlu.tsv -o result_"$d"_nlu.tsv
    python3 calc_score.py -nlg -i ../parameter/"$d"_nlg/test_nlg.tsv -o result_"$d"_nlg.tsv
end
