#! /bin/csh

set mode="nlg nlu"
set data="atis e2e snips"
set dir="test train"

foreach d ($data)
    foreach m ($mode)
	foreach r ($dir)
	    python3 m_eval.py -mode "$m" -p ../parameter/"$d"/ -result "$r"/result_"$d"_chain_"$m".tsv -data ../corpus/"$d"_test.tsv > "$r"/RES_"$d"_chain_"$m".txt
	    python3 m_eval.py -mode "$m" -p ../parameter/"$d"_"$m"/ -result "$r"/result_"$d"_"$m".tsv -data ../corpus/"$d"_test.tsv > "$r"/RES_"$d"_"$m".txt
        end
    end
end
