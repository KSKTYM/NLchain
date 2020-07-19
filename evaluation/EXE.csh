#! /bin/csh

set mode="nlg nlu"
set data="e2e"
set dir="test train"

foreach d ($data)
    foreach r ($dir)
	foreach m ($mode)
	    python3 m_eval.py -mode "$m" -p ../parameter/"$d"_chain/ -result "$r"/result_"$d"_chain_"$m".tsv -data ../corpus/"$d"_"$r".tsv > "$r"/RES_"$d"_chain_"$m".txt
	    python3 m_eval.py -mode "$m" -p ../parameter/"$d"_"$m"/ -result "$r"/result_"$d"_"$m".tsv -data ../corpus/"$d"_"$r".tsv > "$r"/RES_"$d"_"$m".txt

	    python3 calc_score.py -mode "$m" -data "$d" -i "$r"/result_"$d"_chain_"$m".tsv -o result/result_"$d"_chain_"$m"_"$r".tsv
	    python3 calc_score.py -mode "$m" -data "$d" -i "$r"/result_"$d"_"$m".tsv -o result/result_"$d"_"$m"_"$r".tsv
        end
    end
end
