#! /bin/csh

set mode="nlg nlu"
set data="atis e2e snips"
set dir="train test"
#set dir="train"
#set dir="test"

foreach d ($data)
    foreach m ($mode)
	foreach r ($dir)
	    python3 calc_score.py -mode "$m" -data "$d" -i "$r"/result_"$d"_chain_"$m".tsv -o result/result_"$d"_chain_"$m"_"$r".tsv
	    python3 calc_score.py -mode "$m" -data "$d" -i "$r"/result_"$d"_"$m".tsv -o result/result_"$d"_"$m"_"$r".tsv
	end
    end
end
