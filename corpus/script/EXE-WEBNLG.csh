#! /bin/csh

set name = "train dev"

# 1triples
set nameAA = "Airport Artist Astronaut Athlete Building CelestialBody City ComicsCharacter Company Food MeanOfTransportation Monument Politician SportsTeam University WrittenWork"
set nameA = "Airport Artist Astronaut Athlete Building CelestialBody City ComicsCharacter Company Food MeanOfTransportation Monument Politician SportsTeam University WrittenWork"

# 2-5 triples
set nameB = "Airport Artist Astronaut Athlete Building CelestialBody City ComicsCharacter Company Food MeanOfTransportation Monument Politician SportsTeam University WrittenWork"

# 6-7 triples
set nameC = "Astronaut Company Monument University"

foreach c ($name)
    # 1triples
    #python3 conv_webnlg.py -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/1triples/Airport_allSolutions.xml -json ../webnlg/"$c"/1triples/Airport.json
    #/bin/cp -pv tmp_new.tsv tmp_old.tsv
    foreach attr ($nameAA)
	#python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/1triples/"$attr"_allSolutions.xml -json ../webnlg/"$c"/1triples/"$attr".json
	#/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end

    # 2triples
    python3 conv_webnlg.py -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/2triples/Airport.xml -json ../webnlg/"$c"/2triples/Airport.json
    /bin/cp -pv tmp_new.tsv tmp_old.tsv
    #foreach attr ($nameA)
    foreach attr ($nameAA)
	python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/2triples/"$attr".xml -json ../webnlg/"$c"/2triples/"$attr".json
	/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end

    # 3triples
    #python3 conv_webnlg.py -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/3triples/Airport.xml -json ../webnlg/"$c"/3triples/Airport.json
    #/bin/cp -pv tmp_new.tsv tmp_old.tsv
    #foreach attr ($nameA)
    foreach attr ($nameAA)
	#python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/3triples/"$attr".xml -json ../webnlg/"$c"/3triples/"$attr".json
	#/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end

    # 4triples
    foreach attr ($nameA)
	#python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/4triples/"$attr".xml -json ../webnlg/"$c"/4triples/"$attr".json
	#/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end

    # 5triples
    foreach attr ($nameA)
	#python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/5triples/"$attr".xml -json ../webnlg/"$c"/5triples/"$attr".json
	#/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end

    # 6triples
    foreach attr ($nameC)
	#python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/6triples/"$attr".xml -json ../webnlg/"$c"/6triples/"$attr".json
	#/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end

    # 7triples
    foreach attr ($nameC)
	#python3 conv_webnlg.py -old tmp_old.tsv -new tmp_new.tsv -xml ../webnlg/challenge2020_train_dev/en/"$c"/7triples/"$attr".xml -json ../webnlg/"$c"/7triples/"$attr".json
	#/bin/cp -pv tmp_new.tsv tmp_old.tsv
    end
    /bin/cp -pv tmp_new.tsv ../webnlg_"$c".tsv
end
/bin/mv ../webnlg_dev.tsv ../webnlg_valid.tsv
