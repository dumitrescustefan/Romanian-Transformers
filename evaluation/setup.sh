#!/bin/bash

udifydir="udify"
rrtdir="dataset-rrt"
ronecdir="dataset-ronec"

function nice_print {
	title=$1
	printf '\n%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

	printf "\n%*s" $((($(tput cols))/2 - 1 - (${#title})/2 + `if [ $(( $(tput cols) % 2 )) -eq 1 ]; then echo 1; else echo 0; fi`)) | tr ' ' '#' 
	printf " $title "
	printf "%*s\n" $((($(tput cols))/2 - (${#title})/2 - 1)) | tr ' ' '#'

	printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'
}

nice_print "Instaling Udify..."

if [ ! -d $udifydir ]; then
	git clone https://github.com/Hyperparticle/udify.git
	cd udify
	pip3 install -r requirements.txt

	mkdir -p config/ud/ro/
	cp ../config/udify_finetune_ro_rrt.json config/ud/ro/udify_finetune_ro_rrt.json

	cd ..
else
 	printf "Udify already installed.\n"
fi

nice_print "Downloading UD Romanian RRT..."

if [ ! -d $rrtdir ]; then 
	mkdir $rrtdir
	cd $rrtdir
	printf "Download train.conllu...\n"
	curl -o train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-train.conllu
	printf "\nDownload dev.conllu...\n"
	curl -o dev.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-dev.conllu
	printf "\nDownload test.conllu...\n"
	curl -o test.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-test.conllu
	cd ..
else
	echo "UD Romanian RRT already downloaded."
fi

nice_print "Downloading RONEC..."

if [ ! -d $ronecdir ]; then
	mkdir $ronecdir
	cd $ronecdir
	printf "\nDownloading train.conllu...\n"
	curl -o train.conllu https://raw.githubusercontent.com/dumitrescustefan/ronec/master/ronec/conllup/raw/train.conllu
	printf "\nDownloading dev.conllu...\n"
	curl -o dev.conllu https://raw.githubusercontent.com/dumitrescustefan/ronec/master/ronec/conllup/raw/dev.conllu
	printf "\nDownloading test.conllu...\n"
	curl -o test.conllu https://raw.githubusercontent.com/dumitrescustefan/ronec/master/ronec/conllup/raw/test.conllu
	cd ..
else
	echo "RONEC already downloaded."
fi

echo ""
