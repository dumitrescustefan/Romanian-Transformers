#!/bin/bash

udify_config="config/ud/ro/udify_finetune_ro_rrt.json"
udify_original_config="../config/udify_finetune_ro_rrt.json"
save_path="logs/ro_rrt"

function nice_print {
	title=$1
	printf '\n%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

	printf "\n%*s" $((($(tput cols))/2 - 1 - (${#title})/2 + `if [ $(( $(tput cols) % 2 )) -eq 1 ]; then echo 1; else echo 0; fi`)) | tr ' ' '#'
	printf " $title "
	printf "%*s\n" $((($(tput cols))/2 - (${#title})/2 - 1)) | tr ' ' '#'

	printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'
}

if [ -z "$1" ]; then
	echo "Please specify a model path or a model name from HuggingFace's repositoy."
	exit 1
else
	model=$1
fi

if [ -n "$2" ]; then
	device=$2
else
	device="cuda"
fi

nice_print "Training Udify model on UD Romanian RRT..."

cd udify

[ -d "$udify_config" ] && rm "$udify_config"
cp "$udify_original_config" "$udify_config"

sed -i '11i\        "pretrained_model": "'"$model"'",' "$udify_config"

if [ "$device" == "cpu" ]
then
  sed -i '83i\    "cuda_device": -1,' "$udify_config"
fi

if [[ $model == *"uncased"* ]]; then
  sed -i '12s/.*/        "do_lowercase": true,/' "$udify_config"
fi

[ -d "$save_path" ] && rm -r "$save_path"

python3 train.py --config "$udify_config" --name ro_rrt

nice_print "Evaluating Udify model on UD Romanian RRT..."

[ ! -d "../models/$model" ] && mkdir -p "../models/$model"
[ ! -d "../outputs/$model" ] && mkdir -p "../outputs/$model"

model_path="$(find $save_path -name model.tar.gz)"
cp "$model_path" "../models/$model/udify-model.tar.gz"

python3 predict.py "$model_path" ../dataset-rrt/test.conllu "../outputs/$model/predict.udify.conllu" --device -1

cd ..
