#!/bin/bash

function nice_print {
	title=$1
	printf '\n%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

	printf "\n%*s" $((($(tput cols))/2 - 1 - (${#title})/2 + `if [ $(( $(tput cols) % 2 )) -eq 1 ]; then echo 1; else echo 0; fi`)) | tr ' ' '#'
	printf " $title "
	printf "%*s\n" $((($(tput cols))/2 - (${#title})/2 - 1)) | tr ' ' '#'

	printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'
}

if [ -z "$1" ]; then
	echo "Please specify a model path/model name from HuggingFace's repositoy."
	exit 1
else
	model=$1
fi

if [ -n "$2" ]; then
	device=$2
else
	if [ -n which nvcc ]
	then
		device="cuda"
	else
		printf "\nWARNING: Nvidia driver was not automatically detected. The training will start on cpu. If you are sure you have cuda installed, please specify the specific driver as the second argument to the script.\n"
		device="cpu"
	fi
fi

model_frozen_dir="models/$model/ronec_frozen"
model_dir="models/$model/ronec"

nice_print "Training model on RONEC..."

printf "Model: %s\n" "$1"
printf "Save path: %s\n" "$model_frozen_dir"
printf "Frozen: True\n"
printf "Device: %s\n" $device

[ ! -d "$model_frozen_dir" ] && mkdir -p "$model_frozen_dir"

python3 tools/train.py dataset-ronec/train.conllu dataset-ronec/dev.conllu 10 --save_path "$model_frozen_dir" --lang_model_name "$model" --device $device

printf "\nFinished.\n"

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

printf "Model: %s\n" "$1"
printf "Save path: %s\n" "$model_dir"
printf "Frozen: False\n"
printf "Device: %s\n" $device

[ ! -d "$model_dir" ] && mkdir -p "$model_dir"

python3 tools/train.py dataset-ronec/train.conllu dataset-ronec/dev.conllu 10 --save_path "$model_dir" --lang_model_name "$model" --device $device --fine_tune --epochs 3 --learning_rate 2e-5

printf "\nFinished.\n"

nice_print "Evaluating model on RONEC..."

printf "Model: %s\n" "$1"
printf "Load path: %s\n" "$model_frozen_dir"
printf "Device: %s\n\n" $device

python3 tools/predict.py dataset-ronec/test.conllu "$model_frozen_dir" 10 --lang_model_name "$model" --output_path "output/$model/predict_ronec_frozen.conllu" --device $device
python3 tools/ner_eval.py dataset-ronec/test.conllu output/predict_ronec_frozen.conllu

printf "\nFinished.\n"

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

printf "Model: %s\n" "$1"
printf "Load path: %s\n" "$model_dir"
printf "Device: %s\n\n" $device

python3 tools/predict.py dataset-ronec/test.conllu "$model_dir" 10 --lang_model_name "$model" --output_path "output/$model/predict_ronec.conllu" --device $device
python3 tools/ner_eval.py dataset-ronec/test.conllu output/predict_ronec.conllu

printf "\nFinished.\n"

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

