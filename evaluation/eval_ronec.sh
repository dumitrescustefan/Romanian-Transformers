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

device="cuda"

if [ -n "$2" ]; then
  if  [[ "$2" =~ ^[0-9]+$ ]]
  then
	  iterations=$2
	else
	  device="$2"
	  iterations=1
	fi
else
	iterations=1
fi

if [ -n "$3" ]; then
	device="$3"
fi

model_basename=$(basename $model)
model_frozen_dir="models/$model_basename/ronec_frozen"
model_dir="models/$model_basename/ronec"

nice_print "Training model on RONEC..."

printf "Model: %s\n" "$1"
printf "Save path: %s\n" "$model_frozen_dir"
printf "Frozen: True\n"
printf "Device: %s\n\n" $device

[ ! -d "$model_frozen_dir" ] && mkdir -p "$model_frozen_dir"

python3 tools/train.py dataset-ronec/train.conllu dataset-ronec/dev.conllu 10 --save_path "$model_frozen_dir" --lang_model_name "$model" --device $device --iterations "$iterations" --remove_o_label

printf "\nFinished.\n"

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

printf "Model: %s\n" "$1"
printf "Save path: %s\n" "$model_dir"
printf "Frozen: False\n"
printf "Device: %s\n" $device

[ ! -d "$model_dir" ] && mkdir -p "$model_dir"

python3 tools/train.py dataset-ronec/train.conllu dataset-ronec/dev.conllu 10 --save_path "$model_dir" --lang_model_name "$model" --device $device --fine_tune --epochs 10 --learning_rate 2e-5 --iterations "$iterations" --batch_size 16 --remove_o_label

printf "\nFinished.\n"

nice_print "Evaluating model on RONEC..."

[ ! -d "outputs/$model_basename" ] && mkdir -p "outputs/$model_basename"
[ ! -d "results/$model_basename" ] && mkdir -p "results/$model_basename"

printf "Model: %s\n" "$1"
printf "Load path: %s\n" "$model_frozen_dir"
printf "Device: %s\n\n" $device

python3 tools/predict.py dataset-ronec/test.conllu "$model_frozen_dir" 10 --lang_model_name "$model" --output_path "outputs/$model_basename/predict_ronec_frozen.conllu" --device $device --iterations "$iterations"
output=$(python3 tools/ner_eval.py dataset-ronec/test.conllu "outputs/$model_basename/predict_ronec_frozen.conllu" --iterations "$iterations")
echo "$output"
echo "$output" > "results/$model_basename/ronec_frozen.txt"

printf "\nFinished.\n"

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

printf "Model: %s\n" "$1"
printf "Load path: %s\n" "$model_dir"
printf "Device: %s\n\n" $device

python3 tools/predict.py dataset-ronec/test.conllu "$model_dir" 10 --lang_model_name "$model" --output_path "outputs/$model_basename/predict_ronec.conllu" --device $device --iterations "$iterations"
output=$(python3 tools/ner_eval.py dataset-ronec/test.conllu "outputs/$model_basename/predict_ronec.conllu" --iterations "$iterations")
echo "$output"
echo "$output" > "results/$model_basename/ronec.txt"

printf "\nFinished.\n"

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'


