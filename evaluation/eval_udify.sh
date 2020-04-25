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

model_basename=$(basename "$model")
vocab="$model/vocab.txt"

if curl --output /dev/null --silent --head --fail "https://s3.amazonaws.com/models.huggingface.co/bert/$model.tar.gz"; then
  printf "Model exists at the URL: 'https://s3.amazonaws.com/models.huggingface.co/bert/%s.tar.gz'. No local download is required.\n" "$model"
else
  printf "Model does not exists at the URL: 'https://s3.amazonaws.com/models.huggingface.co/bert/%s.tar.gz'. Downloading...\n" "$model"

  cd udify
  [ ! -d "pretrained_models" ] && mkdir "pretrained_models"
  cd pretrained_models

  if [ ! -e "$model_basename.tar.gz" ]
  then
    printf "\nDownloading 'bert_config.json'...\n"
    curl -o bert_config.json "https://s3.amazonaws.com/models.huggingface.co/bert/$model/config.json"

    printf "\nDownloading 'vocab.txt'...\n"
    curl -o vocab.txt "https://s3.amazonaws.com/models.huggingface.co/bert/$model/vocab.txt"

    printf "\nDownloading 'pytorch_model.bin'...\n"
    curl -o pytorch_model.bin "https://s3.amazonaws.com/models.huggingface.co/bert/$model/pytorch_model.bin"

    tar -czvf "$model_basename.tar.gz" pytorch_model.bin bert_config.json vocab.txt

    rm pytorch_model.bin bert_config.json vocab.txt
  fi

  vocab="https://s3.amazonaws.com/models.huggingface.co/bert/$model/vocab.txt"
  model="pretrained_models/$model_basename.tar.gz"
  cd ../..
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

sed -i '34s\.*\          "pretrained_model": "'"$model"'",\' "$udify_config"

if [[ $model == *"uncased"* ]]; then
  sed -i '12s/.*/        "do_lowercase": true,/' "$udify_config"
fi

sed -i '11i\        "pretrained_model": "'"$vocab"'",' "$udify_config"

if [ "$device" == "cpu" ]
then
  sed -i '83i\    "cuda_device": -1,' "$udify_config"
fi


[ -d "$save_path" ] && rm -r "$save_path"

python3 train.py --config "$udify_config" --name ro_rrt

nice_print "Evaluating Udify model on UD Romanian RRT..."

[ ! -d "../models/$model_basename" ] && mkdir -p "../models/$model_basename"
[ ! -d "../outputs/$model_basename" ] && mkdir -p "../outputs/$model_basename"

model_path="$(find $save_path -name model.tar.gz)"
cp "$model_path" "../models/$model_basename/udify_model.tar.gz"

python3 predict.py "$model_path" ../dataset-rrt/test.conllu "../outputs/$model_basename/predict.udify.conllu" --device -1

cd ..
