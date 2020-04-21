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
	device="cuda"
fi

model_dirs=("models/$model/rrt_upos_frozen" "models/$model/rrt_xpos_frozen" "models/$model/rrt_upos" "models/$model/rrt_xpos")
output_paths=("output/$model/predict_rrt_upos_frozen.conllu" "output/$model/predict_rrt_xpos_frozen.conllu" "output/$model/predict_rrt_upos.conllu" "output/$model/predict_rrt_xpos.conllu")
goals=("UPOS" "XPOS" "UPOS" "XPOS")
predict_cols=(3 4 3 4)
frozen=(true true false false)

nice_print "Training model on UD Romanian RRT..."

for i in {0..3}
do
  printf "Model: %s\n" "$1"
  printf "Save path: %s\n" "${model_dirs[i]}"
  printf "Frozen: %s\n" "${frozen[i]}"
  printf "Device: %s\n" $device
  printf "Training goal: %s\n\n" "${goals[i]}"


  [ ! -d "${model_dirs[i]}" ] && mkdir -p "${model_dirs[i]}"

  if [ ! "${frozen[i]}" ]; then
    fine_tune="--fine_tune"
    epochs=3
    learning_rate=2e-5
  else
    fine_tune=""
    epochs=100
    learning_rate=2e-4
  fi

  python3 tools/train.py dataset-rrt/train.conllu dataset-rrt/dev.conllu "${predict_cols[i]}" --save_path "${model_dirs[i]}" --lang_model_name "$model" --device $device $fine_tune --epochs $epochs --learning_rate $learning_rate

  printf "\nFinished.\n"

  printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'
done

nice_print "Evaluating model on UD Romanian RRT..."

[ ! -d "output/$model" ] && mkdir "output/$model"

for i in {0..3}
do
	printf "Model: %s\n" "$1"
 	printf "Load path: %s\n" "${model_dirs[i]}"
	printf "Device: %s\n" $device
	printf "Evaluation goal: %s\n\n" "${goals[i]}"

	python3 tools/predict.py dataset-rrt/test.conllu "${model_dirs[i]}" "${predict_cols[i]}" --lang_model_name "$model" --output_path "${output_paths[i]}" --device $device

	output=$(python3 tools/ud_eval.py dataset-rrt/test.conllu "${output_paths[i]}"  --verbose)
	echo "$output" | head -n 2
	echo "$output" | sed "$((${predict_cols[i]} + 3))q;d"

	printf "\nFinished.\n"

	printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'
done

printf "Concatenated Frozen Evaluation\n\n"

python3 tools/ud_unite.py --frozen
python3 tools/ud_eval.py dataset-rrt/test.conllu output/$model/predict_all_frozen.conllu  --verbose

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

printf "Concatenated Non-Frozen Evaluation\n\n"

python3 tools/ud_unite.py
python3 tools/ud_eval.py dataset-rrt/test.conllu output/$model/predict_all.conllu  --verbose

printf '\n%*s\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' '#'

