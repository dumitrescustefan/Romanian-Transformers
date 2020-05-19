![Python](https://img.shields.io/badge/Python-3-brightgreen)
![Latest%20BERT](https://img.shields.io/badge/Latest%20BERT-v1.0-informational)

# Romanian Transformers

This repo is meant as a space to centralize Romanian Transformers.
We start small, with a pretrained BERT, and we'll add more models as resources become available. 

We're using [HuggingFace's Transformers](https://github.com/huggingface/transformers) lib, an awesome tool for NLP. What's BERT you ask? 

Here's a [clear and condensed article](https://skok.ai/2020/05/11/Top-Down-Introduction-to-BERT.html) about what BERT is and what it can do. 

## Releases

#### BERT

| Model name                    	| Task fine-tuned               	| Version/Date       	|
|-------------------------------	|--------------------------	|--------------------	|
| [bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)   	    | Standard (no fine-tuning) 	| ![v1.0](https://img.shields.io/badge/v1.0-21%20Apr%202020-ff6666) 	|
| [bert-base-romanian-uncased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1) 	| Standard (no fine-tuning) 	| ![v1.0](https://img.shields.io/badge/v1.0-21%20Apr%202020-ff6666) 	|

## How to use

Using HuggingFace's Transformers lib, instantiate a model and replace the model name name as necessary:
```python
from transformers import AutoTokenizer, AutoModel
import torch

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

# tokenize a sentence and run through the model
input_ids = torch.tensor(tokenizer.encode("Acesta este un test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

We're in the process of writing a guide with practical examples. Check back soon!

## Evaluation

Evaluation is performed on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) UPOS, XPOS and LAS, and on a NER task based on [RONEC](https://github.com/dumitrescustefan/ronec). Details, as well as more in-depth tests not shown here, are given in the dedicated [evaluation page](evaluation/README.md). 

The baseline is the [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) model ``bert-base-multilingual-(un)cased``, as currently it is the only available BERT model that works on Romanian.

| Model                          |  UPOS |  XPOS  |  NER  |  LAS  |
|--------------------------------|:-----:|:------:|:-----:|:-----:|
| bert-base-multilingual-uncased | 97.65 |  95.72 | 83.91 | 87.65 |
| bert-base-multilingual-cased   | 97.87 |  96.16 | 84.13 | 88.04 |
| bert-base-romanian-uncased-v1  | **98.18** |  **96.84** | 85.26 | 89.61 |
| bert-base-romanian-cased-v1    | 98.00 |  96.46 | **85.88** | **89.69** |

## Corpus 

The **standard** models (non-finetuned) are trained on the following corpora (stats in the table below are after cleaning):

| Corpus    	| Lines(M) 	| Words(M) 	| Chars(B) 	| Size(GB) 	|
|-----------	|:--------:	|:--------:	|:--------:	|:--------:	|
| OPUS      	|   55.05  	|  635.04  	|   4.045  	|    3.8   	|
| OSCAR     	|   33.56  	|  1725.82 	|  11.411  	|    11    	|
| Wikipedia 	|   1.54   	|   60.47  	|   0.411  	|    0.4   	|
| **Total**     	|   **90.15**  	|  **2421.33** 	|  **15.867**  	|   **15.2**   	|

## Help us out

If you like this repo, you could help out with:
- Raw Romanian texts: more corpus (usually) means better models
- Text cleaning: improve the current cleaning tools in the ``corpus`` folder
- Evaluation: we'd like to extend our evaluation on more datasets; give us a heads up if you have a dataset suitable to benchmark models on, like sentiment analysis, textual entailment, etc. 
- More models: We'd like to train larger models (e.g. RoBERTa) as well as smaller ones (ALBERT, ELECTRA, DistillBERT). Want to help?

## Contributors

[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/0)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/0)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/1)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/1)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/2)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/2)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/3)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/3)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/4)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/4)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/5)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/5)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/6)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/6)[![](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/images/7)](https://sourcerer.io/fame/avramandrei/dumitrescustefan/Romanian-Transformers/links/7)

## Acknowledgements

- We'd like to thank [Sampo Pyysalo](https://github.com/spyysalo) from TurkuNLP for helping us out with the compute needed to pretrain the v1.0 BERT models. He's awesome!
