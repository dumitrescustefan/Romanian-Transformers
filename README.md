# Romanian Transformers

This repo is meant as a space to centralize Romanian Transformers and to provide a uniform evaluation. Contributions are welcome.

We're using [HuggingFace's Transformers](https://github.com/huggingface/transformers) lib, an awesome tool for NLP. What's BERT you ask?  Here's a [clear and condensed article](https://skok.ai/2020/05/11/Top-Down-Introduction-to-BERT.html) about what BERT is and what it can do. Also check out this summary of [different transformer models](https://huggingface.co/transformers/summary.html).

What follows is the list of Romanian transformer models, both masked and conditional language models.

Feel free to open an issue and add your model/eval here!

## Masked Language Models (MLMs)

| **Model**                                                                                                               	| **Type**   	| **Size** 	| **Article/Citation/Source**                                                                	 | **Pre-trained / Fine-tuned**                                                                                                                                                                 	   | **Release Date** 	|
|-------------------------------------------------------------------------------------------------------------------------	|------------	|----------	|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------	|
| [dumitrescustefan/bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)     	| BERT       	| 124M     	| [PDF](https://arxiv.org/abs/2009.08712) / [Cite](https://aclanthology.org/2020.findings-emnlp.387/)                                                   	 | Pre-trained                                                                                                                                                                                    	 | Apr, 2020        	|
| [dumitrescustefan/bert-base-romanian-uncased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1) 	| BERT       	| 124M     	| [PDF](https://arxiv.org/abs/2009.08712)  / [Cite](https://aclanthology.org/2020.findings-emnlp.387/)                                                   	 | Pre-trained	                                                                                                                                                                                     | Apr, 2020        	|
| [racai/distillbert-base-romanian-cased](https://huggingface.co/racai/distilbert-base-romanian-cased)                    	| DistilBERT 	| 81M      	| -                                                                                        	   | Pre-trained	                                                                                                                                                                                     | Apr, 2021        	|
| [readerbench/RoBERT-small](https://huggingface.co/readerbench/RoBERT-small)                                             	| BERT       	| 19M      	| [PDF](https://www.aclweb.org/anthology/2020.coling-main.581/)                              	 | 	Pre-trained                                                                                                                                                                                     | May, 2021        	|
| [readerbench/RoBERT-base](https://huggingface.co/readerbench/RoBERT-base)                                               	| BERT       	| 114M     	| [PDF](https://www.aclweb.org/anthology/2020.coling-main.581/)                              	 | 	Pre-trained                                                                                                                                                                                     | May, 2021        	|
| [readerbench/RoBERT-large](https://huggingface.co/readerbench/RoBERT-large)                                             	| BERT       	| 341M     	| [PDF](https://www.aclweb.org/anthology/2020.coling-main.581/)                              	 | 	Pre-trained                                                                                                                                                                                     | May, 2021        	|
| [dumitrescustefan/bert-base-romanian-ner](https://huggingface.co/dumitrescustefan/bert-base-romanian-ner)               	| BERT       	| 124M     	| [HF Space](https://huggingface.co/spaces/dumitrescustefan/NamedEntityRecognition-Romanian) 	 | Named Entity Recognition on [RONECv2](https://github.com/dumitrescustefan/ronec)                                                                                                                                                 	        | Jan, 2022        	|
| [snisioi/bert-legal-romanian-cased-v1](https://huggingface.co/snisioi/bert-legal-romanian-cased-v1)                     	| BERT       	| 124M     	| -                                                                                        	   | Legal documents on [MARCELLv2](https://elrc-share.eu/repository/browse/marcell-romanian-legislative-subcorpus-v2/2da548428b9d11eb9c1a00155d026706ce94a6b59ffc4b0e9fb5cd9cebe6889e/) 	            | Jan, 2022        	|
| [readerbench/jurBERT-base](https://huggingface.co/readerbench/jurBERT-base)                                             	| BERT       	| 111M     	| [PDF](https://aclanthology.org/2021.nllp-1.8/)                                             	 | Legal documents                                                                                                                                                                     	            | Oct, 2021        	|
| [readerbench/jurBERT-large](https://huggingface.co/readerbench/jurBERT-large)                                           	| BERT       	| 337M     	| [PDF](https://aclanthology.org/2021.nllp-1.8/)                                             	 | Legal documents                                                                                                                                                                     	            | Oct, 2021        	|


## Generative Language Models (CLMs)

| **Model**                                                                                                               	| **Type**   	      | **Size** 	 | **Article/Citation/Source**                                                                	 | **Pre-trained / Fine-tuned**                                                                                                                                                                   	| **Release Date** 	|
|-------------------------------------------------------------------------------------------------------------------------	|-------------------|------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|------------------	|
| [dumitrescustefan/gpt-neo-romanian-780m](https://huggingface.co/dumitrescustefan/gpt-neo-romanian-780m)                  	| GPT-Neo      	    | 780M     	 | not yet / [HF Space](https://huggingface.co/spaces/dumitrescustefan/romanian-text-generation)                                        	                                             |  Pre-trained                                                                                                                                                                                   	| Jul, 2021        	|
| [readerbench/RoGPT2-base](https://huggingface.co/readerbench/RoGPT2-base)                                               	| GPT2       	      | 124M     	 | [PDF](https://ieeexplore.ieee.org/document/9643330)                                        	 |  Pre-trained                                                                                                                                                                                   	| Jul, 2021        	|
| [readerbench/RoGPT2-medium](https://huggingface.co/readerbench/RoGPT2-medium)                                           	| GPT2       	      | 354M     	 | [PDF](https://ieeexplore.ieee.org/document/9643330)                                        	 |   Pre-trained                                                                                                                                                                                  	| Jul, 2021        	|
| [readerbench/RoGPT2-large](https://huggingface.co/readerbench/RoGPT2-large)                                             	| GPT2       	      | 774M     	 | [PDF](https://ieeexplore.ieee.org/document/9643330)                                        	 |   Pre-trained                                                                                                                                                                                  	| Jul, 2021        	|


NEW: Check out this HF Space to play with Romanian generative models: [https://huggingface.co/spaces/dumitrescustefan/romanian-text-generation](https://huggingface.co/spaces/dumitrescustefan/romanian-text-generation)

## Model evaluation

Models are evaluated using the [public Colab script available here](https://colab.research.google.com/drive/1fa9Yd_7xTSCxwmqoX8ptrZsr0SfpB6_0?usp=sharing). All results reported are the average score of 5 runs, using the same parameters. For larger models, if it was possible, a larger batch-size was simulated by accumulating gradients, such that all models should have the same effective batch size. Only standard models (not finetuned for a particular task) and that could fit in 16GB of RAM are evaluated. 

The tests cover the following fields, and, for brevity, we select a single metric from each field:

* **Named Entity Recognition**: on [RONECv2](https://github.com/dumitrescustefan/ronec) we measure the test strict match measure. A model must correctly detect whether a word is an entity and tag it with its correct class.  
* **Part of Speech Tagging**: on [ro-pos-tagger](https://github.com/dumitrescustefan/ro-pos-tagger) we measure the test UPOS F1 score. This test should reveal how well a model understands the language's structure. 
* **Semantic Textual Similarity**: on [RO-STS](https://github.com/dumitrescustefan/RO-STS) we measure the test Pearson correlation coefficient. Given two sentences the model must predict whether they are entailed, contradictory or are on different subjects (neutral). This test should highlight how well a model can embed the meaning of a sentence.
* **Emotion Detection**: on the [REDv2](https://github.com/Alegzandra/RED-Romanian-Emotions-Dataset) emotion detection in Romanian Tweets we measure the test Hamming loss in the classification setting (_lower is better_). This test should show how well a model can "understand" emotions from short texts.  
* **Perplexity**: on [wiki-ro](https://github.com/dumitrescustefan/wiki-ro)'s test split, we measure CLM-only models' perplexity with a stride of 512 and a batch size of 4.

### MLM model evaluation

| **Model**                                                                                                               	| **Type**   	| **Size** 	| **NER/EM_strict** 	| **RoSTS/Pearson** 	| **Ro-pos-tagger/UPOS F1** 	| **REDv2/hamming_loss** 	 |
|-------------------------------------------------------------------------------------------------------------------------	|------------	|----------	|-------------------	|-------------------	|---------------------------	|--------------------------|
| [dumitrescustefan/bert-base-romanian-cased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)     	| BERT       	| 124M     	| 0.8815            	| 0.7966            	| 0.982                     	| 0.1039                 	 |
| [dumitrescustefan/bert-base-romanian-uncased-v1](https://huggingface.co/dumitrescustefan/bert-base-romanian-uncased-v1) 	| BERT       	| 124M     	| 0.8572            	| 0.8149            	| 0.9826                    	| 0.1038                 	 |
| [racai/distillbert-base-romanian-cased](https://huggingface.co/racai/distilbert-base-romanian-cased)                    	| DistilBERT 	| 81M      	| 0.8573            	| 0.7285            	| 0.9637                    	| 0.1119                 	 |
| [readerbench/RoBERT-small](https://huggingface.co/readerbench/RoBERT-small)                                             	| BERT       	| 19M      	| 0.8512            	| 0.7827            	| 0.9794                    	| 0.1085                 	 |
| [readerbench/RoBERT-base](https://huggingface.co/readerbench/RoBERT-base)                                               	| BERT       	| 114M     	| 0.8768            	| 0.8102            	| 0.9819                    	| 0.1041                 	 |

### CLM model evaluation

| **Model**                                                                     	| **Type** 	| **Size** 	| **NER/EM_strict** 	| **RoSTS/Pearson** 	| **Ro-pos-tagger/UPOS F1** 	| **REDv2/hamming_loss** 	| **Perplexity** 	|
|-------------------------------------------------------------------------------	|----------	|----------	|-------------------	|-------------------	|---------------------------	|------------------------	|----------------	|
| [readerbench/RoGPT2-base](https://huggingface.co/readerbench/RoGPT2-base)     	| GPT2     	| 124M     	| 0.6865            	| 0.7963            	| 0.9009                    	| 0.1068                 	| 52.34          	|
| [readerbench/RoGPT2-medium](https://huggingface.co/readerbench/RoGPT2-medium) 	| GPT2     	| 354M     	| 0.7123            	| 0.7979            	| 0.9098                    	| 0.114                  	| 31.26          	|

### What you can do with these models

Using HuggingFace's Transformers lib, instantiate a model and replace the model name as necessary. Then use an appropriate model head depending on your task. Here are a few examples:

##### Get token embeddings 

```python
from transformers import AutoTokenizer, AutoModel
import torch

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

# tokenize a sentence and run through the model
input_ids = tokenizer.encode("Acesta este un test.", add_special_tokens=True, return_tensors="pt")
outputs = model(input_ids)

# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

* For _dumitrescustefan/*_ models, remember to correct the ș/ț diacritics before feeding it to the model (it was trained only with the correct, comma-style diacritics, and will see the cedilla ş an ţ as UNKs and thus decrease overall performance): 
```
text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
```

#### Write text with generative models

Give a prompt to a generative model and let it [write](https://huggingface.co/blog/how-to-generate): 

```python
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/gpt-neo-romanian-125m")
model = AutoModelForCausalLM.from_pretrained("dumitrescustefan/gpt-neo-romanian-125m")

input_ids = tokenizer.encode("Cine a fost Mihai Eminescu? A fost", return_tensors='pt')

text = model.generate(input_ids, max_length=128, do_sample=True, no_repeat_ngram_size=2, top_k=50, top_p=0.9, early_stopping=True)

print(tokenizer.decode(text[0], skip_special_tokens=True))
```

P.S. You can test all generative models here: [https://huggingface.co/spaces/dumitrescustefan/romanian-text-generation](https://huggingface.co/spaces/dumitrescustefan/romanian-text-generation)

### Final note

* While this repo initially started as an in-depth of a single transformer model back in 2020, with the express hope that more models would be added quickly, it turned out that training a good model is not that easy, and it takes a lot of effort to curate the data and then have access to sufficient compute power. So, I feel it's no longer useful to just list a couple of models, and it would make more impact to list all the models I could find that are Romanian-only, and have a minimal level of performance/documentation. Here you go :)
* This repo contained some code to download and clean a Romanian corpus. I have removed this part as Oscar is now offered on HuggingFace (new version), and OPUS's API is no longer working as it should (some manual filtering is now required, not to mention new resources are being added constantly) - thus maintaining this code is not really feasible.   
* Please contribute to this repo with new Romanian models you mihgt find, or with citations or updates to existing models. 
