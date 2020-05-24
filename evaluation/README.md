# Evaluation

Here are the tools used to evaluate our models on. We currently evaluate on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) (RRT) as well as on [Romanian Named Entity Corpus](https://github.com/dumitrescustefan/ronec) (RONEC). All results reported below are the averages of 5 random-seed runs.

We are evaluating on three token-labeling tasks: 

### 1. **Simple Universal Dependencies** 
A model is evaluated on the UPOS (Universal Part-of-Speech) tag and the XPOS (eXtended Part-of-Speech) tag. The model must correctly predict the UPOS & XPOS tags for each word in a sentence. The model itself is basic: the transformer encodes each word and a single dense layer with a fixed dropout predicts the output label. The script runs the model 4 times: on UPOS and XPOS with the transformer parameters frozen, and the same UPOS and XPOS with all the parameters unfrozen (trainable). To evaluate, run script ``eval_rrt.sh``. 

| Model                          | UPOS <br> (frozen) | XPOS <br> (frozen) | UPOS  |  XPOS |
|--------------------------------|:-------------:|:-------------:|:-----:|:-----:|
| bert-base-multilingual-uncased |     95.48     |      89.84    | 97.65 | 95.72 |
| bert-base-multilingual-cased   |     94.46     |      89.50    | 97.87 | 96.16 |
| bert-base-romanian-uncased-v1  |     **96.55**     |      **95.14**    | **98.18** | **96.84** |
| bert-base-romanian-cased-v1    |     96.49     |      95.01    | 98.00 | 96.46 |

### 2. **Joined Universal Dependencies**

We've automated the [UDify](https://github.com/Hyperparticle/udify) tool to fine-tune the model and report results on test data. UDify is a single model that parses Universal Dependencies (UPOS, UFeats, Lemmas, Deps) jointly, using a transformer to provide word embeddings. We report results on the UPOS, UFeats, Lemma and LAS tasks. To evaluate, run script ``eval_udify.sh``.


| Model                          | UPOS | UFeats | Lemma | LAS |
|--------------------------------|:----:|:----:|:------:|:---:|
| bert-base-multilingual-uncased |   97.72  |   96.54  |    94.67   |  87.65  |
| bert-base-multilingual-cased   |   97.90 | 96.71 | **95.20** | 88.05 |
| bert-base-romanian-uncased-v1  |   **97.91** | 97.01 | 94.93 | 89.61  |
| bert-base-romanian-cased-v1    |   97.90 | **97.22** | 94.88 | **89.69**  |


### 3. **Named Entity Recognition**
The model used to perform NER is a transformer that, for each token, predicts a BIO-style label on one of the 16 classes in [RONEC](https://github.com/dumitrescustefan/ronec). To evaluate, run script ``eval_ronec.sh``.

| Model                          | Entity Type | Partial | Strict | Exact |
|--------------------------------|:-----------:|:-------:|:------:|:-----:|
| bert-base-multilingual-uncased |    84.75    |  86.06  |  80.81 | 83.91 |
| bert-base-multilingual-cased   |    84.52    |  86.27  |  80.6  | 84.13 |
| bert-base-romanian-uncased-v1  |    85.53    |  87.17  |  82.01 | 85.26 |
| bert-base-romanian-cased-v1    |    **86.21**    |  **87.84**  |  **82.54** | **85.88** |


## Setup & run

To replicate these results, or evaluate on your own model, run the `setup.sh` script to download the datasets and install prerequisites. We'd recommended running everything inside a virtual environment.

Run `eval_*.sh` to start the evaluation of a model on one of the tasks. Each script will save the models in `models/`, the outputs in `outputs/` and the results (for each model and the average) in `results/`.

```
.\eval_*.sh [language_model_name] [no_iterations] [device]
```

where:

- `lang_model_name` - The name of the model (from HuggingFace repository) or the path to the model (saved in HuggingFace format).
- `no_iterations` (optional) - Number of experiments for each task. Default value: `1`.
- `device` (optional) - The device to run the evaluation on. Default value: `cuda`, alternative `cpu`.

For example, run a cased multilingual BERT on RONEC five times (it will automatically average results over the five iterations in ``results/``:
```
.\eval_ronec.sh bert-base-multilingual-cased 5
```
.. or, run the uncased Romanian BERT on UDify once (Note: UDify is the only tool that does not (yet) have automatic iteration results averaging):
```
.\eval_udify.sh dumitrescustefan/bert-base-romanian-uncased-v1 1
```
