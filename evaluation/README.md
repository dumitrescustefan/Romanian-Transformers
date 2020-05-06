# Evaluation (under construction)

Here are the tools used to evaluate our models on. 

We currently evaluate on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) tasks as well as on the [Romanian Named Entity Corpus](https://github.com/dumitrescustefan/ronec) task. Details are given in the [evaluation readme](evauation/README.md) NER task. 

Note that the ``eval_rrt.sh`` evaluates the model with **frozen** weights, meaning that the parameters of the model are untouched; during the evaluation only the head (a single feedforward with dropout) is trained. The ``eval_udify.sh`` uses UDify which **fine-tunes** the weights of the model jointly with the (more-complex) heads.

## Setup
Run the `setup.sh` script to download and configure the [UD Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) (RRT), [Romanian Named Entity Corpus](https://github.com/dumitrescustefan/ronec) (RONEC) and [Udify](https://github.com/Hyperparticle/udify). 

```
.\setup.sh
```

## Evaluation

Run the `eval_rrt.sh`, `eval_ronec.sh`, `eval_udify.sh` to evaluate a model on RRT, RONEC and on RRT with Udify, respectively. You must provide as argument either the [model name](https://huggingface.co/transformers/pretrained_models.html) or the model path, saved in [HuggingFace's format](https://huggingface.co/transformers/main_classes/model.html#pretrainedmodel). For instance, the following will start training a multilingual BERT cased on RRT:

```
.\eval_rrt.sh bert-base-multilingual-cased
```

**Warning!** If no GPU is automatically detected, the evaluation process will start on the CPU (this is totally not recommended). However, you can provide the running `device` as the second argument if the automatic detection fails. Training on `cuda`, using the above example: 

```
.\eval_rrt.sh bert-base-multilingual-cased cuda
```

#### Universal Dependencies

| Model                          | UPOS <br> (frozen) | XPOS <br> (frozen) | UPOS  |  XPOS |
|--------------------------------|:-------------:|:-------------:|:-----:|:-----:|
| bert-base-multilingual-uncased |     95.48     |      89.84    | 97.65 | 95.72 |
| bert-base-multilingual-cased   |     94.46     |      89.50    | 97.87 | 96.16 |
| bert-base-romanian-uncased-v1  |     **96.55**     |      **95.14**    | **98.18** | **96.84** |
| bert-base-romanian-cased-v1    |     96.49     |      95.01    | 98.00 | 96.46 |

#### Universal Dependencies with UDify

| Model                          | UPOS | UFeats | Lemma | LAS |
|--------------------------------|:----:|:----:|:------:|:---:|
| bert-base-multilingual-uncased |   97.74  |   96.61  |    94.50   |  87.58  |
| bert-base-multilingual-cased   |   -  |   -  |    -   |  -  |
| bert-base-romanian-uncased-v1  |   97.90  |   97.02  |    94.92   |  89.6  |
| bert-base-romanian-cased-v1    |   -  |   -  |    -   |  -  |

#### Named Entity Recognition

| Model                          | Entity Type | Partial | Strict | Exact |
|--------------------------------|:-----------:|:-------:|:------:|:-----:|
| bert-base-multilingual-uncased |    84.75    |  86.06  |  80.81 | 83.91 |
| bert-base-multilingual-cased   |    84.52    |  86.27  |  80.6  | 84.13 |
| bert-base-romanian-uncased-v1  |    85.53    |  87.17  |  82.01 | 85.26 |
| bert-base-romanian-cased-v1    |    **86.21**    |  **87.84**  |  **82.54** | **85.88** |
