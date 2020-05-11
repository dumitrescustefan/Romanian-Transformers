# Evaluation (under construction)

Here are the tools used to evaluate our models on. We currently evaluate on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) (RRT) as well as on [Romanian Named Entity Corpus](https://github.com/dumitrescustefan/ronec) (RONEC). 

## Scripts

Run the `setup.sh` script to download the datasets and configure the environment. 

```
.\setup.sh
```

Run `eval_*.sh` to start the evaluation of a model on one of the tasks. You must specify either the name of the model (from HuggingFace repository), or the path to the model (saved in HuggingFace format). You can also specify the number of iteration for each experiment as the second (optional) parameter of the script (the default is 1). The evaluation will automatically start on the GPU. You can specify the device you want to evaluate on as the third argument of the script.

```
eval_*.sh [language_model_name] [no_iteration] [device]
```

## Results

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
| bert-base-multilingual-uncased |   97.72  |   96.54  |    94.67   |  87.65  |
| bert-base-multilingual-cased   |   -  |   -  |    -   |  -  |
| bert-base-romanian-uncased-v1  |   97.91 | 97.01 | 94.93 | 89.61  |
| bert-base-romanian-cased-v1    |   97.90 | 97.22 | 94.88 | **89.69**  |

#### Named Entity Recognition

| Model                          | Entity Type | Partial | Strict | Exact |
|--------------------------------|:-----------:|:-------:|:------:|:-----:|
| bert-base-multilingual-uncased |    84.75    |  86.06  |  80.81 | 83.91 |
| bert-base-multilingual-cased   |    84.52    |  86.27  |  80.6  | 84.13 |
| bert-base-romanian-uncased-v1  |    85.53    |  87.17  |  82.01 | 85.26 |
| bert-base-romanian-cased-v1    |    **86.21**    |  **87.84**  |  **82.54** | **85.88** |
