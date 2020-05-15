## Sentiment analysis

### Description

The goal of this project is to fine-tune BERT model on Romanian sentiment dataset. 
The code is modular for easier reading and reusability. All the modules are imported in `sentiment_analysis.ipynb` 
notebook which is then runned in Google's colaboratory.

## How to run

Here is an example on how to run experiment on local machine:

```python
# clone project
git clone git@github.com:dumitrescustefan/Romanian-Transformers.git

# install dependencies
cd Romanian-Transformers/examples/sentiment_analysis
pip install -r requirements.txt

# run jupyter notebook
jupyter lab
```

Here you can find the colab example on how to run the experiment:
 * [sentiment_analysis](https://colab.research.google.com/drive/1vKv1Kp9omFr9y4HlFWUGmiYjHgk5XD0m#scrollTo=FU-8vkP25DfU)

### Dataset

Dataset can be found at [dataset](https://github.com/katakonst/sentiment-analysis-tensorflow/tree/master/datasets/ro). 
Label `0` denotes a negative examples and label `1` denotes a positive example. 
It contains aproximatively 4000 negative examples and 11000 positive examples.

In `data_exploration.ipynb` we can observe that the dataset is imbalanced, so we used a custom sampler to balance the dataset. Also we observe that more
than 75% of dataset has lenght under 124, so we set max_length for sentences to 256.

### Evaluation Results

Evaluation is described in the following table:

|           	| Precision | Recall 	| F1-Score 	| Support 	|
|-----------	|:--------:	|:--------:	|:--------:	|:--------:	|
| 0         	|   0.75  	|  0.89  	|   0.82  	|    4828   |
| 1         	|   0.90  	|  0.77  	|   0.83  	|    6177   |
| accuracy      |       	|       	|   0.82  	|    11005  |
| macro avg     |   0.83  	|  0.83 	|   0.83  	|    11005  |
| weigt avg 	|   0.83   	|  0.82  	|   0.82  	|    11005  |
