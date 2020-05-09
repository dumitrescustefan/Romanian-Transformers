### Sentiment analysis

Here you can find a colab example on how to use the transformers for sentiment analysis:
 * [sentiment_analysis](https://colab.research.google.com/drive/1vKv1Kp9omFr9y4HlFWUGmiYjHgk5XD0m#scrollTo=FU-8vkP25DfU)

In `data_exploration.ipynb` we can observe that the dataset is imbalanced, so we used a custom sampler to balance the dataset. Also we observe that
than 75% of dataset has lenght under 124, so we set max_length for BERT input to 256.

## Evaluation Results

Evaluation is performed on a Romanian [movies reviews](https://github.com/katakonst/sentiment-analysis-tensorflow/tree/master/datasets/ro) dataset.

|           	| Precision | Recall 	| F1-Score 	| Support 	|
|-----------	|:--------:	|:--------:	|:--------:	|:--------:	|
| 0         	|   0.75  	|  0.89  	|   0.82  	|    4828   |
| 1         	|   0.90  	|  0.77  	|   0.83  	|    6177   |
| accuracy      |       	|       	|   0.82  	|    11005  |
| macro avg     |   0.83  	|  0.83 	|   0.83  	|    11005  |
| weigt avg 	|   0.83   	|  0.82  	|   0.82  	|    11005  |
