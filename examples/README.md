#### Summary
Here you can find a few examples on how to use the transformers in common scenarios:
 * How to [tokenize a text](####short-example:-tokenize-a-text)
 * How to [get a token embedding and compare similarity to another token](####short-example:-token-embeddings)
 * How to [get a sentence embedding and compare similarity to another sentence](####short-example:-sentence-embeddings)
 * Advanced: How to train and use a transformer-based model for [Named Entity Recognition](ner/README.md)
 * Advanced: How to train and use a transformer-based model for [Document Classification](doc_classification/README.md)
 * Advanced: How to train and use a transformer-based model for [Sentiment Analysis](sentiment_analysis/README.md)

#### Short example: Tokenize a text

Tokenizing a text means passing a string to the Tokenizer object to obtain a list of tokens that you can then pass to the model. The ``add_special_tokens=True`` option is necessary to automatically mark the sentence with special tokens needed by the model. The tokens themselves are integers, as the vocabulary is a simple map from strings to integers.
```python
from transformers import AutoTokenizer

# load a BERT cased tokenizer
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

# tokenize a sentence and obtain list of integers 
input_ids = tokenizer.encode("Acesta este un test.", add_special_tokens=True)

# print tokens
print(input_ids)
# will print: [2, 1330, 443, 396, 4231, 18, 3]
```
Note that a word will not always map to a token; sometimes it will map to multiple subtokens. This is normal and expected by the model. 

The [AutoTokenizer](https://huggingface.co/transformers/model_doc/auto.html#autotokenizer) is a class that automatically wraps up several transformer models. For example, here are the [operations](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer) you can do if you load the BERT model.

#### Short example: Token embeddings




#### Short example: Sentence embeddings



 