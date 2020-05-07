### Summary
Here you can find a few examples on how to use the transformers in common scenarios:
 * How to [tokenize a text](#short-example-tokenize-a-text)
 * How to [load a pre-trained model](#short-example-tokenize-a-text)
 * How to [get a token embedding and compare similarity to another token](#short-example-token-embeddings)
 * How to [get a sentence embedding and compare similarity to another sentence](#short-example-sentence-embeddings)
 * Advanced: How to train and use a transformer-based model for [Named Entity Recognition](ner/README.md)
 * Advanced: How to train and use a transformer-based model for [Document Classification](doc_classification/README.md)
 * Advanced: How to train and use a transformer-based model for [Sentiment Analysis](sentiment_analysis/README.md)

### Short example: Tokenize a text

Tokenizing a text means passing a string to the Tokenizer object to obtain a list of tokens that you can then pass to the model. BERT model comes with it's own tokenizer. The ``add_special_tokens=True`` option is necessary to automatically mark the sentence with special tokens needed by the model. The tokens themselves are integers, as the vocabulary is a simple map from strings to integers.
```python
from transformers import AutoTokenizer

# load a BERT cased tokenizer
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

# tokenize a sentence and obtain list of tokens and subtokens
sentence = "Am împrumutat cartea acesta."
marked_sentence = "[CLS]" + sentence + "[SEP]"
tokenized_sentence = tokenizer.tokenize(marked_sentence)

print(tokenized_sentence)
# will print: ["[CLS]", "am", "imprumut", "##at", "cartea", "aceasta", "."," [SEP]"]

# tokenize a sentence and obtain list of integers
input_ids = tokenizer.encode(sentence, add_special_tokens=True)
# 'add_special_tokens=True' is equivalent to "'[CLS]' + text + '[SEP]'"

print(input_ids)
# will print: [2, 474, 48617, 368, 4435, 1330, 18, 3]
```
Note that a word will not always map to a token. As we can see in the example above  sometimes a word will map to multiple subtokens. The two hash signs at the beginning of the subtokens tells the tokenizer that it is part of a bigger word. That is because BERT 
tokenizer was created with a WordPiece model to deal with out of vocabulary words. If a word is not found, it tries to break the word into the largest possible subwords contained in the vocabulary. More details about WordPiece can be found in [WordPiece](https://blog.floydhub.com/tokenization-nlp/)

The [AutoTokenizer](https://huggingface.co/transformers/model_doc/auto.html#autotokenizer) is a class that automatically wraps up several transformer models. For example, here are the [operations](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer) you can do if you load the BERT model.

### Short example: Load a Pre-Trained model
The BERT model requires that the input should be in torch tensor format. After converting from python lists to torch tensors,
the shape remains the same. More details about BERT can be found here [BERT](http://jalammar.github.io/illustrated-bert/)

``from_pretrained`` method will download a pretrained model from HuggingFace. In order to output all hidden layers in the model,
we create a  ``BertConfig`` variable in which we set ``output_hidden_states=True``.

```python
import torch
from transformers import AutoModel, BertConfig

name = "dumitrescustefan/bert-base-romanian-cased-v1"
config = BertConfig.from_pretrained(name, output_hidden_states=True)
model = AutoModel.from_pretrained(name, config=config)

input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1

with torch.no_grad():
    outputs = model(input_tensor)

# tensor of shape (batch_size, sequence_length, hidden_size)
last_hidden_states = outputs[0]
print(last_hidden_states.shape)

# tensor of shape (batch_size, hidden_size)
pooler_output = outputs[1]
print(pooler_output.shape)
```
When fetching the output of the model, ``torch.no_grad()`` deactivates the gradient calculation and speed up computation by saving memory.
We don't need gradients since we are running only forward pass of the input.

### Short example: Token embeddings

For each token in the input, the ``hidden_states`` from the output will contain 13 separate vectors of size 768. 
To obtain an individual vector for each token we may take the summ of the last 6 vectors from the ``hidden_states``. Other methods like concatenation can be used.
Feel free to experiment with different methods.

``get_tokkens_embeddings_summ`` will return a list of embeddings for each word in input sentence. We have to chose a max_length for the input_tensor. BERT has a limit at 512 tokens for an input tensor. Longer inputs must be truncated. This is a well known problem for BERT where longer documents classifications is a challenge. We will face this challenge in an advanced tutorial. 

``tokens_similarity`` compares the cosine distance between embeddings at certain positions in the embeddings list.

```python
def get_tokkens_embeddings_summ(sentence, max_length=32):
    token_vecs_summ = []

    ids_sentence = tokenizer.encode(sentence, add_special_tokens=True,
                                                pad_to_max_length=True,
                                                 max_length=max_length)

    input_tensor = torch.tensor(ids_sentence).unsqueeze(0)  # Batch size 1

    with torch.no_grad():
        _, _, hidden_states = model(input_tensor)
                                    
    token_embeddings = torch.stack(hidden_states, dim=0).squeeze(1).permute(1,0,2)
    
    # For each token in the sentence...
    for token in token_embeddings:
        sum_vec = torch.sum(token[-6:], dim=0)

        token_vecs_summ.append(sum_vec)

    return token_vecs_summ

def tokens_similarity(tokens_emb, idx1, idx2):
    
    tok1_emb = tokens_emb[idx1]
    tok2_emb = tokens_emb[idx2]

    output = cosine(tok1_emb, tok2_emb)
    
    return 1-output

sentence1 = "Am mers pe lac si am inotat in lac."
sentence2 = "Lebedele inotau pe lac si am dat cu lac pe unghii."
tokens_emb1 = get_tokkens_embeddings_summ(sentence1)
tokens_emb2 = get_tokkens_embeddings_summ(sentence2)

print("Semantic similarity between tokens 'lac' in the first sentence is {}".format(tokens_similarity(tokens_emb1, 3, 9)))
# Semantic similarity between tokens 'lac' in the first sentence is 0.7684007287025452
print("Semantic similarity between tokens 'lac' in the second sentence is {}".format(tokens_similarity(tokens_emb2, 6, 11)))
# Semantic similarity between tokens 'lac' in the second sentence is 0.6081207394599915
```

As we can see in the first sentence the word 'lac' has the same meaning in both uses and the model returns a greater score than in the second sentence
where the word has different meanings.


### Short example: Sentence embeddings

To obtain a sentence embeddings we have multiple options. A simple approach could be to average all the hidden layers of each token and obtain a vector of shape (768,).

``get_tokkens_embeddings_summ`` will return a vector of shape (768,) which will represent the sentence embedding.

``sentence_similarity`` compares the cosine distance between two sentence embeddings similar to word embeddings.


```python
from scipy.spatial.distance import cosine

def get_sentence_embeddings(sentence, max_length=16):
    
    ids_sentences = tokenizer.encode(sentence, add_special_tokens=True,
                                    pad_to_max_length=True,
                                    max_length=max_length)

    input_tensor = torch.tensor(ids_sentences).unsqueeze(0)  # Batch size 1

    with torch.no_grad():
        _, _, hidden_states = model(input_tensor)
    
    token_vecs = hidden_states[12][0]

    sentence_embedding = torch.mean(token_vecs, dim=0)
    
    return sentence_embedding

def sentence_similarity(sent1, sent2):
    
    sent1_emb = get_sentence_embeddings(sent1)
    sent2_emb = get_sentence_embeddings(sent2)
    
    output = cosine(sent1_emb, sent2_emb)
    
    return 1-output

sentence1 = "Am fost la doctor la un control"
sentence2 = "Medicul a operat in salon"
sentence3 = "Nu am fost niciodata atat de vesela"

print("Semantic similarity between first and second sentence is {}".format(sentence_similarity(sentence1, sentence2)))
# Semantic similarity between first and second sentence is 0.8724686503410339
print("Semantic similarity between second and third sentence is {}".format(sentence_similarity(sentence2, sentence3)))
# Semantic similarity between second and third sentence is 0.7341931462287903
print("Semantic similarity between first and third sentence is {}".format(sentence_similarity(sentence1, sentence3)))
# Semantic similarity between first and third sentence is 0.7526880502700806
```
As we can see in the first sentence the meaning is closer to the second sentence than to the third. This is reflected by the cosine distance.
Interesting to observe is that the third sentence is more close to the first sentence than to the second, probably because they both have 'am fost' words in them.

More details can be found here [colab](https://colab.research.google.com/drive/1L2_UnNSxJn6FCYL0as5Q1DoOGdh3ejqM#scrollTo=Wl9OUN_TvNDQ)