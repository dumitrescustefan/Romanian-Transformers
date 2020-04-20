# Corpus 

Here's how to download and prepare the corpus for LM training.

The corpus is, in version 1, composed of:
- Romanian Wikipedia 
- OSCAR (based on CommonCrawl)
- OPUS (all the Romanian monolingual data)

## Corpus details
Please see [here](CORPUS_DETAILS.md) all the details about size and cleaning statistics.
The rest of this page is dedicated to running the download and cleaning process step by step.



## How to get the corpus

Note: Please use Python 3.8 or later. Because of the way unicode strings are handled in Python 3.8+, the cleaning script will not work with earlier versions.

#### Step 0. Prerequesites
Please install all the dependencies with:
```shell script
pip3 install -r requirements.txt
```

#### Step 1. Download the corpus
Run the following shell commands:
```shell script
./wiki_download.sh
./opus_download.sh
./oscar_download.sh
```
The scrips will create the ``raw`` folder and download the corpora there.

#### Step 2. Clean the corpus
Run the following python scripts:
```shell script
python3 wiki_clean.py
python3 opus_clean.py
python3 oscar_clean.py
``` 
The scripts will take a few hours to clean the corpora. The folder ``clean`` will be created with the txt files for each corpus. 

#### Step 3. Merge the corpora
The script will concatenate the corpora while extracting a validation subset of sentences that respects the distribution of line in each composing corpus.
```shell script
python3 merge_corpora.py
```
The folder ``merged`` will now contain a ``train.txt`` and a ``valid.txt``.
Edit the ``valid_count = 5000`` in ``merge_corpora.py`` to change the default number of lines that will be extracted for validation.

