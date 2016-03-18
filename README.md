# Abstractive Sentence Summarization

## The Algorithm
An implementation of the Rush, Chopra, Weston work
<http://arxiv.org/abs/1509.00685>

Our implementation differs in that we fix the context and summary token
embedding matrix. Two scenarios are possible:

1. Both embedding matrices are initialised from GloVe
2. Context embedding matrix are initialised from Skip-Thought Vectors <http://arxiv.org/abs/1506.06726>, summary from GloVe

## The Code
Helptext is available for running our code by doing:

```bash
python3 ass_train.py -h
```

The training corpuses are under `data/`. Each JSON file contains three fields `title`, `full_text`, `summary`. They're downloaded with scripts in `download_data/`.

GloVe data needs to be downloaded and unzipped under `glove/`. The code uses the first 10k most frequent tokens by default. To generate the embeddings for them, 

```bash
cd glove
head -n 10000 glove.6B.300d.txt >glove.10k.300d.txt
``` 

SkipThoughts data needs to be downloaded to access the SkipThoughts embeddings (to be done).


