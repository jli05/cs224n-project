# Abstractive Sentence Summarization

An implementation of the Rush, Chopra, Weston work
<http://arxiv.org/abs/1509.00685>

Our implementation differs in that we fix the context and summary token
embedding matrix. Two scenarios are possible:

1. Both embedding matrices are initialised from GloVe
2. Context embedding matrix are initialised from Skip-Thought Vectors <http://arxiv.org/abs/1506.06726>, summary from
   GloVe

Helptext is available for running our code by doing:

```bash
python ass_train.py -h
```

To download our dataset, feel free to use our utility code available in
`download_data/`.

GloVe data needs to be downloaded to access the GloVe embedding (default).
SkipThoughts data needs to be downloaded to access the SkipThoughts embeddings.

We reccommend training on a system with GPU support.
