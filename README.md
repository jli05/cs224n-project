Helptext is available for running our code by doing:

```bash
python ass_train.py -h
```

The embedding can be swapped to `SkipThoughts()` to implement the SkipThoughts implementation. Default is our standard GloVe word embeddings.

To download our dataset, feel free to use our utility code available in
`download_data/`.

GloVe data needs to be downloaded to access the GloVe embedding (default).
SkipThoughts data needs to be downloaded to access the SkipThoughts embeddings.

We reccommend training on a system with GPU support.
