# Semantle bot

Complete search bot to assist in playing the word-guessing game [Semantle](https://semantle.com/).

## Instructions

To download the datasets, run `./setup.py`. Note that it will download **~1.6 GB** of data and could take several minutes depending on your internet speed. This will create the files `word2vec.model`, `word2vec.model.vectors.npy`, and `english-words.txt`.

To run the bot, run `./main.py` and follow the prompts.

## References

- List of English words from https://github.com/dwyl/english-words
- Google Word2Vec dataset from https://code.google.com/archive/p/word2vec, downloaded with [gensim](https://radimrehurek.com/gensim/models/word2vec.html)
