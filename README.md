# Semantle bot

Complete search bot to assist in playing the word-guessing game [Semantle](https://semantle.com/).

## Instructions

To download the datasets, run `./setup.py`. Note that it will download **~1.6 GB** of data and could take several minutes depending on your internet speed. This will create the files `word2vec.model`, `word2vec.model.vectors.npy`, and `english-words.txt`.

To run the bot, run `./main.py` and follow the prompts. Note that the initial complete search will probably take around 10 minutes to retrieve word vectors from the file. Subsequent complete searches will be quicker.

## Dependencies

- [gensim](https://pypi.org/project/gensim/) for word vector download
- [tqdm](https://github.com/tqdm/tqdm) for progress bars

## References

- My explanatory YouTube video: https://youtu.be/EHZYiuxGSX8
- List of English words from https://github.com/dwyl/english-words
- Google Word2Vec dataset from https://code.google.com/archive/p/word2vec
