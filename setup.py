# Run this file first
import os

# Check if libraries are installed

if not {"word2vec.model", "word2vec.model.vectors.npy"}.issubset(set(os.listdir())):
    # Get the pretrained Google Word2Vec dataset
    # This might take a couple minutes
    print("Downloading Google word2vec dataset...")

    import gensim.downloader
    wv = gensim.downloader.load("word2vec-google-news-300")
    wv.save("./word2vec.model")

    print("Download complete.")


if not "english-words.txt" in os.listdir():
    print("Downloading list of English words...")

    import requests
    words = requests.get(
        "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt").content.decode().strip().split("\n")

    word2vec_words = set(wv.index_to_key)
    words = [w.strip() for w in words if w.strip() in word2vec_words]

    with open("./english-words.txt", "w") as fout:
        fout.write("\n".join(words))

    print("Download complete.")


print("Setup complete.")
