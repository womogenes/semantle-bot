from scipy import spatial
from gensim.models import KeyedVectors

import os
from tqdm import tqdm
import random

# Make sure datasets are downloaded
needed_files = ["word2vec.model",
                "word2vec.model.vectors.npy", "english-words.txt"]
existing_files = set(os.listdir())
for file in existing_files:
    if file not in existing_files:
        print("Run 'setup.py' to download the datasets first.")


# Load datasets
print("Loading datasets...")
wv = KeyedVectors.load("./word2vec.model", mmap="r")
with open("./secret-words.txt") as fin:
    english_words = fin.read().strip().split("\n")
print("Datasets loaded.")
print()


def similarity(word1, word2):
    return (1 - spatial.distance.cosine(wv[word1], wv[word2])) * 100


def find_possible(guess, reported_sim, words_to_consider):
    """
    Return a set of possible words given a guess and its the similary to the secret word.
    """
    ans = []
    for w in tqdm(words_to_consider):
        if abs(similarity(guess, w) - reported_sim) <= 0.005:
            ans.append(w)
    return ans


def make_list(words):
    if len(words) == 0:
        return ""
    if len(words) == 1:
        return f"'{words[0]}'"
    if len(words) == 2:
        return f"'{words[0]}' and '{words[1]}'"
    return "'" + "', '".join(words[:-1]) + "', and '" + words[-1] + "'"


# Start guessing process
def do_run():
    possible = set(english_words)

    print("Enter guesses as '<guess>, <similarity>'")
    print(f"There are {len(possible)} possible words remaining.")
    print()

    while len(possible) > 1:
        while True:
            data = input("Guess: ")
            if ", " in data:
                try:
                    guess, reported_sim = data.split(", ")
                    reported_sim = float(reported_sim)
                    break

                except ValueError:
                    print("Reported similarity was not a float.")
                    print()

        possible = possible.intersection(
            set(find_possible(guess, reported_sim, possible)))
        sample_words = random.sample(sorted(possible), min(3, len(possible)))

        if len(possible) > 3:
            print(
                f"There are {len(possible)} possible words remaining (such as {make_list(sample_words)}).")

        elif len(possible) > 1:
            print(
                f"There are {len(possible)} possible words remaining: {make_list(sample_words)}.")

        elif len(possible) == 1:
            print(f"The answer is '{possible.pop()}'!")

        else:
            print(f"There are no possible words remaining--something went wrong.")

        print()


while True:
    do_run()

    while True:
        res = input("Would you like to do another run? [y/N] ").lower()
        if res == "y":
            break
        else:
            exit()
