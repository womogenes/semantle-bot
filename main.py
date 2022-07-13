from scipy import spatial
from gensim.models import KeyedVectors
import os
import numpy as np
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
wv = KeyedVectors.load("word2vec.model", mmap="r")
with open("english-words.txt") as fin:
    english_words = fin.read().strip().split("\n")
print("Datasets loaded.")
print()


def find_possible_vectorized(guess, reported_sim, words_to_consider):
    """
    Returns the list of all words in [words_to_consider] <= 0.05 from [reported_sim] of [guess].
    This version is vectorized and exploits the fast computation of matrix-vector products.

    cos(A, B) = dot(A, B) / sqrt(A^2) * sqrt(B^2)

    In this case, B is a matrix. Therefore, we get a vector of cosines, from which we look up the closest words.
    """
    p_bar = tqdm(range(3))

    # Need to map between indices of the matrix and words
    id_to_word = list(words_to_consider) if not isinstance(words_to_consider, list) else words_to_consider

    guess_v = wv[guess]  # features x 1
    # TODO: This operation takes long because for-loop over words, is there a way to load word2vec directly as a matrix?
    words_matrix = np.stack([wv[word] for word in words_to_consider])  # num_words x features

    p_bar.update(1)

    # num_words x features @ features x 1 = num_words x 1
    numerator = words_matrix @ guess_v  # dot product of every datapoint with guess_v

    # words_matrix @ words_matrix.T --> the squares of the matrix are on the diagonal
    # we want to avoid computing all the non-diagonal elements somehow
    # we can achieve this with einstein summation:
    #   np.einsum('ij,jk') is the normal matrix product
    #   np.einsum('ij,ji') gives us the sum over all diagonal elements of the matrix product
    #   np.einsum('ij,ji->i') unforces the sum operation, so just returns the elements of the diagonal
    # dim: num_words x features \w features x num_words --> num_words x 1
    norms = np.sqrt(np.einsum('ij,ji->i', words_matrix, words_matrix.T))

    denominator = norms * np.linalg.norm(guess_v, 2)  # elem-multiply by norm of guess, denominator --> (num_words x 1)

    # NOTE: We do not have to do 1 - cosine, because spatial.distance.cosine calculates cosine DISTANCE
    # Whereas I here calculate cosine SIMILARITY directly
    cosines = (numerator / denominator) * 100  # num_words x 1, cosines[i] = SEMANTLE cosine of guess with ith wv

    p_bar.update(1)

    # return all the indices where the difference is <= 0.05
    candidates = np.where(np.abs(cosines - reported_sim) <= 0.005)[0]

    p_bar.update(1)

    return [id_to_word[i] for i in candidates]


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
            set(find_possible_vectorized(guess, reported_sim, possible)))
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


if __name__ == "__main__":

    while True:
        do_run()

        while True:
            res = input("Would you like to do another run? [y/N] ").lower()
            if res == "y":
                break
            else:
                exit()
