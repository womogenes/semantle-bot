from main import *

# Need to be able to map indices onto words and vice versa
id_to_word = english_words
word_to_id = {word : id for id, word in enumerate(english_words)}

def find_possible_vectorized(guess, reported_sim, words_to_consider):

    """
    Returns the list of all words in words_to_consider <= 0.05 from reported_sim of guess.
    This version is vectorized and exploits the fast computation of matrix-vector products.

    cos(A, B) = dot(A, B) / sqrt(A^2) * sqrt(B^2)

    In this case, B is a matrix. Therefore we get a vector of cosines, from which we look-up the closest words.
    """

    guess_v = wv[guess] # features x 1

    #TODO: This operation takes long because for-loop over words, is there a way to load word2vec directly as a matrix?
    words_matrix = np.stack([wv[word] for word in words_to_consider]) # num_words x features

    # num_words x features @ features x 1 = num_words x 1
    numerator = words_matrix @ guess_v # dot product of every datapoint with v

    # if we do: words_matrix @ words_matrix.T, the squares of the matrix are on the diagonal
    # we want to avoid computing all the non-diagonal elements somehow
    # we can achieve this with einstein summation:
    #   np.einsum('ij,jk') is the normal matrix product
    #   np.einsum('ij,ji') gives us the sum over all diagonal elements of the matrix product
    #   np.einsum('ij,ji->i') unforces the sum operation, so just returns the elements of the diagonal
    # dim: num_words x features \w features x num_words --> num_words x 1
    norms = np.sqrt(np.einsum('ij,ji->i', words_matrix, words_matrix.T))

    denominator = norms * np.linalg.norm(guess_v, 2) # elem-multiply by norm of guess, denominator --> (num_words x 1)

    # NOTE: We do not have to do -1, because spatial.distance.cosine calculates cosine DISTANCE
    # Whereas I here calculate cosine SIMILARITY directly
    cosines = (numerator / denominator) * 100 # num_words x 1, cosines[i] = SEMANTLE cosine of guess with ith wv

    #return all the indices where the difference is <= 0.05
    candidates = np.where(np.abs(cosines - reported_sim) <= 0.005)[0]

    return [id_to_word[i] for i in candidates]

def benchmark(guess, reported_sim, words_to_consider):
    num_operations = len(words_to_consider) # number of cosines we have to computed
    print('Computing vanilla...')
    s_0 = perf_counter()
    vanilla_set = find_possible(guess, reported_sim, words_to_consider)
    s_1 = perf_counter()

    print('Computing vectorized...')
    s_2 = perf_counter()
    vectorized_set = find_possible_vectorized(guess, reported_sim, words_to_consider)
    s_3 = perf_counter()

    assert set(vectorized_set) == set(vanilla_set)

    print()
    print("{:<25} {:<25} {:<25}".format('Method', 'Elapsed time', 'Time per cosine'))
    print("{:<25} {:<25} {:<25}".format('Vanilla:', str(s_1-s_0), str((s_1-s_0) / num_operations)))
    print("{:<25} {:<25} {:<25}".format('Vectorized:', str(s_3 - s_2), str((s_3 - s_2) / num_operations)))
    print(f'Speed-up factor: {(s_1 - s_0)/(s_3 - s_2)}')

guess = 'house'
sim = 8.43

benchmark(guess, sim, english_words)
