import numpy as np
import sys


# This function get the vectors file name and load the words and their vectors
def load_vectors(vectors_fname):
    words = []
    W =[]
    for line in file(vectors_fname):
        text = line.strip()
        text_list = text.split()
        word, vector = text_list[0], list(map(float, text_list[1:]))
        words.append(word)
        W.append(vector)
    W = np.array(W)
    return W,words


# This function find the k most similar words to the target word
def find_similar_words(W, words, w2i, k, target_word):
    target_word_vec = W[w2i[target_word]]
    sims = W.dot(target_word_vec)
    most_similar_ids = sims.argsort()[-1:-(k+1):-1]
    similar_words = []
    for i in most_similar_ids:
        similar_words.append(words[i])
    return similar_words


def main(argv):
    dependency_fname = argv[0]
    bow_fname = argv[1]
    word_list = ['car','bus','hospital','hotel','gun','bomb','horse','fox','table','bowl','guitar','piano']

    # compute similarities with dependency based W2V
    W, words = load_vectors(dependency_fname)
    w2i = {w: i for i, w in enumerate(words)}
    out = open('W2V_dependency','w')
    for target_word in word_list:
        words_str = ' '
        sim_words = find_similar_words(W, words,w2i,21,target_word)
        out.write(target_word + ' similar words : \n')
        words_str = words_str.join(sim_words[1:])
        out.write(words_str + '\n')
    out.close

    # compute similarities with W2V based on bag of words with window size of 5
    W, words = load_vectors(bow_fname)
    w2i = {w: i for i, w in enumerate(words)}
    out = open('W2V_cbow','w')
    for target_word in word_list:
        words_str = ' '
        sim_words = find_similar_words(W, words,w2i,21,target_word)
        out.write(target_word + ' similar words : \n')
        words_str = words_str.join(sim_words[1:])
        out.write(words_str + '\n')
    out.close


if __name__=='__main__':
    main(sys.argv[1:])