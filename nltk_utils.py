import numpy as np
import nltk
import sys
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Dividir la oración en palabras o tokens
un token puede ser una palabra, un caracter o número
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    derivados = encontrar una palabra raíz
    por ejemplo:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    retorno de bolsa de arrays:
    1 por cada palabra que que existe en la oración, 0 en caso contrario
    ejemplo:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # raíz de cada palabra
    sentence_words = [stem(word) for word in tokenized_sentence]
    # inicializa la bolsa con 0 para cada palabra
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
