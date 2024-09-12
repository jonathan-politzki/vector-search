from gensim.models import KeyedVectors
import numpy as np

def load_model(path='path/to/your/word2vec.bin'):
    return KeyedVectors.load_word2vec_format(path, binary=True)

def perform_operation(model, positive, negative):
    return model.most_similar(positive=positive, negative=negative, topn=1)[0][0]

def find_similar_words(model, word, topn=10):
    return model.most_similar(positive=[word], topn=topn)