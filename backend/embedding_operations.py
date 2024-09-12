from gensim.models import KeyedVectors
import numpy as np
import os

def load_model(path='models/GoogleNews-vectors-negative300.bin'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, path)
    return KeyedVectors.load_word2vec_format(model_path, binary=True)

def perform_operation(model, positive, negative):
    return model.most_similar(positive=positive, negative=negative, topn=1)[0][0]

def find_similar_words(model, word, topn=10):
    return model.most_similar(positive=[word], topn=topn)