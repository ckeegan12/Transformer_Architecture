import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
import gensim
from gensim.models import Word2Vec, KeyedVectors

nltk.download('punkt')

class Word2VecEmbedding:
    def __init__(self, tokens, vector_size, window, min_count=1, workers=4, sg=0):
        """
        args:  
        tokens (list of list of str): Tokenized sentences.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Ignores all words with total frequency lower than this.
        workers (int): Number of worker threads for training.
        sg (int): CBOW.
        """
        self.model = Word2Vec(
            sentences=tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg
        )
    
    def get_embedding(self, word):
        """
        Retrieve the intialized embedding vector for a given word.
        args:
        word: The word to retrieve the embedding for.
        return: The embedding vector as a numpy array, or None if the word is not in the vocabulary.
        """
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            print(f"'{word}' not found in vocabulary.")
            return None
