import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding

class Word_Embedding:
    def __init__(self, vacab_size, tokens, token_window, d_model, df, encoded_reviews):
        self.vocab_size = vocab_size
        self.tokens = tokens
        self.d_model = d_model
        self.df = df
        self.encoded_words = [one_hot(i, vocab_size) for i in len(df)]

    def WE(self)
        padded_words = pad_sequence(encoded_words, maxlen=token_window, padding='post')
