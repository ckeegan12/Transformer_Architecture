import torch
import torch.nn as nn

class WordEmbedding:
    def __init__(self, df, vocab_size, weights):
        self.df = df
        self.vocab_size = vocab_size
        self.weights = weights

    def we(self):
        embedding_dim = 384
        max_norm = 1.0
        embedding_vector = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim, 
                                        padding_idx=None, max_norm=max_norm, _weight=self.weights, btype=torch.float32)
        return embedding_vector
