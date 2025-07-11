import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, Temp, atten_dropout=0.1):
        self.Temp = Temp
        self.atten_dropout = atten_dropout
        self.dropout = nn.Dropout(atten_dropout)

    def fowardpass(self, query, key, value, Mask=None):
        scores = np.matmul(query / self.Temp, key.T) / math.sqrt(key.size(-1))

        if Mask is not None:
            scores += -1e9 * Mask

        weights = self.dropout(F.softmax(scores, dim=-1))

        return np.matmul(weights, value), weights
