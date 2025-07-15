import numpy as np 
import tensorflow as tf

class Postional_Encoding:
    def __init__(self, token_window, d_model):
        """
            Args:
            token_window (int): The maximum number of tokens
            d_model (int): The dimensionality of the model
        """
        self.token_window = token_window
        self.d_model = d_model

    def PE(self):
        position = np.arange(self.max_tokens)[:, np.newaxis] # shape (token_window, 1)
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)) # shape (token_window, 1)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # for even positons 
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) # for odd positions
    
        pos_encoding = angle_rads[np.newaxis, ...] # shape (1, token_window, d_model)
    
        return tf.cast(pos_encoding, dtype=tf.float32)
