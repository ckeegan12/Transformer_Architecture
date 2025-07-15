import nltk
from nltk.tokenize import RegexpTokenizer

class Word_Tokenizer:
    def __init__(self, text):
        self.text = text
        self.tokenizer = RegexpTokenizer(r'\w+')

    def Tokenize(self):
        tokens = self.tokenizer.tokenize(self.text)
        # Normalize tokens
        normalized_tokens = [token.lower() for token in tokens]
        # Add start and stop tokens
        processed_tokens = ['<START>'] + normalized_tokens + ['<STOP>']
        return(processed_tokens)
