import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.get_positional_encoding()
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

    def get_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))
        return pe  

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]
    


