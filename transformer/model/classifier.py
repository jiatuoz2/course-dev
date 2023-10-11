import torch
import torch.nn as nn
import math
from encoder import Encoder

class Classifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, input_vocab_size, output_vocab_size, max_seq_length, dropout=0.1):
        super(Classifier, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ffn, input_vocab_size, max_seq_length, dropout)
        self.output_projection = nn.Linear(d_model, output_vocab_size)
    
    def forward(self, input_seq, src_mask=None):
        enc_output = self.encoder(input_seq)
        output = self.output_projection(enc_output)
        return output