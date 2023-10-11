import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, input_vocab_size, output_vocab_size, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ffn, input_vocab_size, max_seq_length, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ffn, output_vocab_size, max_seq_length, dropout)
        self.output_projection = nn.Linear(d_model, output_vocab_size)
    
    def forward(self, input_seq, target_seq, target_mask, src_mask=None):
        enc_output = self.encoder(input_seq)
        dec_output = self.decoder(target_seq, enc_output, target_mask, src_mask)
        output = self.output_projection(dec_output)
        return output