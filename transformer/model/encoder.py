import torch
import torch.nn as nn 
from attention import MultiHeadAttention
from feed4wd import FeedForward
from norm import LayerNormalization
from embed import WordEmbedding
from embed import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ffn)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.multihead_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.feedforward(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.layer_norm2(x + ffn_output)

        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, vocab_size, max_seq_length, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = WordEmbedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_seq_length, d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, src, mask=None):
        x = self.embedding(src)
        x = self.pe(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x
