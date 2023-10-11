import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed4wd import FeedForward
from norm import LayerNormalization
from embed import WordEmbedding
from embed import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention2 = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ffn)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.layer_norm3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, target_mask, src_mask):
        # Self-attention
        attn_output1 = self.multihead_attention1(x, x, x, target_mask)
        x = self.layer_norm1(x + attn_output1)

        # Cross-attention
        attn_output2 = self.multihead_attention2(x, enc_output, enc_output)
        x = self.layer_norm2(x + attn_output2)

        # Feed-forward
        ffn_output = self.feedforward(x)
        ffn_output = self.dropout(ffn_output)
        x = self.layer_norm3(x + ffn_output)

        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, vocab_size, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = WordEmbedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_seq_length, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, trg, enc_output, target_mask, src_mask):
        x = self.embedding(trg)
        x = self.pe(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, target_mask, src_mask)
        
        return x
