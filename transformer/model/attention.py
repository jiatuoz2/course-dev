import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        
        if mask is not None:
            matmul_qk = matmul_qk.masked_fill(mask == 0, float("-1e20"))
        
        scaled_attention = torch.nn.functional.softmax(matmul_qk / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), dim=-1)
        output = torch.matmul(scaled_attention, value)
        
        return output
    
    def split_heads(self, tensor, batch_size):
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        attention_output = self.scaled_dot_product_attention(query, key, value, mask)
        
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.fc_out(attention_output)
        output = self.dropout(output)
        
        return output
