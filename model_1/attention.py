import torch
from torch import nn
from math import sqrt

class DotAttention(nn.Module):
    def __init__(self, dim_x: int, dim_qk: int, dim_v: int):
        self.q = nn.Linear(dim_x, dim_qk)
        self.k = nn.Linear(dim_x, dim_qk)
        self.v = nn.Linear(dim_x, dim_v)
        self.norm = sqrt(dim_qk)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        d = torch.bmm(q, k.transpose(1, 2))/self.norm
        softmax_out = torch.softmax(d, dim=-1)

        attn_out = torch.bmm(softmax_out, v)
        return attn_out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, attention: nn.Module, num_heads: int):
        self.num_heads = num_heads
        self.head_dim = dim_qk // num_heads
        self.attention = attentiona
        self.fc = nn.Linear(dim_v, dim_x)

    def forward(self, x):
        attn_out = self.attention(x)
        attn_out = attn_out.view(-1, self.num_heads, self.head_dim)
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(-1, self.head_dim * self.num_heads)
        out = self.fc(attn_out)
        return out
    
class MultiLevelAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        