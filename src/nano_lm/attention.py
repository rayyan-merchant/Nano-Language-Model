"""Attention math and head split/merge are deliberately explicit."""
import math
import torch
from torch import nn
from .positions import apply_rope


def scaled_attention(q, k, v, causal=True, dropout=None):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        future = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    probabilities = scores.softmax(dim=-1)
    weights = dropout(probabilities) if dropout is not None else probabilities
    return weights @ v, probabilities


def split_heads(x, heads):
    batch, length, width = x.shape
    return x.reshape(batch, length, heads, width // heads).transpose(1, 2)


def merge_heads(x):
    batch, heads, length, width = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, length, heads * width)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.rope = config.n_heads, config.position == "rope"
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        q, k, v = [split_heads(layer(x), self.heads) for layer in (self.q, self.k, self.v)]
        if self.rope:
            q, k = apply_rope(q), apply_rope(k)
        attended, probabilities = scaled_attention(q, k, v, dropout=self.dropout)
        return self.output(merge_heads(attended)), probabilities
