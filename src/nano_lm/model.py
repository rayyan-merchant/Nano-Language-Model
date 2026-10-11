"""Pre-LayerNorm decoder-only GPT, constructed from basic PyTorch layers."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import MultiHeadAttention
from .positions import sinusoidal


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        activation = nn.GELU if config.activation == "gelu" else nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_model * config.mlp_expansion),
                                 activation(), nn.Linear(config.d_model * config.mlp_expansion,
                                                        config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attended, weights = self.attention(self.norm1(x))
        x = x + self.dropout(attended)
        return x + self.dropout(self.mlp(self.norm2(x))), weights


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (nn.Embedding(config.context_length, config.d_model)
                                   if config.position == "learned" else None)
        self.register_buffer("sinusoidal", sinusoidal(config.context_length, config.d_model),
                             persistent=False)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, ids, targets=None, return_attention=False):
        length = ids.shape[1]
        if not 0 < length <= self.config.context_length:
            raise ValueError("Sequence outside configured context")
        x = self.token_embedding(ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(length, device=ids.device))
        elif self.config.position == "sinusoidal":
            x = x + self.sinusoidal[:length].to(x.dtype)
        x = self.dropout(x)
        attention = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attention:
                attention.append(weights)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss, attention

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
