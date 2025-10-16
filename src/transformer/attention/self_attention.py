# %%
from typing import Optional

import torch
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self,
        dim: int, heads: int,
        bias: bool=False, prob: float=0.1,
        cache_size: int=-1,
    ) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.cache_size = cache_size

        self.dim = dim
        self.n_heads = heads
        self.head_dim = dim // heads

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(prob)

        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(prob)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

    def _calculate_qkv(self,
        query: torch.FloatTensor, memory: torch.FloatTensor,
        use_cache: bool=False
    ):
        B, T, _ = query.shape
        _, S, _ = memory.shape

        q = self.query(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # print("-"*30)
        # print("kv", k.shape, v.shape)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)

            if self.cache_size != -1 and self.cache_k.shape[1] > self.cache_size:
                self.cache_k = self.cache_k[:, -self.cache_size:]
                self.cache_v = self.cache_v[:, -self.cache_size:]

            k = self.cache_k
            v = self.cache_v
        # print("kv", k.shape, v.shape)

        return q, k, v
    
    def _calculate_logits(self, 
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        mask: Optional[torch.BoolTensor],
    ) -> torch.FloatTensor:
        # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)
        logits = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim)**-0.5  # (B, H, T, S)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            logits = logits.masked_fill(mask, -1e4)  # more stable than -torch.inf

        return logits
    
    def _calculate_attention(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        attn = logits.softmax(-1)
        attn = self.attn_dropout(attn)
        return attn

    def forward(self,
        query: torch.FloatTensor, memory: torch.FloatTensor,
        mask: torch.BoolTensor,  # (B, T, S)
        use_cache: bool=False
    ) -> torch.FloatTensor:
        B, T, _ = query.shape
        _, S, _ = memory.shape
        
        q, k, v = self._calculate_qkv(query, memory, use_cache)

        logits = self._calculate_logits(q, k, mask)  # (B, H, T, S)

        attn = self._calculate_attention(logits)
        attn = self.attn_dropout(attn)

        weighted = torch.matmul(attn, v)
        weighted = weighted.transpose(1, 2).contiguous().view(B, T, self.dim)

        out = self.proj(weighted)

        return self.proj_dropout(out), logits


if __name__ == "__main__":
    # demonstrating usage

    d, n_h = 64, 8
    mha = MultiHeadAttention(d, n_h)

