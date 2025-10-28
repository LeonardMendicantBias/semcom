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
        self.bias = bias
        self.prob = prob

        self.head_dim = dim // heads

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(prob)

        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(prob)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    # def build_query(self) -> nn.Module:
    #     return nn.Linear(self.dim, self.dim, bias=self.bias)
    
    # def build_key(self) -> nn.Module:
    #     return nn.Linear(self.dim, self.dim, bias=self.bias)
    
    # def build_value(self) -> nn.Module:
    #     return nn.Linear(self.dim, self.dim, bias=self.bias)
    
    # def build_proj(self) -> nn.Module:
    #     return nn.Linear(self.dim, self.dim)

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
        
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k.detach(), k[:, :, -1:, :]], dim=2)
                self.cache_v = torch.cat([self.cache_v.detach(), v[:, :, -1:, :]], dim=2)

            if self.cache_size != -1 and self.cache_k.shape[1] > self.cache_size:
                self.cache_k = self.cache_k[:, -self.cache_size:]
                self.cache_v = self.cache_v[:, -self.cache_size:]

            k = self.cache_k
            v = self.cache_v

        return q, k, v
    
    def _calculate_logits(self, 
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        mask: Optional[torch.BoolTensor],
    ) -> torch.FloatTensor:
        # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)
        logits = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim)**-0.5  # (B, H, T, S)
        
        if mask is not None:
            # unsqueeze for head dimension
            if mask.ndim == 2: mask = mask.unsqueeze(0)#.repeat(self.n_heads, 1, 1)
            # unsqueeze for batch dimension (when mask is done per head)
            if mask.ndim == 3: mask = mask.unsqueeze(0)
            # mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            logits = logits.masked_fill(mask.to(torch.bool), -1e4)  # more stable than -torch.inf

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
        q, k, v = self._calculate_qkv(query, memory, use_cache)

        logits = self._calculate_logits(q, k, mask)  # (B, H, T, S)

        attn = self._calculate_attention(logits)
        attn = self.attn_dropout(attn)

        weighted = torch.matmul(attn, v)
        weighted = weighted.transpose(1, 2).contiguous().flatten(-2, -1)

        out = self.proj(weighted)

        return self.proj_dropout(out), logits


class RoPEAttention(MultiHeadAttention):
    
    def __init__(self,
        *args,
        **kwargs, 
    ) -> None:
        super().__init__(*args, **kwargs)

        assert self.head_dim % 2 == 0, "dim//n_heads must be even"
        
        # initialize RoPE
        base, seq_length = 10000, 129
        theta = base ** (torch.arange(self.head_dim/2, dtype=torch.float) / self.head_dim)
        idx_theta = torch.einsum('n,d->nd', torch.arange(seq_length), theta)

        # [1, 2, ..., n] -> [1, 1, 2, 2, ..., n, n]
        idx_theta = idx_theta.repeat_interleave(2, dim=-1)

        # accomodate batch and head dimensions
        self.register_buffer("cos", idx_theta.cos()[None, None, :, :])
        self.register_buffer("sin", idx_theta.sin()[None, None, :, :])
    
    def apply_rope(self, x, pos_idx: int=None):
        if pos_idx == None:
            pos_idx = x.shape[2]
        
        inv_x = torch.cat([
            -x[..., 1::2],
            x[..., ::2]
        ], dim=-1)

        return self.cos[:, :, :pos_idx] * x + self.sin[:, :, :pos_idx] * inv_x

    def _calculate_logits(self, 
        query: torch.FloatTensor,   # (B, H, T, D')
        key: torch.FloatTensor,     # (B, H, S, D')
        mask: Optional[torch.BoolTensor],
    ) -> torch.FloatTensor:
        # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)
        query = self.apply_rope(query)
        key = self.apply_rope(key)
        
        return super()._calculate_logits(query, key, mask)


if __name__ == "__main__":
    # demonstrating usage

    d, n_h = 64, 8
    mha = MultiHeadAttention(d, n_h)

