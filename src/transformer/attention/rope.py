from typing import Optional

import torch
from torch import nn

from .self_attention import MultiHeadAttention


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