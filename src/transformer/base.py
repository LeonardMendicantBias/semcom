from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MLP, StochasticDepth

from .attention import MultiHeadAttention


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


# class FeedForwardNetwork(nn.Module):

#     def __init__(self,
#         dim: int, 
#     ):
#         super().__init__()
#         self.net 


class Block(nn.Module):
    
    def __init__(self,
        dim: int, mlp_dim: int,
        heads: int,
        activation,
        drop_prob: float=0.1, depth_prob: float=0.0
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim)
        self.attn: MultiHeadAttention = MultiHeadAttention(dim, heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp: nn.Module = MLP(
            dim, [mlp_dim, dim],
            dropout=drop_prob,
            activation_layer=activation #nn.Tanh
        )
        self.drop_path = StochasticDepth(depth_prob, mode="row")

    def reset_cache(self):
        self.attn.reset_cache()

    def forward(self,
        query: torch.FloatTensor,
        memory: torch.FloatTensor,
        mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        x_norm = self.attn_norm(query)
        mem_norm = self.attn_norm(memory)
        x, attn_logits, k, v = self.attn(x_norm, mem_norm, mask)
        x = self.drop_path(x) + query

        x_norm = self.mlp_norm(x)
        x = self.drop_path(self.mlp(x_norm)) + x
        
        return x, attn_logits, k, v
    

class Transformer(nn.Module):

    def __init__(self, 
        depth: int, 
        dim: int, mlp_dim: int,
        heads: int
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, 4*dim, heads)
            for _ in range(depth)
        ])

    def forward(self, x):
        pass
