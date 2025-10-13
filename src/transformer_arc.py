from typing import Union, Tuple

import math
import numpy as np

import torch
from torch import nn

from einops import rearrange, repeat

from src.utils import init_weights


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    
    return emb


class PositionalEmbedding(nn.Module):
    
    def __init__(self, length, height, width, dim,):
        super().__init__()
        self.length = length
        self.height = height
        self.width = width
        
        initializer_range = 0.02
        self.cls_pos_embedding = nn.Parameter(initializer_range * torch.randn(1, dim))
        initializer_range = initializer_range / math.sqrt(3)
        self.temporal_pos_embedding = nn.Parameter(torch.empty(1, length, dim).normal_(std=initializer_range))
        self.height_pos_embedding = nn.Parameter(torch.empty(1, height, dim).normal_(std=initializer_range))
        self.width_pos_embedding = nn.Parameter(torch.empty(1, width, dim).normal_(std=initializer_range))

    def forward(self):
        pos_embedding = torch.reshape(
            input=(
                self.temporal_pos_embedding.reshape(self.length, 1, 1, -1) +
                self.height_pos_embedding.reshape(1, self.height, 1, -1) +
                self.width_pos_embedding.reshape(1, 1, self.width, -1)
            ),
            shape=(self.length * self.height * self.width, -1)
        )
        return torch.cat([self.cls_pos_embedding, pos_embedding])
        

class PreNorm(nn.Module):

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
    

class TransformerBlock(nn.Module):
    
    def __init__(self,
        dim: int, heads: int, dim_head: int, mlp_dim: int
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.attn_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return


class Transformer(nn.Module):

    def __init__(self,
        dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim)),
            ])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class SpatioTemporalBlock(nn.Module):
    
    def __init__(self,
        dim: int, heads: int, dim_head: int, mlp_dim: int
    ) -> None:
        super().__init__()
        self.temporal_attn_norm = nn.LayerNorm(dim)
        self.temporal_attn = Attention(dim, heads=heads, dim_head=dim_head)

        self.spatial_attn_norm = nn.LayerNorm(dim)
        self.spatial_attn = Attention(dim, heads=heads, dim_head=dim_head)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim))
        
    def forward(self,
        x: torch.Tensor,   # (B, L, H, W)
    ) -> torch.Tensor:
        B, L, H, W = x.shape

        x = self.temporal_attn(self.temporal_attn_norm(x)) + x

        x = self.spatial_attn(self.spatial_attn_norm(x)) + x

        x = self.ffn(self.ffn_norm(x)) + x
        return


class TransformerLayout(nn.Module):
    def __init__(self,
        length: int, height: int, width: int,
        vocab_size: int,
        dim: int,
        layers: int=12,
        heads: int=8,
        dropout: float=0.1,
    ):
        super().__init__()
        self.length = length
        self.height = height
        self.width = width

        self.embedding = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)

        # self.class_param = nn.Parameter(self.initializer_range * torch.randn(dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional_embedding = PositionalEmbedding(height, width, length, dim)

        self.ln_pre = nn.LayerNorm(dim, eps=1e-12)
        self.transformer = Transformer(dim, layers, heads, dim//heads, 2*dim)

        self.cls_pos_emb = nn.Parameter(torch.empty(1, dim).normal_(std=0.02))
        self.temporal_pos_emb = nn.Parameter(torch.empty(1, length, dim).normal_(std=0.02))
        self.height_pos_emb = nn.Parameter(torch.empty(1, height, dim).normal_(std=0.02))
        self.width_pos_emb = nn.Parameter(torch.empty(1, width, dim).normal_(std=0.02))

        # Weight initialization
        self.apply(init_weights)

    def forward(self,
        token: torch.Tensor,  # (B, L, H, W)
    ):
        B, L, H, W = token.shape

        # mask = (token == PAD_TOKEN_ID)      # B, L, H, W
        # if (~mask).all():
        #     mask = None

        x = self.embedding(token)  # B, L, H, W --> B, L, H, W, Dim
        class_token = self.class_token.expand(B, -1, -1, -1)
        x = torch.cat([x, class_token], dim=1)

        pos_emb = torch.cat([
            self.cls_pos_emb,
            self.temporal_pos_emb[:L].reshape(L, 1, 1, -1),
            self.height_pos_emb[:H].reshape(1, H, 1, -1),
            self.width_pos_emb[:W].reshape(1, 1, W, -1),
        ])
        x = x + pos_emb

        x = self.transformer(x)

        return x