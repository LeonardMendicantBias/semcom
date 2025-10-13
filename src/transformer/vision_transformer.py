from typing import Union, Tuple

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from .base import Block

# from src import utils


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


class VisionTransformer(nn.Module):

    def __init__(self,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        depth: int,
        dim: int, heads: int, mlp_dim: int,
        drop_prob:float=0.1, depth_prob: float=0.1
    ):
        super().__init__()
        image_height, image_width = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        patch_height, patch_width = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )

        en_pos_embedding = get_2d_sincos_pos_embed(
            dim, (image_height // patch_height, image_width // patch_width)
        )
        self.en_pos_embedding = nn.Parameter(
            torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False
        )

        self.transformer = nn.ModuleList([
            Block(dim, mlp_dim, heads, nn.Tanh, drop_prob=drop_prob, depth_prob=depth_prob*i/depth)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self,
        x: torch.FloatTensor,
        mask: torch.BoolTensor,
    ):
        x = self.to_patch_embedding(x)

        x = x + self.en_pos_embedding

        for module in self.transformer:
            x, attn = module(x, x, mask)
        return self.norm(x)


class VisionDecoder(nn.Module):

    def __init__(self,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        depth: int,
        dim: int, heads: int, mlp_dim: int,
        drop_prob:float=0.1, depth_prob: float=0.1
    ):
        super().__init__()
        image_height, image_width = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        patch_height, patch_width = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )

        de_pos_embedding = get_2d_sincos_pos_embed(
            dim, (image_height // patch_height, image_width // patch_width)
        )
        self.de_pos_embedding = nn.Parameter(
            torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False
        )

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = 3 * patch_height * patch_width

        self.transformer = nn.ModuleList([
            Block(dim, mlp_dim, heads, nn.Tanh, drop_prob=drop_prob, depth_prob=depth_prob*i/depth)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        self.to_pixel = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=image_height // patch_height),
            nn.ConvTranspose2d(
                dim, 3, kernel_size=patch_size, stride=patch_size
            ),
        )

        # self.apply(utils.init_weights)

    def forward(self,
        token: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        for module in self.transformer:
            x, attn = module(x, x, mask)
        x = self.norm(x)
        x = self.to_pixel(x)
        return x
