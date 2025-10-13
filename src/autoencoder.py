from typing import Union, Tuple

import torch
from torch import nn

import numpy as np

from einops.layers.torch import Rearrange

from src.transformer_arc import get_2d_sincos_pos_embed, Transformer
from src.utils import init_weights


class ViTEncoder(nn.Module):

    def __init__(self,
        image_size: Union[Tuple[int, int], int],
        patch_size: Union[Tuple[int, int], int],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        image_height, image_width = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        patch_height, patch_width = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        en_pos_embedding = get_2d_sincos_pos_embed(
            dim, (image_height // patch_height, image_width // patch_width)
        )

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.en_pos_embedding = nn.Parameter(
            torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x = x + self.en_pos_embedding
        x = self.transformer(x)

        return x


class ViTDecoder(nn.Module):
    
    def __init__(
        self,
        image_size: Union[Tuple[int, int], int],
        patch_size: Union[Tuple[int, int], int],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int=3,
        dim_head: int=64,
    ) -> None:
        super().__init__()
        image_height, image_width = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        patch_height, patch_width = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        de_pos_embedding = get_2d_sincos_pos_embed(
            dim, (image_height // patch_height, image_width // patch_width)
        )

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(
            torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False
        )
        self.to_pixel = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=image_height // patch_height),
            nn.ConvTranspose2d(
                dim, channels, kernel_size=patch_size, stride=patch_size
            ),
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        x = self.to_pixel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight
