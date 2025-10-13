from typing import Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# from .vqgan import ViTVQGAN
from .transformer import TemporalBlock
from .vqgan import ViTVQGAN


class MaskCode(nn.Module):
    """
        find the salient codes among frames
    """

    def __init__(self,
        window_size: Union[int, Tuple[int, int]],
        length: int, height: int, width: int,
        depth: int, heads: int, dim: int, 
        n_codes: int, embed_dim: int,
        drop_prob:float=0.1, depth_prob: float=0.1,
        path: str=None
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.length, self.height, self.width = length, height, width
        self.n_codes = n_codes

        self.embedding = nn.Embedding(n_codes, dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1, dim).normal_(std=0.02))
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, dim).normal_(std=0.02))

        # including the CLS token
        self.t_emb = nn.Parameter(torch.empty(1, length+1, 1, dim).normal_(std=0.02))
        self.h_emb = nn.Parameter(torch.empty(1, 1, height, dim).normal_(std=0.02))
        self.w_emb = nn.Parameter(torch.empty(1, 1, width, dim).normal_(std=0.02))
        self.emb_norm = nn.LayerNorm(dim)

        self.transformer = nn.ModuleList([
            TemporalBlock(
                window_size,
                height, width,
                dim, 2*dim, heads,
                activation=nn.Tanh,
                drop_prob=drop_prob, depth_prob=depth_prob*i/depth
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        self.code_head = nn.Linear(dim, n_codes)

    def create_temporal_mask(self,
        T: int,
        device: torch.device=torch.device("cpu")
    ) -> torch.BoolTensor:
        mask = torch.triu(torch.ones(T+1, T+1, dtype=torch.bool, device=device), diagonal=1)
        mask[0, :] = False
        return mask.unsqueeze(0)  # account for token and batch dimension

    def forward(self,
        code: torch.LongTensor,        # (B, T, HW)
        token_mask: torch.LongTensor,   # (B, T, HW)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        B, T, HW = code.shape

        temporal_mask = self.create_temporal_mask(T, code.device)

        emb = self.embedding(code)  # (B, T, HW, D)
        # detach ?
        if token_mask is not None:
            emb = (1-token_mask).unsqueeze(-1)*self.mask_token + (token_mask.unsqueeze(-1)*emb)
        
        # 1, 2, ..., n, 1, 2, ..., n, ...
        _h_emb = self.h_emb.repeat((1, 1, self.width, 1))
        # 1, 1, ..., 1, 2, 2, ..., 2, ...
        _w_emb = self.w_emb.repeat_interleave(self.height, dim=2)
        pos_emb = self.t_emb[:, 1:T+1] + _h_emb + _w_emb  # 0-th is for cls token
        x = emb + pos_emb

        # always include 0-th pos emb for CLS token
        cls_token = self.cls_token + self.t_emb[:, 0:1]
        cls_token = cls_token.repeat((B, 1, HW, 1))
        x = torch.cat([cls_token, x], dim=1)  # (B, T', HW, D)
        x = self.emb_norm(x)

        for layer in self.transformer:
            x, attn = layer(x, None, temporal_mask)

        logits = self.code_head(self.norm(x))

        return logits, attn


class MaskVideo(MaskCode):
    
    def __init__(self,
        window_size: Union[int, Tuple[int, int]],
        length: int, height: int, width: int,
        depth: int, heads: int, dim: int, 
        n_codes: int, embed_dim: int,
        drop_prob:float=0.1, depth_prob: float=0.1,
        path: str=None, vitvq_path: str=None
    ):
        super().__init__(
            window_size,
            length, height, width,
            depth, heads, dim,
            n_codes, embed_dim,
            drop_prob, depth_prob,
            path
        )
        vitvq = ViTVQGAN(
            image_size=256, patch_size=8,
            dim=768, depth=12, heads=12,
            n_codes=8192, embed_dim=32
        )
        if vitvq_path is not None:
            vitvq.init_from_ckpt(vitvq_path)
            
    def forward(self,
        frames: torch.FloatTensor,  # (B, T, C, H, W)
    ):
        code = self.vitvq.get_code(frames)  # (B, T, HW)
        return super().forward(code)

