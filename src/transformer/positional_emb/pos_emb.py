from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpatialTemporalPositionalEmbedding(nn.Module):

    def __init__(self,
        length: int, height: int, width: int,
        dim: int,
        drop_prob: float=0.1,
    ):
        super().__init__()

        self.t_emb = nn.Parameter(torch.empty(1, length+1, 1, dim).normal_(std=0.02))
        self.h_emb = nn.Parameter(torch.empty(1, 1, height, dim).normal_(std=0.02))
        self.w_emb = nn.Parameter(torch.empty(1, 1, width, dim).normal_(std=0.02))

        self.drop = nn.Dropout(drop_prob)

    def forward(self,
        x: torch.FloatTensor,  # (B, T', HW, D)
    ) -> torch.FloatTensor:
        B, T_, HW, D = x.shape
        
        # 1, 2, ..., n, 1, 2, ..., n, ...
        _h_emb = self.h_emb.repeat((1, 1, self.width, 1))
        # 1, 1, ..., 1, 2, 2, ..., 2, ...
        _w_emb = self.w_emb.repeat_interleave(self.height, dim=2)

        pos_emb = self.t_emb[:, 1:T_-1] + _h_emb + _w_emb

        x[:, 1:] = x[:, 1:] + pos_emb
        x[:, 0:1] = x[:, 0:1] + self.t_emb[:, 0:1]

        return self.drop(x)
