from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class RelativePositionalEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
        x: torch.FloatTensor, 
        pos: torch.LongTensor=None,
    ) -> torch.FloatTensor:
        return x