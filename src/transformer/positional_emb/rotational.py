from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class RoPE(nn.Module):

    def __init__(self,
        length: int, dim: int,
        drop_prob: float=0.1,
    ):
        super().__init__()
        
