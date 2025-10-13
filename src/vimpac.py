from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F

from src.autoencoder import ViTEncoder as Encoder, ViTDecoder as Decoder
from src.quantizer import VectorQuantizer


class MyPAC(nn.Module):

    def __init__(
        self,
        image_size, patch_size,
        dim, mlp_dim, depth, heads,
        n_codes, quant_dim,
        checkpoint: Optional[str]=None,
        ignore_keys: List[str]=list(),
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            image_size=image_size, patch_size=patch_size,
            dim=dim, mlp_dim=mlp_dim,
            depth=depth, heads=heads,
        )
        self.decoder = Decoder(
            image_size=image_size, patch_size=patch_size,
            dim=dim, mlp_dim=mlp_dim,
            depth=depth, heads=heads,
        )
        self.init_vitvq(checkpoint, ignore_keys)

        self.quantizer = VectorQuantizer(quant_dim, n_codes)
        self.pre_quant = nn.Linear(dim, quant_dim)
        self.post_quant = nn.Linear(quant_dim, dim)
        
        self.emb = nn.Embedding(self.n_embed, self.embed_dim)
        self.t_emb = nn.Parameter()
        self.h_emb = nn.Parameter()
        self.w_emb = nn.Parameter()
        # self.temporal_transformer = nn.Module([

        # ])

    def process_frame(self,
        x: torch.FloatTensor,           
    ) -> torch.FloatTensor:
        pass

    def forward(self, xs: torch.FloatTensor) -> None:
        pass


class LightningPAC(nn.Module):

    def __init__(
        self,
        image_size, patch_size,
        dim, mlp_dim, depth, heads,
        n_codes, quant_dim,
        checkpoint: Optional[str]=None,
        ignore_keys: List[str]=list(),
    ) -> None:
        super().__init__(
            image_size, patch_size,
            dim, mlp_dim, depth, heads,
            n_codes, quant_dim,
            checkpoint, ignore_keys
        )

        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, x, batch_idx):
        # calculate attention
        # apply mask
        # calculate logits
        # calculate loss
        pass

    def validation_step(self, x, batch_idx): pass
