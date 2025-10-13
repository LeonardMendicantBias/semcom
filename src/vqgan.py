from typing import Optional, Union, List, Tuple, Any

import torch
from torch import nn

from torch.optim import lr_scheduler
import pytorch_lightning as pl

from src.transformer import VisionTransformer, VisionDecoder
from .quantizer import VectorQuantizer

from src.autoencoder import ViTEncoder as Encoder, ViTDecoder as Decoder


class ViTVQGAN(nn.Module):

    def __init__(self,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        depth: int, heads: int,
        dim: int, embed_dim: int,
        n_codes: int, 
        path: str=None
    ):
        super().__init__()
        self.path = path

        self.encoder = VisionTransformer(image_size, patch_size, depth, dim, heads, 4*dim)
        self.decoder = VisionDecoder(image_size, patch_size, depth, dim, heads, 4*dim)
        self.quantizer = VectorQuantizer(embed_dim, n_codes)
        self.pre_quant = nn.Linear(dim, embed_dim)
        self.post_quant = nn.Linear(embed_dim, dim)

        if path is not None:
            self.init_from_ckpt(path)

    @torch.no_grad()
    def init_from_ckpt(self, path: str):
        sd = torch.load(path)["state_dict"]
        for key, item in sd.items():
            if key.startswith("quantizer"):
                self.quantizer.embedding.weight.data.copy_(item)
            elif key.startswith("pre_quant"):
                if "weight" in key:
                    self.pre_quant.weight.data.copy_(item)
                elif "bias" in key:
                    self.pre_quant.bias.data.copy_(item)
            elif key.startswith("post_quant"):
                if "weight" in key:
                    self.post_quant.weight.data.copy_(item)
                elif "bias" in key:
                    self.post_quant.bias.data.copy_(item)
            elif key.startswith("encoder"):
                # Encoder
                if "transformer" in key:
                    if "layers" in key:
                        layer_idx = key.index("layers")
                        dot_idx = key.find(".", layer_idx+len("layers")+1)
                        idx = int(key[layer_idx+len("layers")+1:dot_idx])
                        if "0" in key[dot_idx:]:
                            if "norm" in key[dot_idx:]:
                                if "weight" in key[dot_idx:]:
                                    self.encoder.transformer[idx].attn_norm.weight.data.copy_(item)
                                if "bias" in key[dot_idx:]:
                                    self.encoder.transformer[idx].attn_norm.bias.data.copy_(item)
                            if "fn" in key[dot_idx:]:
                                if "to_qkv" in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        d = item.shape[1]
                                        self.encoder.transformer[idx].attn.query.weight.copy_(item[:d])
                                        self.encoder.transformer[idx].attn.key.weight.copy_(item[d:-d])
                                        self.encoder.transformer[idx].attn.value.weight.copy_(item[-d:])
                                    if "bias" in key[dot_idx:]: pass
                                if "to_out" in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        self.encoder.transformer[idx].attn.proj.weight.data.copy_(item)
                                    if "bias" in key[dot_idx:]:
                                        self.encoder.transformer[idx].attn.proj.bias.data.copy_(item)
                        if "1" in key[dot_idx:]:
                            if "norm" in key[dot_idx:]:
                                if "weight" in key[dot_idx:]:
                                    self.encoder.transformer[idx].mlp_norm.weight.copy_(item)
                                if "bias" in key[dot_idx:]:
                                    self.encoder.transformer[idx].mlp_norm.bias.copy_(item)
                            if "net" in key[dot_idx:]:
                                if ".net.0." in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        self.encoder.transformer[idx].mlp[0].weight.copy_(item)
                                    if "bias" in key[dot_idx:]:
                                        self.encoder.transformer[idx].mlp[0].bias.copy_(item)
                                if ".net.2." in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        self.encoder.transformer[idx].mlp[3].weight.copy_(item)
                                    if "bias" in key[dot_idx:]:
                                        self.encoder.transformer[idx].mlp[3].bias.copy_(item)
                    elif "encoder.transformer.norm" in key:
                        if "weight" in key:
                            self.encoder.norm.weight.data.copy_(item)
                        if "bias" in key:
                            self.encoder.norm.bias.data.copy_(item)
                elif "en_pos_embedding" in key:
                    self.encoder.en_pos_embedding.copy_(item)
                elif "to_patch_embedding" in key:
                    if "weight" in key:
                        self.encoder.to_patch_embedding[0].weight.copy_(item)
                    elif "bias" in key:
                        self.encoder.to_patch_embedding[0].bias.copy_(item)
            elif key.startswith("decoder"):
                if "de_pos_embedding" in key:
                    self.decoder.de_pos_embedding.copy_(item)
                elif "transformer" in key:
                    if "layers" in key:
                        layer_idx = key.index("layers")
                        dot_idx = key.find(".", layer_idx+len("layers")+1)
                        idx = int(key[layer_idx+len("layers")+1:dot_idx])
                        if "0" in key[dot_idx:]:
                            if "norm" in key[dot_idx:]:
                                if "weight" in key[dot_idx:]:
                                    self.decoder.transformer[idx].attn_norm.weight.data.copy_(item)
                                if "bias" in key[dot_idx:]:
                                    self.decoder.transformer[idx].attn_norm.bias.data.copy_(item)
                            if "fn" in key[dot_idx:]:
                                if "to_qkv" in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        d = item.shape[1]
                                        self.decoder.transformer[idx].attn.query.weight.copy_(item[:d])
                                        self.decoder.transformer[idx].attn.key.weight.copy_(item[d:-d])
                                        self.decoder.transformer[idx].attn.value.weight.copy_(item[-d:])
                                    if "bias" in key[dot_idx:]: pass
                                if "to_out" in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        self.decoder.transformer[idx].attn.proj.weight.data.copy_(item)
                                    if "bias" in key[dot_idx:]:
                                        self.decoder.transformer[idx].attn.proj.bias.data.copy_(item)
                        if "1" in key[dot_idx:]:
                            if "norm" in key[dot_idx:]:
                                if "weight" in key[dot_idx:]:
                                    self.decoder.transformer[idx].mlp_norm.weight.copy_(item)
                                if "bias" in key[dot_idx:]:
                                    self.decoder.transformer[idx].mlp_norm.bias.copy_(item)
                            if "net" in key[dot_idx:]:
                                if ".net.0." in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        self.decoder.transformer[idx].mlp[0].weight.copy_(item)
                                    if "bias" in key[dot_idx:]:
                                        self.decoder.transformer[idx].mlp[0].bias.copy_(item)
                                if ".net.2." in key[dot_idx:]:
                                    if "weight" in key[dot_idx:]:
                                        self.decoder.transformer[idx].mlp[3].weight.copy_(item)
                                    if "bias" in key[dot_idx:]:
                                        self.decoder.transformer[idx].mlp[3].bias.copy_(item)
                    elif "decoder.transformer.norm" in key:
                        if "weight" in key:
                            self.decoder.norm.weight.data.copy_(item)
                        if "bias" in key:
                            self.decoder.norm.bias.data.copy_(item)
                elif "to_pixel" in key:
                    if "weight" in key:
                        self.decoder.to_pixel[1].weight.copy_(item)
                    if "bias" in key:
                        self.decoder.to_pixel[1].bias.copy_(item)
            # else:
            #     print(key)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def encode(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        features = self.encoder(x, None)
        h = self.pre_quant(features)
        quant, emb_loss, _ = self.quantizer(h)

        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant, None)

        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        features = self.encoder(x, None)
        h = self.pre_quant(features)
        _, _, codes = self.quantizer(h)

        return features, codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)

        if self.quantizer.use_residual:
            quant = quant.sum(-2)

        dec = self.decode(quant)

        return dec
    
    @classmethod
    def get_vit_vqgan(cls, size: str):
        if size == "small":
            return cls.get_vit_vqgan_small()
        elif size == "base":
            return cls.get_vit_vqgan_base()
        else:
            raise ValueError 

    @classmethod
    def get_vit_vqgan_small(cls):
        return ViTVQGAN(
            image_size=256, patch_size=8,
            dim=512, depth=8, heads=8,
            n_codes=8192, embed_dim=32
        )
    
    @classmethod
    def get_vit_vqgan_base(cls):
        return ViTVQGAN(
            image_size=256, patch_size=8,
            dim=768, depth=12, heads=12,
            n_codes=8192, embed_dim=32
        )
