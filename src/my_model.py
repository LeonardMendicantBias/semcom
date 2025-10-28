from typing import Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# from .vqgan import ViTVQGAN
from .quantizer import VectorQuantizer
from .transformer import TemporalBlock, TemporalTransformer, VisionTransformer, VisionDecoder
from .vqgan import ViTVQGAN


class Encoder(nn.Module):
	
	def __init__(self,
		vit: nn.Module,
		pre_quant: nn.Linear,
		quantizer: VectorQuantizer,
		temporal_transformer: nn.Module
	):
		super().__init__()
		self.vit = vit
		self.pre_quant = pre_quant
		self.quantizer = quantizer
		self.transformer = temporal_transformer
	
	@torch.no_grad()
	def get_mask_from_logits(self,
		logits: torch.FloatTensor,  # (B, T, 3, _, _)
		K: int=None,
	) -> torch.BoolTensor:
		B, HW, _, T_, _ = logits.shape
		
		# which tokens are salient according to the cls token
		cls_logits = logits[:, :, :, 0, 1:]  # (B, HW, n_heads, T)
		cls_logits = cls_logits.permute(0, 2, 3, 1).flatten(-2, -1)  # (B, n_heads, T*HW)
		# average over all heads
		attn = cls_logits.softmax(-1).mean(1)
		# if self.training:
		#     attn = F.dropout(attn, 0.1)
		
		# stat for top-p
		sorted_logits, sorted_indices = torch.sort(attn, descending=True)
		cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

		prob = cumulative_probs > self.top_p
		prob[..., 1:] = prob[..., :-1].clone()
		prob[..., 0] = False
		
		mask = ~prob.scatter(
			dim=-1,
			index=sorted_indices,
			src=prob
		).reshape((B, T_-1, HW))
		return mask.to(torch.long)
	
	def process_frames(self, frames: torch.FloatTensor) -> torch.LongTensor:
		B, T = frames.shape[:2]  # T = 1 during inference

		# process each frame
		features = self.vit(frames.flatten(0, 1), None)  # (BT, HW, D)
		h = self.pre_quant(features)
		_, _, codes = self.quantizer(h)
		HW = codes.shape[-1]
		return codes.reshape(B, T, HW)  # (B, T, HW)
	
	def forward(self,
		frames: torch.FloatTensor,  # (B, T, 3, H', W'), T should be 1
	) -> Tuple[torch.LongTensor, torch.BoolTensor]:
		codes = self.process_frames(frames)

		_, attn_logits = self.transformer(codes, None, use_cache=True)

		return codes[:, -1], self.get_mask_from_codes(attn_logits)[:, -1]


class Decoder(nn.Module):
	
	def __init__(self,
		temporal_transformer: nn.Module,
		decoder: nn.Module,
	):
		super().__init__()
		self.transformer = temporal_transformer
		self.decoder = decoder

	def forward(self,
		code: torch.LongTensor,  # (B, T, 3, H', W')
		token_mask: torch.LongTensor=None,
		use_cache: bool=False
	) -> Tuple[torch.LongTensor, torch.BoolTensor]:
		output = self.transformer(code, token_mask, use_cache)

		return output


class MaskVideo(nn.Module):

	def __init__(self,
		# setting for vit
		is_finetune: bool,
		image_size: Union[int, Tuple[int, int]],
		patch_size: Union[int, Tuple[int, int]],
		vit_depth: int, vit_heads: int, vit_dim: int,
		n_codes: int, embed_dim: int,
		# setting for transformer
		window_size: Union[int, Tuple[int, int]],
		length: int, height: int, width: int,
		temporal_depth: int, temporal_heads: int, temporal_dim: int,
		
		drop_prob:float=0.1, depth_prob: float=0.1,
	):
		super().__init__()
		self.is_finetune = is_finetune
		self.image_size, self.patch_size = image_size, patch_size
		self.n_codes = n_codes

		self.vit = VisionTransformer(
			image_size=image_size, patch_size=patch_size,
			depth=vit_depth, heads=vit_heads,
			dim=vit_dim, mlp_dim=4*vit_dim,
			drop_prob=drop_prob, depth_prob=depth_prob
		)
		self.pre_quant = nn.Linear(vit_dim, embed_dim)
		self.quantizer = VectorQuantizer(embed_dim, n_codes)

		self.transformer = TemporalTransformer(
			window_size=window_size,
			length=length, height=height, width=width,
			depth=temporal_depth, n_heads=temporal_heads, dim=temporal_dim,
			n_codes=n_codes,
			drop_prob=drop_prob, depth_prob=depth_prob
		)

		self.code_head = nn.Linear(temporal_dim, n_codes)

		if not is_finetune:
			for param in self.vit.parameters():
				param.requires_grad = False
			for param in self.pre_quant.parameters():
				param.requires_grad = False
			for param in self.quantizer.parameters():
				param.requires_grad = False
		else: pass  # implement LoRA

	def export_encoder(self) -> None:
		encoder = Encoder()
		dummy_input = torch.randn(1, 1, 3, 256, 256)
		torch.onnx.export(encoder, (dummy_input, None, True), "encoder.onnx", export_params=True)
		print("ONNX model saved: encoder.onnx")
	
	def export_decoder(self) -> None:
		decoder = Decoder()
		dummy_input = torch.randn(1, 16, 1024)
		dummy_mask = torch.randn(1, 16, 1024)
		torch.onnx.export(decoder, (dummy_input, dummy_mask, True), "decoder.onnx", export_params=True)
		print("ONNX model saved: decoder.onnx")
	
	def reset_cache(self):
		self.current_pos = 1
		self.transformer.reset_cache()

	def process_frames(self, frames: torch.FloatTensor) -> torch.LongTensor:
		B, T = frames.shape[:2]  # T = 1 during inference

		# process each frame
		features = self.vit(frames.flatten(0, 1), None)  # (BT, HW, D)
		h = self.pre_quant(features)
		_, _, codes = self.quantizer(h)
		HW = codes.shape[-1]
		return codes.reshape(B, T, HW)  # (B, T, HW)

	@torch.no_grad()
	def get_mask_from_frames(self,
		frames: torch.FloatTensor,  # (B, T, 3, _, _)
		top_p: float,
		K: int=None,
		use_cache: bool=False,
	) -> torch.BoolTensor:
		codes = self.process_frames(frames)
		_, attn_logits = self.transformer(codes, None, use_cache)
		return self.transformer.get_mask_from_logits(attn_logits)
	
	def forward(self,
		frames: torch.FloatTensor,
		token_mask: torch.LongTensor=None,
		use_cache: bool=False
	) -> Tuple[torch.BoolTensor, torch.LongTensor, torch.FloatTensor]:
		codes = self.process_frames(frames)

		x, attn_logits = self.transformer(codes, token_mask, use_cache)

		dec = self.code_head(x)

		return codes, x, attn_logits, dec
	