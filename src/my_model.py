from typing import Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# from .vqgan import ViTVQGAN
from .quantizer import VectorQuantizer
from .transformer import TemporalBlock, VisionTransformer, VisionDecoder
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
				drop_prob=drop_prob, depth_prob=depth_prob*i/depth,
				cache_size=length,
			) for i in range(depth)
		])

		self.norm = nn.LayerNorm(dim)

		self.code_head = nn.Linear(dim, n_codes)
		# self.current_pos = 0
	
	def reset_cache(self):
		# self.current_pos = 0
		for block in self.transformer:
			block.reset_cache()

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
		use_cache: bool=False
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
			x, attn = layer(x, None, temporal_mask, use_cache)

		logits = self.code_head(self.norm(x))
		
		return logits, attn


class Encoder(nn.Module):

	def __init__(self,
		is_finetune: bool,
		image_size: Union[int, Tuple[int, int]],
		patch_size: Union[int, Tuple[int, int]],
		depth: int, dim: int, heads: int,
		n_codes: int, embed_dim: int,

		window_size: Union[int, Tuple[int, int]],
		length: int, height: int, width: int,
		temporal_depth: int, temporal_heads: int, temporal_dim: int, 

		drop_prob:float=0.1, depth_prob: float=0.1,
		vitvq_path: str=None
	):
		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.length, self.height, self.width = length, height, width
		self.n_codes = n_codes

		# ViT-VQGAN
		self.vit = VisionTransformer(
			image_size=image_size, patch_size=patch_size,
			depth=depth, heads=heads, dim=dim, mlp_dim=4*dim,
			# n_codes=n_codes, embed_dim=embed_dim,
			drop_prob=drop_prob, depth_prob=depth_prob
		)
		self.pre_quant = nn.Linear(dim, embed_dim)
		self.quantizer = VectorQuantizer(embed_dim, n_codes)

		# Temporal Transformer
		self.embedding = nn.Embedding(n_codes, temporal_dim)
		self.cls_token = nn.Parameter(torch.empty(1, 1, 1, temporal_dim).normal_(std=0.02))
		self.mask_token = nn.Parameter(torch.empty(1, 1, 1, temporal_dim).normal_(std=0.02))
		self.emb_norm = nn.LayerNorm(temporal_dim)

		# positional embedding
		self.h_emb = nn.Parameter(torch.empty(1, 1, height, temporal_dim).normal_(std=0.02))
		self.w_emb = nn.Parameter(torch.empty(1, 1, width, temporal_dim).normal_(std=0.02))
		# self.t_emb = nn.Parameter(torch.empty(1, length+1, 1, temporal_dim).normal_(std=0.02))

		# Temporal Transformer
		self.transformer = nn.ModuleList([
			TemporalBlock(
				window_size, height, width,
				temporal_dim, 2*temporal_dim, temporal_heads,
				activation=nn.Tanh,
				drop_prob=drop_prob, depth_prob=depth_prob*i/temporal_depth,
				cache_size=length+1,
			) for i in range(temporal_depth)
		])
		self.norm = nn.LayerNorm(temporal_dim)

		# head
		self.code_head = nn.Linear(temporal_dim, n_codes)

		self.register_buffer("cache_frames", None, persistent=False)

		if vitvq_path is not None:
			self.init_vit_from_ckpt(vitvq_path)

		if not is_finetune:
			for param in self.vit.parameters():
				param.requires_grad = False
			for param in self.pre_quant.parameters():
				param.requires_grad = False
			for param in self.quantizer.parameters():
				param.requires_grad = False

		self.current_pos = 1

	@torch.no_grad()
	def init_vit_from_ckpt(self, path: str):
		sd = torch.load(path)["state_dict"]
		for key, item in sd.items():
			if key.startswith("quantizer"):
				self.quantizer.embedding.weight.data.copy_(item)
			elif key.startswith("pre_quant"):
				if "weight" in key:
					self.pre_quant.weight.data.copy_(item)
				elif "bias" in key:
					self.pre_quant.bias.data.copy_(item)
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
									self.vit.transformer[idx].attn_norm.weight.data.copy_(item)
								if "bias" in key[dot_idx:]:
									self.vit.transformer[idx].attn_norm.bias.data.copy_(item)
							if "fn" in key[dot_idx:]:
								if "to_qkv" in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										d = item.shape[1]
										self.vit.transformer[idx].attn.query.weight.copy_(item[:d])
										self.vit.transformer[idx].attn.key.weight.copy_(item[d:-d])
										self.vit.transformer[idx].attn.value.weight.copy_(item[-d:])
									if "bias" in key[dot_idx:]: pass
								if "to_out" in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										self.vit.transformer[idx].attn.proj.weight.data.copy_(item)
									if "bias" in key[dot_idx:]:
										self.vit.transformer[idx].attn.proj.bias.data.copy_(item)
						if "1" in key[dot_idx:]:
							if "norm" in key[dot_idx:]:
								if "weight" in key[dot_idx:]:
									self.vit.transformer[idx].mlp_norm.weight.copy_(item)
								if "bias" in key[dot_idx:]:
									self.vit.transformer[idx].mlp_norm.bias.copy_(item)
							if "net" in key[dot_idx:]:
								if ".net.0." in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										self.vit.transformer[idx].mlp[0].weight.copy_(item)
									if "bias" in key[dot_idx:]:
										self.vit.transformer[idx].mlp[0].bias.copy_(item)
								if ".net.2." in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										self.vit.transformer[idx].mlp[3].weight.copy_(item)
									if "bias" in key[dot_idx:]:
										self.vit.transformer[idx].mlp[3].bias.copy_(item)
					elif "encoder.transformer.norm" in key:
						if "weight" in key:
							self.vit.norm.weight.data.copy_(item)
						if "bias" in key:
							self.vit.norm.bias.data.copy_(item)
				elif "en_pos_embedding" in key:
					self.vit.en_pos_embedding.copy_(item)
				elif "to_patch_embedding" in key:
					if "weight" in key:
						self.vit.to_patch_embedding[0].weight.copy_(item)
					elif "bias" in key:
						self.vit.to_patch_embedding[0].bias.copy_(item)

	def create_temporal_mask(self,
		T: int,
		device: torch.device=torch.device("cpu")
	) -> torch.BoolTensor:
		mask = torch.triu(torch.ones(T+1, T+1, dtype=torch.bool, device=device), diagonal=1)
		# ClS token from other tokens, but NOT vice versa, facilitating KV caching
		mask[:, 0] = True
		mask[0, :] = False
		return mask#.unsqueeze(0)  # account for token and batch dimension

	def apply_rope(self, x, pos_idx: int=None):
		if pos_idx == None:
			pos_idx = x.shape[2]
		
		inv_x = torch.cat([
			-x[..., 1::2],
			x[..., ::2]
		], dim=-1)

		return self.cos[:, :, :pos_idx] * x + self.sin[:, :, :pos_idx] * inv_x

	def reset_cache(self):
		self.current_pos = 1
		self.cache_frames = None
		for block in self.transformer:
			block.reset_cache()
		# self.decoder.reset_cache()

	def forward(self,
		frames: torch.FloatTensor,
		token_mask: torch.LongTensor=None,
		use_cache: bool=False,
	) -> Tuple[torch.FloatTensor, torch.LongTensor]:
		"""
			During inference, input consists of current frame 
				while other previous frames are cached
			frames: (B, T, 3, 256, 256) /  (B, 1, 3, 256, 256)
		"""
		B, T, *_ = frames.shape

		features = self.vit(frames.flatten(0, 1), None)  # (BT, HW, D)
		h = self.pre_quant(features)
		_, _, code = self.quantizer(h)
		HW = code.shape[-1]
		code = code.reshape(B, T, HW)  # (B, T, HW)

		temporal_mask = self.create_temporal_mask(T, code.device) if not use_cache else torch.zeros(
			(2, self.current_pos+1), dtype=torch.float, device=code.device
		)
		if use_cache:
			temporal_mask[:, 0] = True
			temporal_mask[0, :] = False

		emb = self.embedding(code)  # (B, T/1, HW, D)
		
		if token_mask is not None:
			# detach?
			emb = (1-token_mask).unsqueeze(-1)*self.mask_token + (token_mask.unsqueeze(-1)*emb)
			# emb = (1-token_mask).unsqueeze(-1)*self.mask_token + (token_mask.unsqueeze(-1)*emb.detach())
		
		# 1, 2, ..., n, 1, 2, ..., n, ...
		_h_emb = self.h_emb.repeat((1, 1, self.width, 1))
		# 1, 1, ..., 1, 2, 2, ..., 2, ...
		_w_emb = self.w_emb.repeat_interleave(self.height, dim=2)
		# for temporal, using RoPE, which is implemented inside RoPEAttention
		x = emb + _h_emb + _w_emb  # (B, T, HW, D)

		cls_token = self.cls_token.repeat((B, 1, HW, 1))
		x = torch.cat([cls_token, x], dim=1)  # (B, T', HW, D)
		x = self.emb_norm(x)

		for layer in self.transformer:
			x, attn = layer(x, None, temporal_mask, use_cache)

		logits = self.code_head(self.norm(x))
		# logits = self.code_head(self.norm(x[:, 1:]))
		
		if use_cache:
			self.current_pos = min(self.current_pos+1, self.length)

		return code, logits, attn


class Decoder(nn.Module):

	def __init__(self,
		is_finetune: bool,
		image_size: Union[int, Tuple[int, int]],
		patch_size: Union[int, Tuple[int, int]],
		depth: int, heads: int, dim: int,
		
		# temporal_depth: int, temporal_heads: int, temporal_dim: int, 
		n_codes: int, embed_dim: int,
		drop_prob:float=0.1, depth_prob: float=0.1,
		path: str=None, vitvq_path: str=None
	):
		super().__init__()
		
		self.vit_decoder = VisionDecoder(
			image_size, patch_size,
			depth, dim, heads, 4*dim,
			drop_prob=drop_prob, depth_prob=depth_prob
		)
		self.quantizer = VectorQuantizer(embed_dim, n_codes)
		self.post_quant = nn.Linear(embed_dim, dim)

		if vitvq_path is not None:
			self.init_vit_from_ckpt(vitvq_path)
			
		if not is_finetune:
			for param in self.vit_decoder.parameters():
				param.requires_grad = False
			for param in self.quantizer.parameters():
				param.requires_grad = False
			for param in self.post_quant.parameters():
				param.requires_grad = False
		else:
			# init LORA
			pass

	@torch.no_grad()
	def init_vit_from_ckpt(self, path: str):
		sd = torch.load(path)["state_dict"]
		for key, item in sd.items():
			if key.startswith("quantizer"):
				self.quantizer.embedding.weight.data.copy_(item)
			elif key.startswith("post_quant"):
				if "weight" in key:
					self.post_quant.weight.data.copy_(item)
				elif "bias" in key:
					self.post_quant.bias.data.copy_(item)
			elif key.startswith("decoder"):
				if "de_pos_embedding" in key:
					self.vit_decoder.de_pos_embedding.copy_(item)
				elif "transformer" in key:
					if "layers" in key:
						layer_idx = key.index("layers")
						dot_idx = key.find(".", layer_idx+len("layers")+1)
						idx = int(key[layer_idx+len("layers")+1:dot_idx])
						if "0" in key[dot_idx:]:
							if "norm" in key[dot_idx:]:
								if "weight" in key[dot_idx:]:
									self.vit_decoder.transformer[idx].attn_norm.weight.data.copy_(item)
								if "bias" in key[dot_idx:]:
									self.vit_decoder.transformer[idx].attn_norm.bias.data.copy_(item)
							if "fn" in key[dot_idx:]:
								if "to_qkv" in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										d = item.shape[1]
										self.vit_decoder.transformer[idx].attn.query.weight.copy_(item[:d])
										self.vit_decoder.transformer[idx].attn.key.weight.copy_(item[d:-d])
										self.vit_decoder.transformer[idx].attn.value.weight.copy_(item[-d:])
									if "bias" in key[dot_idx:]: pass
								if "to_out" in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										self.vit_decoder.transformer[idx].attn.proj.weight.data.copy_(item)
									if "bias" in key[dot_idx:]:
										self.vit_decoder.transformer[idx].attn.proj.bias.data.copy_(item)
						if "1" in key[dot_idx:]:
							if "norm" in key[dot_idx:]:
								if "weight" in key[dot_idx:]:
									self.vit_decoder.transformer[idx].mlp_norm.weight.copy_(item)
								if "bias" in key[dot_idx:]:
									self.vit_decoder.transformer[idx].mlp_norm.bias.copy_(item)
							if "net" in key[dot_idx:]:
								if ".net.0." in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										self.vit_decoder.transformer[idx].mlp[0].weight.copy_(item)
									if "bias" in key[dot_idx:]:
										self.vit_decoder.transformer[idx].mlp[0].bias.copy_(item)
								if ".net.2." in key[dot_idx:]:
									if "weight" in key[dot_idx:]:
										self.vit_decoder.transformer[idx].mlp[3].weight.copy_(item)
									if "bias" in key[dot_idx:]:
										self.vit_decoder.transformer[idx].mlp[3].bias.copy_(item)
					elif "decoder.transformer.norm" in key:
						if "weight" in key:
							self.vit_decoder.norm.weight.data.copy_(item)
						if "bias" in key:
							self.vit_decoder.norm.bias.data.copy_(item)
				elif "to_pixel" in key:
					if "weight" in key:
						self.vit_decoder.to_pixel[1].weight.copy_(item)
					if "bias" in key:
						self.vit_decoder.to_pixel[1].bias.copy_(item)

	def reset_cache(self):
		self.vit_decoder.reset_cache()

	def forward(self, code: torch.LongTensor, use_cache: bool=False) -> torch.FloatTensor:
		B, T, _ = code.shape

		code = code.flatten(0, 1)
		quant = self.quantizer.embedding(code)
		quant = self.quantizer.norm(quant)
		quant = self.post_quant(quant)

		if self.quantizer.use_residual:
			quant = quant.sum(-2)

		dec = self.vit_decoder(quant, None)
		H, W = dec.shape[-2:]

		return dec.reshape(B, T, 3, H, W)
	
	def forward_feature(self, quant: torch.FloatTensor) -> torch.FloatTensor:
		quant = self.post_quant(quant)
		dec = self.vit_decoder(quant, None)

		return dec


class MaskVideo(nn.Module):
	
	def __init__(self,
		is_finetune: bool,
		image_size: Union[int, Tuple[int, int]],
		patch_size: Union[int, Tuple[int, int]],
		depth: int, heads: int, dim: int, 
		n_codes: int, embed_dim: int,

		window_size: Union[int, Tuple[int, int]],
		length: int, height: int, width: int,

		temporal_depth: int, temporal_heads: int, temporal_dim: int, 
		drop_prob:float=0.1, depth_prob: float=0.1,
		path: str=None, vitvq_path: str=None,
	):
		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.length, self.height, self.width = length, height, width
		self.n_codes = n_codes

		self.encoder = Encoder(
			is_finetune=is_finetune,
			window_size=window_size,
			length=length, height=height, width=width,
			image_size=image_size, patch_size=patch_size,
			depth=depth, heads=heads, dim=dim,
			n_codes=n_codes, embed_dim=embed_dim,
			temporal_depth=temporal_depth, temporal_heads=temporal_heads, temporal_dim=temporal_dim, 
			drop_prob=drop_prob, depth_prob=depth_prob,
			vitvq_path=vitvq_path
		)
		self.decoder = Decoder(
			is_finetune=is_finetune,
			image_size=image_size, patch_size=patch_size,
			depth=depth, heads=heads, dim=dim,
			n_codes=n_codes, embed_dim=embed_dim,
			drop_prob=drop_prob, depth_prob=depth_prob,
			vitvq_path=vitvq_path
		)
	
	def reset_cache(self):
		self.encoder.reset_cache()
		self.decoder.reset_cache()

	def export(self):
		dummy_input = torch.randn(1, 64, 3, 256, 256)
		torch.onnx.export(self.encoder, dummy_input, "encoder.onnx", export_params=True)
		print("ONNX model saved: encoder.onnx")

		dummy_input = torch.randn(1, 16, 1024)
		torch.onnx.export(self.decoder, dummy_input, "decoder.onnx", export_params=True)
		print("ONNX model saved: decoder.onnx")

	def forward(self,
		frames: torch.FloatTensor,  # (B, T, 3, H, W)
		token_mask: torch.LongTensor,   # (B, T, HW)
		use_cache: bool=False
	):
		code, logits, attn = self.encoder(frames, token_mask, use_cache)
		dec = self.decoder(code)
		return code, logits, attn, dec
		# return code, logits, attn

