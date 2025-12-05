from typing import Union, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

# from .vqgan import ViTVQGAN
from .quantizer import VectorQuantizer
from .transformer import TemporalBlock, VisionTransformer, VisionDecoder
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
		ks: List[torch.FloatTensor]=None, vs: List[torch.FloatTensor]=None,
	) -> Tuple[torch.LongTensor, torch.BoolTensor]:
		codes = self.process_frames(frames)

		x, attn_logits, ks, vs = self.transformer(codes, None, past_keys=ks, past_values=vs)
		mask = self.transformer.get_mask_from_logits(attn_logits)
		return codes[:, -1], mask[:, -1], ks, vs
	
	def export(self) -> None :
		encoder = Encoder(
			self.vit, self.pre_quant, self.quantizer,
			self.transformer
		).eval().cuda()
		L = self.transformer.depth
		HW = self.transformer.height * self.transformer.width
		n_heads = self.transformer.n_heads
		dummy_frames = torch.randn(1, 1, 3, 256, 256).cuda()
		dummy_ks = torch.randn(L, HW, n_heads, 1, 16).cuda()
		dummy_vs = torch.randn(L, HW, n_heads, 1, 16).cuda()
		torch.onnx.export(
			encoder,     #torch model
			(dummy_frames, dummy_ks, dummy_vs),  #inputs
			"encoder.onnx",   #path of the output onnx model
			input_names=["frames", "prev_k", "prev_v"],
			output_names=["code", "mask", "curr_k", "curr_v"],
			dynamic_axes={
				# 'frames': {1: 'sequence'}, temporal length of frames is always ONE: the current frame
				'prev_k': {3: 'sequence'},
				'prev_v': {3: 'sequence'},
				'curr_k': {3: 'sequence'},
				'curr_v': {3: 'sequence'},
			},
			export_params=True,
			opset_version=14
		)
		print("ONNX model saved: encoder.onnx")


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
		temporal_depth: int, temporal_heads: int, # temporal_dim: int,
		
		drop_prob:float=0.1, depth_prob: float=0.1,
		vitvq_path: str=None,
	):
		super().__init__()
		self.is_finetune = is_finetune
		self.image_size, self.patch_size = image_size, patch_size
		self.n_codes = n_codes
		self.dim = vit_dim

		self.vit = VisionTransformer(
			image_size=image_size, patch_size=patch_size,
			depth=vit_depth, heads=vit_heads,
			dim=vit_dim, mlp_dim=4*vit_dim,
			drop_prob=drop_prob, depth_prob=depth_prob
		)
		self.pre_quant = nn.Linear(vit_dim, embed_dim)
		self.quantizer = VectorQuantizer(embed_dim, n_codes)

		temporal_dim = vit_dim

		self.cls_token = nn.Parameter(torch.empty(1, 1, 1, temporal_dim).normal_(std=0.02))
		self.mask_token = nn.Parameter(torch.empty(1, 1, 1, temporal_dim).normal_(std=0.02))
		self.transformer = nn.ModuleList([
			TemporalBlock(
				window_size, height, width,
				temporal_dim, 4*temporal_dim, temporal_heads,
				activation=nn.Tanh,
				drop_prob=drop_prob, depth_prob=depth_prob*i/temporal_depth,
				cache_size=length,
			) for i in range(temporal_depth)
		])
		self.transformer_norm = nn.LayerNorm(temporal_dim)

		self.code_head = nn.Linear(temporal_dim, n_codes)
		self.rgb_head = nn.Linear(temporal_dim, 3)
		self.feat_head = nn.Linear(temporal_dim, temporal_dim)
		self.rgb_head = nn.Sequential(
			Rearrange(
				"b (h w) c -> b c h w",
				h=image_size[0]//patch_size[0], w=image_size[1]//patch_size[1]
			),
			nn.ConvTranspose2d(
				vit_dim, 3, kernel_size=patch_size, stride=patch_size
			),
		)

		if vitvq_path is not None: self.init_vit_from_ckpt(vitvq_path)

		if not is_finetune:
			for param in self.vit.parameters():
				param.requires_grad = False
			for param in self.pre_quant.parameters():
				param.requires_grad = False
			for param in self.quantizer.parameters():
				param.requires_grad = False
		else: pass  # implement LoRA

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
	
	def export_decoder(self) -> None:
		decoder = Decoder(self.transformer, self.code_head)
		dummy_input = torch.randn(1, 16, 1024)
		dummy_mask = torch.randn(1, 16, 1024)
		torch.onnx.export(decoder, (dummy_input, dummy_mask, True), "decoder.onnx", export_params=True)
		print("ONNX model saved: decoder.onnx")

	def process_frames(self, frames: torch.FloatTensor) -> torch.LongTensor:
		B, T = frames.shape[:2]  # T = 1 during inference

		# process each frame
		features = self.vit(frames.flatten(0, 1), None)  # (BT, HW, D)
		h = self.pre_quant(features)
		_, _, codes = self.quantizer(h)
		HW = codes.shape[-1]
		return codes.reshape(B, T, HW), features.reshape(B, T, HW, -1)  # (B, T, HW)

	@torch.no_grad()
	def get_mask_from_logits(self,
		# logits are extarcted from self-attention module
		logits: torch.FloatTensor,  # (B, HW, heads, T_, T_)
		# past_mask: torch.FloatTensor,
		K: int=None,
		top_p: float=0.9,
	) -> torch.BoolTensor:
		"""
			mask patches with low attention scores

		"""
		# print("logits", logits.shape)
		B, HW, _, S, T_ = logits.shape
		
		# which patches are salient according to the cls token
		cls_logits = logits[:, :, :, 0, 1:]  # (B, HW, n_heads, T)

		# first softmax over spatial dimension then temporal
		# to ensure some patches are choosen at all time steps
		cls_logits = cls_logits.permute(0, 2, 3, 1)  # (B, n_heads, T, HW)
		cls_logits = cls_logits.softmax(dim=-1).flatten(-2, -1)  # (B, n_heads, T*HW)
		# average over all heads -> (B, T*HW)
		attn = cls_logits.softmax(-1).mean(1)
		
		if self.training:
			attn = F.dropout(attn, 0.1)
		# sorted_attn, sorted_indices = torch.sort(attn, descending=True, dim=1)
		sorted_attn, sorted_indices = torch.topk(attn, attn.shape[-1], largest=True, sorted=True, dim=1)
		cumulative_probs = torch.cumsum(sorted_attn, dim=1)

		# top patches whose commulative attention scores are more than top_p are unmasked (set to False)
		prob = cumulative_probs > top_p
		prob[..., 1:] = prob[..., :-1].clone()
		prob[..., 0] = False
		
		mask = prob.scatter(
			dim=-1,
			index=sorted_indices,  # while ordered, the indices are of original sequence
			src=prob
		).reshape((B, T_-1, HW))
		
		return mask.to(torch.long)
	
	@torch.no_grad()
	def get_mask_from_frames(self,
		frames: torch.FloatTensor,  # (B, T, 3, _, _)
		top_p: float,
		K: int=None,
		past_keys: torch.FloatTensor=None, past_values: torch.FloatTensor=None,
	) -> torch.BoolTensor:
		codes, features = self.process_frames(frames)
		B, T, HW = features.shape[:3]
		# _, attn_logits, _, _ = self.transformer(codes, None, past_keys, past_values)
		cls_token = self.cls_token.repeat((B, 1, HW, 1))
		x = torch.cat([cls_token, features], dim=1)  # (B, T', HW, D)

		for i, block in enumerate(self.transformer):
			_k = past_keys[i] if past_keys is not None else None
			_v = past_values[i] if past_values is not None else None
			x, attn_logits, k, v = block(x, None, _k, _v)

		return self.get_mask_from_logits(attn_logits, top_p=top_p)

	def create_temporal_mask(self,
		S: int, T: int,
		device: torch.device=torch.device("cpu")
	) -> torch.BoolTensor:
		# ClS token from other tokens, but NOT vice versa, facilitating KV caching
		mask = torch.triu(torch.ones(S+1, T+1, device=device), diagonal=1)
		mask[:, 0] = True
		mask[0, :] = False
		return mask
	
	def forward(self,
		frames: torch.FloatTensor,  # (B, T, 3, _, _)
		token_mask: torch.BoolTensor=None,  # True = keep, False = mask
	) -> Tuple[torch.BoolTensor, torch.LongTensor, torch.FloatTensor]:
		# 1/ process each frame
		codes, features = self.process_frames(frames)  # (B, T, HW), (B, T, HW, D)
		B, T, HW = features.shape[:3]
		raw_features = features.clone()

		# 2/ apply masking
		if token_mask is not None:
			token_mask = token_mask.unsqueeze(-1)
			features = token_mask*self.mask_token + (1-token_mask)*features

		# 3/ preparing inputs
		cls_token = self.cls_token.repeat((B, 1, HW, 1))
		x = torch.cat([cls_token, features], dim=1)  # (B, T', HW, D)

		temporal_mask = self.create_temporal_mask(T, T, codes.device)

		for i, block in enumerate(self.transformer):
			x, attn_logits, _, _ = block(x, None, temporal_mask)
		x = self.transformer_norm(x[:, 1:])

		# ignore CLS token for decoding
		dec_code = self.code_head(x)
		dec_rgb = self.rgb_head(x.flatten(0, 1))
		dec_rgb = dec_rgb.reshape(B, T, *dec_rgb.shape[-3:])
		dec_feat = self.feat_head(x)

		dec = {
			"rgb": dec_rgb,
			"code": dec_code,
			"feat": dec_feat,
		}

		return codes, raw_features, x, attn_logits, dec
	