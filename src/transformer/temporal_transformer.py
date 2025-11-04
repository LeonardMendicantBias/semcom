from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import MLP, StochasticDepth

from .attention import MultiHeadAttention, RoPEAttention
from .base import Block
from .vision_transformer import get_1d_sincos_pos_embed_from_grid,\
	get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid


class TemporalBlock(nn.Module):
	
	def __init__(self,
		window_size: int,
		height: int, width: int,
		dim: int, mlp_dim: int, n_heads: int,
		activation: nn.Module,
		drop_prob: float=0.1, depth_prob: float=0,
		cache_size: int=-1,
	):
		super().__init__()
		self.window_size = window_size
		self.height, self.width = height, width
		self.dim, self.n_heads = dim, n_heads

		self.spatial_attn_norm = nn.LayerNorm(dim)
		self.spatial_attn = MultiHeadAttention(dim, n_heads, cache_size)

		self.temporal_attn_norm = nn.LayerNorm(dim)
		self.temporal_attn = RoPEAttention(dim, n_heads, cache_size)

		self.mlp_norm = nn.LayerNorm(dim)
		self.mlp: nn.Module = MLP(
			dim, [mlp_dim, dim],
			dropout=drop_prob,
			activation_layer=activation  # nn.Tanh
		)
		self.drop_path = StochasticDepth(depth_prob, mode="row")

	def extract_spatial_feature(self,
		# query is without CLS token
		query: torch.FloatTensor,	# (B, T, HW, D)
	) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
		B, T, HW, D = query.shape
		padding_size = (self.window_size[0]//2, self.window_size[1]//2)
		size = self.window_size[0] * self.window_size[1]
		
		# exclude CLS token and obtain visual tokens
		visual_x = query.reshape(B, T, self.height, self.width, D)
		# reshape for F.unfold function
		visual_x = visual_x.flatten(0, 1)  # (BT, H, W, D)
		neighbor_x = F.unfold(
			visual_x.permute(0, 3, 1, 2), self.window_size,
			padding=padding_size, stride=1,
		)  # (BT, 9*D, HW)
		neighbor_x = neighbor_x.reshape(B*T, D, size, HW)
		neighbor_x = neighbor_x.permute(0, 3, 2, 1)  # (BT, HW, 9, D)
		
		# reshaping for MHA
		visual_x = visual_x.flatten(1, 2).flatten(0, 1).unsqueeze(-2)	# (B', 1, D)
		neighbor_x = neighbor_x.reshape(B*T*HW, size, D)		# (B', 9, D)

		visual_x_norm = self.spatial_attn_norm(visual_x)		# (BTHW, 1, D)
		neighbor_x_norm = self.spatial_attn_norm(neighbor_x)	# (BTHW, 9, D)

		# mask the padding spatial location
		spatial_mask = torch.ones(
			(B*T, 1, self.height, self.width),
			dtype=torch.float, device=query.device
		)
		spatial_mask = 1 - F.unfold(
			spatial_mask, self.window_size,
			padding=padding_size, stride=1,
		)  # (1, size*1, HW)
		spatial_mask = spatial_mask.reshape(B*T, size, 1, HW).permute(0, 3, 2, 1)
		spatial_mask = spatial_mask.flatten(1, 2).reshape(B*T*HW, 1, size).unsqueeze(1)
		# print("spatial_mask", spatial_mask.shape)

		# (B', 2, D), (B', 9, D)
		x, attn_logits, k, v = self.spatial_attn(visual_x_norm, neighbor_x_norm, mask=spatial_mask)
		x = self.drop_path(x) + visual_x.reshape(B*T*HW, 1, D)
		x = x.reshape(B, T, HW, D)

		return x, attn_logits, k, v
	
	def extract_temporal_feature(self,
		query: torch.FloatTensor,
		mask: torch.BoolTensor,
		past_key: torch.FloatTensor=None, past_value: torch.FloatTensor=None,
	) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
		B, T_, HW, D = query.shape
		T = T_ - 1

		x = query.permute(0, 2, 1, 3).flatten(0, 1)  # (BHW, T', D)
		x_norm = self.temporal_attn_norm(x)
		
		x_, attn_logits, k, v = self.temporal_attn(x_norm, x_norm, mask, past_key, past_value)
		x = self.drop_path(x_) + x

		x = x.reshape(B, HW, T_, D).permute(0, 2, 1, 3)
		# print(x.shape)
		avg_cls = x[:, 0:1, :].mean(dim=2)
		# print("avg_cls", avg_cls.shape, x[:, 0:1, :].shape, x[:, 0, ...].shape)
		x[:, 0, ...] = avg_cls.repeat((1, HW, 1))
		
		attn_logits = attn_logits.reshape(B, HW, self.n_heads, *attn_logits.shape[-2:])

		return x, attn_logits, k, v

	def forward(self,
		query: torch.FloatTensor,			# (B, T', HW, D)
		spatial_mask: torch.BoolTensor,		# (1, T', T')
		temporal_mask: torch.BoolTensor,	# (1, T', T')
		past_key: torch.FloatTensor=None, past_value: torch.FloatTensor=None,
	) -> torch.FloatTensor:
		B, T_, HW, D = query.shape

		x_, spatial_attn_logits, _, _ = self.extract_spatial_feature(query[:, :-1])
		x = torch.cat([x_, query[:, -1:]], dim=1)

		x, temporal_attn_logits, k, v = self.extract_temporal_feature(
			x, temporal_mask, past_key, past_value
		)

		# non-linear
		x_norm = self.mlp_norm(x)
		x = self.drop_path(self.mlp(x_norm)) + x

		return x, temporal_attn_logits, k, v


class TemporalTransformer(nn.Module):

	def __init__(self,
		window_size: Union[int, Tuple[int, int]],
		length: int, height: int, width: int,
		depth: int, n_heads: int, dim: int,
		n_codes: int,
		drop_prob: float=0.1, depth_prob: float=0,
	):
		super().__init__()
		self.window_size = window_size
		self.length, self.height, self.width = length, height, width
		self.depth = depth
		self.dim, self.n_heads = dim, n_heads
		
		# Temporal Transformer
		self.embedding = nn.Embedding(n_codes, dim)
		self.cls_token = nn.Parameter(torch.empty(1, 1, 1, dim).normal_(std=0.02))
		self.mask_token = nn.Parameter(torch.empty(1, 1, 1, dim).normal_(std=0.02))

		self.h_emb = nn.Parameter(torch.empty(1, 1, height, dim).normal_(std=0.02))
		self.w_emb = nn.Parameter(torch.empty(1, 1, width, dim).normal_(std=0.02))
		self.emb_norm = nn.LayerNorm(dim)
	
		self.blocks = nn.ModuleList([
			TemporalBlock(
				window_size, height, width,
				dim, 2*dim, n_heads,
				activation=nn.Tanh,
				drop_prob=drop_prob, depth_prob=depth_prob*i/depth,
				cache_size=length,
			) for i in range(depth)
		])
		self.norm = nn.LayerNorm(dim)

	def create_temporal_mask(self,
		S: int, T: int,
		device: torch.device=torch.device("cpu")
	) -> torch.BoolTensor:
		# mask = torch.triu(torch.ones(S+1, T+1, device=device), diagonal=1)
		# # ClS token from other tokens, but NOT vice versa, facilitating KV caching
		# mask[:, 0] = True
		# mask[0, :] = False
		mask = torch.triu(torch.ones(S+1, T+1, device=device), diagonal=1)
		mask[-1, -1] = 1
		return mask

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
		B, HW, _, S, T_ = logits.shape
		
		# which patches are salient according to the cls token
		cls_logits = logits[:, :, :, -1, :-1]  # (B, HW, n_heads, T)

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
	
	def forward(self,
		codes: torch.LongTensor, # (B, T, HW)
		token_mask: torch.LongTensor=None,  # True = mask, False = unmask
		past_keys: torch.FloatTensor=None, past_values: torch.FloatTensor=None,
	) -> torch.FloatTensor:
		B, T, HW = codes.shape

		emb = self.embedding(codes)  # (B, T, HW, D)
		
		if token_mask is not None:
			token_mask = token_mask.unsqueeze(-1)
			emb = token_mask*self.mask_token + ~token_mask*emb

		# include spatial information via positional embeddings
		# 1, 2, ..., n, 1, 2, ..., n, ...
		_h_emb = self.h_emb.repeat((1, 1, self.width, 1))
		# 1, 1, ..., 1, 2, 2, ..., 2, ...
		_w_emb = self.w_emb.repeat_interleave(self.height, dim=2)
		x = emb + _h_emb + _w_emb  # (B, T, HW, D)

		cls_token = self.cls_token.repeat((B, 1, HW, 1))
		x = torch.cat([x, cls_token], dim=1)  # (B, T', HW, D)
		x = self.emb_norm(x)
		
		temporal_mask = self.create_temporal_mask(
			T, T + past_keys[0].shape[-2] if past_keys is not None else T,
			codes.device
		)
		ks, vs = [], []
		for i, block in enumerate(self.blocks):
			_k = past_keys[i] if past_keys is not None else None
			_v = past_values[i] if past_values is not None else None
			x, attn_logits, k, v = block(x, None, temporal_mask, _k, _v)

			# ignore the k-v pairs of CLS tokens
			# 	since they need to recompute with new info at the next step
			ks.append(k[..., :-1, :])
			vs.append(v[..., :-1, :])

		ks = torch.stack(ks, dim=0)
		vs = torch.stack(vs, dim=0)
		# print(k.shape, ks.shape)

		return self.norm(x), attn_logits, ks, vs
