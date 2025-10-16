from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import MLP, StochasticDepth

from .attention import MultiHeadAttention
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
		self.spatial_attn: MultiHeadAttention = MultiHeadAttention(dim, n_heads, cache_size)

		self.temporal_attn_norm = nn.LayerNorm(dim)
		self.temporal_attn: MultiHeadAttention = MultiHeadAttention(dim, n_heads, cache_size)

		self.mlp_norm = nn.LayerNorm(dim)
		self.mlp: nn.Module = MLP(
			dim, [mlp_dim, dim],
			dropout=drop_prob,
			activation_layer=activation  # nn.Tanh
		)
		self.drop_path = StochasticDepth(depth_prob, mode="row")
		
		self.register_buffer("cache_spatial", None, persistent=False)

	def reset_cache(self):
		self.cache_spatial = None

		self.spatial_attn.reset_cache()
		self.temporal_attn.reset_cache()

	def extract_spatial_feature(self,
		query: torch.FloatTensor,
		mask: torch.BoolTensor,
		use_cache: bool=False
	) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
		"""
			Treat each frame independently by merging the temporal dimension with batch
			Args:
				query: (B, T+1, HW, D) including CLS token
			Process:
				1. for each frame, find neighbors of each spatial location
				2. perform MHA for each spatial location with its neighbors + global CLS token
				3. combine the output of all frames
			Returns:
				x: (B, T', HW, D)
				attn_logits: (B, HW, n_heads, T', T')
		"""
		# stat
		B, T_, HW, D = query.shape
		T = T_ - 1
		padding_size = (self.window_size[0]//2, self.window_size[1]//2)
		size = self.window_size[0] * self.window_size[1]
		
		# exclude CLS token and obtain visual tokens
		visual_x = query[:, 1:].reshape(
			B, T, self.height, self.width, D
		)  # (B, T, H, W, D)
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
		neighbor_x = neighbor_x.reshape(B*T*HW, size, D)	# (B', 9, D)
		# append a global CLS token to each neighbor
		neighbor_x = torch.cat([
			query[:, 0].mean(1, keepdim=True).repeat((T*HW, 1, 1)),  # (B', 1, D) -> (B*T*HW, 1, D)
			neighbor_x,
		], dim=1)  # (B', 1, D)

		visual_x_norm = self.spatial_attn_norm(visual_x)		# (BTHW, 1, D)
		neighbor_x_norm = self.spatial_attn_norm(neighbor_x)	# (BTHW, 9+1, D)

		# mask the padding spatial location
		spatial_mask = torch.ones(
			(B*T, 1, self.height, self.width),
			dtype=torch.bool, device=query.device
		)
		spatial_mask = ~F.unfold(
			spatial_mask, self.window_size,
			padding=padding_size, stride=1,
		)  # (1, size*1, HW)
		spatial_mask = spatial_mask.reshape(B*T, size, 1, HW).permute(0, 3, 2, 1)
		spatial_mask = spatial_mask.flatten(1, 2).reshape(B*T*HW, 1, size)
		# # mask for the global CLS token
		spatial_mask = torch.cat([
			spatial_mask, torch.zeros((B*T*HW, 1, 1), dtype=torch.bool, device=query.device)
		], dim=-1)

		# (B', 1, D), (B', 9+1, D)
		x, attn_logits = self.spatial_attn(
			visual_x_norm, neighbor_x_norm,
			mask=spatial_mask, use_cache=use_cache
		)
		x = self.drop_path(x) + visual_x.reshape(B*T*HW, 1, D)
		x = x.reshape(B, T, HW, D)

		# add back CLS token
		x = torch.cat([query[:, 0:1], x], dim=1)
		# attn = attn.reshape(B, T, HW, 12, 1, 10)

		return x, attn_logits
	
	def extract_temporal_feature(self,
		query: torch.FloatTensor,
		mask: torch.BoolTensor,
		use_cache: bool=False
	) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
		"""
			Perform MHA for each spatial locations across all time steps.

		"""
		B, T_, HW, D = query.shape
		T = T_ - 1

		x = query.permute(0, 2, 1, 3).flatten(0, 1)  # (BHW, T', D)
		x_norm = self.temporal_attn_norm(x)
		x_, attn_logits = self.temporal_attn(x_norm, x_norm, mask, use_cache=use_cache)
		x = self.drop_path(x_) + x

		x = x.reshape(B, HW, T_, D).permute(0, 2, 1, 3)
		avg_cls = x[:, 0:1, :].mean(dim=2)
		x[:, 0, :] = avg_cls

		attn_logits = attn_logits.reshape(B, HW, self.n_heads, T_, T_)

		return x, attn_logits

	def forward(self,
		query: torch.FloatTensor,			# (B, T', HW, D)
		spatial_mask: torch.BoolTensor,		# (1, T', T')
		temporal_mask: torch.BoolTensor,	# (1, T', T')
		use_cache: bool=False
	) -> torch.FloatTensor:
		B, T_, HW, D = query.shape
		
		"""
			Treating each frame independently, thus
			query: (B*T*HW, 1, D) 
			key: (B*T*HW, 10, D) 

			KV caching, the KV of previous time step are already computed, thus
			query: (B*HW, 1, D);	T=1 
			key: (B*HW, 10, D); 	T=1
		"""
		x, spatial_attn_logits = self.extract_spatial_feature(query, spatial_mask, use_cache=False)
		
		if use_cache:
			if self.cache_spatial is None:
				self.cache_spatial = x
			else:
				self.cache_spatial = torch.cat([self.cache_spatial, x], dim=0)
			
			if self.cache_spatial.shape[1] > self.cache_size:
				self.cache_spatial = self.cache_spatial[-self.cache_size:]

		"""
			Treating 
		"""
		x, temporal_attn_logits = self.extract_temporal_feature(x, temporal_mask, use_cache)

		# non-linear
		x_norm = self.mlp_norm(x)
		x = self.drop_path(self.mlp(x_norm)) + x

		return x, temporal_attn_logits
