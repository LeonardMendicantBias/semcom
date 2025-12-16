from typing import Optional, Union, Tuple

import torch
from torch import nn

from torchvision.ops import MLP, StochasticDepth

from .attention import SpatialAttentionLayer, TemporalAttentionLayer
from .base import Block


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
		self.spatial_attn = SpatialAttentionLayer(
			window_size, height, width,
			dim, n_heads, cache_size
		)

		self.temporal_attn_norm = nn.LayerNorm(dim)
		self.temporal_attn = TemporalAttentionLayer(dim, n_heads, cache_size)

		self.mlp_norm = nn.LayerNorm(dim)
		self.mlp: nn.Module = MLP(
			dim, [mlp_dim, dim],
			dropout=drop_prob,
			activation_layer=activation  # nn.Tanh
		)
		self.drop_path = StochasticDepth(depth_prob, mode="row")

	def forward(self,
		query: torch.FloatTensor,			# (B, T', HW, D)
		spatial_mask: torch.BoolTensor,		# (1, T', T')
		temporal_mask: torch.BoolTensor,	# (1, T', T')
		# past_spatial_info: torch.FloatTensor=None, # (B, T, HW, D)
		past_key: torch.FloatTensor=None,  # (B, T, HW, D)
		past_value: torch.FloatTensor=None,
	) -> torch.FloatTensor:
		
		x_norm = self.spatial_attn_norm(query)
		x, spatial_attn_logits, k, v = self.spatial_attn(x_norm, None, None, None)
		x = self.drop_path(x) + query  # (B, T_, HW, D)
		
		x_norm = self.temporal_attn_norm(x)
		x, temporal_attn_logits, k, v = self.temporal_attn(
			x_norm, None, temporal_mask, past_key, past_value
		)
		x = self.drop_path(x) + x

		# non-linear
		x_norm = self.mlp_norm(x)
		x = self.drop_path(self.mlp(x_norm)) + x
		# print(k.shape, v.shape)

		# return x, spatial_attn_logits, k, v
		return x, temporal_attn_logits, k, v
