from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .self_attention import MultiHeadAttention
from .rope import RoPEAttention


class SpatialAttentionLayer(MultiHeadAttention):
	
	def __init__(self,
		window_size: int, height: int, width: int,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.window_size = window_size
		self.height, self.width = height, width
		
		self._padding_size = (self.window_size[0]//2, self.window_size[1]//2)
		self._size = self.window_size[0] * self.window_size[1]

	def get_neighbor(self, x):
		"""
			1/ flatten batch, time, and head dimensions
			2/ 
		"""
		# print("x", x.shape)
		BT, _, HW, _ = x.shape

		_x = x.reshape(BT, self.n_heads, self.height, self.width, self.head_dim)
		_x = _x.flatten(0, 1)  # (BT*n_heads, H, W, D')

		neighbor_x = F.unfold(
			_x.permute(0, 3, 1, 2), self.window_size,
			padding=self._padding_size, stride=1,
		)  # (BT*n_heads, D*9, HW)
		# neighbor_x = neighbor_x.reshape(BT, self.n_heads, self.head_dim, self._size, HW)
		# neighbor_x = neighbor_x.permute(0, 4, 1, 3, 2).flatten(0, 1)
		neighbor_x = neighbor_x.reshape(BT, self.head_dim, self.n_heads, self._size, HW)
		neighbor_x = neighbor_x.permute(0, 4, 2, 3, 1).flatten(0, 1)
		# print("neighbor", neighbor_x.shape)

		# print("+"*30)
		return neighbor_x

	def forward(self,
		query: torch.FloatTensor, memory: torch.FloatTensor,
		mask: torch.BoolTensor,  # (B, T, S)
		past_key: torch.FloatTensor=None, past_value: torch.FloatTensor=None,
	) -> torch.FloatTensor:
		B, T_, HW, D = query.shape
		T = T_ - 1
		_query = query[:, 1:].flatten(0, 1)
		# print("query", query.shape, _query.shape)
		if memory is None: memory = _query
		
		# (BT, self.n_heads, HW, D')
		q, k, v = self._calculate_qkv(_query, memory, past_key, past_value)
		# print("qkv", q.shape, k.shape, v.shape)

		neighbor_k = self.get_neighbor(k)
		neighbor_v = self.get_neighbor(v)

		# q: (BT*HW, n_heads, 1, D')
		# k: (BT*HW, n_heads, 9, D')
		_q = q.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(2)
		logits = self._calculate_logits(_q, neighbor_k, mask)  # (B, H, T, S)

		attn = self._calculate_attention(logits)
		attn = self.attn_dropout(attn)

		weighted = torch.matmul(attn, neighbor_v)
		weighted = weighted.transpose(1, 2).contiguous().flatten(-2, -1)

		out = self.proj_dropout(self.proj(weighted))
		out = out.squeeze(-2).reshape((B, T, HW, D))
		
		out = torch.cat([out, query[:, 0:1]], dim=1)

		return out, logits, k, v


class TemporalAttentionLayer(RoPEAttention):
	
	def forward(self,
		query: torch.FloatTensor, memory: torch.FloatTensor,
		mask: torch.BoolTensor,  # (B, T, S)
		past_key: torch.FloatTensor=None, past_value: torch.FloatTensor=None,
	) -> torch.FloatTensor:
		B, T_, HW, D = query.shape
		_query = query.permute(0, 2, 1, 3)
		_query = _query.flatten(0, 1)
		
		out, logits, k, v = super().forward(
			_query, _query,  # (B*HW, T_, D)
			mask,
			past_key, past_value
		)

		out = out.reshape((B, HW, T_, D)).permute(0, 2, 1, 3)
		logits = logits.reshape((B, HW, self.n_heads, T_, T_))
		
		out[:, 0] = torch.mean(out[:, 0], dim=1, keepdim=True)

		return out, logits, k, v