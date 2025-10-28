from typing import Optional
import copy

import torch
from torch import nn
import torch.nn.functional as F

from torch import optim

from torchmetrics import Accuracy

import pytorch_lightning as pl


class AttentionMaskModeling(pl.LightningModule):

	def __init__(self,
		model: nn.Module,
		top_p: float=0.95,
		lr: float=1e-4,
		T_max: int=5,
		warmup_epochs: int=5,
		weight_decay: float=0.0,
		ema_momentum: float=0.999,
		ema_warmup_steps: int=0,
	):
		super().__init__()
		self.lr = lr
		self.T_max = T_max
		self.warmup_epochs = warmup_epochs

		self.weight_decay = weight_decay
		self.base_ema = ema_momentum
		self.ema_warmup_steps = ema_warmup_steps
		self.top_p = top_p

		self.model = model

		self.teacher = copy.deepcopy(model)
		self._set_teacher_eval_and_freeze()
		
		self._step = 0 

		self.save_hyperparameters(ignore=['model'])

		self.acc = Accuracy(task="multiclass", num_classes=self.model.n_codes)
	
	def _set_teacher_eval_and_freeze(self) -> None:
		for p in self.teacher.parameters():
			p.requires_grad = False
		# self.teacher.eval()

	def configure_optimizers(self):
		optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
		warmup_scheduler = optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=1e-4, end_factor=1.0,
			total_iters=self.warmup_epochs
		)
		cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
		scheduler = optim.lr_scheduler.SequentialLR(
			optimizer,
			schedulers=[warmup_scheduler, cosine_scheduler],
			milestones=[self.warmup_epochs]
		)
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "epoch",
				"frequency": 1,
			},
		}

	def on_train_start(self) -> None:
		self._copy_weights(self.teacher, self.model)

	@torch.no_grad()
	def _copy_weights(self, target: nn.Module, source: nn.Module):
		# copy all params + buffers
		for t_param, s_param in zip(target.parameters(), source.parameters()):
			t_param.data.copy_(s_param.data)
		# copy buffers (e.g., running stats)
		for t_buf, s_buf in zip(target.buffers(), source.buffers()):
			t_buf.data.copy_(s_buf.data)
			
	@torch.no_grad()
	def _ema_update(self, momentum: float):
		# EMA update: teacher = momentum * teacher + (1 - momentum) * student
		for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
			t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))
		# for running stats/buffers we typically copy directly (or EMA if you prefer)
		for t_buf, s_buf in zip(self.teacher.buffers(), self.model.buffers()):
			t_buf.data.copy_(s_buf.data)

	def reset_cache(self):
		self.teacher.reset_cache()
		self.model.reset_cache()

	@torch.no_grad()
	def get_mask_from_logits(self,
		# code: torch.FloatTensor,
		frames: torch.FloatTensor,  # (B, T, 3, _, _)
		K: int=None,
	) -> torch.BoolTensor:
		_, _, attn_logits = self.teacher.encoder(frames, None, use_cache=False)
		B, HW, _, T_, _ = attn_logits.shape
		
		# which tokens are salient according to the cls token
		cls_logits = attn_logits[:, :, :, 0, 1:]  # (B, HW, n_heads, T)
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

	def training_step(self, batch, batch_idx):
		"""
			single frame prediction
		"""
		frames, _ = batch  # (B, T, HW)

		# generate masks for all time steps
		# with torch.no_grad():
		#     # (B, T, HW, C), (B, T, HW)
		#     _, attn_logits = self.teacher(code, None)
		mask = self.get_mask_from_logits(frames)  # (B, T', HW)
		
		code, logits, _, dec = self.model(frames, mask)

		# only train the last time step
		loss = F.cross_entropy(
			logits[:, 1:].flatten(1, 2).flatten(0, 1),
			code.flatten(1, 2).flatten(0, 1),
			label_smoothing=0.01
		)
		acc = self.acc(logits[:, 1:].flatten(0, 1).flatten(0, 1), code.flatten(0, 1).flatten(0, 1))
		masking_ratio = mask.sum(-1) / code.shape[-1]

		self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
		self.log("train_acc", acc, on_step=True, prog_bar=True, logger=True)
		self.log("train_mask_ratio", masking_ratio.mean().item(), on_step=True, prog_bar=True, logger=True)

		return loss
	
	def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int=0):
		# increment and compute momentum (optional warmup)
		self._step += 1

		if self.ema_warmup_steps > 0 and self._step < self.ema_warmup_steps:
			# linear warmup from 0 -> base_ema
			momentum = (self.base_ema * self._step) / float(self.ema_warmup_steps)
		else:
			momentum = float(self.base_ema)

		# perform EMA update
		self._ema_update(momentum)
