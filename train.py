import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import numpy as np
from skimage import data
from skimage.transform import resize

import matplotlib.pyplot as plt

import torch

from src.reader import KineticDataset

from src.vqgan import ViTVQGAN
from src.my_model import MaskCode
from src.trainer import AttentionMaskModeling


if __name__ == "__main__":
    path = "E:/kinetics-dataset/k400"
    split = "train"

    ds = KineticDataset(
        path, split,
        n_frames=32,
    )
    ds_loader = DataLoader(
        ds, 1, True, num_workers=4
    )

    lightning_model = AttentionMaskModeling(
        model=MaskCode(
            window_size=(3, 3),
            length=32, height=32, width=32,
            depth=4, heads=12, dim=96, embed_dim=32,
            n_codes=8192,
        ), 
        threshold=0.95
    )

    trainer = pl.Trainer(
        # training settings
        max_epochs=25,
        val_check_interval=1.0,
        default_root_dir=f"{path}/trained",
        accelerator="cpu" if is_dev else "gpu",
        precision="32-true" if is_dev else "16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        # logging settings
        logger=WandbLogger(
            project="semcom",
        ),
    )
    trainer.fit(
        lightning_model, 
        train_dataloaders=ds_loader,
        # val_dataloaders=None,
    )
