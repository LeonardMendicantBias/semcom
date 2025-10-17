import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
from skimage import data
from skimage.transform import resize

import matplotlib.pyplot as plt

import torch

from src.reader import KineticDataset, KineticDatasetVideo

from src.vqgan import ViTVQGAN
from src.my_model import MaskCode, Encoder, Decoder, MaskVideo
from src.trainer import AttentionMaskModeling


if __name__ == "__main__":
    path = "/mnt/e/kinetics-dataset/k400"
    split = "train"

    ds = KineticDatasetVideo(
        path, split,
        n_frames=16,
    )
    ds_loader = DataLoader(
        ds, 2, True,
        num_workers=4
    )
    is_finetune = False

    image_size = (256, 256)
    patch_size = (8, 8)
    depth, heads, dim, embed_dim = 12, 12, 768, 32

    window_size = (3, 3)
    length, height, width = 32, 32, 32
    temporal_depth, temporal_heads, temporal_dim = 4, 8, 128
    n_codes=8192

    vitvq_path = "./checkpoint/imagenet_vitvq_base.ckpt"

    is_dev = False

    lightning_model = AttentionMaskModeling(
        model=MaskVideo(
            is_finetune=is_finetune,
            image_size=image_size, patch_size=patch_size,
            depth=depth, heads=heads, dim=dim,
            n_codes=n_codes, embed_dim=embed_dim,

            window_size=window_size,
            length=length, height=height, width=width,
            temporal_depth=temporal_depth, temporal_heads=temporal_heads, temporal_dim=temporal_dim,

            drop_prob=0.1, depth_prob=0.1,
            vitvq_path=vitvq_path
        ), 
        top_p=0.95
    )
    # wandb_logger = WandbLogger(
    # 	project="semcom",
    # )
    # wandb_logger.experiment.config.update({
    #     "dim": dim,
    #     "depth": depth
    # })
    trainer = pl.Trainer(
        # training settings
        max_epochs=25,
        val_check_interval=1.0,
        accelerator="cpu" if is_dev else "gpu",
        precision="32-true" if is_dev else "16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        # logging settings
        default_root_dir=f"./checkpoints",
        # logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor="train_acc",
                dirpath="./trained",
                filename="semcom-{epoch:02d}-{train_acc:.2f}",
                save_top_k=3,
                mode="max",
            )
        ],
        # dev setting
        fast_dev_run=is_dev,
    )
    trainer.fit(
        lightning_model, 
        train_dataloaders=ds_loader,
        # val_dataloaders=None,
    )