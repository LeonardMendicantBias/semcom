# %%
from typing import Tuple
# import os
import math
import pathlib
import random
import csv
import ffmpeg
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.utils import make_grid, save_image
from torchcodec.decoders import VideoDecoder

import pytorch_lightning as pl

from PIL import Image

from src.vqgan import ViTVQGAN


class KineticDataset(Dataset):

    def __init__(self,
        path: str, split: str,
        n_frames: int,
        size: int=256, output_size: int=256,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        mini_size: int=32
    ) -> None:
        self.path = path
        self.split = split
        self.n_frames = n_frames
        
        self.size = size
        self.output_size = output_size

        self.transform = Compose([
            Resize((size, size)),
            # ToTensor(),
            # Normalize(mean=mean, std=std)
        ])

        data = []
        file_fmtstr = "{ytid}_{start:06}_{end:06}.mp4"
        with open(f"{path}/annotations/{split}.csv", mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                f = file_fmtstr.format(
                    ytid=row["youtube_id"],
                    start=int(row["time_start"]),
                    end=int(row["time_end"]),
                )
                label = row["label"].replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")

                data.append((f, label))

        vitvq = ViTVQGAN.get_vit_vqgan_base()
        vitvq.init_from_ckpt("./checkpoint/imagenet_vitvq_base.ckpt")
        vitvq.eval()
        vitvq.cuda()
        self.data = []
        
        pathlib.Path(f"{path}/{split}_token").mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i, (video_path, label) in enumerate(data[:100]):
                print(f"\r{i}/{len(data)}", end="")

                p = pathlib.Path(f"{path}/{split}_token/{video_path[:-4]}.npy")
                if not p.is_file():
                    h, w = self._get_video_dim(f"{path}/{split}/{video_path}")
                    if h is None or w is None: continue
                    resize_h, resize_w = self._get_resize_dim(h, w)
                    cmd = (
                        ffmpeg
                        .input(f"{path}/{split}/{video_path}")
                        .filter('scale', resize_w, resize_h)
                    )
                    output_height, output_width = resize_h, resize_w
                    out, _ = (
                        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )

                    frames = np.frombuffer(out, np.uint8).reshape([-1, output_height, output_width, 3])
                    frames = torch.from_numpy(frames.astype('float32'))
                    frames = frames.permute(0, 3, 1, 2)
                    frames = frames / 255.
                    frames = self.transform(frames)

                    if frames is None: continue
                    if frames.ndim < 4: continue

                    token = torch.cat([
                        vitvq.encode_codes(frames[i*mini_size:(i+1)*mini_size].cuda())
                        for i in range(math.ceil(frames.shape[0]//mini_size + 0.5))
                    ], dim=0)
                    np.save(p, token.cpu())
                token = np.load(p)
                if len(token) < n_frames: continue
                self.data.append((p, label))
                # break
        del vitvq
    
    @classmethod
    def _get_video_dim(cls, video_path):
        """get video (height, width) from video path"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams']
                                    if stream['codec_type'] == 'video'), None)
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            return height, width
        except Exception as e:
            # print(e.stderr)
            return None, None

    def _get_resize_dim(self, raw_h, raw_w):
        """returns a new (height, width) so that the shorter side is self.size"""
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif raw_h >= raw_w:
            return int(raw_h * self.size / raw_w), self.size
        else:
            return self.size, int(raw_w * self.size / raw_h)
    
    def __len__(self): return len(self.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        file, label = self.data[index]
        
        token = np.load(file)  # (T)
        rand_idx = random.randint(0, len(token) - self.n_frames)

        return token[rand_idx:rand_idx+self.n_frames], label


# class KineticDataModule(pl.LightningDataModule):

#     def __init__(self,
#         data_dir: str,
#         batch_size: int=32
#     ):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size

#     def setup(self, stage: str):
#         self.train = MNIST(self.data_dir, split="train")
#         self.val = MNIST(self.data_dir, split="val")
#         self.test = MNIST(self.data_dir, split="test")

#     def train_dataloader(self):
#         return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

#     def teardown(self, stage: str): pass
