from typing import Tuple, Union
# import os
from PIL import Image
import os
import numpy as np

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.utils import make_grid, save_image
# from torchcodec.decoders import VideoDecoder


class KittiVideoDataset(Dataset):

    def __init__(self,
        root: str, split: str,
        n_frames: int,
        size: Union[int, Tuple]=(384, 1248),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ) -> None:
        self.root, self.split = root, split
        self.n_frames = n_frames

        self.transform = Compose([
            Resize(size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

        self.data = []
        for dirpath, dirnames, filenames in os.walk(f"{root}/{split}/image_02"):
            if len(filenames) < n_frames: continue
            
            for idx, _ in enumerate(filenames[:-n_frames]):
                self.data.append([
                    f"{dirpath}/{filenames[idx+i]}"
                    for i in range(n_frames)
                ])
        
    def __len__(self): return len(self.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        paths = self.data[index]

        frames = [
            Image.open(path).convert("RGB")
            for path in paths
        ]
            
        frames = [self.transform(f) for f in frames]

        return torch.stack(frames, dim=0)#.permute(0, 3, 1, 2)

    @classmethod
    def get_ds(cls, root, split, n_frames, size):
        return cls(
            root, split,
            n_frames=n_frames,
            size=size,
        )
