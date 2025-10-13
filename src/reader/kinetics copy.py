import os
import json 
import random
import ffmpeg
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision.utils import make_grid, save_image

from PIL import Image


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def get_basename_no_ext(path):
    """ '/data/movienet/240p_keyframe_feats/tt7672188.npz' --> 'tt7672188' """
    return os.path.splitext(os.path.split(path)[1])[0]


class VideoDatasetBase(Dataset):

    def __init__(self,
        datalist, fps: int=2, size: int=224, output_size: int=224,
        crop_type: str="center", hflip: bool=False, data_ratio: float=1.,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    ):
        self.datalist = datalist
        crop_type2offset_factor = dict(top=0, center=1, bottom=2)
        self.crop_type = crop_type
        self.crop_offset_factor = None if self.crop_type == "none" \
            else crop_type2offset_factor[self.crop_type]
        self.hflip = hflip
        self.size = size
        self.output_size = output_size
        self.fps = fps
        self.normalize = Normalize(mean=mean, std=std)
        self.data_ratio = data_ratio
        # subsample data, mostly used for debug
        if data_ratio != 1.:
            random.shuffle(datalist)
            self.datalist = datalist[:int(len(datalist)*data_ratio)]
        self.counter = 0  # #video to display extracted frames
        self.check_steps = 1

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        raise NotImplementedError

    @classmethod
    def _get_video_dim(cls, video_path):
        """get video (height, width) from video path"""
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_resize_dim(self, raw_h, raw_w):
        """returns a new (height, width) so that the shorter side is self.size"""
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif raw_h >= raw_w:
            return int(raw_h * self.size / raw_w), self.size
        else:
            return self.size, int(raw_w * self.size / raw_h)

    def load_video_from_path(self, video_path):
        try:
            h, w = self._get_video_dim(video_path)
        except:
            print('ffprobe failed at: {}'.format(video_path))
            return torch.zeros(1)  # invalid

        # get frames at self.fps
        # resize with shorter side self.size
        resize_h, resize_w = self._get_resize_dim(h, w)
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self.fps)
            .filter('scale', resize_w, resize_h)
        )

        # take [top, center, bottom] crop
        if self.crop_type != "none":
            x = int(self.crop_offset_factor * (resize_w - self.output_size) / 2.0) 
            y = int(self.crop_offset_factor * (resize_h - self.output_size) / 2.0)
            cmd = cmd.crop(x, y, self.output_size, self.output_size)
            output_height, output_width = self.output_size, self.output_size
        else:  # take full spatial frame
            output_height, output_width = resize_h, resize_w

        if self.hflip:
            cmd = cmd.hflip()

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )

        frames = np.frombuffer(out, np.uint8).reshape([-1, output_height, output_width, 3])
        # check captured frames
        # for idx in range(min(5, len(frames))):
        #     save_path = get_basename_no_ext(video_path) + f"img_{idx}.jpg"
        #     save_np_array_to_img(frames[idx], save_path)
        frames = torch.from_numpy(frames.astype('float32'))
        frames = frames.permute(0, 3, 1, 2)  # (#frms, 3, height, width)
        # first [0, 255] --> [0, 1]; then [0, 1] --> [-1, 1]
        frames = frames / 255.
        show_rec = False
        if show_rec and self.counter < self.check_steps:  # visualization
            save_frm_path = "snap/debug/extracted_frames/" + get_basename_no_ext(video_path) \
                + f"_resize{self.size}_out_size{self.output_size}_crop_{self.crop_type}_hflip{int(self.hflip)}.jpg"
            grid = make_grid(frames[:9], nrow=3)
            save_image(grid, save_frm_path)
            self.counter += 1
            print(f"save at {save_frm_path}")
        frames = self.normalize(frames)
        return frames  # torch.Tensor (#frms, 3, height, width)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor