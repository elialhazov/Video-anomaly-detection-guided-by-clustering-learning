import numpy
from einops import rearrange

from misc.utils import save_tensor_video
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import torch
from PIL import Image
import random
import torchvision.transforms.functional as tf
from torch import nn

rng = np.random.RandomState(2023)


def load_images(image_paths, transform):
    imgs = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            img_str = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)

        imgs.append(Image.fromarray(img))
    if transform is not None:
        video = transform(imgs)

    return video


def tensor_normalize(tensor, mean=1, std=0):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    return tensor


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, frames_num=10,
                 num_pred=1, istest=False, label_folder='.', is_load_label=True, image_format='jpg', index_num=3):
        self.dir = video_folder
        self.index_num = index_num
        self.image_format = '.' + image_format
        self.istest = istest
        self.is_load_label = is_load_label
        self.transform = transform
        self.videos = OrderedDict()
        self.frames_num = frames_num
        self._num_pred = num_pred
        self.label_folder = label_folder
        self.labels = self.get_all_video_labels()
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = os.path.basename(video)
            self.videos[video_name] = {
                'path': video,
                'frame': sorted(glob.glob(os.path.join(video, f'*{self.image_format}')))
            }
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_hole_video(self, video_path):
        video_path = glob.glob(os.path.join(video_path, f'*{self.image_format}'))  # <-- updated
        video_path.sort()
        return video_path

    def get_one_video(self, video_path, image_format, index_num=4):
        video_dir = os.path.dirname(video_path)
        all_files = sorted(glob.glob(os.path.join(video_dir, f'*{image_format}')))
        try:
            start_idx = all_files.index(video_path)
        except ValueError:
            print(f"[ERROR] {video_path} not found in {video_dir}")
            return []
        return all_files[start_idx:start_idx + self.frames_num]

    def get_all_video_labels(self):
        labels = []
        videos = glob.glob(os.path.join(self.label_folder, '*'))
        for video in sorted(videos):
            labels.append(video)
        return labels

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        if self.istest:
            return videos
        for video in sorted(videos):
            imgs = glob.glob(os.path.join(video, f'*{self.image_format}'))  # <-- updated
            for i, img_path in enumerate(imgs):
                if i <= (len(imgs) - self.frames_num):
                    frames.append(img_path)
        return frames

    def __getitem__(self, index):
        video_path = self.samples[index]
        if self.istest:
            scene_num = os.path.split(video_path)[-1].split('_')[0]
            video_name = os.path.split(video_path)[-1]
            if self.is_load_label:
                video_labels = np.load(os.path.join(self.label_folder, video_name) + '.npy')
                seq = self.get_hole_video(video_path)
                frames = load_images(seq, self.transform)
                frames = frames.permute(1, 0, 2, 3)
                return frames, index, video_labels, scene_num
            else:
                seq = self.get_hole_video(video_path)
                frames = load_images(seq, self.transform)
                frames = frames.permute(1, 0, 2, 3)
                return frames, index, scene_num

        seq = self.get_one_video(video_path, self.image_format, self.index_num)
        frames = load_images(seq, self.transform)
        frames = frames.permute(1, 0, 2, 3)
        return frames, index

    def __len__(self):
        return len(self.samples)


class DataTransforms(object):
    def __init__(self, size=(112, 112)):
        self.tranfors = torch.nn.Sequential(
            Resize_Normalize(size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    def __call__(self, video):
        return self.tranfors(video)


class Resize_Normalize(nn.Module):
    def __init__(self, size, mean, std):
        super().__init__()
        self.size = size
        self.mean = mean
        self.std = std

    def forward(self, video):
        out = []
        for img in video:
            img = np.array(tf.resize(img, size=self.size)).astype(np.float32)
            img = img / 255
            img = torch.tensor(img)
            img = rearrange(img, 'H W C -> C H W')
            out.append(img)
        return torch.stack(out)
