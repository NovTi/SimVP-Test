import gzip
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

import pdb
from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

from .dl_transforms import GroupNormalize, Stack, RandomHorizontalFlip

class DL(Dataset):

    def __init__(self, root, is_train=True, n_frames_input=11, n_frames_output=11,
                 transform=None):
        super(DL, self).__init__()
        self.is_train = is_train

        broken_lst = [6751, 14879, 6814, 3110]
        self.video_lst = []
        if self.is_train:  # train 
            for i in range(12300):
                if (2000+i) not in broken_lst:
                    self.video_lst.append(f"video_{2000+i}")
        else:   # valid
            for i in range(12300, 13000):
                if (2000+i) not in broken_lst:
                    self.video_lst.append(f"video_{2000+i}")

        self.resize = transforms.Resize((64, 64))
        self.transform = transforms.Compose([
            Stack(p=0.0),
            transforms.ToTensor(),
            RandomHorizontalFlip(p=0.3),
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # use imagenet default mean/std
            Rearrange("(t c) h w -> t c h w", t=22)
        ])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __getitem__(self, idx):
        # first resize
        img_lst = [
            self.resize(Image.open(os.path.join('../../../dataset/dl/unlabeled', f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(22)
        ]

        img_lst = self.transform(img_lst)
        
        return img_lst[:11], img_lst[11:]  # input / output

    def __len__(self):
        return len(self.video_lst)


def load_data(batch_size, val_batch_size, data_root,
              num_workers=10, pre_seq_length=11, aft_seq_length=11):

    train_set = DL(root=data_root, is_train=True,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length)
    test_set = DL(root=data_root, is_train=False,
                           n_frames_input=pre_seq_length,
                           n_frames_output=aft_seq_length)
    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)
    dataloader_vali = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)

    return dataloader_train, dataloader_vali, dataloader_test
