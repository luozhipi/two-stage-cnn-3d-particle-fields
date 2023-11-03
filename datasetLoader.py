import sys
import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import pandas as pd
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform=transform
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir, idx + '.tif')
        img_file = os.path.join(self.imgs_dir, idx + '.tif')
        mask = Image.open(mask_file)#读取到的是RGB， W, H, C
        mask = self.transform(mask)#transform转化image为：C, H, W
        img = Image.open(img_file)
        img = self.transform(img)
        return {
            'image':img,
            'mask': mask
        }

class preBasicDataset(Dataset):
    def __init__(self, imgs_dir, transformer):
        self.imgs_dir = imgs_dir
        self.transformer = transformer
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = os.path.join(self.imgs_dir, idx + '.tif')
        img = Image.open(img_file)
        img = self.transformer(img)
        return {
            'image':img,
            'id': idx+'.tif'
        }
class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')