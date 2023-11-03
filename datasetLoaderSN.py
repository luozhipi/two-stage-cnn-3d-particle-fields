import os
from os.path import splitext
from os import listdir
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2 as cv
import pandas as pd


class BasicDataset(Dataset):
    def __init__(self, z_dir, transformer):
        self.z_dir = z_dir
        self.transformer = transformer
        df = pd.read_csv(self.z_dir, dtype={'fileName':str, 'z':int})
        self.ids = df['fileName']
        self.zvalues = df['z']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_file = self.ids[i]
        z = self.zvalues[i]
        img = Image.open(img_file)
        img = self.transformer(img)
        return {
            'image': img,
            'z': z
        }

class BasicDataset2(Dataset):
    def __init__(self, z_dir, transformer):
        self.z_dir = z_dir
        self.transformer = transformer
        df = pd.read_csv(self.z_dir, dtype={'fileName':str, 'z':int})
        self.ids =df['fileName']
        self.zvalues = df['z']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_file = self.ids[i]
        z = self.zvalues[i]
        img = Image.open(img_file)
        img = self.transformer(img)
        return {
            'image': img,
            'z_truth': z,
            'id':img_file
        }
    
class preBasicDataset(Dataset):
    def __init__(self, csv_dir, transform):
        self.csv_dir = csv_dir
        self.transform = transform
        df = pd.read_csv(self.csv_dir, dtype={'fileName':str, 'imgID':str, 'x':int, 'y':int})
        self.filenames = df['fileName']
        self.ids = df['imgID']
        self.xvalues = df['x']
        self.yvalues = df['y']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        self.filenames[i]
        img = cv.imread(self.filenames[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img = Image.open(self.filenames[i])
        img = self.transform(img)
        x = self.xvalues[i]
        y = self.yvalues[i]
        idx = self.ids[i]
        return {
            'image': img,
            'x': x,
            'y': y, 
            'id': idx+'.tif'
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')