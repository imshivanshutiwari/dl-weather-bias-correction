import os
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import datasets

import glob


class VGG_TropDataset(Dataset):
    
    def __init__(self, root_dir, transform=None,):
        self.root_dir = root_dir
        self.transform = transform
        
        self.lr_path = glob.glob(root_dir + "*LR.nc")[0]
        self.hr_path = glob.glob(root_dir + "*HR.nc")[0]
        self.lr_data = xr.open_dataset(self.lr_path).rf.values
        self.hr_data = xr.open_dataset(self.hr_path).rf.values
        l1 = []
        for i in self.lr_data:
            x = Image.fromarray(i)
            x = x.resize((135,129))
            x = np.array(x)
            l1.append(x)
        self.lr_data = np.array(l1)
        
        self.length = len(self.lr_data) + len(self.hr_data)

    def __getitem__(self, index):
        if index >= len(self.lr_data):
            data = self.hr_data[index - len(self.lr_data)]
            label = 1
        else:
            data = self.lr_data[index]
            label = 0

        data = np.where(np.isnan(data)==True, 0, data)
        data = np.pad(data, ((1,2),(2,3)), 'edge')

        if self.transform:
            data = self.transform(data)

        return {"data" : data, "label" : label}

    def __len__(self):
        return self.length

