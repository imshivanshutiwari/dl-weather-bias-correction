import glob

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GFSdataset(Dataset):
    
    def __init__(self, data_path, label_path,idxs):
        print("Loading dataset")
        self.data=xr.open_dataset(data_path).rf.values[idxs[0]:idxs[1]]
        self.times=xr.open_dataset(data_path).time.values
        self.label=xr.open_dataset(label_path).sel(time=self.times).rf.values[idxs[0]:idxs[1]]
        
        #self.tensor_transform = transforms.Compose([transforms.ToTensor(),])
        self.tensor_transform = torch.Tensor
        self.length=self.data.shape[0]
        
        
    def __getitem__(self, index):
        data1=self.data[index,:,::-1,:].copy()
        dataT=self.tensor_transform(data1)
        labelT=self.tensor_transform(self.label[index])
        
        return {"gfs" : dataT, "imd" : labelT}
        
    def __len__(self):
        
        return self.length