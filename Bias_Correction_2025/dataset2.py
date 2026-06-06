import glob

import numpy as np
import pandas as pd
import xarray as xr
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


climatology_path="/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/scales/GFS_3h_max_climatology.json"

class GFSdataset(Dataset):
    
    def __init__(self, data_path, label_path, idxs, scale_path=climatology_path):
        print("Loading dataset")
        self.data=xr.open_dataset(data_path).rf.values[idxs[0]:idxs[1]]
        self.label=xr.open_dataset(label_path).rf.values[idxs[0]:idxs[1]]
        
        self.time=xr.open_dataset(data_path).time.values[idxs[0]:idxs[1]].astype('datetime64[h]')
        f = open(scale_path)
        self.scale_data = json.load(f)
        
        self.tensor_transform = torch.Tensor
        self.length=self.data.shape[0]
        
        
    def __getitem__(self, index):
        time1=self.time[index]
        
        data1=self.data[index,:,::-1,:].copy()
        #data1=self.normalize(data1,time1)
        dataT=self.tensor_transform(data1)
        
        label1=self.label[index]
        #label1=self.normalize(label1,time1)
        labelT=self.tensor_transform(label1)
        
        return {"gfs" : dataT, "imdaa" : labelT}
        
    def __len__(self):
        
        return self.length
    
    def normalize(self,arr,scale_time):
        scale_time = str(scale_time)[5:]
        scale_factor = self.scale_data[scale_time]
        
        return arr/scale_factor
        
    def normalize2(self,arr,scale_time):
        scale_time = str(scale_time)[5:]
        scale_factor = 400
        arr = arr/scale_factor
        arr = np.where(arr>1,1,arr)
        
        return arr