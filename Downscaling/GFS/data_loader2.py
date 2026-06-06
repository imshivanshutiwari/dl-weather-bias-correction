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
        s1=self.data.shape
        #self.data=self.data.reshape(s1[0]*s1[1],s1[2],s1[3])
        
        self.label=xr.open_dataset(label_path).rf.values[idxs[0]:idxs[1]]
        s2=self.label.shape
        #self.label=self.label.reshape(s2[0]*s2[1],s2[2],s2[3])
        
        self.time=xr.open_dataset(label_path).time.values[idxs[0]:idxs[1]].astype('datetime64[h]')
        f = open(scale_path)
        self.scale_data = json.load(f)
        
        
        self.tensor_transform = torch.Tensor
        self.length=s2[0]*s2[1]
        
        
    def __getitem__(self, index):
        time1=self.time[index // 10]
        index1=index % 10
        
        
        data1=self.data[index//10,index1,::-1,:].copy()
        data1=self.normalize(data1,time1)
        dataT=self.tensor_transform(data1)
        dataT=dataT.unsqueeze(dim=0)
        
        label1=self.label[index//10,index1,::-1,:].copy()
        label1=self.normalize(label1,time1)
        labelT=self.tensor_transform(label1)
        labelT=labelT.unsqueeze(dim=0)
        
        return {"lr" : dataT, "hr" : labelT}
        
    def __len__(self):
        
        return self.length
    
    def normalize(self,arr,scale_time):
        scale_time = str(scale_time)[5:]
        scale_factor = self.scale_data[scale_time]
        
        return arr/scale_factor