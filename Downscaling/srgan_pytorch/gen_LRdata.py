import xarray as xr
import numpy as np

import os
import sys

import torch
from torch.nn.functional import max_pool2d,avg_pool2d,pad



data_path="/home/users/bipink/DIAT/George_Jose/Data/NORM_1975to2023/Train/trainNorm_HR.nc"
save_path="/home/users/bipink/DIAT/George_Jose/Data/NORM_1975to2023/Train/"
mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/Masks/Mask_1deg.npy"

lr_path="/home/users/bipink/DIAT/George_Jose/Data/NORM_1975to2023/Train/trainNorm_LowRes.nc"


hr_data=xr.open_dataset(data_path)
lr_data=xr.open_dataset(lr_path)

x=hr_data.rf.values
x=np.where(np.isnan(x)==True,0,x)
y=torch.tensor(x)
y=pad(y,(2,3,1,2))
y=y.unsqueeze(dim=1)

z=avg_pool2d(y,(2,2))
z=max_pool2d(z,(2,2))
z=z.squeeze()
z1=np.array(z)

mask1=np.load("/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/Masks/Mask_1deg.npy")[::-1]
maskstack=np.repeat(mask1[np.newaxis, :, :], x.shape[0], axis=0) 
z2=np.where(np.isnan(maskstack)==True,np.nan,z1)

lr_data.rf.values=z2

lr_data.to_netcdf(save_path+'trainMPNorm_LR.nc')