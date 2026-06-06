import os
import sys

import torch
from sklearn.neighbors import KernelDensity

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from srgan_utils import PSNR
from models import Generator
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

def normalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    
    return arr/scale_factor
    
def denormalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    
    return arr*scale_factor


hr_path="/home/users/bipink/DIAT/George_Jose/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23.nc"
lr_path="/home/users/bipink/DIAT/George_Jose/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23_05deg.nc"
oro_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/IMD_DATA/oro.nc"
mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/masks/Mask_03125deg.npy"
scale_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/scales/GFS_3h_max_climatology.json"
save_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/figures/"

model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/checkpoints/SRGAN_GFS/Jun-10-2025_adv1/GEN_Val-PSNR_48.320.pth"


f = open(scale_path)
scale_data = json.load(f)

lat_03125 = np.arange(6.5, 38.51, 0.03125)
lon_03125 = np.arange(66.5, 100.01, 0.03125)
lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)

mask1=np.load(mask_path)[-2:1:-1,1:-2]
mask2=np.load("/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/masks/Mask_125deg_New.npy")[::-1]


generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path,map_location='cpu'))
generator.to(device=device)


hr_data=xr.open_dataset(hr_path).sel(index=1).rf.values
lr_data=xr.open_dataset(lr_path).sel(index=1).rf.values
#hr_data=np.where(np.isnan(hr_data)==True,0,hr_data)
lr_data=np.where(np.isnan(lr_data)==True,0,lr_data)


time1=xr.open_dataset(hr_path).sel(index=1).time.values.astype('datetime64[h]')
sr_data = []
for index, value in enumerate(tqdm(lr_data)):
    sample=lr_data[index]
    sample=np.where(np.isnan(sample)==True,0,sample)
    sample=normalize(sample,time1[index])
    sample=torch.from_numpy(sample).unsqueeze(dim=0).unsqueeze(dim=0).to(device=device, dtype=torch.float)
    #sample=normalize(sample,time1[index])
    #sample=np.stack((sample,oro_data),axis=0)
    #sample=torch.from_numpy(sample).unsqueeze(dim=0).to(device=device, dtype=torch.float)

    with torch.no_grad():
        out_1 = generator(sample)
    
    out_1=out_1.squeeze().detach().cpu().numpy()
    out_1=denormalize(out_1,time1[index])
    out_1=np.where(np.isnan(mask2)==True,np.nan,out_1[1:-2,1:-2])
    sr_data.append(out_1)

sr_data=np.array(sr_data)
sr_data=np.where(sr_data<0,0,sr_data)

hr_data=np.where(np.isnan(sr_data)==True,np.nan,hr_data)

corrs=np.zeros(hr_data.shape[1:],dtype=float)

def correlation(x,y,z):
    shp=x.shape
    for i in range(shp[1]):
        for j in range(shp[2]):
            z[i,j] = np.corrcoef(x[:,i,j], y[:,i,j])[0,1]

    return

correlation(hr_data,sr_data,corrs)
corr_s=corrs.flatten()
corr_s=corr_s[np.logical_not(np.isnan(corr_s))]

fig, ax = plt.subplots()
plt.xticks([i/10 for i in range(-1,11)])
plt.xlabel("Correlation")
ax=sns.kdeplot(corr_s)
ax.set_xlim([0.0,1.1])

fig_data = ax.lines[0].get_xydata()
max_xy=fig_data[np.where(fig_data[:, 1] == max(fig_data[:, 1]))].flatten()
print("max=",max_xy)
ax.axvline(x=max_xy[0],color='r',linestyle='dashed')

#plt.show()

plt.savefig(save_path+"correlationKDE.png",dpi=400)
plt.close('all')


