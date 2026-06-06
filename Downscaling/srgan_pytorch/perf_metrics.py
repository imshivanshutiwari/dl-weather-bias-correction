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


def denormalize(arr,date):
    date = date.astype('datetime64[s]').astype('O')
    date = date.strftime('%Y-%m-%d')
    d = date.split('-')
    day, month, year = d[-1], d[1], d[0]
    scale_date = f'{month}-{day}'
    scale_factor = scale_data[scale_date]
    
    return arr*scale_factor


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)


#model_path="/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/checkpoints/Make_Dataset/Set0/Mar-19-2025_adv5/GEN_Val-PSNR_42.708.pth"
#model_path="/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/checkpoints/Make_Dataset/Set1/Mar-21-2025_adv1/GEN_Val-PSNR_41.518.pth"
#model_path="/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/checkpoints/Make_Dataset/Set2/Mar-22-2025_adv1/GEN_Val-PSNR_42.678.pth"
#model_path="/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/checkpoints/Make_Dataset/Set3/Mar-22-2025_1/Val-PSNR_44.176.pth"
#model_path="/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/checkpoints/Make_Dataset/Set4/Mar-22-2025_1/Val-PSNR_44.728.pth"
model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/checkpoints/SRGAN_2025/May-06-2025_1/Val-PSNR_44.000.pth"

generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path))
generator.to(device=device)


hr_path="/home/users/bipink/DIAT/George_Jose/Data/SRGAN_4x/data_files/HR_1975to2023.nc"
lr_path="/home/users/bipink/DIAT/George_Jose/Data/NORM_1975to2023/Combined/MPNorm_LR.nc"

save_path="/home/users/bipink/DIAT/George_Jose/Report_figures/"

timeSet=[pd.to_datetime(['1975-01-01','1983-12-31']).tolist(),
         pd.to_datetime(['1984-01-01','1992-12-31']).tolist(),
         pd.to_datetime(['1993-01-01','2001-12-31']).tolist(),
         pd.to_datetime(['2002-01-01','2010-12-31']).tolist(),
         pd.to_datetime(['2011-01-01','2014-12-31']).tolist(),
         pd.to_datetime(['2015-01-01','2023-12-31']).tolist(),
         pd.to_datetime(['2021-01-01','2023-12-31']).tolist()
         ]
setnum=6

hr_data=xr.open_dataset(hr_path).sel(TIME=slice(timeSet[setnum][0],timeSet[setnum][1])).RAINFALL.values
lr_data=xr.open_dataset(lr_path).sel(time=slice(timeSet[setnum][0],timeSet[setnum][1])).rf.values
#hr_data=np.where(np.isnan(hr_data)==True,0,hr_data)
lr_data=np.where(np.isnan(lr_data)==True,0,lr_data)

mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/Masks/Mask_0625deg.npy"
mask1=np.load(mask_path)[::-1]

scale_path="/home/users/bipink/DIAT/George_Jose/Data/SRGAN_scales/scales1975to2023.json"
f = open(scale_path)
scale_data = json.load(f)

time1=xr.open_dataset(hr_path).sel(TIME=slice(timeSet[setnum][0],timeSet[setnum][1])).TIME.values

sr_data = []

for index, value in enumerate(tqdm(lr_data)):
    sample=lr_data[index]
    sample=np.where(np.isnan(sample)==True,0,sample)
    sample=torch.from_numpy(sample).unsqueeze(dim=0).unsqueeze(dim=0).to(device=device, dtype=torch.float)
    #sample=normalize(sample,time1[index])
    #sample=np.stack((sample,oro_data),axis=0)
    #sample=torch.from_numpy(sample).unsqueeze(dim=0).to(device=device, dtype=torch.float)

    with torch.no_grad():
        out_1 = generator(sample)
    
    out_1=out_1.squeeze().detach().cpu().numpy()
    out_1=denormalize(out_1,time1[index])
    out_1=np.where(np.isnan(hr_data[1])==True,np.nan,out_1[1:-2,2:-3])
    sr_data.append(out_1)

sr_data=np.array(sr_data)
sr_data=np.where(sr_data<0,0,sr_data)


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
#corr_s=corr_s.reshape(len(corr_s),1)


fig, ax = plt.subplots()
plt.xticks([i/10 for i in range(-1,11)])
plt.xlabel("Correlation")
ax=sns.kdeplot(corr_s)

fig_data = ax.lines[0].get_xydata()
max_xy=fig_data[np.where(fig_data[:, 1] == max(fig_data[:, 1]))].flatten()
ax.axvline(x=max_xy[0],color='r',linestyle='dashed')

plt.show()

#plt.savefig(save_path+"correlationKDE_set"+str(setnum)+".png",dpi=400)
#plt.close('all')


