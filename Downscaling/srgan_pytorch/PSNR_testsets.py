import os
import sys

import torch

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


model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/checkpoints/Make_Dataset2/Set6/May-12-2025_1/Val-PSNR_43.567.pth"

generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path))
generator.to(device=device)


#hr_path="/home/hpcs_rnd/George_DIAT/Data/SRGAN_4x/data_files/HR_1975to2023.nc"
hr_path="/home/users/bipink/DIAT/George_Jose/Data/NORM_1975to2023/Combined/Norm_HR.nc"
lr_path="/home/users/bipink/DIAT/George_Jose/NORM_1975to2023/Combined/MPNorm_LR.nc"
oro_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/IMD_DATA/ORO_SCALED.nc"

save_path="/home/users/bipink/DIAT/George_Jose/Report_figures/"

timeSet=[pd.to_datetime(['2017-01-01','2023-12-31']).tolist(),
         pd.to_datetime(['2010-01-01','2016-12-31']).tolist(),
         pd.to_datetime(['2003-01-01','2009-12-31']).tolist(),
         pd.to_datetime(['1996-01-01','2002-12-31']).tolist(),
         pd.to_datetime(['1989-01-01','1995-12-31']).tolist(),
         pd.to_datetime(['1982-01-01','1988-12-31']).tolist(),
         pd.to_datetime(['1975-01-01','1981-12-31']).tolist(),
         pd.to_datetime(['2012-01-01','2016-12-31']).tolist()
         ]
setnum=6

hr_data=xr.open_dataset(hr_path).sel(time=slice(timeSet[setnum][0],timeSet[setnum][1])).rf.values
lr_data=xr.open_dataset(lr_path).sel(time=slice(timeSet[setnum][0],timeSet[setnum][1])).rf.values
#hr_data=np.where(np.isnan(hr_data)==True,0,hr_data)
lr_data=np.where(np.isnan(lr_data)==True,0,lr_data)
oro_data = xr.open_dataset(oro_path).topology.values

scale_path="/home/users/bipink/DIAT/George_Jose/Data/SRGAN_scales/scales1975to2023.json"
f = open(scale_path)
scale_data = json.load(f)

time1=xr.open_dataset(hr_path).sel(time=slice(timeSet[setnum][0],timeSet[setnum][1])).time.values

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
    #out_1=denormalize(out_1,time1[index])
    out_1=np.where(np.isnan(hr_data[1])==True,np.nan,out_1[1:-2,2:-3])
    sr_data.append(out_1)

sr_data=np.array(sr_data)
sr_data=np.where(sr_data<0,0,sr_data)
sr_data=np.where(np.isnan(sr_data)==True,0,sr_data)
sr_data=torch.Tensor(sr_data)

hr_data=np.where(np.isnan(hr_data)==True,0,hr_data)
hr_data=torch.Tensor(hr_data)

'''
def psnr_metric(img1, img2):
    mse = torch.mean((img1 - img2) ** 2, axis=(1,2))
    #return 20 * torch.log10(255.0 / torch.sqrt(mse))
    return torch.mean(20 * torch.log10(1.0 / torch.sqrt(mse)))'''

psnr_metric = PSNR()

print(psnr_metric(sr_data,hr_data))

mean_psnr=0
for i in range(sr_data.shape[0]):
    mean_psnr += psnr_metric(sr_data[0],hr_data[0]).item()

mean_psnr/=sr_data.shape[0]
print(mean_psnr)

