import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

#sys.path.insert(0, "/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/")
from models import Generator


def normalize(arr,date):
    date = date.astype('datetime64[s]').astype('O')
    date = date.strftime('%Y-%m-%d')
    d = date.split('-')
    day, month, year = d[-1], d[1], d[0]
    scale_date = f'{month}-{day}'
    scale_factor = scale_data[scale_date]
    
    return arr/scale_factor
    
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

data_path="/home/users/bipink/DIAT/George_Jose/Data/NORM_1975to2023/Combined/"
save_path="/home/users/bipink/DIAT/George_Jose/Data/SRGAN_4x_7yTestSets_1975to2023/"
scale_path="/home/users/bipink/DIAT/George_Jose/Data/SRGAN_scales/scales1975to2023.json"

#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/ckps/SRGAN_5k/Feb-03-2022/Val-PSNR_40.587.pth"
model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/checkpoints/Make_Dataset2/Set0/May-12-2025_1/Val-PSNR_44.990.pth"

if len(sys.argv) > 1:
    if sys.argv[1]!="x":
        model_path=sys.argv[1]


#oro_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/IMD_DATA/oro.nc"
oro_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/IMD_DATA/ORO_SCALED.nc"
oro_data = xr.open_dataset(oro_path).topology.values
oro_data = oro_data / np.nanmax(oro_data)
oro_data = np.where(np.isnan(oro_data)==True, 0, oro_data)
oro_data = np.where(oro_data < 0, 0, oro_data)

f = open(scale_path)
scale_data = json.load(f)

mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/Masks/Mask_0625deg.npy"
mask1=np.load(mask_path)[::-1]


timeSet=[pd.to_datetime(['2017-01-01','2023-12-31']).tolist(),
         pd.to_datetime(['2010-01-01','2016-12-31']).tolist(),
         pd.to_datetime(['2003-01-01','2009-12-31']).tolist(),
         pd.to_datetime(['1996-01-01','2002-12-31']).tolist(),
         pd.to_datetime(['1989-01-01','1995-12-31']).tolist(),
         pd.to_datetime(['1982-01-01','1988-12-31']).tolist(),
         pd.to_datetime(['1975-01-01','1981-12-31']).tolist()
         ]
setnum=6


lat_0625 = np.arange(6.5, 38.5, 0.0625)
lon_0625 = np.arange(66.5, 100, 0.0625)

lat_25 = np.arange(6.5, 38.51, 0.25)
lon_25 = np.arange(66.5, 100.1, 0.25)

lat_0125 = np.arange(6.5, 38.51, 0.125)
lon_0125 = np.arange(66.5, 100.1, 0.125)

#generator = Generator()
generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path))
generator.to(device=device)


for x in os.listdir(data_path):
    if not x.endswith('HR.nc'):
        continue
    
    print("Processing "+x)
    data1=xr.open_dataset(data_path+x).sel(time=slice(timeSet[setnum][0],timeSet[setnum][1])).rf.values
    time1=xr.open_dataset(data_path+x).sel(time=slice(timeSet[setnum][0],timeSet[setnum][1])).time.values
    
    out_4x = []

    for index, value in enumerate(tqdm(data1)):
        sample=data1[index]
        sample=np.where(np.isnan(sample)==True,0,sample)
        sample=torch.from_numpy(sample).unsqueeze(dim=0).unsqueeze(dim=0).to(device=device, dtype=torch.float)
        #sample=normalize(sample,time1[index])
        #sample=np.stack((sample,oro_data),axis=0)
        #sample=torch.from_numpy(sample).unsqueeze(dim=0).to(device=device, dtype=torch.float)

        with torch.no_grad():
            out_1 = generator(sample)
        
        out_1=out_1.squeeze().detach().cpu().numpy()
        out_1=denormalize(out_1,time1[index])
        out_1=np.where(np.isnan(mask1)==True,np.nan,out_1[2:-2,2:-2])
        out_4x.append(out_1)
    
    out_4x=np.array(out_4x)
    out_4x=np.where(out_4x<0,0,out_4x)
    outDS=xr.Dataset(
        {
            'rf': (('time', 'lat', 'lon'), out_4x)
        },
        coords={
            'lon': lon_0625,
            'lat': lat_0625,
            'time': time1
        }
    )
    
    filename='Set'+str(setnum)+'_'+str(timeSet[setnum][0].year)+'to'+str(timeSet[setnum][1].year)+'.nc'
    outDS.to_netcdf(save_path+filename)
        
        