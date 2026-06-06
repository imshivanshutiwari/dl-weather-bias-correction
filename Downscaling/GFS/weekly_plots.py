import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from models import Generator


def normalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    
    return arr/scale_factor
    
def denormalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    
    return arr*scale_factor


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

data_path="/home/users/bipink/DIAT/George_Jose/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23.nc"
save_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/figures/"
scale_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/scales/GFS_3h_max_climatology.json"


model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/checkpoints/SRGAN_GFS/Jun-10-2025_adv1/GEN_Val-PSNR_45.811.pth"

oro_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/IMD_DATA/oro.nc"
oro_data = xr.open_dataset(oro_path).topology.values
oro_data = oro_data / np.nanmax(oro_data)
oro_data = np.where(np.isnan(oro_data)==True, 0, oro_data)
oro_data = np.where(oro_data < 0, 0, oro_data)

f = open(scale_path)
scale_data = json.load(f)

mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/masks/Mask_03125deg.npy"
mask1=np.load(mask_path)[-2:1:-1,1:-2]
mask2=np.load("/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/masks/Mask_125deg_New.npy")[::-1]


lat_03125 = np.arange(6.5, 38.51, 0.03125)
lon_03125 = np.arange(66.5, 100.01, 0.03125)
lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)

#generator = Generator()
generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
generator.to(device=device)



data1=xr.open_dataset(data_path).sel(index=0).rf.values
time1=xr.open_dataset(data_path).time.values.astype("datetime64[h]")

###############################################################################################################

out_4x = []

for index, value in enumerate(tqdm(data1)):
    sample=data1[index]
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
    out_1=np.where(np.isnan(mask1)==True,np.nan,out_1[1:-2,1:-2])
    out_4x.append(out_1)

out_4x=np.array(out_4x)
out_4x=np.where(out_4x<0,0,out_4x)

#mask_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/Masks/Mask_0625deg.npy"
#mask1=np.load(mask_path)[::-1]
#maskstack=np.repeat(mask1[np.newaxis, :, :], out_4x.shape[0], axis=0)
#out_4x=np.where(np.isnan(maskstack)==True,np.nan,out_4x)

outDS=xr.Dataset(
    {
        'rf': (('time', 'lat', 'lon'), out_4x)
    },
    coords={
        'lon': lon_03125,
        'lat': lat_03125,
        'time': time1
    }
)


#filename='SR_Set'+str(setnum)+'.nc'
#outDS.to_netcdf(save_path+filename)

##########################################################################################################################

years=[2023]
monthnum=3
weeknum=1
startday=[152,182,213,244][monthnum-1]
month_name=["June","July","Aughust","September"][monthnum-1]

#data1=xr.open_dataset(data_path).sel(TIME=slice(timeSet[setnum][0],timeSet[setnum][1]))
data1=xr.open_dataset(data_path).sel(index=0)
data1=data1.isel(time=data1.time.dt.year.isin(years))

data_set=outDS
data_set=data_set.isel(time=data_set.time.dt.year.isin(years))


# week1 (week 23 in year) start from day 152
lrweek_data=data1.isel(time=data1.time.dt.dayofyear.isin(list(range(startday+7*(weeknum-1),startday+7*weeknum))))
lr_weekly=np.sum(lrweek_data.rf.values,axis=0)
lr_weekly=np.where(np.isnan(mask2)==True,np.nan,lr_weekly)

week_data=data_set.isel(time=data_set.time.dt.dayofyear.isin(list(range(startday+7*(weeknum-1),startday+7*weeknum))))
weekly_data=np.sum(week_data.rf.values,axis=0)


all_values = lr_weekly.flatten()
all_values = all_values[~np.isnan(all_values)]

p60 = max(np.ceil(np.percentile(all_values, 60)),1.0)
p90 = np.ceil(np.percentile(all_values, 90))
p99 = np.ceil(np.percentile(all_values, 99))

levels_1 = [x for x in np.arange(0, p60, p60/2)]
levels_2 = [x for x in np.arange(p60, p90, (p90-p60)/4)]
levels_3 = [x for x in np.arange(p90, p99+0.01, (p99-p90)/6)]
levels = np.unique(np.concatenate([levels_1, levels_2, levels_3]))


#Lev=list(range(0,10,1))
Lev=levels
Cmap="rainbow"
#Cmap="Reds"

plt.figure(figsize=(10,5))
plt.suptitle('Week'+str(weeknum)+' '+month_name+' '+str(years[0]))

plt.subplot(1,2,1)
plt.title('0.125x0.125')
plt.contourf(lon_125, lat_125, lr_weekly,
                cmap=Cmap, levels=Lev, extend = "max")
plt.colorbar()

plt.subplot(1,2,2)
plt.title('0.03125x0.03125')
plt.contourf(lon_03125, lat_03125, weekly_data,
                cmap=Cmap, levels=Lev, extend = "max")
plt.colorbar()

#plt.show()
plt.savefig("/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/figures/week"+str(weeknum)+month_name+str(years[0])+".png",dpi=400)
