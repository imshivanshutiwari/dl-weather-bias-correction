import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
from tqdm import tqdm
import json

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


data_path="/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/"
save_path="/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/"
files=os.listdir(data_path)
files.sort()

lat_0125 = np.arange(6.5, 38.51, 0.125)
lon_0125 = np.arange(66.5, 100.1, 0.125)


# Accumulation

out_data=[]
for x in files:
    if not x.startswith('GFS_JJASindia'):
        continue
    
    print("Processing "+x)
    data1=xr.open_dataset(data_path+x).rf.values
    
    for i in tqdm(range(data1.shape[0]//81)):
        for j in range(10):
            acc_daily=np.sum(data1[i*81+j*8:i*81+(j+1)*8],axis=0)
            
            out_data.append(acc_daily)

    

out_data=np.array(out_data)
out_data=np.where(out_data<0,0,out_data)
out_data=np.where(np.isnan(out_data)==True,0,out_data)
outDS=xr.Dataset(
    {
        'rf': (('index', 'lat', 'lon'), out_data)
    },
    coords={
        'lon': lon_0125,
        'lat': lat_0125,
        'index': np.arange(out_data.shape[0])
    }
)

filename='gfs_acc_daily.nc'
outDS.to_netcdf(save_path+filename)


# Labeling

data=xr.open_dataset("/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/gfs_acc_daily.nc")
data1=data.rf.values

data2=np.reshape(data1,new_shape=(600,10,257,269))

data3=np.flip(data2, axis=0)

data4=[]

for j in range(5):
    data3_slice=data3[120*j:120*(j+1)]
    for i in tqdm(range(9,-111,-1)):
        sample=data3_slice.diagonal(offset=i, axis1=0, axis2=1)
        sample=np.transpose(sample,(2,0,1))
        
        if sample.shape[0]<10:
            sample = np.concatenate([sample[:1]] * (10-sample.shape[0]) + [sample], axis=0)
        data4.append(sample)
        
data4=np.array(data4)

date_range = pd.date_range(start='2019-06-01', end='2023-09-30', freq='D')
date_range = date_range[date_range.month >= 6]
date_range = date_range[date_range.month <= 9]
date_range = date_range[date_range.day != 31]
times=np.array(date_range)

ensemble_num=np.arange(10)


outDS=xr.Dataset(
    {
        'rf': (('time','ensemble', 'lat', 'lon'), data4)
    },
    coords={
        'lon': lon_0125,
        'lat': lat_0125,
        'ensemble': ensemble_num,
        'time':times
    }
)

filename='GFS_daily_2019to23.nc'
outDS.to_netcdf(save_path+filename)
