import xarray as xr
import numpy as np

import os
import sys
from tqdm import tqdm
import gc

import torch
from torch.nn.functional import max_pool2d,avg_pool2d,pad


data_path="/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23.nc"
save_path="/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/"


lat_05 = np.arange(6.5, 38.51, 0.5)
lon_05 = np.arange(66.5, 100.1, 0.5)

time_3h = np.load("/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/jjas_datetime_3hr_incl_31s_and_oct1.npy")

gfs_dataset=xr.open_dataset(data_path)


x=gfs_dataset.rf.values
x=np.where(np.isnan(x)==True,0,x)
y=torch.tensor(x)
y=pad(y,(1,2,1,2))

z=avg_pool2d(y,(2,2))
z=max_pool2d(z,(2,2))
z1=np.array(z)


gfs_05deg_dataset=xr.Dataset(
        {
            'rf': (('time', 'index', 'lat', 'lon'), z1)
        },
        coords={
            'lon': lon_05,
            'lat': lat_05,
            'index': np.arange(10),
            'time' : time_3h
        }
    )


gfs_05deg_dataset.to_netcdf(save_path+'GFS_3h_JJAS_india_2019to23_05deg.nc')