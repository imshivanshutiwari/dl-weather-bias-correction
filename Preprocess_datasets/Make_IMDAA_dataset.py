import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
import torch
import torch.nn.functional as F

from tqdm import tqdm
import gc
import json

import os
from pathlib import Path
from datetime import datetime, timedelta

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


save_path = "/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/"
imdaa_path = "/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/ncum_imdaa_merged.nc"
imd_4x_path = "/home/hpcs_rnd/George_DIAT/Data/SRGAN_4x_7yTestSets_1975to2023/Set0_2017to2023_0625deg.nc"




lat_0125 = np.arange(6.5, 38.51, 0.125)
lon_0125 = np.arange(66.5, 100.1, 0.125)
time_3h = np.load("/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/jjas_datetime_3hr_incl_31s_and_oct1.npy")

mask125=np.load("/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/Masks/Mask_125deg_New.npy")[::-1]

imdaa_dataset=xr.Dataset(
        {
            'rf': (('time', 'lat', 'lon'), np.zeros((time_3h.shape[0],lat_0125.shape[0],lon_0125.shape[0])))
        },
        coords={
            'lon': lon_0125,
            'lat': lat_0125,
            'time' : time_3h
        }
    )

imdaa_data=xr.open_dataset(imdaa_path)
imd_data=xr.open_dataset(imd_4x_path)


for ts in tqdm(time_3h):
    
    try:
        data1=imdaa_data.sel(time = ts - np.timedelta64(2, 'h')).APCP_sfc.values
        data2=imdaa_data.sel(time = ts - np.timedelta64(1, 'h')).APCP_sfc.values
        data3=imdaa_data.sel(time = ts).APCP_sfc.values
        acc_3h=data1+data2+data3
        image = Image.fromarray(acc_3h)
        image = image.resize((269, 257), Image.NEAREST)
        acc_3h=np.array(image)
        acc_3h=np.where(np.isnan(mask125)==True, 0, acc_3h)
    except KeyError:
        ts_d=ts.astype('datetime64[D]')
        acc_1d=imd_data.sel(time = ts_d).rf.values
        acc_1d=np.where(np.isnan(acc_1d)==True, 0, acc_1d)
        acc_3h=acc_1d/8
        acc_3h=torch.tensor(acc_3h)
        acc_3h=F.pad(acc_3h,(1,1,1,1))
        acc_3h=acc_3h.unsqueeze(dim=0).unsqueeze(dim=0)
        acc_3h=F.max_pool2d(acc_3h,(2,2))
        acc_3h=acc_3h.squeeze().squeeze()
        acc_3h=np.array(acc_3h)
    
    imdaa_dataset["rf"].loc[dict(time=ts)] = acc_3h
    
    gc.collect()

filename='IMDAA_3h_JJAS_india_2019to23_filled.nc'
imdaa_dataset.to_netcdf(save_path+filename)








