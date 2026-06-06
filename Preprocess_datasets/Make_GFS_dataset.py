import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygrib
import xarray as xr
from PIL import Image
from tqdm import tqdm
import gc
import json

import os
from pathlib import Path
from datetime import datetime, timedelta

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


save_path = "/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/"
folder = "/home/hpcs_rnd/Montly_IMD_model_data/imdgfs_rain/"
grib_files = []
for file_path in Path(folder).rglob("*.grib"):
    grib_files.append(str(file_path))


lat_0125 = np.arange(6.5, 38.51, 0.125)
lon_0125 = np.arange(66.5, 100.1, 0.125)
time_3h = np.load("/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/jjas_datetime_3hr_incl_31s_and_oct1.npy")

mask125=np.load("/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/Masks/Mask_125deg_New.npy")

gfs_dataset=xr.Dataset(
        {
            'rf': (('time', 'index', 'lat', 'lon'), np.zeros((time_3h.shape[0],10,lat_0125.shape[0],lon_0125.shape[0])))
        },
        coords={
            'lon': lon_0125,
            'lat': lat_0125,
            'index': np.arange(10),
            'time' : time_3h
        }
    )


def yyyymmdd_to_day_number(date_int):
    date_str = str(date_int)
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return date_obj.timetuple().tm_yday

def get_timestamp(grib_msg):
    year=grib_msg['validityDate']//10000
    month=grib_msg['validityDate']//100%100
    day=grib_msg['validityDate']%100
    hour=grib_msg['validityTime']//100

    timestamp=datetime(year, month, day, hour)
    timestamp=np.datetime64(timestamp, 'h')
    return timestamp

def get_index(grib_msg):
    index=yyyymmdd_to_day_number(grib_msg['validityDate']) - yyyymmdd_to_day_number(grib_msg['dataDate']) + [0,-1][grib_msg['validityTime']//100 == 0]
    return index


for grib in grib_files:
    print("Processing: "+grib)
    grbs = pygrib.open(grib)
    
    for i in tqdm(range(1,81)):
        data1=grbs[i].values[668:411:-1,532:801]
        data1=np.where(np.isnan(mask125)==True, 0, data1)
        index1=get_index(grbs[i])
        time1=get_timestamp(grbs[i])
        if time1 not in time_3h:
            continue
        gfs_dataset["rf"].loc[dict(time=time1, index=index1)] = data1
    grbs.close()
    gc.collect()


filename='GFS_3h_JJAS_india_2019to23.nc'
gfs_dataset.to_netcdf(save_path+filename)

'''
gfs_dataset=xr.open_dataset("/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23.nc")
missing0_times_ranges = [datetime(2019, 7, 31, 0), datetime(2020, 7, 31, 0), datetime(2021, 7, 31, 0), datetime(2022, 7, 31, 0), datetime(2023, 7, 31, 0),
                        datetime(2019, 8, 31, 0), datetime(2020, 8, 31, 0), datetime(2021, 8, 31, 0), datetime(2022, 8, 31, 0), datetime(2023, 8, 31, 0)]

for t in missing0_times_ranges:
    time_range = [t+timedelta(hours=3)*i for i in range(1,8+1)]
    for ts in time_range:
        gfs_dataset["rf"].loc[dict(time=ts, index=0)] = gfs_dataset.sel(time=ts, index=1).rf.values

'''
# grbs[80]['forecastTime']
# grbs[1]['validityTime']
# grbs[80]['validityDate']