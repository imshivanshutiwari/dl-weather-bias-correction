import pandas as pd
import xarray as xr
import pygrib
import numpy as np
from datetime import datetime

import sys
import os
from pathlib import Path
import json
from tqdm import tqdm


save_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/Bias_Correction_2025/scales/"
folder="/home/hpcs_rnd/Montly_IMD_model_data/imdgfs_rain/"
grib_files = []
for file_path in Path(folder).rglob("*.grib"):
    grib_files.append(str(file_path))

mask125=np.load("/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/Masks/Mask_125deg_New.npy")

maxRF={}


def get_timestamp(grib_msg):
    year=grib_msg['validityDate']//10000
    month=grib_msg['validityDate']//100%100
    day=grib_msg['validityDate']%100
    hour=grib_msg['validityTime']//100

    timestamp=datetime(year, month, day, hour)
    timestamp=np.datetime64(timestamp, 'h')
    return timestamp


for grib in tqdm(grib_files):
    print("Processing: "+grib)
    grbs = pygrib.open(grib)
    
    for i in tqdm(range(1,82)):
        data1=grbs[i].values[668:411:-1,532:801]
        data1=np.where(np.isnan(mask125)==True, 0, data1)
        maxval=np.max(data1.flatten())
        
        time1=get_timestamp(grbs[i])
        scale_time=str(time1)[5:]
        
        if scale_time in maxRF:
            if maxval>maxRF[scale_time]:
                maxRF[scale_time]=maxval
        else:
            maxRF[scale_time]=maxval
    grbs.close()
    gc.collect()
        
with open(save_path+"GFS_3h_max_climatology.json", "w") as scale_file:
    json.dump(maxRF,scale_file)