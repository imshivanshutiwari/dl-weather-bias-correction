import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
from tqdm import tqdm
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


data_path="/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/"
save_path="/home/hpcs_rnd/George_DIAT/Data/IMDAA_2019to23/"


lat_0125 = np.arange(6.5, 38.51, 0.125)
lon_0125 = np.arange(66.5, 100.1, 0.125)


out_data=[]
out_time=np.empty((0,),dtype='datetime64[ns]')
for x in os.listdir(data_path):
    if not x.startswith('ncum_imdaa'):
        continue
    
    print("Processing "+x)
    data1=xr.open_dataset(data_path+x).APCP_sfc.values
    time1=xr.open_dataset(data_path+x).time.values
    
    for i in tqdm(range(data1.shape[0]//3)):
        acc_3h=data1[3*i]+data1[3*i+1]+data1[3*i+2]
        
        image = Image.fromarray(acc_3h)
        image = image.resize((269, 257), Image.NEAREST)
        acc_3h=np.array(image)
        
        out_data.append(acc_3h)
    out_time=np.concatenate((out_time,time1[2::3]))
    

out_data=np.array(out_data)
out_data=np.where(out_data<0,0,out_data)
outDS=xr.Dataset(
    {
        'rf': (('time', 'lat', 'lon'), out_data)
    },
    coords={
        'lon': lon_0125,
        'lat': lat_0125,
        'time': out_time
    }
)

filename='IMDAA_3h.nc'
outDS.to_netcdf(save_path+filename)