import pygrib
import numpy as np
import xarray as xr

import os
import sys
from tqdm import tqdm



#grib = "/home/hpcs_rnd/George_DIAT/Daily_Model_IMD_Data/2020/gfsrain_2020060100.grib"
file_path="/home/hpcs_rnd/Montly_IMD_model_data/imdgfs_rain/2019/"
files=[x for x in os.listdir(file_path) if (x.endswith('grib')) and ('_2019' in x)]
files.sort()

save_path="/home/hpcs_rnd/George_DIAT/Data/GFS_model_data/"

data=[]
for grib in files:
    print("Processing: "+grib)
    grbs = pygrib.open(file_path+grib)
    
    for i,x in enumerate(tqdm(grbs)):
        data.append(x.values[668:411:-1,532:801])
    
    grbs.close()


data=np.array(data,dtype=np.float32)

'''
print("Shape: ", data.shape)
print("Minimum: ", np.min(data))
print("Maximum: ", np.max(data))
print("Mean: ", np.mean(data))
print("Variance: ",np.var(data))
'''

#data_india=data[:,668:411:-1,532:801]
mask125=np.load("/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/Masks/Mask_125deg_New.npy")
mask_stack=np.repeat(mask125[np.newaxis,:,:],9720,axis=0)
data=np.where(np.isnan(mask_stack)==True,np.nan,data)
'''
print("Shape: ", data_india.shape)
print("Minimum: ", np.nanmin(data_india))
print("Maximum: ", np.nanmax(data_india))
print("Mean: ", np.nanmean(data_india))
print("Variance: ",np.nanvar(data_india))
'''

grbs = pygrib.open(file_path+grib)
grb = grbs.select()[0]
lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni']) )[532:801]
lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj']) )[668:411:-1]
grbs.close()


outDS=xr.Dataset(
        {
            'rf': (('index', 'lat', 'lon'), data)
        },
        coords={
            'lon': lons,
            'lat': lats,
            'index': np.arange(data.shape[0])
        }
    )
    
filename='GFS_JJASindia_2019.nc'
outDS.to_netcdf(save_path+filename)