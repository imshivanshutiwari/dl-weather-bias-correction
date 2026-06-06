import tensorflow as tf
from tensorflow.keras.models import load_model

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]= "4"


#cnnbc = load_model("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/models/my_cnntry8model.h5")
cnnbc = load_model("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/models/cnnbcmodel_1.h5")

train_year=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/train_year.npy")
test_year=np.load("/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/npy_files/test_year.npy")

#data = np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/CFS_rainfall_cnnbc.npy")
data = np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/data_10x32x32.npy")
#label = np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/pr_rainfall_cnnbc.npy")
label=np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/gpcp_32x32.npy")
label=np.where(np.isnan(label)==True,0,label)

###################################
index=3
if len(sys.argv) > 1:
    if sys.argv[1]!="x":
        index=sys.argv[1]
        index=int(index)

dataT=tf.convert_to_tensor(data)
labelT=tf.convert_to_tensor(label)

results=cnnbc.predict(dataT)
results=np.reshape(results,(1290,32,32))


mask1=np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/mask32x32.npy")
maskstack=np.repeat(mask1[np.newaxis, :, :], 1290, axis=0)
maskstack2=np.repeat(mask1[np.newaxis, :, :], 1550, axis=0)

results=np.where(np.isnan(maskstack)==True,np.nan,results)

data = np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/CFS_rainfall_cnnbc.npy")
data=np.reshape(data[:1550],(1550,32,32))
data=np.where(np.isnan(maskstack2)==True,np.nan,data)
label=np.reshape(label,(1290,32,32))
label=np.where(np.isnan(maskstack)==True,np.nan,label)

ds = xr.open_dataset("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/Subodh_Data/GPCP_1_1.nc", decode_times=False)
longitude = ds['longitude'].values
latitude = ds['latitude'].values


level=[0,14]

plt.figure(figsize=(15,5))
plt.suptitle("Year:"+str(index//52)+"  Week:"+str(index%52))
plt.subplot(1,3,1)
plt.title('Model data')
#a=plt.contourf(data[index],cmap="rainbow", extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],data[index], cmap='brg', vmin = 0, vmax = level[1])
plt.colorbar()

plt.subplot(1,3,2)
plt.title('Bias corrected data')
#plt.contourf(results[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],results[int(index/1560*1290)], cmap='brg', vmin = 0, vmax = level[1])#, transform=ccrs.PlateCarree())
plt.colorbar()

plt.subplot(1,3,3)
plt.title('Observation data')
#plt.contourf(label[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],label[index], cmap='brg', vmin = 0, vmax = level[1])
plt.colorbar()

plt.show()


