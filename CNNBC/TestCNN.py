import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch

from model_cnnbc import CNNBC


#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
#print('Using device:', device)

#Paths
data_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/data_10x32x32.npy"
label_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/gpcp_32x32.npy"
mask_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/mask32x32.npy"
save_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/plots/"

model_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/checkpoints/CNNBC_2025/Mar-21-2025/Val-Loss_14.661.pth"


mask1=np.load(mask_path)
maskstack=np.repeat(mask1[np.newaxis, :, :], 1290, axis=0)

ds = xr.open_dataset("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/Subodh_Data/GPCP_1_1.nc", decode_times=False)
longitude = ds['longitude'].values
latitude = ds['latitude'].values

if len(sys.argv) > 1:
    if sys.argv[1]!="x":
        model_path=sys.argv[1]

cnnbc_model = CNNBC()
cnnbc_model.load_state_dict(torch.load(model_path,map_location='cpu'))
cnnbc_model.to(device=device)


data1=np.load(data_path)
data1=torch.from_numpy(data1).squeeze().unsqueeze(dim=1).to(device=device, dtype=torch.float)
labels=np.load(label_path)
labels=labels.reshape((1290, 32, 32))


results=cnnbc_model(data1)

results=results.squeeze().squeeze().detach().cpu().numpy()
data1=data1.squeeze().detach().cpu().numpy()
data1=data1[:,9,:,:]

data1=np.where(np.isnan(maskstack)==True,np.nan,data1)
labels=np.where(np.isnan(maskstack)==True,np.nan,labels)
results=np.where(np.isnan(maskstack)==True,np.nan,results)


# RMSE and Correlation
rmse_d=np.mean((labels-data1)**2,axis=0)**0.5
rmse_r=np.mean((labels-results)**2,axis=0)**0.5

corr_d=np.zeros((32,32),dtype=float)
corr_r=np.zeros((32,32),dtype=float)

def correlation(x,y,z):
    shp=x.shape
    for i in range(shp[1]):
        for j in range(shp[2]):
            z[i,j] = np.corrcoef(x[:,i,j], y[:,i,j])[0,1]
            
    return

correlation(data1,labels,corr_d)
correlation(results,labels,corr_r)

rmse_d=np.where(np.isnan(mask1)==True,np.nan,rmse_d)
rmse_r=np.where(np.isnan(mask1)==True,np.nan,rmse_r)
corr_d=np.where(np.isnan(mask1)==True,np.nan,corr_d)
corr_r=np.where(np.isnan(mask1)==True,np.nan,corr_r)


mean_d=np.mean(data1[1032:],axis=0)
mean_r=np.mean(results[1032:],axis=0)
mean_l=np.mean(labels[1032:],axis=0)

mean_d=np.where(np.isnan(mask1)==True,np.nan,mean_d)
mean_r=np.where(np.isnan(mask1)==True,np.nan,mean_r)
mean_l=np.where(np.isnan(mask1)==True,np.nan,mean_l)


# Mean plots
level=[0,14]

plt.figure(figsize=(15,5))
plt.suptitle("Mean PR 2005-10")
plt.subplot(1,3,1)
plt.title('Model data')
#a=plt.contourf(data[index],cmap="rainbow", extend = "max")
plt.pcolormesh(longitude, latitude,mean_d, cmap='brg', vmin = 0, vmax = level[1])
plt.colorbar()

plt.subplot(1,3,2)
plt.title('Bias corrected data')
#plt.contourf(results[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude, latitude,mean_r, cmap='brg', vmin = 0, vmax = level[1])#, transform=ccrs.PlateCarree())
plt.colorbar()

plt.subplot(1,3,3)
plt.title('Observation data')
#plt.contourf(label[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude, latitude,mean_l, cmap='brg', vmin = 0, vmax = level[1])
plt.colorbar()


# RMSE plot
plt.figure(figsize=(10,5))
plt.suptitle("RMSE")
plt.subplot(1,2,1)
plt.title('Model data')
#a=plt.contourf(data[index],cmap="rainbow", extend = "max")
plt.pcolormesh(longitude, latitude,rmse_d, cmap='brg', vmin = 0, vmax = level[1])
plt.colorbar()

plt.subplot(1,2,2)
plt.title('Bias corrected data')
#plt.contourf(results[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude, latitude,rmse_r, cmap='brg', vmin = 0, vmax = level[1])#, transform=ccrs.PlateCarree())
plt.colorbar()


# Correlation plot
plt.figure(figsize=(10,5))
plt.suptitle("Correlation")
plt.subplot(1,2,1)
plt.title('Model data')
#a=plt.contourf(data[index],cmap="rainbow", extend = "max")
plt.pcolormesh(longitude, latitude,corr_d, cmap='brg', vmin = 0, vmax = 1)
plt.colorbar()

plt.subplot(1,2,2)
plt.title('Bias corrected data')
#plt.contourf(results[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude, latitude,corr_r, cmap='brg', vmin = 0, vmax = 1)#, transform=ccrs.PlateCarree())
plt.colorbar()

plt.show()




'''
import seaborn as sns

fig, ax = plt.subplots()
plt.xticks([i/10 for i in range(-1,11)])
ax=sns.kdeplot(corr_r.flatten())
ax=sns.kdeplot(corr_d.flatten())

plt.xlabel('Correlation')
ax.legend(['Bias corrected data','Bias data'])

fig_data = ax.lines[0].get_xydata()
max_xy=fig_data[np.where(fig_data[:, 1] == max(fig_data[:, 1]))].flatten()
ax.axvline(x=max_xy[0],color='r',linestyle='dashed')

plt.show()
'''