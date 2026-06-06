import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



results=np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/results.npy")
results=np.where(np.isnan(results)==True,0,results)
data = np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/CFS_rainfall_cnnbc.npy")
data=np.reshape(data,(1560,32,32))
i1=[i for i in range(len(data)) if i%52<43]
data1=np.copy(data[i1])
label=np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/gpcp_32x32.npy")
label=np.reshape(label,(1290,32,32))

mask1=np.load("/home/hpcs_rnd/George_DIAT/Bias_Correction/CNNBC/mask32x32.npy")


rmse_d=np.mean((label-data1)**2,axis=0)**0.5
rmse_r=np.mean((label-results)**2,axis=0)**0.5

corr_d=np.zeros((32,32),dtype=float)
corr_r=np.zeros((32,32),dtype=float)


def correlation(x,y,z):
    shp=x.shape
    for i in range(shp[1]):
        for j in range(shp[2]):
            '''X=x[:,i,j]
            Y=y[:,i,j]
            Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
            Sum_x_squared = sum((X-X.mean())**2)
            Sum_y_squared = sum((Y-Y.mean())**2)
            z[i,j] = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
            '''
            z[i,j] = np.corrcoef(x[:,i,j], y[:,i,j])[0,1]

    return

correlation(data1,label,corr_d)
correlation(results,label,corr_r)

rmse_d=np.where(np.isnan(mask1)==True,np.nan,rmse_d)
rmse_r=np.where(np.isnan(mask1)==True,np.nan,rmse_r)
corr_d=np.where(np.isnan(mask1)==True,np.nan,corr_d)
corr_r=np.where(np.isnan(mask1)==True,np.nan,corr_r)


ds = xr.open_dataset("/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/Subodh_Data/GPCP_1_1.nc", decode_times=False)
longitude = ds['longitude'].values
latitude = ds['latitude'].values

level=[0,14]

plt.figure(figsize=(10,5))
plt.suptitle("RMSE")
plt.subplot(1,2,1)
plt.title('Model data')
#a=plt.contourf(data[index],cmap="rainbow", extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],rmse_d, cmap='brg', vmin = 0, vmax = level[1])
plt.colorbar()

plt.subplot(1,2,2)
plt.title('Bias corrected data')
#plt.contourf(results[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],rmse_r, cmap='brg', vmin = 0, vmax = level[1])#, transform=ccrs.PlateCarree())
plt.colorbar()

#plt.show()


plt.figure(figsize=(10,5))
plt.suptitle("Correlation")
plt.subplot(1,2,1)
plt.title('Model data')
#a=plt.contourf(data[index],cmap="rainbow", extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],corr_d, cmap='brg', vmin = -1, vmax = 1)
plt.colorbar()

plt.subplot(1,2,2)
plt.title('Bias corrected data')
#plt.contourf(results[index],cmap="rainbow", levels=a.levels, extend = "max")
plt.pcolormesh(longitude[:32], latitude[:32],corr_r, cmap='brg', vmin = -1, vmax = 1)#, transform=ccrs.PlateCarree())
plt.colorbar()

plt.show()

