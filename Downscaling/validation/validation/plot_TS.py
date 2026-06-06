import torch

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys


model_data_path="/home/hpcs_rnd/George_DIAT/Data/SRGAN_corr/data_files/SR_valNorm_HR.nc"
model_data=xr.open_dataset(model_data_path)
md_jjas2020=model_data.sel(time=slice('2020-06-01','2020-09-30')).sel(lon=79.125).sel(lat=13.4375).rf.values


station={'name' : 'Pakala',
            'coords' : (13.4667,79.1333),
            'data' : np.array([0.0, 4.6, 0.0, 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    2.2,    9.6,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   11.4,   4.2,   36.4,    0.0,    0.0,    2.2,   63.4,    36.4,   11.4,   14.2,    2.2,    0.0,    0.0,    0.0,   13.2,   38.4,   56.2,   30.2,    0.0,   84.2,  10.4,    6.4,   12.2,   38.2,    0.0,    35.2,   65.2,    0.0,    0.0,    0.0,   13.2,   40.2,   21.2,    0.0,    0.0,    0.0,   13.2,   37.6,   39.2,    7.2,   30.6,    0.0,    1.2,    0.0,    0.0,    6.4,    0.0,    0.0,    0.0,    7.4,    0.0,    0.0,    0.0,    0.0,    0.0,    9.4,    0.0,    0.0,    0.0,    0.0,   21.2,   32.4,   12.2,    0.0,    0.0,    0.0,    0.0,   0.0,    0.0,    0.0,    0.0,    3.4,    0.0,    0.0,    0.0,    0.0,   14.6,   48.2,   23.2,    0.0,    6.2,    7.2,   14.2,    6.2,    0.0,    0.0,    0.0,   34.6,    0.0,    7.2,    0.0,    0.0,    0.0,    5.2,   64.2,    0.0,    0.0,    0.0,    0.0]),
            'year' : 2020
}

corr=np.corrcoef(md_jjas2020, station['data'])[0,1]
mse=np.mean((md_jjas2020-station['data'])**2)
print('MSE=',mse,' | Correlation=',corr)


fig,ax=plt.subplots(figsize=(10,5))

ax.set_title(station['name'] +' '+ str(station['year']) + ' JJAS')
ax.plot(station['data'],label='Station')
ax.plot(md_jjas2020,label='Model')
ax.legend()

plt.show()
