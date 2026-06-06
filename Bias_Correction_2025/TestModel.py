import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch
from torch import nn

from config import CONFIG
#from model_cnnbc1 import CNNBC
from model_4 import ResCNNv4



#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
#print('Using device:', device)


# Paths
data_path=CONFIG["DATA_PATH"]
label_path=CONFIG["LABEL_PATH"]
mask_path="/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/masks/Mask_125deg_New.npy"
save_path="/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/figures/"

model_path="/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/checkpoints/CNNBC_2025_oldmodel/May-17-2025_1/Val-Loss_4.561.pth"


lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)

mask1=np.load(mask_path)[::-1]

scale_path="/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/scales/GFS_3h_max_climatology.json"
f = open(scale_path)
scale_data = json.load(f)

def normalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    #scale_factor = 400
    
    return arr/scale_factor

def denormalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    #scale_factor = 400
    
    return arr*scale_factor


# Arguments
date_string='2019-07-11T03'
if len(sys.argv) == 2:
    date_string=sys.argv[1]
else:
    if len(sys.argv) > 1:
        if sys.argv[1]!="x":
            model_path=sys.argv[1]
        if sys.argv[2]!="x":
            date_string=sys.argv[2]


# Model instance
#model = CNNBC()
model = ResCNNv4()
model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
model.to(device=device)

# Get data samples
data1=xr.open_dataset(data_path).sel(time=date_string).rf.values
data1=normalize(data1,date_string)
data1=torch.Tensor(data1).unsqueeze(dim=0)#.unsqueeze(dim=0)
label1=xr.open_dataset(label_path).sel(time=date_string).rf.values


# Result
with torch.no_grad():
    out_1 = model(data1)
out_1=denormalize(out_1,date_string)
out_1 = np.where(out_1 < 0, 0, out_1)
out_1 = np.where(np.isnan(mask1)==True, np.nan, out_1)

data1=np.array(data1)
data1=denormalize(data1,date_string)
data1 = np.where(np.isnan(mask1)==True, np.nan, data1)
label1 = np.where(np.isnan(mask1)==True, np.nan, label1)


# Plots

plt.figure(figsize=(15,5))
plt.suptitle(date_string)

Lev=list(range(0,50,5))

plt.subplot(1,3,1)
plt.title('Model data')
a=plt.contourf(lon_125, lat_125, data1[0,0,:,:], cmap="rainbow", levels=Lev, extend = "max")
#plt.pcolormesh(lon_125, lat_125,data1, cmap='brg')
plt.colorbar()

plt.subplot(1,3,2)
plt.title('Bias corrected data')
plt.contourf(lon_125, lat_125, out_1[0,0,:,:], cmap="rainbow", levels=a.levels, extend = "max")
#plt.pcolormesh(lon_125, lat_125,out_1, cmap='brg')
plt.colorbar()

plt.subplot(1,3,3)
plt.title('Observation data')
plt.contourf(lon_125, lat_125, label1, cmap="rainbow", levels=a.levels, extend = "max")
#plt.pcolormesh(lon_125, lat_125,label1, cmap='brg')
plt.colorbar()

plt.show()
#plt.savefig(save_path+'BC_125deg_'+date_string.replace('-','_')+".png",dpi=1000)
#plt.close('all')