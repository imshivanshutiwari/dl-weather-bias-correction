import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch



sys.path.insert(0, "/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/")
from models import Generator


def normalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    
    return arr/scale_factor
    
def denormalize(arr,scale_time):
    scale_time = str(scale_time)[5:]
    scale_factor = scale_data[scale_time]
    
    return arr*scale_factor


#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
#print('Using device:', device)


# Paths

data_path="/home/users/bipink/DIAT/George_Jose/Data/GFS_model_data/GFS_3h_JJAS_india_2019to23.nc"
oro_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/IMD_DATA/oro.nc"
mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/masks/Mask_03125deg.npy"
scale_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/scales/GFS_3h_max_climatology.json"

save_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/figures/"

model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/checkpoints/SRGAN_GFS/Jun-10-2025_adv1/GEN_Val-PSNR_45.811.pth"

# Orography, scales, coordinates, model and mask
oro_data = xr.open_dataset(oro_path).topology.values
oro_data = oro_data / np.nanmax(oro_data)
oro_data = np.where(np.isnan(oro_data)==True, 0, oro_data)
oro_data = np.where(oro_data < 0, 0, oro_data)

f = open(scale_path)
scale_data = json.load(f)

lat_03125 = np.arange(6.5, 38.51, 0.03125)
lon_03125 = np.arange(66.5, 100.01, 0.03125)
lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)


mask1=np.load(mask_path)[-2:1:-1,1:-2]
mask2=np.load("/home/users/bipink/DIAT/George_Jose/Bias_Correction/Bias_Correction_2025/masks/Mask_125deg_New.npy")[::-1]

# Date
date_string='2019-07-11T21'
if len(sys.argv) == 2:
    date_string=sys.argv[1]
else:
    if len(sys.argv) > 1:
        if sys.argv[1]!="x":
            model_path=sys.argv[1]
        if sys.argv[2]!="x":
            date_string=sys.argv[2]


#generator = Generator()
generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path,map_location='cpu'))
generator.to(device=device)
    
# Loading data
data1=xr.open_dataset(data_path)
sample1=data1.sel(time=date_string,index=0).rf.values

sample=sample1
sample=np.where(np.isnan(sample)==True,0,sample)
sample=normalize(sample,date_string)
#sample=np.stack((sample,oro_data),axis=0)
#sample=np.stack((sample,np.zeros_like(oro_data)),axis=0)
#sample=torch.from_numpy(sample).unsqueeze(dim=0).to(device=device, dtype=torch.float)
sample=torch.from_numpy(sample).unsqueeze(dim=0).unsqueeze(dim=0).to(device=device, dtype=torch.float)

# Result
with torch.no_grad():
    out_1 = generator(sample)
out_1=out_1.squeeze().detach().cpu().numpy()
out_1=denormalize(out_1,date_string)
sample2 = np.where(out_1 < 0, 0, out_1)
sample2 = np.where(np.isnan(mask1)==True, np.nan, sample2[1:-2,1:-2])
sample1 = np.where(np.isnan(mask2)==True, np.nan, sample1)



all_values = sample1.flatten()
all_values = all_values[~np.isnan(all_values)]

p60 = max(np.ceil(np.percentile(all_values, 60)),1.0)
p90 = np.ceil(np.percentile(all_values, 90))
p99 = np.ceil(np.percentile(all_values, 99))

levels_1 = [x for x in np.arange(0, p60, p60/2)]
levels_2 = [x for x in np.arange(p60, p90, (p90-p60)/4)]
levels_3 = [x for x in np.arange(p90, p99+0.01, (p99-p90)/6)]
levels = np.unique(np.concatenate([levels_1, levels_2, levels_3]))



# Plotting
plt.figure(figsize=(10,5))

plt.suptitle(date_string)
#Lev=12
#Lev=list(range(0,161,20))
#Lev=list(range(0,50,5))
#Lev=list(range(0,221,20))
Lev=levels

plt.subplot(1,2,1)
plt.title('0.125x0.125(dataset)')
a=plt.contourf(lon_125, lat_125, sample1,
                cmap="rainbow", levels=Lev, extend = "max")

plt.colorbar()

plt.subplot(1,2,2)
plt.title('0.03125x0.03125(model)')
plt.contourf(lon_03125, lat_03125, sample2,
              cmap="rainbow", levels=a.levels, extend = "max")

plt.colorbar()

plt.show()
#plt.savefig(save_path+"Set5_"+date_string.replace('-','_')+".png",dpi=1000)
