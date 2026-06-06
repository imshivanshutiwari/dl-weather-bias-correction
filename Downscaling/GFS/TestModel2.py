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


def normalize(arr,date):
    d = date.split('-')
    day, month, year = d[-1], d[1], d[0]
    scale_date = f'{month}-{day}'
    scale_factor = scale_data[scale_date]
    
    return arr/scale_factor
    
def denormalize(arr,date):
    d = date.split('-')
    day, month, year = d[-1], d[1], d[0]
    scale_date = f'{month}-{day}'
    scale_factor = scale_data[scale_date]
    
    return arr*scale_factor


#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
#print('Using device:', device)


# Paths
#data_path="/home/hpcs_rnd/George_DIAT/Data/rfData_2010to23/RF25_ind2019_rfp25.nc"
#data_path="/home/hpcs_rnd/George_DIAT/Data/rfData_2005to23/RFdata_2005to23.nc"
data_path="/home/hpcs_rnd/George_DIAT/Data/rfData_2005to23/RF_1975to2023.nc"
oro_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/IMD_DATA/oro.nc"
mask_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/Masks/Mask_0625deg.npy"
scale_path="/home/hpcs_rnd/George_DIAT/Data/SRGAN_scales/scales1975to2023.json"
#scale_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/scales/val_hr.json"

save_path="/home/hpcs_rnd/George_DIAT/Data/miscTests/"

#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/SRGAN-PSNR_38.349.pth"
#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/checkpoints/SRGAN_Val_new/Sep-14-2024/Val-PSNR_47.607.pth"
#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/checkpoints/DISC_Val_BIG1/Sep-20-2024/GEN_Val-PSNR_7.018.pth"
#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/checkpoints/DISC_trial_24/Oct-11-2024/GEN_Val-PSNR_31.891.pth"
#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/checkpoints/DISC_trial_24/Oct-11-2024/GEN_Val-PSNR_31.349.pth"
#model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/checkpoints/SRGAN_Val_new/Oct-14-2024/Val-PSNR_44.185.pth"
model_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/checkpoints/SRGAN_Val_new/Oct-15-2024/Val-PSNR_47.469.pth"

# Orography, scales, coordinates, model and mask
oro_data = xr.open_dataset(oro_path).topology.values
oro_data = oro_data / np.nanmax(oro_data)
oro_data = np.where(np.isnan(oro_data)==True, 0, oro_data)
oro_data = np.where(oro_data < 0, 0, oro_data)

f = open(scale_path)
scale_data = json.load(f)

lat_0625 = np.arange(6.5, 38.5, 0.0625)
lon_0625 = np.arange(66.5, 100, 0.0625)
lat_25 = np.arange(6.5, 38.51, 0.25)
lon_25 = np.arange(66.5, 100.1, 0.25)


mask1=np.load(mask_path)[::-1]

# Date
date_string='2019-07-11'
if len(sys.argv) > 1:
    if sys.argv[1]!="x":
        model_path=sys.argv[1]
    if sys.argv[2]!="x":
        date_string=sys.argv[2]


generator = Generator()
#generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(model_path,map_location='cpu'))
generator.to(device=device)
    
# Loading data
data1=xr.open_dataset(data_path)
sample1=data1.sel(TIME=date_string).RAINFALL.values

sample=sample1
sample=np.where(np.isnan(sample)==True,0,sample)
sample=normalize(sample,date_string)
sample=np.stack((sample,oro_data),axis=0)
#sample=np.stack((sample,np.zeros_like(oro_data)),axis=0)
sample=torch.from_numpy(sample).unsqueeze(dim=0).to(device=device, dtype=torch.float)
#sample=torch.from_numpy(sample).unsqueeze(dim=0).unsqueeze(dim=0).to(device=device, dtype=torch.float)

# Result
with torch.no_grad():
    out_1 = generator(sample)
out_1=out_1.squeeze().detach().cpu().numpy()
out_1=denormalize(out_1,date_string)
sample2 = np.where(out_1 < 0, 0, out_1)
sample2 = np.where(np.isnan(mask1)==True, np.nan, sample2[2:-2,2:-2])

# Plotting
plt.figure(figsize=(10,5))
plt.suptitle(date_string)
#Lev=12
Lev=list(range(0,161,20))
#Lev=list(range(0,221,20))

plt.subplot(1,2,1)
plt.title('0.25x0.25(dataset)')
a=plt.contourf(lon_25, lat_25, sample1,
                cmap="rainbow", levels=Lev, extend = "max")
plt.colorbar()

plt.subplot(1,2,2)
plt.title('0.0625x0.0625(model)')
plt.contourf(lon_0625, lat_0625, sample2,
              cmap="rainbow", levels=a.levels, extend = "max")
plt.colorbar()

plt.show()
#plt.savefig(save_path+"new2_rf_SRplot_"+date_string.replace('-','_')+".png",dpi=400)
#plt.savefig(save_path+"test.png",dpi=400)