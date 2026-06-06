import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch
import geopandas as gpd

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.patches import Rectangle


#import matplotlib as mpl
#plt.rcParams['figure.dpi'] = 300
#plt.rcParams.update({'figure.dpi': 1000})
#plt.rcParams.update({'font.size': 1})
#plt.rcParams.update({'axes.linewidth':0.01})
#plt.rcParams.update({'ytick.labelsize':0.01})



#sys.path.insert(0, "/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/")
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
data_path="/home/users/bipink/DIAT/George_Jose/Data/rfData_2005to23/RF_1975to2023.nc"
oro_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/IMD_DATA/oro.nc"
mask_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/Masks/Mask_0625deg.npy"
scale_path="/home/users/bipink/DIAT/George_Jose/Data/SRGAN_scales/scales1975to2023.json"
#scale_path="/home/hpcs_rnd/George_DIAT/From_Kaustubh/srgan_pytorch/scales/val_hr.json"

#save_path="/home/hpcs_rnd/George_DIAT/Data/miscTests/"
save_path="/home/users/bipink/DIAT/George_Jose/figures_april2025/Gwalior_figures/"


model_path="/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/checkpoints/Seq_run2/Jan-08-2025_gen_tune4/Val-PSNR_40.204.pth"

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
df = gpd.read_file("/home/users/bipink/DIAT/George_Jose/Data/Shape_files/2011_Dist.shp")

# Date
date_string='2019-07-11'
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
sample1=data1.sel(TIME=date_string).RAINFALL.values

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
sample2 = np.where(np.isnan(mask1)==True, np.nan, sample2[2:-2,2:-2])


#Coordinates*
tirupati=[13.6333, 79.4192]
narl=[13.462450323299574, 79.17467286768722]
pakala_aws=[13.452190913051945, 79.11450382930401]

kanpur=[26.4333, 80.3667]
csa_uni=[26.491150, 80.307251]
iit_kanpur=[26.509444444444444, 80.24805555555555]
dm_kanpur_dehat=[26.3554, 79.9599]
ankinghat=[26.9167, 80.0833]
bilhaur=[26.8500, 80.0833]
ghatampur=[26.1667, 80.1667]

delhi_ridge=[28.6833, 77.2500]
du=[28.5833, 77.2000]
aya_nagar=[28.6833, 77.0167]
najafgarh=[28.6333, 77.0000]
rashtrapati_bhav=[28.5833, 77.2000]
lodi_road=[28.5833, 77.2167]


baramati=[18.1500, 74.5833]
ghoda_ambegaon=[19.0500, 73.8333]
pune_city=[18.5000, 73.8333]
indapur=[18.1167, 75.0333]
pashan=[18.5333, 73.8500]#Pune observatory

podar_mc=[19.00484001, 72.84044004]
colaba=[18.91542, 72.82592]
santacruz=[19.08177, 72.84205]
borivali=[19.22936,	72.85751]
andheri=[19.11847, 72.84176999]

bhandar=[25.7333, 78.7500]
gwalior=[26.2333, 78.2500]
iiitm=[26.248468, 78.174592]
ghatigaon=[26.0500, 77.9500]
pichhore=[25.9500, 78.4000]



#bounds=[[25.8,27], [79.4,80.8]]            # Kanpur
#bounds=[[28.4, 29],[76.5, 77.5]]        # Delhi
#bounds=[[17.5,20],[73,76]]                # Pune
#bounds=[[18.8,19.4],[72.6,73]]            # Mumbai
bounds=[[25.7,26.4],[77.6,78.8]]

# Plotting
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12, 6))

plt.suptitle(date_string)


#Lev=12
#Lev=list(range(0,161,20))
Lev=list(range(0,50,5))
#Lev=list(range(0,221,20))


ax[0].set_title('0.25x0.25(dataset)')

a=ax[0].contourf(lon_25, lat_25, sample1,cmap="rainbow", levels=Lev, extend = "max", extent=[lon_25[0],lon_0625[-1],lat_25[0],lat_25[-1]])
df.boundary.plot(ax=ax[0], facecolor='none', edgecolor='black',linewidth=0.5)

#ax[0].plot(bhandar[1],bhandar[0],'o',markersize=8, markerfacecolor='darkgreen',label='Bhandar')
ax[0].plot(gwalior[1],gwalior[0],'ks',markersize=8, markeredgewidth=1.5,markerfacecolor='none',  label='Gwalior')
ax[0].plot(iiitm[1],iiitm[0],'o',markersize=10, markeredgewidth=2.0,markerfacecolor='none', markeredgecolor='darkgreen', label='IIITM')
ax[0].plot(ghatigaon[1],ghatigaon[0],'k+',markersize=10,label='Ghatigaon')
ax[0].plot(pichhore[1],pichhore[0],'k1',markersize=10,label='Pichhore')
ax[0].legend(loc='upper left',fontsize=8)

ax[0].set_xlim(bounds[1])
ax[0].set_ylim(bounds[0])

ax[0].xaxis.set_major_locator(MultipleLocator(0.25))                #enable grids
ax[0].yaxis.set_major_locator(MultipleLocator(0.25))
ax[0].grid(which='major', color='#CCCCCC', linestyle='--')
ax[0].tick_params(axis='x', rotation=90)
fig.colorbar(a,ax=ax[0])


ax[1].set_title('0.0625x0.0625(model)')

a2=ax[1].contourf(lon_0625, lat_0625, sample2,cmap="rainbow", levels=a.levels, extend = "max", extent=[lon_0625[0],lon_0625[-1],lat_0625[0],lat_0625[-1]])
df.boundary.plot(ax=ax[1], facecolor='none', edgecolor='black',linewidth=0.5)


#ax[1].plot(bhandar[1],bhandar[0],'o',markersize=8, markerfacecolor='darkgreen',label='Bhandar')
ax[1].plot(gwalior[1],gwalior[0],'ks',markersize=8, markeredgewidth=1.5,markerfacecolor='none',  label='Gwalior')
ax[1].plot(iiitm[1],iiitm[0],'o',markersize=10, markeredgewidth=2.0,markerfacecolor='none', markeredgecolor='darkgreen', label='IIITM')
ax[1].plot(ghatigaon[1],ghatigaon[0],'k+',markersize=10,label='Ghatigaon')
ax[1].plot(pichhore[1],pichhore[0],'k1',markersize=10,label='Pichhore')
ax[1].legend(loc='upper left',fontsize=8)

ax[1].set_xlim(bounds[1])
ax[1].set_ylim(bounds[0])

ax[1].xaxis.set_major_locator(MultipleLocator(0.0625))
ax[1].yaxis.set_major_locator(MultipleLocator(0.0625))
ax[1].grid(which='major', color='#CCCCCC', linestyle='--')
ax[1].tick_params(axis='x', rotation=90)
fig.colorbar(a2,ax=ax[1])

#plt.show()
plt.savefig(save_path+"gwalior_"+date_string.replace('-','_')+".png",dpi=400)
plt.close('all')
