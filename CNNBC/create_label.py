import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image



data_path="/home/hpcs_rnd/George_DIAT/Bias_Correction/bias_correction/Subodh_Data/GPCP_1_1.nc"




gpcp1=xr.open_dataset(data_path, decode_times=False)

pr1=gpcp1.pr.values

data=np.reshape(pr1,(30,32,32))

label=[]
for i in data:
    o_i=np.repeat(i[np.newaxis, :, :], 43, axis=0)
    label.append(o_i)
    
#label=np.array(label)

label=np.concatenate(label,axis=0)
label=np.reshape(label,(1290, 32, 32, 1))

np.save("gpcp_32x32.npy",label)