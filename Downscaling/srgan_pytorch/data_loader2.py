import glob

import numpy as np
import pandas as pd
import xarray as xr


from torch.utils.data import Dataset
import torchvision.transforms as transforms


#full_range=pd.to_datetime(['2015-01-01','2023-12-31']).tolist()
full_range=pd.to_datetime(['2015-01-01','2020-12-31']).tolist()

class TropDataset(Dataset):
    
    def __init__(self, data_path, oro_path, val_range=full_range, isval=False):
        self.lr_path = glob.glob(data_path + "*LR.nc")[0]
        print(f"> Loading LR Data from: {self.lr_path}")
        self.hr_path = glob.glob(data_path + "*HR.nc")[0]
        print(f"> Loading HR Data from: {self.hr_path}")
        self.oro_path = oro_path
        self.lr_data = xr.open_dataset(self.lr_path)
        self.hr_data = xr.open_dataset(self.hr_path)
        self.oro_data = xr.open_dataset(self.oro_path).topology.values
        
        if isval:
            self.lr_data = self.lr_data.where((self.lr_data.time >= val_range[0]) & (self.lr_data.time <= val_range[1]),drop=True).rf.values
            self.hr_data = self.hr_data.where((self.hr_data.time >= val_range[0]) & (self.hr_data.time <= val_range[1]),drop=True).rf.values
        else:
            self.lr_data = self.lr_data.where((self.lr_data.time <= val_range[0]) | (self.lr_data.time >= val_range[1]),drop=True).rf.values
            self.hr_data = self.hr_data.where((self.hr_data.time <= val_range[0]) | (self.hr_data.time >= val_range[1]),drop=True).rf.values
        
        self.length = len(self.lr_data)
        self.tensor_transform = transforms.Compose([transforms.ToTensor(),])
    
    def __getitem__(self, index):
        
        lr_arr = self.lr_data[index]    #[index:index+5] #- Multi Time Step
        hr_arr = self.hr_data[index]

        # nan ---> 0
        lr_arr = np.where(np.isnan(lr_arr)==True, 0, lr_arr)
        hr_arr = np.where(np.isnan(hr_arr)==True, 0, hr_arr)
        oro_arr = np.where(np.isnan(self.oro_data)==True, 0, self.oro_data)

        # standardization
        #lr_arr = lr_standardize(lr_arr)
        #hr_arr = hr_standardize(hr_arr)

        # Adding the topology channel to lr_arr
        #lr_arr = np.dstack((lr_arr , oro_arr))

        # apply padding to HR data : (129, 135) --> (132, 140)
        hr_arr = np.pad(hr_arr, ((1,2),(2,3)), 'edge')

        return {"lr" : self.tensor_transform(lr_arr), "hr" : self.tensor_transform(hr_arr)}
        #return {"lr" : self.tensor_transform(lr_arr).permute(1,2,0), "hr" : self.tensor_transform(hr_arr)}  #- Multi Time Step

    def __len__(self):
        
        return self.length