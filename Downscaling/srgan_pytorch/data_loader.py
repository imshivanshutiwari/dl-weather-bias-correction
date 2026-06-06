import glob

import numpy as np
import xarray as xr


from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TropDataset(Dataset):
    
    def __init__(self, data_path, oro_path):
        self.stage = "Training" if data_path.split("/")[-2] == "Train" else "Validation"
        self.lr_path = glob.glob(data_path + "*LR.nc")[0]
        print(f"> Loading {self.stage} LR Data from: {self.lr_path}")
        self.hr_path = glob.glob(data_path + "*HR.nc")[0]
        print(f"> Loading {self.stage} HR Data from: {self.hr_path}")
        self.oro_path = oro_path
        self.lr_data = xr.open_dataset(self.lr_path).rf.values
        self.hr_data = xr.open_dataset(self.hr_path).rf.values
        self.oro_data = xr.open_dataset(self.oro_path).topology.values

        assert len(self.lr_data) == len(self.hr_data), "The lengths of LR and HR data do not match."
        self.length = len(self.lr_data)
        print(f"> {self.stage} dataset Size: {self.length}")
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
        lr_arr = np.dstack((lr_arr , oro_arr))

        # apply padding to HR data : (129, 135) --> (132, 140)
        hr_arr = np.pad(hr_arr, ((1,2),(2,3)), 'edge')

        return {"lr" : self.tensor_transform(lr_arr), "hr" : self.tensor_transform(hr_arr)}
        #return {"lr" : self.tensor_transform(lr_arr).permute(1,2,0), "hr" : self.tensor_transform(hr_arr)}  #- Multi Time Step

    def __len__(self):
        
        return self.length               