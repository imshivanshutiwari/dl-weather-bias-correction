import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CFSdataset(Dataset):
    
    def __init__(self, data_path, label_path,idxs):
        print("Loading dataset")
        self.data=np.load(data_path)[idxs[0]:idxs[1]]
        self.label=np.load(label_path)[idxs[0]:idxs[1]]
        
        #self.tensor_transform = transforms.Compose([transforms.ToTensor(),])
        self.tensor_transform = torch.Tensor
        self.length=self.data.shape[0]
        
        
    def __getitem__(self, index):
        dataT=self.tensor_transform(self.data[index])
        #dataT=dataT.squeeze().unsqueeze(dim=1)
        labelT=self.tensor_transform(self.label[index])
        
        return {"cfs" : dataT, "gpcp" : labelT}
        
    def __len__(self):
        
        return self.length