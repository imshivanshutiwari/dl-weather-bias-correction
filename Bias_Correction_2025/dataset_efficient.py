import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import json

class MemoryEfficientDataset(Dataset):
    """
    Memory-efficient dataset that loads samples one by one from NetCDF files
    """
    def __init__(self, data_path, label_path, start_idx=0, end_idx=None, normalize=True):
        # Open datasets to get metadata without loading all data
        self.data_path = data_path
        self.label_path = label_path
        
        # Get time indices
        with xr.open_dataset(data_path) as ds:
            self.times = ds.time.values
            if end_idx is None:
                end_idx = len(self.times)
            self.times = self.times[start_idx:end_idx]
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.normalize = normalize
        self.norm_factor = 400.0  # Simple normalization factor
        
        print(f"Dataset created with {len(self.times)} samples (indices {start_idx} to {end_idx})")
    
    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, idx):
        # Get the actual time index
        time_idx = self.start_idx + idx
        time_str = str(self.times[idx])
        
        # Load single sample from each dataset
        with xr.open_dataset(self.data_path) as ds:
            gfs_data = ds.isel(time=time_idx).rf.values
        
        with xr.open_dataset(self.label_path) as ds:
            imdaa_data = ds.isel(time=time_idx).rf.values
        
        # Normalize if requested
        if self.normalize:
            gfs_data = gfs_data / self.norm_factor
            imdaa_data = imdaa_data / self.norm_factor
        
        # Convert to tensors
        gfs_tensor = torch.FloatTensor(gfs_data)
        imdaa_tensor = torch.FloatTensor(imdaa_data).unsqueeze(0)  # Add channel dimension
        
        return {
            'gfs': gfs_tensor,
            'imdaa': imdaa_tensor,
            'time': time_str
        }

class SimpleDataset(Dataset):
    """
    Simple dataset that loads a small chunk of data into memory
    """
    def __init__(self, data_path, label_path, start_idx=0, num_samples=200, normalize=True):
        print(f"Loading {num_samples} samples starting from index {start_idx}...")
        
        # Load small chunk of data
        with xr.open_dataset(data_path) as ds:
            self.gfs_data = ds.isel(time=slice(start_idx, start_idx + num_samples)).rf.values
            self.times = ds.isel(time=slice(start_idx, start_idx + num_samples)).time.values
        
        with xr.open_dataset(label_path) as ds:
            self.imdaa_data = ds.isel(time=slice(start_idx, start_idx + num_samples)).rf.values
        
        self.normalize = normalize
        self.norm_factor = 400.0
        
        print(f"Loaded GFS data shape: {self.gfs_data.shape}")
        print(f"Loaded IMDAA data shape: {self.imdaa_data.shape}")
    
    def __len__(self):
        return len(self.gfs_data)
    
    def __getitem__(self, idx):
        gfs_sample = self.gfs_data[idx].copy()
        imdaa_sample = self.imdaa_data[idx].copy()
        
        if self.normalize:
            gfs_sample = gfs_sample / self.norm_factor
            imdaa_sample = imdaa_sample / self.norm_factor
        
        gfs_tensor = torch.FloatTensor(gfs_sample)
        imdaa_tensor = torch.FloatTensor(imdaa_sample).unsqueeze(0)  # Add channel dimension
        
        return {
            'gfs': gfs_tensor,
            'imdaa': imdaa_tensor,
            'time': str(self.times[idx])
        }
