import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from datetime import date, datetime

from config import CONFIG
from model_4_fixed import ResCNNv4_Fixed
from cnnbc_utils import PSNR, EarlyStopper, plotLosses, plotPSNRs

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Simple dataset class that loads data in chunks
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, start_idx=0, end_idx=100, normalize_data=True):
        print(f"Loading data from {start_idx} to {end_idx}")
        
        # Open datasets but DON'T load into memory yet
        self.data_file = data_path
        self.label_file = label_path
        
        # Open once to get the dataset object for later slicing
        self.data_ds = xr.open_dataset(data_path)
        self.label_ds = xr.open_dataset(label_path)
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        self.normalize_data = normalize_data
        self.length = end_idx - start_idx
        
        # Simple normalization factor
        self.norm_factor = 400.0
        
        print(f"Dataset will lazily load {self.length} samples from {start_idx} to {end_idx}")
        
    def __getitem__(self, index):
        # Load individual sample on demand (lazy loading)
        actual_idx = self.start_idx + index
        
        data1 = self.data_ds.rf.values[actual_idx]
        label1 = self.label_ds.rf.values[actual_idx]
        
        if self.normalize_data:
            data1 = data1 / self.norm_factor
            label1 = label1 / self.norm_factor
            
        return {"gfs": torch.FloatTensor(data1), "imdaa": torch.FloatTensor(label1)}
    
    def __len__(self):
        return self.length

# Training configuration
BATCH_SIZE = 8  # Small batch size to avoid memory issues
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SIZE = 100  # Use only first 100 samples for quick training
VAL_SIZE = 20

print("Creating datasets...")
train_dataset = SimpleDataset(
    data_path=CONFIG["DATA_PATH"], 
    label_path=CONFIG["LABEL_PATH"],
    start_idx=0, 
    end_idx=TRAIN_SIZE,
    normalize_data=True
)

val_dataset = SimpleDataset(
    data_path=CONFIG["DATA_PATH"], 
    label_path=CONFIG["LABEL_PATH"],
    start_idx=TRAIN_SIZE, 
    end_idx=TRAIN_SIZE + VAL_SIZE,
    normalize_data=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Creating model...")
model = ResCNNv4_Fixed(in_channels=10, out_channels=1, n_residual_blocks=8)
model.to(device=device)

# Loss and optimizer
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
psnr_metric = PSNR()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Save path
save_path = "checkpoints/Fixed_Model_Simple/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

best_loss = float('inf')
train_losses = []
val_losses = []
train_psnrs = []
val_psnrs = []

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    
    # Training
    model.train()
    train_epoch_loss = 0.0
    train_epoch_psnr = 0.0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        gfs_data = batch["gfs"].to(device=device, dtype=torch.float)
        imdaa_data = batch["imdaa"].to(device=device, dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(gfs_data)
        loss = mse_loss(output, imdaa_data)
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        with torch.no_grad():
            psnr_val = psnr_metric(output, imdaa_data)
            train_epoch_psnr += psnr_val.item()
    
    train_epoch_loss /= len(train_loader)
    train_epoch_psnr /= len(train_loader)
    
    # Validation
    model.eval()
    val_epoch_loss = 0.0
    val_epoch_psnr = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            gfs_data = batch["gfs"].to(device=device, dtype=torch.float)
            imdaa_data = batch["imdaa"].to(device=device, dtype=torch.float)
            
            output = model(gfs_data)
            loss = mse_loss(output, imdaa_data)
            psnr_val = psnr_metric(output, imdaa_data)
            
            val_epoch_loss += loss.item()
            val_epoch_psnr += psnr_val.item()
    
    val_epoch_loss /= len(val_loader)
    val_epoch_psnr /= len(val_loader)
    
    print(f"Train Loss: {train_epoch_loss:.6f}, Train PSNR: {train_epoch_psnr:.3f}")
    print(f"Val Loss: {val_epoch_loss:.6f}, Val PSNR: {val_epoch_psnr:.3f}")
    
    # Save best model
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        torch.save(model.state_dict(), save_path + f"best_model_epoch_{epoch}.pth")
        print(f"Model saved! Best loss: {best_loss:.6f}")
    
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)
    train_psnrs.append(train_epoch_psnr)
    val_psnrs.append(val_epoch_psnr)

print("\nTraining completed!")
print(f"Best validation loss: {best_loss:.6f}")

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_psnrs': train_psnrs,
    'val_psnrs': val_psnrs
}

np.save(save_path + "training_history.npy", history)
print(f"Training history saved to {save_path}")
