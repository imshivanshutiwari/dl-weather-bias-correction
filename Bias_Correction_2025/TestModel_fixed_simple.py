import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import json
import torch
from torch import nn

from config import CONFIG
from model_4_fixed import ResCNNv4_Fixed  # Using FIXED model

device = "cpu"
print('Using device:', device)

# Paths
data_path = CONFIG["DATA_PATH"]
label_path = CONFIG["LABEL_PATH"]
mask_path = "masks/Mask_125deg_New.npy"
save_path = "figures/"

# Create save directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)

# Load mask
mask1 = np.load(mask_path)[::-1]

# Load scale data if available
try:
    scale_path = "scales/GFS_3h_max_climatology.json"
    f = open(scale_path)
    scale_data = json.load(f)
    has_scale_data = True
except:
    print("Scale data not found, using simple normalization")
    has_scale_data = False

def normalize(arr, scale_time):
    if has_scale_data:
        scale_time = str(scale_time)[5:]
        scale_factor = scale_data[scale_time]
        return arr/scale_factor
    else:
        # Simple normalization
        return arr/400.0

def denormalize(arr, scale_time):
    if has_scale_data:
        scale_time = str(scale_time)[5:]
        scale_factor = scale_data[scale_time]
        return arr*scale_factor
    else:
        # Simple denormalization
        return arr*400.0

# Arguments
date_string = '2019-07-11T03'
if len(sys.argv) == 2:
    date_string = sys.argv[1]

print(f"Testing FIXED bias correction for date: {date_string}")

# Load data
print("Loading data...")
data1 = xr.open_dataset(data_path).sel(time=date_string).rf.values
label1 = xr.open_dataset(label_path).sel(time=date_string).rf.values

print(f"Data shape: {data1.shape}")
print(f"Label shape: {label1.shape}")

# Normalize data
data1_norm = normalize(data1, date_string)
data1_tensor = torch.Tensor(data1_norm).unsqueeze(dim=0)  # Add batch dimension

print(f"Input data range: {np.nanmin(data1_norm):.4f} to {np.nanmax(data1_norm):.4f}")

# Create FIXED model instance (untrained for testing)
print("Creating FIXED model...")
model = ResCNNv4_Fixed(in_channels=10, out_channels=1, n_residual_blocks=8)
model.to(device=device)
model.eval()

# Test inference
print("Running inference with FIXED model...")
with torch.no_grad():
    out_1 = model(data1_tensor)

# Convert to numpy and denormalize
out_1 = out_1.squeeze().cpu().numpy()
out_1 = denormalize(out_1, date_string)

# Remove negative values and apply mask
out_1 = np.where(out_1 < 0, 0, out_1)
out_1 = np.where(np.isnan(mask1) == True, np.nan, out_1)

# Denormalize input for plotting
data1_plot = denormalize(data1_norm, date_string)
data1_plot = np.where(np.isnan(mask1) == True, np.nan, data1_plot)
label1_plot = np.where(np.isnan(mask1) == True, np.nan, label1)

# Print statistics
print(f"Input data range: {np.nanmin(data1_plot):.4f} to {np.nanmax(data1_plot):.4f}")
print(f"Output data range: {np.nanmin(out_1):.4f} to {np.nanmax(out_1):.4f}")
print(f"Label data range: {np.nanmin(label1_plot):.4f} to {np.nanmax(label1_plot):.4f}")

# Check if output is constant (blue map issue)
output_std = np.nanstd(out_1)
print(f"Output standard deviation: {output_std:.6f}")

if output_std < 0.001:
    print("⚠️  WARNING: Output appears to be constant (blue map issue detected)")
else:
    print("✅ Output shows variation (FIXED model working!)")

# Create visualization
plt.figure(figsize=(15, 5))
plt.suptitle(f'FIXED Bias Correction Test: {date_string}')

# Determine color levels
max_val = max(np.nanmax(data1_plot), np.nanmax(out_1), np.nanmax(label1_plot))
Lev = list(range(0, int(max_val) + 5, 5))

# Input data (take first channel for visualization)
plt.subplot(1, 3, 1)
plt.title('Input Model Data (First Channel)')
a = plt.contourf(lon_125, lat_125, data1_plot[0], cmap="rainbow", levels=Lev, extend="max")
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Bias corrected output
plt.subplot(1, 3, 2)
plt.title('FIXED Bias Corrected Output')
plt.contourf(lon_125, lat_125, out_1, cmap="rainbow", levels=a.levels, extend="max")
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Ground truth
plt.subplot(1, 3, 3)
plt.title('Ground Truth (Observations)')
plt.contourf(lon_125, lat_125, label1_plot, cmap="rainbow", levels=a.levels, extend="max")
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.savefig(save_path + f'BC_FIXED_test_{date_string.replace("-", "_").replace(":", "_")}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Results saved to: {save_path}")
print("FIXED model test completed!")
