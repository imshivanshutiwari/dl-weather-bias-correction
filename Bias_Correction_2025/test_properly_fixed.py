import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
from torch import nn

from config import CONFIG
from model_4_properly_fixed import ResCNNv4_ProperlyFixed

device = "cpu"
print('Using device:', device)

# Paths
data_path = CONFIG["DATA_PATH"]
label_path = CONFIG["LABEL_PATH"]
mask_path = "masks/Mask_125deg_New.npy"
save_path = "figures/"

# Create save directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)

# Load mask
mask1 = np.load(mask_path)[::-1]

# Simple normalization
def normalize(arr):
    return arr / 400.0

def denormalize(arr):
    return arr * 400.0

# Test date
date_string = '2019-07-11T03'

print(f"Testing PROPERLY FIXED bias correction for date: {date_string}")

# Load single sample using xarray's lazy loading
print("Loading data...")
with xr.open_dataset(data_path) as ds:
    data1 = ds.sel(time=date_string).rf.values
with xr.open_dataset(label_path) as ds:
    label1 = ds.sel(time=date_string).rf.values

print(f"Data shape: {data1.shape}")
print(f"Label shape: {label1.shape}")

# Normalize data
data1_norm = normalize(data1)
data1_tensor = torch.Tensor(data1_norm).unsqueeze(dim=0)

print(f"Input data range: {np.nanmin(data1_norm):.4f} to {np.nanmax(data1_norm):.4f}")

# Create PROPERLY FIXED model
model = ResCNNv4_ProperlyFixed(in_channels=10, out_channels=1, n_residual_blocks=8)
model.to(device=device)
model.eval()

print("Using PROPERLY FIXED model with:")
print("- No activation function in final layer")
print("- Proper weight initialization")
print("- Increased feature maps (64 vs 40)")
print("- Adjusted group size (8 vs 10)")

# Test inference
print("Running inference...")
with torch.no_grad():
    out_1 = model(data1_tensor)

# Convert to numpy and denormalize
out_1 = out_1.squeeze().cpu().numpy()
out_1 = denormalize(out_1)

# Remove negative values and apply mask
out_1 = np.where(out_1 < 0, 0, out_1)
out_1 = np.where(np.isnan(mask1) == True, np.nan, out_1)

# Denormalize input for plotting
data1_plot = denormalize(data1_norm)
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
    print("✅ Output shows variation (model architecture is working!)")

# Check if output range is reasonable
output_range = np.nanmax(out_1) - np.nanmin(out_1)
label_range = np.nanmax(label1_plot) - np.nanmin(label1_plot)
print(f"Output range: {output_range:.2f}")
print(f"Label range: {label_range:.2f}")

if output_range < 1.0:
    print("⚠️  WARNING: Output range is too small")
elif output_range > label_range * 2:
    print("⚠️  WARNING: Output range is too large")
else:
    print("✅ Output range looks reasonable")

# Create visualization
plt.figure(figsize=(15, 5))
plt.suptitle(f'PROPERLY FIXED Bias Correction Test: {date_string}')

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
plt.title('PROPERLY FIXED Bias Corrected Output')
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
plt.savefig(save_path + f'BC_PROPERLY_FIXED_test_{date_string.replace("-", "_").replace(":", "_")}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Results saved to: {save_path}")
print("Test completed!")

# Summary
print("\n" + "="*60)
print("PROPERLY FIXED MODEL SUMMARY:")
print("="*60)
print("1. ✅ Removed activation function from final layer")
print("2. ✅ Added proper weight initialization")
print("3. ✅ Increased feature maps for better learning")
print("4. ✅ Adjusted group size for better performance")
print("5. ✅ This should produce more realistic outputs")
print("6. ⚠️  Still needs training to learn actual bias patterns")
print("="*60)
