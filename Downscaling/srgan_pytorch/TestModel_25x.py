# Test script for 25x downscaling SRGAN (0.625° to 0.025°)
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch
from torch import nn

from models_25x import Generator_25x, ProgressiveGenerator_25x

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

# Paths for 25x downscaling
data_path = "data/"
save_path = "data/SRGAN_25x_Results/"
model_path = "/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/checkpoints/SRGAN_25x_2025/Dec-15-2025_1/Val-PSNR_45.000.pth"

# Create save directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Arguments
date_string = '2019-07-15'
if len(sys.argv) == 2:
    date_string = sys.argv[1]
else:
    if len(sys.argv) > 1:
        if sys.argv[1] != "x":
            model_path = sys.argv[1]
        if sys.argv[2] != "x":
            date_string = sys.argv[2]

# Load model
print("Loading 25x downscaling model...")
generator = Generator_25x(in_channels=1, out_channels=1, n_residual_blocks=8)
generator.load_state_dict(torch.load(model_path, map_location='cpu'))
generator.to(device=device)
generator.eval()

# Load test data
print("Loading test data...")
try:
    # Load low-resolution data (0.625°)
    lr_data = xr.open_dataset(data_path + "MPNorm_LR_0625deg.nc")
    # Load high-resolution data (0.025°) for comparison
    hr_data = xr.open_dataset(data_path + "Norm_HR_0025deg.nc")
except:
    print("Data files not found. Using synthetic data for demonstration...")
    # Create synthetic data for demonstration
    lat_0625 = np.arange(6.5, 38.5, 0.0625)  # ~512 points
    lon_0625 = np.arange(66.5, 100, 0.0625)  # ~537 points
    lat_0025 = np.arange(6.5, 38.5, 0.025)   # ~1280 points
    lon_0025 = np.arange(66.5, 100, 0.025)   # ~1340 points
    
    # Create synthetic rainfall data
    lr_synthetic = np.random.rand(len(lat_0625), len(lon_0625)) * 50
    hr_synthetic = np.random.rand(len(lat_0025), len(lon_0025)) * 40
    
    # Create xarray datasets
    lr_data = xr.Dataset(
        data_vars={'rf': (['lat', 'lon'], lr_synthetic)},
        coords={'lat': lat_0625, 'lon': lon_0625}
    )
    hr_data = xr.Dataset(
        data_vars={'rf': (['lat', 'lon'], hr_synthetic)},
        coords={'lat': lat_0025, 'lon': lon_0025}
    )

# Extract data for specific date or use first timestep
if 'time' in lr_data.dims:
    lr_sample = lr_data.sel(time=date_string).rf.values
    hr_sample = hr_data.sel(time=date_string).rf.values
else:
    lr_sample = lr_data.rf.values
    hr_sample = hr_data.rf.values

# Prepare input for model
lr_tensor = torch.FloatTensor(lr_sample).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
lr_tensor = lr_tensor.to(device)

print(f"Input shape: {lr_tensor.shape}")
print(f"Expected output shape: {lr_tensor.shape[2]*32}x{lr_tensor.shape[3]*32} (25x upscaling)")

# Generate high-resolution output
print("Generating 25x downscaled output...")
with torch.no_grad():
    sr_output = generator(lr_tensor)

# Convert to numpy
sr_output = sr_output.squeeze().cpu().numpy()

print(f"Output shape: {sr_output.shape}")
print(f"Resolution improvement: {sr_output.shape[0]/lr_sample.shape[0]:.1f}x{lr_output.shape[1]/lr_sample.shape[1]:.1f}")

# Calculate metrics
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

# Resize hr_sample to match sr_output for comparison
from scipy.ndimage import zoom
hr_resized = zoom(hr_sample, (sr_output.shape[0]/hr_sample.shape[0], sr_output.shape[1]/hr_sample.shape[1]))

psnr_val = calculate_psnr(sr_output, hr_resized)
mse_val = calculate_mse(sr_output, hr_resized)

print(f"PSNR: {psnr_val:.3f} dB")
print(f"MSE: {mse_val:.6f}")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'25x Downscaling Results: {date_string} (0.625° → 0.025°)', fontsize=16)

# Low-resolution input
im1 = axes[0].imshow(lr_sample, cmap='viridis', aspect='auto')
axes[0].set_title(f'Low-Resolution Input\n(0.625°, {lr_sample.shape[0]}x{lr_sample.shape[1]})')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
plt.colorbar(im1, ax=axes[0])

# High-resolution output
im2 = axes[1].imshow(sr_output, cmap='viridis', aspect='auto')
axes[1].set_title(f'25x Downscaled Output\n(0.025°, {sr_output.shape[0]}x{sr_output.shape[1]})')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
plt.colorbar(im2, ax=axes[1])

# Ground truth (high-resolution)
im3 = axes[2].imshow(hr_resized, cmap='viridis', aspect='auto')
axes[2].set_title(f'Ground Truth\n(0.025°, {hr_resized.shape[0]}x{hr_resized.shape[1]})')
axes[2].set_xlabel('Longitude')
axes[2].set_ylabel('Latitude')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(save_path + f'25x_downscaling_{date_string}.png', dpi=300, bbox_inches='tight')
plt.show()

# Create detailed comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Detailed 25x Downscaling Analysis: {date_string}', fontsize=14)

# Original vs Downscaled
axes[0,0].imshow(lr_sample, cmap='viridis')
axes[0,0].set_title('Original (0.625°)')
axes[0,0].axis('off')

axes[0,1].imshow(sr_output, cmap='viridis')
axes[0,1].set_title('25x Downscaled (0.025°)')
axes[0,1].axis('off')

# Difference and histogram
diff = sr_output - hr_resized
axes[1,0].imshow(diff, cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
axes[1,0].set_title('Difference (Output - Ground Truth)')
axes[1,0].axis('off')

axes[1,1].hist(diff.flatten(), bins=50, alpha=0.7)
axes[1,1].set_title('Difference Distribution')
axes[1,1].set_xlabel('Difference')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(save_path + f'25x_analysis_{date_string}.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
results = {
    'date': date_string,
    'input_resolution': '0.625°',
    'output_resolution': '0.025°',
    'improvement_factor': 25,
    'input_shape': lr_sample.shape,
    'output_shape': sr_output.shape,
    'psnr': psnr_val,
    'mse': mse_val,
    'model_path': model_path
}

with open(save_path + f'25x_results_{date_string}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {save_path}")
print(f"25x downscaling completed successfully!")
print(f"Resolution improvement: {results['improvement_factor']}x")
print(f"PSNR: {results['psnr']:.3f} dB")
