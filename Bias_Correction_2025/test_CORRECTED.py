"""
CORRECTED test script - fixes negative values and color scale issues
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
from torch import nn

from config import CONFIG
from model_4_fixed import ResCNNv4_Fixed

# SETTINGS
device = "cpu"
print("Using device:", device)

data_path = CONFIG["DATA_PATH"]
label_path = CONFIG["LABEL_PATH"]
mask_path = "masks/Mask_125deg_New.npy"
save_path = "figures/"

lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)
date_string = "2019-07-11T03"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(arr):
    return arr / 400.0

def denormalize(arr):
    return arr * 400.0

def to_chw(t):
    if t.ndim == 3:
        if t.shape[0] <= 32:
            return t
        if t.shape[2] <= 32:
            return np.transpose(t, (2, 0, 1))
        return t
    elif t.ndim == 2:
        return t[np.newaxis, ...]
    else:
        raise ValueError("Unsupported tensor shape")

def print_stats(name, arr):
    arr_flat = arr[np.isfinite(arr)]
    if arr_flat.size == 0:
        print(f"{name}: ALL NaN")
        return
    p = np.percentile(arr_flat, [0, 5, 50, 95, 99, 100])
    nan_pct = 100.0 * (1.0 - arr_flat.size / arr.size)
    std_dev = np.std(arr_flat)
    print(f"{name}:")
    print(f"  Range: {p[0]:.2f} to {p[5]:.2f} (p50={p[2]:.2f})")
    print(f"  Mean: {np.mean(arr_flat):.2f}, Std: {std_dev:.2f}")
    print(f"  NaN%: {nan_pct:.1f}%")

def main():
    print("="*70)
    print("BIAS CORRECTION TEST - CORRECTED VERSION")
    print("="*70)
    
    ensure_dir(save_path)

    # Load mask
    print("\n[STEP 1] Loading MASK...")
    mask = np.load(mask_path)
    print(f"  Mask shape: {mask.shape}")
    
    mask_for_plot = mask[::-1, :]
    print(f"  Valid pixels: {np.sum(~np.isnan(mask_for_plot))}")

    # Load data
    print(f"\n[STEP 2] Loading data for: {date_string}...")
    with xr.open_dataset(data_path) as ds:
        data1 = ds.sel(time=date_string).rf.values
    
    with xr.open_dataset(label_path) as ds:
        label1 = ds.sel(time=date_string).rf.values

    print(f"  Predictor shape: {data1.shape}")
    print(f"  Label shape: {label1.shape}")

    # Prepare data
    print("\n[STEP 3] Preparing data...")
    data1_chw = to_chw(data1)
    
    if label1.ndim == 3 and label1.shape[0] == 1:
        label2d = label1[0]
    elif label1.ndim == 2:
        label2d = label1
    elif label1.ndim == 3:
        label2d = label1[..., 0]
    else:
        raise ValueError("Unsupported label shape")

    # Normalize
    data1_norm = normalize(data1_chw.astype(np.float32))
    data1_tensor = torch.from_numpy(data1_norm).unsqueeze(0)
    
    print(f"  Input tensor shape: {data1_tensor.shape}")

    # Load model
    print("\n[STEP 4] Loading model...")
    in_channels = int(data1_tensor.shape[1])
    model = ResCNNv4_Fixed(in_channels=in_channels, out_channels=1, n_residual_blocks=8)
    model.to(device=device)
    model.eval()
    
    print(f"  Model created (untrained weights)")

    # Inference
    print("\n[STEP 5] Running inference...")
    with torch.no_grad():
        out = model(data1_tensor.to(device=device))
    
    out_1 = out.squeeze(0).squeeze(0).cpu().numpy()
    out_1 = denormalize(out_1)
    
    print(f"  Output shape: {out_1.shape}")
    print(f"  Output range (before clipping): {np.nanmin(out_1):.2f} to {np.nanmax(out_1):.2f}")
    
    # IMPORTANT FIX: Clip negative values to 0 (rainfall can't be negative!)
    out_1 = np.clip(out_1, 0, None)
    print(f"  Output range (after clipping): {np.nanmin(out_1):.2f} to {np.nanmax(out_1):.2f}")

    # Prepare for plotting
    print("\n[STEP 6] Preparing arrays for visualization...")
    data1_first_channel = denormalize(data1_chw[0])
    out_1_plot = np.where(np.isnan(mask_for_plot), np.nan, out_1)
    data1_plot = np.where(np.isnan(mask_for_plot), np.nan, data1_first_channel)
    label1_plot = np.where(np.isnan(mask_for_plot), np.nan, label2d)

    # Print diagnostics
    print("\n" + "="*70)
    print("STATISTICS:")
    print("="*70)
    print_stats("Input (GFS)", data1_plot)
    print_stats("Output (Model)", out_1_plot)
    print_stats("Ground Truth (IMDAA)", label1_plot)

    # Ensure latitude increasing
    lat_plot = np.array(lat_125)
    lon_plot = np.array(lon_125)
    if lat_plot[0] > lat_plot[-1]:
        lat_plot = lat_plot[::-1]
        data1_plot = data1_plot[::-1, :]
        out_1_plot = out_1_plot[::-1, :]
        label1_plot = label1_plot[::-1, :]

    # Create figure with INDEPENDENT color scales for each subplot
    print("\n[STEP 7] Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Bias Correction Results: {date_string}", fontsize=14, fontweight='bold')

    # Get appropriate levels for each subplot
    valid_input = data1_plot[np.isfinite(data1_plot)]
    valid_output = out_1_plot[np.isfinite(out_1_plot)]
    valid_label = label1_plot[np.isfinite(label1_plot)]

    # Input plot
    if len(valid_input) > 0:
        input_max = np.percentile(valid_input, 95)
        input_levels = np.linspace(0, input_max, 21)
        im1 = axes[0].contourf(lon_plot, lat_plot, data1_plot, levels=input_levels, cmap="rainbow", extend="max")
        axes[0].set_title("Input: GFS Model Data (First Channel)", fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0], label="mm/day")
    
    # Output plot
    if len(valid_output) > 0:
        output_max = np.percentile(valid_output, 95)
        output_levels = np.linspace(0, max(output_max, 1.0), 21)
        im2 = axes[1].contourf(lon_plot, lat_plot, out_1_plot, levels=output_levels, cmap="rainbow", extend="max")
        axes[1].set_title("Output: Bias Corrected (Untrained Model)", fontweight='bold', color='orange')
        cbar2 = plt.colorbar(im2, ax=axes[1], label="mm/day")
    
    # Ground truth plot
    if len(valid_label) > 0:
        label_max = np.percentile(valid_label, 95)
        label_levels = np.linspace(0, label_max, 21)
        im3 = axes[2].contourf(lon_plot, lat_plot, label1_plot, levels=label_levels, cmap="rainbow", extend="max")
        axes[2].set_title("Reference: IMDAA Ground Truth", fontweight='bold')
        cbar3 = plt.colorbar(im3, ax=axes[2], label="mm/day")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    
    out_file = os.path.join(save_path, f"BC_test_CORRECTED_{date_string.replace('-','_').replace(':','_')}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to: {out_file}")
    
    print("\n" + "="*70)
    print("IMPORTANT NOTES:")
    print("="*70)
    print("1. Output uses UNTRAINED model (random weights)")
    print("2. To get proper bias correction, train model: python train_simple.py")
    print("3. Each subplot uses independent color scale (95th percentile)")
    print("4. Negative values clipped to 0 (rainfall can't be negative)")
    print("="*70)

if __name__ == "__main__":
    main()
