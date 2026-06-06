"""
Fixed test script with comprehensive debugging for bias correction visualization
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
# Use the best available trained model - had Val-PSNR of 37.796
model_path = "checkpoints/ResCNN_2025/May-31-2025_1/Val-PSNR_37.796.pth"

lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)
date_string = "2019-07-11T03"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(arr):
    """Normalize to [-1, 1] or [0, 1] range"""
    return arr / 400.0

def denormalize(arr):
    """Reverse normalization"""
    return arr * 400.0

def to_chw(t):
    """Ensure [C, H, W] ordering for Conv2d"""
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
    """Print detailed statistics about an array"""
    arr_flat = arr[np.isfinite(arr)]
    if arr_flat.size == 0:
        print(f"{name}: ALL NaN ⚠️")
        return
    p = np.percentile(arr_flat, [0, 5, 50, 95, 99, 100])
    nan_pct = 100.0 * (1.0 - arr_flat.size / arr.size)
    std_dev = np.std(arr_flat)
    print(f"{name}:")
    print(f"  Range: {p[0]:.3f} to {p[5]:.3f} (p50={p[2]:.3f})")
    print(f"  Std Dev: {std_dev:.6f}, NaN%: {nan_pct:.1f}%")
    if std_dev < 0.001:
        print(f"  ⚠️  ALERT: Very low standard deviation! Check data/model")

def prepare_levels(*arrays, vmin=None, vmax=None, nlev=21):
    """Build common contour levels across subplots"""
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a is not None and np.isfinite(a).any()])
    if vals.size == 0:
        vmin_eff, vmax_eff = 0.0, 1.0
    else:
        vmin_eff = 0.0 if vmin is None else vmin
        vmax_eff = np.percentile(vals, 99) if vmax is None else vmax
        if vmax_eff <= vmin_eff:
            vmax_eff = vmin_eff + 1.0
    levels = np.linspace(vmin_eff, vmax_eff, nlev)
    return levels

def main():
    print("="*70)
    print("BIAS CORRECTION TEST - FIXED VERSION")
    print("="*70)
    
    ensure_dir(save_path)

    # Load and validate mask
    print("\n[STEP 1] Loading and validating MASK...")
    mask = np.load(mask_path)
    print(f"  Mask shape: {mask.shape}, Expected: ({len(lat_125)}, {len(lon_125)})")
    
    if mask.ndim != 2 or mask.shape[0] != len(lat_125) or mask.shape[1] != len(lon_125):
        raise ValueError(f"Mask shape mismatch!")
    
    mask_for_plot = mask[::-1, :]
    print(f"  Valid pixels (not NaN): {np.sum(~np.isnan(mask_for_plot))}")
    print(f"  Invalid pixels (NaN): {np.sum(np.isnan(mask_for_plot))}")

    # Load data
    print(f"\n[STEP 2] Loading data for date: {date_string}...")
    with xr.open_dataset(data_path) as ds:
        data1 = ds.sel(time=date_string).rf.values
        units = ds.rf.attrs.get("units", "").lower()
        if "kg" in units and "s-1" in units:
            data1 = data1 * 86400.0
            print("  Converted units from kg m-2 s-1 to mm/day")
    
    with xr.open_dataset(label_path) as ds:
        label1 = ds.sel(time=date_string).rf.values
        units_lab = ds.rf.attrs.get("units", "").lower()
        if "kg" in units_lab and "s-1" in units_lab:
            label1 = label1 * 86400.0

    print(f"  Predictor shape: {data1.shape}")
    print(f"  Label shape: {label1.shape}")

    # Prepare data for model
    print("\n[STEP 3] Preparing data for model...")
    data1_chw = to_chw(data1)
    print(f"  Data CHW shape: {data1_chw.shape}")
    
    if label1.ndim == 3 and label1.shape[0] == 1:
        label2d = label1[0]
    elif label1.ndim == 2:
        label2d = label1
    elif label1.ndim == 3:
        label2d = label1[..., 0]
    else:
        raise ValueError("Unsupported label shape")
    
    print(f"  Label 2D shape: {label2d.shape}")

    # Normalize
    data1_norm = normalize(data1_chw.astype(np.float32))
    data1_tensor = torch.from_numpy(data1_norm).unsqueeze(0)
    
    print(f"  Normalized input shape: {data1_tensor.shape}")
    print(f"  Normalized input range: {torch.min(data1_tensor):.6f} to {torch.max(data1_tensor):.6f}")

    # Load model
    print("\n[STEP 4] Loading model...")
    in_channels = int(data1_tensor.shape[1])
    model = ResCNNv4_Fixed(in_channels=in_channels, out_channels=1, n_residual_blocks=8)
    model.to(device=device)
    model.eval()
    
    if os.path.exists(model_path):
        print(f"  ✓ Loading checkpoint from: {model_path}")
        state = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Model loaded successfully")
        if missing:
            print(f"    Missing keys: {missing}")
        if unexpected:
            print(f"    Unexpected keys: {unexpected}")
    else:
        print(f"  ⚠️  WARNING: Checkpoint not found at {model_path}")
        print(f"  Using UNTRAINED model weights - output will be random/poor!")
        checkpoint_dir = os.path.dirname(model_path)
        if os.path.exists(checkpoint_dir):
            print(f"  Available files in {checkpoint_dir}:")
            for item in os.listdir(checkpoint_dir)[:10]:
                print(f"    - {item}")

    # Inference
    print("\n[STEP 5] Running inference...")
    with torch.no_grad():
        out = model(data1_tensor.to(device=device))
    
    out_1 = out.squeeze(0).squeeze(0).cpu().numpy()
    out_1 = denormalize(out_1)
    
    print(f"  Output shape: {out_1.shape}")
    print(f"  Output range (raw): {np.nanmin(out_1):.3f} to {np.nanmax(out_1):.3f}")

    # Prepare for plotting
    print("\n[STEP 6] Preparing arrays for visualization...")
    data1_first_channel = denormalize(data1_chw[0])
    out_1_plot = np.where(np.isnan(mask_for_plot), np.nan, out_1)
    data1_plot = np.where(np.isnan(mask_for_plot), np.nan, data1_first_channel)
    label1_plot = np.where(np.isnan(mask_for_plot), np.nan, label2d)

    # Print comprehensive diagnostics
    print("\n" + "="*70)
    print("DIAGNOSTIC STATISTICS:")
    print("="*70)
    print_stats("Input (predictor)", data1_plot)
    print_stats("Output (prediction)", out_1_plot)
    print_stats("Label (observation)", label1_plot)

    # Check for issues
    output_std = np.nanstd(out_1_plot)
    if output_std < 1e-3:
        print("\n" + "⚠️ "*35)
        print("MAJOR PROBLEM DETECTED:")
        print("Output is nearly constant (std = {:.2e})".format(output_std))
        print("This causes the entire map to appear purple!")
        print("")
        print("Possible causes:")
        print("  1. Model not trained or checkpoint corrupted")
        print("  2. Data normalization incorrect")
        print("  3. Channel ordering wrong")
        print("  4. Checkpoint file doesn't exist or is empty")
        print("⚠️ "*35)

    # Ensure latitude is increasing
    lat_plot = np.array(lat_125)
    lon_plot = np.array(lon_125)
    if lat_plot[0] > lat_plot[-1]:
        lat_plot = lat_plot[::-1]
        data1_plot = data1_plot[::-1, :]
        out_1_plot = out_1_plot[::-1, :]
        label1_plot = label1_plot[::-1, :]

    # Prepare levels
    levels = prepare_levels(data1_plot, out_1_plot, label1_plot, vmin=0.0, vmax=None, nlev=21)

    # Plotting
    print("\n[STEP 7] Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Bias Correction Results: {date_string}", fontsize=14, fontweight='bold')

    # Input
    im1 = axes[0].contourf(lon_plot, lat_plot, data1_plot, levels=levels, cmap="rainbow", extend="max")
    axes[0].set_title("Input: Model Data (GFS)", fontweight='bold')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="mm/day")

    # Output
    im2 = axes[1].contourf(lon_plot, lat_plot, out_1_plot, levels=levels, cmap="rainbow", extend="max")
    axes[1].set_title("Output: Bias Corrected", fontweight='bold', color='green')
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label="mm/day")

    # Ground truth
    im3 = axes[2].contourf(lon_plot, lat_plot, label1_plot, levels=levels, cmap="rainbow", extend="max")
    axes[2].set_title("Reference: Ground Truth (IMDAA)", fontweight='bold')
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[2], label="mm/day")

    plt.tight_layout()
    
    out_file = os.path.join(save_path, f"BC_test_{date_string.replace('-','_').replace(':','_')}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved to: {out_file}")
    
    plt.show()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    if output_std < 1e-3:
        print("❌ OUTPUT IS NEARLY CONSTANT - PLOT WILL BE PURPLE")
        print("   Run diagnose_purple_map.py for detailed debugging")
    else:
        print("✓ Output looks reasonable!")
        print("  Check the visualization for bias correction quality")
    print("="*70)

if __name__ == "__main__":
    main()
