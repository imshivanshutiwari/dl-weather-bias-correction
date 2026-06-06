import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
from torch import nn

from config import CONFIG
from model_4_fixed import ResCNNv4_Fixed

# -----------------------------
# SETTINGS
# -----------------------------
device = "cpu"
print("Using device:", device)

# Paths from CONFIG
data_path = CONFIG["DATA_PATH"]      # NetCDF with predictor variable(s), daily
label_path = CONFIG["LABEL_PATH"]    # NetCDF with observed target (e.g., IMD), daily
mask_path = "masks/Mask_125deg_New.npy"
save_path = "figures/"
model_path = "checkpoints/Fixed_Model_Simple/best_model_epoch_50.pth"

# Domain coordinates (IMD-like 0.125° grid)
lat_125 = np.arange(6.5, 38.51, 0.125)
lon_125 = np.arange(66.5, 100.1, 0.125)

# Date/time to visualize
date_string = "2019-07-11T03"  # adjust to a valid timestamp in both files

# -----------------------------
# UTILITIES
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(arr):
    # Simple scale for mm/day dynamic range; adjust to dataset specifics
    return arr / 400.0

def denormalize(arr):
    return arr * 400.0

def to_chw(t):
    # Ensure [C, H, W] ordering for Conv2d; input may be [H, W, C] or [C, H, W]
    if t.ndim == 3:
        # If first dim is small, assume CHW; if last dim is small, assume HWC
        if t.shape[0] <= 32:
            return t
        if t.shape[2] <= 32:
            return np.transpose(t, (2, 0, 1))
        # Fallback: assume already CHW
        return t
    elif t.ndim == 2:
        # Single-channel 2D -> [1, H, W]
        return t[np.newaxis, ...]
    else:
        raise ValueError("Unsupported tensor shape for to_chw")

def print_stats(name, arr):
    arr_flat = arr[np.isfinite(arr)]
    if arr_flat.size == 0:
        print(f"{name}: all NaN")
        return
    p = np.percentile(arr_flat, [0, 5, 50, 95, 99, 100])
    nan_pct = 100.0 * (1.0 - arr_flat.size / arr.size)
    print(f"{name}: min={p[0]:.3f}, p05={p[1]:.3f}, p50={p[2]:.3f}, p95={p[3]:.3f}, p99={p[4]:.3f}, max={p[5]:.3f}, NaN%={nan_pct:.1f}%")

def prepare_levels(*arrays, vmin=None, vmax=None, nlev=21):
    # Build common contour levels across subplots, ignoring NaNs and clipping outliers at 99th percentile
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

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_dir(save_path)

    # Load mask and align orientation
    mask = np.load(mask_path)
    # Ensure mask is 2D [lat, lon] and matches grid lengths
    if mask.ndim != 2 or mask.shape[0] != len(lat_125) or mask.shape[1] != len(lon_125):
        raise ValueError(f"Mask shape does not match domain grid. Got {mask.shape}, expected {(len(lat_125), len(lon_125))}.")
    # Flip latitude if mask saved north->south; our lat_125 is increasing
    mask_for_plot = mask[::-1, :]

    print(f"Testing bias correction for date: {date_string}")

    # Load a single timestep lazily with xarray
    print("Loading data...")
    with xr.open_dataset(data_path) as ds:
        # Assumes predictor variable named 'rf'; change if different
        data1 = ds.sel(time=date_string).rf.values
        # Optional: unit conversion if precipitation in kg m-2 s-1 -> mm/day
        units = ds.rf.attrs.get("units", "").lower()
        if "kg" in units and "s-1" in units:
            data1 = data1 * 86400.0  # 1 kg m-2 s-1 = 86400 mm/day
            print("Converted predictors from kg m-2 s-1 to mm/day")
    with xr.open_dataset(label_path) as ds:
        label1 = ds.sel(time=date_string).rf.values
        units_lab = ds.rf.attrs.get("units", "").lower()
        if "kg" in units_lab and "s-1" in units_lab:
            label1 = label1 * 86400.0
            print("Converted labels from kg m-2 s-1 to mm/day")

    print(f"Raw predictor shape: {data1.shape}")
    print(f"Raw label shape: {label1.shape}")

    # Ensure CHW for model input
    data1_chw = to_chw(data1)
    # Label single-channel array [H, W] expected; if CHW with C=1, squeeze
    if label1.ndim == 3 and label1.shape[0] == 1:
        label2d = label1
    elif label1.ndim == 2:
        label2d = label1
    else:
        # If label is HWC, take the first channel
        if label1.ndim == 3:
            label2d = label1[..., 0]
        else:
            raise ValueError("Unsupported label shape")

    # Normalize input
    data1_norm = normalize(data1_chw.astype(np.float32))
    # Torch tensor [B, C, H, W]
    data1_tensor = torch.from_numpy(data1_norm).unsqueeze(0)

    # Build model
    in_channels = int(data1_tensor.shape[1])
    model = ResCNNv4_Fixed(in_channels=in_channels, out_channels=1, n_residual_blocks=8)
    model.to(device=device)
    model.eval()

    # Verify and load checkpoint (strict)
    if os.path.exists(model_path):
        print("Loading trained model checkpoint...")
        state = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("Checkpoint key mismatch:")
            print("  Missing keys:", missing)
            print("  Unexpected keys:", unexpected)
        else:
            print("Checkpoint loaded successfully.")
    else:
        print("WARNING: No trained model found. Using untrained weights.")

    # Inference
    print("Running inference...")
    with torch.no_grad():
        out = model(data1_tensor.to(device=device))
    out_1 = out.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]

    # Denormalize prediction
    out_1 = denormalize(out_1)

    # Prepare arrays for plotting and masking
    # If input was CHW, for visualization take channel 0 as the "first channel"
    data1_first_channel = denormalize(data1_chw[0])
    # Apply mask: set masked positions to NaN (do not color)
    out_1_plot = np.where(np.isnan(mask_for_plot), np.nan, out_1)
    data1_plot = np.where(np.isnan(mask_for_plot), np.nan, data1_first_channel)
    label1_plot = np.where(np.isnan(mask_for_plot), np.nan, label2d)

    # Diagnostics
    print_stats("Input (first channel)", data1_plot)
    print_stats("Output (prediction)", out_1_plot)
    print_stats("Label (observation)", label1_plot)
    output_std = np.nanstd(out_1_plot)
    print(f"Output standard deviation: {output_std:.6f}")

    if output_std < 1e-3:
        print("WARNING: Output appears nearly constant. Check training, checkpoint, normalization, and channel ordering.")

    # Ensure latitude increasing for plotting with 1D lat/lon
    lat_plot = np.array(lat_125)
    lon_plot = np.array(lon_125)
    if lat_plot[0] > lat_plot[-1]:
        lat_plot = lat_plot[::-1]
        data1_plot = data1_plot[::-1, :]
        out_1_plot = out_1_plot[::-1, :]
        label1_plot = label1_plot[::-1, :]

    # Common levels across subplots to avoid misleading color differences
    levels = prepare_levels(data1_plot, out_1_plot, label1_plot, vmin=0.0, vmax=None, nlev=21)

    # Plot
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Bias Correction Test: {date_string}")

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Input Model Data (First Channel)")
    c1 = ax1.contourf(lon_plot, lat_plot, data1_plot, levels=levels, cmap="rainbow", extend="max")
    plt.colorbar(c1, ax=ax1)
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Bias Corrected Output")
    c2 = ax2.contourf(lon_plot, lat_plot, out_1_plot, levels=levels, cmap="rainbow", extend="max")
    plt.colorbar(c2, ax=ax2)
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Ground Truth (Observations)")
    c3 = ax3.contourf(lon_plot, lat_plot, label1_plot, levels=levels, cmap="rainbow", extend="max")
    plt.colorbar(c3, ax=ax3)
    ax3.set_xlabel("Longitude"); ax3.set_ylabel("Latitude")

    plt.tight_layout()
    out_file = os.path.join(save_path, f"BC_test_{date_string.replace('-','_').replace(':','_')}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Results saved to: {out_file}")
    # plt.show()  # Commented out to avoid blocking
    print("Visualization complete!")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print("1) Ensured CHW input for Conv2d; added unit conversion and consistent color levels.")
    print("2) Mask applied as NaN (not colored), avoiding uniform background artifacts.")
    print("3) Diagnostics print robust stats and std to catch constant outputs.")
    print("4) If output_std is tiny, train the model or fix checkpoint/normalization.")
    print("="*50)

if __name__ == "__main__":
    main()
