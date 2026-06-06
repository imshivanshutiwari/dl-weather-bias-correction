"""
Diagnostic script to identify why the entire map is purple (constant output)
Run this BEFORE running the main test script
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from config import CONFIG
from model_4_fixed import ResCNNv4_Fixed

# Settings
device = "cpu"
data_path = CONFIG["DATA_PATH"]
label_path = CONFIG["LABEL_PATH"]
mask_path = "masks/Mask_125deg_New.npy"
model_path = "checkpoints/Fixed_Model_Simple/best_model_epoch_50.pth"
date_string = "2019-07-11T03"

print("="*70)
print("DIAGNOSTIC: Purple Map Issue")
print("="*70)

# Step 1: Check mask
print("\n[1] Checking MASK...")
try:
    mask = np.load(mask_path)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask data type: {mask.dtype}")
    print(f"   NaN pixels: {np.sum(np.isnan(mask))} / {mask.size}")
    print(f"   Valid pixels: {np.sum(~np.isnan(mask))} / {mask.size}")
    if np.sum(np.isnan(mask)) == 0:
        print("   ⚠️  WARNING: Mask has NO NaN values! This might be wrong.")
    if np.sum(np.isnan(mask)) == mask.size:
        print("   ❌ ERROR: Mask is ALL NaN! Nothing will be plotted.")
except Exception as e:
    print(f"   ❌ ERROR loading mask: {e}")

# Step 2: Check data loading
print("\n[2] Checking DATA LOADING...")
try:
    with xr.open_dataset(data_path) as ds:
        data1 = ds.sel(time=date_string).rf.values
    print(f"   Data shape: {data1.shape}")
    print(f"   Data dtype: {data1.dtype}")
    valid_data = data1[np.isfinite(data1)]
    if len(valid_data) > 0:
        print(f"   Data range: {np.nanmin(data1):.3f} to {np.nanmax(data1):.3f}")
        print(f"   Data mean: {np.nanmean(data1):.3f}, std: {np.nanstd(data1):.3f}")
    else:
        print(f"   ❌ ERROR: Data is all NaN!")
except Exception as e:
    print(f"   ❌ ERROR loading data: {e}")

# Step 3: Check label loading
print("\n[3] Checking LABEL (Ground Truth) LOADING...")
try:
    with xr.open_dataset(label_path) as ds:
        label1 = ds.sel(time=date_string).rf.values
    print(f"   Label shape: {label1.shape}")
    print(f"   Label dtype: {label1.dtype}")
    valid_label = label1[np.isfinite(label1)]
    if len(valid_label) > 0:
        print(f"   Label range: {np.nanmin(label1):.3f} to {np.nanmax(label1):.3f}")
        print(f"   Label mean: {np.nanmean(label1):.3f}, std: {np.nanstd(label1):.3f}")
    else:
        print(f"   ❌ ERROR: Label is all NaN!")
except Exception as e:
    print(f"   ❌ ERROR loading label: {e}")

# Step 4: Check normalization
print("\n[4] Checking NORMALIZATION SCALE...")
try:
    if len(valid_data) > 0:
        normalized = valid_data / 400.0
        print(f"   Original data range: {np.min(valid_data):.3f} to {np.max(valid_data):.3f}")
        print(f"   After norm (÷400): {np.min(normalized):.6f} to {np.max(normalized):.6f}")
        if np.max(normalized) < 0.01:
            print("   ⚠️  WARNING: Normalized values are very small (<0.01)")
        if np.max(normalized) > 100:
            print("   ⚠️  WARNING: Normalized values are very large (>100)")
except:
    pass

# Step 5: Check model architecture
print("\n[5] Checking MODEL ARCHITECTURE...")
try:
    model = ResCNNv4_Fixed(in_channels=1, out_channels=1, n_residual_blocks=8)
    print(f"   Model created successfully")
    print(f"   Model device: {device}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"   Dummy input shape: {dummy_input.shape}")
    print(f"   Dummy output shape: {dummy_output.shape}")
    print(f"   Dummy output range: {torch.min(dummy_output):.6f} to {torch.max(dummy_output):.6f}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Step 6: Check checkpoint
print("\n[6] Checking MODEL CHECKPOINT...")
if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location=device)
        print(f"   ✓ Checkpoint found at: {model_path}")
        print(f"   Checkpoint keys: {list(state.keys())}")
        
        # Check if checkpoint has weights
        if len(state) == 0:
            print("   ❌ WARNING: Checkpoint is empty!")
        else:
            sample_key = list(state.keys())[0]
            print(f"   Sample weight shape ({sample_key}): {state[sample_key].shape}")
    except Exception as e:
        print(f"   ❌ ERROR loading checkpoint: {e}")
else:
    print(f"   ❌ ERROR: Checkpoint NOT FOUND at {model_path}")
    print(f"   Available checkpoints in directory:")
    checkpoint_dir = os.path.dirname(model_path)
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            print(f"      - {item}")
    else:
        print(f"      Directory {checkpoint_dir} does not exist")

# Step 7: Full inference test
print("\n[7] RUNNING FULL INFERENCE TEST...")
try:
    # Load real data
    with xr.open_dataset(data_path) as ds:
        data_test = ds.sel(time=date_string).rf.values
    
    # Convert to tensor
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
    
    data_chw = to_chw(data_test)
    data_norm = data_chw.astype(np.float32) / 400.0
    data_tensor = torch.from_numpy(data_norm).unsqueeze(0)
    
    print(f"   Input tensor shape: {data_tensor.shape}")
    print(f"   Input tensor range: {torch.min(data_tensor):.6f} to {torch.max(data_tensor):.6f}")
    
    # Load model with checkpoint
    model = ResCNNv4_Fixed(in_channels=int(data_tensor.shape[1]), out_channels=1, n_residual_blocks=8)
    model.to(device=device)
    model.eval()
    
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"   ✓ Model weights loaded from checkpoint")
    else:
        print(f"   ⚠️  Using untrained model weights")
    
    # Inference
    with torch.no_grad():
        output = model(data_tensor.to(device=device))
    
    output_np = output.squeeze().cpu().numpy()
    output_denorm = output_np * 400.0
    
    print(f"   Output tensor shape: {output.shape}")
    print(f"   Output (normalized) range: {torch.min(output):.6f} to {torch.max(output):.6f}")
    print(f"   Output (denormalized) range: {np.nanmin(output_denorm):.3f} to {np.nanmax(output_denorm):.3f}")
    print(f"   Output std dev: {np.nanstd(output_denorm):.6f}")
    
    if np.nanstd(output_denorm) < 0.001:
        print(f"   ❌ PROBLEM FOUND: Output is nearly constant (std = {np.nanstd(output_denorm):.6e})")
        print(f"      This will cause the entire map to be one color (purple)")
        print(f"      Likely causes:")
        print(f"      - Model not trained or weights not loaded")
        print(f"      - Checkpoint file is wrong or corrupted")
        print(f"      - Data preprocessing/normalization issue")
    else:
        print(f"   ✓ Output has good variation (std = {np.nanstd(output_denorm):.3f})")
    
except Exception as e:
    print(f"   ❌ ERROR during inference: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nNEXT STEPS:")
print("1. Fix issues identified above")
print("2. Re-run this diagnostic script")
print("3. Once all issues are fixed, run: python test_trained_model_updated.py")
print("="*70)
