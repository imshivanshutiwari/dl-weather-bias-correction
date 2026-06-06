#!/usr/bin/env python3
"""
Quick check script for bias correction setup
Run this FIRST to identify your configuration
"""

import os
import sys
import numpy as np

print("="*70)
print("BIAS CORRECTION - QUICK SETUP CHECK")
print("="*70)

# Check 1: Current directory
print("\n[1] Current Working Directory:")
cwd = os.getcwd()
print(f"    {cwd}")
files_here = os.listdir(cwd)
print(f"    Files in directory: {len(files_here)}")
if 'config.py' in files_here:
    print("    ✓ config.py found")
else:
    print("    ❌ config.py NOT found")

# Check 2: Python packages
print("\n[2] Required Python Packages:")
packages = ['numpy', 'matplotlib', 'xarray', 'torch', 'netCDF4']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"    ✓ {pkg}: {version}")
    except ImportError:
        print(f"    ❌ {pkg}: NOT INSTALLED")

# Check 3: Directory structure
print("\n[3] Expected Directory Structure:")
dirs_to_check = [
    'checkpoints',
    'figures',
    'masks',
    'data',
    '../data/gfs model',
    '../data/imdaa data truth image',
]

for dir_path in dirs_to_check:
    exists = os.path.exists(dir_path)
    status = "✓" if exists else "❌"
    print(f"    {status} {dir_path}")

# Check 4: Key files
print("\n[4] Key Files:")
files_to_check = {
    'masks/Mask_125deg_New.npy': 'Mask file',
    '../data/gfs model/GFS_3h_JJAS_india_2019to23_filled.nc': 'GFS data',
    '../data/imdaa data truth image/IMDAA_3h_JJAS_india_2019to23_filled.nc': 'IMDAA label',
    'checkpoints/Fixed_Model_Simple/best_model_epoch_50.pth': 'Trained model',
}

for file_path, desc in files_to_check.items():
    exists = os.path.exists(file_path)
    status = "✓" if exists else "❌"
    # Get file size if exists
    if exists:
        try:
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"    {status} {desc}: {file_path} ({size_mb:.1f} MB)")
        except:
            print(f"    {status} {desc}: {file_path}")
    else:
        print(f"    {status} {desc}: {file_path}")

# Check 5: Config.py contents
print("\n[5] Config.py Settings:")
if 'config.py' in files_here:
    try:
        from config import CONFIG
        print("    Data path:", CONFIG.get("DATA_PATH", "NOT SET"))
        print("    Label path:", CONFIG.get("LABEL_PATH", "NOT SET"))
        print("    Checkpoint path:", CONFIG.get("CHECKPOINT_PATH", "NOT SET"))
    except Exception as e:
        print(f"    Error reading config: {e}")
else:
    print("    config.py not found - cannot read settings")

# Check 6: Checkpoint options
print("\n[6] Available Checkpoints:")
if os.path.exists('checkpoints'):
    found_any = False
    for root, dirs, files in os.walk('checkpoints'):
        for f in files:
            if f.endswith(('.pth', '.pt', '.keras')):
                found_any = True
                full_path = os.path.join(root, f)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                print(f"    {full_path} ({size_mb:.1f} MB)")
    if not found_any:
        print("    ❌ No checkpoint files (.pth, .pt, .keras) found!")
else:
    print("    ❌ checkpoints/ directory not found")

# Check 7: Data file inspection
print("\n[7] Data Files Inspection:")
data_file = '../data/gfs model/GFS_3h_JJAS_india_2019to23_filled.nc'
if os.path.exists(data_file):
    try:
        import xarray as xr
        ds = xr.open_dataset(data_file)
        print(f"    GFS data:")
        print(f"      Dimensions: {dict(ds.dims)}")
        print(f"      Variables: {list(ds.data_vars)}")
        print(f"      Time steps: {ds.dims.get('time', 'unknown')}")
        print(f"      Coordinates: {list(ds.coords)}")
        ds.close()
    except Exception as e:
        print(f"    Error reading GFS data: {e}")
else:
    print(f"    ❌ GFS data file not found: {data_file}")

label_file = '../data/imdaa data truth image/IMDAA_3h_JJAS_india_2019to23_filled.nc'
if os.path.exists(label_file):
    try:
        import xarray as xr
        ds = xr.open_dataset(label_file)
        print(f"    IMDAA data:")
        print(f"      Dimensions: {dict(ds.dims)}")
        print(f"      Variables: {list(ds.data_vars)}")
        print(f"      Time steps: {ds.dims.get('time', 'unknown')}")
        ds.close()
    except Exception as e:
        print(f"    Error reading IMDAA data: {e}")
else:
    print(f"    ❌ IMDAA data file not found: {label_file}")

# Check 8: Mask file
print("\n[8] Mask File Inspection:")
mask_file = 'masks/Mask_125deg_New.npy'
if os.path.exists(mask_file):
    try:
        mask = np.load(mask_file)
        print(f"    Mask shape: {mask.shape}")
        print(f"    Mask dtype: {mask.dtype}")
        nan_count = np.sum(np.isnan(mask))
        total = mask.size
        print(f"    Valid pixels: {total - nan_count} / {total} ({100*(total-nan_count)/total:.1f}%)")
        print(f"    Invalid pixels (NaN): {nan_count} / {total} ({100*nan_count/total:.1f}%)")
    except Exception as e:
        print(f"    Error reading mask: {e}")
else:
    print(f"    ❌ Mask file not found: {mask_file}")

# Summary
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS:")
print("="*70)

issues = []

# Check critical files
if not os.path.exists('checkpoints/Fixed_Model_Simple/best_model_epoch_50.pth'):
    issues.append("❌ Trained model checkpoint not found - train model first!")

if not os.path.exists('masks/Mask_125deg_New.npy'):
    issues.append("❌ Mask file not found")

if not os.path.exists('../data/gfs model/GFS_3h_JJAS_india_2019to23_filled.nc'):
    issues.append("❌ GFS data not found")

if not os.path.exists('../data/imdaa data truth image/IMDAA_3h_JJAS_india_2019to23_filled.nc'):
    issues.append("❌ IMDAA label data not found")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
    print("\n📝 ACTIONS:")
    print("   1. Update paths in config.py to match your system")
    print("   2. Train model: python train_simple.py")
    print("   3. Place data files in correct locations")
else:
    print("\n✓ All critical files found!")
    print("You can now run:")
    print("   python diagnose_purple_map.py")
    print("   python test_trained_model_fixed_v2.py")

print("="*70)
