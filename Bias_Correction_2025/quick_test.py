"""Quick test of the trained bias correction model."""
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from robust.models.msua_net import MSUANet
from robust.data.datasets import RobustNormalizer

print("=" * 60)
print("Quick Model Test - IITM Pune Bias Correction")
print("=" * 60)

# Load checkpoint
checkpoint_path = "checkpoints/robust_bc/best_model.pth"
print(f"\nLoading checkpoint: {checkpoint_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
print(f"Checkpoint keys: {checkpoint.keys()}")

# Create model
model = MSUANet(
    in_channels=10,
    out_channels=1,
    base_filters=64,
    num_levels=4,
    uncertainty=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Model loaded successfully!")

# Load a single test sample
nwp_path = "d:/shivanshu/MY FILES/data/gfs model/GFS_3h_JJAS_india_2019to23_filled.nc"
obs_path = "d:/shivanshu/MY FILES/data/imdaa data truth image/IMDAA_3h_JJAS_india_2019to23_filled.nc"

print(f"\nLoading test data...")
with xr.open_dataset(nwp_path) as ds:
    nwp_sample = ds.rf.isel(time=4000).values  # Pick a sample from validation set
    
with xr.open_dataset(obs_path) as ds:
    obs_sample = ds.rf.isel(time=4000).values

print(f"NWP shape: {nwp_sample.shape}")
print(f"OBS shape: {obs_sample.shape}")

# Normalize (simple percentile normalization)
nwp_sample = np.nan_to_num(nwp_sample, nan=0.0)
nwp_sample = np.maximum(nwp_sample, 0)
nwp_sample = np.clip(nwp_sample, 0, 7.1875) / 7.1875

obs_sample = np.nan_to_num(obs_sample, nan=0.0)
obs_sample = np.maximum(obs_sample, 0)

# Run inference
nwp_tensor = torch.from_numpy(nwp_sample.astype(np.float32)).unsqueeze(0).to(device)
print(f"Input tensor shape: {nwp_tensor.shape}")

with torch.no_grad():
    pred_mean, pred_var = model(nwp_tensor)
    
print(f"Output mean shape: {pred_mean.shape}")
print(f"Output variance shape: {pred_var.shape}")

# Convert back to numpy
pred = pred_mean.squeeze().cpu().numpy()
pred = pred * 7.1875  # Denormalize

# Calculate metrics
nwp_mean = nwp_sample.mean() * 7.1875
obs_mean = obs_sample.mean()
pred_mean_val = pred.mean()

mae_before = np.abs(nwp_sample.mean() * 7.1875 - obs_sample.mean())
mae_after = np.abs(pred.mean() - obs_sample.mean())

print(f"\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"NWP mean:         {nwp_mean:.4f} mm")
print(f"Observation mean: {obs_mean:.4f} mm")
print(f"Prediction mean:  {pred_mean_val:.4f} mm")
print(f"\nMAE before correction: {mae_before:.4f}")
print(f"MAE after correction:  {mae_after:.4f}")
print(f"Improvement: {((mae_before - mae_after) / mae_before * 100):.1f}%")

# Save comparison plot
print(f"\nGenerating comparison plot...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# NWP input (first channel)
im0 = axes[0].imshow(nwp_sample[0] * 7.1875, cmap='Blues', vmin=0, vmax=50)
axes[0].set_title('NWP Input (Channel 1)')
plt.colorbar(im0, ax=axes[0], label='mm')

# Observation
im1 = axes[1].imshow(obs_sample, cmap='Blues', vmin=0, vmax=50)
axes[1].set_title('Observation (Ground Truth)')
plt.colorbar(im1, ax=axes[1], label='mm')

# Prediction
im2 = axes[2].imshow(pred, cmap='Blues', vmin=0, vmax=50)
axes[2].set_title('Model Prediction')
plt.colorbar(im2, ax=axes[2], label='mm')

# Difference
diff = pred - obs_sample
im3 = axes[3].imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20)
axes[3].set_title('Error (Pred - Obs)')
plt.colorbar(im3, ax=axes[3], label='mm')

plt.tight_layout()
plt.savefig('quick_test_result.png', dpi=150)
print(f"Saved: quick_test_result.png")

print(f"\n✅ Model test completed successfully!")
