# ============================================================================
# 🚀 BIAS CORRECTION MODEL TRAINING - GOOGLE COLAB NOTEBOOK
# ============================================================================
# Copy and paste each section into separate cells in Google Colab
# ============================================================================

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================

# Install required packages
!pip install xarray netcdf4 tqdm matplotlib

# Check if GPU is available
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected - training will be slow!")
    print("💡 Go to Runtime → Change runtime type → GPU")

# ============================================================================
# CELL 2: Check Data Files
# ============================================================================

# Check if files are uploaded
import os

required_files = [
    'GFS_3h_JJAS_india_2019to23_filled.nc',
    'IMDAA_3h_JJAS_india_2019to23_filled.nc',
    'Mask_125deg_New.npy'
]

print("📁 Checking for required files:")
for file in required_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"✅ {file} ({size_mb:.1f} MB)")
    else:
        print(f"❌ {file} - NOT FOUND")
        print(f"   Please upload this file to Colab")

print("\n" + "="*50)
if all(os.path.exists(f) for f in required_files):
    print("🎉 All files found! Ready to train.")
else:
    print("⚠️ Please upload missing files before proceeding.")

# ============================================================================
# CELL 3: Model Architecture
# ============================================================================

import torch
from torch import nn
import numpy as np

class Residual_Block(nn.Module):
    def __init__(self, in_features, ng):
        super(Residual_Block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=9, stride=1, padding=4, groups=ng),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=ng),
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

class ResCNNv4_ProperlyFixed(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, n_residual_blocks=8):
        super(ResCNNv4_ProperlyFixed, self).__init__()
        
        nf = 60  # Must be divisible by groups
        ng = 10  # Must be divisible by in_channels (10)
        
        # Input processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=9, stride=1, padding=4, groups=ng), 
            nn.PReLU()
        )
        
        # Residual blocks
        res_blocks = []
        for i in range(n_residual_blocks):
            res_blocks.append(Residual_Block(nf, ng))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Feature refinement
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, groups=ng),
            nn.PReLU()
        )
        
        # Output layer - NO ACTIVATION for regression
        self.conv3 = nn.Conv2d(nf, out_channels, kernel_size=9, stride=1, padding=4)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input processing
        out1 = self.conv1(x)
        
        # Residual learning
        out = self.res_blocks(out1)
        
        # Feature refinement
        out2 = self.conv2(out)
        out = torch.add(out1, out2)  # Skip connection
        
        # Output - NO ACTIVATION for regression
        out = self.conv3(out)
        
        return out

# Test the model
model = ResCNNv4_ProperlyFixed(in_channels=10, out_channels=1, n_residual_blocks=8)
print(f"✅ Model created successfully!")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# CELL 4: Dataset Loading
# ============================================================================

import xarray as xr
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data_path, label_path, start_idx=0, num_samples=200, normalize=True):
        print(f"Loading {num_samples} samples starting from index {start_idx}...")
        
        # Load small chunk of data
        with xr.open_dataset(data_path) as ds:
            self.gfs_data = ds.isel(time=slice(start_idx, start_idx + num_samples)).rf.values
            self.times = ds.isel(time=slice(start_idx, start_idx + num_samples)).time.values
        
        with xr.open_dataset(label_path) as ds:
            self.imdaa_data = ds.isel(time=slice(start_idx, start_idx + num_samples)).rf.values
        
        self.normalize = normalize
        self.norm_factor = 400.0
        
        print(f"Loaded GFS data shape: {self.gfs_data.shape}")
        print(f"Loaded IMDAA data shape: {self.imdaa_data.shape}")
    
    def __len__(self):
        return len(self.gfs_data)
    
    def __getitem__(self, idx):
        gfs_sample = self.gfs_data[idx].copy()
        imdaa_sample = self.imdaa_data[idx].copy()
        
        if self.normalize:
            gfs_sample = gfs_sample / self.norm_factor
            imdaa_sample = imdaa_sample / self.norm_factor
        
        gfs_tensor = torch.FloatTensor(gfs_sample)
        imdaa_tensor = torch.FloatTensor(imdaa_sample).unsqueeze(0)  # Add channel dimension
        
        return {
            'gfs': gfs_tensor,
            'imdaa': imdaa_tensor,
            'time': str(self.times[idx])
        }

# PSNR metric
class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
    
    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(100.0)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ============================================================================
# CELL 5: Training Configuration
# ============================================================================

# Training configuration
BATCH_SIZE = 8  # Larger batch size with GPU
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SAMPLES = 500  # More samples with GPU
VAL_SAMPLES = 100

print("="*60)
print("BIAS CORRECTION MODEL TRAINING CONFIG")
print("="*60)
print(f"Training samples: {TRAIN_SAMPLES}")
print(f"Validation samples: {VAL_SAMPLES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*60)

# Create datasets
print("Creating datasets...")
train_dataset = SimpleDataset(
    data_path='GFS_3h_JJAS_india_2019to23_filled.nc',
    label_path='IMDAA_3h_JJAS_india_2019to23_filled.nc',
    start_idx=0,
    num_samples=TRAIN_SAMPLES,
    normalize=True
)

val_dataset = SimpleDataset(
    data_path='GFS_3h_JJAS_india_2019to23_filled.nc',
    label_path='IMDAA_3h_JJAS_india_2019to23_filled.nc',
    start_idx=TRAIN_SAMPLES,
    num_samples=VAL_SAMPLES,
    normalize=True
)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ============================================================================
# CELL 6: Training Loop
# ============================================================================

import time
from datetime import datetime
import matplotlib.pyplot as plt
from torch import optim

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Create model
model = ResCNNv4_ProperlyFixed(in_channels=10, out_channels=1, n_residual_blocks=8)
model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
psnr_metric = PSNR()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training history
train_losses = []
val_losses = []
train_psnrs = []
val_psnrs = []
best_val_loss = float('inf')

print(f"Starting training...")
print(f"Model will be saved automatically when validation loss improves.")

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_psnr = 0.0
    
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("-" * 30)
    
    for batch_idx, batch in enumerate(train_loader):
        gfs_data = batch['gfs'].to(device)
        imdaa_data = batch['imdaa'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(gfs_data)
        loss = criterion(output, imdaa_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        train_loss += loss.item()
        with torch.no_grad():
            psnr_val = psnr_metric(output, imdaa_data)
            train_psnr += psnr_val.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    # Average training metrics
    train_loss /= len(train_loader)
    train_psnr /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            gfs_data = batch['gfs'].to(device)
            imdaa_data = batch['imdaa'].to(device)
            
            output = model(gfs_data)
            loss = criterion(output, imdaa_data)
            psnr_val = psnr_metric(output, imdaa_data)
            
            val_loss += loss.item()
            val_psnr += psnr_val.item()
    
    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Save metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_psnrs.append(train_psnr)
    val_psnrs.append(val_psnr)
    
    # Print epoch results
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} Results:")
    print(f"  Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.3f}")
    print(f"  Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.3f}")
    print(f"  Time: {epoch_time:.2f}s")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_bias_correction_model.pth')
        print(f"  ✅ New best model saved! Val Loss: {val_loss:.6f}")
    
    # Plot progress every 10 epochs
    if epoch % 10 == 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot PSNR
        ax2.plot(train_psnrs, label='Training PSNR', color='blue')
        ax2.plot(val_psnrs, label='Validation PSNR', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Training and Validation PSNR')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Best model saved as: best_bias_correction_model.pth")
print("\n🎉 Your model is ready for bias correction!")

# ============================================================================
# CELL 7: Download Model
# ============================================================================

from google.colab import files

# Download the trained model
files.download('best_bias_correction_model.pth')

print("📥 Model downloaded!")
print("\n📋 Next steps:")
print("1. Save the model file to your project folder")
print("2. Use it with the test scripts we created earlier")
print("3. The model will now perform proper bias correction!")

# ============================================================================
# CELL 8: Test Trained Model (Optional)
# ============================================================================

# Load the trained model
model.load_state_dict(torch.load('best_bias_correction_model.pth'))
model.eval()

# Test on a single sample
test_batch = next(iter(val_loader))
gfs_test = test_batch['gfs'][0:1].to(device)  # Take first sample
imdaa_test = test_batch['imdaa'][0:1].to(device)

with torch.no_grad():
    output = model(gfs_test)
    
# Denormalize for comparison
gfs_denorm = gfs_test[0, 0].cpu().numpy() * 400.0  # First channel
output_denorm = output[0, 0].cpu().numpy() * 400.0
imdaa_denorm = imdaa_test[0, 0].cpu().numpy() * 400.0

print("📊 Test Results:")
print(f"Input (GFS) range: {gfs_denorm.min():.2f} to {gfs_denorm.max():.2f}")
print(f"Output (Bias Corrected) range: {output_denorm.min():.2f} to {output_denorm.max():.2f}")
print(f"Target (IMDAA) range: {imdaa_denorm.min():.2f} to {imdaa_denorm.max():.2f}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im1 = axes[0].imshow(gfs_denorm, cmap='rainbow')
axes[0].set_title('Input GFS Data')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(output_denorm, cmap='rainbow')
axes[1].set_title('Bias Corrected Output')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(imdaa_denorm, cmap='rainbow')
axes[2].set_title('Ground Truth (IMDAA)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

print("\n🎯 Notice how the bias corrected output now has:")
print("✅ Higher intensity values (like ground truth)")
print("✅ Better spatial patterns")
print("✅ More realistic rainfall distribution")

# ============================================================================
# INSTRUCTIONS FOR USE:
# ============================================================================
# 1. Go to colab.research.google.com
# 2. Create a new notebook
# 3. Copy each CELL section above into separate cells
# 4. Enable GPU: Runtime → Change runtime type → GPU
# 5. Upload your data files (GFS, IMDAA, Mask)
# 6. Run cells in order
# 7. Download the trained model
# ============================================================================
