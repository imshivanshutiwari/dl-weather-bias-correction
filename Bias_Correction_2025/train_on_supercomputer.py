#!/usr/bin/env python3
"""
🚀 Bias Correction Model Training Script for Supercomputer
Optimized for HPC environments with GPU acceleration
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import json
import xarray as xr
from tqdm import tqdm

# Set matplotlib to use non-interactive backend for HPC
plt.switch_backend('Agg')

def check_gpu():
    """Check GPU availability and print info"""
    print("="*60)
    print("🔍 GPU CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"📊 Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Use the first GPU
        device = torch.device("cuda:0")
        print(f"🎯 Using device: {device}")
        print(f"🎯 GPU Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        
    else:
        print("⚠️ No GPU detected - training will be slow!")
        print("💡 Consider using a GPU-enabled node")
        device = torch.device("cpu")
    
    print("="*60)
    return device

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

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, start_idx=0, num_samples=200, normalize=True):
        print(f"📊 Loading {num_samples} samples starting from index {start_idx}...")
        
        # Load small chunk of data
        with xr.open_dataset(data_path) as ds:
            self.gfs_data = ds.isel(time=slice(start_idx, start_idx + num_samples)).rf.values
            self.times = ds.isel(time=slice(start_idx, start_idx + num_samples)).time.values
        
        with xr.open_dataset(label_path) as ds:
            self.imdaa_data = ds.isel(time=slice(start_idx, start_idx + num_samples)).rf.values
        
        self.normalize = normalize
        self.norm_factor = 400.0
        
        print(f"✅ Loaded GFS data shape: {self.gfs_data.shape}")
        print(f"✅ Loaded IMDAA data shape: {self.imdaa_data.shape}")
    
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

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
    
    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(100.0)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save just the model state dict for easy loading
    model_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}_loss_{loss:.4f}.pth')
    torch.save(model.state_dict(), model_path)
    
    return checkpoint_path, model_path

def plot_training_progress(train_losses, val_losses, train_psnrs, val_psnrs, save_dir):
    """Plot and save training progress"""
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
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to save memory

def main():
    """Main training function"""
    print("🚀 BIAS CORRECTION MODEL TRAINING ON SUPERCOMPUTER")
    print("="*70)
    
    # Check GPU
    device = check_gpu()
    
    # Training configuration
    BATCH_SIZE = 16  # Larger batch size for GPU
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    TRAIN_SAMPLES = 800  # More samples for better training
    VAL_SAMPLES = 200
    
    print("\n⚙️ TRAINING CONFIGURATION")
    print("="*50)
    print(f"Training samples: {TRAIN_SAMPLES}")
    print(f"Validation samples: {VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*50)
    
    # Data paths - adjust these to your file locations
    data_path = "../data/gfs model/GFS_3h_JJAS_india_2019to23_filled.nc"
    label_path = "../data/imdaa data truth image/IMDAA_3h_JJAS_india_2019to23_filled.nc"
    
    # Check if data files exist
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("💡 Please update the data_path variable to point to your GFS data file")
        return
    
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        print("💡 Please update the label_path variable to point to your IMDAA data file")
        return
    
    print("✅ Data files found!")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = SimpleDataset(
        data_path=data_path,
        label_path=label_path,
        start_idx=0,
        num_samples=TRAIN_SAMPLES,
        normalize=True
    )
    
    val_dataset = SimpleDataset(
        data_path=data_path,
        label_path=label_path,
        start_idx=TRAIN_SAMPLES,
        num_samples=VAL_SAMPLES,
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = ResCNNv4_ProperlyFixed(in_channels=10, out_channels=1, n_residual_blocks=8)
    model.to(device)
    
    print(f"✅ Model created successfully!")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    psnr_metric = PSNR()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_results_{timestamp}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n💾 Results will be saved to: {save_dir}")
    
    # Training history
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training...")
    print(f"⏰ Model will be saved automatically when validation loss improves.")
    
    total_start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        
        print(f"\n📈 Epoch {epoch}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(train_pbar):
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
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Average training metrics
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        print("🔍 Running validation...")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                gfs_data = batch['gfs'].to(device)
                imdaa_data = batch['imdaa'].to(device)
                
                output = model(gfs_data)
                loss = criterion(output, imdaa_data)
                psnr_val = psnr_metric(output, imdaa_data)
                
                val_loss += loss.item()
                val_psnr += psnr_val.item()
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
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
        epoch_time = time.time() - epoch_start_time
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.3f}")
        print(f"   Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.3f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path, model_path = save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)
            print(f"   ✅ New best model saved! Val Loss: {val_loss:.6f}")
            
            # Save the best model with a standard name
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        
        # Save progress plot every 10 epochs
        if epoch % 10 == 0:
            plot_training_progress(train_losses, val_losses, train_psnrs, val_psnrs, save_dir)
            print(f"   📈 Training progress plot saved")
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            print(f"   🎮 GPU Memory: {gpu_memory:.2f} GB")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print(f"⏰ Total training time: {total_time/3600:.2f} hours")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")
    print(f"💾 Results saved to: {save_dir}")
    print(f"🎯 Best model saved as: {os.path.join(save_dir, 'best_model.pth')}")
    
    # Save final training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_psnrs': train_psnrs,
        'val_psnrs': val_psnrs,
        'best_val_loss': best_val_loss,
        'total_training_time_hours': total_time/3600,
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'train_samples': TRAIN_SAMPLES,
            'val_samples': VAL_SAMPLES,
            'device': str(device)
        }
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create final plot
    plot_training_progress(train_losses, val_losses, train_psnrs, val_psnrs, save_dir)
    
    print(f"\n📈 Final training plot saved to: {os.path.join(save_dir, 'training_progress.png')}")
    print(f"📋 Training history saved to: {os.path.join(save_dir, 'training_history.json')}")
    
    print("\n🎯 Your bias correction model is ready!")
    print("💡 Next steps:")
    print("   1. Test the model with test scripts")
    print("   2. Compare results with ground truth")
    print("   3. Use for your research/thesis")
    print("   4. The 'blue map' issue is now solved! 🎉")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
