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

from config import CONFIG
from model_4_properly_fixed import ResCNNv4_ProperlyFixed
from dataset_efficient import SimpleDataset
from cnnbc_utils import PSNR

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
    plt.show()

def train_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Training configuration
    BATCH_SIZE = 4  # Small batch size to avoid memory issues
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    TRAIN_SAMPLES = 300  # Use 300 samples for training
    VAL_SAMPLES = 50     # Use 50 samples for validation
    
    print("="*60)
    print("BIAS CORRECTION MODEL TRAINING")
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
        data_path=CONFIG["DATA_PATH"],
        label_path=CONFIG["LABEL_PATH"],
        start_idx=0,
        num_samples=TRAIN_SAMPLES,
        normalize=True
    )
    
    val_dataset = SimpleDataset(
        data_path=CONFIG["DATA_PATH"],
        label_path=CONFIG["LABEL_PATH"],
        start_idx=TRAIN_SAMPLES,
        num_samples=VAL_SAMPLES,
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = ResCNNv4_ProperlyFixed(in_channels=10, out_channels=1, n_residual_blocks=8)
    model.to(device)
    
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
    
    # Training history
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []
    best_val_loss = float('inf')
    
    print(f"Starting training...")
    print(f"Results will be saved to: {save_dir}")
    
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
            checkpoint_path, model_path = save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)
            print(f"  ✅ New best model saved! Val Loss: {val_loss:.6f}")
            
            # Save the best model with a standard name
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        
        # Save progress plot every 5 epochs
        if epoch % 5 == 0:
            plot_training_progress(train_losses, val_losses, train_psnrs, val_psnrs, save_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {save_dir}")
    print(f"Best model saved as: {os.path.join(save_dir, 'best_model.pth')}")
    
    # Save final training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_psnrs': train_psnrs,
        'val_psnrs': val_psnrs,
        'best_val_loss': best_val_loss,
        'config': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'train_samples': TRAIN_SAMPLES,
            'val_samples': VAL_SAMPLES
        }
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create final plot
    plot_training_progress(train_losses, val_losses, train_psnrs, val_psnrs, save_dir)
    
    return os.path.join(save_dir, "best_model.pth"), save_dir

if __name__ == "__main__":
    try:
        best_model_path, results_dir = train_model()
        print(f"\n🎉 Training successful!")
        print(f"📁 Best model: {best_model_path}")
        print(f"📁 All results: {results_dir}")
        print(f"\n🚀 Next step: Test the trained model to see the bias correction working!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
