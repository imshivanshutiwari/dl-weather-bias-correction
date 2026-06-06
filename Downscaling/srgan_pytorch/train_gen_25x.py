# Training script for 25x downscaling SRGAN (0.625° to 0.025°)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from datetime import date, datetime

from models_25x import Generator_25x, Discriminator_25x, ProgressiveGenerator_25x
from srgan_utils import PSNR, EarlyStopper, plotLosses, plotPSNRs

# Configuration for 25x downscaling
CONFIG_25x = {
    "GPU_NUM": "1",
    "BATCH_SIZE": 32,  # Reduced batch size due to larger model
    "NUM_EPOCHS": 200,
    "EVAL_EVERY": 1,
    "LEARNING_RATE": 1e-07,
    "MOMENTUM": 0.9,
    "PATIENCE": 25,
    "FACTOR": 0.1,
    "DISC_LEAD": 0,
    
         "TRAIN_PATH": "data/gfs model/",
     "VAL_PATH": "data/imdaa data truth image/",
     "COMB_PATH": "data/",
    "ORO_PATH": "IMD_DATA/ORO_SCALED.nc",
    
    "CHECKPOINT_PATH": "checkpoints/SRGAN_25x_2025/"
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
if device == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')

# Data loading functions for 25x downscaling
def load_25x_data(data_path, is_training=True):
    """Load data for 25x downscaling (0.625° to 0.025°)"""
    print(f"Loading {'training' if is_training else 'validation'} data for 25x downscaling")
    
    # Load low-resolution data (0.625°)
    lr_data = xr.open_dataset(data_path + "MPNorm_LR_0625deg.nc").rf.values
    # Load high-resolution data (0.025°)
    hr_data = xr.open_dataset(data_path + "Norm_HR_0025deg.nc").rf.values
    
    return lr_data, hr_data

def create_25x_dataloader(lr_data, hr_data, batch_size=32, shuffle=True):
    """Create dataloader for 25x downscaling"""
    # Convert to tensors
    lr_tensor = torch.FloatTensor(lr_data)
    hr_tensor = torch.FloatTensor(hr_data)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(lr_tensor, hr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    return dataloader

# Training function for 25x downscaling
def train_25x_srgan():
    """Train SRGAN for 25x downscaling"""
    
    # Load data
    lr_train, hr_train = load_25x_data(CONFIG_25x["TRAIN_PATH"], is_training=True)
    lr_val, hr_val = load_25x_data(CONFIG_25x["VAL_PATH"], is_training=False)
    
    # Create dataloaders
    train_loader = create_25x_dataloader(lr_train, hr_train, CONFIG_25x["BATCH_SIZE"], shuffle=True)
    val_loader = create_25x_dataloader(lr_val, hr_val, CONFIG_25x["BATCH_SIZE"], shuffle=False)
    
    # Initialize models
    generator = Generator_25x(in_channels=1, out_channels=1, n_residual_blocks=8)
    discriminator = Discriminator_25x(input_shape=(1, 512, 512))  # Adjusted for 25x output
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    psnr_metric = PSNR()
    
    # Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=CONFIG_25x["LEARNING_RATE"])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=CONFIG_25x["LEARNING_RATE"])
    
    # Learning rate schedulers
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=50, gamma=0.5)
    
    # Early stopping
    early_stopper = EarlyStopper(patience=CONFIG_25x["PATIENCE"], min_delta=0.0)
    
    # Save path
    todaysdate = date.today().strftime("%b-%d-%Y") + "_1"
    save_path = os.path.join(CONFIG_25x["CHECKPOINT_PATH"], todaysdate, "")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save parameters
    params = {
        'lr': CONFIG_25x["LEARNING_RATE"],
        'batch_size': CONFIG_25x["BATCH_SIZE"],
        'remark': '25x downscaling SRGAN (0.625° to 0.025°)',
        'architecture': 'Generator_25x with 5 upsampling layers'
    }
    with open(save_path + "params.json", "w") as f1:
        json.dump(params, f1)
    
    print(f">>> Training Started at: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Target: 25x downscaling (0.625° to 0.025°)")
    
    best_psnr = 0.0
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []
    val_epochs = []
    
    for epoch in range(1, CONFIG_25x["NUM_EPOCHS"] + 1):
        print(f"> Starting epoch {epoch}/{CONFIG_25x['NUM_EPOCHS']}")
        
        # Training phase
        generator.train()
        discriminator.train()
        
        train_epoch_loss = 0.0
        train_epoch_psnr = 0.0
        
        t0 = time.time()
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(tqdm(train_loader)):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            # Train Discriminator
            disc_optimizer.zero_grad()
            
            # Real images
            real_labels = torch.ones(hr_batch.size(0), 1).to(device)
            real_output = discriminator(hr_batch)
            d_real_loss = bce_loss(real_output, real_labels)
            
            # Fake images
            fake_images = generator(lr_batch)
            fake_labels = torch.zeros(fake_images.size(0), 1).to(device)
            fake_output = discriminator(fake_images.detach())
            d_fake_loss = bce_loss(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            disc_optimizer.step()
            
            # Train Generator
            gen_optimizer.zero_grad()
            
            # Content loss (MSE)
            content_loss = mse_loss(fake_images, hr_batch)
            
            # Adversarial loss
            fake_output = discriminator(fake_images)
            adversarial_loss = bce_loss(fake_output, real_labels)
            
            # Total generator loss
            g_loss = content_loss + 0.001 * adversarial_loss
            g_loss.backward()
            gen_optimizer.step()
            
            # Calculate PSNR
            with torch.no_grad():
                psnr_val = psnr_metric(fake_images, hr_batch)
                train_epoch_psnr += psnr_val.item()
            
            train_epoch_loss += g_loss.item()
        
        # Calculate averages
        train_epoch_loss /= len(train_loader)
        train_epoch_psnr /= len(train_loader)
        
        t1 = time.time()
        print(f"Train Loss: {train_epoch_loss:.6f}")
        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f">>> Epoch {epoch} finished in {t1-t0:.2f} seconds")
        
        # Validation phase
        if epoch % CONFIG_25x["EVAL_EVERY"] == 0:
            print(f"> Starting validation after epoch {epoch}")
            
            generator.eval()
            val_epoch_loss = 0.0
            val_epoch_psnr = 0.0
            
            with torch.no_grad():
                for lr_batch, hr_batch in tqdm(val_loader):
                    lr_batch = lr_batch.to(device)
                    hr_batch = hr_batch.to(device)
                    
                    fake_images = generator(lr_batch)
                    val_loss = mse_loss(fake_images, hr_batch)
                    val_psnr = psnr_metric(fake_images, hr_batch)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_psnr += val_psnr.item()
            
            val_epoch_loss /= len(val_loader)
            val_epoch_psnr /= len(val_loader)
            
            print(f"Val Loss: {val_epoch_loss:.6f}")
            print(f"Val PSNR: {val_epoch_psnr:.3f}")
            
            # Save best model
            if val_epoch_psnr > best_psnr:
                best_psnr = val_epoch_psnr
                torch.save(generator.state_dict(), save_path + f"Val-PSNR_{val_epoch_psnr:.3f}.pth")
                print(f"Model saved on epoch: {epoch} with validation PSNR: {val_epoch_psnr:.3f}")
            
            # Store metrics
            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)
            train_psnrs.append(train_epoch_psnr)
            val_psnrs.append(val_epoch_psnr)
            val_epochs.append(epoch)
            
            # Early stopping
            if early_stopper.early_stop(val_epoch_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Update learning rates
        gen_scheduler.step()
        disc_scheduler.step()
        
        # Save training log
        if epoch % CONFIG_25x["EVAL_EVERY"] == 0:
            df = pd.DataFrame({
                'Epoch': val_epochs,
                'Train_Loss': train_losses,
                'Val_Loss': val_losses,
                'Train_PSNR': train_psnrs,
                'Val_PSNR': val_psnrs
            })
            df.to_csv(save_path + "TrainLog.csv", index=False)
            
            # Plot metrics
            plotLosses(train_losses, val_losses, val_epochs, "25x SRGAN Loss", save_path)
            plotPSNRs(train_psnrs, val_psnrs, val_epochs, save_path)
    
    print("Training completed!")
    print(f"Best PSNR achieved: {best_psnr:.3f}")

if __name__ == "__main__":
    train_25x_srgan()
