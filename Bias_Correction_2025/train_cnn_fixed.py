#Code for CNNBC training with FIXED components
import os
import sys
from config import CONFIG
RUN_NAME = "ResCNNv4_Fixed_2025"

import time
from datetime import date, datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
import json
import gc

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from dataset2_fixed import GFSdataset_Fixed
from model_4_fixed import ResCNNv4_Fixed
from cnnbc_utils import PSNR,EarlyStopper,plotLosses,plotPSNRs

torch.backends.cudnn.benchmark = True
# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
#Additional Info when using cuda
if device == 'cuda':
    print("GPU : ", torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
    print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
    

# Dataset and dataloader - Using FIXED dataset with proper normalization
split_ratio = [0.7,0.15,0.15]  #70% traning 15% validation 15% testing
split_index = [int(split_ratio[0]*4880), int(sum(split_ratio[:2])*4880), int(sum(split_ratio[:])*4880)]

# IMPORTANT: Set normalize_data=True to ensure consistent normalization
train_dataset = GFSdataset_Fixed(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"], 
                                idxs=(0,split_index[0]), normalize_data=True)
val_dataset = GFSdataset_Fixed(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"], 
                              idxs=(split_index[0],split_index[1]), normalize_data=True)

train_loader = DataLoader(dataset=train_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = True, num_workers=8)
val_loader = DataLoader(dataset=val_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = False, num_workers=8)

# Model instance - Using FIXED model
model = ResCNNv4_Fixed()
model = nn.DataParallel(model)
model.to(device=device)

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
psnr_metric = PSNR()
optimizer = optim.Adam(model.parameters(), lr = CONFIG["LEARNING_RATE"])
num_epochs = CONFIG["NUM_EPOCHS"]
early_stopper = EarlyStopper(patience=CONFIG["PATIENCE"], min_delta=0.0)

#########################################################################################################################################################

todaysdate=date.today().strftime("%b-%d-%Y") + "_1"
# save path
save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, todaysdate, "")
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save parameters
params={'lr':CONFIG["LEARNING_RATE"],'batch_size':CONFIG["BATCH_SIZE"], 'split':split_ratio,
        'remark':'FIXED MODEL: IMDAA 3h as ground truth, WITH Normalization, Fixed architecture'}
with open(save_path+"params.json", "w") as f1:
    json.dump(params,f1)

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Started at: {now}")
best_loss=float('inf')
best_psnr = 0.0
Val_Loss=0.0
train_losses=[]
val_losses=[]
train_psnrs=[]
val_psnrs=[]
train_maes=[]
val_maes=[]
val_epochs=[]
stopflag=False

for epoch in range(1, num_epochs + 1):
    print(f"> Starting epoch {epoch}/{num_epochs}")
    
    train_epoch_loss = 0.0
    train_epoch_psnr = 0.0
    train_epoch_mae = 0.0
    
    t0 = time.time()
    
    # Training phase
    model.train()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        gfs_data = batch["gfs"].to(device=device, dtype=torch.float)
        imdaa_data = batch["imdaa"].to(device=device, dtype=torch.float)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(gfs_data)
            loss = mse_loss(output, imdaa_data)
        
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        
        # Calculate PSNR and MAE
        with torch.no_grad():
            psnr_val = psnr_metric(output, imdaa_data)
            mae_val = mae_loss(output, imdaa_data)
            train_epoch_psnr += psnr_val.item()
            train_epoch_mae += mae_val.item()
    
    # Calculate averages
    train_epoch_loss /= len(train_loader)
    train_epoch_psnr /= len(train_loader)
    train_epoch_mae /= len(train_loader)
    
    t1 = time.time()
    print(f"Train Loss:  {train_epoch_loss}")
    print(f"Train PSNR:  {train_epoch_psnr}")
    print(f"Train MAE:  {train_epoch_mae}")
    print(f">>> Epoch {epoch} finished in {t1-t0:.2f} seconds")
    
    # Validation phase
    if epoch % CONFIG["EVAL_EVERY"] == 0:
        print(f"> Starting validation after epoch {epoch}")
        
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_psnr = 0.0
        val_epoch_mae = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                gfs_data = batch["gfs"].to(device=device, dtype=torch.float)
                imdaa_data = batch["imdaa"].to(device=device, dtype=torch.float)
                
                with torch.cuda.amp.autocast():
                    output = model(gfs_data)
                    loss = mse_loss(output, imdaa_data)
                
                val_epoch_loss += loss.item()
                
                # Calculate PSNR and MAE
                psnr_val = psnr_metric(output, imdaa_data)
                mae_val = mae_loss(output, imdaa_data)
                val_epoch_psnr += psnr_val.item()
                val_epoch_mae += mae_val.item()
        
        # Calculate averages
        val_epoch_loss /= len(val_loader)
        val_epoch_psnr /= len(val_loader)
        val_epoch_mae /= len(val_loader)
        
        print(f"Val Loss:  {val_epoch_loss}")
        print(f"Val PSNR:  {val_epoch_psnr}")
        print(f"Val MAE:  {val_epoch_mae}")
        
        # Save best model
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.module.state_dict(), save_path + f"Val-Loss_{val_epoch_loss:.3f}.pth")
            print(f"Model saved on epoch: {epoch} with validation loss: {val_epoch_loss:.3f}")
        
        if val_epoch_psnr > best_psnr:
            best_psnr = val_epoch_psnr
            torch.save(model.module.state_dict(), save_path + f"Val-PSNR_{val_epoch_psnr:.3f}.pth")
            print(f"Model saved on epoch: {epoch} with validation PSNR: {val_epoch_psnr:.3f}")
        
        # Store metrics
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_psnrs.append(train_epoch_psnr)
        val_psnrs.append(val_epoch_psnr)
        train_maes.append(train_epoch_mae)
        val_maes.append(val_epoch_mae)
        val_epochs.append(epoch)
        
        # Early stopping
        if early_stopper.early_stop(val_epoch_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save training log
    if epoch % CONFIG["EVAL_EVERY"] == 0:
        df = pd.DataFrame({
            'Epoch': val_epochs,
            'Train_Loss': train_losses,
            'Val_Loss': val_losses,
            'Train_PSNR': train_psnrs,
            'Val_PSNR': val_psnrs,
            'Train_MAE': train_maes,
            'Val_MAE': val_maes
        })
        df.to_csv(save_path + "TrainLog.csv", index=False)
        
        # Plot metrics
        plotLosses(train_losses, val_losses, val_epochs, "MSE Model loss", save_path)
        plotLosses(train_maes, val_maes, val_epochs, "MAE Model loss", save_path)
        plotPSNRs(train_psnrs, val_psnrs, val_epochs, save_path)

print("Training completed!")
