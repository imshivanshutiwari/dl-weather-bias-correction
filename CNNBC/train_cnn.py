#Code for CNNBC training
import os
import sys
from config import CONFIG
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
RUN_NAME = "CNNBC_2025"

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

from dataset import CFSdataset
from model_cnnbc import CNNBC
from cnnbc_utils import EarlyStopper,plotLosses


torch.backends.cudnn.benchmark = True
# setting device on GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)
#Additional Info when using cuda
if device == 'cuda':
    print("GPU : ", torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
    print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
    

# Dataset and dataloader
train_split = 0.8
split_index = int(train_split*1290)

train_dataset = CFSdataset(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"],idxs=(0,split_index))
val_dataset = CFSdataset(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"],idxs=(split_index,None))

train_loader = DataLoader(dataset=train_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = True, num_workers=8)
val_loader = DataLoader(dataset=val_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = False, num_workers=8)


# Model instance
cnnbc_model = CNNBC()
cnnbc_model.to(device=device)

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
optimizer = optim.Adam(cnnbc_model.parameters(), lr = CONFIG["LEARNING_RATE"])
num_epochs = CONFIG["NUM_EPOCHS"]
early_stopper = EarlyStopper(patience=CONFIG["PATIENCE"], min_delta=0.0)

##############################################################################

todaysdate=date.today().strftime("%b-%d-%Y") + "_2"
# save path
save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, todaysdate, "")
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save parameters
params={'lr':CONFIG["LEARNING_RATE"],'batch_size':CONFIG["BATCH_SIZE"],
        'remark':''}
with open(save_path+"params.json", "w") as f1:
    json.dump(params,f1)


now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Started at: {now}")
best_loss=float('inf')
Val_Loss=0.0
train_losses=[]
val_losses=[]
train_maes=[]
val_maes=[]
val_epochs=[]
stopflag=False

for epoch in range(1, num_epochs + 1):
    print(f"> Starting epoch {epoch}/{num_epochs}")
    
    train_epoch_loss = 0.0
    train_epoch_mae = 0.0
    
    t0 = time.time()
    cnnbc_model.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader)):

        # load data - cuda
        data = train_data["cfs"].to(device = device, dtype=torch.float)
        data = data.squeeze().unsqueeze(dim=1)
        labels = train_data["gpcp"].to(device = device, dtype=torch.float)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward 
        with torch.cuda.amp.autocast():
            output = cnnbc_model(data)
            output = output.squeeze()
            labels = labels.squeeze()
            train_batch_loss = mse_loss(output, labels)#*0 + vgg_loss(output, hr_data)

        # backward
        train_batch_loss.backward()
        optimizer.step()
        
        # batch statistics
        train_epoch_loss += output.shape[0] * train_batch_loss.item()
        
        train_batch_mae = mae_loss(output, labels)
        train_epoch_mae += output.shape[0] * train_batch_mae.item()
        
    Train_Loss = train_epoch_loss / len(train_dataset)
    print("Train Loss: ", Train_Loss)
    Train_MAE = train_epoch_mae / len(train_dataset)
    print("Train MAE: ", Train_MAE)
    Epoch_Time = time.time() - t0
    print(f'>>> Epoch {epoch} finished in {Epoch_Time:.2f} seconds')
    
    
    # Validation
    print(f"> Starting validation after epoch {epoch}")
    
    val_epoch_loss = 0.0
    val_epoch_mae = 0.0
    
    cnnbc_model.eval()
    

    with torch.inference_mode():
        
        for batch_ix, val_data in enumerate(tqdm(val_loader)):
            
            # load data - cuda
            data = val_data["cfs"].to(device = device, dtype=torch.float)
            data = data.squeeze().unsqueeze(dim=1)
            labels = val_data["gpcp"].to(device = device, dtype=torch.float)

            # forward
            output = cnnbc_model(data)
            output = output.squeeze()
            labels = labels.squeeze()
            val_batch_loss = mse_loss(output, labels) #* (10**6)

            # batch statistics
            val_epoch_loss += output.shape[0] * val_batch_loss.item()

            val_batch_mae = mae_loss(output, labels)
            val_epoch_mae += output.shape[0] * val_batch_mae.item()
            
        Val_Loss = val_epoch_loss / len(val_dataset)
        print("Val Loss: ", Val_Loss)
        Val_MAE = val_epoch_mae / len(val_dataset)
        print("Val MAE: ", Val_MAE)
        
        # save model
        if Val_Loss < best_loss:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(cnnbc_model.state_dict(), save_path + f"Val-Loss_{Val_Loss:.3f}.pth")
            print(f"Model saved on epoch: {epoch} with validation Loss: {Val_Loss:.5f}")
            best_loss = Val_Loss
        
        train_losses.append(Train_Loss)
        val_losses.append(Val_Loss)
        train_maes.append(Train_MAE)
        val_maes.append(Val_MAE)
        val_epochs.append(epoch)
        
        plotLosses(train_losses,val_losses,val_epochs,"MSE Loss",save_path)
        plotLosses(train_maes,val_maes,val_epochs,"MAE Loss",save_path)
        
        log_data={'epochs':val_epochs,
                    'train_losses':train_losses,
                    'val_losses':val_losses,
                    'train_maes':train_maes,
                    'val_maes':val_maes}
        log_data=pd.DataFrame(log_data)
        log_data.to_csv(save_path+'GenTrainLog.csv')
        
        #stopflag=early_stopper.early_stop_conv(Val_Loss,Train_Loss)
        stopflag=early_stopper.early_stop(Val_Loss)

        cnnbc_model.train()
        
    gc.collect()
    if stopflag:             
        break



now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Finished at: {now}")


print('Memory Usage:')
print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')

