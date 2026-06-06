#Code for CNNBC training
import os
import sys
from config import CONFIG
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"
#RUN_NAME = "ResCNNv4_2025"
RUN_NAME = "ResCNNv4_2025_part2"

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

from dataset2 import GFSdataset
#from model_cnnbc1 import CNNBC
from model_4 import ResCNNv4
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
    

# Dataset and dataloader
split_ratio = [0.7,0.15,0.15]  #70% traning 15% validation 15% testing
split_index = [int(split_ratio[0]*4880), int(sum(split_ratio[:2])*4880), int(sum(split_ratio[:])*4880)]

train_dataset = GFSdataset(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"],idxs=(0,split_index[0]))
val_dataset = GFSdataset(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"],idxs=(split_index[0],split_index[1]))

train_loader = DataLoader(dataset=train_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = True, num_workers=8)
val_loader = DataLoader(dataset=val_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = False, num_workers=8)


# Model instance
#model = CNNBC()
model = ResCNNv4()
model = nn.DataParallel(model)#,device_ids = [0,1,2])
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
        'remark':'IMDAA 3h as ground truth, No Normalization, no BN'}
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
    model.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader)):

        # load data - cuda
        data = train_data["gfs"].to(device = device, dtype=torch.float)
        #data = data.squeeze().unsqueeze(dim=1)
        labels = train_data["imdaa"].to(device = device, dtype=torch.float)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward 
        with torch.cuda.amp.autocast():
            output = model(data)
            output = output.squeeze()
            labels = labels.squeeze()
            train_batch_loss = mse_loss(output, labels) #+ 1/psnr_metric(output, labels)

        # backward
        train_batch_loss.backward()
        optimizer.step()
        
        # batch statistics
        train_epoch_loss += output.shape[0] * train_batch_loss.item()
        
        train_batch_mae = mae_loss(output.detach(), labels)
        train_epoch_mae += output.shape[0] * train_batch_mae.item()
        
        train_batch_psnr = psnr_metric(output.detach(), labels)
        train_epoch_psnr += output.shape[0] * train_batch_psnr.item()
        
    Train_Loss = train_epoch_loss / len(train_dataset)
    print("Train Loss: ", Train_Loss)
    Train_PSNR = train_epoch_psnr / len(train_dataset)
    print("Train PSNR: ", Train_PSNR)
    Train_MAE = train_epoch_mae / len(train_dataset)
    print("Train MAE: ", Train_MAE)
    Epoch_Time = time.time() - t0
    print(f'>>> Epoch {epoch} finished in {Epoch_Time:.2f} seconds')
    
    
    # Validation
    if epoch % CONFIG["EVAL_EVERY"] != 0:
        continue
    print(f"> Starting validation after epoch {epoch}")
    
    val_epoch_loss = 0.0
    val_epoch_psnr = 0.0
    val_epoch_mae = 0.0
    
    model.eval()
    

    with torch.inference_mode():
        
        for batch_ix, val_data in enumerate(tqdm(val_loader)):
            
            # load data - cuda
            data = val_data["gfs"].to(device = device, dtype=torch.float)
            #data = data.squeeze().unsqueeze(dim=1)
            labels = val_data["imdaa"].to(device = device, dtype=torch.float)

            # forward
            output = model(data)
            output = output.squeeze()
            labels = labels.squeeze()
            val_batch_loss = mse_loss(output, labels) #+ 1/psnr_metric(output, labels)

            # batch statistics
            val_epoch_loss += output.shape[0] * val_batch_loss.item()

            val_batch_mae = mae_loss(output, labels)
            val_epoch_mae += output.shape[0] * val_batch_mae.item()
            
            val_batch_psnr = psnr_metric(output.detach(), labels)
            val_epoch_psnr += output.shape[0] * val_batch_psnr.item()
            
        Val_Loss = val_epoch_loss / len(val_dataset)
        print("Val Loss: ", Val_Loss)
        Val_PSNR = val_epoch_psnr / len(val_dataset)
        print("Val PSNR: ", Val_PSNR)
        Val_MAE = val_epoch_mae / len(val_dataset)
        print("Val MAE: ", Val_MAE)
        
        # save model
        if Val_PSNR > best_psnr:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.module.state_dict(), save_path + f"Val-PSNR_{Val_PSNR:.3f}.pth")
            print(f"Model saved on epoch: {epoch} with validation PSNR: {Val_PSNR:.5f}")
            best_psnr = Val_PSNR
        
        train_losses.append(Train_Loss)
        val_losses.append(Val_Loss)
        train_psnrs.append(Train_PSNR)
        val_psnrs.append(Val_PSNR)
        train_maes.append(Train_MAE)
        val_maes.append(Val_MAE)
        val_epochs.append(epoch)
        
        plotLosses(train_losses,val_losses,val_epochs,"MSE Loss",save_path)
        plotLosses(train_maes,val_maes,val_epochs,"MAE Loss",save_path)
        plotPSNRs(train_psnrs,val_psnrs,val_epochs,save_path)
        
        log_data={'epochs':val_epochs,
                    'train_losses':train_losses,
                    'val_losses':val_losses,
                    'train_psnrs':train_psnrs,
                    'val_psnrs':val_psnrs,
                    'train_maes':train_maes,
                    'val_maes':val_maes}
        log_data=pd.DataFrame(log_data)
        log_data.to_csv(save_path+'TrainLog.csv')
        
        #stopflag=early_stopper.early_stop_conv(Val_Loss,Train_Loss)
        stopflag=early_stopper.early_stop(Val_Loss)

        model.train()
        
    gc.collect()
    if stopflag:             
        break



now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Finished at: {now}")


print('Memory Usage:')
print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')

