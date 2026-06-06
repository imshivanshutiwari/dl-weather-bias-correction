# Code for SRGAN: Generator (MSE Training)
import os
import sys
from config import CONFIG
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#RUN_NAME = "SRGAN_with2023data_noTop"
#RUN_NAME = "SRGAN_with2023data"
#RUN_NAME = "SRGAN_Val_new"
#RUN_NAME = "Seq_run2"
#RUN_NAME = "Seq_run3"
#RUN_NAME = "NewVGG_runs"
#RUN_NAME = "Make_Dataset"
#RUN_NAME = "SRGAN_GFS_2"
RUN_NAME = "Train1"

import time
from datetime import date, datetime
import pandas as pd

from tqdm import tqdm
#import wandb
import json
import gc

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_loader3 import GFSdataset
from models import Generator
from srgan_utils import PSNR, VGGLoss
from training_utils import EarlyStopper,plotLosses,plotPSNRs

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

#os.environ["WANDB_API_KEY"] = WANDB_CONFIG["api_key"]


# Datset and Dataloader
'''split_ratio = [0.8,0.1,0.1]
sample_num=4880
split_index = [int(split_ratio[0]*sample_num), int(sum(split_ratio[:2])*sample_num), int(sum(split_ratio[:])*sample_num)]'''

index_sets =   [[[0,1,2,3,4,5,6],[7]],
                [[0,1,2,3,4,8,9],[5]],
                [[0,1,2,6,7,8,9],[3]],
                [[0,4,5,6,7,8,9],[1]],
                [[3,4,5,6,7,8,9],[2]]]

set_num=4
train_index=index_sets[set_num][0]
val_index=index_sets[set_num][1]

train_dataset = GFSdataset(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"],idxs=train_index)
val_dataset = GFSdataset(data_path=CONFIG["DATA_PATH"], label_path=CONFIG["LABEL_PATH"],idxs=val_index)

train_loader = DataLoader(dataset=train_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = True, num_workers=8)
val_loader = DataLoader(dataset=val_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = False, num_workers=8)



# Model
model_path=""
#model_path="/home/hpcs_rnd/George_DIAT/Downscaling/srgan_pytorch/checkpoints/Make_Dataset2/Set6/May-12-2025_adv1/GEN_Val-PSNR_42.403.pth"

generator = Generator(in_channels=1)
if len(model_path) != 0:
    generator.load_state_dict(torch.load(model_path))
generator = nn.DataParallel(generator)
generator.to(device=device)


mse_loss = nn.MSELoss() #nn.L1Loss() 
#vgg_loss = VGGLoss()
psnr_metric = PSNR()
optimizer = optim.Adam(generator.parameters(), lr = CONFIG["LEARNING_RATE"])
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG["PATIENCE"], factor=CONFIG["FACTOR"], verbose=True)
scaler =  torch.cuda.amp.GradScaler()

num_epochs = CONFIG["NUM_EPOCHS"]

#early_stopper = EarlyStopper(patience=5, min_delta=0.0)
early_stopper = EarlyStopper(patience=CONFIG["PATIENCE"], min_delta=0.0)

###############################################################

todaysdate=date.today().strftime("%b-%d-%Y") + "_1"
# save path
#save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, todaysdate, "")
save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, "Set"+str(set_num), todaysdate, "")
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save parameters
params={'lr':CONFIG["LEARNING_RATE"],'batch_size':CONFIG["BATCH_SIZE"],'disc_lead':CONFIG["DISC_LEAD"],'eval_every':CONFIG["EVAL_EVERY"],
        'checkpoint':model_path,'remark':'avgpool & maxpooled LR data'}
with open(save_path+"params.json", "w") as f1:
    json.dump(params,f1)

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Started at: {now}")
best_psnr = 0.0
Val_Loss=0.0
train_losses=[]
val_losses=[]
train_psnrs=[]
val_psnrs=[]
val_epochs=[]
stopflag=False
for epoch in range(1, num_epochs + 1):
    print(f"> Starting epoch {epoch}/{num_epochs}")

    train_epoch_loss = 0.0
    train_epoch_psnr = 0.0
    
    t0 = time.time()
    generator.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader)):

        # load data - cuda
        lr_data = train_data["lr"].to(device = device, dtype=torch.float)
        hr_data = train_data["hr"].to(device = device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward 
        with torch.cuda.amp.autocast():
            output = generator(lr_data)
            train_batch_loss = mse_loss(output[:,:,1:-2,1:-2], hr_data)#*0 + vgg_loss(output, hr_data)

        # backward
        scaler.scale(train_batch_loss).backward()
        scaler.step(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.update()

        # batch statistics
        train_epoch_loss += output.shape[0] * train_batch_loss.item()

        train_batch_psnr = psnr_metric(output[:,:,1:-2,1:-2].detach(), hr_data)
        train_epoch_psnr += output.shape[0] * train_batch_psnr.item()

    Train_Loss = train_epoch_loss / len(train_dataset)
    print("Train Loss: ", Train_Loss)
    Train_PSNR = train_epoch_psnr / len(train_dataset)
    print("Train PSNR: ", Train_PSNR)
    Epoch_Time = time.time() - t0
    print(f'>>> Epoch {epoch} finished in {Epoch_Time:.2f} seconds')
    

    # Validation
    if epoch % CONFIG["EVAL_EVERY"] == 0:
        print(f"> Starting validation after epoch {epoch}")
        
        val_epoch_loss = 0.0
        val_epoch_psnr = 0.0
        
        generator.eval()

        #with torch.no_grad():
        with torch.inference_mode():

            for batch_ix, val_data in enumerate(tqdm(val_loader)):

                # load data - cuda
                lr_data = val_data["lr"].to(device = device, dtype=torch.float)
                hr_data = val_data["hr"].to(device = device, dtype=torch.float)

                # forward
                output = generator(lr_data)
                val_batch_loss = mse_loss(output[:,:,1:-2,1:-2], hr_data) #* (10**6)

                # batch statistics
                val_epoch_loss += output.shape[0] * val_batch_loss.item()

                val_batch_psnr = psnr_metric(output[:,:,1:-2,1:-2].detach(), hr_data)
                val_epoch_psnr += output.shape[0] * val_batch_psnr.item() 

            Val_Loss = val_epoch_loss / len(val_dataset)
            #scheduler.step(Val_Loss)
            print("Val Loss: ", Val_Loss)
            Val_PSNR = val_epoch_psnr / len(val_dataset)
            print("Val PSNR: ", Val_PSNR)            
            
            
            # save model
            if Val_PSNR > best_psnr:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(generator.module.state_dict(), save_path + f"Val-PSNR_{Val_PSNR:.3f}.pth")
                print(f"Model saved on epoch: {epoch} with validation PSNR: {Val_PSNR:.5f}")
                best_psnr = Val_PSNR
            
            train_losses.append(Train_Loss)
            val_losses.append(Val_Loss)
            train_psnrs.append(Train_PSNR)
            val_psnrs.append(Val_PSNR)
            val_epochs.append(epoch)
            
            plotLosses(train_losses,val_losses,val_epochs,save_path)
            plotPSNRs(train_psnrs,val_psnrs,val_epochs,save_path)
            
            log_data={'epochs':val_epochs,
                        'train_losses':train_losses,
                        'val_losses':val_losses,
                        'train_psnrs':train_psnrs,
                        'val_psnrs':val_psnrs}
            log_data=pd.DataFrame(log_data)
            log_data.to_csv(save_path+'GenTrainLog.csv')
            
            #stopflag=early_stopper.early_stop_conv(Val_Loss,Train_Loss)
            stopflag=early_stopper.early_stop(Val_Loss)

        generator.train()
        
    gc.collect()
    if stopflag:             
        break



now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Finished at: {now}")


print('Memory Usage:')
print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')