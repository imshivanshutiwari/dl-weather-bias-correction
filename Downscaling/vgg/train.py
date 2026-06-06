import os
import time
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import json


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import VGG_TropDataset
from config import CONFIG 
from wandb_config import WANDB_CONFIG
from VGG import VGG19
from training_utils import EarlyStopper,plotLosses,plotAccuracies

os.environ["CUDA_VISIBLE_DEVICES"]= "4"
RUN_NAME = "Retrain_1975to2023"

# setting device on GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

#Additional Info when using cuda
if device == 'cuda':
    print("GPU : ", torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
    print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')

'''
os.environ["WANDB_API_KEY"] = WANDB_CONFIG["api_key"]
status = wandb.login()

if status == True:
  wandb.init(project= WANDB_CONFIG["project_name"],config=CONFIG)
  wandb.run.name = WANDB_CONFIG["run_name"]
  print("wandb login successfully")'''

train_dataset = VGG_TropDataset(root_dir = CONFIG["TRAIN_ROOT_DIR"], transform = transforms.ToTensor())
val_dataset = VGG_TropDataset(root_dir = CONFIG["VAL_ROOT_DIR"], transform=transforms.ToTensor()) 

train_loader = DataLoader(dataset=train_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle=True)

# Model
model = VGG19(num_classes=CONFIG["NUM_CLASSES"])
model.to(device = device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = CONFIG["LEARNING_RATE"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG["PATIENCE"], factor=CONFIG["FACTOR"], verbose=True)

num_epochs = CONFIG["NUM_EPOCHS"]

early_stopper = EarlyStopper(patience=3, min_delta=0.0)

###############################################################

todaysdate=date.today().strftime("%b-%d-%Y") #+ "_2"
# save path
save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, todaysdate, "")
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save parameters
params={'lr':CONFIG["LEARNING_RATE"],'batch_size':CONFIG["BATCH_SIZE"],'eval_every':CONFIG["EVAL_EVERY"]}
with open(save_path+"params.json", "w") as f1:
    json.dump(params,f1)

val_accuracy_log = [0]
train_accuracy_log = []
train_losses=[]
val_losses=[]
val_epochs=[]
stopflag=False

for epoch in range(1, num_epochs+1):
    print(f"Starting Epoch: {epoch}/{num_epochs}")

    train_epoch_loss = 0.0
    total = 0
    correct = 0
    t0 = time.time()
    
    for batch_idx, train_data in enumerate(tqdm(train_loader)):
        
        # load data - cuda
        data = train_data["data"].to(device = device, dtype=torch.float)
        label = train_data["label"].to(device = device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        output = model(data)
        train_batch_loss = criterion(output, label)

        # backward
        train_batch_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # gradient descent - optimize
        optimizer.step()

        # batch statistics
        train_epoch_loss += output.shape[0] * train_batch_loss.item()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    Train_Loss = train_epoch_loss / len(train_dataset)
    print("Train Loss: ", Train_Loss)
    Train_Accuracy = 100 * correct / total
    print("Train Accuracy: ", Train_Accuracy)
    Epoch_Time = time.time() - t0
    print(f'>>> Epoch {epoch} finished in {Epoch_Time:.2f} seconds')
    #wandb.log({"Train Accuracy" : Train_Accuracy, "Train Loss" : Train_Loss, "Epoch Time" : Epoch_Time})

    # Validation
    if epoch % CONFIG["EVAL_EVERY"] == 0:

        num_correct = 0
        num_samples = 0
        val_epoch_loss = 0

        model.eval()

        with torch.no_grad():

            for batch_idx, val_data in enumerate(tqdm(val_loader)):

                data = val_data["data"].to(device = device, dtype=torch.float)
                label = val_data["label"].to(device = device)

                output = model(data)
                val_batch_loss = criterion(output, label)

                val_epoch_loss += output.shape[0] * val_batch_loss.item()
                _, predicted = torch.max(output.data, 1)

                num_correct += (predicted == label).sum().item()
                num_samples += label.size(0)
            
            Val_Loss = val_epoch_loss / len(val_dataset)
            print("Val Loss: ", Val_Loss)
            Val_Accuracy = 100 * num_correct / num_samples
            print("Val Accuracy: ", Val_Accuracy)
            #wandb.log({"Val Accuracy" : Val_Accuracy, "Val Loss" : Val_Loss})
            
            # save best model
            if Val_Accuracy > max(val_accuracy_log):
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(model, save_path + f"Val-Acc_{Val_Accuracy:.3f}.pth")
                print(f"Model saved on epoch: {epoch} with validation accuracy: {Val_Accuracy:.2f}")
            
            train_accuracy_log.append(Train_Accuracy)
            val_accuracy_log.append(Val_Accuracy)
            train_losses.append(Train_Loss)
            val_losses.append(Val_Loss)
            val_epochs.append(epoch)
            
            plotLosses(train_losses,val_losses,val_epochs,save_path)
            plotAccuracies(train_accuracy_log,val_accuracy_log[1:],val_epochs,save_path)
            
            log_data={'epochs':val_epochs,
                        'train_losses':train_losses,
                        'val_losses':val_losses,
                        'train_accuracy':train_accuracy_log,
                        'val_accuracy':val_accuracy_log[1:]}
            log_data=pd.DataFrame(log_data)
            log_data.to_csv(save_path+'vggTrainLog.csv')
            
            stopflag=early_stopper.early_stop_conv(Val_Loss,Train_Loss)
            
        model.train()
        
    if stopflag:
        break

print("Finished Training ...")
#wandb.finish()

print('Memory Usage:')
print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')



