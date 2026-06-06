# Code for SRGAN: Generator + Discriminator (Adversarial Training)
import os
from config import CONFIG
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#RUN_NAME = "DISC_Val_BIG1"
#RUN_NAME = "DISC_24_noTop"
#RUN_NAME = "DISC_trial_24"
#RUN_NAME = "Seq_run3"
#RUN_NAME = "No_PreTrain"
#RUN_NAME = "NewVGG_runs"
#RUN_NAME = "Make_Dataset2"
RUN_NAME = "SRGAN_2025"

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

from data_loader3 import TropDataset
from models import Discriminator, Generator
from srgan_utils import PSNR, VGGLoss
from wandb_config import WANDB_CONFIG
from training_utils import EarlyStopper,plotLosses,plotPSNRs,plotGANlosses

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


# Datset and Dataloader
'''
train_dataset = TropDataset(data_path = CONFIG["TRAIN_PATH"], oro_path = CONFIG["ORO_PATH"])
val_dataset = TropDataset(data_path = CONFIG["VAL_PATH"], oro_path = CONFIG["ORO_PATH"])
'''
timeSet=[[pd.to_datetime(['1975-01-01','2011-12-31']).tolist(),pd.to_datetime(['2012-01-01','2016-12-31']).tolist()],
         [pd.to_datetime(['1975-01-01','2004-12-31']).tolist(),pd.to_datetime(['2017-01-01','2023-12-31']).tolist(),pd.to_datetime(['2005-01-01','2009-12-31']).tolist()],
         [pd.to_datetime(['1975-01-01','1999-12-31']).tolist(),pd.to_datetime(['2010-01-01','2023-12-31']).tolist(),pd.to_datetime(['2000-01-01','2004-12-31']).tolist()],
         [pd.to_datetime(['1975-01-01','1990-12-31']).tolist(),pd.to_datetime(['2003-01-01','2023-12-31']).tolist(),pd.to_datetime(['1991-01-01','1995-12-31']).tolist()],
         [pd.to_datetime(['1975-01-01','1983-12-31']).tolist(),pd.to_datetime(['1996-01-01','2023-12-31']).tolist(),pd.to_datetime(['1984-01-01','1988-12-31']).tolist()],
         [pd.to_datetime(['1975-01-01','1976-12-31']).tolist(),pd.to_datetime(['1989-01-01','2023-12-31']).tolist(),pd.to_datetime(['1977-01-01','1981-12-31']).tolist()],
         [pd.to_datetime(['1987-01-01','2023-12-31']).tolist(),pd.to_datetime(['1982-01-01','1986-12-31']).tolist()],
         ]
set_num = 0
train_dataset = TropDataset(data_path = CONFIG["COMB_PATH"], oro_path = CONFIG["ORO_PATH"], date_ranges=timeSet[set_num][:-1])
val_dataset = TropDataset(data_path = CONFIG["COMB_PATH"], oro_path = CONFIG["ORO_PATH"], date_ranges=timeSet[set_num][-1:])
#train_dataset = TropDataset(data_path = CONFIG["COMB_PATH"], oro_path = CONFIG["ORO_PATH"], isval=False)
#val_dataset = TropDataset(data_path = CONFIG["COMB_PATH"], oro_path = CONFIG["ORO_PATH"], isval=True)


train_loader = DataLoader(dataset=train_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = True, num_workers=8)
val_loader = DataLoader(dataset=val_dataset, batch_size= CONFIG["BATCH_SIZE"], shuffle = False, num_workers=8)



# Model - Generator

gen_load_path = "/home/users/bipink/DIAT/George_Jose/Downscaling/srgan_pytorch/checkpoints/SRGAN_2025/May-14-2025_1/Val-PSNR_45.322.pth"



print(f">> Loading GEN checkpoint from: {gen_load_path}")
generator = Generator()
#generator = Generator(in_channels=1)
generator.load_state_dict(torch.load(gen_load_path))
generator.to(device=device)

# Model - Discriminator
print(">> Initialising Discriminator")
discriminator = Discriminator(input_shape = (1, 132, 140))
#discriminator.load_state_dict(torch.load(disc_load_path))
discriminator.to(device=device)

psnr_metric = PSNR()
mae_loss = nn.MSELoss() #nn.L1Loss()
vgg_loss = VGGLoss()
bce = nn.BCEWithLogitsLoss()
#loss_coeff=[1e-3,1,1e-1] #order : [bce,vgg,mse]
loss_coeff=[5e-3,5,1e-1]

gen_optimizer = optim.Adam(generator.parameters(), lr = CONFIG["LEARNING_RATE"])#*100)
disc_optimizer = optim.Adam(discriminator.parameters(), lr = CONFIG["LEARNING_RATE"]/10)

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, patience=CONFIG["PATIENCE"], factor=CONFIG["FACTOR"], verbose=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(disc_optimizer, patience=CONFIG["PATIENCE"], factor=CONFIG["FACTOR"], verbose=True)
#scaler =  torch.cuda.amp.GradScaler()

num_epochs = CONFIG["NUM_EPOCHS"]
early_stopper = EarlyStopper(patience=CONFIG["PATIENCE"], min_delta=0.0)

#########################################################

todaysdate=date.today().strftime("%b-%d-%Y") + "_adv2"
# save path
save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, todaysdate, "")
#save_path = os.path.join(CONFIG["CHECKPOINT_PATH"], RUN_NAME, "Set"+str(set_num), todaysdate, "")
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save parameters
#remark=""
remark="with oro; Diff lrs x1,0.1;"
#remark='Diff lrs x1,0.1; '
params={'lr':CONFIG["LEARNING_RATE"],'batch_size':CONFIG["BATCH_SIZE"],'disc_lead':CONFIG["DISC_LEAD"],
        'eval_every':CONFIG["EVAL_EVERY"],'loss_coeff':loss_coeff,'checkpoint':gen_load_path,"remark":remark}
with open(save_path+"params.json", "w") as f1:
    json.dump(params,f1)


now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")                
print(f">>> Training Started at: {now}")
best_psnr = 0.0
Gen_Val_Loss=0.0
train_losses=[]
disc_losses=[]
val_losses=[]
train_psnrs=[]
val_psnrs=[]
val_epochs=[]
stopflag=False

for epoch in range(1, num_epochs + 1):
    print(f"> Starting epoch {epoch}/{num_epochs}")

    train_disc_epoch_loss = 0.0
    train_gen_epoch_loss = 0.0
    train_epoch_psnr = 0.0
    
    t0 = time.time()
    generator.train()
    discriminator.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader)):

        # load data - cuda
        train_lr_data = train_data["lr"].to(device = device, dtype=torch.float)
        train_hr_data = train_data["hr"].to(device = device, dtype=torch.float)

        # Train Discriminator:
        disc_optimizer.zero_grad()
        ## forward 
        train_gen_output = generator(train_lr_data)
        train_disc_real = discriminator(train_hr_data)
        train_disc_fake = discriminator(train_gen_output.detach())

        train_disc_loss_real = bce(train_disc_real, torch.ones_like(train_disc_real)) #- 0.1 * torch.rand_like(train_disc_real)
        train_disc_loss_real.backward()
        train_disc_loss_fake = bce(train_disc_fake, torch.zeros_like(train_disc_fake))
        train_disc_loss_fake.backward()
        
        ## backward
        train_disc_batch_loss = train_disc_loss_fake + train_disc_loss_real
        disc_optimizer.step()

        ## batch statistics
        train_disc_epoch_loss += train_disc_real.shape[0] * train_disc_batch_loss.item()
        
        train_batch_psnr = psnr_metric(train_gen_output.detach(), train_hr_data)
        train_epoch_psnr += train_gen_output.shape[0] * train_batch_psnr.item()

        # Train Generator:
        ## forward
        train_disc_fake = discriminator(train_gen_output)
        train_adversarial_loss = bce(train_disc_fake, torch.ones_like(train_disc_fake)) * loss_coeff[0]
        train_loss_for_vgg = vgg_loss(train_gen_output, train_hr_data.detach()) * loss_coeff[1]
        train_mae_loss = mae_loss(train_gen_output, train_hr_data.detach()) * loss_coeff[2]
        train_gen_batch_loss = train_adversarial_loss + train_loss_for_vgg + train_mae_loss

        ## backward
        if epoch > CONFIG["DISC_LEAD"]:
            gen_optimizer.zero_grad()
            train_gen_batch_loss.backward()
            gen_optimizer.step()

        ## batch statistics
        train_gen_epoch_loss += train_gen_output.shape[0] * train_gen_batch_loss.item()

    Disc_Train_Loss = train_disc_epoch_loss / len(train_dataset)
    print("Disc Loss: ", Disc_Train_Loss)
    Train_PSNR = train_epoch_psnr / len(train_dataset)
    print("Train PSNR: ", Train_PSNR)
    Gen_Train_Loss = train_gen_epoch_loss / len(train_dataset)
    print("Gen Loss: ", Gen_Train_Loss)
    Epoch_Time = time.time() - t0
    print(f'>>> Epoch {epoch} finished in {Epoch_Time:.2f} seconds')

    # Validation
    if epoch % CONFIG["EVAL_EVERY"] == 0:
        print(f"> Starting validation after epoch {epoch}")

        val_disc_epoch_loss = 0.0
        val_gen_epoch_loss = 0.0
        val_epoch_psnr = 0.0
        
        generator.eval()
        discriminator.eval()

        #with torch.no_grad():
        with torch.inference_mode():

            for batch_ix, val_data in enumerate(tqdm(val_loader)):

                # load data - cuda
                val_lr_data = val_data["lr"].to(device = device, dtype=torch.float)
                val_hr_data = val_data["hr"].to(device = device, dtype=torch.float)

                # Discriminator:
                ## forward
                val_gen_output = generator(val_lr_data)
                val_disc_real = discriminator(val_hr_data)
                val_disc_fake = discriminator(val_gen_output.detach())

                val_disc_loss_real = bce(val_disc_real, torch.ones_like(val_disc_real)) #- 0.1 * torch.rand_like(disc_real) 
                val_disc_loss_fake = bce(val_disc_fake, torch.zeros_like(val_disc_fake))
                val_disc_batch_loss = val_disc_loss_fake + val_disc_loss_real

                ## batch statistics
                val_disc_epoch_loss += val_disc_real.shape[0] * val_disc_batch_loss.item()     # Two loss values, but scaling from one loss value only ???

                val_batch_psnr = psnr_metric(val_gen_output.detach(), val_hr_data)
                val_epoch_psnr += val_gen_output.shape[0] * val_batch_psnr.item()

                # Generator:
                ## forward
                val_disc_fake = discriminator(val_gen_output.detach())
                val_adversarial_loss = bce(val_disc_fake, torch.ones_like(val_disc_fake)) * loss_coeff[0]
                val_loss_for_vgg = vgg_loss(val_gen_output.detach(), val_hr_data.detach()) * loss_coeff[1]
                val_mae_loss = mae_loss(val_gen_output.detach(), val_hr_data.detach()) * loss_coeff[2]
                val_gen_batch_loss = val_adversarial_loss + val_loss_for_vgg + val_mae_loss

                ## batch statistics
                val_gen_epoch_loss += val_gen_output.shape[0] * val_gen_batch_loss.item()

            Disc_Val_Loss = val_disc_epoch_loss / len(val_dataset)
            print("Disc Loss: ", Disc_Val_Loss)
            Val_PSNR = val_epoch_psnr / len(val_dataset)
            print("Val PSNR: ", Val_PSNR)
            Gen_Val_Loss = val_gen_epoch_loss / len(val_dataset)
            print("Gen Loss: ", Gen_Val_Loss)

            # save model
            if (Val_PSNR > best_psnr) and (epoch > CONFIG["DISC_LEAD"]):
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(generator.state_dict(), save_path + f"GEN_Val-PSNR_{Val_PSNR:.3f}.pth")
                torch.save(discriminator.state_dict(), save_path + f"DISC_Val-PSNR_{Val_PSNR:.3f}.pth")
                print(f"Model saved on epoch: {epoch} with validation PSNR: {Val_PSNR:.5f}")
                #best_psnr = Val_PSNR
                
            train_losses.append(Gen_Train_Loss)
            disc_losses.append(Disc_Train_Loss)
            val_losses.append(Gen_Val_Loss)
            train_psnrs.append(Train_PSNR)
            val_psnrs.append(Val_PSNR)
            val_epochs.append(epoch)
            
            plotLosses(train_losses,val_losses,val_epochs,save_path)
            plotGANlosses(train_losses,disc_losses,val_epochs,save_path)
            plotPSNRs(train_psnrs,val_psnrs,val_epochs,save_path)
            
            log_data={'epochs':val_epochs,
                        'train_losses':train_losses,
                        'disc_losses':disc_losses,
                        'val_losses':val_losses,
                        'train_psnrs':train_psnrs,
                        'val_psnrs':val_psnrs}
            log_data=pd.DataFrame(log_data)
            log_data.to_csv(save_path+'AdvTrainLog.csv')
            
            stopflag=early_stopper.early_stop(Gen_Val_Loss)
            
        generator.train()
        discriminator.train()
    
    gc.collect()    
    if stopflag:             
        break

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f">>> Training Finished at: {now}")

print('Memory Usage:')
print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')
print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)}/{round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB')