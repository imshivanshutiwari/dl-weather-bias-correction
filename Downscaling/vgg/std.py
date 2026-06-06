import os
import numpy as np 
from tqdm import tqdm
import wandb
from wand_config import WAND_CONFIG

load_path = "/home/mtechstudent/IISER_Pune/kaustubh/VGG19/VGG/DATA/Val/HR/"
save_path = "/home/mtechstudent/IISER_Pune/kaustubh/VGG19/VGG/DATA/Val/NEW_HR/"


Mean_LR = 4.7381673
Mean_HR = 3.1606863
Std_LR = 12.848967
Std_HR = 10.908616

for i in tqdm(os.listdir(load_path)):
    x = np.load(load_path + i)
    new_x = (x - Mean_HR) / Std_HR
    np.save(save_path + i, new_x)