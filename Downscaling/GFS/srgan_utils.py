from torch._C import device
import torch.nn as nn
import numpy as np
import torch

#vgg_path = "/home/mtechstudent6/kaustubh/srgan_pytorch/Val-Acc_99.795.pth"
#vgg_path = "Val-Acc_99.795.pth"
#vgg_path = "/home/hpcs_rnd/George_DIAT/Downscaling/VGG/checkpoints/Retrain_1975to2023/Nov-14-2024/Val-Acc_98.601.pth"
vgg_path = "/home/users/bipink/DIAT/George_Jose/Downscaling/GFS_4x/Val-Acc_98.844.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

class PSNR:
    """Peak Signal to Noise Ratio"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        #return 20 * torch.log10(255.0 / torch.sqrt(mse))
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg =torch.load(vgg_path).conv_layers[:51].eval().to(device=device)
        self.loss = nn.MSELoss() #nn.L1Loss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        with torch.inference_mode():
            vgg_input_features = self.vgg(input)
            vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)