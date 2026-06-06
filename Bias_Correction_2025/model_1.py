import torch
from torch import nn 

## VGG

class Residual_Block(nn.Module):

    def __init__(self, in_features):
        super(Residual_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),)
    
    def forward(self, x):
        return x + self.conv_block(x)

class ResCNNv1(nn.Module):

    def __init__(self, in_channels=10, out_channels=1, n_residual_blocks=8):
        super(ResCNNv1, self).__init__()

        nf =64
        
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, nf, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(Residual_Block(nf))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(nf, 0.8))

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(nf, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        return out
