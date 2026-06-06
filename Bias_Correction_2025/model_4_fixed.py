import torch
from torch import nn 

class Residual_Block(nn.Module):

    def __init__(self, in_features, ng):
        super(Residual_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=9, stride=1, padding=4, groups=ng),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=ng),
            )
    
    def forward(self, x):
        return x + self.conv_block(x)

class ResCNNv4_Fixed(nn.Module):

    def __init__(self, in_channels=10, out_channels=1, n_residual_blocks=8):
        super(ResCNNv4_Fixed, self).__init__()

        nf = 40
        ng = 10
        
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, nf, kernel_size=9, stride=1, padding=4, groups=ng), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for i in range(n_residual_blocks):
            res_blocks.append(Residual_Block(nf, ng))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, groups=ng))
        
        # REMOVED: Problematic shuffling layers that were upscaling then downscaling
        
        # Final output layer - Changed from ReLU to Linear activation
        self.conv3 = nn.Sequential(nn.Conv2d(nf, out_channels, kernel_size=9, stride=1, padding=4))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        # REMOVED: shuffling layers
        out = self.conv3(out)
        return out
