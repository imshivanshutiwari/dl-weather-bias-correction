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

class ResCNNv4_ProperlyFixed(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, n_residual_blocks=8):
        super(ResCNNv4_ProperlyFixed, self).__init__()
        
        nf = 60  # Must be divisible by groups
        ng = 10  # Must be divisible by in_channels (10)
        
        # Input processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=9, stride=1, padding=4, groups=ng), 
            nn.PReLU()
        )
        
        # Residual blocks
        res_blocks = []
        for i in range(n_residual_blocks):
            res_blocks.append(Residual_Block(nf, ng))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Feature refinement
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, groups=ng),
            nn.PReLU()
        )
        
        # Output layer - CRITICAL FIX: No activation function for regression
        self.conv3 = nn.Conv2d(nf, out_channels, kernel_size=9, stride=1, padding=4)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input processing
        out1 = self.conv1(x)
        
        # Residual learning
        out = self.res_blocks(out1)
        
        # Feature refinement
        out2 = self.conv2(out)
        out = torch.add(out1, out2)  # Skip connection
        
        # Output - NO ACTIVATION for regression
        out = self.conv3(out)
        
        return out
