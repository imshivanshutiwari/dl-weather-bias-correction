import torch
from torch import nn 

## Enhanced SRGAN for 25x Downscaling (0.625° to 0.025°)

class Residual_Block(nn.Module):

    def __init__(self, in_features):
        super(Residual_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),)
    
    def forward(self, x):
        return x + self.conv_block(x)

class Generator_25x(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, n_residual_blocks=8):
        super(Generator_25x, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(Residual_Block(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Enhanced upsampling layers for 25x improvement (5 layers = 2^5 = 32x)
        # 0.625° → 0.3125° → 0.156° → 0.078° → 0.039° → 0.025°
        upsampling = []
        for out_features in range(5):  # 5 layers for 32x upscaling (covers 25x target)
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator_25x(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_25x, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        self.body = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(18432, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),)

    def forward(self, x):
        x = self.body(x)
        return self.classifier(x)

# Alternative: Progressive Upsampling Generator
class ProgressiveGenerator_25x(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, n_residual_blocks=8):
        super(ProgressiveGenerator_25x, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(Residual_Block(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Progressive upsampling stages
        # Stage 1: 0.625° → 0.3125° (2x)
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )
        
        # Stage 2: 0.3125° → 0.156° (2x)
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )
        
        # Stage 3: 0.156° → 0.078° (2x)
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )
        
        # Stage 4: 0.078° → 0.039° (2x)
        self.upsample_4 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )
        
        # Stage 5: 0.039° → 0.025° (1.56x - final adjustment)
        self.upsample_5 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        
        # Progressive upsampling
        out = self.upsample_1(out)  # 2x
        out = self.upsample_2(out)  # 4x
        out = self.upsample_3(out)  # 8x
        out = self.upsample_4(out)  # 16x
        out = self.upsample_5(out)  # 32x (covers 25x target)
        
        out = self.conv3(out)
        return out
