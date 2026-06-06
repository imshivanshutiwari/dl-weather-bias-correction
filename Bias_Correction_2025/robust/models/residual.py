"""
Residual Blocks for Bias Correction Network
============================================

This module implements advanced residual block architectures for 
deep feature learning in meteorological bias correction.

Features:
- Pre-activation residual blocks (improved gradients)
- Dense connections option
- Squeeze-Excitation integration
- Proper initialization for stable training

Author: IITM Pune - Government of India Project
Reference: He et al., "Deep Residual Learning" (CVPR 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .attention import SqueezeExcitation


class ResidualBlock(nn.Module):
    """
    Pre-activation Residual Block with optional SE attention.
    
    Uses BatchNorm  ReLU  Conv ordering (pre-activation) which provides
    better gradient flow compared to original ResNet ordering.
    
    Architecture:
        Input x
             (identity)
             BN  GELU  Conv  BN  GELU  Conv  [SE]  output
        Output = x + output
    
    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size (default: 3)
        use_se: Whether to use Squeeze-Excitation attention
        se_reduction: Reduction ratio for SE block
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_se: bool = True,
        se_reduction: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
        )
        
        self.se = SqueezeExcitation(channels, se_reduction) if use_se else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        residual = self.block(x)
        residual = self.se(residual)
        return x + residual


class ResidualBlockV2(nn.Module):
    """
    Enhanced Residual Block with bottleneck architecture.
    
    Uses 1x1  3x3  1x1 convolutions for efficiency in deeper networks.
    Includes SE attention and optional expansion.
    
    Architecture:
        Input x [B, C, H, W]
             (identity or projection if channels change)
             BN  GELU  Conv1x1  BN  GELU  Conv3x3  BN  GELU  Conv1x1  SE
        Output = projection(x) + output
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        bottleneck_ratio: Ratio for bottleneck (default: 0.25)
        use_se: Whether to use SE attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: float = 0.25,
        use_se: bool = True
    ):
        super().__init__()
        
        hidden_channels = int(out_channels * bottleneck_ratio)
        
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
        )
        
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        
        # Skip connection projection if channels differ
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv2d(in_channels, out_channels, 1, bias=False)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.se(residual)
        return self.skip(x) + residual


class ResidualGroup(nn.Module):
    """
    Group of Residual Blocks with optional global skip connection.
    
    Stacks multiple residual blocks with a long skip connection from
    input to output, improving gradient flow in very deep networks.
    
    Architecture:
        Input x
             (identity)
             ResBlock  ResBlock  ...  ResBlock  Conv  output
        Output = x + output
    
    Args:
        channels: Number of channels
        num_blocks: Number of residual blocks in group
        kernel_size: Kernel size for convolutions
        use_se: Whether to use SE attention in blocks
    """
    
    def __init__(
        self,
        channels: int,
        num_blocks: int = 4,
        kernel_size: int = 3,
        use_se: bool = True
    ):
        super().__init__()
        
        blocks = [
            ResidualBlock(channels, kernel_size, use_se)
            for _ in range(num_blocks)
        ]
        
        self.blocks = nn.Sequential(*blocks)
        
        # Final convolution for the group
        self.conv = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with long skip connection.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        out = self.blocks(x)
        out = self.conv(out)
        return x + out


class DenseResidualBlock(nn.Module):
    """
    Dense Residual Block combining residual and dense connections.
    
    Each layer receives inputs from all preceding layers, providing
    maximum feature reuse while maintaining residual learning.
    
    Args:
        channels: Number of channels
        growth_rate: Number of channels added by each layer
        num_layers: Number of dense layers
    """
    
    def __init__(
        self,
        channels: int,
        growth_rate: int = 32,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)
                )
            )
        
        # Transition layer to reduce channels back
        final_channels = channels + num_layers * growth_rate
        self.transition = nn.Sequential(
            nn.BatchNorm2d(final_channels),
            nn.GELU(),
            nn.Conv2d(final_channels, channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dense connections.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        features = [x]
        
        for layer in self.layers:
            combined = torch.cat(features, dim=1)
            out = layer(combined)
            features.append(out)
        
        combined = torch.cat(features, dim=1)
        out = self.transition(combined)
        
        return x + out


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block with normalization and activation.
    
    Standard building block for encoder/decoder paths.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride (default: 1)
        padding: Padding (default: kernel_size // 2)
        bias: Whether to use bias
        norm: Normalization type ('batch', 'group', 'instance', 'none')
        activation: Activation type ('relu', 'gelu', 'leaky', 'none')
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        norm: str = 'batch',
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        ]
        
        # Normalization
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'group':
            layers.append(nn.GroupNorm(8, out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
            
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """
    Double Convolution Block for U-Net.
    
    Two consecutive convolutions with normalization and activation.
    Standard building block for U-Net encoder/decoder.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        mid_channels: Middle channels (default: same as out_channels)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
