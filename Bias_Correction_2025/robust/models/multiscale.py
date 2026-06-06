"""
Multi-Scale Feature Extractor for Bias Correction
==================================================

This module implements multi-scale feature extraction to capture 
meteorological patterns at different spatial scales:

- 33: Local precipitation patterns, convective cells
- 55: Mesoscale features, rain bands
- 77: Synoptic-scale features, frontal systems
- 99: Large-scale patterns, monsoon trough

The multi-scale approach is crucial for bias correction as NWP model
biases manifest differently at different spatial scales.

Author: IITM Pune - Government of India Project
Reference: Szegedy et al., "Going Deeper with Convolutions" (CVPR 2015) - Inception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .attention import SqueezeExcitation, CBAM


class MultiScaleConv(nn.Module):
    """
    Single multi-scale convolution layer.
    
    Applies convolutions with different kernel sizes in parallel
    and concatenates the results.
    
    Args:
        in_channels: Number of input channels
        out_channels_per_branch: Channels per kernel branch
        kernel_sizes: List of kernel sizes (default: [3, 5, 7, 9])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels_per_branch: int = 64,
        kernel_sizes: List[int] = [3, 5, 7, 9]
    ):
        super().__init__()
        
        self.branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_per_branch, 
                         kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels_per_branch),
                nn.GELU()
            )
            self.branches.append(branch)
            
        self.out_channels = out_channels_per_branch * len(kernel_sizes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Concatenated multi-scale features [B, C*num_branches, H, W]
        """
        features = [branch(x) for branch in self.branches]
        return torch.cat(features, dim=1)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Complete Multi-Scale Feature Extractor with attention fusion.
    
    Extracts features at 4 different spatial scales (33, 55, 77, 99)
    to capture meteorological patterns ranging from convective cells
    to large-scale monsoon patterns.
    
    Architecture:
        Input [B, in_channels, H, W]
             Branch 33: Conv  BN  GELU
             Branch 55: Conv  BN  GELU  
             Branch 77: Conv  BN  GELU
             Branch 99: Conv  BN  GELU
                 Concatenate
            [B, base_filters * 4, H, W]
                 Fusion Conv
            [B, out_channels, H, W]
                 CBAM Attention
            Output [B, out_channels, H, W]
    
    Args:
        in_channels: Number of input channels (e.g., 10 for multi-temporal data)
        base_filters: Base number of filters per branch
        out_channels: Final output channels
        use_attention: Whether to apply CBAM attention after fusion
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        base_filters: int = 64,
        out_channels: int = 256,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Multi-scale branches
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU()
        )
        
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU()
        )
        
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU()
        )
        
        self.branch_9x9 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU()
        )
        
        # Pooling branch for global context
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, base_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU()
        )
        
        # Fusion layer
        concat_channels = base_filters * 5  # 4 spatial branches + 1 global
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Optional attention
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
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
        Forward pass extracting multi-scale features.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Multi-scale features [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # Extract features at different scales
        f3 = self.branch_3x3(x)  # Local patterns
        f5 = self.branch_5x5(x)  # Mesoscale
        f7 = self.branch_7x7(x)  # Synoptic scale
        f9 = self.branch_9x9(x)  # Large scale
        
        # Global context
        f_global = self.branch_pool(x)
        f_global = F.interpolate(f_global, size=(H, W), mode='bilinear', align_corners=True)
        
        # Concatenate all scales
        fused = torch.cat([f3, f5, f7, f9, f_global], dim=1)
        
        # Fusion and attention
        out = self.fusion(fused)
        out = self.attention(out)
        
        return out


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for multi-scale context aggregation.
    
    Pools features at multiple scales (11, 22, 33, 66) to capture
    global and local context simultaneously. Particularly useful for
    capturing different precipitation intensities.
    
    Reference: Zhao et al., "Pyramid Scene Parsing Network" (CVPR 2017)
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        pool_sizes: List of pooling output sizes
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: List[int] = [1, 2, 3, 6]
    ):
        super().__init__()
        
        self.stages = nn.ModuleList()
        
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.GELU()
            ))
            
        # Fusion after concatenation with original
        total_channels = in_channels + (in_channels // len(pool_sizes)) * len(pool_sizes)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Pyramid-pooled features [B, out_channels, H, W]
        """
        H, W = x.shape[2:]
        
        # Pool at multiple scales and upsample back
        pyramid_features = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=True)
            pyramid_features.append(upsampled)
            
        # Concatenate and fuse
        concatenated = torch.cat(pyramid_features, dim=1)
        out = self.fusion(concatenated)
        
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    
    Captures multi-scale context using dilated convolutions with
    different dilation rates. Effective for dense prediction tasks.
    
    Reference: Chen et al., "DeepLab" series
    
    Args:
        in_channels: Input channels
        out_channels: Output channels per branch
        dilations: List of dilation rates
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilations: List[int] = [1, 6, 12, 18]
    ):
        super().__init__()
        
        modules = []
        
        # 11 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ))
        
        # Dilated convolutions
        for dilation in dilations[1:]:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                         dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ))
            
        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ))
        
        self.aspp = nn.ModuleList(modules)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(modules), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            ASPP features [B, out_channels, H, W]
        """
        H, W = x.shape[2:]
        
        features = []
        for module in self.aspp[:-1]:
            features.append(module(x))
            
        # Global pooling branch
        global_feat = self.aspp[-1](x)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=True)
        features.append(global_feat)
        
        # Concatenate and fuse
        concatenated = torch.cat(features, dim=1)
        out = self.fusion(concatenated)
        
        return out
