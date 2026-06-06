"""
Attention Modules for Bias Correction Network
==============================================

This module implements state-of-the-art attention mechanisms for 
meteorological bias correction:

1. Squeeze-and-Excitation (SE) Block - Channel attention
2. Spatial Attention Module - Spatial focus
3. Attention Gate - For U-Net skip connections
4. Self-Attention (Transformer-style) - Global dependencies
5. CBAM - Convolutional Block Attention Module (combined)

Author: IITM Pune - Government of India Project
Reference: 
    - Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    - Oktay et al., "Attention U-Net" (MIDL 2018)
    - Woo et al., "CBAM" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    
    Adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels. This is crucial for 
    precipitation bias correction as different atmospheric variables
    (channels) have varying importance for different weather patterns.
    
    Architecture:
        Input [B, C, H, W] 
             Global Avg Pool  [B, C, 1, 1]
             FC  ReLU  FC  Sigmoid  [B, C, 1, 1]
             Scale input  [B, C, H, W]
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Attention-weighted tensor of shape [B, C, H, W]
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        squeezed = self.squeeze(x).view(batch_size, channels)
        
        # Excitation: FC layers with sigmoid
        excited = self.excitation(squeezed).view(batch_size, channels, 1, 1)
        
        # Scale: Multiply input by attention weights
        return x * excited.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Focuses on WHERE to pay attention by computing spatial attention maps.
    Uses both max-pooled and average-pooled features across channels.
    Important for identifying precipitation hotspots and frontal boundaries.
    
    Architecture:
        Input [B, C, H, W]
             Concat(AvgPool(dim=1), MaxPool(dim=1))  [B, 2, H, W]
             Conv 77  Sigmoid  [B, 1, H, W]
             Scale input  [B, C, H, W]
    
    Args:
        kernel_size: Size of convolution kernel (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Spatially-attended tensor of shape [B, C, H, W]
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.conv(combined)  # [B, 1, H, W]
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel attention (SE) and spatial attention sequentially.
    Provides comprehensive attention mechanism for bias correction.
    
    Architecture:
        Input  Channel Attention  Spatial Attention  Output
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention
        spatial_kernel_size: Kernel size for spatial attention
    """
    
    def __init__(
        self, 
        channels: int, 
        reduction: int = 16,
        spatial_kernel_size: int = 7
    ):
        super().__init__()
        
        self.channel_attention = SqueezeExcitation(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying channel then spatial attention.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Attention-enhanced tensor of shape [B, C, H, W]
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    
    Focuses on relevant features from encoder skip connections by using
    gating signals from the decoder. Essential for preserving fine-grained
    precipitation patterns while filtering out noise.
    
    Reference: Oktay et al., "Attention U-Net: Learning Where to Look 
               for the Pancreas" (MIDL 2018)
    
    Architecture:
        g (gating signal from decoder): [B, F_g, H, W]
        x (skip connection from encoder): [B, F_l, H, W]
        
        W_g(g) + W_x(x)  ReLU  psi  Sigmoid  attention map
        Output: x * attention_map
    
    Args:
        F_g: Number of channels in gating signal
        F_l: Number of channels in skip connection
        F_int: Number of intermediate channels
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            g: Gating signal from decoder [B, F_g, H_g, W_g]
            x: Skip connection from encoder [B, F_l, H_x, W_x]
            
        Returns:
            Attention-gated skip connection [B, F_l, H_x, W_x]
        """
        # Apply 1x1 convolutions
        g1 = self.W_g(g)  # [B, F_int, H_g, W_g]
        x1 = self.W_x(x)  # [B, F_int, H_x, W_x]
        
        # Upsample gating signal if sizes don't match
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        # Compute attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H_x, W_x]
        
        # Apply attention to skip connection
        return x * psi


class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (Transformer-style).
    
    Captures long-range spatial dependencies in meteorological fields.
    Essential for modeling teleconnections and large-scale weather patterns
    like the monsoon that span the entire Indian subcontinent.
    
    Uses efficient implementation with linear projections and 
    optional positional encoding.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head (default: channels // num_heads)
        dropout: Dropout rate
        use_positional_encoding: Whether to add positional encoding
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_positional_encoding = use_positional_encoding
        
        inner_dim = self.num_heads * self.head_dim
        
        # Query, Key, Value projections
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, kernel_size=1, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        
        # Positional encoding (learnable)
        if use_positional_encoding:
            # Will be initialized lazily based on input size
            self.pos_embedding = None
            
    def _get_positional_embedding(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate 2D sinusoidal positional embedding."""
        if self.pos_embedding is not None and self.pos_embedding.shape[2:] == (H, W):
            return self.pos_embedding.to(device)
            
        # Create 2D positional encoding
        y_embed = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        
        dim = self.channels // 4
        omega = 1.0 / (10000 ** (torch.arange(dim, device=device).float() / dim))
        
        # Compute sin/cos embeddings
        y_embed = y_embed.unsqueeze(-1) * omega
        x_embed = x_embed.unsqueeze(-1) * omega
        
        pos_embed = torch.cat([
            torch.sin(y_embed), torch.cos(y_embed),
            torch.sin(x_embed), torch.cos(x_embed)
        ], dim=-1)
        
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Pad or truncate to match channels
        if pos_embed.shape[1] < self.channels:
            padding = torch.zeros(1, self.channels - pos_embed.shape[1], H, W, device=device)
            pos_embed = torch.cat([pos_embed, padding], dim=1)
        elif pos_embed.shape[1] > self.channels:
            pos_embed = pos_embed[:, :self.channels]
            
        self.pos_embedding = pos_embed
        return pos_embed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Self-attended tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            pos_embed = self._get_positional_embedding(H, W, x.device)
            x_norm = x_norm + pos_embed
        
        # Compute Q, K, V
        qkv = self.to_qkv(x_norm)  # [B, 3*inner_dim, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, -1, H, W)  # [B, inner_dim, H, W]
        
        # Output projection
        out = self.to_out(out)
        
        # Residual connection
        return x + out


class MultiHeadAttention2D(nn.Module):
    """
    Efficient 2D Multi-Head Attention for dense prediction tasks.
    
    Optimized for spatial feature maps with memory-efficient computation.
    Uses depthwise separable convolutions for efficiency.
    
    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
        window_size: Size of local attention window (0 for global)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # Efficient QKV using depthwise separable convolutions
        self.qkv_dwconv = nn.Conv2d(
            channels, channels * 3, 
            kernel_size=3, padding=1, 
            groups=channels, bias=False
        )
        self.qkv_pwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=1, bias=False)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_drop = nn.Dropout(dropout)
        
        # Layer norm
        self.norm = nn.GroupNorm(8, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attended tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Efficient QKV computation
        qkv = self.qkv_pwconv(self.qkv_dwconv(x_norm))
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Output
        out = (v @ attn).reshape(B, C, H, W)
        out = self.proj_drop(self.proj(out))
        
        return x + out
