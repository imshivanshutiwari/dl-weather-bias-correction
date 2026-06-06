"""
SSIM Loss for Structural Similarity
====================================

Structural Similarity Index (SSIM) measures perceptual similarity
between images. For precipitation, this preserves:
- Rainfall patterns and gradients
- Spatial structure of weather systems
- Local contrast (important for extremes)

Author: IITM Pune - Government of India Project
Reference: Wang et al., "Image Quality Assessment" (IEEE TIP 2004)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


def gaussian_kernel(
    window_size: int, 
    sigma: float, 
    channels: int = 1, 
    device: torch.device = None
) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.
    
    Args:
        window_size: Size of the kernel (must be odd)
        sigma: Standard deviation of Gaussian
        channels: Number of channels
        device: Torch device
        
    Returns:
        Gaussian kernel [channels, 1, window_size, window_size]
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    # 2D kernel
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)  # [H, W]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # [C, 1, H, W]
    
    return kernel_2d


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    
    SSIM combines three components:
    - Luminance: Comparison of mean intensity
    - Contrast: Comparison of variance
    - Structure: Comparison of covariance/correlation
    
    Loss = 1 - SSIM (to minimize)
    
    Args:
        window_size: Size of sliding window (default: 11)
        sigma: Gaussian sigma (default: 1.5)
        size_average: Whether to average over batch
        data_range: Range of input data (default: 1.0 for normalized)
        channel: Number of channels
        K1: Stability constant for luminance
        K2: Stability constant for contrast
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        size_average: bool = True,
        data_range: float = 1.0,
        channel: int = 1,
        K1: float = 0.01,
        K2: float = 0.03
    ):
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.data_range = data_range
        self.channel = channel
        
        # Stability constants
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        
        # Create Gaussian window (will be registered as buffer)
        self.register_buffer('window', self._create_window(window_size, sigma, channel))
        
    def _create_window(
        self, 
        window_size: int, 
        sigma: float, 
        channel: int
    ) -> torch.Tensor:
        """Create Gaussian window."""
        return gaussian_kernel(window_size, sigma, channel)
    
    def _ssim(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor,
        window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SSIM between two images.
        
        Returns:
            ssim_map: Per-pixel SSIM values
            cs_map: Contrast-structure map (for MS-SSIM)
        """
        channel = img1.size(1)
        padding = self.window_size // 2
        
        # Compute means using Gaussian filter
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
        
        # Ensure non-negative variances
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        
        # Compute SSIM
        ssim_numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ssim_denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = ssim_numerator / ssim_denominator
        
        # Contrast-structure component (for MS-SSIM)
        cs_map = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        
        return ssim_map, cs_map
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            SSIM loss (1 - SSIM)
        """
        # Ensure window is on correct device
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
            
        # Handle channel mismatch
        if pred.size(1) != self.channel:
            self.window = gaussian_kernel(
                self.window_size, self.sigma, pred.size(1), pred.device
            )
            self.channel = pred.size(1)
            
        ssim_map, _ = self._ssim(pred, target, self.window)
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=[1, 2, 3])


class MultiScaleSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM Loss.
    
    Computes SSIM at multiple scales by iteratively downsampling.
    Better captures structure at both fine and coarse scales.
    
    Reference: Wang et al., "Multiscale Structural Similarity" (ACSSC 2003)
    
    Args:
        window_size: Size of sliding window
        sigma: Gaussian sigma
        data_range: Range of input data
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights: list = None
    ):
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = weights
        self.num_scales = len(weights)
        
        self.ssim = SSIMLoss(window_size, sigma, True, data_range)
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MS-SSIM loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            MS-SSIM loss
        """
        device = pred.device
        channel = pred.size(1)
        
        mcs_values = []
        
        window = gaussian_kernel(self.window_size, self.sigma, channel, device)
        
        for i in range(self.num_scales):
            ssim_map, cs_map = self.ssim._ssim(pred, target, window)
            
            if i < self.num_scales - 1:
                # Store contrast-structure component
                mcs_values.append(cs_map.mean())
                
                # Downsample for next scale
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                
                # Resize window if needed
                if min(pred.shape[2:]) < self.window_size:
                    break
                    
            else:
                # Final scale: use full SSIM
                mcs_values.append(ssim_map.mean())
                
        # Compute weighted product
        ms_ssim = 1.0
        for i, mcs in enumerate(mcs_values):
            ms_ssim *= torch.pow(mcs, self.weights[i])
            
        return 1 - ms_ssim
