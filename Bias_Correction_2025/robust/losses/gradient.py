"""
Gradient Loss for Edge Preservation
====================================

Gradient loss penalizes differences in spatial gradients between
prediction and target. This preserves:
- Sharp precipitation boundaries (fronts)
- Rain band edges
- Topographic precipitation gradients

Author: IITM Pune - Government of India Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GradientLoss(nn.Module):
    """
    Gradient Loss for preserving spatial gradients.
    
    Computes the L1 or L2 difference between gradients of 
    prediction and target in both x and y directions.
    
    Architecture:
        pred, target  Sobel filter  gradient_x, gradient_y
        loss = |pred_grad - target_grad|
    
    Args:
        loss_type: 'l1' or 'l2' (default: 'l1')
        edge_weight: Weight for edge pixels (default: 1.0)
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        edge_weight: float = 1.0
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.edge_weight = edge_weight
        
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def _compute_gradients(
        self, 
        x: torch.Tensor
    ) -> tuple:
        """Compute spatial gradients using Sobel filter."""
        
        # Ensure kernels are on correct device
        if self.sobel_x.device != x.device:
            self.sobel_x = self.sobel_x.to(x.device)
            self.sobel_y = self.sobel_y.to(x.device)
        
        channels = x.size(1)
        
        # Expand kernels for multi-channel input
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)
        
        # Compute gradients
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
        
        return grad_x, grad_y
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Gradient loss
        """
        # Compute gradients
        pred_grad_x, pred_grad_y = self._compute_gradients(pred)
        target_grad_x, target_grad_y = self._compute_gradients(target)
        
        # Compute difference
        if self.loss_type == 'l1':
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        else:  # l2
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)
            
        return (loss_x + loss_y) * self.edge_weight


class LaplacianLoss(nn.Module):
    """
    Laplacian Loss for preserving second-order derivatives.
    
    Penalizes differences in Laplacian () between prediction and target.
    Useful for preserving precipitation gradients and curvature.
    
    Args:
        loss_type: 'l1' or 'l2'
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        self.loss_type = loss_type
        
        # Laplacian kernel
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('laplacian', laplacian)
        
    def _compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian."""
        if self.laplacian.device != x.device:
            self.laplacian = self.laplacian.to(x.device)
            
        channels = x.size(1)
        kernel = self.laplacian.repeat(channels, 1, 1, 1)
        
        return F.conv2d(x, kernel, padding=1, groups=channels)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian loss."""
        pred_lap = self._compute_laplacian(pred)
        target_lap = self._compute_laplacian(target)
        
        if self.loss_type == 'l1':
            return F.l1_loss(pred_lap, target_lap)
        else:
            return F.mse_loss(pred_lap, target_lap)


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for smoothness regularization.
    
    Encourages piece-wise smooth outputs while preserving edges.
    Can help reduce noise in precipitation predictions.
    
    Args:
        weight: Loss weight
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss."""
        
        # Differences in x and y directions
        diff_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        diff_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # Sum of absolute differences
        tv = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
        
        return self.weight * tv
