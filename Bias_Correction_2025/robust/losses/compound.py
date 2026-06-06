"""
Compound Loss for Bias Correction
==================================

Comprehensive loss function combining multiple objectives:
1. MSE - Overall accuracy
2. SSIM - Structural preservation
3. Gradient - Edge preservation  
4. Focal - Extreme event focus
5. Spectral - Frequency accuracy
6. NLL - Uncertainty calibration

Author: IITM Pune - Government of India Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .ssim import SSIMLoss, MultiScaleSSIMLoss
from .gradient import GradientLoss, LaplacianLoss
from .focal import FocalLoss, WeightedMSELoss, AsymmetricLoss
from .spectral import SpectralLoss, PowerSpectrumLoss


class NLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for uncertainty calibration.
    
    Assumes Gaussian distribution with predicted mean and variance.
    Encourages well-calibrated uncertainty estimates.
    
    Formula:
        NLL = 0.5 * log() + 0.5 * (y - ) / 
        
    Args:
        eps: Small constant for numerical stability
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NLL loss.
        
        Args:
            mean: Predicted mean [B, C, H, W]
            variance: Predicted variance [B, C, H, W]
            target: Target values [B, C, H, W]
            
        Returns:
            NLL loss
        """
        # Ensure positive variance
        variance = torch.clamp(variance, min=self.eps)
        
        # NLL for Gaussian
        nll = 0.5 * torch.log(variance) + 0.5 * ((target - mean) ** 2) / variance
        
        return nll.mean()


class CompoundBiasCorrectionLoss(nn.Module):
    """
    Comprehensive Compound Loss for Bias Correction.
    
    Combines multiple loss terms to optimize for:
    - Pixel-wise accuracy (MSE/MAE)
    - Structural similarity (SSIM)
    - Edge preservation (Gradient)
    - Extreme event prediction (Focal)
    - Frequency accuracy (Spectral)
    - Uncertainty calibration (NLL)
    
    This is the MAIN LOSS FUNCTION for training.
    
    Args:
        config: Dictionary with loss weights and settings
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        if config is None:
            config = {}
            
        # Default weights (can be overridden by config)
        self.weights = {
            'mse': config.get('mse_weight', 1.0),
            'ssim': config.get('ssim_weight', 0.3),
            'gradient': config.get('gradient_weight', 0.2),
            'focal': config.get('focal_weight', 0.5),
            'spectral': config.get('spectral_weight', 0.1),
            'nll': config.get('nll_weight', 0.3),
        }
        
        # Initialize individual losses
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
        self.ssim = SSIMLoss(
            window_size=config.get('ssim_window', 11),
            data_range=config.get('data_range', 1.0)
        )
        
        self.gradient = GradientLoss(
            loss_type=config.get('gradient_type', 'l1')
        )
        
        self.focal = FocalLoss(
            gamma=config.get('focal_gamma', 2.0),
            alpha=config.get('focal_alpha', 0.25),
            threshold=config.get('focal_threshold', 0.1)
        )
        
        self.spectral = SpectralLoss(
            loss_type=config.get('spectral_type', 'l1'),
            amplitude_weight=config.get('spectral_amp_weight', 1.0),
            phase_weight=config.get('spectral_phase_weight', 0.1)
        )
        
        self.nll = NLLLoss()
        
        # Track components for logging
        self.last_losses = {}
        
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_var: Optional[torch.Tensor],
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute compound loss.
        
        Args:
            pred_mean: Predicted mean [B, C, H, W]
            pred_var: Predicted variance [B, C, H, W] (optional)
            target: Target values [B, C, H, W]
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss (and optionally component dict)
        """
        losses = {}
        
        # 1. MSE Loss - Basic accuracy
        losses['mse'] = self.weights['mse'] * self.mse(pred_mean, target)
        
        # 2. SSIM Loss - Structural similarity
        if self.weights['ssim'] > 0:
            losses['ssim'] = self.weights['ssim'] * self.ssim(pred_mean, target)
        
        # 3. Gradient Loss - Edge preservation
        if self.weights['gradient'] > 0:
            losses['gradient'] = self.weights['gradient'] * self.gradient(pred_mean, target)
        
        # 4. Focal Loss - Extreme events
        if self.weights['focal'] > 0:
            losses['focal'] = self.weights['focal'] * self.focal(pred_mean, target)
        
        # 5. Spectral Loss - Frequency accuracy
        if self.weights['spectral'] > 0:
            losses['spectral'] = self.weights['spectral'] * self.spectral(pred_mean, target)
        
        # 6. NLL Loss - Uncertainty calibration
        if pred_var is not None and self.weights['nll'] > 0:
            losses['nll'] = self.weights['nll'] * self.nll(pred_mean, pred_var, target)
        
        # Total loss
        total = sum(losses.values())
        losses['total'] = total
        
        # Store for logging
        self.last_losses = {k: v.item() for k, v in losses.items()}
        
        if return_components:
            return total, losses
        return total
    
    def get_last_losses(self) -> Dict[str, float]:
        """Get the last computed loss components."""
        return self.last_losses


class AdaptiveCompoundLoss(CompoundBiasCorrectionLoss):
    """
    Adaptive Compound Loss with learnable weights.
    
    Learns optimal weights for each loss component during training
    using uncertainty-based weighting.
    
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty 
               to Weigh Losses" (CVPR 2018)
    """
    
    def __init__(self, config: Dict = None, num_losses: int = 6):
        super().__init__(config)
        
        # Learnable log-variance for each loss (uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_var: Optional[torch.Tensor],
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """Compute adaptive loss with learned weights."""
        
        losses_list = []
        losses_dict = {}
        
        # Compute individual losses
        loss_mse = self.mse(pred_mean, target)
        losses_list.append(loss_mse)
        losses_dict['mse'] = loss_mse
        
        loss_ssim = self.ssim(pred_mean, target)
        losses_list.append(loss_ssim)
        losses_dict['ssim'] = loss_ssim
        
        loss_gradient = self.gradient(pred_mean, target)
        losses_list.append(loss_gradient)
        losses_dict['gradient'] = loss_gradient
        
        loss_focal = self.focal(pred_mean, target)
        losses_list.append(loss_focal)
        losses_dict['focal'] = loss_focal
        
        loss_spectral = self.spectral(pred_mean, target)
        losses_list.append(loss_spectral)
        losses_dict['spectral'] = loss_spectral
        
        if pred_var is not None:
            loss_nll = self.nll(pred_mean, pred_var, target)
            losses_list.append(loss_nll)
            losses_dict['nll'] = loss_nll
        
        # Adaptive weighting using learned log-variances
        # Loss =  (loss_i / (2 * _i) + log(_i))
        total = 0
        for i, loss in enumerate(losses_list):
            precision = torch.exp(-self.log_vars[i])  # 1 / 
            total += precision * loss + self.log_vars[i]
        
        losses_dict['total'] = total
        self.last_losses = {k: v.item() for k, v in losses_dict.items()}
        
        if return_components:
            return total, losses_dict
        return total


class CurriculumLoss(nn.Module):
    """
    Curriculum Loss that gradually increases difficulty.
    
    Starts with simple MSE and gradually adds more complex
    loss components as training progresses.
    
    Args:
        config: Loss configuration
        warmup_epochs: Epochs before adding each loss component
    """
    
    def __init__(
        self,
        config: Dict = None,
        warmup_epochs: int = 10
    ):
        super().__init__()
        
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Individual losses
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.gradient = GradientLoss()
        self.focal = FocalLoss()
        self.spectral = SpectralLoss()
        self.nll = NLLLoss()
        
        # Schedule: epoch threshold for each loss
        self.schedule = {
            'mse': 0,
            'ssim': warmup_epochs,
            'gradient': warmup_epochs * 2,
            'focal': warmup_epochs * 3,
            'spectral': warmup_epochs * 4,
        }
        
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum."""
        self.current_epoch = epoch
        
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_var: Optional[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute curriculum loss based on current epoch."""
        
        total = self.mse(pred_mean, target)
        
        if self.current_epoch >= self.schedule['ssim']:
            total += 0.3 * self.ssim(pred_mean, target)
            
        if self.current_epoch >= self.schedule['gradient']:
            total += 0.2 * self.gradient(pred_mean, target)
            
        if self.current_epoch >= self.schedule['focal']:
            total += 0.5 * self.focal(pred_mean, target)
            
        if self.current_epoch >= self.schedule['spectral']:
            total += 0.1 * self.spectral(pred_mean, target)
            
        if pred_var is not None:
            total += 0.3 * self.nll(pred_mean, pred_var, target)
            
        return total


def create_loss(config: Dict) -> nn.Module:
    """
    Factory function to create loss from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss module
    """
    loss_type = config.get('type', 'compound')
    
    if loss_type == 'compound':
        return CompoundBiasCorrectionLoss(config)
    elif loss_type == 'adaptive':
        return AdaptiveCompoundLoss(config)
    elif loss_type == 'curriculum':
        return CurriculumLoss(config)
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
