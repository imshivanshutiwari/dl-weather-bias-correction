"""
Focal Loss for Extreme Event Prediction
========================================

Focal loss down-weights easy examples and focuses on hard ones.
For precipitation, this is crucial because:
- Most pixels have low/no rainfall (easy)
- Extreme events are rare but critical (hard)
- Standard MSE underpredicts extremes

Author: IITM Pune - Government of India Project
Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss adapted for regression (continuous values).
    
    Focuses training on hard examples (extreme precipitation events)
    by down-weighting loss from easy examples (normal/no rain).
    
    Formula:
        FL(p_t) = -_t (1 - p_t)^ log(p_t)
        
    For regression, we adapt this to:
        FL_reg = (|error| / max_error)^  base_loss
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balance factor (weight for positive class in classification)
        threshold: Threshold for defining extreme events (e.g., 50mm)
        extreme_weight: Extra weight for extreme events
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        threshold: float = 0.1,  # Normalized threshold
        extreme_weight: float = 2.0
    ):
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        self.extreme_weight = extreme_weight
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for regression.
        
        Args:
            pred: Predicted values [B, C, H, W]
            target: Target values [B, C, H, W]
            
        Returns:
            Focal loss
        """
        # Compute absolute error
        error = torch.abs(pred - target)
        
        # Normalize error for focal weight (0-1 range)
        max_error = error.max().detach() + 1e-6
        normalized_error = error / max_error
        
        # Focal weight: higher for larger errors
        focal_weight = torch.pow(normalized_error, self.gamma)
        
        # Extreme event weight: higher for large target values
        extreme_mask = (target > self.threshold).float()
        event_weight = 1.0 + (self.extreme_weight - 1.0) * extreme_mask
        
        # Combined weight
        weight = self.alpha * focal_weight * event_weight
        
        # Weighted MSE loss
        loss = weight * (error ** 2)
        
        return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss with intensity-based weighting.
    
    Applies higher weights to pixels with higher precipitation,
    ensuring the model accurately predicts heavy rainfall events.
    
    Args:
        base_weight: Base weight for all pixels
        intensity_power: Power for intensity-based weighting
        min_weight: Minimum weight (prevents ignoring zero pixels)
        max_weight: Maximum weight (prevents explosion)
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        intensity_power: float = 0.5,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        super().__init__()
        
        self.base_weight = base_weight
        self.intensity_power = intensity_power
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            pred: Predicted values [B, C, H, W]
            target: Target values [B, C, H, W]
            
        Returns:
            Weighted MSE loss
        """
        # Compute intensity-based weights
        weights = self.base_weight + torch.pow(target.abs() + 1e-6, self.intensity_power)
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        
        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = weights * mse
        
        return weighted_mse.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss penalizing under-prediction more than over-prediction.
    
    For precipitation forecasting, under-predicting heavy rainfall
    can have severe consequences (floods, disasters), so we penalize
    under-prediction more heavily.
    
    Args:
        under_weight: Weight for under-prediction errors
        over_weight: Weight for over-prediction errors
    """
    
    def __init__(
        self,
        under_weight: float = 2.0,
        over_weight: float = 1.0
    ):
        super().__init__()
        
        self.under_weight = under_weight
        self.over_weight = over_weight
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            pred: Predicted values [B, C, H, W]
            target: Target values [B, C, H, W]
            
        Returns:
            Asymmetric loss
        """
        diff = pred - target
        
        # Under-prediction (pred < target)
        under_mask = (diff < 0).float()
        
        # Over-prediction (pred > target)
        over_mask = (diff >= 0).float()
        
        # Weighted squared error
        loss = (
            self.under_weight * under_mask * (diff ** 2) +
            self.over_weight * over_mask * (diff ** 2)
        )
        
        return loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile Loss for probabilistic prediction.
    
    Predicts specific quantiles of the distribution rather than
    just the mean. Useful for ensemble/probabilistic forecasting.
    
    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        
        self.quantiles = quantiles
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        quantile_idx: int = 1  # Default to median
    ) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            pred: Predicted quantile value [B, C, H, W]
            target: Target values [B, C, H, W]
            quantile_idx: Index of quantile in self.quantiles
            
        Returns:
            Quantile loss
        """
        q = self.quantiles[quantile_idx]
        error = target - pred
        
        # Asymmetric loss based on quantile
        loss = torch.max(q * error, (q - 1) * error)
        
        return loss.mean()
