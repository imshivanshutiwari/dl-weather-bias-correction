"""
Uncertainty Quantification for Bias Correction
===============================================

This module implements uncertainty estimation for precipitation
bias correction. Uncertainty quantification is crucial for:

1. Identifying regions where the model is confident/uncertain
2. Propagating uncertainty to downstream applications
3. Calibrating ensemble predictions
4. Decision-making in operational weather forecasting

Methods implemented:
- Heteroscedastic regression (predicting both mean and variance)
- Monte Carlo Dropout
- Deep Ensembles
- Temperature scaling for calibration

Author: IITM Pune - Government of India Project
Reference: 
    - Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning 
      for Computer Vision?" (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class UncertaintyHead(nn.Module):
    """
    Dual-head output for heteroscedastic uncertainty estimation.
    
    Predicts both the mean () and variance () of the output distribution.
    The variance is predicted in log-space for numerical stability and
    passed through softplus to ensure positivity.
    
    Architecture:
        Input [B, C, H, W]
             Mean Head: Conv  Conv  output [B, 1, H, W]
             Var Head: Conv  Conv  Softplus  output [B, 1, H, W]
    
    Args:
        in_channels: Number of input channels
        hidden_channels: Hidden layer channels
        min_variance: Minimum variance to prevent numerical issues
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        min_variance: float = 1e-6
    ):
        super().__init__()
        
        self.min_variance = min_variance
        
        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        # Mean prediction head (NO ACTIVATION - critical for regression!)
        self.mean_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
            # NO ACTIVATION! This was a key bug in the original code.
        )
        
        # Variance prediction head (Softplus for positivity)
        self.var_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Softplus()  # Ensures positive variance
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting mean and variance.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            mean: Predicted mean [B, 1, H, W]
            variance: Predicted variance [B, 1, H, W]
        """
        shared_features = self.shared(x)
        
        mean = self.mean_head(shared_features)
        variance = self.var_head(shared_features) + self.min_variance
        
        return mean, variance


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.
    
    Applies dropout at inference time to estimate model uncertainty
    through multiple forward passes. High variance across passes
    indicates model uncertainty.
    
    Args:
        dropout_rate: Dropout probability
        num_samples: Number of MC samples for inference
    """
    
    def __init__(self, dropout_rate: float = 0.1, num_samples: int = 10):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_samples = num_samples
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout (also during inference if in MC mode)."""
        return self.dropout(x)
    
    def mc_forward(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform multiple forward passes with dropout.
        
        Args:
            model: The model to run MC dropout on
            x: Input tensor
            
        Returns:
            mean: Mean of MC samples
            variance: Variance of MC samples (epistemic uncertainty)
        """
        model.train()  # Enable dropout
        
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                samples.append(output)
                
        samples = torch.stack(samples, dim=0)  # [num_samples, B, C, H, W]
        
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)
        
        model.eval()
        return mean, variance


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification.
    
    Trains multiple models with different initializations and combines
    their predictions. Disagreement between models indicates uncertainty.
    
    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive 
               Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017)
    
    Args:
        model_class: Class of the base model
        model_kwargs: Keyword arguments for model initialization
        num_models: Number of ensemble members
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict,
        num_models: int = 5
    ):
        super().__init__()
        
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(num_models)
        ])
        self.num_models = num_models
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            mean: Ensemble mean prediction
            variance: Ensemble variance (uncertainty)
        """
        predictions = []
        
        for model in self.models:
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]  # Get mean if model outputs (mean, var)
            predictions.append(output)
            
        predictions = torch.stack(predictions, dim=0)  # [num_models, B, C, H, W]
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean, variance
    
    def load_ensemble(self, checkpoint_paths: list):
        """Load checkpoints for each ensemble member."""
        assert len(checkpoint_paths) == self.num_models
        for model, path in zip(self.models, checkpoint_paths):
            model.load_state_dict(torch.load(path))


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for uncertainty calibration.
    
    Learns a single temperature parameter to scale the predicted
    variance for better calibration.
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
    
    Args:
        init_temperature: Initial temperature value
    """
    
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))
        
    def forward(
        self, 
        mean: torch.Tensor, 
        variance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scale variance by temperature.
        
        Args:
            mean: Predicted mean [B, C, H, W]
            variance: Predicted variance [B, C, H, W]
            
        Returns:
            mean: Unchanged mean
            scaled_variance: Calibrated variance
        """
        scaled_variance = variance * self.temperature
        return mean, scaled_variance
    
    def calibrate(
        self, 
        predictions: torch.Tensor,
        variances: torch.Tensor,
        targets: torch.Tensor,
        num_iterations: int = 100
    ):
        """
        Calibrate temperature on validation set.
        
        Args:
            predictions: Model predictions
            variances: Predicted variances
            targets: Ground truth
            num_iterations: Optimization iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=num_iterations)
        
        def closure():
            optimizer.zero_grad()
            scaled_var = variances * self.temperature
            nll = 0.5 * torch.log(scaled_var) + 0.5 * ((targets - predictions)**2 / scaled_var)
            loss = nll.mean()
            loss.backward()
            return loss
            
        optimizer.step(closure)


class UncertaintyMetrics:
    """
    Metrics for evaluating uncertainty quality.
    """
    
    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        For well-calibrated uncertainty, ~p% of observations should
        fall within the p% confidence interval.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted standard deviations
            targets: Ground truth
            num_bins: Number of calibration bins
            
        Returns:
            ECE value (lower is better, 0 = perfectly calibrated)
        """
        errors = np.abs(predictions - targets)
        
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties.flatten())
        sorted_errors = errors.flatten()[sorted_indices]
        sorted_uncs = uncertainties.flatten()[sorted_indices]
        
        # Bin by uncertainty quantiles
        bin_size = len(sorted_errors) // num_bins
        ece = 0.0
        
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_errors)
            
            bin_errors = sorted_errors[start:end]
            bin_uncs = sorted_uncs[start:end]
            
            # Expected coverage vs actual coverage
            expected_coverage = (i + 0.5) / num_bins
            actual_coverage = np.mean(bin_errors <= 1.96 * bin_uncs)  # 95% CI
            
            ece += np.abs(expected_coverage - actual_coverage) / num_bins
            
        return ece
    
    @staticmethod
    def prediction_interval_coverage(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Compute Prediction Interval Coverage Probability (PICP).
        
        Measures what fraction of observations fall within the
        predicted confidence interval.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted standard deviations
            targets: Ground truth
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            PICP (should be close to confidence level for calibrated model)
        """
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties
        
        coverage = np.mean((targets >= lower) & (targets <= upper))
        
        return coverage
    
    @staticmethod
    def mean_prediction_interval_width(
        uncertainties: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Compute Mean Prediction Interval Width (MPIW).
        
        Measures the average width of prediction intervals.
        Lower is better (but only if PICP is maintained).
        
        Args:
            uncertainties: Predicted standard deviations
            confidence: Confidence level
            
        Returns:
            MPIW
        """
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        interval_width = 2 * z_score * uncertainties
        
        return np.mean(interval_width)
