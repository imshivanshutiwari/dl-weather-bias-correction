"""
Spectral Loss for Frequency Domain Accuracy
============================================

Spectral loss computes differences in the frequency domain using FFT.
Ensures prediction has correct:
- Power spectrum (energy distribution across scales)
- Phase structure (spatial organization)

Important for precipitation which has characteristic spectral signatures.

Author: IITM Pune - Government of India Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpectralLoss(nn.Module):
    """
    Spectral Loss in frequency domain.
    
    Computes loss in Fourier space to ensure predictions have
    correct frequency characteristics. Precipitation fields have
    characteristic power spectra that should be preserved.
    
    Args:
        loss_type: 'l1' or 'l2'
        log_amplitude: Whether to use log amplitude
        amplitude_weight: Weight for amplitude loss
        phase_weight: Weight for phase loss
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        log_amplitude: bool = True,
        amplitude_weight: float = 1.0,
        phase_weight: float = 0.1
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.log_amplitude = log_amplitude
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Spectral loss
        """
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Amplitude (magnitude)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        if self.log_amplitude:
            pred_amp = torch.log(pred_amp + 1e-8)
            target_amp = torch.log(target_amp + 1e-8)
        
        # Phase
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Compute losses
        if self.loss_type == 'l1':
            amp_loss = F.l1_loss(pred_amp, target_amp)
            phase_loss = F.l1_loss(pred_phase, target_phase)
        else:
            amp_loss = F.mse_loss(pred_amp, target_amp)
            phase_loss = F.mse_loss(pred_phase, target_phase)
        
        loss = self.amplitude_weight * amp_loss + self.phase_weight * phase_loss
        
        return loss


class PowerSpectrumLoss(nn.Module):
    """
    Power Spectrum Loss.
    
    Compares radially averaged power spectra to ensure prediction
    has correct energy distribution across spatial scales.
    
    Args:
        num_bins: Number of radial bins
        weight_by_frequency: Weight loss by frequency (more weight on larger scales)
    """
    
    def __init__(
        self,
        num_bins: int = 32,
        weight_by_frequency: bool = True
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.weight_by_frequency = weight_by_frequency
        
    def _radial_average(
        self, 
        fft_magnitude: torch.Tensor
    ) -> torch.Tensor:
        """Compute radially averaged power spectrum."""
        B, C, H, W = fft_magnitude.shape
        
        # Create radial coordinate grid
        y = torch.arange(H, device=fft_magnitude.device) - H // 2
        x = torch.arange(W, device=fft_magnitude.device) - W // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
        
        max_radius = min(H, W) // 2
        bin_edges = torch.linspace(0, max_radius, self.num_bins + 1, device=fft_magnitude.device)
        
        power_spectrum = torch.zeros(B, C, self.num_bins, device=fft_magnitude.device)
        
        for i in range(self.num_bins):
            mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
            if mask.sum() > 0:
                power_spectrum[:, :, i] = (fft_magnitude * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1)) / mask.sum()
        
        return power_spectrum
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute power spectrum loss."""
        # Compute FFT and shift
        pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, norm='ortho'))
        target_fft = torch.fft.fftshift(torch.fft.fft2(target, norm='ortho'))
        
        # Power (magnitude squared)
        pred_power = torch.abs(pred_fft) ** 2
        target_power = torch.abs(target_fft) ** 2
        
        # Radially average
        pred_spectrum = self._radial_average(pred_power)
        target_spectrum = self._radial_average(target_power)
        
        # Log transform for stability
        pred_spectrum = torch.log(pred_spectrum + 1e-8)
        target_spectrum = torch.log(target_spectrum + 1e-8)
        
        # Compute loss
        if self.weight_by_frequency:
            # Weight by 1/frequency (larger scales more important)
            weights = 1.0 / (torch.arange(self.num_bins, device=pred.device).float() + 1)
            weights = weights / weights.sum()
            loss = (weights * (pred_spectrum - target_spectrum) ** 2).sum(dim=-1).mean()
        else:
            loss = F.mse_loss(pred_spectrum, target_spectrum)
        
        return loss


class WassersteinLoss(nn.Module):
    """
    Wasserstein Distance (Earth Mover's Distance) Loss.
    
    Measures the "work" required to transform one distribution
    into another. Better captures differences in precipitation
    distribution than pixel-wise losses.
    
    Uses 1D approximation along flattened dimension for efficiency.
    
    Args:
        p: Order of Wasserstein distance (1 or 2)
    """
    
    def __init__(self, p: int = 1):
        super().__init__()
        self.p = p
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Wasserstein distance."""
        B = pred.size(0)
        
        # Flatten spatial dimensions
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)
        
        # Sort along the flat dimension
        pred_sorted, _ = torch.sort(pred_flat, dim=1)
        target_sorted, _ = torch.sort(target_flat, dim=1)
        
        # Wasserstein distance is the L_p norm of sorted differences
        if self.p == 1:
            loss = torch.mean(torch.abs(pred_sorted - target_sorted))
        else:
            loss = torch.mean((pred_sorted - target_sorted) ** self.p) ** (1.0 / self.p)
        
        return loss
