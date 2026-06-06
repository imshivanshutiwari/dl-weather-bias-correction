# Losses module
from .compound import CompoundBiasCorrectionLoss
from .ssim import SSIMLoss
from .gradient import GradientLoss
from .focal import FocalLoss
from .spectral import SpectralLoss

__all__ = [
    'CompoundBiasCorrectionLoss',
    'SSIMLoss',
    'GradientLoss',
    'FocalLoss',
    'SpectralLoss'
]
