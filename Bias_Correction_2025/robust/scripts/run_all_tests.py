#!/usr/bin/env python
"""
Comprehensive Test Suite for Robust Bias Correction System
============================================================

This script tests ALL components to ensure they work correctly.
Run this before deploying to lab system.

Author: IITM Pune - Government of India Project
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import traceback

# Track test results
PASSED = []
FAILED = []

def test_result(name, success, error=None):
    if success:
        print(f" PASS: {name}")
        PASSED.append(name)
    else:
        print(f" FAIL: {name}")
        if error:
            print(f"   Error: {error}")
        FAILED.append((name, error))

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ============================================================
# TEST 1: All Module Imports
# ============================================================
print_separator("TEST 1: Module Imports")

try:
    from robust.models.attention import SqueezeExcitation, AttentionGate, SelfAttention, CBAM
    test_result("Import attention module", True)
except Exception as e:
    test_result("Import attention module", False, str(e))

try:
    from robust.models.residual import ResidualBlock, ResidualGroup, DoubleConv
    test_result("Import residual module", True)
except Exception as e:
    test_result("Import residual module", False, str(e))

try:
    from robust.models.multiscale import MultiScaleFeatureExtractor, PyramidPoolingModule, ASPP
    test_result("Import multiscale module", True)
except Exception as e:
    test_result("Import multiscale module", False, str(e))

try:
    from robust.models.uncertainty import UncertaintyHead, MCDropout
    test_result("Import uncertainty module", True)
except Exception as e:
    test_result("Import uncertainty module", False, str(e))

try:
    from robust.models.msua_net import MSUANet, MSUANetLite, create_model
    test_result("Import msua_net module", True)
except Exception as e:
    test_result("Import msua_net module", False, str(e))

try:
    from robust.losses.ssim import SSIMLoss, MultiScaleSSIMLoss
    test_result("Import ssim loss", True)
except Exception as e:
    test_result("Import ssim loss", False, str(e))

try:
    from robust.losses.gradient import GradientLoss, LaplacianLoss
    test_result("Import gradient loss", True)
except Exception as e:
    test_result("Import gradient loss", False, str(e))

try:
    from robust.losses.focal import FocalLoss, WeightedMSELoss, AsymmetricLoss
    test_result("Import focal loss", True)
except Exception as e:
    test_result("Import focal loss", False, str(e))

try:
    from robust.losses.spectral import SpectralLoss, PowerSpectrumLoss
    test_result("Import spectral loss", True)
except Exception as e:
    test_result("Import spectral loss", False, str(e))

try:
    from robust.losses.compound import CompoundBiasCorrectionLoss, create_loss
    test_result("Import compound loss", True)
except Exception as e:
    test_result("Import compound loss", False, str(e))

try:
    from robust.data.datasets import BiasCorrectionDataset, RobustNormalizer, create_dataloaders
    test_result("Import data module", True)
except Exception as e:
    test_result("Import data module", False, str(e))

try:
    from robust.evaluation.metrics import BiasCorrectionMetrics, print_metrics_table
    test_result("Import metrics module", True)
except Exception as e:
    test_result("Import metrics module", False, str(e))

try:
    from robust.training.trainer import BiasCorrectionTrainer, TrainingConfig, EarlyStopping
    test_result("Import trainer module", True)
except Exception as e:
    test_result("Import trainer module", False, str(e))

# ============================================================
# TEST 2: Attention Modules
# ============================================================
print_separator("TEST 2: Attention Modules")

try:
    se = SqueezeExcitation(64)
    x = torch.randn(2, 64, 32, 32)
    out = se(x)
    assert out.shape == x.shape, f"SE output shape mismatch: {out.shape}"
    test_result("SqueezeExcitation forward", True)
except Exception as e:
    test_result("SqueezeExcitation forward", False, str(e))

try:
    cbam = CBAM(64)
    x = torch.randn(2, 64, 32, 32)
    out = cbam(x)
    assert out.shape == x.shape, f"CBAM output shape mismatch: {out.shape}"
    test_result("CBAM forward", True)
except Exception as e:
    test_result("CBAM forward", False, str(e))

try:
    ag = AttentionGate(F_g=64, F_l=64, F_int=32)
    g = torch.randn(2, 64, 16, 16)
    x = torch.randn(2, 64, 32, 32)
    out = ag(g, x)
    assert out.shape == x.shape, f"AttentionGate output shape mismatch: {out.shape}"
    test_result("AttentionGate forward", True)
except Exception as e:
    test_result("AttentionGate forward", False, str(e))

try:
    sa = SelfAttention(64, num_heads=4)
    x = torch.randn(2, 64, 16, 16)
    out = sa(x)
    assert out.shape == x.shape, f"SelfAttention output shape mismatch: {out.shape}"
    test_result("SelfAttention forward", True)
except Exception as e:
    test_result("SelfAttention forward", False, str(e))

# ============================================================
# TEST 3: Residual Blocks
# ============================================================
print_separator("TEST 3: Residual Blocks")

try:
    rb = ResidualBlock(64)
    x = torch.randn(2, 64, 32, 32)
    out = rb(x)
    assert out.shape == x.shape, f"ResidualBlock output shape mismatch: {out.shape}"
    test_result("ResidualBlock forward", True)
except Exception as e:
    test_result("ResidualBlock forward", False, str(e))

try:
    rg = ResidualGroup(64, num_blocks=3)
    x = torch.randn(2, 64, 32, 32)
    out = rg(x)
    assert out.shape == x.shape, f"ResidualGroup output shape mismatch: {out.shape}"
    test_result("ResidualGroup forward", True)
except Exception as e:
    test_result("ResidualGroup forward", False, str(e))

try:
    dc = DoubleConv(32, 64)
    x = torch.randn(2, 32, 32, 32)
    out = dc(x)
    assert out.shape == (2, 64, 32, 32), f"DoubleConv output shape mismatch: {out.shape}"
    test_result("DoubleConv forward", True)
except Exception as e:
    test_result("DoubleConv forward", False, str(e))

# ============================================================
# TEST 4: Multi-Scale Feature Extractor
# ============================================================
print_separator("TEST 4: Multi-Scale Feature Extractor")

try:
    msfe = MultiScaleFeatureExtractor(in_channels=10, base_filters=32, out_channels=128)
    x = torch.randn(2, 10, 64, 64)
    out = msfe(x)
    assert out.shape == (2, 128, 64, 64), f"MSFE output shape mismatch: {out.shape}"
    test_result("MultiScaleFeatureExtractor forward", True)
except Exception as e:
    test_result("MultiScaleFeatureExtractor forward", False, str(e))

try:
    ppm = PyramidPoolingModule(256, 128)
    x = torch.randn(2, 256, 32, 32)
    out = ppm(x)
    assert out.shape == (2, 128, 32, 32), f"PPM output shape mismatch: {out.shape}"
    test_result("PyramidPoolingModule forward", True)
except Exception as e:
    test_result("PyramidPoolingModule forward", False, str(e))

try:
    aspp = ASPP(256, 128)
    x = torch.randn(2, 256, 32, 32)
    out = aspp(x)
    assert out.shape == (2, 128, 32, 32), f"ASPP output shape mismatch: {out.shape}"
    test_result("ASPP forward", True)
except Exception as e:
    test_result("ASPP forward", False, str(e))

# ============================================================
# TEST 5: Uncertainty Module
# ============================================================
print_separator("TEST 5: Uncertainty Module")

try:
    uh = UncertaintyHead(64)
    x = torch.randn(2, 64, 32, 32)
    mean, var = uh(x)
    assert mean.shape == (2, 1, 32, 32), f"Mean shape mismatch: {mean.shape}"
    assert var.shape == (2, 1, 32, 32), f"Var shape mismatch: {var.shape}"
    assert (var > 0).all(), "Variance must be positive"
    test_result("UncertaintyHead forward", True)
except Exception as e:
    test_result("UncertaintyHead forward", False, str(e))

# ============================================================
# TEST 6: MSUA-Net Full Model
# ============================================================
print_separator("TEST 6: MSUA-Net Full Model")

try:
    model = MSUANet(in_channels=10, out_channels=1, base_filters=32, num_levels=3, uncertainty=True)
    x = torch.randn(2, 10, 64, 64)
    mean, var = model(x)
    assert mean.shape == (2, 1, 64, 64), f"Mean shape mismatch: {mean.shape}"
    assert var.shape == (2, 1, 64, 64), f"Var shape mismatch: {var.shape}"
    test_result("MSUANet forward (with uncertainty)", True)
except Exception as e:
    test_result("MSUANet forward (with uncertainty)", False, str(e))

try:
    model = MSUANet(in_channels=10, out_channels=1, base_filters=32, num_levels=3, uncertainty=False)
    x = torch.randn(2, 10, 64, 64)
    out, _ = model(x)
    assert out.shape == (2, 1, 64, 64), f"Output shape mismatch: {out.shape}"
    test_result("MSUANet forward (no uncertainty)", True)
except Exception as e:
    test_result("MSUANet forward (no uncertainty)", False, str(e))

try:
    model = MSUANetLite(in_channels=10, out_channels=1, base_filters=16)
    x = torch.randn(2, 10, 64, 64)
    out, _ = model(x)
    assert out.shape == (2, 1, 64, 64), f"Output shape mismatch: {out.shape}"
    test_result("MSUANetLite forward", True)
except Exception as e:
    test_result("MSUANetLite forward", False, str(e))

# ============================================================
# TEST 7: Loss Functions
# ============================================================
print_separator("TEST 7: Loss Functions")

pred = torch.randn(2, 1, 32, 32)
target = torch.randn(2, 1, 32, 32)

try:
    loss_fn = SSIMLoss()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    test_result("SSIMLoss", True)
except Exception as e:
    test_result("SSIMLoss", False, str(e))

try:
    loss_fn = GradientLoss()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    test_result("GradientLoss", True)
except Exception as e:
    test_result("GradientLoss", False, str(e))

try:
    loss_fn = FocalLoss()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    test_result("FocalLoss", True)
except Exception as e:
    test_result("FocalLoss", False, str(e))

try:
    loss_fn = SpectralLoss()
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    test_result("SpectralLoss", True)
except Exception as e:
    test_result("SpectralLoss", False, str(e))

try:
    loss_fn = CompoundBiasCorrectionLoss()
    var = torch.abs(torch.randn(2, 1, 32, 32)) + 0.1
    loss = loss_fn(pred, var, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    test_result("CompoundBiasCorrectionLoss", True)
except Exception as e:
    test_result("CompoundBiasCorrectionLoss", False, str(e))

# ============================================================
# TEST 8: Data Module
# ============================================================
print_separator("TEST 8: Data Module")

try:
    normalizer = RobustNormalizer(method='percentile')
    data = np.random.randn(100) * 10 + 5
    normalizer.fit(data)
    normalized = normalizer.transform(data)
    assert normalized.min() >= 0, "Normalized min should be >= 0"
    assert normalized.max() <= 1, "Normalized max should be <= 1"
    denormalized = normalizer.inverse_transform(normalized)
    # Note: percentile normalization clips values, so we check most values (98%) match
    # Values outside 1st-99th percentile get clipped and won't exactly match
    close_count = np.sum(np.isclose(data, denormalized, rtol=0.01))
    assert close_count >= 95, f"At least 95% should match, got {close_count}"
    test_result("RobustNormalizer percentile", True)
except Exception as e:
    test_result("RobustNormalizer percentile", False, str(e))

try:
    normalizer = RobustNormalizer(method='zscore')
    data = np.random.randn(100) * 10 + 5
    normalizer.fit(data)
    normalized = normalizer.transform(data)
    denormalized = normalizer.inverse_transform(normalized)
    np.testing.assert_allclose(data, denormalized, rtol=1e-5, atol=1e-5)
    test_result("RobustNormalizer zscore", True)
except Exception as e:
    test_result("RobustNormalizer zscore", False, str(e))

# ============================================================
# TEST 9: Metrics
# ============================================================
print_separator("TEST 9: Metrics")

try:
    metrics = BiasCorrectionMetrics()
    pred = np.random.randn(100, 64, 64)
    target = pred + np.random.randn(100, 64, 64) * 0.1
    results = metrics.compute_all(pred, target)
    
    assert 'MAE' in results, "MAE missing"
    assert 'RMSE' in results, "RMSE missing"
    assert 'Pearson_R' in results, "Pearson_R missing"
    assert 'NSE' in results, "NSE missing"
    assert 'KGE' in results, "KGE missing"
    
    # Check reasonable values
    assert results['MAE'] >= 0, "MAE should be non-negative"
    assert results['RMSE'] >= 0, "RMSE should be non-negative"
    assert -1 <= results['Pearson_R'] <= 1, "Pearson_R should be in [-1, 1]"
    
    test_result("BiasCorrectionMetrics.compute_all", True)
except Exception as e:
    test_result("BiasCorrectionMetrics.compute_all", False, str(e))

try:
    fss = BiasCorrectionMetrics.compute_fss(
        pred=np.random.rand(64, 64) * 10,
        target=np.random.rand(64, 64) * 10,
        threshold=5.0,
        scale=3
    )
    assert 0 <= fss <= 1, f"FSS should be in [0, 1], got {fss}"
    test_result("FSS metric", True)
except Exception as e:
    test_result("FSS metric", False, str(e))

# ============================================================
# TEST 10: Training Components
# ============================================================
print_separator("TEST 10: Training Components")

try:
    config = TrainingConfig({
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'use_amp': False
    })
    assert config.batch_size == 8
    assert config.learning_rate == 1e-4
    test_result("TrainingConfig", True)
except Exception as e:
    test_result("TrainingConfig", False, str(e))

try:
    es = EarlyStopping(patience=5)
    for i in range(10):
        # Simulating decreasing loss
        stop = es(1.0 - i * 0.1)
    assert not stop, "Should not stop with decreasing loss"
    test_result("EarlyStopping (decreasing)", True)
except Exception as e:
    test_result("EarlyStopping (decreasing)", False, str(e))

try:
    es = EarlyStopping(patience=3)
    results = []
    for i in range(10):
        # Simulating constant loss
        results.append(es(1.0))
    assert results[-1] == True, "Should stop after patience exceeded"
    test_result("EarlyStopping (constant)", True)
except Exception as e:
    test_result("EarlyStopping (constant)", False, str(e))

# ============================================================
# TEST 11: Gradient Flow (Backpropagation)
# ============================================================
print_separator("TEST 11: Gradient Flow (Backpropagation)")

try:
    model = MSUANet(in_channels=10, base_filters=16, num_levels=2, uncertainty=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = CompoundBiasCorrectionLoss()
    
    x = torch.randn(2, 10, 32, 32)
    target = torch.randn(2, 1, 32, 32)
    
    mean, var = model(x)
    loss = loss_fn(mean, var, target)
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed"
    test_result("Gradient flow check", True)
except Exception as e:
    test_result("Gradient flow check", False, str(e))

try:
    # One optimization step
    optimizer.step()
    test_result("Optimizer step", True)
except Exception as e:
    test_result("Optimizer step", False, str(e))

# ============================================================
# TEST 12: GPU Compatibility (if available)
# ============================================================
print_separator("TEST 12: GPU Compatibility")

if torch.cuda.is_available():
    try:
        device = torch.device('cuda')
        model = MSUANet(in_channels=10, base_filters=16, num_levels=2).to(device)
        x = torch.randn(2, 10, 32, 32).to(device)
        mean, var = model(x)
        assert mean.device.type == 'cuda', "Output not on CUDA"
        test_result("GPU forward pass", True)
    except Exception as e:
        test_result("GPU forward pass", False, str(e))
else:
    print("   CUDA not available, skipping GPU tests")

# ============================================================
# SUMMARY
# ============================================================
print_separator("TEST SUMMARY")

print(f"\n  Total tests: {len(PASSED) + len(FAILED)}")
print(f"   Passed: {len(PASSED)}")
print(f"   Failed: {len(FAILED)}")

if FAILED:
    print("\n  Failed tests:")
    for name, error in FAILED:
        print(f"    - {name}: {error}")
    print("\n SOME TESTS FAILED! Please fix before running on lab system.")
    sys.exit(1)
else:
    print("\n ALL TESTS PASSED! Code is ready for lab system.")
    sys.exit(0)
