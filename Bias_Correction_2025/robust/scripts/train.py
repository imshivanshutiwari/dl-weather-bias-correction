#!/usr/bin/env python
"""
Train Bias Correction Model
============================

Main training script for the IITM Pune Government of India
production-grade bias correction system.

Usage:
    python train.py --config config/production.yaml
    python train.py --config config/production.yaml --epochs 100 --lr 1e-4

Author: IITM Pune - Government of India Project
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from scripts -> robust -> Bias_Correction_2025
sys.path.insert(0, str(PROJECT_ROOT))

from robust.models.msua_net import MSUANet, MSUANetLite, create_model
from robust.data.datasets import BiasCorrectionDataset, MultiTemporalDataset, create_dataloaders, RobustNormalizer
from robust.losses.compound import CompoundBiasCorrectionLoss, create_loss
from robust.training.trainer import BiasCorrectionTrainer, TrainingConfig
from robust.evaluation.metrics import BiasCorrectionMetrics, print_metrics_table


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config with command line arguments."""
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        config['data']['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device
    if args.checkpoint_dir:
        config['training']['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        config['training']['log_dir'] = args.log_dir
        
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train IITM Pune Bias Correction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=str, 
        default='robust/config/production.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Override learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Override device (cuda/cpu)'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default=None,
        help='Override checkpoint directory'
    )
    parser.add_argument(
        '--log_dir', type=str, default=None,
        help='Override log directory'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode (smaller dataset)'
    )
    
    args = parser.parse_args()
    
    # Load and override config
    print("\n" + "=" * 60)
    print(" IITM Pune - Government of India")
    print("   Production Bias Correction System")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f" Config file not found: {config_path}")
        print("   Creating default config...")
        # Use default config
        config = get_default_config()
    else:
        config = load_config(config_path)
        
    config = override_config(config, args)
    
    # Print config
    print("\n Configuration:")
    print(f"   Model: {config['model']['name']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Device: {config['device']}")
    
    # Set seed
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    print(f"   Random seed: {seed}")
    
    # Setup device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("    CUDA not available, using CPU")
        device = 'cpu'
        config['device'] = device
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
    # Create directories
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    log_dir = Path(config['training']['log_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to checkpoint dir
    config_save_path = checkpoint_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   Config saved: {config_save_path}")
    
    # Check data paths
    data_config = config['data']
    nwp_path = Path(data_config['nwp_path'])
    obs_path = Path(data_config['obs_path'])
    
    if not nwp_path.exists():
        print(f"\n NWP data not found: {nwp_path}")
        print("   Please update the data paths in the config file.")
        print("   Using demo mode with synthetic data...")
        demo_mode = True
    else:
        demo_mode = False
        
    # Create model
    print("\n Creating model...")
    model = create_model(config['model'], device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    if demo_mode or args.debug:
        # Create synthetic data for testing
        print("\n Creating synthetic demo data...")
        train_loader, val_loader, normalizer = create_synthetic_data(config)
    else:
        # Create real dataloaders
        print("\n Loading data...")
        train_loader, val_loader, normalizer = create_dataloaders(
            data_config,
            train_years=data_config.get('train_years', [2019, 2020, 2021, 2022]),
            val_years=data_config.get('val_years', [2023])
        )
        
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n Setting up trainer...")
    training_config = TrainingConfig(config['training'])
    trainer = BiasCorrectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        loss_config=config.get('loss', {})
    )
    
    # Resume if specified
    if args.resume:
        print(f"\n Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        
    # Train
    print("\n" + "=" * 60)
    history = trainer.train()
    
    # Final evaluation
    print("\n Final Evaluation:")
    model.load_state_dict(
        torch.load(checkpoint_dir / 'best_model.pth', map_location=device)['model_state_dict']
    )
    val_loss, metrics = trainer.validate()
    print_metrics_table(metrics, "Final Validation Metrics")
    
    # Save final metrics
    import json
    metrics_path = checkpoint_dir / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved: {metrics_path}")
    
    print("\n Training complete!")
    print(f"   Best model: {checkpoint_dir / 'best_model.pth'}")
    print(f"   Logs: {log_dir}")
    print("=" * 60 + "\n")
    

def create_synthetic_data(config: dict):
    """Create synthetic data for demo/testing."""
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = config['training']['batch_size']
    in_channels = config['model']['in_channels']
    
    # Create synthetic tensors
    n_train = 200
    n_val = 50
    H, W = 64, 64
    
    # Training data
    train_inputs = torch.randn(n_train, in_channels, H, W)
    train_targets = torch.randn(n_train, 1, H, W) * 0.5 + train_inputs[:, 0:1, :, :]
    train_targets = torch.clamp(train_targets, 0, 1)
    
    # Validation data
    val_inputs = torch.randn(n_val, in_channels, H, W)
    val_targets = torch.randn(n_val, 1, H, W) * 0.5 + val_inputs[:, 0:1, :, :]
    val_targets = torch.clamp(val_targets, 0, 1)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    # Wrapper to convert TensorDataset to expected format
    class WrapperDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            return {'input': x, 'target': y, 'time': '', 'index': idx}
            
    train_loader = DataLoader(
        WrapperDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        WrapperDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    normalizer = RobustNormalizer()
    normalizer.fitted = True
    normalizer.lower_bound = 0
    normalizer.upper_bound = 1
    normalizer.mean = 0.5
    normalizer.std = 0.2
    
    return train_loader, val_loader, normalizer


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        'experiment': {
            'name': 'default',
            'seed': 42
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data': {
            'nwp_path': 'data/nwp.nc',
            'obs_path': 'data/obs.nc',
            'batch_size': 8,
            'num_workers': 0,
            'augment': False
        },
        'model': {
            'name': 'MSUANetLite',
            'in_channels': 10,
            'out_channels': 1,
            'base_filters': 32,
            'uncertainty': False
        },
        'loss': {
            'type': 'compound',
            'mse_weight': 1.0,
            'ssim_weight': 0.1,
            'gradient_weight': 0.1,
            'focal_weight': 0.1,
            'spectral_weight': 0.0,
            'nll_weight': 0.0
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'T_0': 10,
            'warmup_epochs': 2,
            'use_amp': False,
            'gradient_clipping': 1.0,
            'early_stopping_patience': 20,
            'checkpoint_dir': 'checkpoints/default',
            'log_dir': 'logs/default',
            'save_every': 5
        }
    }


if __name__ == '__main__':
    main()
