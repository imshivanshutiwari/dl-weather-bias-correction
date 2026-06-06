"""
Production-Grade Trainer for Bias Correction
=============================================

Complete training infrastructure with:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate warmup and scheduling
- Comprehensive logging (TensorBoard)
- Automatic checkpointing
- Early stopping
- Distributed training support

Author: IITM Pune - Government of India Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from tqdm import tqdm

from ..losses.compound import CompoundBiasCorrectionLoss, create_loss
from ..evaluation.metrics import BiasCorrectionMetrics, print_metrics_table


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Training
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 200)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.gradient_clipping = config.get('gradient_clipping', 1.0)
        
        # Optimizer
        self.optimizer = config.get('optimizer', 'AdamW')
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.betas = config.get('betas', (0.9, 0.999))
        
        # Scheduler
        self.scheduler = config.get('scheduler', 'CosineAnnealingWarmRestarts')
        self.T_0 = config.get('T_0', 10)
        self.T_mult = config.get('T_mult', 2)
        self.eta_min = config.get('eta_min', 1e-7)
        
        # Warmup
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.warmup_start_lr = config.get('warmup_start_lr', 1e-7)
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 30)
        self.early_stopping_min_delta = config.get('early_stopping_min_delta', 1e-5)
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.save_every = config.get('save_every', 5)
        self.keep_last = config.get('keep_last', 10)
        
        # Logging
        self.log_dir = config.get('log_dir', 'logs')
        self.eval_every = config.get('eval_every', 1)
        
        # Device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training when validation loss doesn't improve for patience epochs.
    """
    
    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 1e-5,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
                
        return False


class ModelCheckpoint:
    """
    Automatic model checkpointing.
    
    Saves best model and optionally periodic checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 5,
        keep_last: int = 10,
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        self.mode = mode
        self.best_score = None
        self.saved_checkpoints = []
        
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epoch: int,
        score: float,
        config: Dict
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'score': score,
            'config': config
        }
        
        # Check if best
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == 'min' and score < self.best_score:
            is_best = True
        elif self.mode == 'max' and score > self.best_score:
            is_best = True
            
        if is_best:
            self.best_score = score
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f" Saved best model (score: {score:.6f})")
            
        # Periodic save
        if epoch % self.save_every == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, path)
            self.saved_checkpoints.append(path)
            
            # Remove old checkpoints
            while len(self.saved_checkpoints) > self.keep_last:
                old_path = self.saved_checkpoints.pop(0)
                if old_path.exists():
                    old_path.unlink()
                    
    def load_best(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ) -> Tuple[nn.Module, int, float]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pth'
        if not best_path.exists():
            raise FileNotFoundError(f"No best model found at {best_path}")
            
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint['epoch'], checkpoint['score']


class BiasCorrectionTrainer:
    """
    Complete trainer for bias correction model.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate warmup
    - Comprehensive logging
    - Automatic checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_config: Dict = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Move model to device
        self.device = config.device
        self.model = self.model.to(self.device)
        
        # Setup loss
        self.criterion = create_loss(loss_config or {})
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
        
        # Checkpointing
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            save_every=config.save_every,
            keep_last=config.keep_last
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.logger = self._setup_logger()
        
        # Metrics
        self.metrics_calculator = BiasCorrectionMetrics()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas
            )
        elif self.config.optimizer == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas
            )
        elif self.config.optimizer == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.eta_min
            )
        elif self.config.scheduler == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        elif self.config.scheduler == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
            
    def _setup_logger(self) -> logging.Logger:
        """Setup file logger."""
        logger = logging.getLogger('BiasCorrectionTrainer')
        logger.setLevel(logging.INFO)
        
        log_path = Path(self.config.log_dir) / 'training.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        
        return logger
        
    def _warmup_lr(self, epoch: int, step: int, total_steps: int):
        """Apply learning rate warmup."""
        if epoch >= self.config.warmup_epochs:
            return
            
        warmup_factor = (epoch * total_steps + step) / (self.config.warmup_epochs * total_steps)
        lr = self.config.warmup_start_lr + warmup_factor * (
            self.config.learning_rate - self.config.warmup_start_lr
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        # Use tqdm for progress bar
        pbar = tqdm(enumerate(self.train_loader), total=num_batches, 
                    desc=f"Epoch {self.current_epoch+1}", unit="batch",
                    ncols=100, leave=True)
        
        for batch_idx, batch in pbar:
            # Warmup
            self._warmup_lr(self.current_epoch, batch_idx, num_batches)
            
            # Get data
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    pred_mean, pred_var = self.model(inputs)
                    loss = self.criterion(pred_mean, pred_var, targets)
                    loss = loss / self.config.gradient_accumulation_steps
                    
                # Backward with scaling
                self.scaler.scale(loss).backward()
            else:
                pred_mean, pred_var = self.model(inputs)
                loss = self.criterion(pred_mean, pred_var, targets)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clipping
                    )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                self.global_step += 1
                
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Update progress bar
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
                
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in self.val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            if self.config.use_amp:
                with autocast():
                    pred_mean, pred_var = self.model(inputs)
                    loss = self.criterion(pred_mean, pred_var, targets)
            else:
                pred_mean, pred_var = self.model(inputs)
                loss = self.criterion(pred_mean, pred_var, targets)
                
            total_loss += loss.item()
            
            all_preds.append(pred_mean.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        metrics = self.metrics_calculator.compute_all(preds, targets)
        metrics['val_loss'] = avg_loss
        
        return avg_loss, metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns:
            Training history
        """
        print("\n" + "=" * 60)
        print(" Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.use_amp}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\n Epoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Get current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', lr, epoch)
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")
            
            if 'Pearson_R' in metrics:
                print(f"  Pearson R:  {metrics['Pearson_R']:.4f}")
            if 'NSE' in metrics:
                print(f"  NSE:        {metrics['NSE']:.4f}")
                
            # Save checkpoint
            self.checkpoint.save(
                self.model, self.optimizer, self.scheduler,
                epoch, val_loss, vars(self.config)
            )
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n Early stopping at epoch {epoch + 1}")
                break
                
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(" Training Complete!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best val loss: {self.checkpoint.best_score:.6f}")
        print("=" * 60)
        
        # Save final history
        history_path = Path(self.config.log_dir) / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        self.writer.close()
        
        return self.history


def train_model(config: Dict) -> Dict:
    """
    Main training function.
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        Training history
    """
    from ..models.msua_net import create_model
    from ..data.datasets import create_dataloaders
    
    # Create model
    model = create_model(config.get('model', {}), config.get('device', 'cuda'))
    
    # Create dataloaders
    train_loader, val_loader, normalizer = create_dataloaders(config.get('data', {}))
    
    # Create trainer
    training_config = TrainingConfig(config.get('training', {}))
    trainer = BiasCorrectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        loss_config=config.get('loss', {})
    )
    
    # Train
    history = trainer.train()
    
    # Save normalizer
    normalizer.save(str(Path(training_config.checkpoint_dir) / 'normalizer.json'))
    
    return history
