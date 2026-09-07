"""
Training script for Velocity UNet using preprocessed data with magnitude masks.

This script trains the ComplexToRealUNet model using clean, preprocessed velocity
ground truth with magnitude-weighted loss to focus learning on signal regions.

Key Features:
- Uses PreprocessedVelocityDataset with magnitude masks
- MagnitudeWeightedVelocityLoss for clean training
- Gradient accumulation and mixed precision training
- TensorBoard logging and checkpointing

Usage:
    python -m src.training.train_velocity_preprocessed \
        --data_dir data/preprocessed \
        --experiment_name velocity_clean_v1 \
        --num_epochs 100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.architectures.velocity_unet import ComplexToRealUNet
from src.training.velocity_training import MagnitudeWeightedVelocityLoss

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for preprocessed velocity training."""
    
    # Data
    data_dir: str = "data/preprocessed"
    train_split: float = 0.85
    batch_size: int = 2
    num_workers: int = 4
    
    # Model - ComplexToRealUNet parameters
    in_channels: int = 1  # Channels per complex component
    base_channels: int = 32
    encoder_depths: Tuple[int, ...] = (32, 64, 128, 256)
    upsampling_factor: int = 20
    use_motion_interpolation: bool = True
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Targets
    use_loupas_targets: bool = True  # Loupas estimator targets (recommended)
    
    # Loss weights
    l1_weight: float = 1.0
    gdl_weight: float = 0.5
    temporal_weight: float = 0.1
    use_temporal_gradient: bool = True
    
    # Training
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    experiment_name: str = "velocity_preprocessed"
    output_dir: str = "outputs"
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-5
    
    # Random seed
    seed: int = 42


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.should_stop


class PreprocessedVelocityTrainer:
    """
    Trainer for velocity model using preprocessed data with magnitude masks.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Setup
        self._create_output_dir()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        # TensorBoard
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _create_output_dir(self):
        """Create output directory."""
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = self.output_dir / "training.txt"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Logging to: {log_file}")
    
    def _setup_data(self):
        """Setup data loaders using physics-based train/val split."""
        from src.utils.velocity_dataset import create_preprocessed_velocity_dataloaders
        
        # Use physics-based split from train/ and val/ directories
        self.train_loader, self.val_loader = create_preprocessed_velocity_dataloaders(
            preprocessed_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            return_mask=True,
            use_loupas_targets=self.config.use_loupas_targets,
        )
        target_type = 'loupas' if self.config.use_loupas_targets else 'raw kasai'
        logger.info(f"Target type: {target_type}")
    
    def _setup_model(self):
        """Setup model."""
        self.model = ComplexToRealUNet(
            in_channels=self.config.in_channels,
            base_channels=self.config.base_channels,
            encoder_depths=list(self.config.encoder_depths),
            upsampling_factor=self.config.upsampling_factor,
            use_motion_interpolation=self.config.use_motion_interpolation,
        ).to(self.device)
        
        # Log model info
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # OneCycleLR scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_pct = self.config.warmup_epochs / self.config.num_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy='cos',
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
    
    def _setup_loss(self):
        """Setup loss function."""
        self.criterion = MagnitudeWeightedVelocityLoss(
            l1_weight=self.config.l1_weight,
            gdl_weight=self.config.gdl_weight,
            use_temporal_gradient=self.config.use_temporal_gradient,
            temporal_weight=self.config.temporal_weight,
        ).to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        meters = {
            'loss': AverageMeter(),
            'l1_loss': AverageMeter(),
            'gdl_loss': AverageMeter(),
            'mask_coverage': AverageMeter(),
        }
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=False,
        )
        
        for batch_idx, (inputs, targets, masks) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(inputs)
                
                # Match temporal dimensions: model outputs T_dense+1, but velocity is T_dense
                # (because Kasai velocity uses consecutive frame pairs)
                T_target = targets.shape[2]
                if outputs.shape[2] > T_target:
                    outputs = outputs[:, :, :T_target, :, :]
                
                loss, loss_dict = self.criterion(
                    outputs, targets, masks, return_components=True
                )
                
                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update meters
            batch_size = inputs.size(0)
            meters['loss'].update(loss_dict['total_loss'].item(), batch_size)
            meters['l1_loss'].update(loss_dict['l1_loss'].item(), batch_size)
            meters['gdl_loss'].update(loss_dict['gdl_loss'].item(), batch_size)
            meters['mask_coverage'].update(loss_dict['mask_coverage'].item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{meters['loss'].avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        return {k: v.avg for k, v in meters.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        meters = {
            'val_loss': AverageMeter(),
            'val_l1_loss': AverageMeter(),
            'val_gdl_loss': AverageMeter(),
        }
        
        for inputs, targets, masks in tqdm(self.val_loader, desc="Validating", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(inputs)
                
                # Match temporal dimensions
                T_target = targets.shape[2]
                if outputs.shape[2] > T_target:
                    outputs = outputs[:, :, :T_target, :, :]
                
                _, loss_dict = self.criterion(
                    outputs, targets, masks, return_components=True
                )
            
            batch_size = inputs.size(0)
            meters['val_loss'].update(loss_dict['total_loss'].item(), batch_size)
            meters['val_l1_loss'].update(loss_dict['l1_loss'].item(), batch_size)
            meters['val_gdl_loss'].update(loss_dict['gdl_loss'].item(), batch_size)
        
        return {k: v.avg for k, v in meters.items()}
    
    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {self.best_val_loss:.6f}")
    
    def train(self):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("Starting training with preprocessed data")
        logger.info("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Mask Coverage: {train_metrics['mask_coverage']:.3f}"
            )
            
            # TensorBoard logging
            if self.writer is not None:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            self.save_checkpoint("latest.pt", is_best=is_best)
            
            # Early stopping
            if self.early_stopping.step(val_metrics['val_loss']):
                logger.info("Early stopping triggered")
                break
        
        logger.info("=" * 60)
        logger.info(f"Training complete! Best val_loss: {self.best_val_loss:.6f}")
        logger.info("=" * 60)
        
        # Close TensorBoard
        if self.writer is not None:
            self.writer.close()
        
        # Close file logging handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        
        return self.best_val_loss


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train velocity model with preprocessed data")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                       help='Directory with preprocessed data')
    parser.add_argument('--train_split', type=float, default=0.85,
                       help='Fraction of data for training')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels in UNet')
    parser.add_argument('--upsampling_factor', type=int, default=20,
                       help='Temporal upsampling factor')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    
    # Targets
    parser.add_argument('--use_loupas_targets', type=bool, default=True,
                       help='Use Loupas estimator velocity targets (recommended)')
    parser.add_argument('--no_loupas', action='store_true',
                       help='Disable Loupas targets, use raw Kasai instead')
    
    # Loss
    parser.add_argument('--l1_weight', type=float, default=1.0,
                       help='L1 loss weight')
    parser.add_argument('--gdl_weight', type=float, default=0.5,
                       help='GDL loss weight')
    parser.add_argument('--temporal_weight', type=float, default=0.1,
                       help='Temporal gradient loss weight')
    
    # Output
    parser.add_argument('--experiment_name', type=str, default='velocity_preprocessed_v1',
                       help='Experiment name for output directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Base output directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        train_split=args.train_split,
        use_loupas_targets=not args.no_loupas,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        upsampling_factor=args.upsampling_factor,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        l1_weight=args.l1_weight,
        gdl_weight=args.gdl_weight,
        temporal_weight=args.temporal_weight,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Train
    trainer = PreprocessedVelocityTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
