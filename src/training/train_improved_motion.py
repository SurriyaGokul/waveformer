"""
Training script for Improved Motion-Aware PI-CV-RecUNet.

Key improvements:
1. Residual motion interpolation (linear baseline + motion correction)
2. Reduced flow loss weights (0.001 instead of 0.01)
3. Flow smoothness regularization
4. Better monitoring of motion weights
5. Optional: disable flow losses for pure reconstruction focus

Author: GitHub Copilot
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# Local imports
from src.utils.loss_functions import PhysicsInformedLoss, SupervisedOnlyLoss
from src.utils.data_loader import SWEDataModule, DataConfig, ComplexNormalizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class ImprovedMotionAwareConfig:
    """Training configuration for Improved Motion-Aware PI-CV-RecUNet."""
    
    def __init__(self):
        # Data settings
        self.data_dir: str = "data/CIRS_phantom_dataset"
        self.output_dir: str = "outputs"
        self.experiment_name: str = "improved_motion_aware"
        
        # Model settings
        self.in_channels: int = 1
        self.out_channels: int = 1
        self.base_channels: int = 32
        self.num_levels: int = 4
        self.lstm_hidden_channels: int = 128
        self.lstm_num_layers: int = 1
        self.upsampling_factor: int = 20
        
        # Motion-aware specific settings (IMPROVED)
        self.flow_hidden_channels: int = 64
        self.flow_scale: float = 0.5  # Reduced from 1.0 for stability
        self.use_residual_fusion: bool = True  # Key improvement
        self.flow_upsample_factor: int = 4  # Estimate flow at 4x resolution (8x8 -> 32x32)
        
        # Flow regularization losses (REDUCED weights)
        self.use_flow_losses: bool = True
        self.use_flow_consistency_loss: bool = True
        self.use_photometric_loss: bool = True
        self.use_flow_smoothness_loss: bool = True  # NEW
        self.lambda_flow_consistency: float = 0.001  # Reduced from 0.01
        self.lambda_photometric: float = 0.001  # Reduced from 0.01
        self.lambda_flow_smoothness: float = 0.0005  # NEW
        self.flow_loss_warmup_epochs: int = 10  # Increased from 5
        
        # Training settings
        self.epochs: int = 100
        self.batch_size: int = 2
        self.learning_rate: float = 5e-4
        self.weight_decay: float = 1e-5
        self.gradient_clip: float = 50.0
        
        # Learning rate scheduler
        self.lr_scheduler: str = "cosine"
        self.lr_step_size: int = 30
        self.lr_gamma: float = 0.5
        self.lr_patience: int = 10
        self.min_lr: float = 1e-7
        
        # Loss settings
        self.lambda_data: float = 1.0
        self.lambda_physics: float = 0.0
        self.lambda_physics_target: float = 0.1
        self.lambda_smooth: float = 0.01
        self.lambda_magnitude: float = 0.5
        self.use_physics_loss: bool = True
        self.physics_warmup_epochs: int = 10
        self.normalize_physics_loss: bool = True
        self.physics_lr_multiplier: float = 0.1
        
        # Data settings
        self.sparse_frame_step: int = 20
        self.normalization: str = "per_sample"
        self.train_ratio: float = 0.7
        self.val_ratio: float = 0.15
        self.test_ratio: float = 0.15
        self.use_augmentation: bool = False
        self.num_workers: int = 4
        
        # Training optimization
        self.use_amp: bool = True
        self.accumulation_steps: int = 1
        
        # Checkpointing
        self.save_every: int = 10
        self.validate_every: int = 1
        self.log_every: int = 10
        
        # Early stopping
        self.early_stopping: bool = True
        self.patience: int = 20
        
        # Device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Random seed
        self.seed: int = 42
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ImprovedMotionAwareConfig':
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_json(cls, path: str) -> 'ImprovedMotionAwareConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ImprovedMetrics:
    """Tracking metrics including motion-specific stats."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.data_loss = 0.0
        self.physics_loss = 0.0
        self.smooth_loss = 0.0
        self.magnitude_loss = 0.0
        self.flow_consistency_loss = 0.0
        self.photometric_loss = 0.0
        self.flow_smoothness_loss = 0.0
        self.motion_weight_mean = 0.0
        self.count = 0
        self.psnr_sum = 0.0
    
    def update(self, loss_components: Dict, batch_size: int = 1):
        self.total_loss += loss_components.get('total', 0.0) * batch_size
        self.data_loss += loss_components.get('data', 0.0) * batch_size
        self.physics_loss += loss_components.get('physics', 0.0) * batch_size
        self.smooth_loss += loss_components.get('smoothness', 0.0) * batch_size
        self.magnitude_loss += loss_components.get('magnitude', 0.0) * batch_size
        self.flow_consistency_loss += loss_components.get('flow_consistency', 0.0) * batch_size
        self.photometric_loss += loss_components.get('photometric', 0.0) * batch_size
        self.flow_smoothness_loss += loss_components.get('flow_smoothness', 0.0) * batch_size
        self.motion_weight_mean += loss_components.get('motion_weight_mean', 0.0) * batch_size
        self.count += batch_size
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_mag = torch.sqrt(pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2)
        target_mag = torch.sqrt(target[:, :, 0] ** 2 + target[:, :, 1] ** 2)
        
        mse = torch.mean((pred_mag - target_mag) ** 2)
        if mse == 0:
            return float('inf')
        
        max_val = torch.max(target_mag)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    def add_psnr(self, psnr: float, batch_size: int = 1):
        self.psnr_sum += psnr * batch_size
    
    def get_averages(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        
        return {
            'total_loss': self.total_loss / self.count,
            'data_loss': self.data_loss / self.count,
            'physics_loss': self.physics_loss / self.count,
            'smooth_loss': self.smooth_loss / self.count,
            'magnitude_loss': self.magnitude_loss / self.count,
            'flow_consistency_loss': self.flow_consistency_loss / self.count,
            'photometric_loss': self.photometric_loss / self.count,
            'flow_smoothness_loss': self.flow_smoothness_loss / self.count,
            'motion_weight_mean': self.motion_weight_mean / self.count,
            'psnr': self.psnr_sum / self.count if self.psnr_sum > 0 else 0.0
        }


def create_improved_motion_model(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 4,
    lstm_hidden_channels: int = 128,
    lstm_num_layers: int = 1,
    upsampling_factor: int = 20,
    flow_hidden_channels: int = 64,
    flow_scale: float = 0.5,
    use_residual_fusion: bool = True,
    flow_upsample_factor: int = 4
) -> nn.Module:
    """
    Create an Improved Motion-Aware PI-CV-RecUNet model.
    
    Uses ResidualMotionInterpolation with adaptive fusion.
    
    Args:
        flow_upsample_factor: Factor to upsample features before flow estimation.
            Higher values (e.g., 4) allow finer motion capture at higher resolution.
    """
    from src.architectures.pinn_recunet import Encoder, Decoder
    from src.architectures.improved_motion_interpolation import ImprovedMotionAwareTemporalBottleneck
    from src.architectures.complex_modules import ComplexConv2d as CC2d
    
    class ImprovedMotionAware_PI_CV_RecUNet(nn.Module):
        """
        Improved Motion-Aware Physics-Informed Complex-Valued Recurrent U-Net.
        
        Key improvements:
        1. Residual motion interpolation (linear + learned correction)
        2. Adaptive per-pixel fusion
        3. Flow magnitude constraints
        4. Better initialization for stability
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            base_channels: int,
            num_levels: int,
            lstm_hidden_channels: int,
            lstm_num_layers: int,
            upsampling_factor: int,
            flow_hidden_channels: int,
            flow_scale: float,
            use_residual_fusion: bool,
            flow_upsample_factor: int = 4
        ):
            super(ImprovedMotionAware_PI_CV_RecUNet, self).__init__()
            
            self.upsampling_factor = upsampling_factor
            
            # Encoder
            self.encoder = Encoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_levels=num_levels
            )
            
            encoder_out_channels = self.encoder.output_channels
            
            # Improved Motion-Aware Temporal Bottleneck
            self.temporal_bottleneck = ImprovedMotionAwareTemporalBottleneck(
                in_channels=encoder_out_channels,
                hidden_channels=lstm_hidden_channels,
                num_layers=lstm_num_layers,
                kernel_size=3,
                upsampling_factor=upsampling_factor,
                flow_hidden_channels=flow_hidden_channels,
                flow_scale=flow_scale,
                use_residual_fusion=use_residual_fusion,
                flow_upsample_factor=flow_upsample_factor
            )
            
            # Decoder
            self.decoder = Decoder(
                in_channels=encoder_out_channels,
                base_channels=base_channels,
                num_levels=num_levels
            )
            
            # Output head
            self.output_head = CC2d(
                base_channels,
                out_channels,
                kernel_size=1,
                padding=0
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input sparse frames (B, T_sparse, 2, C, H, W)
            
            Returns:
                output: Dense frames (B, T_dense, 2, C, H, W)
            """
            batch_size, t_sparse, _, channels, height, width = x.shape
            
            # Encode each sparse frame
            encoded_frames = []
            skip_features_list = []
            
            for t in range(t_sparse):
                frame = x[:, t, :, :, :, :]
                encoded, skips = self.encoder(frame)
                encoded_frames.append(encoded)
                skip_features_list.append(skips)
            
            # Stack encoded frames
            encoded_stack = torch.stack(encoded_frames, dim=1)
            
            # Motion-aware temporal interpolation
            dense_encoded = self.temporal_bottleneck(encoded_stack)
            
            t_dense = dense_encoded.shape[1]
            
            # Interpolate skip connections (trilinear - fast and good for smooth data)
            interpolated_skips = self._interpolate_skip_connections(
                skip_features_list, t_dense
            )
            
            # Decode each dense frame
            outputs = []
            for t in range(t_dense):
                encoded_frame = dense_encoded[:, t, :, :, :, :]
                skips = [s[:, t] for s in interpolated_skips]
                
                decoded = self.decoder(encoded_frame, skips)
                output = self.output_head(decoded)
                outputs.append(output)
            
            output = torch.stack(outputs, dim=1)
            
            return output
        
        def _interpolate_skip_connections(self, skip_features_list, t_dense: int):
            """Interpolate skip connections using trilinear interpolation."""
            t_sparse = len(skip_features_list)
            num_levels = len(skip_features_list[0])
            
            stacked_skips = []
            for level in range(num_levels):
                level_skips = torch.stack(
                    [skip_features_list[t][level] for t in range(t_sparse)],
                    dim=1
                )
                stacked_skips.append(level_skips)
            
            interpolated_skips = []
            for level_skips in stacked_skips:
                B, T, two, C, H, W = level_skips.shape
                
                level_skips_flat = level_skips.permute(0, 2, 3, 1, 4, 5).reshape(
                    B * 2 * C, T, H, W
                )
                
                level_skips_interp = torch.nn.functional.interpolate(
                    level_skips_flat.unsqueeze(1),
                    size=(t_dense, H, W),
                    mode='trilinear',
                    align_corners=True
                ).squeeze(1)
                
                level_skips_interp = level_skips_interp.reshape(
                    B, 2, C, t_dense, H, W
                ).permute(0, 3, 1, 2, 4, 5)
                
                interpolated_skips.append(level_skips_interp)
            
            return interpolated_skips
        
        def get_flow_estimator(self):
            """Get the flow estimator."""
            return self.temporal_bottleneck.get_flow_estimator()
        
        def get_motion_interpolation(self):
            """Get the motion interpolation module."""
            return self.temporal_bottleneck.get_motion_interpolation()
        
        def get_input_projection(self):
            """Get the input projection layer."""
            return self.temporal_bottleneck.input_proj
        
        def get_projected_features(self, sparse_encoded: torch.Tensor) -> torch.Tensor:
            """Project encoder features to hidden space."""
            input_proj = self.get_input_projection()
            if input_proj is None:
                return sparse_encoded
            
            B, T, _, C, H, W = sparse_encoded.shape
            sparse_flat = sparse_encoded.view(-1, 2, C, H, W)
            projected_flat = input_proj(sparse_flat)
            hidden_channels = projected_flat.shape[2]
            projected = projected_flat.view(B, T, 2, hidden_channels, H, W)
            return projected
    
    model = ImprovedMotionAware_PI_CV_RecUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        lstm_hidden_channels=lstm_hidden_channels,
        lstm_num_layers=lstm_num_layers,
        upsampling_factor=upsampling_factor,
        flow_hidden_channels=flow_hidden_channels,
        flow_scale=flow_scale,
        use_residual_fusion=use_residual_fusion,
        flow_upsample_factor=flow_upsample_factor
    )
    
    return model


class ImprovedMotionAwareTrainer:
    """Training orchestrator for Improved Motion-Aware PI-CV-RecUNet."""
    
    def __init__(self, config: ImprovedMotionAwareConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seeds()
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for logging
        file_handler = logging.FileHandler(self.output_dir / "training.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Save config
        config.save(self.output_dir / "config.json")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.data_module = None
        
        # TensorBoard writer
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
            logger.info("TensorBoard not available.")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _set_seeds(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup(self):
        """Set up all training components."""
        logger.info("Setting up Improved Motion-Aware training...")
        
        self._setup_data()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        
        if self.config.use_amp:
            self.scaler = GradScaler()
        
        logger.info("Setup complete!")
        self._log_setup_info()
    
    def _setup_data(self):
        data_config = DataConfig(
            data_dir=self.config.data_dir,
            sparse_frame_step=self.config.sparse_frame_step,
            normalization=self.config.normalization,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            use_augmentation=self.config.use_augmentation,
            num_workers=self.config.num_workers,
            seed=self.config.seed
        )
        
        self.data_module = SWEDataModule(data_config)
        self.data_module.setup()
        
        self.train_loader = self.data_module.get_train_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.val_loader = self.data_module.get_val_dataloader(
            batch_size=self.config.batch_size
        )
    
    def _setup_model(self):
        self.model = create_improved_motion_model(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            num_levels=self.config.num_levels,
            lstm_hidden_channels=self.config.lstm_hidden_channels,
            lstm_num_layers=self.config.lstm_num_layers,
            upsampling_factor=self.config.upsampling_factor,
            flow_hidden_channels=self.config.flow_hidden_channels,
            flow_scale=self.config.flow_scale,
            use_residual_fusion=self.config.use_residual_fusion,
            flow_upsample_factor=self.config.flow_upsample_factor
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        flow_params = sum(p.numel() for p in self.model.get_flow_estimator().parameters())
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"  Flow estimator: {flow_params:,} parameters")
    
    def _setup_optimizer(self):
        param_groups = [
            {'params': self.model.parameters(), 'lr': self.config.learning_rate},
        ]
        
        if hasattr(self.criterion, 'parameters'):
            criterion_params = list(self.criterion.parameters())
            if criterion_params:
                param_groups.append({
                    'params': criterion_params,
                    'lr': self.config.learning_rate * self.config.physics_lr_multiplier,
                    'weight_decay': 0.0
                })
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_gamma,
                patience=self.config.lr_patience,
                min_lr=self.config.min_lr
            )
        else:
            self.scheduler = None
    
    def _setup_criterion(self):
        if self.config.use_physics_loss:
            self.criterion = PhysicsInformedLoss(
                lambda_data=self.config.lambda_data,
                lambda_physics=self.config.lambda_physics,
                lambda_smooth=self.config.lambda_smooth,
                lambda_magnitude=self.config.lambda_magnitude,
                normalize_physics=self.config.normalize_physics_loss
            )
        else:
            self.criterion = SupervisedOnlyLoss(
                lambda_data=self.config.lambda_data,
                lambda_smooth=self.config.lambda_smooth
            )
        
        self.criterion = self.criterion.to(self.device)
    
    def _log_setup_info(self):
        logger.info("=" * 60)
        logger.info("Improved Motion-Aware Training Configuration:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Train samples: {len(self.data_module.train_dataset)}")
        logger.info(f"  Val samples: {len(self.data_module.val_dataset)}")
        logger.info("Motion-Aware Settings:")
        logger.info(f"  Flow hidden channels: {self.config.flow_hidden_channels}")
        logger.info(f"  Flow scale: {self.config.flow_scale}")
        logger.info(f"  Use residual fusion: {self.config.use_residual_fusion}")
        logger.info(f"  Flow losses enabled: {self.config.use_flow_losses}")
        if self.config.use_flow_losses:
            logger.info(f"    Consistency weight: {self.config.lambda_flow_consistency}")
            logger.info(f"    Photometric weight: {self.config.lambda_photometric}")
            logger.info(f"    Smoothness weight: {self.config.lambda_flow_smoothness}")
            logger.info(f"    Warmup epochs: {self.config.flow_loss_warmup_epochs}")
        logger.info(f"  Physics loss: {self.config.use_physics_loss}")
        logger.info(f"  Mixed precision: {self.config.use_amp}")
        logger.info("=" * 60)
    
    def _compute_flow_losses(
        self,
        sparse_encoded: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute flow regularization losses with improved stability."""
        from src.architectures.improved_motion_interpolation import (
            compute_flow_consistency_loss_improved,
            compute_photometric_loss_improved,
            compute_flow_smoothness_loss
        )
        
        components = {
            'flow_consistency': 0.0,
            'photometric': 0.0,
            'flow_smoothness': 0.0,
            'motion_weight_mean': 0.0
        }
        
        # Check if flow losses are enabled
        if not self.config.use_flow_losses:
            return torch.tensor(0.0, device=self.device), components
        
        # Check warmup
        if epoch < self.config.flow_loss_warmup_epochs:
            return torch.tensor(0.0, device=self.device), components
        
        # Get flow estimator and motion interpolation
        flow_estimator = self.model.get_flow_estimator()
        motion_interp = self.model.get_motion_interpolation()
        
        B, T_sparse, _, C, H, W = sparse_encoded.shape
        
        total_consistency = 0.0
        total_photometric = 0.0
        total_smoothness = 0.0
        count = 0
        
        for t in range(T_sparse - 1):
            frame_t = sparse_encoded[:, t]
            frame_t1 = sparse_encoded[:, t + 1]
            
            # Get flows
            forward_flow, backward_flow = flow_estimator(frame_t, frame_t1)
            
            # Flow consistency loss
            if self.config.use_flow_consistency_loss:
                consistency = compute_flow_consistency_loss_improved(
                    forward_flow, backward_flow
                )
                total_consistency += consistency
            
            # Photometric loss
            if self.config.use_photometric_loss:
                photometric = compute_photometric_loss_improved(
                    frame_t, frame_t1, forward_flow
                )
                total_photometric += photometric
            
            # Flow smoothness loss
            if self.config.use_flow_smoothness_loss:
                smoothness = compute_flow_smoothness_loss(forward_flow)
                smoothness += compute_flow_smoothness_loss(backward_flow)
                total_smoothness += smoothness / 2
            
            count += 1
        
        if count > 0:
            total_consistency /= count
            total_photometric /= count
            total_smoothness /= count
        
        # Get motion weight stats
        motion_stats = motion_interp.get_motion_weight_stats()
        
        components['flow_consistency'] = total_consistency.item() if torch.is_tensor(total_consistency) else total_consistency
        components['photometric'] = total_photometric.item() if torch.is_tensor(total_photometric) else total_photometric
        components['flow_smoothness'] = total_smoothness.item() if torch.is_tensor(total_smoothness) else total_smoothness
        components['motion_weight_mean'] = motion_stats.get('motion_weight_mean', 0.0)
        
        # Combine with weights
        total_flow_loss = (
            self.config.lambda_flow_consistency * total_consistency +
            self.config.lambda_photometric * total_photometric +
            self.config.lambda_flow_smoothness * total_smoothness
        )
        
        return total_flow_loss, components
    
    def _get_physics_weight(self, epoch: int) -> float:
        """Get physics loss weight with warmup."""
        if epoch < self.config.physics_warmup_epochs:
            return 0.0
        
        warmup_epochs_remaining = min(
            10,
            self.config.epochs - self.config.physics_warmup_epochs
        )
        warmup_progress = min(
            1.0,
            (epoch - self.config.physics_warmup_epochs) / warmup_epochs_remaining
        )
        
        return self.config.lambda_physics_target * warmup_progress
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = ImprovedMetrics()
        
        # Update physics weight
        if hasattr(self.criterion, 'lambda_physics'):
            self.criterion.lambda_physics = self._get_physics_weight(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            sparse = batch['sparse'].to(self.device)
            dense = batch['dense'].to(self.device)
            
            batch_size = sparse.shape[0]
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    pred = self.model(sparse)
                    loss, components = self.criterion(pred, dense, return_components=True)
                
                # Flow losses (separate for stability)
                with autocast():
                    encoded_frames = []
                    t_sparse = sparse.shape[1]
                    for t in range(t_sparse):
                        frame = sparse[:, t]
                        encoded, _ = self.model.encoder(frame)
                        encoded_frames.append(encoded)
                    sparse_encoded = torch.stack(encoded_frames, dim=1)
                    sparse_encoded = self.model.get_projected_features(sparse_encoded)
                    
                    flow_loss, flow_components = self._compute_flow_losses(
                        sparse_encoded, self.current_epoch
                    )
                    
                    total_loss = loss + flow_loss
                    total_loss = total_loss / self.config.accumulation_steps
                
                self.scaler.scale(total_loss).backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                pred = self.model(sparse)
                loss, components = self.criterion(pred, dense, return_components=True)
                
                encoded_frames = []
                t_sparse = sparse.shape[1]
                for t in range(t_sparse):
                    frame = sparse[:, t]
                    encoded, _ = self.model.encoder(frame)
                    encoded_frames.append(encoded)
                sparse_encoded = torch.stack(encoded_frames, dim=1)
                sparse_encoded = self.model.get_projected_features(sparse_encoded)
                
                flow_loss, flow_components = self._compute_flow_losses(
                    sparse_encoded, self.current_epoch
                )
                
                total_loss = (loss + flow_loss) / self.config.accumulation_steps
                total_loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
            
            # Update metrics
            components['total'] = total_loss.item() * self.config.accumulation_steps
            components.update(flow_components)
            metrics.update(components, batch_size)
            
            # Compute PSNR
            with torch.no_grad():
                psnr = metrics.compute_psnr(pred.detach(), dense)
                metrics.add_psnr(psnr, batch_size)
            
            self.global_step += 1
            
            # Log batch
            if batch_idx % self.config.log_every == 0:
                avg = metrics.get_averages()
                logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={avg.get('total_loss', 0):.4f} "
                    f"data={avg.get('data_loss', 0):.4f} "
                    f"motion_wt={avg.get('motion_weight_mean', 0):.3f}"
                )
        
        return metrics.get_averages()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics = ImprovedMetrics()
        
        for batch in self.val_loader:
            sparse = batch['sparse'].to(self.device)
            dense = batch['dense'].to(self.device)
            
            batch_size = sparse.shape[0]
            
            if self.config.use_amp:
                with autocast():
                    pred = self.model(sparse)
                    loss, components = self.criterion(pred, dense, return_components=True)
            else:
                pred = self.model(sparse)
                loss, components = self.criterion(pred, dense, return_components=True)
            
            components['total'] = loss.item()
            metrics.update(components, batch_size)
            
            psnr = metrics.compute_psnr(pred, dense)
            metrics.add_psnr(psnr, batch_size)
        
        return metrics.get_averages()
    
    def train(self) -> Dict[str, float]:
        """Full training loop."""
        logger.info("Starting training...")
        
        best_metrics = {}
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate()
                
                # Log
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics.get('total_loss', 0):.4f} "
                    f"val_loss={val_metrics.get('total_loss', 0):.4f} "
                    f"val_psnr={val_metrics.get('psnr', 0):.2f} dB "
                    f"train_data_loss={train_metrics.get('data_loss', 0):.4f}"
                )
                
                # TensorBoard logging
                if self.writer:
                    for k, v in train_metrics.items():
                        self.writer.add_scalar(f'train/{k}', v, epoch)
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f'val/{k}', v, epoch)
                
                # Check for best model
                val_loss = val_metrics.get('total_loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_metrics = val_metrics
                    self.patience_counter = 0
                    
                    # Save best model
                    self._save_checkpoint('best_model.pt', val_metrics)
                    logger.info(f"  New best model! val_loss={val_loss:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.config.early_stopping and self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('total_loss', 0))
                else:
                    self.scheduler.step()
            
            # Periodic save
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', train_metrics)
        
        # Final save
        self._save_checkpoint('final_model.pt', train_metrics)
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return best_metrics
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def test(self) -> Dict[str, float]:
        """Test the model on test set."""
        logger.info("Running test evaluation...")
        
        # Load best model
        best_path = self.output_dir / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for testing")
        
        self.model.eval()
        metrics = ImprovedMetrics()
        
        test_loader = self.data_module.get_test_dataloader(
            batch_size=self.config.batch_size
        )
        
        with torch.no_grad():
            for batch in test_loader:
                sparse = batch['sparse'].to(self.device)
                dense = batch['dense'].to(self.device)
                
                batch_size = sparse.shape[0]
                
                if self.config.use_amp:
                    with autocast():
                        pred = self.model(sparse)
                        loss, components = self.criterion(pred, dense, return_components=True)
                else:
                    pred = self.model(sparse)
                    loss, components = self.criterion(pred, dense, return_components=True)
                
                components['total'] = loss.item()
                metrics.update(components, batch_size)
                
                psnr = metrics.compute_psnr(pred, dense)
                metrics.add_psnr(psnr, batch_size)
        
        test_metrics = metrics.get_averages()
        
        logger.info("Test Results:")
        logger.info(f"  Total Loss: {test_metrics.get('total_loss', 0):.4f}")
        logger.info(f"  Data Loss: {test_metrics.get('data_loss', 0):.4f}")
        logger.info(f"  PSNR: {test_metrics.get('psnr', 0):.2f} dB")
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Improved Motion-Aware PI-CV-RecUNet')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--no_flow_losses', action='store_true', help='Disable flow losses')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ImprovedMotionAwareConfig.from_json(args.config)
    else:
        config = ImprovedMotionAwareConfig()
    
    # Override with command line args
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_flow_losses:
        config.use_flow_losses = False
    
    # Create trainer and run
    trainer = ImprovedMotionAwareTrainer(config)
    trainer.setup()
    
    try:
        trainer.train()
        trainer.test()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if trainer.writer:
            trainer.writer.close()


if __name__ == '__main__':
    main()
