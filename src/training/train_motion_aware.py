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


class MotionAwareTrainConfig:
    """Training configuration for Motion-Aware PI-CV-RecUNet."""
    
    def __init__(self):
        # Data settings
        self.data_dir: str = "data/CIRS_phantom_dataset"
        self.output_dir: str = "outputs"
        self.experiment_name: str = "motion_aware_recunet"
        
        # Model settings
        self.in_channels: int = 1
        self.out_channels: int = 1
        self.base_channels: int = 32
        self.num_levels: int = 4
        self.lstm_hidden_channels: int = 128
        self.lstm_num_layers: int = 1
        self.upsampling_factor: int = 20
        
        # Motion-aware specific settings
        self.flow_hidden_channels: int = 64
        self.flow_scale: float = 1.0
        self.use_learned_fusion: bool = True
        
        # Flow regularization losses
        self.use_flow_consistency_loss: bool = True
        self.use_photometric_loss: bool = True
        self.lambda_flow_consistency: float = 0.01
        self.lambda_photometric: float = 0.01
        self.flow_loss_warmup_epochs: int = 5  # Epochs before applying flow losses
        
        # Training settings
        self.epochs: int = 100
        self.batch_size: int = 2
        self.learning_rate: float = 5e-4
        self.weight_decay: float = 1e-5
        self.gradient_clip: float = 50.0
        
        # Learning rate scheduler
        self.lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
        self.lr_step_size: int = 30
        self.lr_gamma: float = 0.5
        self.lr_patience: int = 10
        self.min_lr: float = 1e-7
        
        # Loss settings
        self.lambda_data: float = 1.0
        self.lambda_physics: float = 0.0  # Start at 0, warmup later
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
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MotionAwareTrainConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_json(cls, path: str) -> 'MotionAwareTrainConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MotionAwareMetrics:
    """Tracking and computation of metrics including flow-based losses."""
    
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
        self.count = 0
        self.psnr_sum = 0.0
    
    def update(self, loss_components: Dict, batch_size: int = 1):
        """Update metrics with batch results."""
        self.total_loss += loss_components.get('total', 0.0) * batch_size
        self.data_loss += loss_components.get('data', 0.0) * batch_size
        self.physics_loss += loss_components.get('physics', 0.0) * batch_size
        self.smooth_loss += loss_components.get('smoothness', 0.0) * batch_size
        self.magnitude_loss += loss_components.get('magnitude', 0.0) * batch_size
        self.flow_consistency_loss += loss_components.get('flow_consistency', 0.0) * batch_size
        self.photometric_loss += loss_components.get('photometric', 0.0) * batch_size
        self.count += batch_size
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR for complex data (based on magnitude)."""
        pred_mag = torch.sqrt(pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2)
        target_mag = torch.sqrt(target[:, :, 0] ** 2 + target[:, :, 1] ** 2)
        
        mse = torch.mean((pred_mag - target_mag) ** 2)
        if mse == 0:
            return float('inf')
        
        max_val = torch.max(target_mag)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    def add_psnr(self, psnr: float, batch_size: int = 1):
        """Add PSNR measurement."""
        self.psnr_sum += psnr * batch_size
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics."""
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
            'psnr': self.psnr_sum / self.count if self.psnr_sum > 0 else 0.0
        }


def create_motion_aware_model(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 4,
    lstm_hidden_channels: int = 128,
    lstm_num_layers: int = 1,
    upsampling_factor: int = 20,
    flow_hidden_channels: int = 64,
    flow_scale: float = 1.0,
    use_learned_fusion: bool = True
) -> nn.Module:
    """
    Create a Motion-Aware PI-CV-RecUNet model.
    
    This function creates the model by replacing the standard TemporalBottleneck
    with the MotionAwareTemporalBottleneck.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels (doubled at each level)
        num_levels: Number of encoder/decoder levels
        lstm_hidden_channels: Hidden channels in LSTM bottleneck
        lstm_num_layers: Number of LSTM layers
        upsampling_factor: Temporal upsampling factor
        flow_hidden_channels: Hidden channels in flow estimator
        flow_scale: Scaling factor for flow output
        use_learned_fusion: Whether to use learned fusion weights
    
    Returns:
        Motion-aware PI-CV-RecUNet model
    """
    from src.architectures.pinn_recunet import Encoder, Decoder, ComplexConvBlock, ComplexConv2d
    from src.architectures.complex_motion_interpolation import MotionAwareTemporalBottleneck
    from src.architectures.complex_modules import ComplexConv2d as CC2d
    
    class MotionAware_PI_CV_RecUNet(nn.Module):
        """
        Motion-Aware Physics-Informed Complex-Valued Recurrent U-Net.
        
        Uses optical flow-based motion interpolation instead of linear
        interpolation for temporal super-resolution.
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
            use_learned_fusion: bool
        ):
            super(MotionAware_PI_CV_RecUNet, self).__init__()
            
            self.upsampling_factor = upsampling_factor
            
            # Encoder
            self.encoder = Encoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_levels=num_levels
            )
            
            encoder_out_channels = self.encoder.output_channels
            
            # Motion-Aware Temporal Bottleneck
            self.temporal_bottleneck = MotionAwareTemporalBottleneck(
                in_channels=encoder_out_channels,
                hidden_channels=lstm_hidden_channels,
                num_layers=lstm_num_layers,
                kernel_size=3,
                upsampling_factor=upsampling_factor,
                flow_hidden_channels=flow_hidden_channels,
                flow_scale=flow_scale,
                use_learned_fusion=use_learned_fusion
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
                frame = x[:, t, :, :, :, :]  # (B, 2, C, H, W)
                encoded, skips = self.encoder(frame)
                encoded_frames.append(encoded)
                skip_features_list.append(skips)
            
            # Stack encoded frames: (B, T_sparse, 2, C_enc, H_enc, W_enc)
            encoded_stack = torch.stack(encoded_frames, dim=1)
            
            # Motion-aware temporal interpolation
            dense_encoded = self.temporal_bottleneck(encoded_stack)
            
            t_dense = dense_encoded.shape[1]
            
            # Interpolate skip connections
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
            
            # Stack outputs: (B, T_dense, 2, C_out, H, W)
            output = torch.stack(outputs, dim=1)
            
            return output
        
        def _interpolate_skip_connections(
            self,
            skip_features_list,
            t_dense: int
        ):
            """
            Interpolate skip connections from sparse to dense time steps.
            
            Uses linear interpolation for skip connections (simpler than
            motion interpolation since these are at different spatial scales).
            """
            t_sparse = len(skip_features_list)
            num_levels = len(skip_features_list[0])
            
            # Stack skips for each level: (B, T_sparse, 2, C, H, W)
            stacked_skips = []
            for level in range(num_levels):
                level_skips = torch.stack(
                    [skip_features_list[t][level] for t in range(t_sparse)],
                    dim=1
                )
                stacked_skips.append(level_skips)
            
            # Interpolate each level
            interpolated_skips = []
            for level_skips in stacked_skips:
                B, T, two, C, H, W = level_skips.shape
                
                # Reshape for interpolation: (B*2*C, T, H, W) -> (B*2*C, T_dense, H, W)
                level_skips_flat = level_skips.permute(0, 2, 3, 1, 4, 5).reshape(
                    B * 2 * C, T, H, W
                )
                
                # Interpolate along temporal dimension
                level_skips_interp = torch.nn.functional.interpolate(
                    level_skips_flat.unsqueeze(1),  # (B*2*C, 1, T, H, W)
                    size=(t_dense, H, W),
                    mode='trilinear',
                    align_corners=True
                ).squeeze(1)  # (B*2*C, T_dense, H, W)
                
                # Reshape back: (B, T_dense, 2, C, H, W)
                level_skips_interp = level_skips_interp.reshape(
                    B, 2, C, t_dense, H, W
                ).permute(0, 3, 1, 2, 4, 5)
                
                interpolated_skips.append(level_skips_interp)
            
            return interpolated_skips
        
        def get_flow_estimator(self):
            """Get the flow estimator for computing flow regularization losses."""
            return self.temporal_bottleneck.motion_interpolation.flow_estimator
        
        def get_motion_interpolation(self):
            """Get the motion interpolation module."""
            return self.temporal_bottleneck.motion_interpolation
        
        def get_input_projection(self):
            """Get the input projection layer from the temporal bottleneck."""
            return self.temporal_bottleneck.input_proj
        
        def get_projected_features(self, sparse_encoded: torch.Tensor) -> torch.Tensor:
            """
            Project encoder features to hidden space for flow loss computation.
            
            Args:
                sparse_encoded: Encoder output (B, T_sparse, 2, C_enc, H, W)
            
            Returns:
                projected: Projected features (B, T_sparse, 2, C_hidden, H, W)
            """
            input_proj = self.get_input_projection()
            if input_proj is None:
                return sparse_encoded
            
            B, T, _, C, H, W = sparse_encoded.shape
            # Reshape for projection
            sparse_flat = sparse_encoded.view(-1, 2, C, H, W)
            projected_flat = input_proj(sparse_flat)
            hidden_channels = projected_flat.shape[2]
            projected = projected_flat.view(B, T, 2, hidden_channels, H, W)
            return projected
    
    # Create and return the model
    model = MotionAware_PI_CV_RecUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        lstm_hidden_channels=lstm_hidden_channels,
        lstm_num_layers=lstm_num_layers,
        upsampling_factor=upsampling_factor,
        flow_hidden_channels=flow_hidden_channels,
        flow_scale=flow_scale,
        use_learned_fusion=use_learned_fusion
    )
    
    return model


class MotionAwareTrainer:
    """Training orchestrator for Motion-Aware PI-CV-RecUNet."""
    
    def __init__(self, config: MotionAwareTrainConfig):
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
        
        # TensorBoard writer (optional)
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
            logger.info("TensorBoard not available. Install with: pip install tensorboard")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup(self):
        """Set up all training components."""
        logger.info("Setting up Motion-Aware training...")
        
        # Data
        self._setup_data()
        
        # Model
        self._setup_model()
        
        # Loss
        self._setup_criterion()
        
        # Optimizer and scheduler
        self._setup_optimizer()
        
        # Mixed precision
        if self.config.use_amp:
            self.scaler = GradScaler()
        
        logger.info("Setup complete!")
        self._log_setup_info()
    
    def _setup_data(self):
        """Set up data loading."""
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
        """Set up motion-aware model."""
        self.model = create_motion_aware_model(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            num_levels=self.config.num_levels,
            lstm_hidden_channels=self.config.lstm_hidden_channels,
            lstm_num_layers=self.config.lstm_num_layers,
            upsampling_factor=self.config.upsampling_factor,
            flow_hidden_channels=self.config.flow_hidden_channels,
            flow_scale=self.config.flow_scale,
            use_learned_fusion=self.config.use_learned_fusion
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Count flow-specific parameters
        flow_params = sum(
            p.numel() for p in self.model.get_flow_estimator().parameters()
        )
        fusion_params = sum(
            p.numel() for p in self.model.get_motion_interpolation().fusion_net.parameters()
        )
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"  Flow estimator: {flow_params:,} parameters")
        logger.info(f"  Fusion network: {fusion_params:,} parameters")
    
    def _setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        # Separate parameter groups
        model_params = list(self.model.parameters())
        
        physics_params = []
        physics_lr_mult = getattr(self.config, 'physics_lr_multiplier', 0.1)
        
        if hasattr(self.criterion, 'parameters'):
            criterion_params = list(self.criterion.parameters())
            if criterion_params:
                physics_params = criterion_params
                logger.info(f"Including {len(criterion_params)} learnable physics parameters")
        
        param_groups = [
            {'params': model_params, 'lr': self.config.learning_rate},
        ]
        
        if physics_params:
            param_groups.append({
                'params': physics_params,
                'lr': self.config.learning_rate * physics_lr_mult,
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
        """Set up loss function."""
        if self.config.use_physics_loss:
            normalize_physics = getattr(self.config, 'normalize_physics_loss', True)
            self.criterion = PhysicsInformedLoss(
                lambda_data=self.config.lambda_data,
                lambda_physics=self.config.lambda_physics,
                lambda_smooth=self.config.lambda_smooth,
                lambda_magnitude=self.config.lambda_magnitude,
                normalize_physics=normalize_physics
            )
        else:
            self.criterion = SupervisedOnlyLoss(
                lambda_data=self.config.lambda_data,
                lambda_smooth=self.config.lambda_smooth
            )
        
        self.criterion = self.criterion.to(self.device)
    
    def _log_setup_info(self):
        """Log setup information."""
        logger.info("=" * 60)
        logger.info("Motion-Aware Training Configuration:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Train samples: {len(self.data_module.train_dataset)}")
        logger.info(f"  Val samples: {len(self.data_module.val_dataset)}")
        logger.info(f"  Upsampling factor: {self.config.upsampling_factor}")
        logger.info("Motion-Aware Settings:")
        logger.info(f"  Flow hidden channels: {self.config.flow_hidden_channels}")
        logger.info(f"  Flow scale: {self.config.flow_scale}")
        logger.info(f"  Learned fusion: {self.config.use_learned_fusion}")
        logger.info(f"  Flow consistency loss: {self.config.use_flow_consistency_loss}")
        logger.info(f"  Photometric loss: {self.config.use_photometric_loss}")
        logger.info(f"  Physics loss: {self.config.use_physics_loss}")
        logger.info(f"  Mixed precision: {self.config.use_amp}")
        logger.info("=" * 60)
    
    def _compute_flow_losses(
        self,
        sparse_encoded: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute flow regularization losses.
        
        Args:
            sparse_encoded: Encoded sparse features (B, T_sparse, 2, C, H, W)
            epoch: Current epoch for warmup
        
        Returns:
            total_flow_loss: Combined flow losses
            components: Dictionary of individual loss values
        """
        from src.architectures.complex_motion_interpolation import (
            compute_flow_consistency_loss,
            compute_photometric_loss
        )
        
        components = {
            'flow_consistency': 0.0,
            'photometric': 0.0
        }
        
        # Check warmup
        if epoch < self.config.flow_loss_warmup_epochs:
            return torch.tensor(0.0, device=self.device), components
        
        # Get flow estimator
        flow_estimator = self.model.get_flow_estimator()
        
        B, T_sparse, _, C, H, W = sparse_encoded.shape
        
        total_consistency = 0.0
        total_photometric = 0.0
        count = 0
        
        for t in range(T_sparse - 1):
            frame_t = sparse_encoded[:, t]
            frame_t1 = sparse_encoded[:, t + 1]
            
            # Get flows
            forward_flow, backward_flow = flow_estimator(frame_t, frame_t1)
            
            # Flow consistency loss
            if self.config.use_flow_consistency_loss:
                consistency = compute_flow_consistency_loss(
                    forward_flow, backward_flow, frame_t, frame_t1
                )
                total_consistency += consistency
            
            # Photometric loss
            if self.config.use_photometric_loss:
                photometric = compute_photometric_loss(
                    frame_t, frame_t1, forward_flow
                )
                total_photometric += photometric
            
            count += 1
        
        if count > 0:
            total_consistency /= count
            total_photometric /= count
        
        components['flow_consistency'] = total_consistency.item() if torch.is_tensor(total_consistency) else total_consistency
        components['photometric'] = total_photometric.item() if torch.is_tensor(total_photometric) else total_photometric
        
        # Combine with weights
        total_flow_loss = (
            self.config.lambda_flow_consistency * total_consistency +
            self.config.lambda_photometric * total_photometric
        )
        
        return total_flow_loss, components
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = MotionAwareMetrics()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            sparse = batch['sparse'].to(self.device)
            dense = batch['dense'].to(self.device)
            
            batch_size = sparse.shape[0]
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    pred = self.model(sparse)
                    loss, components = self.criterion(pred, dense, return_components=True)
                    
                    # Add flow losses (computed outside autocast for stability)
                with autocast():
                    # Get encoded features for flow loss computation
                    # Note: We recompute encoding for flow losses
                    encoded_frames = []
                    t_sparse = sparse.shape[1]
                    for t in range(t_sparse):
                        frame = sparse[:, t]
                        encoded, _ = self.model.encoder(frame)
                        encoded_frames.append(encoded)
                    sparse_encoded = torch.stack(encoded_frames, dim=1)
                    
                    # Project to hidden space (same as inside bottleneck)
                    sparse_encoded = self.model.get_projected_features(sparse_encoded)
                    
                    flow_loss, flow_components = self._compute_flow_losses(
                        sparse_encoded, self.current_epoch
                    )
                    
                    total_loss = loss + flow_loss
                    total_loss = total_loss / self.config.accumulation_steps
                
                # Update components
                components.update(flow_components)
                components['total'] = total_loss.item() * self.config.accumulation_steps
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                pred = self.model(sparse)
                loss, components = self.criterion(pred, dense, return_components=True)
                
                # Flow losses
                encoded_frames = []
                t_sparse = sparse.shape[1]
                for t in range(t_sparse):
                    frame = sparse[:, t]
                    encoded, _ = self.model.encoder(frame)
                    encoded_frames.append(encoded)
                sparse_encoded = torch.stack(encoded_frames, dim=1)
                
                # Project to hidden space (same as inside bottleneck)
                sparse_encoded = self.model.get_projected_features(sparse_encoded)
                
                flow_loss, flow_components = self._compute_flow_losses(
                    sparse_encoded, self.current_epoch
                )
                
                total_loss = loss + flow_loss
                total_loss = total_loss / self.config.accumulation_steps
                
                components.update(flow_components)
                components['total'] = total_loss.item() * self.config.accumulation_steps
                
                total_loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            metrics.update(components, batch_size)
            
            # Compute PSNR
            with torch.no_grad():
                psnr = metrics.compute_psnr(pred, dense)
                metrics.add_psnr(psnr, batch_size)
            
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                grad_norms = self._compute_gradient_norms()
                
                logger.info(
                    f"  Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {components['total']:.4f}, "
                    f"Data: {components.get('data', 0):.4f}, "
                    f"Flow: {components.get('flow_consistency', 0) + components.get('photometric', 0):.4f}, "
                    f"PSNR: {psnr:.2f} dB"
                )
        
        return metrics.get_averages()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics = MotionAwareMetrics()
        
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
            
            metrics.update(components, batch_size)
            psnr = metrics.compute_psnr(pred, dense)
            metrics.add_psnr(psnr, batch_size)
        
        return metrics.get_averages()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if hasattr(self.criterion, 'state_dict'):
            checkpoint['criterion_state_dict'] = self.criterion.state_dict()
        
        # Log learned physics parameters
        if hasattr(self.criterion, 'wave_equation'):
            we = self.criterion.wave_equation
            if hasattr(we, 'rho') and isinstance(we.rho, nn.Parameter):
                checkpoint['learned_physics'] = {
                    'rho': we.rho.item(),
                    'mu': we.mu.item(),
                    'eta': we.eta.item()
                }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if hasattr(self.criterion, 'load_state_dict') and checkpoint.get('criterion_state_dict'):
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _update_physics_warmup(self, epoch: int):
        """Update physics loss weight based on warmup schedule."""
        if not self.config.use_physics_loss:
            return
        
        warmup_epochs = getattr(self.config, 'physics_warmup_epochs', 10)
        target_lambda = getattr(self.config, 'lambda_physics_target', 0.1)
        
        if epoch < warmup_epochs:
            current_lambda = 0.0
        else:
            ramp_epochs = 20
            progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
            current_lambda = target_lambda * (1 - np.cos(np.pi * progress)) / 2
        
        if hasattr(self.criterion, 'set_lambda_physics'):
            self.criterion.set_lambda_physics(current_lambda)
        else:
            self.criterion.lambda_physics = current_lambda
        
        if epoch == warmup_epochs:
            logger.info(f"Physics warmup complete. Starting ramp-up to λ={target_lambda:.2e}")
    
    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for monitoring."""
        grad_norms = {}
        
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        grad_norms['model_total'] = total_norm ** 0.5
        
        # Flow estimator gradients
        flow_norm = 0.0
        for param in self.model.get_flow_estimator().parameters():
            if param.grad is not None:
                flow_norm += param.grad.data.norm(2).item() ** 2
        grad_norms['flow_estimator'] = flow_norm ** 0.5
        
        return grad_norms
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Motion-Aware training...")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Update physics warmup
            self._update_physics_warmup(epoch)
            
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            logger.info("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics
            physics_loss = train_metrics.get('physics_loss', 0)
            flow_loss = (train_metrics.get('flow_consistency_loss', 0) + 
                        train_metrics.get('photometric_loss', 0))
            
            logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB")
            logger.info(f"  Data: {train_metrics.get('data_loss', 0):.4f}, "
                       f"Flow: {flow_loss:.4f}, "
                       f"Physics: {physics_loss:.6f}")
            
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
            
            # Validate
            if (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validate()
                
                logger.info(f"Val   - Loss: {val_metrics['total_loss']:.4f}, PSNR: {val_metrics['psnr']:.2f} dB")
                
                if self.writer:
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f"val/{key}", value, epoch)
                
                # Check for best model
                val_loss = val_metrics['total_loss']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best_model.pt", is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.config.early_stopping and self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.writer:
                self.writer.add_scalar("train/learning_rate", current_lr, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        
        logger.info("\nTraining complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.writer:
            self.writer.close()
    
    def test(self) -> Dict[str, float]:
        """Test the model on test set."""
        logger.info("Evaluating on test set...")
        
        test_loader = self.data_module.get_test_dataloader(batch_size=self.config.batch_size)
        
        self.model.eval()
        metrics = MotionAwareMetrics()
        
        with torch.no_grad():
            for batch in test_loader:
                sparse = batch['sparse'].to(self.device)
                dense = batch['dense'].to(self.device)
                batch_size = sparse.shape[0]
                
                pred = self.model(sparse)
                loss, components = self.criterion(pred, dense, return_components=True)
                
                metrics.update(components, batch_size)
                psnr = metrics.compute_psnr(pred, dense)
                metrics.add_psnr(psnr, batch_size)
        
        test_metrics = metrics.get_averages()
        
        logger.info("Test Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Motion-Aware PI-CV-RecUNet")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    # Data
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    # Model
    parser.add_argument("--base_channels", type=int, help="Base channels for U-Net")
    parser.add_argument("--lstm_hidden", type=int, help="LSTM hidden channels")
    parser.add_argument("--upsampling_factor", type=int, help="Temporal upsampling factor")
    
    # Motion-aware settings
    parser.add_argument("--flow_hidden", type=int, help="Flow estimator hidden channels")
    parser.add_argument("--flow_scale", type=float, help="Flow scale factor")
    parser.add_argument("--no_learned_fusion", action="store_true", help="Disable learned fusion")
    parser.add_argument("--no_flow_consistency", action="store_true", help="Disable flow consistency loss")
    parser.add_argument("--no_photometric", action="store_true", help="Disable photometric loss")
    
    # Training
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--no_physics", action="store_true", help="Disable physics loss")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    
    # Misc
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load or create config
    if args.config:
        config = MotionAwareTrainConfig.from_json(args.config)
    else:
        config = MotionAwareTrainConfig()
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.base_channels:
        config.base_channels = args.base_channels
    if args.lstm_hidden:
        config.lstm_hidden_channels = args.lstm_hidden
    if args.upsampling_factor:
        config.upsampling_factor = args.upsampling_factor
    if args.flow_hidden:
        config.flow_hidden_channels = args.flow_hidden
    if args.flow_scale:
        config.flow_scale = args.flow_scale
    if args.no_learned_fusion:
        config.use_learned_fusion = False
    if args.no_flow_consistency:
        config.use_flow_consistency_loss = False
    if args.no_photometric:
        config.use_photometric_loss = False
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_physics:
        config.use_physics_loss = False
    if args.no_amp:
        config.use_amp = False
    if args.seed:
        config.seed = args.seed
    
    # Create trainer
    trainer = MotionAwareTrainer(config)
    trainer.setup()
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training or testing
    if args.test_only:
        best_model_path = trainer.output_dir / "best_model.pt"
        if best_model_path.exists():
            trainer.load_checkpoint(str(best_model_path))
        trainer.test()
    else:
        trainer.train()
        
        # Test with best model
        best_model_path = trainer.output_dir / "best_model.pt"
        if best_model_path.exists():
            trainer.load_checkpoint(str(best_model_path))
        trainer.test()


if __name__ == "__main__":
    main()
