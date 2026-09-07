"""
Training script for Input-Guided Motion-Aware PI-CV-RecUNet.

Key difference from previous versions:
- Flow is estimated from ORIGINAL high-resolution input frames
- Flow is then downsampled and scaled to match bottleneck resolution
- This captures true pixel-level motion, not abstracted features

Usage:
    python -m src.training.train_input_guided \
        --config configs/config_input_guided.json \
        --experiment_name input_guided_trial1
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class InputGuidedMetrics:
    """Tracking metrics for Kasai velocity-centric training.
    
    Simplified to track only:
    - Kasai autocorrelation loss (primary velocity metric)
    - Data loss (complex MSE)
    - Flow losses (consistency, photometric, smoothness)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.kasai_loss = 0.0
        self.data_loss = 0.0
        self.flow_consistency_loss = 0.0
        self.photometric_loss = 0.0
        self.flow_smoothness_loss = 0.0
        self.count = 0
    
    def update(self, loss_components: Dict, batch_size: int = 1):
        self.total_loss += loss_components.get('total', 0.0) * batch_size
        self.kasai_loss += loss_components.get('kasai', 0.0) * batch_size
        self.data_loss += loss_components.get('data', 0.0) * batch_size
        self.flow_consistency_loss += loss_components.get('flow_consistency', 0.0) * batch_size
        self.photometric_loss += loss_components.get('photometric', 0.0) * batch_size
        self.flow_smoothness_loss += loss_components.get('flow_smoothness', 0.0) * batch_size
        self.count += batch_size
    
    def get_averages(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        
        return {
            'total_loss': self.total_loss / self.count,
            'kasai_loss': self.kasai_loss / self.count,
            'data_loss': self.data_loss / self.count,
            'flow_consistency_loss': self.flow_consistency_loss / self.count,
            'photometric_loss': self.photometric_loss / self.count,
            'flow_smoothness_loss': self.flow_smoothness_loss / self.count,
        }


@dataclass
class InputGuidedConfig:
    """Configuration for Input-Guided Motion-Aware training.
    
    Uses original loss functions: data + phase + magnitude + smoothness
    Plus flow-related losses for motion estimation supervision.
    Does NOT use velocity loss or PDE consistency loss.
    """
    
    # Experiment
    experiment_name: str = "input_guided_v1"
    data_dir: str = "data/CIRS_phantom_dataset"
    output_dir: str = "outputs"
    device: str = "cuda"
    seed: int = 42
    
    # Model architecture
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4
    lstm_hidden_channels: int = 128
    lstm_num_layers: int = 1
    upsampling_factor: int = 20
    
    # Flow estimation (operates on high-res input)
    flow_hidden_channels: int = 32  # Lighter network for high-res
    flow_scale: float = 0.1  # Conservative scaling
    use_residual_fusion: bool = True
    
    # Flow losses (for motion estimation supervision)
    use_flow_losses: bool = True
    use_flow_consistency_loss: bool = True
    use_photometric_loss: bool = True
    use_flow_smoothness_loss: bool = True
    lambda_flow_consistency: float = 0.001
    lambda_photometric: float = 0.001
    lambda_flow_smoothness: float = 0.0005
    flow_loss_warmup_epochs: int = 10
    
    # Training
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 50.0
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"
    lr_step_size: int = 30
    lr_gamma: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-7
    
    # Loss weights (original losses only - no velocity/PDE)
    lambda_data: float = 1.0
    lambda_phase: float = 0.5
    lambda_smooth: float = 0.01
    lambda_magnitude: float = 0.5
    phase_threshold_ratio: float = 0.05
    
    # Data
    sparse_frame_step: int = 20
    normalization: str = "per_sample"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_augmentation: bool = False
    num_workers: int = 4
    
    # Mixed precision
    use_amp: bool = True
    accumulation_steps: int = 1
    
    # Checkpointing
    save_every: int = 10
    early_stopping_patience: int = 20
    use_early_stopping: bool = True  # Whether to enable early stopping
    use_train_loss_for_best: bool = False  # Use training loss instead of val loss for model selection
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'InputGuidedConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle physics_params specially (it's a dict, not a simple field)
        physics_params = data.pop('physics_params', None)
        # Filter to valid dataclass fields
        valid_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != 'physics_params'}
        config = cls(**valid_data)
        config.physics_params = physics_params
        return config


def create_input_guided_model(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 4,
    lstm_hidden_channels: int = 128,
    lstm_num_layers: int = 1,
    upsampling_factor: int = 20,
    flow_hidden_channels: int = 32,
    flow_scale: float = 0.1,
    use_residual_fusion: bool = True,
) -> nn.Module:
    """
    Create an Input-Guided Motion-Aware PI-CV-RecUNet model.
    
    Key difference: Flow is estimated from original high-res inputs,
    not from bottleneck features.
    """
    from src.architectures.pinn_recunet import Encoder, Decoder
    from src.architectures.input_guided_motion import InputGuidedTemporalBottleneck
    from src.architectures.complex_modules import ComplexConv2d as CC2d
    
    class InputGuided_PI_CV_RecUNet(nn.Module):
        """
        Input-Guided Motion-Aware Physics-Informed Complex-Valued Recurrent U-Net.
        
        Architecture:
        1. Encode each sparse frame independently
        2. Pass encoded features + ORIGINAL high-res inputs to temporal bottleneck
        3. Bottleneck estimates flow from high-res inputs, applies to features
        4. Decode dense features back to full resolution
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
        ):
            super(InputGuided_PI_CV_RecUNet, self).__init__()
            
            self.upsampling_factor = upsampling_factor
            self.in_channels = in_channels
            
            # Encoder
            self.encoder = Encoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_levels=num_levels
            )
            
            encoder_out_channels = self.encoder.output_channels
            
            # Input-Guided Temporal Bottleneck
            self.temporal_bottleneck = InputGuidedTemporalBottleneck(
                encoder_channels=encoder_out_channels,
                hidden_channels=lstm_hidden_channels,
                input_channels=in_channels,  # Original input channels
                num_layers=lstm_num_layers,
                kernel_size=3,
                upsampling_factor=upsampling_factor,
                flow_hidden_channels=flow_hidden_channels,
                flow_scale=flow_scale,
                use_residual_fusion=use_residual_fusion,
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
            
            # Store original high-res inputs for flow estimation
            inputs_high_res = x  # (B, T_sparse, 2, C_in, H_large, W_large)
            
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
            # encoded_stack: (B, T_sparse, 2, C_enc, H_small, W_small)
            
            # Input-guided temporal interpolation
            # Pass BOTH encoded features AND original high-res inputs
            dense_encoded = self.temporal_bottleneck(
                encoded_stack,
                inputs_high_res  # Original high-res frames for flow estimation
            )
            
            t_dense = dense_encoded.shape[1]
            
            # Interpolate skip connections using trilinear interpolation
            # This provides smoother temporal transitions than simple linear blending
            interpolated_skips = self._interpolate_skip_connections(skip_features_list, t_dense)
            
            # Decode each dense frame with trilinearly interpolated skips
            outputs = []
            for t in range(t_dense):
                encoded_frame = dense_encoded[:, t, :, :, :, :]
                
                # Get trilinearly interpolated skips for this timestep
                skips = [level_skips[:, t, :, :, :, :] for level_skips in interpolated_skips]
                
                decoded = self.decoder(encoded_frame, skips)
                output = self.output_head(decoded)
                outputs.append(output)
            
            output = torch.stack(outputs, dim=1)
            
            return output
        
        def _interpolate_skip_connections(self, skip_features_list, t_dense: int):
            """
            Interpolate skip connections using trilinear interpolation.
            
            This provides smoother temporal transitions for the skip connections
            compared to simple linear blending between first and last frames.
            
            Args:
                skip_features_list: List of skip features from each sparse frame
                t_dense: Number of dense frames to interpolate to
                
            Returns:
                List of interpolated skip features for each level
            """
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
        
        def get_flow_stats(self) -> dict:
            """Get flow statistics from the last forward pass."""
            return self.temporal_bottleneck.get_flow_stats()
    
    model = InputGuided_PI_CV_RecUNet(
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
    )
    
    return model


class InputGuidedTrainer:
    """Training orchestrator for Input-Guided Motion-Aware PI-CV-RecUNet."""
    
    def __init__(self, config: InputGuidedConfig):
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
        self.scaler = None  # Initialize in setup()
        self.criterion = None
        
        # TensorBoard writer
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
            logger.info("TensorBoard not available.")
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def _set_seeds(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup(self):
        """Setup all training components."""
        logger.info("Setting up Input-Guided Motion-Aware training...")
        
        self._setup_data()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        
        if self.config.use_amp:
            self.scaler = GradScaler('cuda')
        
        logger.info("Setup complete!")
    
    def _setup_data(self):
        """Setup data loaders with OOD validation split."""
        from src.utils.input_guided_data_loader import InputGuidedDataConfig, InputGuidedDataModule
        
        # Get fixed_sparse_frames from config, default to 3
        fixed_sparse_frames = getattr(self.config, 'fixed_sparse_frames', 3)
        
        data_config = InputGuidedDataConfig(
            data_dir=self.config.data_dir,
            sparse_frame_step=self.config.sparse_frame_step,
            fixed_sparse_frames=fixed_sparse_frames,
            normalization=self.config.normalization,
            use_augmentation=self.config.use_augmentation,
            num_workers=self.config.num_workers,
            seed=self.config.seed
        )
        
        self.data_module = InputGuidedDataModule(data_config)
        self.data_module.setup()
        
        self.train_loader = self.data_module.get_train_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.val_loader = self.data_module.get_val_dataloader(
            batch_size=self.config.batch_size
        )
        self.test_loader = self.data_module.get_test_dataloader(
            batch_size=self.config.batch_size
        )
    
    def _setup_model(self):
        self.model = create_input_guided_model(
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
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        flow_params = sum(p.numel() for p in self.model.get_flow_estimator().parameters())
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"  Flow estimator: {flow_params:,} parameters")
    
    def _setup_criterion(self):
        """Setup simplified loss functions: Kasai + Data + Flow.
        
        For shear wave elastography, we use:
        - KASAI AUTOCORRELATION: The gold standard for velocity estimation.
          Computes R = P(t) * conj(P(t-1)) and minimizes |R_pred - R_true|.
          This directly measures the phase change (Δφ) which is proportional
          to tissue particle velocity.
        - DATA LOSS: Complex MSE for general signal reconstruction.
        - FLOW LOSSES: For motion estimation quality (consistency, photometric, smoothness).
        
        All other losses (phase, magnitude, smoothness, wave equation) are removed
        to reduce clutter and focus on what matters for elastography.
        """
        from src.utils.loss_functions import ComplexMSELoss, KasaiAutocorrelationLoss
        
        # Get loss weights from config with defaults
        lambda_kasai = getattr(self.config, 'lambda_kasai', 10.0)  # PRIMARY: velocity estimation
        lambda_data = getattr(self.config, 'lambda_data', 1.0)     # SECONDARY: signal reconstruction
        kasai_threshold_ratio = getattr(self.config, 'kasai_threshold_ratio', 0.05)
        
        logger.info(f"="*60)
        logger.info(f"Simplified Loss Setup: Kasai + Data + Flow")
        logger.info(f"="*60)
        logger.info(f"  KASAI AUTOCORRELATION (velocity):")
        logger.info(f"    lambda_kasai={lambda_kasai}")
        logger.info(f"    Computes: L = |R_pred - R_true| where R = P(t) * conj(P(t-1))")
        logger.info(f"    This measures phase change (Δφ) = tissue velocity")
        logger.info(f"  DATA LOSS (reconstruction):")
        logger.info(f"    lambda_data={lambda_data}")
        logger.info(f"  FLOW LOSSES (motion estimation):")
        logger.info(f"    lambda_flow_consistency={getattr(self.config, 'lambda_flow_consistency', 0.001)}")
        logger.info(f"    lambda_photometric={getattr(self.config, 'lambda_photometric', 0.001)}")
        logger.info(f"    lambda_flow_smoothness={getattr(self.config, 'lambda_flow_smoothness', 0.0005)}")
        logger.info(f"="*60)
        
        # Store loss weights
        self.lambda_kasai = lambda_kasai
        self.lambda_data = lambda_data
        
        # Initialize loss functions
        self.kasai_loss_fn = KasaiAutocorrelationLoss(
            reduction='mean',
            threshold_ratio=kasai_threshold_ratio
        )
        self.data_loss_fn = ComplexMSELoss(reduction='mean')
        
        # No criterion object needed - we compute losses directly
        self.criterion = None
    
    def _setup_optimizer(self):
        # Only model parameters - no learnable physics params anymore
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
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
                patience=getattr(self.config, 'lr_patience', 10),
                min_lr=self.config.min_lr
            )
    
    def _get_flow_loss_weight(self, epoch: int) -> float:
        """Get flow loss weight with warmup."""
        if epoch < self.config.flow_loss_warmup_epochs:
            return 0.0
        return 1.0
    
    def _compute_flow_losses(self, epoch: int) -> Dict[str, torch.Tensor]:
        """Compute flow-related losses."""
        losses = {}
        flow_weight = self._get_flow_loss_weight(epoch)
        
        if flow_weight == 0 or not self.config.use_flow_losses:
            return losses
        
        motion_interp = self.model.get_motion_interpolation()
        
        # Get cached high-res flows and frames
        flows_high_res = motion_interp.get_cached_flows_high_res()
        frames_high_res = motion_interp.get_cached_frames_high_res()
        
        if not flows_high_res:
            return losses
        
        from src.architectures.input_guided_motion import (
            flow_consistency_loss,
            flow_smoothness_loss,
            photometric_loss
        )
        
        if self.config.use_flow_consistency_loss:
            consistency = flow_consistency_loss(flows_high_res)
            losses['flow_consistency'] = self.config.lambda_flow_consistency * consistency * flow_weight
        
        if self.config.use_flow_smoothness_loss:
            smoothness = flow_smoothness_loss(flows_high_res)
            losses['flow_smoothness'] = self.config.lambda_flow_smoothness * smoothness * flow_weight
        
        if self.config.use_photometric_loss and frames_high_res:
            photo = photometric_loss(flows_high_res, frames_high_res)
            losses['photometric'] = self.config.lambda_photometric * photo * flow_weight
        
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with Kasai autocorrelation loss.
        
        Simplified to use only:
        - Kasai autocorrelation loss (primary velocity metric)
        - Data loss (complex MSE)
        - Flow losses (consistency, photometric, smoothness)
        """
        self.model.train()
        
        total_loss = 0.0
        total_kasai_loss = 0.0
        total_data_loss = 0.0
        total_flow_consistency = 0.0
        total_flow_photometric = 0.0
        total_flow_smoothness = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Handle both dict and tuple formats from dataloader
            if isinstance(batch, dict):
                sparse_frames = batch['sparse'].to(self.device)
                dense_frames = batch['dense'].to(self.device)
            else:
                sparse_frames, dense_frames = batch[0].to(self.device), batch[1].to(self.device)
            
            with autocast('cuda', enabled=self.config.use_amp):
                # Forward pass
                predictions = self.model(sparse_frames)
                
                # ============ KASAI AUTOCORRELATION LOSS ============
                # PRIMARY: Measures velocity (phase change) directly
                # L = |R_pred - R_true| where R = P(t) * conj(P(t-1))
                l_kasai = self.kasai_loss_fn(predictions, dense_frames)
                
                # ============ DATA LOSS ============
                # SECONDARY: Complex MSE for signal reconstruction
                l_data = self.data_loss_fn(predictions, dense_frames)
                
                # Combine main losses
                loss = self.lambda_kasai * l_kasai + self.lambda_data * l_data
                
                # ============ FLOW LOSSES ============
                flow_losses = self._compute_flow_losses(epoch)
                flow_loss_total = sum(flow_losses.values()) if flow_losses else torch.tensor(0.0, device=self.device)
                loss = loss + flow_loss_total
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_kasai_loss += l_kasai.item()
            total_data_loss += l_data.item()
            if flow_losses:
                total_flow_consistency += flow_losses.get('flow_consistency', torch.tensor(0.0)).item() if isinstance(flow_losses.get('flow_consistency'), torch.Tensor) else flow_losses.get('flow_consistency', 0.0)
                total_flow_photometric += flow_losses.get('photometric', torch.tensor(0.0)).item() if isinstance(flow_losses.get('photometric'), torch.Tensor) else flow_losses.get('photometric', 0.0)
                total_flow_smoothness += flow_losses.get('flow_smoothness', torch.tensor(0.0)).item() if isinstance(flow_losses.get('flow_smoothness'), torch.Tensor) else flow_losses.get('flow_smoothness', 0.0)
            num_batches += 1
            
            if batch_idx % 10 == 0:
                # Get flow stats
                flow_stats = self.model.get_flow_stats()
                flow_info = f" flow_mag={flow_stats.get('flow_mean', 0):.2f}" if flow_stats else ""
                
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} kasai={l_kasai.item():.4f} data={l_data.item():.4f}{flow_info}"
                )
        
        avg_loss = total_loss / num_batches
        avg_kasai_loss = total_kasai_loss / num_batches
        avg_data_loss = total_data_loss / num_batches
        avg_flow_consistency = total_flow_consistency / num_batches
        avg_flow_photometric = total_flow_photometric / num_batches
        avg_flow_smoothness = total_flow_smoothness / num_batches
        
        return {
            'loss': avg_loss,
            'kasai_loss': avg_kasai_loss,
            'data_loss': avg_data_loss,
            'flow_consistency_loss': avg_flow_consistency,
            'flow_photometric_loss': avg_flow_photometric,
            'flow_smoothness_loss': avg_flow_smoothness,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate with the same losses used for training.
        
        Uses Kasai autocorrelation loss, data loss, and flow losses.
        No IQ reconstruction metrics - we care about velocity estimation.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_kasai_loss = 0.0
        total_data_loss = 0.0
        total_flow_consistency = 0.0
        total_flow_photometric = 0.0
        total_flow_smoothness = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # Handle both dict and tuple formats from dataloader
            if isinstance(batch, dict):
                sparse_frames = batch['sparse'].to(self.device)
                dense_frames = batch['dense'].to(self.device)
            else:
                sparse_frames, dense_frames = batch[0].to(self.device), batch[1].to(self.device)
            
            predictions = self.model(sparse_frames)
            
            # Kasai autocorrelation loss (primary)
            l_kasai = self.kasai_loss_fn(predictions, dense_frames)
            
            # Data loss (secondary)
            l_data = self.data_loss_fn(predictions, dense_frames)
            
            # Total loss
            loss = self.lambda_kasai * l_kasai + self.lambda_data * l_data
            
            # Flow losses (for validation we compute but don't backprop)
            # We need forward pass to populate flow caches first
            motion_interp = self.model.get_motion_interpolation()
            flows_high_res = motion_interp.get_cached_flows_high_res()
            frames_high_res = motion_interp.get_cached_frames_high_res()
            
            flow_consistency = 0.0
            flow_photometric = 0.0
            flow_smoothness = 0.0
            
            if flows_high_res and self.config.use_flow_losses:
                from src.architectures.input_guided_motion import (
                    flow_consistency_loss,
                    flow_smoothness_loss,
                    photometric_loss
                )
                
                if self.config.use_flow_consistency_loss:
                    flow_consistency = flow_consistency_loss(flows_high_res).item()
                if self.config.use_flow_smoothness_loss:
                    flow_smoothness = flow_smoothness_loss(flows_high_res).item()
                if self.config.use_photometric_loss and frames_high_res:
                    flow_photometric = photometric_loss(flows_high_res, frames_high_res).item()
            
            total_loss += loss.item()
            total_kasai_loss += l_kasai.item()
            total_data_loss += l_data.item()
            total_flow_consistency += flow_consistency
            total_flow_photometric += flow_photometric
            total_flow_smoothness += flow_smoothness
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'kasai_loss': total_kasai_loss / num_batches,
            'data_loss': total_data_loss / num_batches,
            'flow_consistency_loss': total_flow_consistency / num_batches,
            'flow_photometric_loss': total_flow_photometric / num_batches,
            'flow_smoothness_loss': total_flow_smoothness / num_batches,
        }
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test with the same losses used for training.
        
        Uses Kasai autocorrelation loss, data loss, and flow losses.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_kasai_loss = 0.0
        total_data_loss = 0.0
        total_flow_consistency = 0.0
        total_flow_photometric = 0.0
        total_flow_smoothness = 0.0
        num_batches = 0
        
        for batch in self.test_loader:
            # Handle both dict and tuple formats from dataloader
            if isinstance(batch, dict):
                sparse_frames = batch['sparse'].to(self.device)
                dense_frames = batch['dense'].to(self.device)
            else:
                sparse_frames, dense_frames = batch[0].to(self.device), batch[1].to(self.device)
            
            predictions = self.model(sparse_frames)
            
            # Kasai autocorrelation loss (primary)
            l_kasai = self.kasai_loss_fn(predictions, dense_frames)
            
            # Data loss (secondary)
            l_data = self.data_loss_fn(predictions, dense_frames)
            
            # Total loss
            loss = self.lambda_kasai * l_kasai + self.lambda_data * l_data
            
            # Flow losses
            motion_interp = self.model.get_motion_interpolation()
            flows_high_res = motion_interp.get_cached_flows_high_res()
            frames_high_res = motion_interp.get_cached_frames_high_res()
            
            flow_consistency = 0.0
            flow_photometric = 0.0
            flow_smoothness = 0.0
            
            if flows_high_res and self.config.use_flow_losses:
                from src.architectures.input_guided_motion import (
                    flow_consistency_loss,
                    flow_smoothness_loss,
                    photometric_loss
                )
                
                if self.config.use_flow_consistency_loss:
                    flow_consistency = flow_consistency_loss(flows_high_res).item()
                if self.config.use_flow_smoothness_loss:
                    flow_smoothness = flow_smoothness_loss(flows_high_res).item()
                if self.config.use_photometric_loss and frames_high_res:
                    flow_photometric = photometric_loss(flows_high_res, frames_high_res).item()
            
            total_loss += loss.item()
            total_kasai_loss += l_kasai.item()
            total_data_loss += l_data.item()
            total_flow_consistency += flow_consistency
            total_flow_photometric += flow_photometric
            total_flow_smoothness += flow_smoothness
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'kasai_loss': total_kasai_loss / num_batches,
            'data_loss': total_data_loss / num_batches,
            'flow_consistency_loss': total_flow_consistency / num_batches,
            'flow_photometric_loss': total_flow_photometric / num_batches,
            'flow_smoothness_loss': total_flow_smoothness / num_batches,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch}.pt")
        
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pt")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop with Kasai autocorrelation loss."""
        logger.info("=" * 60)
        logger.info("Input-Guided Motion-Aware Training (Kasai Loss)")
        logger.info("=" * 60)
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"  Val samples: {len(self.val_loader.dataset)}")
        logger.info("Loss Configuration:")
        logger.info(f"  lambda_kasai: {self.lambda_kasai} (Kasai autocorrelation)")
        logger.info(f"  lambda_data: {self.lambda_data} (Complex MSE)")
        logger.info("Flow Loss Settings:")
        logger.info(f"  Flow losses enabled: {self.config.use_flow_losses}")
        if self.config.use_flow_losses:
            logger.info(f"    Consistency weight: {self.config.lambda_flow_consistency}")
            logger.info(f"    Photometric weight: {self.config.lambda_photometric}")
            logger.info(f"    Smoothness weight: {self.config.lambda_flow_smoothness}")
            logger.info(f"    Warmup epochs: {self.config.flow_loss_warmup_epochs}")
        logger.info(f"  Mixed precision: {self.config.use_amp}")
        logger.info(f"  Early stopping: {self.config.use_early_stopping} (patience={self.config.early_stopping_patience})")
        logger.info(f"  Use train loss for best model: {self.config.use_train_loss_for_best}")
        logger.info("=" * 60)
        
        logger.info("Starting training...")
        
        best_metric = float('inf')  # Track best metric (train or val loss based on config)
        
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Determine which metric to use for model selection
            current_metric = train_metrics['loss'] if self.config.use_train_loss_for_best else val_metrics['loss']
            metric_name = 'train_loss' if self.config.use_train_loss_for_best else 'val_loss'
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                f"train_kasai={train_metrics['kasai_loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_kasai={val_metrics['kasai_loss']:.4f}"
            )
            
            # TensorBoard logging
            if self.writer:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
            
            # Check for improvement (based on configured metric)
            if current_metric < best_metric:
                best_metric = current_metric
                self.best_val_loss = val_metrics['loss']  # Still track val_loss for logging
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"  New best model! {metric_name}={current_metric:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping (only if enabled)
            if self.config.use_early_stopping and self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training complete!")
        logger.info(f"Best {metric_name}: {best_metric:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Run test evaluation
        logger.info("Running test evaluation...")
        best_checkpoint = torch.load(self.output_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info("Loaded best model for testing")
        
        test_metrics = self.test()
        logger.info("Test Results:")
        logger.info(f"  Total Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Kasai Loss: {test_metrics['kasai_loss']:.4f}")
        logger.info(f"  Data Loss: {test_metrics['data_loss']:.4f}")
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Input-Guided Motion-Aware model")
    parser.add_argument("--config", type=str, default="configs/config_input_guided.json",
                        help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment name")
    parser.add_argument("--no_flow_losses", action="store_true",
                        help="Disable flow losses")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = InputGuidedConfig.load(Path(args.config))
    else:
        config = InputGuidedConfig()
    
    # Override with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.no_flow_losses:
        config.use_flow_losses = False
    if args.epochs:
        config.epochs = args.epochs
    
    # Create trainer and run
    trainer = InputGuidedTrainer(config)
    trainer.setup()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if trainer.writer:
            trainer.writer.close()


if __name__ == "__main__":
    main()
