"""
Velocity Fidelity Loss and Training Utilities for SWE.

This module provides:
    1. VelocityFidelityLoss - Combined L1 + Gradient Difference Loss with boundary masking
    2. Training loop with gradient accumulation
    3. AdamW optimizer configuration
    4. Logging and checkpointing utilities

Loss Components:
    - L1 Loss: Direct pixel-wise velocity error
    - Gradient Difference Loss (GDL): Sobel-filtered edge preservation
    - Boundary Masking: Ignores outer 15 pixels to avoid edge artifacts

Training Configuration:
    - Batch Size: 4 (with gradient accumulation support)
    - Optimizer: AdamW with lr=1e-4
    - Learning Rate Scheduler: Cosine annealing with warmup
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Sobel Filters for Gradient Computation
# =============================================================================

def get_sobel_kernels(device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create Sobel kernels for gradient computation.
    
    Returns:
        Tuple of (sobel_x, sobel_y) kernels, each shape (1, 1, 3, 3)
    """
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3)
    
    if device is not None:
        sobel_x = sobel_x.to(device)
        sobel_y = sobel_y.to(device)
    
    return sobel_x, sobel_y


def compute_spatial_gradients(
    x: torch.Tensor,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial gradients using Sobel filters.
    
    Args:
        x: Input tensor (B, C, H, W) or (B, C, T, H, W)
        sobel_x: Sobel kernel for x gradient
        sobel_y: Sobel kernel for y gradient
        
    Returns:
        Tuple of (grad_x, grad_y) with same shape as input
    """
    # Handle 5D input (B, C, T, H, W)
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        # Reshape to (B*C*T, 1, H, W) for 2D convolution
        x_flat = x.reshape(B * C * T, 1, H, W)
        
        grad_x = F.conv2d(x_flat, sobel_x, padding=1)
        grad_y = F.conv2d(x_flat, sobel_y, padding=1)
        
        grad_x = grad_x.reshape(B, C, T, H, W)
        grad_y = grad_y.reshape(B, C, T, H, W)
    else:
        # Standard 4D input (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        
        grad_x = F.conv2d(x_flat, sobel_x, padding=1)
        grad_y = F.conv2d(x_flat, sobel_y, padding=1)
        
        grad_x = grad_x.reshape(B, C, H, W)
        grad_y = grad_y.reshape(B, C, H, W)
    
    return grad_x, grad_y


# =============================================================================
# Boundary Mask Creation
# =============================================================================

def create_boundary_mask(
    shape: Tuple[int, ...],
    boundary_size: int = 15,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a mask that zeros out boundary pixels.
    
    Args:
        shape: Shape of the tensor to mask (*, H, W)
        boundary_size: Number of pixels to mask on each edge
        device: Device for the mask tensor
        
    Returns:
        Binary mask tensor with 0s at boundaries, 1s in center
    """
    H, W = shape[-2], shape[-1]
    
    mask = torch.ones(*shape, device=device)
    
    # Zero out boundaries
    if boundary_size > 0:
        mask[..., :boundary_size, :] = 0  # Top
        mask[..., -boundary_size:, :] = 0  # Bottom
        mask[..., :, :boundary_size] = 0  # Left
        mask[..., :, -boundary_size:] = 0  # Right
    
    return mask


# =============================================================================
# Velocity Fidelity Loss
# =============================================================================

class VelocityFidelityLoss(nn.Module):
    """
    Combined Loss Function for Velocity Estimation.
    
    Components:
        1. L1 Loss: Pixel-wise absolute error between predicted and target velocity
        2. Gradient Difference Loss (GDL): Preserves sharp wave edges using Sobel filters
    
    Features:
        - Automatic boundary masking to ignore edge artifacts
        - Configurable loss weights
        - Optional temporal gradient loss
    
    Total Loss = λ_l1 * L1_loss + λ_gdl * GDL_loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        gdl_weight: float = 0.5,
        boundary_mask_size: int = 15,
        use_temporal_gradient: bool = False,
        temporal_weight: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Args:
            l1_weight: Weight for L1 loss term
            gdl_weight: Weight for gradient difference loss term
            boundary_mask_size: Pixels to mask at image boundaries
            use_temporal_gradient: Whether to include temporal gradient loss
            temporal_weight: Weight for temporal gradient term
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.gdl_weight = gdl_weight
        self.boundary_mask_size = boundary_mask_size
        self.use_temporal_gradient = use_temporal_gradient
        self.temporal_weight = temporal_weight
        self.reduction = reduction
        
        # Register Sobel kernels as buffers (moved to device with model)
        sobel_x, sobel_y = get_sobel_kernels()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Cache for boundary mask
        self._mask_cache: Dict[Tuple, torch.Tensor] = {}
        
        logger.info(
            f"VelocityFidelityLoss initialized: "
            f"L1 weight={l1_weight}, GDL weight={gdl_weight}, "
            f"boundary_mask={boundary_mask_size}px"
        )
    
    def _get_boundary_mask(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Get or create boundary mask (cached for efficiency)."""
        cache_key = (shape, str(device))
        
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = create_boundary_mask(
                shape, self.boundary_mask_size, device
            )
        
        return self._mask_cache[cache_key]
    
    def compute_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked L1 loss.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W)
            target: Target velocity (B, 1, T, H, W)
            mask: Binary mask (B, 1, T, H, W)
            
        Returns:
            L1 loss scalar
        """
        l1_error = torch.abs(pred - target) * mask
        
        if self.reduction == 'mean':
            # Normalize by number of non-masked pixels
            return l1_error.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return l1_error.sum()
        else:
            return l1_error
    
    def compute_gradient_difference_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gradient Difference Loss using Sobel filters.
        
        GDL = |∇x(pred) - ∇x(target)| + |∇y(pred) - ∇y(target)|
        
        This enforces sharp edges in the velocity maps by penalizing
        differences in spatial gradients.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W)
            target: Target velocity (B, 1, T, H, W)
            mask: Binary mask (B, 1, T, H, W)
            
        Returns:
            GDL loss scalar
        """
        # Compute gradients
        pred_gx, pred_gy = compute_spatial_gradients(pred, self.sobel_x, self.sobel_y)
        target_gx, target_gy = compute_spatial_gradients(target, self.sobel_x, self.sobel_y)
        
        # Gradient differences
        gdl_x = torch.abs(pred_gx - target_gx) * mask
        gdl_y = torch.abs(pred_gy - target_gy) * mask
        
        gdl = gdl_x + gdl_y
        
        if self.reduction == 'mean':
            return gdl.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return gdl.sum()
        else:
            return gdl
    
    def compute_temporal_gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal gradient loss.
        
        Enforces smooth temporal transitions by matching temporal derivatives.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W)
            target: Target velocity (B, 1, T, H, W)
            mask: Binary mask (B, 1, T, H, W)
            
        Returns:
            Temporal gradient loss scalar
        """
        # Temporal gradients (difference between consecutive frames)
        pred_dt = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_dt = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        mask_t = mask[:, :, 1:, :, :]  # Adjusted mask
        
        temporal_error = torch.abs(pred_dt - target_dt) * mask_t
        
        if self.reduction == 'mean':
            return temporal_error.sum() / (mask_t.sum() + 1e-8)
        elif self.reduction == 'sum':
            return temporal_error.sum()
        else:
            return temporal_error
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the total velocity fidelity loss.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W), range [-1, 1]
            target: Target velocity (B, 1, T, H, W), range [-1, 1]
            return_components: Whether to return individual loss components
            
        Returns:
            If return_components=False: Total loss scalar
            If return_components=True: (total_loss, loss_dict)
        """
        # Validate shapes
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={pred.shape}, target={target.shape}"
            )
        
        # Get boundary mask
        mask = self._get_boundary_mask(pred.shape, pred.device)
        
        # Compute loss components
        l1_loss = self.compute_l1_loss(pred, target, mask)
        gdl_loss = self.compute_gradient_difference_loss(pred, target, mask)
        
        # Total loss
        total_loss = self.l1_weight * l1_loss + self.gdl_weight * gdl_loss
        
        # Optional temporal gradient loss
        if self.use_temporal_gradient and pred.shape[2] > 1:
            temporal_loss = self.compute_temporal_gradient_loss(pred, target, mask)
            total_loss = total_loss + self.temporal_weight * temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=pred.device)
        
        if return_components:
            loss_dict = {
                'l1_loss': l1_loss.detach(),
                'gdl_loss': gdl_loss.detach(),
                'temporal_loss': temporal_loss.detach(),
                'total_loss': total_loss.detach(),
            }
            return total_loss, loss_dict
        
        return total_loss


class MagnitudeWeightedVelocityLoss(nn.Module):
    """
    Magnitude-Weighted Velocity Loss for Clean Training.
    
    This loss function uses magnitude masks from preprocessing to:
    1. Focus learning on signal regions (high magnitude)
    2. Ignore noise regions (low magnitude)
    3. Provide soft weighting for smooth transitions
    
    Components:
        - Weighted L1 Loss: |pred - target| * magnitude_mask
        - Weighted GDL: |∇pred - ∇target| * magnitude_mask
    
    Use with PreprocessedVelocityDataset which provides magnitude masks.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        gdl_weight: float = 0.5,
        use_temporal_gradient: bool = False,
        temporal_weight: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Args:
            l1_weight: Weight for L1 loss term
            gdl_weight: Weight for gradient difference loss term
            use_temporal_gradient: Whether to include temporal gradient loss
            temporal_weight: Weight for temporal gradient term
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.gdl_weight = gdl_weight
        self.use_temporal_gradient = use_temporal_gradient
        self.temporal_weight = temporal_weight
        self.reduction = reduction
        
        # Register Sobel kernels as buffers
        sobel_x, sobel_y = get_sobel_kernels()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        logger.info(
            f"MagnitudeWeightedVelocityLoss initialized: "
            f"L1 weight={l1_weight}, GDL weight={gdl_weight}"
        )
    
    def compute_weighted_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute magnitude-weighted L1 loss.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W)
            target: Target velocity (B, 1, T, H, W)
            mask: Magnitude mask (B, 1, T, H, W), values in [0, 1]
            
        Returns:
            Weighted L1 loss scalar
        """
        l1_error = torch.abs(pred - target) * mask
        
        if self.reduction == 'mean':
            # Normalize by sum of mask weights
            return l1_error.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return l1_error.sum()
        else:
            return l1_error
    
    def compute_weighted_gdl_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute magnitude-weighted Gradient Difference Loss.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W)
            target: Target velocity (B, 1, T, H, W)
            mask: Magnitude mask (B, 1, T, H, W)
            
        Returns:
            Weighted GDL loss scalar
        """
        # Compute gradients
        pred_gx, pred_gy = compute_spatial_gradients(pred, self.sobel_x, self.sobel_y)
        target_gx, target_gy = compute_spatial_gradients(target, self.sobel_x, self.sobel_y)
        
        # Gradient differences weighted by mask
        gdl_x = torch.abs(pred_gx - target_gx) * mask
        gdl_y = torch.abs(pred_gy - target_gy) * mask
        
        gdl = gdl_x + gdl_y
        
        if self.reduction == 'mean':
            return gdl.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return gdl.sum()
        else:
            return gdl
    
    def compute_weighted_temporal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute magnitude-weighted temporal gradient loss.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W)
            target: Target velocity (B, 1, T, H, W)
            mask: Magnitude mask (B, 1, T, H, W)
            
        Returns:
            Temporal gradient loss scalar
        """
        # Temporal gradients (difference between consecutive frames)
        pred_dt = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_dt = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        mask_t = mask[:, :, 1:, :, :]  # Adjusted mask
        
        temporal_error = torch.abs(pred_dt - target_dt) * mask_t
        
        if self.reduction == 'mean':
            return temporal_error.sum() / (mask_t.sum() + 1e-8)
        elif self.reduction == 'sum':
            return temporal_error.sum()
        else:
            return temporal_error
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the magnitude-weighted velocity loss.
        
        Args:
            pred: Predicted velocity (B, 1, T, H, W), range [-1, 1]
            target: Target velocity (B, 1, T, H, W), range [-1, 1]
            mask: Magnitude mask (B, 1, T, H, W), range [0, 1]
            return_components: Whether to return individual loss components
            
        Returns:
            If return_components=False: Total loss scalar
            If return_components=True: (total_loss, loss_dict)
        """
        # Validate shapes
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={pred.shape}, target={target.shape}"
            )
        if pred.shape != mask.shape:
            raise ValueError(
                f"Mask shape mismatch: pred={pred.shape}, mask={mask.shape}"
            )
        
        # Compute loss components
        l1_loss = self.compute_weighted_l1_loss(pred, target, mask)
        gdl_loss = self.compute_weighted_gdl_loss(pred, target, mask)
        
        # Total loss
        total_loss = self.l1_weight * l1_loss + self.gdl_weight * gdl_loss
        
        # Optional temporal gradient loss
        if self.use_temporal_gradient and pred.shape[2] > 1:
            temporal_loss = self.compute_weighted_temporal_loss(pred, target, mask)
            total_loss = total_loss + self.temporal_weight * temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=pred.device)
        
        if return_components:
            loss_dict = {
                'l1_loss': l1_loss.detach(),
                'gdl_loss': gdl_loss.detach(),
                'temporal_loss': temporal_loss.detach(),
                'total_loss': total_loss.detach(),
                'mask_coverage': mask.mean().detach(),  # Average mask weight
            }
            return total_loss, loss_dict
        
        return total_loss


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class VelocityTrainingConfig:
    """Configuration for velocity estimation training."""
    
    # Model
    model_name: str = "velocity_unet"
    
    # Data
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = field(init=False)
    
    # Loss weights
    l1_weight: float = 1.0
    gdl_weight: float = 0.5
    boundary_mask_size: int = 15
    
    # Training schedule
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_dir: str = "outputs/velocity_model"
    save_every_n_epochs: int = 10
    save_best_only: bool = True
    
    # Logging
    log_every_n_steps: int = 10
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-5
    
    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


# =============================================================================
# Training Utilities
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# Main Trainer Class
# =============================================================================

class VelocityTrainer:
    """
    Trainer class for velocity estimation model.
    
    Features:
        - Gradient accumulation for larger effective batch sizes
        - Mixed precision training (AMP)
        - Learning rate scheduling with warmup
        - Automatic checkpointing
        - Early stopping
        - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[VelocityTrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The velocity estimation model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on
        """
        self.config = config or VelocityTrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.criterion = VelocityFidelityLoss(
            l1_weight=self.config.l1_weight,
            gdl_weight=self.config.gdl_weight,
            boundary_mask_size=self.config.boundary_mask_size,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        # Create save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"VelocityTrainer initialized on {self.device}. "
            f"Effective batch size: {self.config.effective_batch_size}"
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        
        meters = {
            'loss': AverageMeter(),
            'l1_loss': AverageMeter(),
            'gdl_loss': AverageMeter(),
        }
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=False,
        )
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(
                    outputs, targets, return_components=True
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update meters
            batch_size = inputs.size(0)
            meters['loss'].update(loss_dict['total_loss'].item(), batch_size)
            meters['l1_loss'].update(loss_dict['l1_loss'].item(), batch_size)
            meters['gdl_loss'].update(loss_dict['gdl_loss'].item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{meters['loss'].avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        return {k: v.avg for k, v in meters.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of average validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        meters = {
            'val_loss': AverageMeter(),
            'val_l1_loss': AverageMeter(),
            'val_gdl_loss': AverageMeter(),
        }
        
        for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(
                    outputs, targets, return_components=True
                )
            
            batch_size = inputs.size(0)
            meters['val_loss'].update(loss_dict['total_loss'].item(), batch_size)
            meters['val_l1_loss'].update(loss_dict['l1_loss'].item(), batch_size)
            meters['val_gdl_loss'].update(loss_dict['gdl_loss'].item(), batch_size)
        
        return {k: v.avg for k, v in meters.items()}
    
    def save_checkpoint(
        self,
        filename: str = "checkpoint.pt",
        is_best: bool = False,
    ):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
        
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self) -> Dict[str, List]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary containing training and validation history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_history.append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            log_str = (
                f"Epoch {epoch + 1}/{self.config.num_epochs} "
                f"({epoch_time:.1f}s) - "
                f"train_loss: {train_metrics['loss']:.4f}"
            )
            if val_metrics:
                log_str += f" - val_loss: {val_metrics['val_loss']:.4f}"
            logger.info(log_str)
            
            # Check for best model
            current_val_loss = val_metrics.get('val_loss', train_metrics['loss'])
            is_best = current_val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = current_val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            if is_best or not self.config.save_best_only:
                self.save_checkpoint("checkpoint_latest.pt", is_best=is_best)
            
            # Early stopping
            if self.early_stopping(current_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        logger.info(f"Training complete. Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
        }


# =============================================================================
# Training Script Entry Point
# =============================================================================

def train_velocity_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[VelocityTrainingConfig] = None,
    resume_from: Optional[str] = None,
) -> Dict:
    """
    High-level training function.
    
    Args:
        model: Velocity estimation model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Training history dictionary
    """
    trainer = VelocityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    
    return trainer.train()


# =============================================================================
# Additional Loss Functions
# =============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    
    Can be combined with L1/GDL for additional perceptual quality.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 1,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        
        # Create gaussian window
        gaussian = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / (2 * sigma**2))
            for x in range(window_size)
        ])
        gaussian = gaussian / gaussian.sum()
        
        window_1d = gaussian.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        
        self.register_buffer('window', window)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class CombinedVelocityLoss(nn.Module):
    """
    Combined loss with multiple components.
    
    Combines VelocityFidelityLoss with optional SSIM loss.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        gdl_weight: float = 0.5,
        ssim_weight: float = 0.1,
        boundary_mask_size: int = 15,
    ):
        super().__init__()
        
        self.velocity_loss = VelocityFidelityLoss(
            l1_weight=l1_weight,
            gdl_weight=gdl_weight,
            boundary_mask_size=boundary_mask_size,
        )
        
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None
        self.ssim_weight = ssim_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Compute combined loss."""
        
        if return_components:
            total_loss, loss_dict = self.velocity_loss(pred, target, return_components=True)
        else:
            total_loss = self.velocity_loss(pred, target)
            loss_dict = {}
        
        if self.ssim_loss is not None and self.ssim_weight > 0:
            # Compute SSIM for each frame and average
            B, C, T, H, W = pred.shape
            pred_flat = pred.view(B * T, C, H, W)
            target_flat = target.view(B * T, C, H, W)
            
            ssim_loss = self.ssim_loss(pred_flat, target_flat)
            total_loss = total_loss + self.ssim_weight * ssim_loss
            
            if return_components:
                loss_dict['ssim_loss'] = ssim_loss.detach()
        
        if return_components:
            loss_dict['total_loss'] = total_loss.detach()
            return total_loss, loss_dict
        
        return total_loss
