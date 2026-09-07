#!/usr/bin/env python3
"""
Training script for IQ → Velocity prediction.

Trains a ComplexToRealUNet model that maps sparse complex IQ frames to
dense particle-velocity maps.  Velocity targets are generated offline
by the 2-D Loupas autocorrelator pipeline (scripts/iq_to_velocity.py).

Key Features
~~~~~~~~~~~~
- Complex-valued encoder that respects IQ phase/magnitude structure
- ConvLSTM temporal bottleneck with input-guided motion-aware flow estimation
- Multi-component loss: L1 + MSE + gradient-difference + temporal + SSIM + wave-eq
- Optional flow auxiliary losses (consistency, photometric, smoothness)
- Magnitude-mask weighting to focus on the tissue ROI
- Mixed-precision training with gradient accumulation
- Optional FiLM-based PRF conditioning (500 Hz – 20 kHz)
- TensorBoard logging, checkpointing, and early stopping

Usage
~~~~~
    python -m src.training.train_iq_to_velocity \\
        --config configs/config_iq_to_velocity.json \\
        --experiment_name iq_to_vel_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.velocity_unet import ComplexToRealUNet, create_velocity_model
from src.utils.iq_velocity_dataset import (
    IQVelocityDataset,
    create_iq_velocity_dataloaders,
)
from src.losses.physics_loss import EnhancedVelocityLoss

# Flow loss helpers from input-guided module
from src.architectures.input_guided_motion import (
    flow_consistency_loss,
    flow_smoothness_loss,
    photometric_loss,
)

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Complete training configuration for IQ → Velocity."""

    # --- Data ---
    data_dir: str = "data/phantom_velocity"
    batch_size: int = 2
    num_workers: int = 4
    iq_norm_scalar: Optional[float] = None
    vel_norm_scalar: Optional[float] = None
    per_sample_iq_norm: bool = False
    per_sample_vel_norm: bool = False

    # --- Model ---
    in_channels: int = 1
    base_channels: int = 32
    encoder_depths: Tuple[int, ...] = (32, 64, 128, 256)
    upsampling_factor: int = 12
    use_motion_interpolation: bool = True
    complex_to_real_mode: str = "concat"
    use_naive_bn: bool = True
    flow_hidden_channels: int = 32
    flow_scale: float = 0.1
    use_prf_conditioning: bool = False
    prf_hidden_dim: int = 64

    # --- Optimizer ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 5

    # --- Training ---
    num_epochs: int = 200
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    max_grad_norm: float = 1.0

    # --- Velocity loss weights ---
    l1_weight: float = 1.0
    mse_weight: float = 0.5
    gdl_weight: float = 0.3
    temporal_weight: float = 0.2
    ssim_weight: float = 0.3
    wave_weight: float = 0.05
    mask_weight: bool = True
    use_enhanced_loss: bool = True

    # --- Flow auxiliary losses ---
    use_flow_losses: bool = True
    lambda_flow_consistency: float = 0.001
    lambda_photometric: float = 0.001
    lambda_flow_smoothness: float = 0.0005
    flow_loss_warmup_epochs: int = 10

    # --- Scheduler ---
    scheduler_type: str = "CosineAnnealingWarmRestarts"
    scheduler_T0: int = 50
    scheduler_Tmult: int = 2
    scheduler_eta_min: float = 1e-6

    # --- Early stopping ---
    patience: int = 30
    min_delta: float = 1e-5

    # --- Logging / IO ---
    log_interval: int = 5
    save_interval: int = 10
    experiment_name: str = "iq_to_velocity_v2"
    output_dir: str = "outputs"

    # --- Misc ---
    seed: int = 42


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from a JSON file."""
    with open(config_path) as f:
        raw = json.load(f)

    flat: Dict = {}

    section_map = {
        "model": [
            "in_channels", "base_channels", "encoder_depths",
            "upsampling_factor", "use_motion_interpolation",
            "complex_to_real_mode", "use_naive_bn",
            "flow_hidden_channels", "flow_scale",
            "use_prf_conditioning", "prf_hidden_dim",
        ],
        "data": [
            "data_dir", "batch_size", "num_workers",
            "iq_norm_scalar", "vel_norm_scalar", "per_sample_iq_norm",
            "per_sample_vel_norm",
        ],
        "training": [
            "num_epochs", "learning_rate", "weight_decay", "betas",
            "warmup_epochs", "gradient_accumulation_steps", "use_amp",
            "max_grad_norm",
        ],
        "loss": [
            "l1_weight", "mse_weight", "gdl_weight", "temporal_weight",
            "mask_weight", "ssim_weight", "wave_weight", "use_enhanced_loss",
        ],
        "flow_losses": [
            "use_flow_losses", "lambda_flow_consistency",
            "lambda_photometric", "lambda_flow_smoothness",
            "flow_loss_warmup_epochs",
        ],
        "early_stopping": ["patience", "min_delta"],
        "logging": [
            "log_interval", "save_interval", "experiment_name", "output_dir",
        ],
    }

    for section, keys in section_map.items():
        sec_data = raw.get(section, {})
        for k in keys:
            if k in sec_data:
                flat[k] = sec_data[k]

    # Scheduler
    sched = raw.get("scheduler", {})
    if "type" in sched:
        flat["scheduler_type"] = sched["type"]
    if "T_0" in sched:
        flat["scheduler_T0"] = sched["T_0"]
    if "T_mult" in sched:
        flat["scheduler_Tmult"] = sched["T_mult"]
    if "eta_min" in sched:
        flat["scheduler_eta_min"] = sched["eta_min"]

    if "seed" in raw:
        flat["seed"] = raw["seed"]

    # Convert list → tuple where needed
    if "encoder_depths" in flat and isinstance(flat["encoder_depths"], list):
        flat["encoder_depths"] = tuple(flat["encoder_depths"])
    if "betas" in flat and isinstance(flat["betas"], list):
        flat["betas"] = tuple(flat["betas"])

    return TrainingConfig(**flat)


# ============================================================================
# Loss: Combined Velocity Loss (lightweight, no SSIM / wave)
# ============================================================================

class CombinedVelocityLoss(nn.Module):
    """
    Multi-component pixel-level loss for velocity prediction.

    Components (all optionally masked by the tissue ROI):
        L1        — |pred − target|
        MSE       — (pred − target)²
        GDL       — |∇pred − ∇target|   (spatial gradient difference)
        Temporal  — |∂pred/∂t − ∂target/∂t|
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.5,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.1,
        use_mask: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.use_mask = use_mask

    def _masked_loss(self, loss_map: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None and self.use_mask:
            while mask.dim() < loss_map.dim():
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(loss_map)
            return (loss_map * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_map.mean()

    def _spatial_gradient(self, x: torch.Tensor):
        return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        comps: Dict[str, float] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if self.l1_weight > 0:
            l1 = self._masked_loss(torch.abs(pred - target), mask)
            total = total + self.l1_weight * l1
            comps["l1"] = l1.item()

        if self.mse_weight > 0:
            mse = self._masked_loss((pred - target) ** 2, mask)
            total = total + self.mse_weight * mse
            comps["mse"] = mse.item()

        if self.gdl_weight > 0:
            pdx, pdy = self._spatial_gradient(pred)
            tdx, tdy = self._spatial_gradient(target)
            mdx = mask[..., :, 1:] if mask is not None else None
            mdy = mask[..., 1:, :] if mask is not None else None
            gdl = (
                self._masked_loss(torch.abs(pdx - tdx), mdx)
                + self._masked_loss(torch.abs(pdy - tdy), mdy)
            ) / 2.0
            total = total + self.gdl_weight * gdl
            comps["gdl"] = gdl.item()

        if self.temporal_weight > 0 and pred.shape[2] > 1:
            pdt = pred[:, :, 1:] - pred[:, :, :-1]
            tdt = target[:, :, 1:] - target[:, :, :-1]
            temp = self._masked_loss(torch.abs(pdt - tdt), mask)
            total = total + self.temporal_weight * temp
            comps["temporal"] = temp.item()

        comps["total"] = total.item()
        return total, comps


# ============================================================================
# Utilities
# ============================================================================

class AverageMeter:
    """Running average tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Halt training when validation loss plateaus."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered: no improvement for {self.counter} epochs"
                )
        return self.should_stop


# ============================================================================
# Trainer
# ============================================================================

class IQVelocityTrainer:
    """End-to-end trainer for the ComplexToRealUNet IQ → Velocity pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # Build everything
        self._create_output_dir()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()

        # Add learnable loss params (e.g. wave-eq β) to the optimizer
        criterion_params = list(self.criterion.parameters())
        if criterion_params:
            self.optimizer.add_param_group(
                {"params": criterion_params, "lr": config.learning_rate}
            )
            logger.info(
                f"Added {sum(p.numel() for p in criterion_params)} "
                "learnable loss parameters to optimizer"
            )

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)

        # TensorBoard
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(str(self.output_dir / "tensorboard"))
        else:
            self.writer = None
            logger.warning("TensorBoard not available; logging to console only")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _create_output_dir(self):
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Output directory: {self.output_dir}")

    def _setup_data(self):
        cfg = self.config
        self.train_loader, self.val_loader, self.data_info = create_iq_velocity_dataloaders(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            iq_norm_scalar=cfg.iq_norm_scalar,
            vel_norm_scalar=cfg.vel_norm_scalar,
            per_sample_iq_norm=cfg.per_sample_iq_norm,
            per_sample_vel_norm=cfg.per_sample_vel_norm,
        )
        logger.info(f"Data info: {self.data_info}")

    def _setup_model(self):
        cfg = self.config

        T_sparse = self.data_info.get("T_sparse", 5)
        T_dense = self.data_info.get("T_dense", 49)
        computed_factor = (T_dense - 1) // (T_sparse - 1) if T_sparse > 1 else T_dense
        upsampling_factor = cfg.upsampling_factor or computed_factor
        expected_T = 1 + (T_sparse - 1) * upsampling_factor

        logger.info(
            f"Temporal: T_sparse={T_sparse}, T_dense={T_dense}, "
            f"upsampling_factor={upsampling_factor}, model_output_T={expected_T}"
        )

        self.model = create_velocity_model(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            encoder_depths=list(cfg.encoder_depths),
            upsampling_factor=upsampling_factor,
            use_motion_interpolation=cfg.use_motion_interpolation,
            complex_to_real_mode=cfg.complex_to_real_mode,
            use_naive_bn=cfg.use_naive_bn,
            flow_hidden_channels=cfg.flow_hidden_channels,
            flow_scale=cfg.flow_scale,
            use_prf_conditioning=cfg.use_prf_conditioning,
            prf_hidden_dim=cfg.prf_hidden_dim,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    def _setup_optimizer(self):
        cfg = self.config
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )

        if cfg.scheduler_type == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.scheduler_T0,
                T_mult=cfg.scheduler_Tmult,
                eta_min=cfg.scheduler_eta_min,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.num_epochs,
                eta_min=cfg.scheduler_eta_min,
            )

        self.scaler = GradScaler("cuda", enabled=cfg.use_amp)

    def _setup_loss(self):
        cfg = self.config
        if cfg.use_enhanced_loss:
            self.criterion = EnhancedVelocityLoss(
                l1_weight=cfg.l1_weight,
                mse_weight=cfg.mse_weight,
                gdl_weight=cfg.gdl_weight,
                temporal_weight=cfg.temporal_weight,
                ssim_weight=cfg.ssim_weight,
                wave_weight=cfg.wave_weight,
                use_mask=cfg.mask_weight,
            ).to(self.device)
            logger.info(
                f"Using EnhancedVelocityLoss  ssim_w={cfg.ssim_weight}  "
                f"wave_w={cfg.wave_weight}"
            )
        else:
            self.criterion = CombinedVelocityLoss(
                l1_weight=cfg.l1_weight,
                mse_weight=cfg.mse_weight,
                gdl_weight=cfg.gdl_weight,
                temporal_weight=cfg.temporal_weight,
                use_mask=cfg.mask_weight,
            )

    # ------------------------------------------------------------------
    # Flow auxiliary losses
    # ------------------------------------------------------------------

    def _compute_flow_losses(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute flow auxiliary losses from cached high-res flows.

        Returns (loss_tensor, component_dict).  If flow losses are disabled
        or the model has no motion interpolation, returns (0, {}).
        """
        cfg = self.config
        if not cfg.use_flow_losses or not cfg.use_motion_interpolation:
            return torch.tensor(0.0, device=self.device), {}

        motion_interp = self.model.get_motion_interpolation()
        if motion_interp is None:
            return torch.tensor(0.0, device=self.device), {}

        # Warm-up schedule: linearly ramp flow loss weight over warmup epochs
        if self.current_epoch < cfg.flow_loss_warmup_epochs:
            ramp = self.current_epoch / max(cfg.flow_loss_warmup_epochs, 1)
        else:
            ramp = 1.0

        flows = motion_interp.get_cached_flows_high_res()
        frames = motion_interp.get_cached_frames_high_res()

        total_flow = torch.tensor(0.0, device=self.device)
        comps: Dict[str, float] = {}

        if flows:
            if cfg.lambda_flow_consistency > 0:
                fc = flow_consistency_loss(flows)
                total_flow = total_flow + ramp * cfg.lambda_flow_consistency * fc
                comps["flow_consistency"] = fc.item()

            if cfg.lambda_flow_smoothness > 0:
                fs = flow_smoothness_loss(flows)
                total_flow = total_flow + ramp * cfg.lambda_flow_smoothness * fs
                comps["flow_smoothness"] = fs.item()

            if cfg.lambda_photometric > 0 and frames:
                fp = photometric_loss(flows, frames)
                total_flow = total_flow + ramp * cfg.lambda_photometric * fp
                comps["photometric"] = fp.item()

        comps["flow_total"] = total_flow.item()
        return total_flow, comps

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        cfg = self.config

        meter_keys = ["total", "l1", "mse", "gdl", "temporal"]
        if cfg.use_enhanced_loss:
            meter_keys += ["ssim", "wave"]
        if cfg.use_flow_losses and cfg.use_motion_interpolation:
            meter_keys += ["flow_consistency", "flow_smoothness", "photometric", "flow_total"]
        meters = {k: AverageMeter() for k in meter_keys}

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{cfg.num_epochs} [Train]",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            inputs = batch["input"].to(self.device)   # (B, 2, T_sparse, H, W)
            targets = batch["target"].to(self.device)  # (B, 1, T_vel, H, W)
            masks = batch["mask"].to(self.device)      # (B, 1, 1, H, W)

            log_prf = None
            if cfg.use_prf_conditioning and "log_prf" in batch:
                log_prf = batch["log_prf"].to(self.device)

            # ----- Forward pass -----
            with autocast("cuda", enabled=cfg.use_amp):
                predictions = self.model(inputs, log_prf=log_prf)
                # (B, 1, T_model, H, W)

                T_target = targets.shape[2]
                T_pred = predictions.shape[2]
                assert T_target == T_pred, (
                    f"Temporal shape mismatch: model T={T_pred} vs target T={T_target}. "
                    "Check upsampling_factor / sparse_step alignment."
                )

                # Velocity loss
                vel_loss, vel_comps = self.criterion(predictions, targets, masks)

                # Flow auxiliary losses
                flow_loss, flow_comps = self._compute_flow_losses()

                loss = (vel_loss + flow_loss) / cfg.gradient_accumulation_steps

            # ----- Backward -----
            self.scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # ----- Meters -----
            bs = inputs.shape[0]
            all_comps = {**vel_comps, **flow_comps}
            for k in meters:
                if k in all_comps:
                    meters[k].update(all_comps[k], bs)

            pbar.set_postfix(
                loss=f"{meters['total'].avg:.4f}",
                l1=f"{meters['l1'].avg:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        cfg = self.config

        meter_keys = ["total", "l1", "mse", "gdl", "temporal"]
        if cfg.use_enhanced_loss:
            meter_keys += ["ssim", "wave"]
        meters = {k: AverageMeter() for k in meter_keys}

        prf_meters: Dict[int, Dict[str, AverageMeter]] = {}

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)

            log_prf = None
            if cfg.use_prf_conditioning and "log_prf" in batch:
                log_prf = batch["log_prf"].to(self.device)

            with autocast("cuda", enabled=cfg.use_amp):
                predictions = self.model(inputs, log_prf=log_prf)

                T_target = targets.shape[2]
                T_pred = predictions.shape[2]
                assert T_target == T_pred, (
                    f"Val temporal mismatch: model T={T_pred} vs target T={T_target}."
                )

                _, comps = self.criterion(predictions, targets, masks)

            bs = inputs.shape[0]
            for k in meters:
                if k in comps:
                    meters[k].update(comps[k], bs)

            # Per-PRF tracking
            if "prf_hz" in batch:
                for i in range(bs):
                    prf_val = int(batch["prf_hz"][i].item())
                    if prf_val not in prf_meters:
                        prf_meters[prf_val] = {k: AverageMeter() for k in meters}
                    for k in meters:
                        if k in comps:
                            prf_meters[prf_val][k].update(comps[k], 1)

        if prf_meters:
            for pv in sorted(prf_meters):
                pm = prf_meters[pv]
                logger.info(
                    f"  [Val PRF={pv}Hz]  loss={pm['total'].avg:.6f}  l1={pm['l1'].avg:.6f}"
                )

        return {k: m.avg for k, m in meters.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, is_best: bool = False, tag: str = ""):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "data_info": self.data_info,
        }

        # Save learnable loss parameters (e.g. wave-eq β)
        if hasattr(self.criterion, "state_dict") and len(list(self.criterion.parameters())) > 0:
            ckpt["criterion_state_dict"] = self.criterion.state_dict()

        ckpt_dir = self.output_dir / "checkpoints"
        path = ckpt_dir / (f"checkpoint_{tag}.pt" if tag else f"checkpoint_epoch{self.current_epoch:04d}.pt")
        torch.save(ckpt, path)

        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            logger.info(f"  ★ Saved best model (val_loss={self.best_val_loss:.6f})")
            saved_model_path = PROJECT_ROOT / "saved_models" / f"{self.config.experiment_name}_best.pt"
            torch.save(ckpt, saved_model_path)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}  "
            f"lr={lr:.2e}  "
            f"train_loss={train_metrics['total']:.6f}  "
            f"val_loss={val_metrics['total']:.6f}  "
            f"val_l1={val_metrics.get('l1', 0):.6f}"
        )

        # Log loss component scales on first epoch
        if self.current_epoch == 0:
            logger.info(
                f"  [Loss scales]  L1={train_metrics.get('l1', 0):.6f}  "
                f"MSE={train_metrics.get('mse', 0):.6f}  "
                f"GDL={train_metrics.get('gdl', 0):.6f}  "
                f"Temporal={train_metrics.get('temporal', 0):.6f}  "
                f"SSIM={train_metrics.get('ssim', 0):.6f}  "
                f"Wave={train_metrics.get('wave', 0):.6f}"
            )
            if self.config.use_flow_losses:
                logger.info(
                    f"  [Flow scales]  consistency={train_metrics.get('flow_consistency', 0):.6f}  "
                    f"smoothness={train_metrics.get('flow_smoothness', 0):.6f}  "
                    f"photometric={train_metrics.get('photometric', 0):.6f}"
                )

        # Log flow stats from the model
        flow_stats = self.model.get_flow_stats()
        if flow_stats and self.current_epoch % self.config.log_interval == 0:
            logger.info(
                f"  [Flow]  mean={flow_stats.get('flow_mean', 0):.3f}  "
                f"max={flow_stats.get('flow_max', 0):.3f}  "
                f"max_param={flow_stats.get('max_flow_param', 0):.1f}"
            )

        if self.writer is not None:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, self.current_epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
            self.writer.add_scalar("lr", lr, self.current_epoch)
            for k, v in flow_stats.items():
                self.writer.add_scalar(f"flow/{k}", v, self.current_epoch)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def train(self):
        """Run the full training loop."""
        cfg = self.config
        logger.info("=" * 60)
        logger.info("Starting IQ → Velocity Training (v2)")
        logger.info("=" * 60)

        start = time.time()

        for epoch in range(cfg.num_epochs):
            self.current_epoch = epoch

            train_metrics = self._train_one_epoch()
            val_metrics = self._validate()
            self.scheduler.step()
            self._log_metrics(train_metrics, val_metrics)

            val_loss = val_metrics["total"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if is_best or (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(is_best=is_best)

            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(
            f"Training complete in {elapsed / 60:.1f} min.  "
            f"Best val_loss: {self.best_val_loss:.6f}"
        )
        logger.info("=" * 60)

        if self.writer:
            self.writer.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train IQ → Velocity model")
    parser.add_argument(
        "--config", type=str, default="configs/config_iq_to_velocity.json",
        help="Path to config JSON",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    args = parser.parse_args()

    config = load_config(args.config)

    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    trainer = IQVelocityTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
