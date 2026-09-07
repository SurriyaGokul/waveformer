#!/usr/bin/env python3
"""
Training script for WavePINN v5.

Builds on train_iq_to_velocity.py with three changes:

1. Uses ``ComplexToRealUNetV5`` (temporal skip refinement at levels 3 & 4)
2. Uses ``EnhancedVelocityLossV3`` with dt-aware wave regularization
3. Uses ``IQVelocityDatasetV3`` for PRF-diverse training
4. Wave-equation weight ramps from 0 to ``wave_weight_max`` over training
5. Physics loss parameters use a reduced learning rate (0.1x by default)

Usage
~~~~~
    python -m src.training.train_wavepinn_v5 \\
        --config configs/config_wavepinn_v5.json
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v5 import ComplexToRealUNetV5, create_wavepinn_v5
from src.utils.iq_velocity_dataset_v3 import (
    create_iq_velocity_dataloaders_v3,
)
from src.losses.physics_loss_v3 import EnhancedVelocityLossV3
from src.architectures.input_guided_motion import (
    flow_consistency_loss,
    flow_smoothness_loss,
    photometric_loss,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore[assignment,misc]

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
class TrainingConfigV5:
    """Training configuration for WavePINN v5."""

    # --- Data ---
    data_dir: str = "data/fullscale_velocity"
    batch_size: int = 2
    num_workers: int = 4
    iq_norm_scalar: Optional[float] = None
    vel_norm_scalar: Optional[float] = None
    per_sample_iq_norm: bool = True
    per_sample_vel_norm: bool = True
    enable_prf_augmentation: bool = True
    augmentation_steps: List[int] = field(default_factory=lambda: [6, 8, 10, 12])
    augmentation_step_weights: Dict[str, float] = field(
        default_factory=lambda: {"6": 0.15, "8": 0.15, "10": 0.20, "12": 0.50}
    )

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
    skip_refine_levels: List[int] = field(default_factory=lambda: [3, 4])
    skip_refine_kernel: int = 5

    # --- Optimizer ---
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 5
    physics_lr_multiplier: float = 0.1

    # --- Training ---
    num_epochs: int = 250
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    max_grad_norm: float = 1.0

    # --- Wave ramp ---
    wave_ramp_start_epoch: int = 50
    wave_ramp_end_epoch: int = 200
    wave_weight_max: float = 0.05

    # --- Loss weights ---
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    gdl_weight: float = 0.3
    temporal_weight: float = 0.15
    ssim_weight: float = 0.0
    wave_weight: float = 0.0  # initial (ramped)
    use_mask: bool = True
    init_cs: float = 3.0
    dx_m: float = 1.013e-4
    dy_m: float = 5.07e-5

    # --- Flow losses ---
    use_flow_losses: bool = False
    lambda_flow_consistency: float = 0.0
    lambda_photometric: float = 0.0
    lambda_flow_smoothness: float = 0.0
    flow_loss_warmup_epochs: int = 0

    # --- Scheduler ---
    scheduler_type: str = "CosineAnnealingWarmRestarts"
    scheduler_T0: int = 50
    scheduler_Tmult: int = 2
    scheduler_eta_min: float = 1e-6

    # --- Early stopping ---
    patience: int = 40
    min_delta: float = 1e-6

    # --- Logging ---
    log_interval: int = 5
    save_interval: int = 10
    experiment_name: str = "wavepinn_v5"
    output_dir: str = "outputs"

    seed: int = 42


def load_config_v5(config_path: str) -> TrainingConfigV5:
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
            "skip_refine_levels", "skip_refine_kernel",
        ],
        "data": [
            "data_dir", "batch_size", "num_workers",
            "iq_norm_scalar", "vel_norm_scalar", "per_sample_iq_norm",
            "per_sample_vel_norm", "enable_prf_augmentation",
            "augmentation_steps", "augmentation_step_weights",
        ],
        "training": [
            "num_epochs", "learning_rate", "weight_decay", "betas",
            "warmup_epochs", "gradient_accumulation_steps", "use_amp",
            "max_grad_norm", "wave_ramp_start_epoch", "wave_ramp_end_epoch",
            "wave_weight_max", "physics_lr_multiplier",
        ],
        "loss": [
            "l1_weight", "mse_weight", "gdl_weight", "temporal_weight",
            "ssim_weight", "wave_weight", "use_mask", "init_cs", "dx_m", "dy_m",
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

    return TrainingConfigV5(**flat)


# ============================================================================
# Utilities
# ============================================================================

class AverageMeter:
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

class WavePINNv5Trainer:
    """End-to-end trainer for WavePINN v5."""

    def __init__(self, config: TrainingConfigV5):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self._create_output_dir()
        self._setup_file_logger()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)

        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(str(self.output_dir / "tensorboard"))
        else:
            self.writer = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _create_output_dir(self):
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        logger.info(f"Output directory: {self.output_dir}")

    def _setup_file_logger(self):
        """Attach a per-run file handler so all logs are persisted."""
        self.log_file_path = self.output_dir / "training.log"
        self.metrics_jsonl_path = self.output_dir / "epoch_metrics.jsonl"

        existing_handler = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == self.log_file_path:
                existing_handler = h
                break

        if existing_handler is None:
            fh = logging.FileHandler(self.log_file_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        logger.info(f"Training log file: {self.log_file_path}")
        logger.info(f"Epoch metrics JSONL: {self.metrics_jsonl_path}")

    def _setup_data(self):
        cfg = self.config
        # Convert string-keyed weights to int-keyed
        step_weights = None
        if cfg.augmentation_step_weights:
            step_weights = {int(k): v for k, v in cfg.augmentation_step_weights.items()}

        self.train_loader, self.val_loader, self.data_info = (
            create_iq_velocity_dataloaders_v3(
                data_dir=cfg.data_dir,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                enable_prf_augmentation=cfg.enable_prf_augmentation,
                augmentation_steps=cfg.augmentation_steps,
                augmentation_step_weights=step_weights,
                upsampling_factor=cfg.upsampling_factor,
                iq_norm_scalar=cfg.iq_norm_scalar,
                vel_norm_scalar=cfg.vel_norm_scalar,
                per_sample_iq_norm=cfg.per_sample_iq_norm,
                per_sample_vel_norm=cfg.per_sample_vel_norm,
            )
        )
        logger.info(f"Data info: {self.data_info}")

    def _setup_model(self):
        cfg = self.config
        self.model = create_wavepinn_v5(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            encoder_depths=list(cfg.encoder_depths),
            upsampling_factor=cfg.upsampling_factor,
            use_motion_interpolation=cfg.use_motion_interpolation,
            complex_to_real_mode=cfg.complex_to_real_mode,
            use_naive_bn=cfg.use_naive_bn,
            flow_hidden_channels=cfg.flow_hidden_channels,
            flow_scale=cfg.flow_scale,
            use_prf_conditioning=cfg.use_prf_conditioning,
            prf_hidden_dim=cfg.prf_hidden_dim,
            skip_refine_levels=cfg.skip_refine_levels,
            skip_refine_kernel=cfg.skip_refine_kernel,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

        component_params = {
            "encoder": sum(
                p.numel()
                for m in [self.model.enc1, self.model.enc2, self.model.enc3, self.model.enc4]
                for p in m.parameters()
                if p.requires_grad
            ),
            "bottleneck": sum(
                p.numel() for p in self.model.bottleneck.parameters() if p.requires_grad
            ),
            "skip_refiners": sum(
                p.numel() for p in self.model._skip_refiners.parameters() if p.requires_grad
            ),
            "decoder": sum(
                p.numel()
                for m in [self.model.dec4, self.model.dec3, self.model.dec2, self.model.dec1]
                for p in m.parameters()
                if p.requires_grad
            ),
            "heads_and_bridges": sum(
                p.numel()
                for m in [
                    self.model.bottleneck_bridge,
                    self.model.skip_bridge_l1,
                    self.model.skip_bridge_l2,
                    self.model.skip_bridge_l3,
                    self.model.skip_bridge_l4,
                    self.model.output_conv,
                ]
                for p in m.parameters()
                if p.requires_grad
            ),
        }
        logger.info(
            "Architecture param split: "
            + "  ".join(f"{k}={v:,}" for k, v in component_params.items())
        )

    def _setup_loss(self):
        cfg = self.config
        self.criterion = EnhancedVelocityLossV3(
            l1_weight=cfg.l1_weight,
            mse_weight=cfg.mse_weight,
            gdl_weight=cfg.gdl_weight,
            temporal_weight=cfg.temporal_weight,
            ssim_weight=cfg.ssim_weight,
            wave_weight=cfg.wave_weight,
            use_mask=cfg.use_mask,
            init_cs=cfg.init_cs,
            dx_m=cfg.dx_m,
            dy_m=cfg.dy_m,
        ).to(self.device)
        logger.info(
            f"Loss: L1={cfg.l1_weight} MSE={cfg.mse_weight} GDL={cfg.gdl_weight} "
            f"Temporal={cfg.temporal_weight} Wave=0→{cfg.wave_weight_max} "
            f"(ramp {cfg.wave_ramp_start_epoch}→{cfg.wave_ramp_end_epoch})"
        )
        logger.info(
            f"Physics: init_cs={cfg.init_cs:.3f} m/s dx={cfg.dx_m:.3e} m dy={cfg.dy_m:.3e} m"
        )

    def _setup_optimizer(self):
        cfg = self.config
        physics_params = list(self.criterion.parameters())

        self.optimizer = AdamW(
            [
                {
                    "params": self.model.parameters(),
                    "lr": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                },
                {
                    "params": physics_params,
                    "lr": cfg.learning_rate * cfg.physics_lr_multiplier,
                    "weight_decay": 0.0,
                },
            ],
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

    @staticmethod
    def _module_grad_norm(modules: List[nn.Module]) -> float:
        total = 0.0
        for m in modules:
            for p in m.parameters():
                if p.grad is not None:
                    total += float((p.grad.detach() ** 2).sum().item())
        return total ** 0.5

    def _append_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        payload = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "physics_learning_rate": self.optimizer.param_groups[1]["lr"],
            "wave_weight": self._get_wave_weight(self.current_epoch),
            "learned_cs": self.criterion.wave_reg.cs_value,
            "train": train_metrics,
            "val": val_metrics,
            "skip_refiner_alpha": {
                f"L{k}": float(v.alpha.item()) for k, v in self.model._skip_refiners.items()
            },
        }
        with open(self.metrics_jsonl_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    # ------------------------------------------------------------------
    # Wave weight ramping
    # ------------------------------------------------------------------

    def _get_wave_weight(self, epoch: int) -> float:
        """Linear ramp from 0 to wave_weight_max."""
        cfg = self.config
        if epoch < cfg.wave_ramp_start_epoch:
            return 0.0
        if epoch >= cfg.wave_ramp_end_epoch:
            return cfg.wave_weight_max
        progress = (epoch - cfg.wave_ramp_start_epoch) / max(
            cfg.wave_ramp_end_epoch - cfg.wave_ramp_start_epoch, 1
        )
        return cfg.wave_weight_max * progress

    # ------------------------------------------------------------------
    # Flow losses (optional, inherited from v2)
    # ------------------------------------------------------------------

    def _compute_flow_losses(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        if not cfg.use_flow_losses or not cfg.use_motion_interpolation:
            return torch.tensor(0.0, device=self.device), {}

        motion_interp = self.model.get_motion_interpolation()
        if motion_interp is None:
            return torch.tensor(0.0, device=self.device), {}

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

        # Update wave weight for this epoch
        wave_w = self._get_wave_weight(self.current_epoch)
        self.criterion.set_weights(wave=wave_w)

        meter_keys = [
            "total", "l1", "mse", "gdl", "temporal", "wave", "wave_cs",
            "pred_mean", "pred_std", "pred_abs_max", "target_std", "mask_frac",
            "grad_total", "grad_encoder", "grad_bottleneck", "grad_decoder", "grad_skip_refiner",
        ]
        meters = {k: AverageMeter() for k in meter_keys}

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{cfg.num_epochs} [Train]",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)
            dt_output = batch["dt_output"].to(self.device)

            log_prf = None
            if cfg.use_prf_conditioning and "log_prf" in batch:
                log_prf = batch["log_prf"].to(self.device)

            with autocast("cuda", enabled=cfg.use_amp):
                predictions = self.model(inputs, log_prf=log_prf)

                if not torch.isfinite(predictions).all():
                    raise RuntimeError("Non-finite predictions detected. Potential architecture/instability issue.")

                T_target = targets.shape[2]
                T_pred = predictions.shape[2]
                assert T_target == T_pred, (
                    f"Temporal shape mismatch: model T={T_pred} vs target T={T_target}. "
                    "Check upsampling_factor / sparse_step alignment."
                )

                vel_loss, vel_comps = self.criterion(
                    predictions,
                    targets,
                    masks,
                    dt_seconds=dt_output,
                )
                flow_loss, flow_comps = self._compute_flow_losses()
                loss = (vel_loss + flow_loss) / cfg.gradient_accumulation_steps

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss detected. Aborting to avoid corrupt training.")

            if step == 0:
                logger.info(
                    f"  [Batch0 diagnostics] input={tuple(inputs.shape)} target={tuple(targets.shape)} "
                    f"pred={tuple(predictions.shape)} mask={tuple(masks.shape)} dt_output_mean={dt_output.mean().item():.6f}"
                )
                logger.info(
                    f"  [Batch0 ranges] input_std={inputs.std().item():.4f} target_std={targets.std().item():.4f} "
                    f"pred_std={predictions.std().item():.4f} pred_abs_max={predictions.abs().max().item():.4f}"
                )

            self.scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)

                grad_encoder = self._module_grad_norm(
                    [self.model.enc1, self.model.enc2, self.model.enc3, self.model.enc4]
                )
                grad_bottleneck = self._module_grad_norm([self.model.bottleneck])
                grad_decoder = self._module_grad_norm(
                    [self.model.dec4, self.model.dec3, self.model.dec2, self.model.dec1, self.model.output_conv]
                )
                grad_skip_refiner = self._module_grad_norm([self.model._skip_refiners])
                grad_total = self._module_grad_norm([self.model, self.criterion])

                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.criterion.parameters()),
                    cfg.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                meters["grad_total"].update(grad_total, 1)
                meters["grad_encoder"].update(grad_encoder, 1)
                meters["grad_bottleneck"].update(grad_bottleneck, 1)
                meters["grad_decoder"].update(grad_decoder, 1)
                meters["grad_skip_refiner"].update(grad_skip_refiner, 1)

            bs = inputs.shape[0]
            pred_stats = {
                "pred_mean": predictions.mean().item(),
                "pred_std": predictions.std().item(),
                "pred_abs_max": predictions.abs().max().item(),
                "target_std": targets.std().item(),
                "mask_frac": masks.mean().item(),
            }

            all_comps = {**vel_comps, **flow_comps, **pred_stats}
            for k in meters:
                if k in all_comps:
                    meters[k].update(all_comps[k], bs)

            pbar.set_postfix(
                loss=f"{meters['total'].avg:.4f}",
                l1=f"{meters['l1'].avg:.4f}",
                wave_w=f"{wave_w:.4f}",
                cs=f"{meters['wave_cs'].avg:.3f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        cfg = self.config

        meter_keys = [
            "total", "l1", "mse", "gdl", "temporal", "wave", "wave_cs",
            "pred_mean", "pred_std", "pred_abs_max",
        ]
        meters = {k: AverageMeter() for k in meter_keys}
        prf_meters: Dict[int, Dict[str, AverageMeter]] = {}

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)
            dt_output = batch["dt_output"].to(self.device)

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

                _, comps = self.criterion(
                    predictions,
                    targets,
                    masks,
                    dt_seconds=dt_output,
                )

            comps["pred_mean"] = predictions.mean().item()
            comps["pred_std"] = predictions.std().item()
            comps["pred_abs_max"] = predictions.abs().max().item()

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
                    f"  [Val PRF={pv}Hz]  loss={pm['total'].avg:.6f}  "
                    f"l1={pm['l1'].avg:.6f}"
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
            "criterion_state_dict": self.criterion.state_dict(),
            "best_val_loss": self.best_val_loss,
            "learned_cs": self.criterion.wave_reg.cs_value,
            "config": asdict(self.config),
            "data_info": self.data_info,
        }

        ckpt_dir = self.output_dir / "checkpoints"
        path = ckpt_dir / (
            f"checkpoint_{tag}.pt" if tag
            else f"checkpoint_epoch{self.current_epoch:04d}.pt"
        )
        torch.save(ckpt, path)

        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            logger.info(f"  * Saved best model (val_loss={self.best_val_loss:.6f})")
            saved_model_path = PROJECT_ROOT / "saved_models" / f"{self.config.experiment_name}_best.pt"
            saved_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, saved_model_path)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        lr = self.optimizer.param_groups[0]["lr"]
        wave_w = self._get_wave_weight(self.current_epoch)
        physics_lr = self.optimizer.param_groups[1]["lr"]
        learned_cs = self.criterion.wave_reg.cs_value
        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}  "
            f"lr={lr:.2e}  physics_lr={physics_lr:.2e}  wave_w={wave_w:.4f}  "
            f"train_loss={train_metrics['total']:.6f}  "
            f"val_loss={val_metrics['total']:.6f}  "
            f"val_l1={val_metrics.get('l1', 0):.6f}  cs={learned_cs:.4f}"
        )

        logger.info(
            f"  [Train components] mse={train_metrics.get('mse', 0):.6f}  "
            f"gdl={train_metrics.get('gdl', 0):.6f}  temporal={train_metrics.get('temporal', 0):.6f}  "
            f"wave={train_metrics.get('wave', 0):.6f}  pred_std={train_metrics.get('pred_std', 0):.6f}  "
            f"pred_abs_max={train_metrics.get('pred_abs_max', 0):.6f}"
        )
        logger.info(
            f"  [Val components]   mse={val_metrics.get('mse', 0):.6f}  "
            f"gdl={val_metrics.get('gdl', 0):.6f}  temporal={val_metrics.get('temporal', 0):.6f}  "
            f"wave={val_metrics.get('wave', 0):.6f}  pred_std={val_metrics.get('pred_std', 0):.6f}  "
            f"pred_abs_max={val_metrics.get('pred_abs_max', 0):.6f}"
        )
        logger.info(
            f"  [Grad norms] total={train_metrics.get('grad_total', 0):.4f}  "
            f"encoder={train_metrics.get('grad_encoder', 0):.4f}  "
            f"bottleneck={train_metrics.get('grad_bottleneck', 0):.4f}  "
            f"decoder={train_metrics.get('grad_decoder', 0):.4f}  "
            f"skip_refiner={train_metrics.get('grad_skip_refiner', 0):.4f}"
        )

        if self.current_epoch == 0:
            logger.info(
                f"  [Loss scales]  L1={train_metrics.get('l1', 0):.6f}  "
                f"MSE={train_metrics.get('mse', 0):.6f}  "
                f"GDL={train_metrics.get('gdl', 0):.6f}  "
                f"Temporal={train_metrics.get('temporal', 0):.6f}  "
                f"Wave={train_metrics.get('wave', 0):.6f}"
            )

        # Log skip refiner alpha values
        for lvl_key, refiner in self.model._skip_refiners.items():
            alpha = refiner.alpha.item()
            logger.info(f"  [Skip refiner L{lvl_key}]  alpha={alpha:.6f}")

        if train_metrics.get("pred_std", 1.0) < 1e-3:
            logger.warning(
                "  [Alert] Prediction std is near-zero. Potential collapse or architecture issue."
            )
        if np.isnan(val_metrics.get("total", np.nan)) or np.isinf(val_metrics.get("total", np.nan)):
            logger.error("  [Alert] Validation loss became non-finite.")

        flow_stats = self.model.get_flow_stats()
        if flow_stats and self.current_epoch % self.config.log_interval == 0:
            logger.info(
                f"  [Flow]  mean={flow_stats.get('flow_mean', 0):.3f}  "
                f"max={flow_stats.get('flow_max', 0):.3f}"
            )

        self._append_epoch_metrics(train_metrics, val_metrics)

        if self.writer is not None:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, self.current_epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
            self.writer.add_scalar("lr", lr, self.current_epoch)
            self.writer.add_scalar("physics_lr", physics_lr, self.current_epoch)
            self.writer.add_scalar("wave_weight", wave_w, self.current_epoch)
            self.writer.add_scalar("physics/cs", learned_cs, self.current_epoch)
            for lvl_key, refiner in self.model._skip_refiners.items():
                self.writer.add_scalar(
                    f"skip_refiner/alpha_L{lvl_key}",
                    refiner.alpha.item(),
                    self.current_epoch,
                )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.config
        logger.info("=" * 60)
        logger.info("Starting WavePINN v5 Training")
        logger.info(f"  Skip refinement: levels {cfg.skip_refine_levels}")
        logger.info(f"  Wave equation ramp: 0 -> {cfg.wave_weight_max} "
                     f"(epochs {cfg.wave_ramp_start_epoch}-{cfg.wave_ramp_end_epoch})")
        logger.info(f"  Physics LR multiplier: {cfg.physics_lr_multiplier}")
        logger.info(f"  PRF augmentation: {cfg.enable_prf_augmentation}")
        logger.info(f"  dt-aware physics loss with learnable global cs regularizer")
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
    parser = argparse.ArgumentParser(description="Train WavePINN v5")
    parser.add_argument(
        "--config", type=str, default="configs/config_wavepinn_v5.json",
        help="Path to config JSON",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    args = parser.parse_args()
    config = load_config_v5(args.config)

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

    trainer = WavePINNv5Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
