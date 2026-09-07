"""
Training script for WavePINN v7 — Multi-Task SWE-Optimised.

Usage:
    python -m src.training.train_wavepinn_v7 --config configs/config_wavepinn_v7.json
    python -m src.training.train_wavepinn_v7 --config configs/config_wavepinn_v7.json --resume outputs/wavepinn_v7/checkpoint.pt
    python -m src.training.train_wavepinn_v7 --config configs/config_wavepinn_v7.json --warmstart saved_models/wavepinn_v6_best.pt

Three-Phase Curriculum:
    Phase 1 (Velocity pretrain):  Epochs 1-40,   L1+MSE+GDL+temporal + sparse(0->0.1) + envelope(0->0.2)
    Phase 2 (Physics activation): Epochs 41-120, + wave(0->0.05) + toa(0->0.5) + cs_map(0->0.3) + cs_mean(0->0.2)
    Phase 3 (Coupled finetune):   Epochs 121-180, LR drop, all weights stable

Key differences from V6 trainer:
    - Two-param-group optimizer: backbone lr=3e-4, physics head lr=1e-3
    - Model returns (velocity, cs_map) tuple — handled in train/val loops
    - V7 loss receives vel_scale, dt_output, cs_target_scalar, push_col, cs_map_pred
    - PhantomStratifiedBatchSampler for phantom-diverse batches
    - Warm-start from V6 checkpoint (strict=False, fresh optimizer)
    - Extended val metrics: r_active, cs_mae_head
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from src.architectures.wavepinn_v7 import WavePINNv7, create_wavepinn_v7, count_parameters_v7
from src.losses.swe_loss_v7 import EnhancedVelocityLossV7
from src.utils.iq_velocity_dataset_v3 import create_iq_velocity_dataloaders_v3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =========================================================================
# Phantom-Stratified Batch Sampler
# =========================================================================

class PhantomStratifiedBatchSampler(Sampler):
    """
    Batch sampler ensuring each batch has samples from >= 2 phantom types.

    Groups samples by phantom source (2ft, 3ft, cirs, etc.), then constructs
    batches by drawing from multiple groups.  When fewer than 2 groups have
    remaining samples, falls back to random sampling.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Build group indices
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            source = sample.get("source_phantom", "unknown")
            self.groups[source].append(i)

        self.group_names = list(self.groups.keys())
        self.n_samples = len(dataset)

        logger.info(
            f"PhantomStratifiedBatchSampler: {len(self.group_names)} groups, "
            f"{self.n_samples} samples, batch_size={batch_size}"
        )
        for name, indices in self.groups.items():
            logger.info(f"  {name}: {len(indices)} samples")

    def __iter__(self):
        # Shuffle within each group
        rng = np.random.default_rng()
        group_pools = {
            name: list(rng.permutation(indices))
            for name, indices in self.groups.items()
        }
        group_names = [n for n in self.group_names if len(group_pools[n]) > 0]

        batches = []
        while True:
            active = [n for n in group_names if len(group_pools[n]) > 0]
            if len(active) < 2:
                # Not enough groups for stratification — drain remaining
                remaining = []
                for n in active:
                    remaining.extend(group_pools[n])
                    group_pools[n] = []
                rng.shuffle(remaining)
                while len(remaining) >= self.batch_size:
                    batches.append(remaining[:self.batch_size])
                    remaining = remaining[self.batch_size:]
                if remaining and not self.drop_last:
                    batches.append(remaining)
                break

            # Draw from groups round-robin
            batch = []
            draw_order = list(rng.permutation(active))
            gi = 0
            while len(batch) < self.batch_size:
                if gi >= len(draw_order):
                    # Refresh active groups
                    draw_order = [n for n in draw_order if len(group_pools[n]) > 0]
                    if not draw_order:
                        break
                    gi = 0
                name = draw_order[gi]
                if group_pools[name]:
                    batch.append(group_pools[name].pop())
                gi += 1

            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last and batch:
                batches.append(batch)
            else:
                # Not enough samples left
                break

        # Shuffle batch order
        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# =========================================================================
# Curriculum Phase Logic
# =========================================================================

def get_phase(epoch: int, curriculum: Dict) -> Tuple[str, Dict]:
    """Return (phase_name, phase_config) for 1-based epoch."""
    for key in sorted(curriculum.keys()):
        if not key.startswith("phase_"):
            continue
        phase = curriculum[key]
        start, end = phase["epochs"]
        if start <= epoch <= end:
            return phase["name"], phase
    last_key = sorted(k for k in curriculum if k.startswith("phase_"))[-1]
    return curriculum[last_key]["name"], curriculum[last_key]


def _cosine_ramp(epoch: int, start_epoch: int, ramp_end: int, v_start: float, v_end: float) -> float:
    """Cosine ramp from v_start to v_end over [start_epoch, ramp_end], then plateau."""
    total = max(ramp_end - start_epoch, 1)
    progress = min(1.0, max(0.0, (epoch - start_epoch) / total))
    return v_start + (v_end - v_start) * (1 - math.cos(math.pi * progress)) / 2


def get_loss_weights(epoch: int, phase_cfg: Dict) -> Dict[str, float]:
    """Resolve loss weights for current epoch, handling cosine ramps."""
    weights = dict(phase_cfg["loss_weights"])
    start_epoch = phase_cfg["epochs"][0]

    # Process all ramp pairs: <name>_start / <name>_end
    ramp_keys = [k.replace("_start", "") for k in list(weights.keys()) if k.endswith("_start")]
    for name in ramp_keys:
        v_start = weights.pop(f"{name}_start")
        v_end = weights.pop(f"{name}_end")
        ramp_end_key = f"{name}_ramp_end_epoch"
        ramp_end = phase_cfg.get(ramp_end_key, phase_cfg["epochs"][1])
        weights[name] = _cosine_ramp(epoch, start_epoch, ramp_end, v_start, v_end)

    return weights


def get_lr_for_phase(epoch: int, phase_cfg: Dict, base_lr: float) -> float:
    """Compute learning rate with warmup + cosine decay within phase."""
    phase_lr = phase_cfg.get("lr", base_lr)
    start_epoch, end_epoch = phase_cfg["epochs"]

    warmup = phase_cfg.get("warmup_epochs", 0)
    if warmup > 0 and (epoch - start_epoch) < warmup:
        return phase_lr * (epoch - start_epoch + 1) / warmup

    total = max(end_epoch - start_epoch - warmup, 1)
    progress = min(1.0, (epoch - start_epoch - warmup) / total)
    min_lr = 1e-6
    return min_lr + (phase_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


# =========================================================================
# Validation Metrics
# =========================================================================

def compute_validation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cs_map_pred: Optional[torch.Tensor],
    cs_target_scalar: Optional[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute SWE-relevant validation metrics beyond loss.

    Returns dict with:
      r_active: Pearson correlation in active window (|target|>0.1)
      r_full: Pearson correlation over full volume
      cs_mae_head: MAE of physics head Cs vs ground truth (if available)
    """
    metrics: Dict[str, float] = {}
    B = pred.shape[0]

    # Flatten pred/target to compute correlations
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() if mask is not None else None

    r_active_list = []
    r_full_list = []

    for b in range(B):
        p = pred_np[b, 0]  # (T, H, W)
        t = tgt_np[b, 0]
        if mask_np is not None:
            m = mask_np[b, 0, 0].astype(bool)  # (H, W)
        else:
            m = np.ones(t.shape[1:], dtype=bool)

        # Full correlation (masked)
        p_flat = p[:, m].flatten()
        t_flat = t[:, m].flatten()
        if len(p_flat) > 10 and np.std(p_flat) > 1e-8 and np.std(t_flat) > 1e-8:
            r_full_list.append(float(np.corrcoef(p_flat, t_flat)[0, 1]))

        # Active window: first upsampling_factor frames, |target|>0.1
        T_dense = t.shape[0]
        n_active = min(T_dense, 12)  # first 12 frames ≈ 1 input interval
        active_mask_3d = (np.abs(t[:n_active]) > 0.1) & m[None, :, :]
        p_active = p[:n_active][active_mask_3d]
        t_active = t[:n_active][active_mask_3d]
        if len(p_active) > 10 and np.std(p_active) > 1e-8 and np.std(t_active) > 1e-8:
            r_active_list.append(float(np.corrcoef(p_active, t_active)[0, 1]))

    if r_active_list:
        metrics["r_active"] = float(np.mean(r_active_list))
    if r_full_list:
        metrics["r_full"] = float(np.mean(r_full_list))

    # Cs head diagnostics (always, not just when cs targets valid)
    if cs_map_pred is not None:
        metrics["cs_map_pred_mean"] = float(cs_map_pred.mean().item())
        metrics["cs_map_pred_std"] = float(cs_map_pred.std().item())

    # Cs head MAE
    if cs_map_pred is not None and cs_target_scalar is not None:
        valid = cs_target_scalar > 0
        if valid.any():
            cs_pred_mean = cs_map_pred[valid].mean(dim=(1, 2, 3))  # (B_valid,)
            cs_tgt = cs_target_scalar[valid]
            metrics["cs_mae_head"] = float((cs_pred_mean - cs_tgt).abs().mean().item())

    return metrics


# =========================================================================
# Trainer
# =========================================================================

class TrainerV7:
    """Training orchestrator for WavePINN v7 multi-task."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        exp_name = config.get("logging", {}).get("experiment_name", "wavepinn_v7")
        output_dir = config.get("logging", {}).get("output_dir", "outputs")
        self.output_dir = Path(output_dir) / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(self.output_dir / "training.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.writer = SummaryWriter(self.output_dir / "tensorboard") if HAS_TENSORBOARD else None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.current_phase = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Initialise model, data, loss, optimiser."""
        self._setup_data()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        # AMP is only meaningful on CUDA; disable silently on CPU/MPS to avoid
        # GradScaler('cuda') crashing on non-CUDA environments.
        self.device_type = self.device.type
        use_amp = self.config["training"].get("use_amp", True) and self.device_type == "cuda"
        if use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            if self.config["training"].get("use_amp", True) and self.device_type != "cuda":
                logger.info("AMP requested but device is not CUDA — disabling.")
        self._log_setup()

    def _setup_data(self):
        data_cfg = self.config["data"]
        use_stratified = self.config.get("training", {}).get("use_stratified_sampler", True)

        if use_stratified:
            # Build datasets manually for stratified sampler
            from src.utils.iq_velocity_dataset_v3 import (
                IQVelocityDatasetV3,
                collate_fn_v3,
            )

            upsampling = self.config["model"].get("upsampling_factor", 12)
            ds_kwargs = dict(
                data_dir=data_cfg["data_dir"],
                enable_prf_augmentation=data_cfg.get("enable_prf_augmentation", True),
                augmentation_steps=data_cfg.get("augmentation_steps"),
                augmentation_step_weights=data_cfg.get("augmentation_step_weights"),
                upsampling_factor=upsampling,
                per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
                per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            )

            train_dataset = IQVelocityDatasetV3(split="train", **ds_kwargs)

            val_dataset = IQVelocityDatasetV3(
                data_dir=data_cfg["data_dir"],
                split="val",
                enable_prf_augmentation=False,
                upsampling_factor=upsampling,
                iq_norm_scalar=train_dataset.iq_norm_scalar,
                vel_norm_scalar=train_dataset.vel_norm_scalar,
                per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
                per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            )

            batch_size = data_cfg.get("batch_size", 2)
            num_workers = data_cfg.get("num_workers", 4)
            pin_memory = True

            sampler = PhantomStratifiedBatchSampler(
                train_dataset, batch_size=batch_size, drop_last=True,
            )

            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn_v3,
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=collate_fn_v3,
            )

            self.data_info = train_dataset.get_data_info()
            self.data_info["num_val"] = len(val_dataset)
        else:
            self.train_loader, self.val_loader, self.data_info = create_iq_velocity_dataloaders_v3(
                data_dir=data_cfg["data_dir"],
                batch_size=data_cfg.get("batch_size", 2),
                num_workers=data_cfg.get("num_workers", 4),
                enable_prf_augmentation=data_cfg.get("enable_prf_augmentation", True),
                augmentation_steps=data_cfg.get("augmentation_steps"),
                augmentation_step_weights=data_cfg.get("augmentation_step_weights"),
                upsampling_factor=self.config["model"].get("upsampling_factor", 12),
                per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
                per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            )

    def _setup_model(self):
        model_cfg = dict(self.config["model"])
        model_cfg.pop("name", None)
        self.model = create_wavepinn_v7(config=model_cfg)
        self.model = self.model.to(self.device)

        params = count_parameters_v7(self.model)
        logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    def _setup_criterion(self):
        p1_weights = self.config["curriculum"]["phase_1"]["loss_weights"]
        loss_cfg = self.config.get("loss", {})
        self.criterion = EnhancedVelocityLossV7(
            # V6 heritage
            l1_weight=p1_weights.get("l1", 1.0),
            mse_weight=p1_weights.get("mse", 1.0),
            gdl_weight=p1_weights.get("gdl", 0.3),
            temporal_weight=p1_weights.get("temporal", 0.15),
            wave_weight=p1_weights.get("wave", p1_weights.get("wave_start", 0.0)),
            amplitude_weighting=loss_cfg.get("amplitude_weighting", True),
            amplitude_weight_gamma=loss_cfg.get("amplitude_weight_gamma", 1.0),
            amplitude_weight_bias=loss_cfg.get("amplitude_weight_bias", 0.01),
            # V7 new
            toa_weight=p1_weights.get("toa", 0.0),
            cs_map_weight=p1_weights.get("cs_map", 0.0),
            cs_mean_weight=p1_weights.get("cs_mean", 0.0),
            sparse_weight=p1_weights.get("sparse", p1_weights.get("sparse_start", 0.0)),
            envelope_weight=p1_weights.get("envelope", p1_weights.get("envelope_start", 0.0)),
            # DiffToA config
            toa_step=loss_cfg.get("toa_step", 5),
            toa_push_exclusion=loss_cfg.get("toa_push_exclusion", 15),
            toa_loss_clamp=loss_cfg.get("toa_loss_clamp", 3.0),
            softargmax_tau=loss_cfg.get("softargmax_tau", 10.0),
            sparse_threshold=loss_cfg.get("sparse_threshold", 0.05),
        )
        self.criterion = self.criterion.to(self.device)

    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        base_lr = train_cfg["learning_rate"]
        physics_lr = train_cfg.get("physics_head_lr", base_lr * 3.33)

        # Two param groups: backbone + physics head
        backbone_params = []
        physics_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("physics_head."):
                physics_params.append(param)
            else:
                backbone_params.append(param)

        logger.info(
            f"Optimizer: backbone={len(backbone_params)} tensors (lr={base_lr:.1e}), "
            f"physics_head={len(physics_params)} tensors (lr={physics_lr:.1e})"
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": base_lr, "name": "backbone"},
                {"params": physics_params, "lr": physics_lr, "name": "physics_head"},
            ],
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        )

    # ------------------------------------------------------------------
    # Gradient clipping hooks (override in subclasses for per-group clip)
    # ------------------------------------------------------------------

    def _clip_and_step(self, max_grad_norm: float) -> None:
        """Non-AMP path: clip then step. Override for custom clipping."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

    def _clip_and_step_amp(self, max_grad_norm: float) -> None:
        """AMP path: unscale, clip, step, update. Override for custom clipping."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _log_setup(self):
        logger.info("=" * 60)
        logger.info("WavePINN v7 Training — Multi-Task SWE-Optimised")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size: {self.config['data']['batch_size']}")
        logger.info(f"  Train samples: {self.data_info.get('num_train', '?')}")
        logger.info(f"  Val samples: {self.data_info.get('num_val', '?')}")
        logger.info(f"  Upsampling: {self.config['model']['upsampling_factor']}")
        logger.info(f"  Cs range: [{self.config['model'].get('cs_min', 0.3)}, "
                     f"{self.config['model'].get('cs_max', 8.0)}] m/s")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Warm-start from V6
    # ------------------------------------------------------------------

    def warmstart_from_v6(self, v6_path: str):
        """Load V6 checkpoint into V7 backbone (strict=False). Physics head stays random."""
        ckpt = torch.load(v6_path, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"]

        missing, unexpected = self.model.load_state_dict(state, strict=False)

        n_loaded = len(state) - len(unexpected)
        logger.info(
            f"Warm-start from V6: loaded {n_loaded} keys, "
            f"missing (new in V7): {len(missing)}, unexpected: {len(unexpected)}"
        )
        if missing:
            logger.info(f"  Missing keys (expected — physics head): {missing}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")

        # Fresh optimizer — V6 param groups don't match V7
        logger.info("  Optimizer NOT loaded (fresh for V7 param groups)")

    # ------------------------------------------------------------------
    # Curriculum update
    # ------------------------------------------------------------------

    def _update_curriculum(self, epoch: int):
        """Update loss weights and LR for current epoch."""
        curriculum = self.config["curriculum"]
        phase_name, phase_cfg = get_phase(epoch, curriculum)
        loss_weights = get_loss_weights(epoch, phase_cfg)

        base_lr = get_lr_for_phase(epoch, phase_cfg, self.config["training"]["learning_rate"])

        # Physics head LR scales proportionally
        physics_lr_ratio = self.config["training"].get("physics_head_lr_ratio", 3.33)

        # Phase 3: both groups drop to lower LR
        phase_lr_multiplier = phase_cfg.get("lr_multiplier", 1.0)
        backbone_lr = base_lr * phase_lr_multiplier
        physics_head_lr = base_lr * physics_lr_ratio * phase_lr_multiplier

        # Update loss weights
        self.criterion.set_weights(**loss_weights)

        # Update learning rates per param group
        for pg in self.optimizer.param_groups:
            if pg.get("name") == "physics_head":
                pg["lr"] = physics_head_lr
            else:
                pg["lr"] = backbone_lr

        # Phase transition handling
        if phase_name != self.current_phase:
            if self.current_phase is not None:
                logger.info(
                    f"Phase transition '{self.current_phase}' -> '{phase_name}': "
                    f"resetting early stopping (was best={self.best_val_loss:.4f})"
                )
                self.best_val_loss = float("inf")
                self.patience_counter = 0
            self.current_phase = phase_name
            logger.info(
                f"=== Phase '{phase_name}' started (epoch {epoch}) === "
                f"backbone_lr={backbone_lr:.2e}, "
                f"toa={loss_weights.get('toa', 0):.4f}, "
                f"sparse={loss_weights.get('sparse', 0):.4f}"
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        train_cfg = self.config["training"]
        accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
        max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        log_interval = self.config.get("logging", {}).get("log_interval", 5)

        # Important: reset gradients at the start of the epoch.
        # Otherwise, leftover gradients from an incomplete accumulation window
        # can leak into the next epoch when len(dataloader) % accum_steps != 0.
        self.optimizer.zero_grad(set_to_none=True)

        num_batches = len(self.train_loader)
        remainder = num_batches % accum_steps
        flush_start = num_batches - remainder if remainder else num_batches

        running_components: Dict[str, float] = {}
        count = 0
        nan_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x = batch["input"].to(self.device)            # (B, 2, T_sparse, H, W)
            target = batch["target"].to(self.device)       # (B, 1, T_vel, H, W)
            mask = batch["mask"].to(self.device)           # (B, 1, 1, H, W)
            dt_seconds = batch["dt_seconds"].to(self.device)  # (B,)
            dt_output = batch["dt_output"].to(self.device)    # (B,)
            vel_scale = batch["vel_scale"].to(self.device)    # (B,)
            cs_target = batch["cs_target_scalar"].to(self.device)  # (B,)
            push_col = batch["push_col"].to(self.device)      # (B,)

            B = x.shape[0]

            # Use a smaller divisor for the tail remainder so the last optimizer
            # step has the same effective scaling as the full accumulation windows.
            accum_divisor = accum_steps
            if remainder and batch_idx >= flush_start:
                accum_divisor = remainder

            if self.scaler is not None:
                with autocast(self.device_type):
                    velocity, cs_map = self.model(x, dt_seconds=dt_seconds)
                    T_pred = velocity.shape[2]
                    target_crop = target[:, :, :T_pred]
                    loss, components = self.criterion(
                        velocity, target_crop, mask=mask,
                        cs_map_pred=cs_map,
                        cs_target_scalar=cs_target,
                        vel_scale=vel_scale,
                        dt_output=dt_output,
                        push_col=push_col,
                    )
                    loss = loss / accum_divisor

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    logger.warning(f"NaN/Inf loss at batch {batch_idx}. Skipping.")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()

                should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)
                if should_step:
                    self._clip_and_step_amp(max_grad_norm)
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                velocity, cs_map = self.model(x, dt_seconds=dt_seconds)
                T_pred = velocity.shape[2]
                target_crop = target[:, :, :T_pred]
                loss, components = self.criterion(
                    velocity, target_crop, mask=mask,
                    cs_map_pred=cs_map,
                    cs_target_scalar=cs_target,
                    vel_scale=vel_scale,
                    dt_output=dt_output,
                    push_col=push_col,
                )
                loss = loss / accum_divisor

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    logger.warning(f"NaN/Inf loss at batch {batch_idx}. Skipping.")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)
                if should_step:
                    self._clip_and_step(max_grad_norm)
                    self.optimizer.zero_grad(set_to_none=True)

            for k, v in components.items():
                running_components[k] = running_components.get(k, 0) + v * B
            count += B
            self.global_step += 1

            if batch_idx % log_interval == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {components.get('total', 0):.4f}, "
                    f"L1: {components.get('l1', 0):.4f}, "
                    f"ToA: {components.get('toa_cs', 0):.4f}, "
                    f"CsMap: {components.get('cs_map', 0):.4f}, "
                    f"Sparse: {components.get('spatial_sparsity', 0):.4f}"
                )

        if nan_count > 0:
            logger.warning(f"Epoch had {nan_count} NaN/Inf batches (skipped)")

        return {k: v / max(count, 1) for k, v in running_components.items()}

    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model. Returns (loss_metrics, extended_metrics)."""
        self.model.eval()
        running_components: Dict[str, float] = {}
        count = 0

        all_r_active = []
        all_r_full = []
        all_cs_mae_head = []
        all_cs_map_pred_mean = []

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)
            dt_seconds = batch["dt_seconds"].to(self.device)
            dt_output = batch["dt_output"].to(self.device)
            vel_scale = batch["vel_scale"].to(self.device)
            cs_target = batch["cs_target_scalar"].to(self.device)
            push_col = batch["push_col"].to(self.device)
            B = x.shape[0]

            if self.scaler is not None:
                with autocast(self.device_type):
                    velocity, cs_map = self.model(x, dt_seconds=dt_seconds)
                    T_pred = velocity.shape[2]
                    target_crop = target[:, :, :T_pred]
                    _, components = self.criterion(
                        velocity, target_crop, mask=mask,
                        cs_map_pred=cs_map,
                        cs_target_scalar=cs_target,
                        vel_scale=vel_scale,
                        dt_output=dt_output,
                        push_col=push_col,
                    )
            else:
                velocity, cs_map = self.model(x, dt_seconds=dt_seconds)
                T_pred = velocity.shape[2]
                target_crop = target[:, :, :T_pred]
                _, components = self.criterion(
                    velocity, target_crop, mask=mask,
                    cs_map_pred=cs_map,
                    cs_target_scalar=cs_target,
                    vel_scale=vel_scale,
                    dt_output=dt_output,
                    push_col=push_col,
                )

            for k, v in components.items():
                running_components[k] = running_components.get(k, 0) + v * B
            count += B

            # Extended metrics (on CPU, float32 for stability)
            ext = compute_validation_metrics(
                velocity.float().cpu(),
                target_crop.float().cpu(),
                mask.float().cpu(),
                cs_map.float().cpu(),
                cs_target.float().cpu(),
            )
            if "r_active" in ext:
                all_r_active.append(ext["r_active"])
            if "r_full" in ext:
                all_r_full.append(ext["r_full"])
            if "cs_mae_head" in ext:
                all_cs_mae_head.append(ext["cs_mae_head"])
            if "cs_map_pred_mean" in ext:
                all_cs_map_pred_mean.append(ext["cs_map_pred_mean"])

        loss_metrics = {k: v / max(count, 1) for k, v in running_components.items()}

        ext_metrics: Dict[str, float] = {}
        if all_r_active:
            ext_metrics["r_active"] = float(np.mean(all_r_active))
        if all_r_full:
            ext_metrics["r_full"] = float(np.mean(all_r_full))
        if all_cs_mae_head:
            ext_metrics["cs_mae_head"] = float(np.mean(all_cs_mae_head))
        if all_cs_map_pred_mean:
            ext_metrics["cs_map_pred_mean"] = float(np.mean(all_cs_map_pred_mean))

        return loss_metrics, ext_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str, is_best: bool = False):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.scaler:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        path = self.output_dir / filename
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            logger.info(f"Saved best model: {best_path}")

            saved_models_dir = Path("saved_models")
            saved_models_dir.mkdir(exist_ok=True)
            exp_name = self.config.get("logging", {}).get("experiment_name", "wavepinn_v7")
            torch.save(ckpt, saved_models_dir / f"{exp_name}_best.pt")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False,
        )
        if missing:
            logger.warning(f"Checkpoint missing keys: {missing}")
        if unexpected:
            logger.warning(f"Checkpoint unexpected keys: {unexpected}")

        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not restore optimizer ({e}). Fresh optimizer.")

        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop with 3-phase curriculum."""
        num_epochs = self.config["training"]["num_epochs"]
        save_interval = self.config.get("logging", {}).get("save_interval", 10)
        patience = self.config.get("early_stopping", {}).get("patience", 40)

        logger.info("Starting training...")

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            self._update_curriculum(epoch)

            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 40)

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Train - Loss: {train_metrics.get('total', 0):.4f}, "
                f"L1: {train_metrics.get('l1', 0):.4f}, "
                f"ToA: {train_metrics.get('toa_cs', 0):.4f}, "
                f"CsMap: {train_metrics.get('cs_map', 0):.4f}, "
                f"Sparse: {train_metrics.get('spatial_sparsity', 0):.4f}, "
                f"Env: {train_metrics.get('temporal_envelope', 0):.4f}"
            )

            if self.writer:
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, epoch)
                for pg in self.optimizer.param_groups:
                    self.writer.add_scalar(
                        f"train/lr_{pg.get('name', 'default')}", pg["lr"], epoch,
                    )

            # Validate
            val_metrics, ext_metrics = self.validate()
            val_total = val_metrics.get("total", float("inf"))
            # Use velocity-only L1 for early stopping.  Total loss includes
            # ramping physics terms (ToA, CsMap) that climb even when velocity
            # quality is stable or improving — using total for early stopping
            # would trigger premature termination.
            val_es_metric = val_metrics.get("l1", val_total)
            logger.info(
                f"Val   - Loss: {val_total:.4f}, "
                f"L1: {val_metrics.get('l1', 0):.4f}, "
                f"ToA: {val_metrics.get('toa_cs', 0):.4f}, "
                f"r_active: {ext_metrics.get('r_active', 0):.3f}, "
                f"r_full: {ext_metrics.get('r_full', 0):.3f}, "
                f"cs_mae_head: {ext_metrics.get('cs_mae_head', 0):.3f}, "
                f"cs_map_mean: {ext_metrics.get('cs_map_pred_mean', 0):.3f}"
            )

            if self.writer:
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                for k, v in ext_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

            # Best model — tracked on val L1, not total loss
            is_best = val_es_metric < self.best_val_loss
            if is_best:
                self.best_val_loss = val_es_metric
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt", is_best=True)
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        self.save_checkpoint("final_model.pt")
        logger.info(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

        if self.writer:
            self.writer.close()


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train WavePINN v7")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--resume", type=str, default=None, help="Resume from V7 checkpoint")
    parser.add_argument("--warmstart", type=str, default=None, help="Warm-start from V6 checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override base LR")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.no_amp:
        config["training"]["use_amp"] = False

    trainer = TrainerV7(config)
    trainer.setup()

    if args.warmstart:
        trainer.warmstart_from_v6(args.warmstart)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
