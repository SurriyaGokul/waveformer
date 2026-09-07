"""
Training script for WavePINN v9 — Final SWE-Accurate Temporal Super-Resolution.

Usage:
    python -m src.training.train_wavepinn_v9 --config configs/config_wavepinn_v9.json
    python -m src.training.train_wavepinn_v9 --config configs/config_wavepinn_v9.json --resume outputs/wavepinn_v9/checkpoint.pt

Three-Phase Curriculum (cold start — no warm-start path):
    Phase 1 (velocity_pretrain,  epochs 1-60):
        L1+MSE+GDL+temporal only.  ScalarCsHead gradients are DETACHED from
        backbone so Cs loss cannot corrupt encoder features during pretrain.

    Phase 2 (physics_activation, epochs 61-120):
        wave + cs weights cosine-ramp from 0 over epochs 61→90.  Head
        gradients flow to backbone from epoch 61 onwards.

    Phase 3 (coupled_finetune,   epochs 121-180):
        LR drops 10×.  Consistency weight (head↔wave-cs) ramps 121→140.

Key differences from V7/V8:
    - Single optimizer param group (no separate physics head LR)
    - Model forward returns (velocity, cs_scalar) — scalar head, not spatial map
    - No ToA loss (RC2), no amplitude weighting (RC3), no sparse/envelope terms
    - Early stopping monitored on val_pixel_loss_5khz (not total loss)
    - training_frequencies filter excludes 500 Hz native-rate training data (RC5)
    - Phase-1 detach_head blocks Cs loss from corrupting backbone (RC1)
    - Verification gates G1-G5 checked at key epoch boundaries
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

from src.architectures.wavepinn_v9 import WavePINNv9, create_wavepinn_v9, count_parameters_v9
from src.losses.swe_loss_v9 import EnhancedVelocityLossV9
from src.utils.iq_velocity_dataset_v3 import (
    IQVelocityDatasetV3,
    collate_fn_v3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =========================================================================
# Phantom-Stratified Batch Sampler (identical to V7 — reused verbatim)
# =========================================================================

class PhantomStratifiedBatchSampler(Sampler):
    """
    Batch sampler ensuring each batch has samples from >= 2 phantom types.

    Needed for L_cs diversity per step: if every batch is the same phantom,
    the Cs head cannot learn to discriminate between phantoms.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last

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

            batch = []
            draw_order = list(rng.permutation(active))
            gi = 0
            while len(batch) < self.batch_size:
                if gi >= len(draw_order):
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
                break

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


def _cosine_ramp(
    epoch: int, start_epoch: int, ramp_end: int, v_start: float, v_end: float
) -> float:
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
    cs_scalar_pred: torch.Tensor,
    cs_target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute SWE-relevant validation metrics.

    Returns dict with:
      r_active: Pearson r in first-upsampling-factor frames where |target|>0.1
      r_full  : Pearson r over the full masked volume
      cs_mae  : MAE of scalar cs head vs ground-truth (valid samples only)
      cs_mean : mean predicted cs across batch
    """
    metrics: Dict[str, float] = {}
    B = pred.shape[0]

    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() if mask is not None else None

    r_active_list = []
    r_full_list = []

    for b in range(B):
        p = pred_np[b, 0]  # (T, H, W)
        t = tgt_np[b, 0]
        m = mask_np[b, 0, 0].astype(bool) if mask_np is not None else np.ones(t.shape[1:], dtype=bool)

        p_flat = p[:, m].flatten()
        t_flat = t[:, m].flatten()
        if len(p_flat) > 10 and np.std(p_flat) > 1e-8 and np.std(t_flat) > 1e-8:
            r_full_list.append(float(np.corrcoef(p_flat, t_flat)[0, 1]))

        n_active = min(p.shape[0], 12)
        active_mask_3d = (np.abs(t[:n_active]) > 0.1) & m[None, :, :]
        p_active = p[:n_active][active_mask_3d]
        t_active = t[:n_active][active_mask_3d]
        if len(p_active) > 10 and np.std(p_active) > 1e-8 and np.std(t_active) > 1e-8:
            r_active_list.append(float(np.corrcoef(p_active, t_active)[0, 1]))

    if r_active_list:
        metrics["r_active"] = float(np.mean(r_active_list))
    if r_full_list:
        metrics["r_full"] = float(np.mean(r_full_list))

    # Scalar Cs diagnostics
    cs_pred_np = cs_scalar_pred.detach().cpu().numpy()
    cs_tgt_np = cs_target.detach().cpu().numpy()
    metrics["cs_pred_mean"] = float(np.mean(cs_pred_np))
    metrics["cs_pred_std"] = float(np.std(cs_pred_np))

    valid = cs_tgt_np > 0
    if valid.any():
        metrics["cs_mae"] = float(np.mean(np.abs(cs_pred_np[valid] - cs_tgt_np[valid])))

    return metrics


# =========================================================================
# Trainer
# =========================================================================

class TrainerV9:
    """Training orchestrator for WavePINN v9."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        exp_name = config.get("logging", {}).get("experiment_name", "wavepinn_v9")
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
        self.detach_head = True  # True in Phase 1, False in Phase 2+

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Initialise model, data, loss, optimiser."""
        self._setup_data()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
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
        upsampling = self.config["model"].get("upsampling_factor", 12)
        batch_size = data_cfg.get("batch_size", 2)
        num_workers = data_cfg.get("num_workers", 4)
        training_frequencies = data_cfg.get("training_frequencies", None)

        # --- Training dataset (frequency-filtered) ---
        train_dataset = IQVelocityDatasetV3(
            data_dir=data_cfg["data_dir"],
            split="train",
            enable_prf_augmentation=data_cfg.get("enable_prf_augmentation", True),
            augmentation_steps=data_cfg.get("augmentation_steps"),
            augmentation_step_weights=data_cfg.get("augmentation_step_weights"),
            upsampling_factor=upsampling,
            per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
            per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            training_frequencies=training_frequencies,
        )

        # --- Full val dataset (all frequencies — for deployment probe logging) ---
        val_dataset_all = IQVelocityDatasetV3(
            data_dir=data_cfg["data_dir"],
            split="val",
            enable_prf_augmentation=False,
            upsampling_factor=upsampling,
            iq_norm_scalar=train_dataset.iq_norm_scalar,
            vel_norm_scalar=train_dataset.vel_norm_scalar,
            per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
            per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
        )

        # --- 5 kHz val subset (primary early stopping metric) ---
        val_dataset_5khz = IQVelocityDatasetV3(
            data_dir=data_cfg["data_dir"],
            split="val",
            enable_prf_augmentation=False,
            upsampling_factor=upsampling,
            iq_norm_scalar=train_dataset.iq_norm_scalar,
            vel_norm_scalar=train_dataset.vel_norm_scalar,
            per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
            per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            training_frequencies=["5kHz"],
        )

        use_stratified = self.config.get("training", {}).get("use_stratified_sampler", True)

        if use_stratified:
            sampler = PhantomStratifiedBatchSampler(
                train_dataset, batch_size=batch_size, drop_last=True,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn_v3,
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn_v3,
            )

        self.val_loader_5khz = DataLoader(
            val_dataset_5khz,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_v3,
        )

        self.val_loader_all = DataLoader(
            val_dataset_all,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_v3,
        )

        self.data_info = train_dataset.get_data_info()
        self.data_info["num_val_5khz"] = len(val_dataset_5khz)
        self.data_info["num_val_all"] = len(val_dataset_all)
        self.vel_norm_scalar = train_dataset.vel_norm_scalar

    def _setup_model(self):
        model_cfg = dict(self.config["model"])
        model_cfg.pop("name", None)
        self.model = create_wavepinn_v9(config=model_cfg)
        self.model = self.model.to(self.device)

        params = count_parameters_v9(self.model)
        logger.info(
            f"Model parameters: {params['total']:,} total, "
            f"{params['backbone']:,} backbone, {params['cs_head']:,} cs_head"
        )

    def _setup_criterion(self):
        p1_weights = self.config["curriculum"]["phase_1"]["loss_weights"]
        loss_cfg = self.config.get("loss", {})
        self.criterion = EnhancedVelocityLossV9(
            l1_weight=p1_weights.get("l1", 1.0),
            mse_weight=p1_weights.get("mse", 1.0),
            gdl_weight=p1_weights.get("gdl", 0.3),
            temporal_weight=p1_weights.get("temporal", 0.15),
            wave_weight=p1_weights.get("wave", 0.0),
            cs_weight=p1_weights.get("cs", 0.0),
            cons_weight=p1_weights.get("cons", 0.0),
            use_mask=loss_cfg.get("use_mask", True),
        )
        self.criterion = self.criterion.to(self.device)

    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        base_lr = train_cfg["learning_rate"]
        weight_decay = train_cfg.get("weight_decay", 1e-5)
        betas = tuple(train_cfg.get("betas", [0.9, 0.999]))

        # Single param group — simpler, avoids the 2-group LR inconsistency
        # that complicated V7/V8 curriculum management.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
            betas=betas,
        )
        logger.info(
            f"AdamW: lr={base_lr:.1e}, wd={weight_decay:.1e}, single param group"
        )

    def _log_setup(self):
        logger.info("=" * 60)
        logger.info("WavePINN v9 Training — SWE-Accurate Temporal SR")
        logger.info(f"  Device       : {self.device}")
        logger.info(f"  Epochs       : {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size   : {self.config['data']['batch_size']}")
        logger.info(f"  Train samples: {self.data_info.get('num_train', '?')} "
                    f"(freq filter: {self.config['data'].get('training_frequencies')})")
        logger.info(f"  Val 5kHz     : {self.data_info.get('num_val_5khz', '?')}")
        logger.info(f"  Val all      : {self.data_info.get('num_val_all', '?')}")
        logger.info(f"  Upsampling   : {self.config['model']['upsampling_factor']}")
        logger.info(f"  Cs range     : [{self.config['model'].get('cs_min', 0.3)}, "
                     f"{self.config['model'].get('cs_max', 8.0)}] m/s")
        logger.info(f"  Cs init      : {self.config['model'].get('cs_init', 2.54)} m/s")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Verification Gates
    # ------------------------------------------------------------------

    def _run_gate_g1(self):
        """G1: Init sanity — cs_scalar_pred mean ≈ 2.54 m/s before training."""
        gates = self.config.get("verification_gates", {})
        target_mean = gates.get("G1_init_cs_mean_target", 2.54)
        tolerance = gates.get("G1_init_cs_tolerance", 0.1)

        self.model.eval()
        cs_preds = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader_5khz):
                if i >= 3:
                    break
                x = batch["input"].to(self.device)
                dt_seconds = batch["dt_seconds"].to(self.device)
                _, cs_scalar = self.model(x, dt_seconds=dt_seconds, detach_head=True)
                cs_preds.append(cs_scalar.cpu())

        if not cs_preds:
            logger.warning("G1: No val_5khz batches available for init check")
            return

        cs_all = torch.cat(cs_preds).numpy()
        cs_mean = float(np.mean(cs_all))
        cs_std = float(np.std(cs_all))
        diff = abs(cs_mean - target_mean)

        status = "PASS" if diff <= tolerance else "WARN"
        logger.info(
            f"[G1 {status}] Init cs_scalar: mean={cs_mean:.3f} m/s, "
            f"std={cs_std:.3f}, target={target_mean:.2f}±{tolerance:.2f} m/s"
        )
        if status == "WARN":
            logger.warning(
                f"G1 WARN: cs init mean {cs_mean:.3f} is {diff:.3f} m/s away from "
                f"target {target_mean:.2f}. Check ScalarCsHead init_bias."
            )
        self.model.train()

    def _check_gate(self, gate_key: str, value: float, label: str, lower_is_better: bool = True):
        """Generic verification gate check with logging."""
        gates = self.config.get("verification_gates", {})
        threshold = gates.get(gate_key)
        if threshold is None:
            return
        passed = (value <= threshold) if lower_is_better else (value >= threshold)
        status = "PASS" if passed else "WARN"
        cmp = "≤" if lower_is_better else "≥"
        logger.info(
            f"[{gate_key} {status}] {label}={value:.4f} (target: {cmp}{threshold:.4f})"
        )
        if not passed:
            logger.warning(
                f"Verification gate {gate_key} NOT met: {label}={value:.4f} "
                f"(required {cmp}{threshold}). "
                f"Consider extending the current phase."
            )

    # ------------------------------------------------------------------
    # Curriculum update
    # ------------------------------------------------------------------

    def _update_curriculum(self, epoch: int):
        """Update loss weights, LR, and detach_head flag for current epoch."""
        curriculum = self.config["curriculum"]
        phase_name, phase_cfg = get_phase(epoch, curriculum)
        loss_weights = get_loss_weights(epoch, phase_cfg)

        # Phase-level LR (single group) with cosine decay within phase
        base_lr = self.config["training"]["learning_rate"]
        phase_lr_multiplier = phase_cfg.get("lr_multiplier", 1.0)
        lr = get_lr_for_phase(epoch, phase_cfg, base_lr) * phase_lr_multiplier
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # detach_head: True in Phase 1, False in Phase 2+
        self.detach_head = phase_cfg.get("detach_head", False)

        # Update loss
        self.criterion.set_weights(**loss_weights)

        # Phase transition
        if phase_name != self.current_phase:
            if self.current_phase is not None:
                es_cfg = self.config.get("early_stopping", {})
                reset_best = es_cfg.get("reset_best_on_phase_transition", False)
                logger.info(
                    f"Phase transition '{self.current_phase}' -> '{phase_name}' "
                    f"(epoch {epoch}): resetting patience counter"
                    + (", resetting best_val_loss." if reset_best else ".")
                )
                self.patience_counter = 0
                if reset_best:
                    # Re-anchor ES baseline at phase boundary so Phase 2's
                    # inherently higher val_L1 is not penalised against Phase 1 best.
                    self.best_val_loss = float("inf")

                # Gate checks at phase boundaries
                if phase_name == "physics_activation":
                    pass  # G2 checked after this epoch's validation
                if phase_name == "coupled_finetune":
                    pass  # G4 checked after this epoch's validation

            self.current_phase = phase_name
            logger.info(
                f"=== Phase '{phase_name}' (epoch {epoch}) === "
                f"lr={lr:.2e}, detach_head={self.detach_head}, "
                f"wave={loss_weights.get('wave', 0):.4f}, "
                f"cs={loss_weights.get('cs', 0):.4f}, "
                f"cons={loss_weights.get('cons', 0):.4f}"
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

        self.optimizer.zero_grad(set_to_none=True)

        num_batches = len(self.train_loader)
        remainder = num_batches % accum_steps
        flush_start = num_batches - remainder if remainder else num_batches

        running_components: Dict[str, float] = {}
        count = 0
        nan_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x = batch["input"].to(self.device)             # (B, 2, T_sparse, H, W)
            target = batch["target"].to(self.device)        # (B, 1, T_vel, H, W)
            mask = batch["mask"].to(self.device)            # (B, 1, 1, H, W)
            dt_seconds = batch["dt_seconds"].to(self.device)  # (B,)
            cs_target = batch["cs_target_scalar"].to(self.device)  # (B,)
            B = x.shape[0]

            accum_divisor = accum_steps
            if remainder and batch_idx >= flush_start:
                accum_divisor = remainder

            if self.scaler is not None:
                with autocast(self.device_type):
                    velocity, cs_scalar = self.model(
                        x, dt_seconds=dt_seconds, detach_head=self.detach_head
                    )
                    T_pred = velocity.shape[2]
                    target_crop = target[:, :, :T_pred]
                    components = self.criterion(
                        velocity, target_crop,
                        cs_scalar_pred=cs_scalar,
                        cs_target=cs_target,
                        mask=mask,
                    )
                    loss = components["total"] / accum_divisor

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    logger.warning(f"NaN/Inf loss at batch {batch_idx}. Skipping.")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()

                should_step = (
                    ((batch_idx + 1) % accum_steps == 0)
                    or ((batch_idx + 1) == num_batches)
                )
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                velocity, cs_scalar = self.model(
                    x, dt_seconds=dt_seconds, detach_head=self.detach_head
                )
                T_pred = velocity.shape[2]
                target_crop = target[:, :, :T_pred]
                components = self.criterion(
                    velocity, target_crop,
                    cs_scalar_pred=cs_scalar,
                    cs_target=cs_target,
                    mask=mask,
                )
                loss = components["total"] / accum_divisor

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    logger.warning(f"NaN/Inf loss at batch {batch_idx}. Skipping.")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                should_step = (
                    ((batch_idx + 1) % accum_steps == 0)
                    or ((batch_idx + 1) == num_batches)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            for k, v in components.items():
                val_item = v.item() if isinstance(v, torch.Tensor) else float(v)
                running_components[k] = running_components.get(k, 0.0) + val_item * B
            count += B
            self.global_step += 1

            if batch_idx % log_interval == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(self.train_loader)} — "
                    f"Loss: {components['total'].item():.4f}, "
                    f"L1: {components['l1'].item():.4f}, "
                    f"wave: {components['wave'].item():.4f}, "
                    f"cs: {components['cs'].item():.4f}, "
                    f"cs_mean: {cs_scalar.mean().item():.3f} m/s"
                )

        if nan_count > 0:
            logger.warning(f"Epoch had {nan_count} NaN/Inf batches (skipped)")

        return {k: v / max(count, 1) for k, v in running_components.items()}

    @torch.no_grad()
    def _validate_loader(
        self, loader: DataLoader, tag: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate on a given loader. Returns (loss_metrics, ext_metrics)."""
        self.model.eval()
        running_components: Dict[str, float] = {}
        count = 0
        all_r_active = []
        all_cs_mae = []
        all_cs_mean = []

        for batch in loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)
            dt_seconds = batch["dt_seconds"].to(self.device)
            cs_target = batch["cs_target_scalar"].to(self.device)
            B = x.shape[0]

            if self.scaler is not None:
                with autocast(self.device_type):
                    velocity, cs_scalar = self.model(
                        x, dt_seconds=dt_seconds, detach_head=False
                    )
                    T_pred = velocity.shape[2]
                    target_crop = target[:, :, :T_pred]
                    components = self.criterion(
                        velocity, target_crop,
                        cs_scalar_pred=cs_scalar,
                        cs_target=cs_target,
                        mask=mask,
                    )
            else:
                velocity, cs_scalar = self.model(
                    x, dt_seconds=dt_seconds, detach_head=False
                )
                T_pred = velocity.shape[2]
                target_crop = target[:, :, :T_pred]
                components = self.criterion(
                    velocity, target_crop,
                    cs_scalar_pred=cs_scalar,
                    cs_target=cs_target,
                    mask=mask,
                )

            for k, v in components.items():
                val_item = v.item() if isinstance(v, torch.Tensor) else float(v)
                running_components[k] = running_components.get(k, 0.0) + val_item * B
            count += B

            ext = compute_validation_metrics(
                velocity.float().cpu(),
                target_crop.float().cpu(),
                mask.float().cpu(),
                cs_scalar.float().cpu(),
                cs_target.float().cpu(),
            )
            if "r_active" in ext:
                all_r_active.append(ext["r_active"])
            if "cs_mae" in ext:
                all_cs_mae.append(ext["cs_mae"])
            if "cs_pred_mean" in ext:
                all_cs_mean.append(ext["cs_pred_mean"])

        self.model.train()

        loss_metrics = {k: v / max(count, 1) for k, v in running_components.items()}

        ext_metrics: Dict[str, float] = {}
        if all_r_active:
            ext_metrics["r_active"] = float(np.mean(all_r_active))
        if all_cs_mae:
            ext_metrics["cs_mae"] = float(np.mean(all_cs_mae))
        if all_cs_mean:
            ext_metrics["cs_pred_mean"] = float(np.mean(all_cs_mean))

        return loss_metrics, ext_metrics

    def validate(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Validate on 5kHz subset (primary) and full val set.

        Returns (loss_5khz, ext_5khz, loss_all)
        """
        loss_5khz, ext_5khz = self._validate_loader(self.val_loader_5khz, "5kHz")
        loss_all, _ = self._validate_loader(self.val_loader_all, "all")
        return loss_5khz, ext_5khz, loss_all

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
            exp_name = self.config.get("logging", {}).get("experiment_name", "wavepinn_v9")
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
        """Main training loop with 3-phase curriculum and verification gates."""
        num_epochs = self.config["training"]["num_epochs"]
        save_interval = self.config.get("logging", {}).get("save_interval", 10)
        es_cfg = self.config.get("early_stopping", {})
        patience = es_cfg.get("patience", 30)
        min_delta = es_cfg.get("min_delta", 1e-5)

        # Phase boundary epochs for gate checks
        phase_boundaries = {}
        for key in sorted(self.config["curriculum"].keys()):
            if not key.startswith("phase_"):
                continue
            p = self.config["curriculum"][key]
            phase_boundaries[p["name"]] = p["epochs"]

        logger.info("Starting training...")

        # G1: Init sanity check before any training
        self._run_gate_g1()

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            self._update_curriculum(epoch)

            logger.info(f"\nEpoch {epoch}/{num_epochs} | Phase: {self.current_phase}")
            logger.info("-" * 50)

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Train — loss: {train_metrics.get('total', 0):.4f}, "
                f"L1: {train_metrics.get('l1', 0):.4f}, "
                f"wave: {train_metrics.get('wave', 0):.4f}, "
                f"cs: {train_metrics.get('cs', 0):.4f}"
            )

            # Validate
            loss_5khz, ext_5khz, loss_all = self.validate()

            # Primary early stopping metric: 5kHz pixel L1
            val_es_metric = loss_5khz.get("l1", float("inf"))
            val_cs_mae = ext_5khz.get("cs_mae", float("nan"))
            val_r_active = ext_5khz.get("r_active", float("nan"))
            cs_pred_mean = ext_5khz.get("cs_pred_mean", float("nan"))

            logger.info(
                f"Val 5kHz — L1: {val_es_metric:.4f}, "
                f"wave: {loss_5khz.get('wave', 0):.4f}, "
                f"r_active: {val_r_active:.3f}, "
                f"cs_mae: {val_cs_mae:.3f} m/s, "
                f"cs_mean: {cs_pred_mean:.3f} m/s"
            )
            logger.info(
                f"Val all  — L1: {loss_all.get('l1', 0):.4f}"
            )

            # TensorBoard
            if self.writer:
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in loss_5khz.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val_5khz/{k}", v, epoch)
                for k, v in ext_5khz.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val_5khz/{k}", v, epoch)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
                self.writer.add_scalar("train/detach_head", int(self.detach_head), epoch)

            # Verification gates at phase end
            if epoch == phase_boundaries.get("velocity_pretrain", [0, 0])[1]:
                logger.info(f"[GATE G2] Phase 1 ended at epoch {epoch}")
                self._check_gate(
                    "G2_phase1_end_val_pixel_loss_max",
                    val_es_metric,
                    "val_pixel_loss_5khz",
                )

            if epoch == phase_boundaries.get("physics_activation", [0, 0])[1]:
                logger.info(f"[GATE G4] Phase 2 ended at epoch {epoch}")
                self._check_gate(
                    "G4_phase2_end_cs_mae_max",
                    val_cs_mae if not np.isnan(val_cs_mae) else 999.0,
                    "val_cs_mae_5khz",
                )

            if epoch == num_epochs:
                logger.info(f"[GATE G5] Training complete at epoch {epoch}")
                self._check_gate(
                    "G5_phase3_end_cs_mae_max",
                    val_cs_mae if not np.isnan(val_cs_mae) else 999.0,
                    "val_cs_mae_5khz",
                )
                self._check_gate(
                    "G5_phase3_end_val_pixel_loss_max",
                    val_es_metric,
                    "val_pixel_loss_5khz",
                )

            # Phase 2 stability gate: pixel loss regression check
            if self.current_phase == "physics_activation" and hasattr(self, "_phase2_start_l1"):
                regression_frac = (val_es_metric - self._phase2_start_l1) / max(self._phase2_start_l1, 1e-8)
                max_regression = self.config.get("verification_gates", {}).get(
                    "G3_phase2_pixel_regression_max_fraction", 0.20
                )
                if regression_frac > max_regression:
                    logger.warning(
                        f"[G3 WARN] Phase 2 pixel loss regression {regression_frac:.1%} "
                        f"> {max_regression:.0%}. "
                        f"Consider halving wave/cs ramp targets."
                    )
            if self.current_phase == "physics_activation" and not hasattr(self, "_phase2_start_l1"):
                self._phase2_start_l1 = val_es_metric

            # Early stopping (on 5kHz pixel L1 only)
            if val_es_metric < self.best_val_loss - min_delta:
                self.best_val_loss = val_es_metric
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt", is_best=True)
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={patience}, best val L1={self.best_val_loss:.4f})"
                )
                break

            if epoch % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        self.save_checkpoint("final_model.pt")
        logger.info(f"\nTraining complete. Best val_pixel_loss_5khz: {self.best_val_loss:.4f}")

        if self.writer:
            self.writer.close()


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train WavePINN v9")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
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

    trainer = TrainerV9(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume} (epoch {trainer.current_epoch})")

    trainer.train()


if __name__ == "__main__":
    main()
