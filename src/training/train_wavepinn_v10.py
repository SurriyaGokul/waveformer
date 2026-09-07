"""
Training script for WavePINN V10 — Single-Phase Pure Pixel Loss.

Usage:
    python -m src.training.train_wavepinn_v10 --config configs/config_wavepinn_v10.json
    python -m src.training.train_wavepinn_v10 --config configs/config_wavepinn_v10.json --resume outputs/wavepinn_v10/checkpoint_epoch_60.pt

Design:
    V10 = WavePINNv6 architecture, trained with PURE PIXEL LOSS for 180 epochs.
    No physics loss. No Cs head. No curriculum phases. No wave coherence.

Why pure pixel loss:
    - V9's wave coherence gradient is ~256x larger than L1 on inactive frames
      (70% of frames near zero). This overwhelmed the backbone at epoch 65-72,
      collapsing r_active from 0.436 to 0.078 and destroying deployment performance.
    - V9-Ep60 (Phase 1 only, 60 epochs pixel loss) achieves deployment cs_MAE=1.98
      vs V2's 2.63. V10 trains for 3x longer with deployment-biased augmentation.

Key differences from V9:
    - Architecture: WavePINNv6 directly (no ScalarCsHead, forward returns velocity only)
    - Loss: VelocityLossV10 (L1+MSE+GDL+Temporal, temporal_weight=0.3)
    - No curriculum — single phase, cosine LR over 180 epochs
    - Collapse alarm: logs WARNING if r_active drops >0.1 from running max
    - Verification gates at epochs 1, 60, 120, 180
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

from src.architectures.wavepinn_v10 import create_wavepinn_v10, count_parameters_v10
from src.losses.velocity_loss_v10 import VelocityLossV10
from src.losses.velocity_loss_v14 import VelocityLossV14
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
# Phantom-Stratified Batch Sampler
# =========================================================================

class PhantomStratifiedBatchSampler(Sampler):
    """Batch sampler ensuring each batch has samples from >= 2 phantom types."""

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
# Interleaved Multi-Dataset Batch Loader
# =========================================================================

class InterleavedBatchLoader:
    """
    Combines several DataLoaders (each internally shape-homogeneous, e.g. one
    per source dataset with its own ROI crop size) into a single iterable of
    batches, shuffled at the batch level so batches from different source
    datasets are interspersed through the epoch rather than run back-to-back.

    Each yielded batch always comes from exactly one underlying loader, so
    collate_fn_v3's torch.stack never has to reconcile different spatial
    shapes across datasets — only individual DataLoaders need internal shape
    consistency, not the combined loader.
    """

    def __init__(self, loaders: List[DataLoader]):
        self.loaders = loaders

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        rng = np.random.default_rng()
        iterators = [iter(loader) for loader in self.loaders]
        remaining = list(range(len(iterators)))
        order = []
        # Pre-compute a shuffled draw order proportional to each loader's length.
        for i, loader in enumerate(self.loaders):
            order.extend([i] * len(loader))
        rng.shuffle(order)
        for i in order:
            try:
                yield next(iterators[i])
            except StopIteration:
                continue


# =========================================================================
# Learning Rate Schedule (cosine over full run)
# =========================================================================

def get_lr(epoch: int, num_epochs: int, base_lr: float, warmup_epochs: int = 5) -> float:
    """Linear warmup then cosine decay to 1e-6."""
    min_lr = 1e-6
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


# =========================================================================
# Validation Metrics
# =========================================================================

def compute_validation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Returns:
      r_active   : Pearson r in first 12 frames where |target| > 0.1
      r_full     : Pearson r over full masked volume
      output_std : std of predictions (collapse detector)
    """
    metrics: Dict[str, float] = {}

    pred_np = pred.detach().cpu().float().numpy()
    tgt_np = target.detach().cpu().float().numpy()
    mask_np = mask.detach().cpu().numpy() if mask is not None else None

    r_active_list = []
    r_full_list = []
    std_list = []

    B = pred_np.shape[0]
    for b in range(B):
        p = pred_np[b, 0]   # (T, H, W)
        t = tgt_np[b, 0]
        m = mask_np[b, 0, 0].astype(bool) if mask_np is not None else np.ones(t.shape[1:], dtype=bool)

        p_flat = p[:, m].flatten()
        t_flat = t[:, m].flatten()
        if len(p_flat) > 10 and np.std(p_flat) > 1e-8 and np.std(t_flat) > 1e-8:
            r_full_list.append(float(np.corrcoef(p_flat, t_flat)[0, 1]))
        std_list.append(float(np.std(p_flat)))

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
    if std_list:
        metrics["output_std"] = float(np.mean(std_list))

    return metrics


# =========================================================================
# Trainer
# =========================================================================

class TrainerV10:
    """Single-phase trainer for WavePINN V10."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        exp_name = config.get("logging", {}).get("experiment_name", "wavepinn_v10")
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

        # Collapse alarm state
        self._r_active_max = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        self._setup_data()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        self.device_type = self.device.type
        use_amp = self.config["training"].get("use_amp", True) and self.device_type == "cuda"
        self.scaler = GradScaler() if use_amp else None
        if self.config["training"].get("use_amp", True) and not use_amp:
            logger.info("AMP requested but device is not CUDA — disabling.")
        self._log_setup()

    def _setup_data(self):
        data_cfg = self.config["data"]
        upsampling = self.config["model"].get("upsampling_factor", 12)
        batch_size = data_cfg.get("batch_size", 2)
        num_workers = data_cfg.get("num_workers", 4)
        training_frequencies = data_cfg.get("training_frequencies", None)

        freq_step_overrides = data_cfg.get("frequency_step_overrides", None)
        # Drop (rather than silently mis-train on) samples whose configured PRF
        # step needs raw IQ that is not on disk.  See IQVelocityDatasetV3.
        strict_iq = bool(data_cfg.get("strict_iq", False))
        # Loss weight for target frames the dataset interpolated (0 = ignore them).
        interp_frame_weight = float(data_cfg.get("interp_frame_weight", 0.0))

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
            frequency_step_overrides=freq_step_overrides,
            strict_iq=strict_iq,
            interp_frame_weight=interp_frame_weight,
        )

        val_dataset_all = IQVelocityDatasetV3(
            data_dir=data_cfg["data_dir"],
            split="val",
            enable_prf_augmentation=False,
            upsampling_factor=upsampling,
            iq_norm_scalar=train_dataset.iq_norm_scalar,
            vel_norm_scalar=train_dataset.vel_norm_scalar,
            per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
            per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            frequency_step_overrides=freq_step_overrides,
            strict_iq=strict_iq,
            interp_frame_weight=interp_frame_weight,
        )

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
            frequency_step_overrides=freq_step_overrides,
            strict_iq=strict_iq,
            interp_frame_weight=interp_frame_weight,
        )

        val_dataset_500hz = IQVelocityDatasetV3(
            data_dir=data_cfg["data_dir"],
            split="val",
            enable_prf_augmentation=False,
            upsampling_factor=upsampling,
            iq_norm_scalar=train_dataset.iq_norm_scalar,
            vel_norm_scalar=train_dataset.vel_norm_scalar,
            per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
            per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            training_frequencies=["500Hz"],
            frequency_step_overrides=freq_step_overrides,
            strict_iq=strict_iq,
            interp_frame_weight=interp_frame_weight,
        )

        use_stratified = self.config.get("training", {}).get("use_stratified_sampler", True)

        def _make_train_loader(dataset) -> DataLoader:
            if use_stratified:
                sampler = PhantomStratifiedBatchSampler(
                    dataset, batch_size=batch_size, drop_last=True
                )
                return DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn_v3,
                )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn_v3,
            )

        train_loaders = [_make_train_loader(train_dataset)]

        # Extra datasets (e.g. a second phantom family with a different ROI
        # crop size) are trained on their own "train" split, kept in their
        # own homogeneous-shape DataLoader, and interleaved with the primary
        # loader at the batch level — collate_fn_v3's torch.stack never has
        # to mix shapes across datasets this way. Their own "val" split is
        # deliberately NOT loaded here, so it stays held-out for evaluation.
        for extra_dir in data_cfg.get("extra_data_dirs", []):
            extra_dataset = IQVelocityDatasetV3(
                data_dir=extra_dir,
                split="train",
                enable_prf_augmentation=data_cfg.get("enable_prf_augmentation", True),
                augmentation_steps=data_cfg.get("augmentation_steps"),
                augmentation_step_weights=data_cfg.get("augmentation_step_weights"),
                upsampling_factor=upsampling,
                per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
                per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
            )
            logger.info(f"Extra train dataset '{extra_dir}': {len(extra_dataset)} samples")
            train_loaders.append(_make_train_loader(extra_dataset))

        self.train_loader = (
            train_loaders[0] if len(train_loaders) == 1 else InterleavedBatchLoader(train_loaders)
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

        self.val_loader_500hz = DataLoader(
            val_dataset_500hz,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_v3,
        ) if len(val_dataset_500hz) > 0 else None

        self.val_loader_all = DataLoader(
            val_dataset_all,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_v3,
        )

        # Determine which val set to use for early stopping.
        # If 500Hz is in the training frequencies, monitor 500Hz val L1 directly.
        training_freqs_lower = [f.lower() for f in (training_frequencies or [])]
        # The 500 Hz val target is a linear interpolation of the same 5 frames
        # the model receives, so selecting on val_500hz_l1 rewards piecewise-
        # linear output -- the exact failure V14 targets.  Setting
        # early_stopping.monitor = "val_5khz_l1" selects on the 5 kHz val set
        # instead, whose dense ground truth is genuine (native step 12,
        # dt_out == dt_native, frame for frame).
        monitor_cfg = str(
            self.config.get("early_stopping", {}).get("monitor", "")
        ).lower()
        self._monitor_500hz = (
            "500hz" in training_freqs_lower
            and self.val_loader_500hz is not None
            and monitor_cfg != "val_5khz_l1"
        )
        if monitor_cfg == "val_5khz_l1":
            logger.info(
                "Early stopping monitors val_5khz_l1 (genuine dense ground truth)"
            )

        self.data_info = train_dataset.get_data_info()
        if len(train_loaders) > 1:
            self.data_info["num_train"] = len(self.train_loader.loaders[0].dataset) + sum(
                len(loader.dataset) for loader in self.train_loader.loaders[1:]
            )
        self.data_info["num_val_5khz"] = len(val_dataset_5khz)
        self.data_info["num_val_500hz"] = len(val_dataset_500hz)
        self.data_info["num_val_all"] = len(val_dataset_all)
        self.data_info["monitor_500hz"] = self._monitor_500hz
        self.vel_norm_scalar = train_dataset.vel_norm_scalar

    def _setup_model(self):
        model_cfg = dict(self.config["model"])
        model_cfg.pop("name", None)
        self.model = create_wavepinn_v10(config=model_cfg)
        self.model = self.model.to(self.device)

        params = count_parameters_v10(self.model)
        logger.info(
            f"Model parameters: {params['total']:,} total, "
            f"{params['trainable']:,} trainable"
        )

    def _setup_criterion(self):
        loss_cfg = self.config.get("loss", {})
        # V14 adds per-frame supervision masking (so linearly interpolated
        # target frames stop teaching the model to be piecewise linear) and an
        # optional temporal spectral term.  Opt-in, so v10b/v12/v13 configs keep
        # producing exactly the same loss.
        self.use_frame_mask = bool(loss_cfg.get("use_frame_mask", False))
        spectral_weight = float(loss_cfg.get("spectral_weight", 0.0))

        if self.use_frame_mask or spectral_weight > 0:
            self.criterion = VelocityLossV14(
                l1_weight=loss_cfg.get("l1_weight", 1.0),
                mse_weight=loss_cfg.get("mse_weight", 1.0),
                gdl_weight=loss_cfg.get("gdl_weight", 0.3),
                temporal_weight=loss_cfg.get("temporal_weight", 0.3),
                spectral_weight=spectral_weight,
                spectral_f_lo=loss_cfg.get("spectral_f_lo", 250.0),
                spectral_f_hi=loss_cfg.get("spectral_f_hi", 1500.0),
                spectral_hf_weight=loss_cfg.get("spectral_hf_weight", 2.0),
                use_mask=loss_cfg.get("use_mask", True),
            )
            logger.info(
                f"Loss: VelocityLossV14 (frame_mask={self.use_frame_mask}, "
                f"spectral_weight={spectral_weight})"
            )
        else:
            self.criterion = VelocityLossV10(
                l1_weight=loss_cfg.get("l1_weight", 1.0),
                mse_weight=loss_cfg.get("mse_weight", 1.0),
                gdl_weight=loss_cfg.get("gdl_weight", 0.3),
                temporal_weight=loss_cfg.get("temporal_weight", 0.3),
                use_mask=loss_cfg.get("use_mask", True),
            )
        self.criterion = self.criterion.to(self.device)

    def _loss_kwargs(self, batch, T_pred: int) -> Dict[str, Any]:
        """Extra criterion arguments used only by the V14 loss."""
        if not isinstance(self.criterion, VelocityLossV14):
            return {}
        kw: Dict[str, Any] = {}
        fm = batch.get("target_frame_mask")
        if self.use_frame_mask and fm is not None:
            kw["frame_mask"] = fm[:, :T_pred].to(self.device)
        dt_out = batch.get("dt_output")
        if dt_out is not None:
            kw["dt_seconds"] = dt_out.to(self.device)
        return kw

    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        base_lr = train_cfg["learning_rate"]
        weight_decay = train_cfg.get("weight_decay", 1e-5)
        betas = tuple(train_cfg.get("betas", [0.9, 0.999]))

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
        train_cfg = self.config["training"]
        logger.info("=" * 60)
        logger.info("WavePINN V10 Training — Pure Pixel Loss, Deployment-Optimised")
        logger.info(f"  Device       : {self.device}")
        logger.info(f"  Epochs       : {train_cfg['num_epochs']}")
        logger.info(f"  Batch size   : {self.config['data']['batch_size']}")
        logger.info(f"  Grad accum   : {train_cfg.get('gradient_accumulation_steps', 4)}")
        logger.info(f"  Warmup       : {train_cfg.get('warmup_epochs', 5)} epochs")
        logger.info(f"  Train samples: {self.data_info.get('num_train', '?')} "
                    f"(freq filter: {self.config['data'].get('training_frequencies')})")
        logger.info(f"  Val 5kHz     : {self.data_info.get('num_val_5khz', '?')}")
        logger.info(f"  Val 500Hz    : {self.data_info.get('num_val_500hz', '?')}")
        logger.info(f"  Val all      : {self.data_info.get('num_val_all', '?')}")
        logger.info(f"  ES monitor   : {'500Hz L1' if self.data_info.get('monitor_500hz') else '5kHz L1'}")
        loss_cfg = self.config.get("loss", {})
        logger.info(
            f"  Loss weights : L1={loss_cfg.get('l1_weight', 1.0)}, "
            f"MSE={loss_cfg.get('mse_weight', 1.0)}, "
            f"GDL={loss_cfg.get('gdl_weight', 0.3)}, "
            f"Temporal={loss_cfg.get('temporal_weight', 0.3)}"
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Verification Gates
    # ------------------------------------------------------------------

    def _check_gate(
        self, gate_key: str, value: float, label: str, lower_is_better: bool = True
    ):
        gates = self.config.get("verification_gates", {})
        threshold = gates.get(gate_key)
        if threshold is None:
            return
        passed = (value <= threshold) if lower_is_better else (value >= threshold)
        status = "PASS" if passed else "WARN"
        cmp = "≤" if lower_is_better else "≥"
        logger.info(
            f"[{gate_key} {status}] {label}={value:.4f} (target {cmp}{threshold:.4f})"
        )
        if not passed:
            logger.warning(
                f"Gate {gate_key} NOT MET: {label}={value:.4f} "
                f"(required {cmp}{threshold})"
            )

    def _check_collapse_alarm(self, r_active: float):
        """Warn if r_active drops more than 0.1 from its running maximum."""
        if r_active > self._r_active_max:
            self._r_active_max = r_active
        elif self._r_active_max > 0.15 and (self._r_active_max - r_active) > 0.1:
            logger.warning(
                f"[COLLAPSE ALARM] r_active={r_active:.3f} dropped "
                f"{self._r_active_max - r_active:.3f} from max {self._r_active_max:.3f}. "
                f"Monitor output_std and loss components closely."
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
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
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            dt_seconds = batch["dt_seconds"].to(self.device)
            B = x.shape[0]

            accum_divisor = accum_steps
            if remainder and batch_idx >= flush_start:
                accum_divisor = remainder

            if self.scaler is not None:
                with autocast(self.device_type):
                    velocity = self.model(x, dt_seconds=dt_seconds)
                    T_pred = velocity.shape[2]
                    target_crop = target[:, :, :T_pred]
                    components = self.criterion(velocity, target_crop, mask=mask, **self._loss_kwargs(batch, T_pred))
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                velocity = self.model(x, dt_seconds=dt_seconds)
                T_pred = velocity.shape[2]
                target_crop = target[:, :, :T_pred]
                components = self.criterion(velocity, target_crop, mask=mask, **self._loss_kwargs(batch, T_pred))
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            for k, v in components.items():
                val_item = v.item() if isinstance(v, torch.Tensor) else float(v)
                running_components[k] = running_components.get(k, 0.0) + val_item * B
            count += B
            self.global_step += 1

            if batch_idx % log_interval == 0:
                logger.info(
                    f"  Batch {batch_idx}/{num_batches} — "
                    f"total: {components['total'].item():.4f}, "
                    f"L1: {components['l1'].item():.4f}, "
                    f"temporal: {components['temporal'].item():.4f}"
                )

        if nan_count > 0:
            logger.warning(f"Epoch had {nan_count} NaN/Inf batches (skipped)")

        return {k: v / max(count, 1) for k, v in running_components.items()}

    @torch.no_grad()
    def _validate_loader(
        self, loader: DataLoader, tag: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.model.eval()
        running_components: Dict[str, float] = {}
        count = 0
        all_r_active = []
        all_output_std = []

        for batch in loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            dt_seconds = batch["dt_seconds"].to(self.device)
            B = x.shape[0]

            if self.scaler is not None:
                with autocast(self.device_type):
                    velocity = self.model(x, dt_seconds=dt_seconds)
                    T_pred = velocity.shape[2]
                    target_crop = target[:, :, :T_pred]
                    components = self.criterion(velocity, target_crop, mask=mask, **self._loss_kwargs(batch, T_pred))
            else:
                velocity = self.model(x, dt_seconds=dt_seconds)
                T_pred = velocity.shape[2]
                target_crop = target[:, :, :T_pred]
                components = self.criterion(velocity, target_crop, mask=mask, **self._loss_kwargs(batch, T_pred))

            for k, v in components.items():
                val_item = v.item() if isinstance(v, torch.Tensor) else float(v)
                running_components[k] = running_components.get(k, 0.0) + val_item * B
            count += B

            ext = compute_validation_metrics(
                velocity.float().cpu(),
                target_crop.float().cpu(),
                mask.float().cpu() if mask is not None else None,
            )
            if "r_active" in ext:
                all_r_active.append(ext["r_active"])
            if "output_std" in ext:
                all_output_std.append(ext["output_std"])

        self.model.train()

        loss_metrics = {k: v / max(count, 1) for k, v in running_components.items()}
        ext_metrics: Dict[str, float] = {}
        if all_r_active:
            ext_metrics["r_active"] = float(np.mean(all_r_active))
        if all_output_std:
            ext_metrics["output_std"] = float(np.mean(all_output_std))

        return loss_metrics, ext_metrics

    def validate(self) -> Tuple[Dict, Dict, Dict, Optional[Dict]]:
        loss_5khz, ext_5khz = self._validate_loader(self.val_loader_5khz, "5kHz")
        loss_all, _ = self._validate_loader(self.val_loader_all, "all")
        if self.val_loader_500hz is not None:
            loss_500hz, ext_500hz = self._validate_loader(self.val_loader_500hz, "500Hz")
        else:
            loss_500hz, ext_500hz = {}, {}
        return loss_5khz, ext_5khz, loss_all, loss_500hz, ext_500hz

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
            saved_dir = Path("saved_models")
            saved_dir.mkdir(exist_ok=True)
            exp_name = self.config.get("logging", {}).get("experiment_name", "wavepinn_v10")
            torch.save(ckpt, saved_dir / f"{exp_name}_best.pt")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
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
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {self.current_epoch}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        num_epochs = self.config["training"]["num_epochs"]
        warmup_epochs = self.config["training"].get("warmup_epochs", 5)
        base_lr = self.config["training"]["learning_rate"]
        save_interval = self.config.get("logging", {}).get("save_interval", 10)
        es_cfg = self.config.get("early_stopping", {})
        patience = es_cfg.get("patience", 40)
        min_delta = es_cfg.get("min_delta", 1e-5)
        gates = self.config.get("verification_gates", {})

        logger.info("Starting V10 training (single-phase pure pixel loss)...")

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch

            # Update LR
            lr = get_lr(epoch, num_epochs, base_lr, warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            logger.info(f"\nEpoch {epoch}/{num_epochs} | lr={lr:.2e}")
            logger.info("-" * 50)

            train_metrics = self.train_epoch()
            logger.info(
                f"Train — total: {train_metrics.get('total', 0):.4f}, "
                f"L1: {train_metrics.get('l1', 0):.4f}, "
                f"MSE: {train_metrics.get('mse', 0):.4f}, "
                f"temporal: {train_metrics.get('temporal', 0):.4f}, "
                f"gdl: {train_metrics.get('gdl', 0):.4f}"
            )

            loss_5khz, ext_5khz, loss_all, loss_500hz, ext_500hz = self.validate()

            r_active_5k  = ext_5khz.get("r_active", float("nan"))
            output_std   = ext_5khz.get("output_std", float("nan"))

            logger.info(
                f"Val 5kHz  — L1: {loss_5khz.get('l1', float('nan')):.4f}, "
                f"MSE: {loss_5khz.get('mse', float('nan')):.4f}, "
                f"r_active: {r_active_5k:.3f}, "
                f"output_std: {output_std:.4f}"
            )
            logger.info(f"Val all   — L1: {loss_all.get('l1', float('nan')):.4f}")

            if loss_500hz:
                r_active_500 = ext_500hz.get("r_active", float("nan"))
                logger.info(
                    f"Val 500Hz — L1: {loss_500hz.get('l1', float('nan')):.4f}, "
                    f"MSE: {loss_500hz.get('mse', float('nan')):.4f}, "
                    f"r_active: {r_active_500:.3f}"
                )

            # Early stopping metric: 500Hz L1 if monitored, else 5kHz L1
            if self._monitor_500hz and loss_500hz:
                val_l1    = loss_500hz.get("l1", float("inf"))
                r_active  = ext_500hz.get("r_active", float("nan"))
            else:
                val_l1    = loss_5khz.get("l1", float("inf"))
                r_active  = r_active_5k

            # Collapse alarm — reset max each epoch to avoid over-sensitivity
            # from a single early-training peak suppressing later valid drops.
            self._r_active_max = max(self._r_active_max * 0.98, 0.0)  # slow decay
            if not np.isnan(r_active):
                self._check_collapse_alarm(r_active)

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
                for k, v in loss_500hz.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val_500hz/{k}", v, epoch)
                for k, v in ext_500hz.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val_500hz/{k}", v, epoch)
                self.writer.add_scalar("train/lr", lr, epoch)

            # Verification gates
            if epoch == gates.get("G1_epoch", 1):
                std_ok = not np.isnan(output_std) and output_std > gates.get("G1_output_std_min", 0.01)
                logger.info(
                    f"[G1 {'PASS' if std_ok else 'WARN'}] "
                    f"output_std={output_std:.4f} "
                    f"(target > {gates.get('G1_output_std_min', 0.01):.3f})"
                )
            if epoch == gates.get("G2_epoch", 60):
                logger.info(f"[GATE G2] epoch {epoch}")
                self._check_gate("G2_val_pixel_loss_max", val_l1, "val_pixel_loss_5khz")
                if not np.isnan(r_active):
                    self._check_gate("G2_r_active_min", r_active, "r_active", lower_is_better=False)
            if epoch == gates.get("G3_epoch", 120):
                logger.info(f"[GATE G3] epoch {epoch}")
                self._check_gate("G3_val_pixel_loss_max", val_l1, "val_pixel_loss_5khz")
                if not np.isnan(r_active):
                    self._check_gate("G3_r_active_min", r_active, "r_active", lower_is_better=False)
            if epoch == num_epochs:
                logger.info(f"[GATE G4] final epoch {epoch}")
                self._check_gate("G4_val_pixel_loss_max", val_l1, "val_pixel_loss_5khz")
                if not np.isnan(r_active):
                    self._check_gate("G4_r_active_min", r_active, "r_active", lower_is_better=False)

            # Primary early stopping: 5kHz pixel L1
            if val_l1 < self.best_val_loss - min_delta:
                self.best_val_loss = val_l1
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
    parser = argparse.ArgumentParser(description="Train WavePINN V10")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    trainer = TrainerV10(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
