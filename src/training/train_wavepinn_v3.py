"""
Training script for WavePINN v3 with 3-phase curriculum learning.

Usage:
    python -m src.training.train_wavepinn_v3 --config configs/config_wavepinn_v3.json
    python -m src.training.train_wavepinn_v3 --config configs/config_wavepinn_v3.json --resume outputs/wavepinn_v3/checkpoint.pt

Curriculum Phases:
    Phase 1 (Pretrain):  Epochs 1-80,   PRF >= 250 Hz,  wave 0 -> 0.05 ramp
    Phase 2 (Mixed):     Epochs 81-180, All PRFs,       wave 0.1, all losses
    Phase 3 (Finetune):  Epochs 181-250, All PRFs,      wave 0.2, lower LR
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from src.architectures.wavepinn_v3 import WavePINNv3, create_wavepinn_v3, count_parameters_v3
from src.losses.physics_loss_v3 import EnhancedVelocityLossV3
from src.utils.iq_velocity_dataset_v3 import create_iq_velocity_dataloaders_v3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =========================================================================
# Curriculum Phase Logic
# =========================================================================

def get_phase(epoch: int, curriculum: Dict) -> Tuple[str, Dict]:
    """Return (phase_name, phase_config) for the given 1-based epoch."""
    for key in ("phase_1", "phase_2", "phase_3"):
        phase = curriculum[key]
        start, end = phase["epochs"]
        if start <= epoch <= end:
            return phase["name"], phase
    # Fallback to last phase
    return curriculum["phase_3"]["name"], curriculum["phase_3"]


def get_loss_weights(epoch: int, phase_cfg: Dict) -> Dict[str, float]:
    """Resolve loss weights for the current epoch, handling ramp-ups."""
    weights = dict(phase_cfg["loss_weights"])

    # Handle wave ramp in phase 1
    if "wave_start" in weights and "wave_end" in weights:
        start_epoch, end_epoch = phase_cfg["epochs"]
        total = max(end_epoch - start_epoch, 1)
        progress = min(1.0, max(0.0, (epoch - start_epoch) / total))
        # Smooth cosine ramp
        w = weights["wave_start"] + (weights["wave_end"] - weights["wave_start"]) * (
            1 - math.cos(math.pi * progress)
        ) / 2
        weights["wave"] = w
        del weights["wave_start"]
        del weights["wave_end"]

    return weights


def get_lr_for_phase(epoch: int, phase_cfg: Dict, base_lr: float) -> float:
    """Compute learning rate for the current epoch within the phase."""
    phase_lr = phase_cfg.get("lr", base_lr)
    start_epoch, end_epoch = phase_cfg["epochs"]

    # Warmup (phase 1 only)
    warmup = phase_cfg.get("warmup_epochs", 0)
    if warmup > 0 and (epoch - start_epoch) < warmup:
        return phase_lr * (epoch - start_epoch + 1) / warmup

    # Cosine decay within phase
    total = max(end_epoch - start_epoch - warmup, 1)
    progress = min(1.0, (epoch - start_epoch - warmup) / total)
    min_lr = 1e-6
    return min_lr + (phase_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


# =========================================================================
# Trainer
# =========================================================================

class TrainerV3:
    """Training orchestrator for WavePINN v3."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reproducibility
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Output directory
        exp_name = config.get("logging", {}).get("experiment_name", "wavepinn_v3")
        output_dir = config.get("logging", {}).get("output_dir", "outputs")
        self.output_dir = Path(output_dir) / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File logger
        fh = logging.FileHandler(self.output_dir / "training.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tensorboard") if HAS_TENSORBOARD else None

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.current_phase = None  # Track phase for reset at transitions

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Initialise model, data, loss, optimiser."""
        self._setup_data()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        if self.config["training"].get("use_amp", True):
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        self._log_setup()

    def _setup_data(self):
        data_cfg = self.config["data"]
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
        self.model = create_wavepinn_v3(config=model_cfg)
        self.model = self.model.to(self.device)

        params = count_parameters_v3(self.model)
        logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    def _setup_criterion(self):
        phys = self.config.get("physics", {})
        # Start with phase 1 weights
        p1_weights = self.config["curriculum"]["phase_1"]["loss_weights"]
        self.criterion = EnhancedVelocityLossV3(
            l1_weight=p1_weights.get("l1", 1.0),
            mse_weight=p1_weights.get("mse", 1.0),
            gdl_weight=p1_weights.get("gdl", 0.0),
            temporal_weight=p1_weights.get("temporal", 0.0),
            ssim_weight=p1_weights.get("ssim", 0.0),
            wave_weight=p1_weights.get("wave", p1_weights.get("wave_start", 0.0)),
            init_cs=phys.get("init_cs", 3.0),
            dx_m=phys.get("dx_m", 1.013e-4),
            dy_m=phys.get("dy_m", 1.013e-4),
        )
        self.criterion = self.criterion.to(self.device)

    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        physics_lr_mult = train_cfg.get("physics_lr_multiplier", 0.1)
        base_lr = train_cfg["learning_rate"]

        model_params = list(self.model.parameters())
        physics_params = list(self.criterion.parameters())

        param_groups = [{"params": model_params, "lr": base_lr}]
        if physics_params:
            param_groups.append({
                "params": physics_params,
                "lr": base_lr * physics_lr_mult,
                "weight_decay": 0.0,
            })

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        )

    def _log_setup(self):
        logger.info("=" * 60)
        logger.info("WavePINN v3 Training")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size: {self.config['data']['batch_size']}")
        logger.info(f"  Train samples: {self.data_info.get('num_train', '?')}")
        logger.info(f"  Val samples: {self.data_info.get('num_val', '?')}")
        logger.info(f"  Upsampling factor: {self.config['model']['upsampling_factor']}")
        logger.info(f"  Initial cs: {self.config.get('physics', {}).get('init_cs', 3.0)} m/s")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Curriculum update
    # ------------------------------------------------------------------

    def _update_curriculum(self, epoch: int):
        """Update loss weights, LR, and reset early stopping at phase transitions."""
        curriculum = self.config["curriculum"]
        phase_name, phase_cfg = get_phase(epoch, curriculum)
        loss_weights = get_loss_weights(epoch, phase_cfg)
        lr = get_lr_for_phase(epoch, phase_cfg, self.config["training"]["learning_rate"])

        # Update loss weights
        self.criterion.set_weights(**loss_weights)

        # Update learning rate
        for i, pg in enumerate(self.optimizer.param_groups):
            if i == 0:
                pg["lr"] = lr
            else:
                # Physics params get reduced LR
                pg["lr"] = lr * self.config["training"].get("physics_lr_multiplier", 0.1)

        # Reset early stopping at phase transitions
        if phase_name != self.current_phase:
            if self.current_phase is not None:
                logger.info(
                    f"Phase transition '{self.current_phase}' -> '{phase_name}': "
                    f"resetting early stopping (was best={self.best_val_loss:.4f}, "
                    f"patience={self.patience_counter})"
                )
                self.best_val_loss = float("inf")
                self.patience_counter = 0
            self.current_phase = phase_name
            logger.info(
                f"=== Phase '{phase_name}' started (epoch {epoch}) === "
                f"LR={lr:.2e}, wave={loss_weights.get('wave', 0):.4f}"
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

        running_components: Dict[str, float] = {}
        count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x = batch["input"].to(self.device)          # (B, 2, T_sparse, H, W)
            target = batch["target"].to(self.device)     # (B, 1, T_vel, H, W)
            mask = batch["mask"].to(self.device)         # (B, 1, 1, H, W)
            dt_seconds = batch["dt_seconds"].to(self.device)  # (B,)
            dt_output = batch["dt_output"].to(self.device)    # (B,)

            B = x.shape[0]

            if self.scaler is not None:
                with autocast('cuda'):
                    pred = self.model(x, dt_seconds=dt_seconds)
                    # Crop target to match pred temporal dim
                    T_pred = pred.shape[2]
                    target_crop = target[:, :, :T_pred]
                    loss, components = self.criterion(
                        pred, target_crop, mask=mask, dt_seconds=dt_output,
                    )
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.criterion.parameters()),
                        max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                pred = self.model(x, dt_seconds=dt_seconds)
                T_pred = pred.shape[2]
                target_crop = target[:, :, :T_pred]
                loss, components = self.criterion(
                    pred, target_crop, mask=mask, dt_seconds=dt_output,
                )
                loss = loss / accum_steps
                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.criterion.parameters()),
                        max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Accumulate metrics
            for k, v in components.items():
                running_components[k] = running_components.get(k, 0) + v * B
            count += B
            self.global_step += 1

            if batch_idx % log_interval == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {components.get('total', 0):.4f}, "
                    f"cs: {components.get('wave_cs', 0):.3f} m/s"
                )

        return {k: v / max(count, 1) for k, v in running_components.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_components: Dict[str, float] = {}
        count = 0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)
            dt_seconds = batch["dt_seconds"].to(self.device)
            dt_output = batch["dt_output"].to(self.device)
            B = x.shape[0]

            if self.scaler is not None:
                with autocast('cuda'):
                    pred = self.model(x, dt_seconds=dt_seconds)
                    T_pred = pred.shape[2]
                    target_crop = target[:, :, :T_pred]
                    _, components = self.criterion(
                        pred, target_crop, mask=mask, dt_seconds=dt_output,
                    )
            else:
                pred = self.model(x, dt_seconds=dt_seconds)
                T_pred = pred.shape[2]
                target_crop = target[:, :, :T_pred]
                _, components = self.criterion(
                    pred, target_crop, mask=mask, dt_seconds=dt_output,
                )

            for k, v in components.items():
                running_components[k] = running_components.get(k, 0) + v * B
            count += B

        return {k: v / max(count, 1) for k, v in running_components.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str, is_best: bool = False):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "learned_cs": self.criterion.wave_reg.cs_value,
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

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "criterion_state_dict" in ckpt:
            self.criterion.load_state_dict(ckpt["criterion_state_dict"])
        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        logger.info(
            f"Loaded checkpoint from epoch {self.current_epoch}, "
            f"cs = {ckpt.get('learned_cs', '?')} m/s"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop with curriculum."""
        num_epochs = self.config["training"]["num_epochs"]
        save_interval = self.config.get("logging", {}).get("save_interval", 10)
        patience = self.config.get("early_stopping", {}).get("patience", 40)

        logger.info("Starting training...")

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch

            # Update curriculum (loss weights + LR)
            self._update_curriculum(epoch)

            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 40)

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Train - Loss: {train_metrics.get('total', 0):.4f}, "
                f"L1: {train_metrics.get('l1', 0):.4f}, "
                f"Wave: {train_metrics.get('wave', 0):.6f}, "
                f"cs: {train_metrics.get('wave_cs', 0):.3f} m/s"
            )

            if self.writer:
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, epoch)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], epoch,
                )

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics.get("total", float("inf"))
            logger.info(f"Val   - Loss: {val_loss:.4f}")

            if self.writer:
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, epoch)

            # Best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt", is_best=True)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Periodic checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")
        logger.info(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

        if self.writer:
            self.writer.close()


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train WavePINN v3")
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

    trainer = TrainerV3(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
