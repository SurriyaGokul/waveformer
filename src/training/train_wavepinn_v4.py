"""
Training script for WavePINN v4 with curriculum learning.

Usage:
    python -m src.training.train_wavepinn_v4 --config configs/config_wavepinn_v4.json
    python -m src.training.train_wavepinn_v4 --config configs/config_wavepinn_v4.json --resume outputs/wavepinn_v4/checkpoint.pt

Architecture: Conditional Wave Propagation Network
    - Bidirectional ConvGRU with dt-conditioning and observation anchoring
    - Wave content gate identifies informative frames
    - Learned wave propagation prior transfers from high-PRF to low-PRF
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

from src.architectures.wavepinn_v4 import WavePINNv4, create_wavepinn_v4, count_parameters_v4
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
        if key not in curriculum:
            continue
        phase = curriculum[key]
        start, end = phase["epochs"]
        if start <= epoch <= end:
            return phase["name"], phase
    # Fallback to last defined phase
    for key in ("phase_3", "phase_2", "phase_1"):
        if key in curriculum:
            return curriculum[key]["name"], curriculum[key]
    raise ValueError("No curriculum phases defined")


def get_loss_weights(epoch: int, phase_cfg: Dict) -> Dict[str, float]:
    """Resolve loss weights for the current epoch, handling ramp-ups."""
    weights = dict(phase_cfg["loss_weights"])

    if "wave_start" in weights and "wave_end" in weights:
        start_epoch, end_epoch = phase_cfg["epochs"]
        total = max(end_epoch - start_epoch, 1)
        progress = min(1.0, max(0.0, (epoch - start_epoch) / total))
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

    warmup = phase_cfg.get("warmup_epochs", 0)
    if warmup > 0 and (epoch - start_epoch) < warmup:
        return phase_lr * (epoch - start_epoch + 1) / warmup

    total = max(end_epoch - start_epoch - warmup, 1)
    progress = min(1.0, (epoch - start_epoch - warmup) / total)
    min_lr = 1e-6
    return min_lr + (phase_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


# =========================================================================
# Trainer
# =========================================================================

class TrainerV4:
    """Training orchestrator for WavePINN v4."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        exp_name = config.get("logging", {}).get("experiment_name", "wavepinn_v4")
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
        self.model = create_wavepinn_v4(config=model_cfg)
        self.model = self.model.to(self.device)

        params = count_parameters_v4(self.model)
        logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    def _setup_criterion(self):
        phys = self.config.get("physics", {})
        p1_weights = self.config["curriculum"]["phase_1"]["loss_weights"]
        self.criterion = EnhancedVelocityLossV3(
            l1_weight=p1_weights.get("l1", 1.0),
            mse_weight=p1_weights.get("mse", 1.0),
            gdl_weight=p1_weights.get("gdl", 0.0),
            temporal_weight=p1_weights.get("temporal", 0.0),
            ssim_weight=p1_weights.get("ssim", 0.0),
            wave_weight=p1_weights.get("wave", p1_weights.get("wave_start", 0.0)),
            fg_l1_weight=p1_weights.get("fg_l1", 0.0),
            init_cs=phys.get("init_cs", 3.0),
            dx_m=phys.get("dx_m", 1.013e-4),
            dy_m=phys.get("dy_m", 5.07e-5),
        )
        self.criterion = self.criterion.to(self.device)

    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        physics_lr_mult = train_cfg.get("physics_lr_multiplier", 0.1)
        propagator_lr_mult = train_cfg.get("propagator_lr_multiplier", 5.0)
        base_lr = train_cfg["learning_rate"]

        # Separate param groups for gradient balancing:
        # 1. Encoder/decoder (large, already learns well) — base LR
        # 2. Propagator + gate (small, starved of gradients) — higher LR
        # 3. Physics params (cs regularizer) — low LR
        encoder_decoder_params = []
        propagator_gate_params = []

        for name, param in self.model.named_parameters():
            if any(k in name for k in ["propagator", "wave_gate", "pre_proj"]):
                propagator_gate_params.append(param)
            else:
                encoder_decoder_params.append(param)

        physics_params = list(self.criterion.parameters())

        param_groups = [
            {"params": encoder_decoder_params, "lr": base_lr, "name": "enc_dec"},
            {"params": propagator_gate_params, "lr": base_lr * propagator_lr_mult, "name": "propagator"},
        ]
        if physics_params:
            param_groups.append({
                "params": physics_params,
                "lr": base_lr * physics_lr_mult,
                "weight_decay": 0.0,
                "name": "physics",
            })

        logger.info(
            f"Optimizer param groups: enc_dec={len(encoder_decoder_params)} params (lr={base_lr}), "
            f"propagator={len(propagator_gate_params)} params (lr={base_lr * propagator_lr_mult}), "
            f"physics={len(physics_params)} params (lr={base_lr * physics_lr_mult})"
        )

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        )

    def _log_setup(self):
        logger.info("=" * 60)
        logger.info("WavePINN v4 Training — Conditional Wave Propagation Network")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size: {self.config['data']['batch_size']}")
        logger.info(f"  Train samples: {self.data_info.get('num_train', '?')}")
        logger.info(f"  Val samples: {self.data_info.get('num_val', '?')}")
        logger.info(f"  Upsampling factor: {self.config['model']['upsampling_factor']}")
        logger.info(f"  Propagator channels: {self.config['model'].get('propagator_channels', 256)}")
        logger.info(f"  Initial cs: {self.config.get('physics', {}).get('init_cs', 3.0)} m/s")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_diagnostics(self, epoch: int):
        """Log gradient norms, wave gate scores, prediction stats per epoch."""
        # Gradient norms per component group
        groups = {
            "encoder": [self.model.enc1, self.model.enc2, self.model.enc3, self.model.enc4],
            "pre_proj": [self.model.pre_proj],
            "wave_gate": [self.model.wave_gate],
            "fwd_gru": [self.model.propagator.fwd_gru],
            "bwd_gru": [self.model.propagator.bwd_gru],
            "fusion": [self.model.propagator.fusion],
            "temporal_refine": [self.model.propagator.temporal_refine],
            "decoder": [self.model.dec4, self.model.dec3, self.model.dec2, self.model.dec1],
            "output_head": [self.model.output_conv],
        }
        grad_strs = []
        for name, modules in groups.items():
            total_norm = 0.0
            count = 0
            for m in modules:
                for p in m.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        count += 1
            total_norm = total_norm ** 0.5
            grad_strs.append(f"{name}={total_norm:.4f}")

        # dt FiLM weight norm (indicates how much dt matters)
        fwd_film_w = self.model.propagator.fwd_gru.dt_film[2].weight.data.norm().item()
        bwd_film_w = self.model.propagator.bwd_gru.dt_film[2].weight.data.norm().item()

        logger.info(f"  Grad norms: {', '.join(grad_strs)}")
        logger.info(f"  dt FiLM weight norms: fwd={fwd_film_w:.4f}, bwd={bwd_film_w:.4f}")

        # Wave gate scores on a quick val batch
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["input"].to(self.device)
                dt_s = batch["dt_seconds"].to(self.device)
                # Run encoder + bridge + projection to get gate input
                x_enc, B, T = self.model._reshape_for_encoder(x)
                for enc in [self.model.enc1, self.model.enc2, self.model.enc3, self.model.enc4]:
                    x_enc, _ = enc(x_enc)
                _, _, C_b, H_b, W_b = x_enc.shape
                x_t = x_enc.reshape(B, T, 2, C_b, H_b, W_b)
                x_real = self.model.bottleneck_bridge(x_t)
                x_proj = self.model.pre_proj(x_real.reshape(B*T, -1, H_b, W_b))
                x_proj = x_proj.reshape(B, T, self.model.propagator_channels, H_b, W_b)
                scores = self.model.wave_gate(x_proj)
                logger.info(f"  Wave gate scores (val batch): {scores.cpu().numpy().round(3)}")

                # Also check prediction stats
                with autocast('cuda'):
                    pred = self.model(x, dt_seconds=dt_s)
                logger.info(
                    f"  Pred stats: min={pred.min():.4f}, max={pred.max():.4f}, "
                    f"mean={pred.mean():.4f}, std={pred.std():.4f}"
                )
                break  # one batch is enough
        self.model.train()

    # ------------------------------------------------------------------
    # Curriculum update
    # ------------------------------------------------------------------

    def _update_curriculum(self, epoch: int):
        curriculum = self.config["curriculum"]
        phase_name, phase_cfg = get_phase(epoch, curriculum)
        loss_weights = get_loss_weights(epoch, phase_cfg)
        lr = get_lr_for_phase(epoch, phase_cfg, self.config["training"]["learning_rate"])

        self.criterion.set_weights(**loss_weights)

        propagator_lr_mult = self.config["training"].get("propagator_lr_multiplier", 5.0)
        physics_lr_mult = self.config["training"].get("physics_lr_multiplier", 0.1)
        for pg in self.optimizer.param_groups:
            name = pg.get("name", "")
            if name == "propagator":
                pg["lr"] = lr * propagator_lr_mult
            elif name == "physics":
                pg["lr"] = lr * physics_lr_mult
            else:
                pg["lr"] = lr

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
        self.model.train()
        train_cfg = self.config["training"]
        accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
        max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        log_interval = self.config.get("logging", {}).get("log_interval", 5)

        running_components: Dict[str, float] = {}
        count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            mask = batch["mask"].to(self.device)
            dt_seconds = batch["dt_seconds"].to(self.device)
            dt_output = batch["dt_output"].to(self.device)
            B = x.shape[0]

            gate_entropy_weight = self.config["training"].get("gate_entropy_weight", 0.5)

            if self.scaler is not None:
                with autocast('cuda'):
                    pred, wave_scores = self.model(x, dt_seconds=dt_seconds, return_gate_scores=True)
                    T_pred = pred.shape[2]
                    target_crop = target[:, :, :T_pred]
                    loss, components = self.criterion(
                        pred, target_crop, mask=mask, dt_seconds=dt_output,
                    )
                    # Gate entropy regularization — prevents gate collapse
                    gate_loss = self.model.gate_entropy_loss(wave_scores)
                    loss = loss + gate_entropy_weight * gate_loss
                    components["gate_entropy"] = gate_loss.item()
                    components["gate_mean"] = wave_scores.mean().item()
                    loss = loss / accum_steps

                # NaN guard: skip step if loss is NaN/Inf
                if not torch.isfinite(loss):
                    logger.warning(f"  NaN/Inf loss at batch {batch_idx}, skipping step")
                    self.optimizer.zero_grad()
                    continue

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
                pred, wave_scores = self.model(x, dt_seconds=dt_seconds, return_gate_scores=True)
                T_pred = pred.shape[2]
                target_crop = target[:, :, :T_pred]
                loss, components = self.criterion(
                    pred, target_crop, mask=mask, dt_seconds=dt_output,
                )
                gate_loss = self.model.gate_entropy_loss(wave_scores)
                loss = loss + gate_entropy_weight * gate_loss
                components["gate_entropy"] = gate_loss.item()
                components["gate_mean"] = wave_scores.mean().item()
                loss = loss / accum_steps

                if not torch.isfinite(loss):
                    logger.warning(f"  NaN/Inf loss at batch {batch_idx}, skipping step")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.criterion.parameters()),
                        max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            for k, v in components.items():
                running_components[k] = running_components.get(k, 0) + v * B
            count += B
            self.global_step += 1

            if batch_idx % log_interval == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {components.get('total', 0):.4f}, "
                    f"cs: {components.get('wave_cs', 0):.3f} m/s, "
                    f"gate_mean: {components.get('gate_mean', 0):.3f}"
                )

        return {k: v / max(count, 1) for k, v in running_components.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
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
        num_epochs = self.config["training"]["num_epochs"]
        save_interval = self.config.get("logging", {}).get("save_interval", 10)
        patience = self.config.get("early_stopping", {}).get("patience", 40)

        logger.info("Starting training...")

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            self._update_curriculum(epoch)

            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 40)

            train_metrics = self.train_epoch()
            logger.info(
                f"Train - Loss: {train_metrics.get('total', 0):.4f}, "
                f"L1: {train_metrics.get('l1', 0):.4f}, "
                f"SSIM: {train_metrics.get('ssim', 0):.4f}, "
                f"Wave: {train_metrics.get('wave', 0):.6f}, "
                f"cs: {train_metrics.get('wave_cs', 0):.3f} m/s, "
                f"gate_mean: {train_metrics.get('gate_mean', 0):.3f}"
            )

            if self.writer:
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, epoch)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], epoch,
                )

            val_metrics = self.validate()
            val_loss = val_metrics.get("total", float("inf"))
            logger.info(
                f"Val   - Loss: {val_loss:.4f}, "
                f"L1: {val_metrics.get('l1', 0):.4f}, "
                f"SSIM: {val_metrics.get('ssim', 0):.4f}"
            )

            if self.writer:
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, epoch)

            # Diagnostics every 5 epochs
            if epoch % 5 == 1 or epoch <= 3:
                self._log_diagnostics(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
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
    parser = argparse.ArgumentParser(description="Train WavePINN v4")
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

    trainer = TrainerV4(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
