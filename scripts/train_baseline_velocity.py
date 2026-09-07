#!/usr/bin/env python3
"""
Train baseline architectures (PI-CV-RecUNet, BiLSTM) to predict velocity directly.

These baselines originally predicted IQ output; this script wraps them with a
complex-to-real velocity head so they can be fairly compared against the final
ComplexToRealUNet (v2) architecture on the same dataset and loss.

Usage:
    python scripts/train_baseline_velocity.py \
        --model pinn --epochs 20

    python scripts/train_baseline_velocity.py \
        --model bilstm --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.pinn_recunet import PI_CV_RecUNet
from src.architectures.bilstm_model import PI_CV_RecUNet_BiLSTM
from src.utils.iq_velocity_dataset import (
    IQVelocityDataset,
    create_iq_velocity_dataloaders,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Wrapper: adds a complex-to-real velocity head to any baseline model
# ============================================================================

class BaselineVelocityWrapper(nn.Module):
    """
    Wraps a baseline complex-output model (PI-CV-RecUNet or BiLSTM) to
    predict velocity directly.

    Input:  (B, 2, T_sparse, H, W) — same as IQVelocityDataset output
    Output: (B, 1, T_dense, H, W)  — normalised velocity in [-1, 1]
    """

    def __init__(self, baseline_model: nn.Module):
        super().__init__()
        self.baseline = baseline_model

        # Learnable complex-to-real velocity head:
        #   concatenate real+imag (2 ch) → conv → velocity (1 ch) → tanh
        self.velocity_head = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, log_prf=None) -> torch.Tensor:
        """
        Args:
            x: (B, 2, T_sparse, H, W) from IQVelocityDataset
        Returns:
            (B, 1, T_dense, H, W) velocity in [-1, 1]
        """
        B, two, T_sparse, H, W = x.shape

        # Reshape to baseline format: (B, T_sparse, 2, 1, H, W)
        x_bl = x.permute(0, 2, 1, 3, 4).unsqueeze(3)

        # Pad spatial dims to multiple of 8 (for 3-level encoder with 2x pooling)
        mult = 2 ** self.baseline.num_levels
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h > 0 or pad_w > 0:
            x_bl = F.pad(x_bl, (0, pad_w, 0, pad_h))

        # Run baseline: (B, T_dense, 2, 1, H_pad, W_pad)
        out = self.baseline(x_bl)

        _, T_dense, _, C_out, H_pad, W_pad = out.shape

        # Extract real and imaginary parts: each (B, T_dense, H_pad, W_pad)
        real = out[:, :, 0, 0]
        imag = out[:, :, 1, 0]

        # Stack to (B*T_dense, 2, H_pad, W_pad) for conv
        real_flat = real.reshape(B * T_dense, 1, H_pad, W_pad)
        imag_flat = imag.reshape(B * T_dense, 1, H_pad, W_pad)
        combined = torch.cat([real_flat, imag_flat], dim=1)

        # Apply velocity head: (B*T_dense, 1, H_pad, W_pad)
        vel = self.velocity_head(combined)

        # Crop padding and reshape
        vel = vel[:, :, :H, :W]
        vel = vel.reshape(B, T_dense, 1, H, W)
        vel = vel.permute(0, 2, 1, 3, 4)  # (B, 1, T_dense, H, W)

        return vel

    def get_flow_stats(self):
        return {}

    def get_motion_interpolation(self):
        return None


# ============================================================================
# Loss
# ============================================================================

class SimpleVelocityLoss(nn.Module):
    """L1 + MSE loss with optional mask weighting (matches v2 config)."""

    def __init__(self, l1_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target, mask=None):
        if mask is not None:
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(pred)
            l1 = (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)
            mse = (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        else:
            l1 = torch.abs(pred - target).mean()
            mse = ((pred - target) ** 2).mean()

        total = self.l1_weight * l1 + self.mse_weight * mse
        return total, {"total": total.item(), "l1": l1.item(), "mse": mse.item()}


# ============================================================================
# Training
# ============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(42)
    np.random.seed(42)

    # --- Data (same as v2, batch_size=1 for memory) ---
    train_loader, val_loader, data_info = create_iq_velocity_dataloaders(
        data_dir="data/fullscale_velocity",
        batch_size=1,
        num_workers=4,
        per_sample_iq_norm=True,
        per_sample_vel_norm=True,
    )
    logger.info(f"Data: {data_info}")

    T_sparse = data_info.get("T_sparse", 5)
    T_dense = data_info.get("T_dense", 49)
    upsampling_factor = (T_dense - 1) // (T_sparse - 1)
    logger.info(f"T_sparse={T_sparse}, T_dense={T_dense}, factor={upsampling_factor}")

    # --- Model ---
    # Use num_levels=3 to keep training feasible (complex decoder decodes
    # all 49 frames sequentially, so smaller models are critical for speed).
    if args.model == "pinn":
        baseline = PI_CV_RecUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            num_levels=3,
            lstm_hidden_channels=128,
            lstm_num_layers=1,
            upsampling_factor=upsampling_factor,
        )
        model_name = "PI-CV-RecUNet"
    elif args.model == "bilstm":
        baseline = PI_CV_RecUNet_BiLSTM(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            num_levels=3,
            lstm_hidden_channels=128,
            lstm_num_layers=1,
            upsampling_factor=upsampling_factor,
            bilstm_combine_mode="concat",
        )
        model_name = "BiLSTM-RecUNet"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = BaselineVelocityWrapper(baseline).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name} (velocity wrapper)  params={n_params:,}")

    # --- Optimizer (same as v2) ---
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=True)
    criterion = SimpleVelocityLoss(l1_weight=1.0, mse_weight=1.0)

    # --- Output ---
    exp_name = f"baseline_{args.model}_velocity"
    out_dir = Path("outputs") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # --- Training loop ---
    best_val_loss = float("inf")
    grad_accum = 4

    for epoch in range(args.epochs):
        model.train()
        train_meter = AverageMeter()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)

            with autocast("cuda", enabled=True):
                pred = model(inputs)
                T_pred = pred.shape[2]
                T_tgt = targets.shape[2]
                if T_pred != T_tgt:
                    # Trim to match
                    min_T = min(T_pred, T_tgt)
                    pred = pred[:, :, :min_T]
                    targets_use = targets[:, :, :min_T]
                else:
                    targets_use = targets
                loss, comps = criterion(pred, targets_use, masks)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_meter.update(comps["total"], inputs.shape[0])
            pbar.set_postfix(loss=f"{train_meter.avg:.4f}")

        # Validation
        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                masks = batch["mask"].to(device)

                with autocast("cuda", enabled=True):
                    pred = model(inputs)
                    T_pred = pred.shape[2]
                    T_tgt = targets.shape[2]
                    min_T = min(T_pred, T_tgt)
                    _, comps = criterion(pred[:, :, :min_T], targets[:, :, :min_T], masks)
                val_meter.update(comps["total"], inputs.shape[0])

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        is_best = val_meter.avg < best_val_loss

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}  lr={lr:.2e}  "
            f"train={train_meter.avg:.6f}  val={val_meter.avg:.6f}"
            f"{'  ★ BEST' if is_best else ''}"
        )

        if is_best:
            best_val_loss = val_meter.avg
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {
                    "model_type": args.model,
                    "model_name": model_name,
                    "upsampling_factor": upsampling_factor,
                    "base_channels": 32,
                    "num_levels": 3,
                    "lstm_hidden_channels": 128,
                },
                "data_info": data_info,
            }
            torch.save(ckpt, out_dir / "best_model.pt")
            # Also save to saved_models for easy access
            saved_dir = PROJECT_ROOT / "saved_models"
            saved_dir.mkdir(exist_ok=True)
            torch.save(ckpt, saved_dir / f"{exp_name}_best.pt")
            logger.info(f"  Saved best model (val={best_val_loss:.6f})")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["pinn", "bilstm"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args)
