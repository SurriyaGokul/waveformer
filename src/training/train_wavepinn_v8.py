"""
Training script for WavePINN v8 — Fixed PhysicsHead gradient flow.

Usage:
    python -m src.training.train_wavepinn_v8 --config configs/config_wavepinn_v8.json
    python -m src.training.train_wavepinn_v8 --config configs/config_wavepinn_v8.json \
        --warmstart outputs/wavepinn_v7/best_model.pt

Changes from V7
---------------
1. Separate gradient clipping for backbone vs PhysicsHead.
   V7 clipped ALL parameters together under a shared max_norm=1.0.
   With a 16M-param backbone, the total gradient norm is backbone-dominated;
   the PhysicsHead's 7 parameter tensors get clipped down by the same ratio
   as the backbone (often ×0.05–0.2), yielding effective head lr orders of
   magnitude below the nominal 1e-3.  V8 clips each group independently:
     backbone:      max_grad_norm (default 1.0)
     physics_head:  physics_head_max_grad_norm (default 5.0)

2. cs_map and cs_mean weights ramp from 0 in phase_2 (not immediate full
   weight).  In V7 they jumped to 0.3 / 0.2 on epoch 11, before the head had
   any stable signal.  V8 ramps them over 30 epochs.

3. NaN guard in DiffToaCsLoss already added to swe_loss_v7.py (shared).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Re-use all V7 infrastructure unchanged
from src.training.train_wavepinn_v7 import TrainerV7, logger

import numpy as np


# =========================================================================
# TrainerV8 — V7 + per-param-group gradient clipping
# =========================================================================

class TrainerV8(TrainerV7):
    """
    Identical to TrainerV7 except:
      - backbone and physics_head gradients are clipped independently
      - experiment label is "wavepinn_v8"
    """

    def _setup_optimizer(self):
        super()._setup_optimizer()
        # Cache param group references for independent clipping.
        # param_groups[0] = backbone (set in TrainerV7._setup_optimizer)
        # param_groups[1] = physics_head
        self._backbone_params = self.optimizer.param_groups[0]["params"]
        self._physics_params  = self.optimizer.param_groups[1]["params"]
        self._physics_max_grad_norm = self.config["training"].get(
            "physics_head_max_grad_norm", 5.0
        )
        logger.info(
            f"V8: separate grad clipping — "
            f"backbone max_norm={self.config['training'].get('max_grad_norm', 1.0)}, "
            f"physics_head max_norm={self._physics_max_grad_norm}"
        )

    # ------------------------------------------------------------------
    # Override clipping hooks (called by TrainerV7.train_epoch)
    # ------------------------------------------------------------------

    def _clip_and_step(self, max_grad_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(self._backbone_params, max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self._physics_params, self._physics_max_grad_norm)
        self.optimizer.step()

    def _clip_and_step_amp(self, max_grad_norm: float) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self._backbone_params, max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self._physics_params, self._physics_max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _log_setup(self):
        import logging
        logger.info("=" * 60)
        logger.info("WavePINN v8 Training — Fixed PhysicsHead Gradient Flow")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size: {self.config['data']['batch_size']}")
        logger.info(f"  Train samples: {self.data_info.get('num_train', '?')}")
        logger.info(f"  Val samples: {self.data_info.get('num_val', '?')}")
        logger.info(f"  Upsampling: {self.config['model']['upsampling_factor']}")
        logger.info(f"  Cs range: [{self.config['model'].get('cs_min', 0.3)}, "
                     f"{self.config['model'].get('cs_max', 8.0)}] m/s")
        logger.info("=" * 60)


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train WavePINN v8")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--warmstart",  type=str, default=None,
                        help="Warm-start from V6 or V7 checkpoint")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--no_amp",     action="store_true")
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

    trainer = TrainerV8(config)
    trainer.setup()

    if args.warmstart:
        trainer.warmstart_from_v6(args.warmstart)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
