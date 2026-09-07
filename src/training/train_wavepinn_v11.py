"""
Training script for WavePINN v11 — deployment-focused hybrid (v9 + spectral adapter).

Usage:
    python -m src.training.train_wavepinn_v11 --config configs/config_wavepinn_v11.json
    python -m src.training.train_wavepinn_v11 --config configs/config_wavepinn_v11.json --resume outputs/wavepinn_v11/checkpoint.pt

v11 reuses the proven v9 curriculum/loss/dataloader stack and only overrides
model construction to instantiate WavePINNv11.
"""

from __future__ import annotations

import argparse
import json

from src.architectures.wavepinn_v11 import create_wavepinn_v11, count_parameters_v11
from src.training.train_wavepinn_v9 import TrainerV9, logger


class TrainerV11(TrainerV9):
    """v11 trainer: identical training loop to v9 with v11 model factory."""

    def _setup_model(self):
        model_cfg = dict(self.config["model"])
        model_cfg.pop("name", None)
        self.model = create_wavepinn_v11(config=model_cfg)
        self.model = self.model.to(self.device)

        params = count_parameters_v11(self.model)
        logger.info(
            f"Model parameters: {params['total']:,} total, "
            f"{params['backbone']:,} backbone, "
            f"{params['spectral_adapter']:,} spectral_adapter, "
            f"{params['cs_head']:,} cs_head"
        )

    def _log_setup(self):
        logger.info("=" * 60)
        logger.info("WavePINN v11 Training — Deployment-Focused Hybrid")
        logger.info(f"  Device       : {self.device}")
        logger.info(f"  Epochs       : {self.config['training']['num_epochs']}")
        logger.info(f"  Batch size   : {self.config['data']['batch_size']}")
        logger.info(
            f"  Train samples: {self.data_info.get('num_train', '?')} "
            f"(freq filter: {self.config['data'].get('training_frequencies')})"
        )
        logger.info(f"  Val 5kHz     : {self.data_info.get('num_val_5khz', '?')}")
        logger.info(f"  Val all      : {self.data_info.get('num_val_all', '?')}")
        logger.info(f"  Upsampling   : {self.config['model']['upsampling_factor']}")
        logger.info(
            f"  Spectral     : enabled={self.config['model'].get('use_spectral_calibrator', True)}, "
            f"hidden={self.config['model'].get('spectral_hidden_channels', 64)}, "
            f"modes=({self.config['model'].get('spectral_num_modes_height', 6)},"
            f"{self.config['model'].get('spectral_num_modes_width', 6)}), "
            f"layers={self.config['model'].get('spectral_num_layers', 2)}"
        )
        try:
            gate = self.model.spectral_gate()
            logger.info(f"  Spectral gate: init={gate:.4f}")
        except Exception:
            pass
        logger.info(
            f"  Cs range     : [{self.config['model'].get('cs_min', 0.3)}, "
            f"{self.config['model'].get('cs_max', 8.0)}] m/s"
        )
        logger.info(f"  Cs init      : {self.config['model'].get('cs_init', 2.54)} m/s")
        logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Train WavePINN v11")
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

    trainer = TrainerV11(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume} (epoch {trainer.current_epoch})")

    trainer.train()


if __name__ == "__main__":
    main()
