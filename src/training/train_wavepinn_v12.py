"""
Training script for WavePINN V12 — per-sample dt fix + leakage-free split.

Usage:
    python -m src.training.train_wavepinn_v12 --config configs/config_wavepinn_v12.json
    python -m src.training.train_wavepinn_v12 --config configs/config_wavepinn_v12.json --resume outputs/wavepinn_v12/checkpoint_epoch_60.pt

Key differences from V10b:
    - Backbone (wavepinn_v6.py) now threads per-sample dt through the
      bottleneck / positional encoding / attention / physics refinement
      instead of dt_seconds.mean() — mixed-PRF batches (batch_size=2) get
      correct per-sample time conditioning. Float-dt path is bit-identical,
      so this applies automatically; V12 needs no architecture changes.
    - Trained on the leakage-free split produced by scripts/regroup_splits.py
      (12 straddling acquisitions consolidated into train; frozen test split
      carved out and excluded from train/val).
    - Recipe (loss weights, epochs, LR, gates) identical to V10b.
"""

# V12 uses exactly the same trainer as V10/V10b — only the config and the
# (already-fixed) backbone differ.
from src.training.train_wavepinn_v10 import TrainerV10, parse_args
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Strip JSON comments (keys starting with "_")
    def strip_comments(d):
        if isinstance(d, dict):
            return {k: strip_comments(v) for k, v in d.items() if not k.startswith("_")}
        return d

    config = strip_comments(config)

    trainer = TrainerV10(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from: {args.resume}")

    trainer.train()


if __name__ == "__main__":
    main()
