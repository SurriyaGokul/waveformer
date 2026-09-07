"""
Training script for WavePINN V13 — v12 + PVA phantom training data.

Usage:
    python -m src.training.train_wavepinn_v13 --config configs/config_wavepinn_v13.json
    python -m src.training.train_wavepinn_v13 --config configs/config_wavepinn_v13.json --resume outputs/wavepinn_v13/checkpoint_epoch_60.pt

Key differences from V12:
    - Adds data/pva_velocity's "train" split (40 samples, real PVA phantom
      acquisitions, previously unused) to training via data.extra_data_dirs.
      PVA's own "val" split (10 samples) is deliberately excluded from
      training, so it stays held-out for a fair comparison against v10b/v12
      (which never saw any PVA data) on the exact same PVA val samples.
    - PVA has a different ROI crop size (72x200) than fullscale_velocity
      (116x160), so it's trained via InterleavedBatchLoader
      (src/training/train_wavepinn_v10.py) rather than being concatenated
      into one DataLoader — each batch is drawn from one dataset only, so
      collate_fn_v3's torch.stack never has to reconcile different spatial
      shapes across datasets.
    - Recipe (loss weights, epochs, LR, gates, backbone) identical to v12.
"""

# V13 uses exactly the same trainer as V10/V10b/V12 — only the config
# (extra_data_dirs) and the trainer's multi-dataset support differ.
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
