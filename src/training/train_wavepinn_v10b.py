"""
Training script for WavePINN V10b — 500Hz-Inclusive Deployment Training.

Usage:
    python -m src.training.train_wavepinn_v10b --config configs/config_wavepinn_v10b.json
    python -m src.training.train_wavepinn_v10b --config configs/config_wavepinn_v10b.json --resume outputs/wavepinn_v10b/checkpoint_epoch_60.pt

Key differences from V10:
    - 500Hz data included in training_frequencies (was excluded in V10)
    - frequency_step_overrides: {"500hz": [1]} forces consecutive-frame input
      (step=1, dt=2ms) for all 500Hz samples — the exact deployment scenario
    - GT temporal resampling: when 500Hz GT (2ms/frame) is coarser than the
      dense output (0.167ms/frame), the GT is linearly resampled to T_dense frames
      within the same input time window
    - Early stopping monitors 500Hz val L1 (deployment metric)
    - Loss: temporal_weight raised to 0.5 (SWE cs depends on temporal structure)
    - Loss bug fix: GDL and Temporal computed on masked volumes (not full padded)
    - Collapse alarm: running max decays 2%/epoch (prevents over-sensitivity)
    - MSE logged separately in training output
"""

# V10b uses exactly the same trainer as V10 — only the config differs.
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
