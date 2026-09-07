"""
Training script for WavePINN V14 — honest supervision, honest model selection.

Usage:
    python -m src.training.train_wavepinn_v14 --config configs/config_wavepinn_v14.json
    python -m src.training.train_wavepinn_v14 --config configs/config_wavepinn_v14b.json

Architecture is identical to V12 (wavepinn_v6.py backbone).  Three things change,
all motivated by measurements rather than intuition:

1. KNOT-MASKED SUPERVISION (loss.use_frame_mask).
   For 500 Hz samples the dense target is a linear interpolation of the same 5
   frames the model receives as input, so 44 of 49 target frames carried no
   information above 250 Hz and actively rewarded piecewise-linear output.
   Those frames are now excluded from the loss.

2. MODEL SELECTION ON GENUINE GROUND TRUTH (early_stopping.monitor).
   V12 early-stopped on val_500hz_l1, i.e. L1 against that same interpolation,
   so checkpoint selection preferred the most interpolation-like model.  V14
   selects on the 5 kHz val split, whose dense ground truth is real (native
   step 12, dt_out == dt_native, frame for frame).

3. TEMPORAL SPECTRAL LOSS (loss.spectral_weight, the v14b variant).
   Matched-ground-truth evaluation of v12 (scripts/evaluate_matched_gt.py)
   showed it reconstructs high-frequency PHASE well -- band-passed coherence
   0.73 over 250-1500 Hz, against ~0.0 for linear interpolation -- while
   carrying only ~46% of the true high-frequency ENERGY.  The deficit is one of
   magnitude, not structure, which is exactly what a log-magnitude spectral term
   penalises.

Data hygiene (data.strict_iq) makes a configured PRF step that cannot be
honoured drop the sample instead of silently reverting to the native step; that
silent fallback previously invalidated a whole run.
"""

import json
import logging
import sys

from src.training.train_wavepinn_v10 import TrainerV10, parse_args

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
