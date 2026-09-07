"""
WavePINN V10 — Deployment-Optimised Temporal Super-Resolution.

V10 IS the WavePINNv6 architecture. No modifications.
Every failure from V7 through V9 was caused by the loss function (wave coherence,
ToA loss, Cs head gradient leakage), not by the backbone.

V10 training recipe differences from V9:
  - Single phase, 180 epochs, pure pixel loss only (no wave coherence, no Cs loss)
  - NO ScalarCsHead — Cs estimation is done by ToA analysis on output velocity
  - Deployment-biased PRF augmentation: step-10 weight doubled (0.15 → 0.30)
  - Stronger temporal loss weight (0.15 → 0.30) for better ToA-relevant signal
  - Cosine LR over full 180 epochs (no phase restarts)

Usage:
    from src.architectures.wavepinn_v10 import WavePINNv10, create_wavepinn_v10
    model = create_wavepinn_v10()

Forward signature (identical to V6):
    velocity = model(x, dt_seconds)
    # x         : (B, 2, T_sparse, H, W)
    # dt_seconds: float or (B,) tensor  [input frame gap in seconds]
    # velocity  : (B, 1, T_dense,  H, W) in [-1, 1]
"""
from __future__ import annotations

from typing import Dict, Optional

from src.architectures.wavepinn_v6 import (
    WavePINNv6,
    create_wavepinn_v6,
    count_parameters_v6,
)

# V10 = V6 architecture, different training recipe
WavePINNv10 = WavePINNv6


def create_wavepinn_v10(
    config: Optional[Dict] = None,
    **kwargs,
) -> WavePINNv10:
    """Create a WavePINNv10 (= WavePINNv6) model from config dict or kwargs."""
    return create_wavepinn_v6(config, **kwargs)


def count_parameters_v10(model: WavePINNv10) -> Dict[str, int]:
    """Count parameters in a WavePINNv10 model."""
    return count_parameters_v6(model)
