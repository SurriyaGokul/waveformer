"""
GT-free velocity normalisation scale estimation for deployment.

During training, the velocity normalisation scale (vel_scale) was the 99th
percentile of |velocity| computed over the **full 50-frame IQ window**
(iq_velocity_dataset.py:267-270).  At deployment, no ground-truth velocity
is available, but the 500 Hz input IQ itself contains the same tissue motion
— so vel_scale can be recovered by running the Loupas estimator on the raw IQ.

Key invariant: the model's deployment output (shear wave speed cs via
normalised-xcorr ToA) is **provably invariant** to vel_scale because ToA
normalised-xcorr divides signal by its own magnitude before cross-correlation.
So even if the proxy vel_scale differs slightly from the training-time value,
cs is unaffected.  The proxy is still useful for converting model outputs to
physical units (mm/s) for other downstream uses.

Usage:
    from src.utils.deployment_scale import estimate_velocity_scale_from_iq

    # raw IQ window, shape (H, W, T), complex64, pre-clutter-filter (mean not subtracted)
    scale = estimate_velocity_scale_from_iq(iq_window, frame_time_s=1/500)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.utils.loupas import (
    LoupasConfig,
    loupas_2d_displacement,
    smooth_displacement,
    compute_velocity,
)

logger = logging.getLogger(__name__)


def estimate_velocity_scale_from_iq(
    iq_window: np.ndarray,
    frame_time_s: float,
    config: Optional[LoupasConfig] = None,
    calibration: float = 1.0,
) -> float:
    """Estimate the velocity normalisation scale from raw IQ without GT velocity.

    Mirrors the training-time normalisation in iq_velocity_dataset.py:267-270:
        vel_scale = max(percentile(|vel|, 99), 1.0)

    Prefer calling with the full available 500 Hz window (T ≥ 50) so the
    percentile is computed over the same temporal extent used during training.
    For inference with only T=5 IQ frames the estimate degrades gracefully
    (fewer finite-difference steps), and the `calibration` scalar can correct
    the systematic bias if measured offline.

    Parameters
    ----------
    iq_window : ndarray (H, W, T), complex64
        Raw pre-clutter-filter IQ data.  Clutter removal (temporal mean
        subtraction) is applied internally, matching the dataset pipeline.
    frame_time_s : float
        Inter-frame interval in seconds (= 1 / PRF).
    config : LoupasConfig or None
        Loupas parameters; defaults to standard LoupasConfig() if None.
    calibration : float
        Optional multiplicative correction factor (default 1.0 = no correction).

    Returns
    -------
    float
        Estimated vel_scale in mm/s, clamped to ≥ 1.0.
    """
    if config is None:
        config = LoupasConfig()

    # Clutter removal — matches dataset pipeline
    iq = iq_window.astype(np.complex64)
    iq = iq - iq.mean(axis=2, keepdims=True)

    try:
        disp, _ = loupas_2d_displacement(iq, config)
        disp_smooth = smooth_displacement(
            disp,
            kernel_ax=config.medfilt_kernel_ax,
            kernel_lat=config.medfilt_kernel_lat,
        )
        vel = compute_velocity(disp_smooth, frame_time_s, interp_factor=1)
        # vel: (H_roi, W_roi, T-1) in mm/s
        scale = float(np.percentile(np.abs(vel), 99))
        scale = max(scale * calibration, 1.0)
    except Exception as exc:
        logger.warning(f"Loupas proxy scale estimation failed: {exc}; returning 1.0")
        scale = 1.0

    return scale
