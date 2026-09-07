"""
2D Loupas autocorrelator pipeline for particle velocity estimation from IQ data.

Faithfully implements the MATLAB IQ2VEl.m pipeline:
    1. Determine ROI from non-zero IQ regions
    2. Axial displacement via 2D Loupas autocorrelator (Loupas et al., 1995)
    3. Smooth displacement with 2D median filter [12×6]
    4. Material velocity via finite difference (mm/s)
    5. (Optional) temporal interpolation before velocity computation

Physics References:
    - Loupas et al., IEEE TUFFC, 1995.
    - Maksuti et al., 2016 (elastography/ultrasound tracking).

Imported by scripts/iq_to_velocity.py (CLI) and
src/utils/deployment_scale.py (GT-free inference normalisation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LoupasConfig:
    """Configuration for the 2D Loupas velocity estimation pipeline."""

    # Physics parameters (from MATLAB IQ2VEl.m comments)
    speed_of_sound: float = 1540.0          # c [m/s]
    center_frequency: float = 7.6e6         # f_c [Hz] — transmit center frequency

    # 2D Loupas autocorrelator kernel sizes
    M: int = 6     # Range gate (axial samples) — ~1.5 wavelengths, low jitter
    N: int = 3     # Ensemble length (temporal frames)

    # Median filter kernel [axial, lateral] pixels
    # At f_c = 7.6 MHz → λ = 0.203 mm, kernel ≈ 3λ × 3λ ≈ 0.6 mm × 0.6 mm
    # Axial pixel = 0.0507 mm → 12 pixels ≈ 0.608 mm
    # Lateral pixel = 0.1013 mm → 6 pixels ≈ 0.608 mm
    medfilt_kernel_ax: int = 12
    medfilt_kernel_lat: int = 6

    # Temporal interpolation
    interp_factor: int = 10            # 10× slow-time interpolation
    do_interpolation: bool = False     # Off by default (native-rate for training)

    # Bandpass filter (MATLAB code overrides this with vel_filtL=vel; so off by default)
    do_bandpass: bool = False
    bandpass_low: float = 10.0         # Hz
    bandpass_high: float = 800.0       # Hz
    bandpass_order: int = 3

    # Sparse IQ selection for model input
    sparse_step: int = 12              # Take every Nth IQ frame (must equal upsampling_factor)

    # Data selection — PRFs to process
    target_frequencies: Tuple[str, ...] = ('1kHz', '5kHz', '10kHz', '20kHz')

    # Train/val split ratio
    train_ratio: float = 0.8

    # Random seed
    seed: int = 42


# ============================================================================
# Core Loupas Pipeline
# ============================================================================

def determine_roi(iq_data: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Determine the acquisition ROI from non-zero IQ samples at the first frame.

    Returns:
        (min_row, max_row, min_col, max_col) — inclusive bounds
    """
    nonzero = np.abs(iq_data[:, :, 0]) > 0
    rows, cols = np.nonzero(nonzero)
    if len(rows) == 0:
        logger.warning("No non-zero samples found; using full image as ROI")
        return (0, iq_data.shape[0] - 1, 0, iq_data.shape[1] - 1)
    return (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))


def loupas_2d_displacement(
    iq_data: np.ndarray,
    config: LoupasConfig,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute axial displacement using the 2D Loupas autocorrelator.

    Implements the vectorised version of the MATLAB IQ2VEl.m loop:
        disp(ax_i, :, nacq_i) = c/(4πf_c) · arctan(Σuu / Σud)
                                 / (1 + arctan(Σdu / Σdd) / (2π)) · 1e6  [μm]

    Args:
        iq_data: Complex IQ data of shape (H, W, T) — (axial, lateral, acquisitions)
        config: LoupasConfig with physics and kernel parameters

    Returns:
        disp: Displacement array (H_roi, W_roi, T) in microns [μm]
        roi_info: Dictionary with ROI coordinates
    """
    H, W, T = iq_data.shape
    c = config.speed_of_sound
    f_c = config.center_frequency
    M = config.M
    N = config.N
    eps = 1e-10

    # --- ROI detection ---
    min_row, max_row, min_col, max_col = determine_roi(iq_data)
    H_roi = max_row - min_row + 1
    W_roi = max_col - min_col + 1

    logger.debug(
        f"ROI: rows [{min_row}, {max_row}] ({H_roi}), "
        f"cols [{min_col}, {max_col}] ({W_roi})"
    )

    # Initialise output
    disp = np.zeros((H_roi, W_roi, T), dtype=np.float64)

    for nacq_idx in range(1, T):
        if nacq_idx + N - 1 >= T:
            continue

        ax_start = min_row + 1
        ax_end = min(max_row, H - M)
        if ax_start > ax_end:
            continue

        ax_starts = np.arange(ax_start, ax_end + 1)
        num_ax = len(ax_starts)

        row_lo = ax_starts[0]
        row_hi = ax_starts[-1] + M
        IQ_block = iq_data[row_lo:row_hi, min_col:max_col + 1, nacq_idx:nacq_idx + N]
        I_block = IQ_block.real
        Q_block = IQ_block.imag

        uu_sum = np.zeros((num_ax, W_roi), dtype=np.float64)
        ud_sum = np.zeros((num_ax, W_roi), dtype=np.float64)
        du_sum = np.zeros((num_ax, W_roi), dtype=np.float64)
        dd_sum = np.zeros((num_ax, W_roi), dtype=np.float64)

        offsets = ax_starts - row_lo

        for m in range(M - 1):
            for n in range(N - 1):
                I_mn   = I_block[offsets + m,     :, n]
                Q_mn   = Q_block[offsets + m,     :, n]
                I_mn1  = I_block[offsets + m,     :, n + 1]
                Q_mn1  = Q_block[offsets + m,     :, n + 1]
                I_m1n  = I_block[offsets + m + 1, :, n]
                Q_m1n  = Q_block[offsets + m + 1, :, n]

                uu_sum += Q_mn * I_mn1 - I_mn * Q_mn1
                ud_sum += I_mn * I_mn1 + Q_mn * Q_mn1
                du_sum += Q_mn * I_m1n - I_mn * Q_m1n
                dd_sum += I_mn * I_m1n + Q_mn * Q_m1n

        phi_temporal = np.arctan(uu_sum / (ud_sum + eps))
        phi_spatial  = np.arctan(du_sum / (dd_sum + eps))
        ax_disp_idx  = ax_starts - min_row
        disp[ax_disp_idx, :, nacq_idx] = (
            c / (4.0 * np.pi * f_c)
            * phi_temporal
            / (1.0 + phi_spatial / (2.0 * np.pi))
            * 1e6
        )

    roi_info = {
        'min_row': min_row, 'max_row': max_row,
        'min_col': min_col, 'max_col': max_col,
        'H_roi': H_roi, 'W_roi': W_roi,
    }
    return disp, roi_info


def smooth_displacement(
    disp: np.ndarray,
    kernel_ax: int = 12,
    kernel_lat: int = 6,
) -> np.ndarray:
    """
    Smooth displacement with a 2D median filter (step 2 of the pipeline).

    Args:
        disp: (H, W, T) displacement array [μm]
        kernel_ax: Axial kernel size (pixels)
        kernel_lat: Lateral kernel size (pixels)

    Returns:
        Smoothed displacement of same shape
    """
    medfiltdisp = np.zeros_like(disp)
    T = disp.shape[2]

    for t in range(1, T):
        frame = disp[:, :, t]
        if np.all(frame == 0):
            continue
        medfiltdisp[:, :, t] = median_filter(frame, size=(kernel_ax, kernel_lat))

    return medfiltdisp


def interpolate_displacement(
    disp: np.ndarray,
    interp_factor: int = 10,
) -> np.ndarray:
    """
    10× slow-time interpolation of displacement data (step 3 of the pipeline).

    Args:
        disp: (H, W, T) smoothed displacement [μm]
        interp_factor: Temporal upsampling factor

    Returns:
        Interpolated displacement (H, W, T_interp)
    """
    H, W, T = disp.shape
    T_interp = (T - 1) * interp_factor + 1
    interpdisp = np.zeros((H, W, T_interp), dtype=np.float64)

    x_coarse = np.arange(T, dtype=np.float64)
    x_fine = np.linspace(0, T - 1, T_interp)

    for ax_i in range(H):
        for lat_i in range(W):
            trace = disp[ax_i, lat_i, :].copy()
            trace[np.isnan(trace)] = 0.0
            f = interp1d(x_coarse, trace, kind='cubic', fill_value='extrapolate')
            interpdisp[ax_i, lat_i, :] = f(x_fine)

    return interpdisp


def compute_velocity(
    disp: np.ndarray,
    frame_time: float,
    interp_factor: int = 1,
) -> np.ndarray:
    """
    Compute material velocity via finite-difference differentiation (step 4).

    Units: displacement [μm] / time [ms] = velocity [mm/s]

    Args:
        disp: (H, W, T) displacement in microns [μm]
        frame_time: Detection pulse repetition period [seconds]
        interp_factor: Temporal interpolation factor

    Returns:
        Velocity (H, W, T-1) in [mm/s], float32
    """
    dt_ms = (frame_time / interp_factor) * 1e3
    vel = np.diff(disp, axis=2) / dt_ms
    return vel.astype(np.float32)


def bandpass_filter_velocity(
    vel: np.ndarray,
    fs: float,
    low: float = 10.0,
    high: float = 800.0,
    order: int = 3,
) -> np.ndarray:
    """
    Bandpass filter velocity between low and high Hz.

    Args:
        vel: (H, W, T) velocity in mm/s
        fs: Sampling frequency in Hz
        low, high: Cutoff frequencies in Hz
        order: Butterworth filter order

    Returns:
        Filtered velocity, same shape
    """
    nyquist = fs / 2.0
    if high >= nyquist:
        high = nyquist * 0.9
        logger.warning(f"Bandpass high cutoff clamped to {high:.1f} Hz (Nyquist={nyquist:.1f})")

    bh, ah = butter(order, low / nyquist, btype='high')
    bl, al = butter(order, high / nyquist, btype='low')

    vel_filtered = np.zeros_like(vel)
    H, W, T = vel.shape

    for i in range(H):
        for j in range(W):
            if T < 3 * max(len(bh), len(bl)):
                vel_filtered[i, j, :] = vel[i, j, :]
                continue
            try:
                temp = filtfilt(bh, ah, vel[i, j, :])
                vel_filtered[i, j, :] = filtfilt(bl, al, temp)
            except ValueError:
                vel_filtered[i, j, :] = vel[i, j, :]

    return vel_filtered
