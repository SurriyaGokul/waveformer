#!/usr/bin/env python3
"""
Inference & Visualization for 500 Hz and 1 kHz CIRS Phantom Data.

Runs the trained ComplexToRealUNet model on all 500Hz and 1kHz IQ data to
produce temporally upsampled velocity predictions, then generates AVI videos
and summary PNG figures for each prediction.

Also computes Loupas ground-truth velocity from the raw IQ data so that
side-by-side comparison videos are generated for every file.  This allows
the user to see both the physics-based result (Loupas) and the ML prediction.

IMPORTANT — PRF mismatch warning:
    The model was trained on 10 kHz / 20 kHz data.  At those PRFs the five
    sparse IQ frames span only 2.4 – 4.8 ms, during which the shear wave
    travels ~5 – 10 mm and is still well within the ROI.
    At 500 Hz / 1 kHz the same five frames span 48 – 96 ms, far longer
    than the shear-wave transit time through the ROI (~16 mm at ~2 m/s ≈ 8 ms).
    Because the model uses simple linear temporal interpolation (no
    motion-aware bottleneck), it cannot extrapolate outward-propagating
    wavefronts from out-of-distribution inputs.  Lower-PRF predictions
    may show a decaying centre-concentrated pattern that appears to
    converge toward the push focus.

Pipeline (per file):
    1. Load raw IQ data (360, 380, 50) complex128
    2. Detect ROI (non-zero region at frame 0)
    3. Crop IQ to ROI (~116×160)
    4. Clutter filter: subtract temporal mean
    5. Sparse subsample: take every 12th frame → 5 sparse frames
    6. Normalise IQ: divide by iq_norm_scalar (4.81e7, from training)
    7. Separate real/imag → (2, 5, H, W) float32
    8. Pad spatial to (128, 160) — multiple of 16 for UNet
    9. Model forward pass → (1, 49, 128, 160) normalised velocity
   10. De-normalise: multiply by vel_norm_scalar (68.9145 mm/s)
   11. Un-pad: crop back to ROI spatial dims (~116×160)
   12. Compute Loupas ground-truth velocity from the full raw IQ
   13. Save velocity .npy and generate comparison video + summary figure

Usage:
    python scripts/inference_500hz_1khz.py
    python scripts/inference_500hz_1khz.py --checkpoint outputs/iq_to_velocity_v1/best_model.pt
    python scripts/inference_500hz_1khz.py --max_files 5  # quick test
    python scripts/inference_500hz_1khz.py --no_gt        # skip Loupas GT
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.velocity_unet import ComplexToRealUNet, create_velocity_model
from scripts.iq_to_velocity import (
    LoupasConfig,
    loupas_2d_displacement,
    smooth_displacement,
    compute_velocity,
)

# ============================================================================
# Constants (must match training pipeline exactly)
# ============================================================================

# Training PRFs — used to issue OOD warnings
TRAINING_PRFS = {10000.0, 20000.0}

# Pixel spacing from MATLAB IQ2VEl.m
AXIAL_PIXEL_MM = 0.0507
LATERAL_PIXEL_MM = 0.1013

# PRF mapping
FREQ_TO_PRF = {
    '500Hz': 500.0,
    '1kHz': 1000.0,
    '5kHz': 5000.0,
    '10kHz': 10000.0,
    '20kHz': 20000.0,
}

# Sparse sampling step (must match upsampling_factor)
SPARSE_STEP = 12

# Spatial padding multiple (UNet requirement)
SPATIAL_MULTIPLE = 16

# Default velocity color range for display (mm/s)
DEFAULT_VMIN = -3.0
DEFAULT_VMAX = 3.0


# ============================================================================
# Data Discovery
# ============================================================================

def discover_files(data_dir: Path, frequencies: Tuple[str, ...] = ('500Hz', '1kHz')) -> List[Dict]:
    """
    Discover all .npy IQ files for the given frequencies.

    Returns:
        List of dicts with: path, focus, inclusion, frequency, dataset, name
    """
    phantom_root = data_dir / 'CIRS_Phantom_Latest'
    if not phantom_root.exists():
        raise FileNotFoundError(f"Phantom data not found: {phantom_root}")

    files = []
    for dataset_dir in sorted(phantom_root.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith('Data set'):
            continue

        for focus_dir in sorted(dataset_dir.iterdir()):
            if not focus_dir.is_dir():
                continue

            for inclusion_dir in sorted(focus_dir.iterdir()):
                if not inclusion_dir.is_dir():
                    continue

                for freq_name in frequencies:
                    freq_dir = inclusion_dir / freq_name
                    if not freq_dir.exists():
                        continue

                    for npy_file in sorted(freq_dir.glob('*.npy')):
                        files.append({
                            'path': npy_file,
                            'dataset': dataset_dir.name,
                            'focus': focus_dir.name,
                            'inclusion': inclusion_dir.name,
                            'frequency': freq_name,
                            'name': npy_file.stem,
                        })

    logger.info(f"Discovered {len(files)} IQ files for frequencies {frequencies}")
    return files


# ============================================================================
# Preprocessing (mirrors training pipeline exactly)
# ============================================================================

def determine_roi(iq_data: np.ndarray) -> Dict:
    """Detect ROI from non-zero IQ at frame 0. Matches iq_to_velocity.py."""
    nonzero = np.abs(iq_data[:, :, 0]) > 0
    rows, cols = np.nonzero(nonzero)
    if len(rows) == 0:
        logger.warning("No non-zero samples found; using full image")
        min_row, max_row = 0, iq_data.shape[0] - 1
        min_col, max_col = 0, iq_data.shape[1] - 1
    else:
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())

    H_roi = max_row - min_row + 1
    W_roi = max_col - min_col + 1

    return {
        'min_row': min_row, 'max_row': max_row,
        'min_col': min_col, 'max_col': max_col,
        'H_roi': H_roi, 'W_roi': W_roi,
    }


def preprocess_iq_for_inference(
    iq_data: np.ndarray,
    iq_norm_scalar: float,
    sparse_step: int = SPARSE_STEP,
    spatial_multiple: int = SPATIAL_MULTIPLE,
) -> Tuple[torch.Tensor, Dict, Tuple[int, int]]:
    """
    Preprocess raw IQ data for model inference.

    Exactly mirrors the training dataset pipeline:
        1. Detect ROI → crop
        2. Clutter filter (subtract temporal mean)
        3. Sparse subsample
        4. Normalise by iq_norm_scalar
        5. Separate real/imag → (2, T_sparse, H, W)
        6. Pad spatial to nearest multiple of 16

    Args:
        iq_data: (H, W, T) complex128/complex64 raw IQ
        iq_norm_scalar: Normalisation scalar from training
        sparse_step: Temporal subsampling step
        spatial_multiple: Pad spatial dims to this multiple

    Returns:
        input_tensor: (1, 2, T_sparse, H_pad, W_pad) float32 on CPU
        roi_info: Dict with ROI coordinates
        pad_amounts: (pad_h, pad_w) number of rows/cols padded
    """
    H, W, T = iq_data.shape

    # 1. ROI detection
    roi_info = determine_roi(iq_data)
    min_row = roi_info['min_row']
    max_row = roi_info['max_row']
    min_col = roi_info['min_col']
    max_col = roi_info['max_col']

    logger.debug(
        f"  ROI: rows [{min_row}, {max_row}] ({roi_info['H_roi']}), "
        f"cols [{min_col}, {max_col}] ({roi_info['W_roi']})"
    )

    # 2. Crop to ROI
    iq_roi = iq_data[min_row:max_row + 1, min_col:max_col + 1, :]  # (H_roi, W_roi, T)

    # 3. Clutter filter: subtract temporal mean
    iq_filtered = iq_roi - iq_roi.mean(axis=2, keepdims=True)

    # 4. Sparse subsampling
    sparse_indices = np.arange(0, T, sparse_step)
    sparse_iq = iq_filtered[:, :, sparse_indices]  # (H_roi, W_roi, T_sparse)

    T_sparse = sparse_iq.shape[2]
    logger.debug(f"  Sparse IQ: {sparse_iq.shape}, T_sparse={T_sparse}")

    # 5. Normalise
    sparse_iq = sparse_iq / iq_norm_scalar

    # 6. Transpose to (T_sparse, H, W) then separate real/imag → (2, T, H, W)
    sparse_thw = np.transpose(sparse_iq, (2, 0, 1))  # (T_sparse, H_roi, W_roi)
    real = sparse_thw.real.astype(np.float32)
    imag = sparse_thw.imag.astype(np.float32)
    input_array = np.stack([real, imag], axis=0)  # (2, T_sparse, H_roi, W_roi)

    # 7. Pad spatial dims to multiple of 16
    _, _, h, w = input_array.shape
    pad_h = (spatial_multiple - h % spatial_multiple) % spatial_multiple
    pad_w = (spatial_multiple - w % spatial_multiple) % spatial_multiple

    if pad_h > 0 or pad_w > 0:
        input_array = np.pad(
            input_array,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0,
        )

    # 8. To tensor with batch dim
    input_tensor = torch.from_numpy(input_array).unsqueeze(0)  # (1, 2, T_sparse, H_pad, W_pad)

    return input_tensor, roi_info, (pad_h, pad_w)


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[ComplexToRealUNet, Dict]:
    """
    Load trained ComplexToRealUNet from checkpoint.

    Returns:
        model: Loaded model in eval mode
        data_info: Training data info dict (contains normalization scalars)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    full_config = ckpt.get('config', {})
    data_info = ckpt.get('data_info', {})

    logger.info(
        f"  Epoch: {ckpt.get('epoch')}, "
        f"Best val_loss: {ckpt.get('best_val_loss', 'N/A')}"
    )
    logger.info(
        f"  upsampling_factor: {full_config.get('upsampling_factor')}, "
        f"complex_to_real_mode: {full_config.get('complex_to_real_mode')}"
    )
    logger.info(
        f"  iq_norm_scalar: {data_info.get('iq_norm_scalar')}, "
        f"vel_norm_scalar: {data_info.get('vel_norm_scalar')}"
    )

    # Filter config to only model constructor args
    MODEL_KEYS = {
        'in_channels', 'base_channels', 'encoder_depths', 'upsampling_factor',
        'use_motion_interpolation', 'complex_to_real_mode', 'use_naive_bn',
        'flow_hidden_channels', 'flow_scale',
        'use_prf_conditioning', 'prf_hidden_dim',
    }
    model_config = {k: v for k, v in full_config.items() if k in MODEL_KEYS}

    # Create model with same config
    model = create_velocity_model(config=model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params:,} parameters")

    return model, data_info


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def run_inference(
    model: ComplexToRealUNet,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run model forward pass.

    Args:
        model: Trained model in eval mode
        input_tensor: (1, 2, T_sparse, H_pad, W_pad) float32
        device: torch device

    Returns:
        output: (T_dense, H_pad, W_pad) float32 in [-1, 1] (normalised velocity)
    """
    x = input_tensor.to(device)
    out = model(x)  # (1, 1, T_dense, H_pad, W_pad)
    return out[0, 0].cpu().numpy()  # (T_dense, H_pad, W_pad)


# ============================================================================
# Post-processing
# ============================================================================

def postprocess_velocity(
    vel_norm: np.ndarray,
    vel_norm_scalar: float,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    """
    De-normalise and un-pad model output.

    Args:
        vel_norm: (T_dense, H_pad, W_pad) in [-1, 1]
        vel_norm_scalar: Multiply to get mm/s
        pad_h: Rows to remove from bottom
        pad_w: Columns to remove from right

    Returns:
        velocity: (T_dense, H_roi, W_roi) in mm/s
    """
    # De-normalise
    vel_mms = vel_norm * vel_norm_scalar

    # Un-pad
    T, H, W = vel_mms.shape
    if pad_h > 0:
        vel_mms = vel_mms[:, :H - pad_h, :]
    if pad_w > 0:
        vel_mms = vel_mms[:, :, :W - pad_w]

    return vel_mms.astype(np.float32)


# ============================================================================
# Visualization (adapted from visualize_velocity_video.py)
# ============================================================================

def compute_axes(roi: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute physical axis vectors (mm) from ROI info."""
    H_roi = roi['H_roi']
    W_roi = roi['W_roi']
    min_row = roi['min_row']
    min_col = roi['min_col']

    axis_depth = (min_row + np.arange(H_roi)) * AXIAL_PIXEL_MM
    full_width_cols = 380
    center_col = full_width_cols / 2.0
    axis_width = ((min_col + np.arange(W_roi)) - center_col) * LATERAL_PIXEL_MM

    return axis_depth, axis_width


def make_velocity_video(
    vel: np.ndarray,
    output_path: Path,
    axis_depth: np.ndarray,
    axis_width: np.ndarray,
    taxis: np.ndarray,
    title_prefix: str = '',
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    step: int = 1,
    fps: int = 10,
    dpi: int = 100,
):
    """Create AVI video of particle velocity frames."""
    T, H, W = vel.shape
    frame_indices = list(range(0, T, step))

    if len(frame_indices) == 0:
        logger.warning(f"No frames to render for {output_path}")
        return

    # Ensure axes match data dims
    axis_depth = axis_depth[:H]
    axis_width = axis_width[:W]

    extent = [axis_width[0], axis_width[-1], axis_depth[-1], axis_depth[0]]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(
        vel[frame_indices[0]],
        aspect='equal',
        extent=extent,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        origin='upper',
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Particle velocity (mm/s)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lateral distance from center (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Axial depth (mm)', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    title_text = ax.set_title('', fontsize=13, fontweight='bold')
    fig.tight_layout()

    def update(frame_idx):
        i = frame_indices[frame_idx]
        im.set_data(vel[i])
        t_str = f"{taxis[i] * 1e3:.3f} ms" if i < len(taxis) else f"frame {i}"
        title = f"{title_prefix}Predicted velocity (mm/s) at t = {t_str}"
        title_text.set_text(title)
        return [im, title_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=1000 // fps, blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(output_path), writer=writer, dpi=dpi)
        logger.info(f"  Saved video: {output_path}")
    except Exception as e:
        logger.error(f"  FFmpeg failed: {e}. Trying GIF fallback.")
        gif_path = output_path.with_suffix('.gif')
        try:
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(gif_path), writer=writer, dpi=dpi)
            logger.info(f"  Saved GIF: {gif_path}")
        except Exception as e2:
            logger.error(f"  Could not save animation: {e2}")

    plt.close(fig)


def make_summary_figure(
    vel: np.ndarray,
    output_path: Path,
    axis_depth: np.ndarray,
    axis_width: np.ndarray,
    taxis: np.ndarray,
    title_prefix: str = '',
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    n_snapshots: int = 6,
):
    """Save a 2×3 panel summary figure at 6 time points."""
    T, H, W = vel.shape

    # Ensure axes match data dims
    axis_depth = axis_depth[:H]
    axis_width = axis_width[:W]

    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    extent = [axis_width[0], axis_width[-1], axis_depth[-1], axis_depth[0]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax, idx in zip(axes_flat, indices):
        im = ax.imshow(
            vel[idx], aspect='equal', extent=extent,
            cmap='jet', vmin=vmin, vmax=vmax,
            interpolation='nearest', origin='upper',
        )
        t_str = f"{taxis[idx] * 1e3:.3f} ms" if idx < len(taxis) else f"frame {idx}"
        ax.set_title(f"t = {t_str}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Lateral (mm)', fontsize=9)
        ax.set_ylabel('Depth (mm)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{title_prefix}Predicted Velocity Snapshots (model upsampled)",
        fontsize=14, fontweight='bold',
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Velocity (mm/s)')
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved summary: {output_path}")


# ============================================================================
# Loupas Ground-Truth Computation
# ============================================================================

def compute_loupas_ground_truth(
    iq_data: np.ndarray,
    prf: float,
    roi_info: Dict,
) -> np.ndarray:
    """
    Compute Loupas ground-truth velocity from raw IQ data.

    Uses the same 2D Loupas autocorrelator as the training pipeline (native rate,
    no temporal interpolation) so the result has the same 49 frames as the model
    output and can be compared side-by-side.

    Args:
        iq_data: Raw complex IQ data, shape (H, W, T)
        prf: Pulse repetition frequency in Hz
        roi_info: Dict with ROI coordinates (min_row, max_row, min_col, max_col, ...)

    Returns:
        gt_vel: Ground-truth velocity in mm/s, shape (T_vel, H_roi, W_roi)
    """
    config = LoupasConfig()
    frame_time = 1.0 / prf

    # Step 1: Loupas 2D displacement
    disp, _ = loupas_2d_displacement(iq_data, config)
    logger.debug(
        f"  Loupas disp: shape={disp.shape}, "
        f"range=[{disp.min():.2f}, {disp.max():.2f}] um"
    )

    # Step 2: Median-filter smoothing
    medfiltdisp = smooth_displacement(
        disp, config.medfilt_kernel_ax, config.medfilt_kernel_lat
    )

    # Step 3: Velocity via finite difference (no temporal interpolation)
    vel = compute_velocity(medfiltdisp, frame_time, interp_factor=1)

    # Transpose to (T_vel, H, W) — same convention as model output
    gt_vel = np.transpose(vel, (2, 0, 1)).astype(np.float32)

    return gt_vel


# ============================================================================
# Side-by-Side Comparison Video (GT left, Prediction right)
# ============================================================================

def make_comparison_video(
    gt_vel: np.ndarray,
    pred_vel: np.ndarray,
    output_path: Path,
    axis_depth: np.ndarray,
    axis_width: np.ndarray,
    taxis: np.ndarray,
    title_prefix: str = '',
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    step: int = 1,
    fps: int = 10,
    dpi: int = 100,
):
    """
    Create side-by-side AVI video: Loupas GT (left) vs Model Prediction (right).

    Both arrays must share the same spatial dimensions.  If temporal lengths
    differ, the shorter one is zero-padded.
    """
    T_gt, H_gt, W_gt = gt_vel.shape
    T_pred, H_pred, W_pred = pred_vel.shape

    # Align spatial dims to the smaller of the two
    H = min(H_gt, H_pred)
    W = min(W_gt, W_pred)
    gt_crop = gt_vel[:, :H, :W]
    pred_crop = pred_vel[:, :H, :W]

    T = min(T_gt, T_pred)
    gt_crop = gt_crop[:T]
    pred_crop = pred_crop[:T]

    frame_indices = list(range(0, T, step))
    if len(frame_indices) == 0:
        logger.warning(f"No frames to render for {output_path}")
        return

    axis_depth = axis_depth[:H]
    axis_width = axis_width[:W]
    extent = [axis_width[0], axis_width[-1], axis_depth[-1], axis_depth[0]]

    # Determine a shared color range from the GT
    shared_vmin = vmin
    shared_vmax = vmax

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(18, 5))

    im_gt = ax_gt.imshow(
        gt_crop[frame_indices[0]], aspect='equal', extent=extent,
        cmap='jet', vmin=shared_vmin, vmax=shared_vmax,
        interpolation='nearest', origin='upper',
    )
    ax_gt.set_xlabel('Lateral (mm)', fontsize=12, fontweight='bold')
    ax_gt.set_ylabel('Depth (mm)', fontsize=12, fontweight='bold')
    ax_gt.tick_params(labelsize=10)
    ax_gt.grid(True, alpha=0.3)
    gt_title = ax_gt.set_title('', fontsize=12, fontweight='bold')

    im_pred = ax_pred.imshow(
        pred_crop[frame_indices[0]], aspect='equal', extent=extent,
        cmap='jet', vmin=shared_vmin, vmax=shared_vmax,
        interpolation='nearest', origin='upper',
    )
    ax_pred.set_xlabel('Lateral (mm)', fontsize=12, fontweight='bold')
    ax_pred.tick_params(labelsize=10)
    ax_pred.grid(True, alpha=0.3)
    pred_title = ax_pred.set_title('', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im_pred, ax=[ax_gt, ax_pred], pad=0.02, shrink=0.9)
    cbar.set_label('Velocity (mm/s)', fontsize=12, fontweight='bold')

    fig.suptitle(f'{title_prefix}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.95, 0.93])

    def update(frame_idx):
        i = frame_indices[frame_idx]
        im_gt.set_data(gt_crop[i])
        im_pred.set_data(pred_crop[i])
        t_str = f"{taxis[i] * 1e3:.3f} ms" if i < len(taxis) else f"frame {i}"
        gt_title.set_text(f"Loupas Ground Truth  |  t = {t_str}")
        pred_title.set_text(f"Model Prediction  |  t = {t_str}")
        return [im_gt, im_pred, gt_title, pred_title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=1000 // fps, blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(str(output_path), writer=writer, dpi=dpi)
        logger.info(f"  Saved comparison video: {output_path}")
    except Exception as e:
        logger.error(f"  FFmpeg failed: {e}. Trying GIF fallback.")
        gif_path = output_path.with_suffix('.gif')
        try:
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(gif_path), writer=writer, dpi=dpi)
            logger.info(f"  Saved comparison GIF: {gif_path}")
        except Exception as e2:
            logger.error(f"  Could not save comparison animation: {e2}")

    plt.close(fig)


def make_comparison_summary(
    gt_vel: np.ndarray,
    pred_vel: np.ndarray,
    output_path: Path,
    axis_depth: np.ndarray,
    axis_width: np.ndarray,
    taxis: np.ndarray,
    title_prefix: str = '',
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    n_snapshots: int = 4,
):
    """Save a 2×N panel figure: top row = GT, bottom row = Prediction."""
    T_gt, H_gt, W_gt = gt_vel.shape
    T_pred, H_pred, W_pred = pred_vel.shape

    H = min(H_gt, H_pred)
    W = min(W_gt, W_pred)
    T = min(T_gt, T_pred)
    gt_crop = gt_vel[:T, :H, :W]
    pred_crop = pred_vel[:T, :H, :W]

    axis_depth = axis_depth[:H]
    axis_width = axis_width[:W]
    extent = [axis_width[0], axis_width[-1], axis_depth[-1], axis_depth[0]]

    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, n_snapshots, figsize=(5 * n_snapshots, 10))

    for col, idx in enumerate(indices):
        t_str = f"{taxis[idx] * 1e3:.3f} ms" if idx < len(taxis) else f"frame {idx}"

        # GT row
        im_gt = axes[0, col].imshow(
            gt_crop[idx], aspect='equal', extent=extent,
            cmap='jet', vmin=vmin, vmax=vmax,
            interpolation='nearest', origin='upper',
        )
        axes[0, col].set_title(f"GT  t={t_str}", fontsize=10, fontweight='bold')
        axes[0, col].tick_params(labelsize=7)
        axes[0, col].grid(True, alpha=0.2)
        if col == 0:
            axes[0, col].set_ylabel('Depth (mm)', fontsize=9)

        # Pred row
        im_pred = axes[1, col].imshow(
            pred_crop[idx], aspect='equal', extent=extent,
            cmap='jet', vmin=vmin, vmax=vmax,
            interpolation='nearest', origin='upper',
        )
        axes[1, col].set_title(f"Pred  t={t_str}", fontsize=10, fontweight='bold')
        axes[1, col].set_xlabel('Lateral (mm)', fontsize=9)
        axes[1, col].tick_params(labelsize=7)
        axes[1, col].grid(True, alpha=0.2)
        if col == 0:
            axes[1, col].set_ylabel('Depth (mm)', fontsize=9)

    fig.suptitle(
        f"{title_prefix}Loupas GT (top) vs Model Prediction (bottom)",
        fontsize=14, fontweight='bold',
    )
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im_pred, cax=cbar_ax, label='Velocity (mm/s)')
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved comparison summary: {output_path}")


# ============================================================================
# Per-File Processing
# ============================================================================

def process_one_file(
    file_info: Dict,
    model: ComplexToRealUNet,
    device: torch.device,
    iq_norm_scalar: float,
    vel_norm_scalar: float,
    output_dir: Path,
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    fps: int = 10,
    save_npy: bool = True,
    compute_gt: bool = True,
) -> Optional[Dict]:
    """
    Full pipeline for a single IQ file: preprocess → infer → postprocess → visualize.

    Also computes Loupas ground-truth and generates side-by-side comparison
    videos when compute_gt is True.

    Returns:
        Metadata dict for this file, or None on failure.
    """
    fpath = file_info['path']
    freq_name = file_info['frequency']
    focus = file_info['focus']
    inclusion = file_info['inclusion']
    file_stem = file_info['name']

    # Descriptive name for output files
    out_name = f"{focus}_{inclusion}_{freq_name}_{file_stem}"

    # Warn about OOD PRFs
    prf = FREQ_TO_PRF[freq_name]
    if prf not in TRAINING_PRFS:
        logger.warning(
            f"  ⚠ PRF {prf:.0f} Hz ({freq_name}) is OUT OF DISTRIBUTION. "
            f"Model was trained on {sorted(TRAINING_PRFS)} Hz.  "
            f"Predictions may be unreliable — wave propagation direction "
            f"and magnitude may be incorrect."
        )

    try:
        # --- Load raw IQ ---
        iq_data = np.load(fpath)
        if not np.iscomplexobj(iq_data):
            raise ValueError(f"Expected complex IQ data, got {iq_data.dtype}")
        H_orig, W_orig, T_orig = iq_data.shape
        logger.info(f"  Loaded: shape={iq_data.shape}, dtype={iq_data.dtype}")

        # --- Preprocess ---
        input_tensor, roi_info, (pad_h, pad_w) = preprocess_iq_for_inference(
            iq_data, iq_norm_scalar,
        )
        logger.info(
            f"  Preprocessed: input={list(input_tensor.shape)}, "
            f"ROI=({roi_info['H_roi']}×{roi_info['W_roi']}), "
            f"pad=({pad_h},{pad_w})"
        )

        # Sanity checks
        assert input_tensor.shape[1] == 2, f"Expected 2 channels, got {input_tensor.shape[1]}"
        T_sparse = input_tensor.shape[2]
        assert T_sparse == 5, f"Expected T_sparse=5, got {T_sparse}"

        # --- Inference ---
        vel_norm = run_inference(model, input_tensor, device)
        T_dense = vel_norm.shape[0]
        expected_T_dense = 1 + (T_sparse - 1) * SPARSE_STEP
        assert T_dense == expected_T_dense, (
            f"T_dense mismatch: got {T_dense}, expected {expected_T_dense}"
        )
        logger.info(
            f"  Model output: shape={vel_norm.shape}, "
            f"range=[{vel_norm.min():.4f}, {vel_norm.max():.4f}]"
        )

        # --- Post-process ---
        vel_mms = postprocess_velocity(vel_norm, vel_norm_scalar, pad_h, pad_w)
        logger.info(
            f"  Velocity (mm/s): shape={vel_mms.shape}, "
            f"range=[{vel_mms.min():.2f}, {vel_mms.max():.2f}], "
            f"std={vel_mms.std():.4f}"
        )

        # --- Compute physical axes ---
        axis_depth, axis_width = compute_axes(roi_info)

        # Time axis: model outputs T_dense frames spanning the sparse frame times
        prf = FREQ_TO_PRF[freq_name]
        frame_time_s = 1.0 / prf
        # The sparse frames are at indices [0, 12, 24, 36, 48] of the original T=50.
        # Model output has T_dense=49 frames covering from sparse[0] to sparse[4].
        # Time between model frames = sparse_step * frame_time_s / (T_dense / (T_sparse - 1))
        # Simpler: each model output frame = original_frame_time * sparse_step / (factors)
        # The spans: sparse indices cover (T_sparse-1)*sparse_step = 48 original frames
        # Model produces T_dense = 49 frames for those 48 inter-frame gaps → dt = 48*frame_time / 48 = frame_time
        # Actually T_dense = 1 + (T_sparse-1)*upsampling_factor = 49
        # Time span = (T_sparse-1) * sparse_step * frame_time_s = 4 * 12 * frame_time_s
        time_span_s = (T_sparse - 1) * SPARSE_STEP * frame_time_s
        taxis = np.linspace(0, time_span_s, T_dense)

        title_prefix = f"{focus}, {inclusion}, {freq_name}\n"

        # --- Create output sub-directory per frequency ---
        freq_dir = output_dir / freq_name
        freq_dir.mkdir(parents=True, exist_ok=True)

        # --- Save velocity .npy ---
        if save_npy:
            npy_path = freq_dir / f"predicted_velocity_{out_name}.npy"
            np.save(npy_path, vel_mms)
            logger.info(f"  Saved .npy: {npy_path}")

        # --- Generate video ---
        video_path = freq_dir / f"Velocity_movie_{out_name}.avi"
        make_velocity_video(
            vel_mms, video_path, axis_depth, axis_width, taxis,
            title_prefix=title_prefix, vmin=vmin, vmax=vmax, fps=fps,
        )

        # --- Generate summary figure ---
        summary_path = freq_dir / f"Velocity_summary_{out_name}.png"
        make_summary_figure(
            vel_mms, summary_path, axis_depth, axis_width, taxis,
            title_prefix=title_prefix, vmin=vmin, vmax=vmax,
        )

        # --- Compute Loupas ground-truth and generate comparisons ---
        gt_vel = None
        if compute_gt:
            try:
                logger.info("  Computing Loupas ground-truth velocity...")
                gt_vel = compute_loupas_ground_truth(iq_data, prf, roi_info)
                logger.info(
                    f"  GT velocity: shape={gt_vel.shape}, "
                    f"range=[{gt_vel.min():.2f}, {gt_vel.max():.2f}] mm/s, "
                    f"std={gt_vel.std():.4f}"
                )

                # Save GT .npy
                if save_npy:
                    gt_npy_path = freq_dir / f"gt_loupas_velocity_{out_name}.npy"
                    np.save(gt_npy_path, gt_vel)
                    logger.info(f"  Saved GT .npy: {gt_npy_path}")

                # Generate GT-only video
                gt_video_path = freq_dir / f"GT_Loupas_movie_{out_name}.avi"
                make_velocity_video(
                    gt_vel, gt_video_path, axis_depth, axis_width, taxis,
                    title_prefix=f"{title_prefix}[Loupas GT] ",
                    vmin=vmin, vmax=vmax, fps=fps,
                )

                # Generate side-by-side comparison video
                comp_video_path = freq_dir / f"Comparison_movie_{out_name}.avi"
                make_comparison_video(
                    gt_vel, vel_mms, comp_video_path,
                    axis_depth, axis_width, taxis,
                    title_prefix=title_prefix,
                    vmin=vmin, vmax=vmax, fps=fps,
                )

                # Generate comparison summary figure
                comp_summary_path = freq_dir / f"Comparison_summary_{out_name}.png"
                make_comparison_summary(
                    gt_vel, vel_mms, comp_summary_path,
                    axis_depth, axis_width, taxis,
                    title_prefix=title_prefix,
                    vmin=vmin, vmax=vmax,
                )

            except Exception as gt_err:
                logger.warning(f"  GT computation failed: {gt_err}")
                gt_vel = None

        result_dict = {
            'name': out_name,
            'frequency': freq_name,
            'focus': focus,
            'inclusion': inclusion,
            'prf_hz': prf,
            'prf_in_distribution': prf in TRAINING_PRFS,
            'roi': roi_info,
            'input_shape': list(iq_data.shape),
            'output_shape': list(vel_mms.shape),
            'velocity_range_mms': [float(vel_mms.min()), float(vel_mms.max())],
            'velocity_std_mms': float(vel_mms.std()),
            'T_sparse': T_sparse,
            'T_dense': T_dense,
        }

        if gt_vel is not None:
            result_dict['gt_velocity_range_mms'] = [float(gt_vel.min()), float(gt_vel.max())]
            result_dict['gt_velocity_std_mms'] = float(gt_vel.std())
            # Frame-level correlation between prediction and GT
            T_cmp = min(vel_mms.shape[0], gt_vel.shape[0])
            H_cmp = min(vel_mms.shape[1], gt_vel.shape[1])
            W_cmp = min(vel_mms.shape[2], gt_vel.shape[2])
            p_flat = vel_mms[:T_cmp, :H_cmp, :W_cmp].flatten()
            g_flat = gt_vel[:T_cmp, :H_cmp, :W_cmp].flatten()
            if p_flat.std() > 0 and g_flat.std() > 0:
                corr = float(np.corrcoef(p_flat, g_flat)[0, 1])
            else:
                corr = 0.0
            result_dict['correlation_with_gt'] = corr
            logger.info(f"  Correlation (pred vs GT): {corr:.4f}")

        return result_dict

    except Exception as e:
        logger.error(f"  FAILED: {e}", exc_info=True)
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run trained model on 500Hz/1kHz CIRS Phantom data and generate videos"
    )
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Root data dir containing CIRS_Phantom_Latest/',
    )
    parser.add_argument(
        '--checkpoint', type=str, default='outputs/iq_to_velocity_v1/best_model.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs/inference_500hz_1khz',
        help='Output directory for predictions and videos',
    )
    parser.add_argument(
        '--frequencies', nargs='+', default=['500Hz', '1kHz'],
        help='Frequency folders to process (default: 500Hz 1kHz)',
    )
    parser.add_argument(
        '--vmin', type=float, default=DEFAULT_VMIN,
        help=f'Min velocity for colorbar (default: {DEFAULT_VMIN})',
    )
    parser.add_argument(
        '--vmax', type=float, default=DEFAULT_VMAX,
        help=f'Max velocity for colorbar (default: {DEFAULT_VMAX})',
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Video frame rate (default: 10)',
    )
    parser.add_argument(
        '--max_files', type=int, default=None,
        help='Max files to process (for testing)',
    )
    parser.add_argument(
        '--no_npy', action='store_true',
        help='Skip saving velocity .npy files',
    )
    parser.add_argument(
        '--no_gt', action='store_true',
        help='Skip Loupas ground-truth computation and comparison videos',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (default: auto-detect cuda/cpu)',
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, data_info = load_model(ckpt_path, device)

    iq_norm_scalar = data_info['iq_norm_scalar']
    vel_norm_scalar = data_info['vel_norm_scalar']
    logger.info(f"Normalization: IQ={iq_norm_scalar:.2e}, Vel={vel_norm_scalar:.4f} mm/s")

    # ---- PRF distribution warning ----
    for freq_name in args.frequencies:
        prf = FREQ_TO_PRF.get(freq_name)
        if prf and prf not in TRAINING_PRFS:
            sparse_span_ms = (5 - 1) * SPARSE_STEP * (1.0 / prf) * 1e3
            logger.warning(
                f"\n{'!'*60}\n"
                f"  PRF WARNING: {freq_name} ({prf:.0f} Hz) is OUT OF DISTRIBUTION.\n"
                f"  Model was trained on {sorted(TRAINING_PRFS)} Hz.\n"
                f"  At {freq_name}, the 5 sparse IQ frames span {sparse_span_ms:.1f} ms.\n"
                f"  Shear waves likely traverse the entire ROI in <10 ms,\n"
                f"  so the model cannot resolve outward-propagating wavefronts.\n"
                f"  Predictions may show artefactual converging patterns.\n"
                f"  Loupas ground-truth will be computed for comparison.\n"
                f"{'!'*60}"
            )

    # Discover files
    data_dir = Path(args.data_dir)
    files = discover_files(data_dir, tuple(args.frequencies))

    if not files:
        logger.error("No IQ files found!")
        sys.exit(1)

    if args.max_files:
        files = files[:args.max_files]

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Processing {len(files)} files → {output_dir}")
    logger.info(f"{'=' * 70}")

    results = []
    n_success = 0
    n_fail = 0

    for i, file_info in enumerate(files):
        logger.info(
            f"\n[{i + 1}/{len(files)}] {file_info['focus']}/{file_info['inclusion']}/"
            f"{file_info['frequency']}/{file_info['name']}"
        )
        result = process_one_file(
            file_info, model, device,
            iq_norm_scalar, vel_norm_scalar, output_dir,
            vmin=args.vmin, vmax=args.vmax, fps=args.fps,
            save_npy=not args.no_npy,
            compute_gt=not args.no_gt,
        )
        if result is not None:
            results.append(result)
            n_success += 1
        else:
            n_fail += 1

    # Save inference metadata
    inference_meta = {
        'checkpoint': str(ckpt_path),
        'iq_norm_scalar': iq_norm_scalar,
        'vel_norm_scalar': vel_norm_scalar,
        'sparse_step': SPARSE_STEP,
        'training_prfs_hz': sorted(TRAINING_PRFS),
        'frequencies': args.frequencies,
        'n_files_processed': n_success,
        'n_files_failed': n_fail,
        'loupas_gt_computed': not args.no_gt,
        'results': results,
    }
    meta_path = output_dir / 'inference_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(inference_meta, f, indent=2)
    logger.info(f"\nMetadata saved: {meta_path}")

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("INFERENCE COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Success: {n_success}/{len(files)}")
    logger.info(f"  Failed:  {n_fail}/{len(files)}")
    logger.info(f"  Output:  {output_dir}")

    # Per-frequency summary
    for freq in args.frequencies:
        freq_results = [r for r in results if r['frequency'] == freq]
        if freq_results:
            vel_ranges = [r['velocity_range_mms'] for r in freq_results]
            all_min = min(v[0] for v in vel_ranges)
            all_max = max(v[1] for v in vel_ranges)
            all_std = np.mean([r['velocity_std_mms'] for r in freq_results])
            logger.info(
                f"  {freq}: {len(freq_results)} files, "
                f"vel range=[{all_min:.2f}, {all_max:.2f}] mm/s, "
                f"mean std={all_std:.4f}"
            )


if __name__ == '__main__':
    main()
