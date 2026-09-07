#!/usr/bin/env python3
"""
Visualize Training Data — Exactly What the Model Sees.

Shows the model input and training target with proper dynamic colour scaling
so that the shear-wave fronts are clearly visible.  The raw Kasai velocity has
amplitudes of ±0.1–0.5 inside the tissue ROI but is zero elsewhere; plotting
with the full theoretical range [-1, 1] makes the waves invisible.  This script
uses the 99th-percentile of |velocity| **within the signal region** as vmax,
revealing the full wave structure.

Rows produced for each sample:
  1. Model Input — sparse IQ magnitude (real²+imag²), normalised by global_max
  2. Raw velocity target (dynamic scale, 99th pctl of signal region)
  3. Clean velocity target (normalised-convolution smoothed, same colour scale)
  4. Magnitude mask [0, 1]
  5. Difference (raw − clean) — the noise that was removed by smoothing

Additionally produces a noise-analysis figure with line profiles and histograms.

Usage:
    python scripts/visualize_smooth_targets.py
    python scripts/visualize_smooth_targets.py --num_samples 5 --split val
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.velocity_dataset import PreprocessedVelocityDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_and_clean(preprocessed_dir: Path, sample_idx: int, split: str):
    """Load the same sample with both raw and clean targets + the model input."""
    split_dir = preprocessed_dir / split
    metadata_path = preprocessed_dir / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    global_max = metadata['global_stats']['global_max']

    ds_raw = PreprocessedVelocityDataset(
        preprocessed_dir=split_dir, global_max=global_max,
        return_mask=True, use_clean_targets=False,
    )
    ds_clean = PreprocessedVelocityDataset(
        preprocessed_dir=split_dir, global_max=global_max,
        return_mask=True, use_clean_targets=True,
    )

    input_t, target_raw, mask_t = ds_raw[sample_idx]
    _, target_clean, _ = ds_clean[sample_idx]
    info = ds_raw.get_sample_info(sample_idx)

    return input_t, target_raw, target_clean, mask_t, info, metadata


def _compute_dynamic_vmax(velocity_np, mask_np, percentile=99):
    """Compute a colour-range limit from the signal region (mask > 0.1)."""
    signal_vals = np.abs(velocity_np[mask_np > 0.1])
    if len(signal_vals) == 0:
        return 0.3  # fallback
    return float(np.percentile(signal_vals, percentile))


def visualize_comparison(
    input_tensor, target_raw, target_clean, mask_tensor,
    sample_info, metadata, output_path,
):
    """
    Five-row comparison figure with dynamic colour scaling.

    Rows:
      1. Input IQ magnitude
      2. Raw velocity target  (dynamic vmax from signal region)
      3. Clean velocity target (same colour scale)
      4. Magnitude mask
      5. Difference raw − clean  (shows removed speckle noise)
    """
    inp = input_tensor.numpy()        # (2, T_sparse, H, W)
    raw = target_raw.numpy()          # (1, T_dense, H, W)
    clean = target_clean.numpy()      # (1, T_dense, H, W)
    mask = mask_tensor.numpy()        # (1, T_dense, H, W)
    diff = raw - clean

    T_sparse, T_dense = inp.shape[1], raw.shape[1]
    H, W = raw.shape[2], raw.shape[3]

    # Dynamic colour limits from the signal region across ALL frames
    vmax_vel = _compute_dynamic_vmax(raw[0], mask[0], percentile=99)
    vmax_diff = _compute_dynamic_vmax(diff[0], mask[0], percentile=99) * 0.8

    # Pick 8 evenly-spaced dense time indices
    t_indices = np.linspace(0, T_dense - 1, 8, dtype=int)

    vel_norm = Normalize(vmin=-vmax_vel, vmax=vmax_vel)
    diff_norm = Normalize(vmin=-vmax_diff, vmax=vmax_diff)
    mask_norm = Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(5, 8, figsize=(28, 17))

    row_labels = [
        'Model Input\n(IQ magnitude)',
        f'Raw Target\n(vel, vmax={vmax_vel:.3f})',
        f'Clean Target\n(smoothed, same scale)',
        'Magnitude Mask\n[0, 1]',
        f'Noise Removed\n(raw − clean)',
    ]

    for col, t in enumerate(t_indices):
        sparse_idx = min(t // 20, T_sparse - 1)

        # Row 0 — IQ magnitude
        ax = axes[0, col]
        iq_mag = np.sqrt(inp[0, sparse_idx]**2 + inp[1, sparse_idx]**2)
        im0 = ax.imshow(iq_mag.T, cmap='magma', origin='lower', aspect='auto')
        ax.set_title(f'sparse t={sparse_idx}', fontsize=8)

        # Row 1 — raw velocity (dynamic scale)
        ax = axes[1, col]
        im1 = ax.imshow(raw[0, t].T, cmap='RdBu_r', norm=vel_norm,
                        origin='lower', aspect='auto')
        ax.set_title(f't={t}', fontsize=8)

        # Row 2 — clean velocity (same dynamic scale)
        ax = axes[2, col]
        im2 = ax.imshow(clean[0, t].T, cmap='RdBu_r', norm=vel_norm,
                        origin='lower', aspect='auto')

        # Row 3 — mask
        ax = axes[3, col]
        im3 = ax.imshow(mask[0, t].T, cmap='viridis', norm=mask_norm,
                        origin='lower', aspect='auto')

        # Row 4 — difference (noise removed)
        ax = axes[4, col]
        im4 = ax.imshow(diff[0, t].T, cmap='coolwarm', norm=diff_norm,
                        origin='lower', aspect='auto')

    # Axis cleanup + row labels
    for row in range(5):
        for col in range(8):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10, rotation=0,
                                labelpad=120, va='center')

    # Colorbars
    for im, row_axes, label in [
        (im0, axes[0, :], 'Magnitude'),
        (im1, axes[1, :], 'Velocity'),
        (im2, axes[2, :], 'Velocity'),
        (im3, axes[3, :], 'Mask'),
        (im4, axes[4, :], 'Diff'),
    ]:
        fig.colorbar(im, ax=row_axes.tolist(), shrink=0.75, pad=0.01, label=label)

    # Stats
    raw_sig_std = raw[0][mask[0] > 0.1].std() if (mask[0] > 0.1).any() else 0
    clean_sig_std = clean[0][mask[0] > 0.1].std() if (mask[0] > 0.1).any() else 0
    mask_cov = (mask > 0.1).mean()

    title = (
        f"Sample: {sample_info['base_name']}  |  "
        f"Input (2,{T_sparse},{H},{W}), Target (1,{T_dense},{H},{W})\n"
        f"Dynamic vmax={vmax_vel:.4f} (99th pctl of signal region)  |  "
        f"Signal std — raw: {raw_sig_std:.4f}, clean: {clean_sig_std:.4f}  |  "
        f"ROI coverage: {mask_cov:.2%}"
    )
    fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout(rect=[0.1, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved → {output_path}")


def visualize_noise_profile(
    input_tensor, target_raw, target_clean, mask_tensor,
    sample_info, output_path,
):
    """
    Detailed noise-analysis figure for a single mid-time frame:
    spatial maps, line profiles (raw vs clean), noise histogram, SNR stats.
    """
    raw = target_raw.numpy()[0]       # (T, H, W)
    clean = target_clean.numpy()[0]
    mask = mask_tensor.numpy()[0]
    diff = raw - clean
    binary = mask > 0.1

    t = raw.shape[0] // 2
    vmax_vel = _compute_dynamic_vmax(raw, mask, percentile=99)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    vel_kw = dict(cmap='RdBu_r', vmin=-vmax_vel, vmax=vmax_vel,
                  origin='lower', aspect='auto')

    # Top row — spatial maps
    axes[0, 0].imshow(raw[t].T, **vel_kw)
    axes[0, 0].set_title(f'Raw velocity (vmax={vmax_vel:.3f})')
    axes[0, 1].imshow(clean[t].T, **vel_kw)
    axes[0, 1].set_title('Clean velocity (same scale)')
    
    diff_vmax = _compute_dynamic_vmax(diff, mask, percentile=99) * 0.8
    im_diff = axes[0, 2].imshow(diff[t].T, cmap='coolwarm', vmin=-diff_vmax,
                                vmax=diff_vmax, origin='lower', aspect='auto')
    axes[0, 2].set_title('Noise removed (raw − clean)')
    fig.colorbar(im_diff, ax=axes[0, 2], shrink=0.8)
    axes[0, 3].imshow(mask[t].T, cmap='viridis', vmin=0, vmax=1,
                      origin='lower', aspect='auto')
    axes[0, 3].set_title('Magnitude mask')

    # Mark profile line through the middle of the ROI
    rows_in_mask = np.where(binary[t].any(axis=1))[0]
    if len(rows_in_mask) > 0:
        mid_row = rows_in_mask[len(rows_in_mask) // 2]
    else:
        mid_row = raw.shape[1] // 2
    for ax in axes[0, :3]:
        ax.axhline(mid_row, color='lime', lw=1, ls='--', alpha=0.7)
    for ax in axes[0]:
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom-left: lateral line profile
    x = np.arange(raw.shape[2])
    axes[1, 0].plot(x, raw[t, mid_row, :], 'r-', alpha=0.5, lw=0.8, label='Raw')
    axes[1, 0].plot(x, clean[t, mid_row, :], 'b-', lw=1.5, label='Clean')
    axes[1, 0].set_title(f'Lateral profile (row {mid_row}, frame {t})')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_ylim(-vmax_vel * 1.2, vmax_vel * 1.2)
    axes[1, 0].set_xlabel('Lateral pixel')
    axes[1, 0].set_ylabel('Velocity')

    # Bottom-middle-left: temporal profile at centre of ROI
    cols_in_mask = np.where(binary[t].any(axis=0))[0]
    if len(cols_in_mask) > 0:
        cx, cy = mid_row, cols_in_mask[len(cols_in_mask) // 2]
    else:
        cx, cy = raw.shape[1] // 2, raw.shape[2] // 2
    axes[1, 1].plot(raw[:, cx, cy], 'r-', alpha=0.5, lw=0.8, label='Raw')
    axes[1, 1].plot(clean[:, cx, cy], 'b-', lw=1.5, label='Clean')
    axes[1, 1].set_title(f'Temporal profile (pixel [{cx},{cy}])')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Velocity')

    # Bottom-middle-right: histogram of noise amplitude (in signal region)
    noise_in_roi = diff[t][binary[t]]
    if len(noise_in_roi) > 0:
        axes[1, 2].hist(noise_in_roi, bins=80, color='gray', alpha=0.7,
                        edgecolor='black', lw=0.3)
        noise_std = noise_in_roi.std()
        axes[1, 2].text(0.95, 0.95, f'σ_noise={noise_std:.4f}',
                        transform=axes[1, 2].transAxes, ha='right', va='top',
                        fontsize=10, bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    axes[1, 2].set_title(f'Noise histogram (signal region, frame {t})')
    axes[1, 2].set_xlabel('raw − clean')
    axes[1, 2].set_ylabel('Count')

    # Bottom-right: summary text
    raw_sig = raw[binary]
    clean_sig = clean[binary]
    noise_all = diff[binary]
    if len(raw_sig) > 0:
        snr_raw = raw_sig.std() / max(noise_all.std(), 1e-10)
        snr_clean = clean_sig.std() / max(noise_all.std(), 1e-10)
    else:
        snr_raw = snr_clean = 0.0

    summary_text = (
        f"SIGNAL QUALITY\n"
        f"{'─' * 30}\n"
        f"Signal region (mask>0.1): {binary.mean()*100:.1f}% of pixels\n"
        f"\n"
        f"Raw  signal  σ = {raw_sig.std():.4f}\n"
        f"Clean signal σ = {clean_sig.std():.4f}\n"
        f"Noise (raw−clean) σ = {noise_all.std():.4f}\n"
        f"\n"
        f"99th pctl |raw|  = {np.percentile(np.abs(raw_sig), 99):.4f}\n"
        f"99th pctl |clean| = {np.percentile(np.abs(clean_sig), 99):.4f}\n"
        f"\n"
        f"Dynamic vmax = {vmax_vel:.4f}\n"
        f"(99th pctl of |velocity| in ROI)\n"
    )
    axes[1, 3].axis('off')
    axes[1, 3].text(0.05, 0.95, summary_text, transform=axes[1, 3].transAxes,
                    va='top', fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', fc='lightyellow'))

    fig.suptitle(f"Noise Analysis — {sample_info['base_name']} (frame {t})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize clean vs raw training targets')
    parser.add_argument('--preprocessed_dir', type=str, default='data/preprocessed')
    parser.add_argument('--output_dir', type=str, default='results/smooth_target_viz')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--sample_indices', type=int, nargs='*', default=None,
                        help='Specific sample indices to visualize (default: 0..num_samples-1)')
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = args.sample_indices if args.sample_indices else list(range(args.num_samples))

    logger.info("=" * 60)
    logger.info("TRAINING TARGET VISUALIZATION (Dynamic Colour Scaling)")
    logger.info("Shows EXACTLY what the model sees during training")
    logger.info("=" * 60)

    for idx in indices:
        try:
            inp, raw, clean, mask, info, meta = load_raw_and_clean(
                preprocessed_dir, idx, args.split
            )
            visualize_comparison(
                inp, raw, clean, mask, info, meta,
                output_dir / f'comparison_{idx}_{info["base_name"]}.png',
            )
            visualize_noise_profile(
                inp, raw, clean, mask, info,
                output_dir / f'noise_analysis_{idx}_{info["base_name"]}.png',
            )
        except Exception as e:
            logger.error(f"Sample {idx}: {e}")
            import traceback; traceback.print_exc()

    logger.info(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
