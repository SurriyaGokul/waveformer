#!/usr/bin/env python3
"""
Generate Clean Silver-Standard Velocity Training Targets.

Raw Kasai velocity has two major problems for learning shear-wave propagation:
  1. **Background noise** — ~90 % of pixels are near-zero speckle noise that
     overwhelms the actual wave signal.
  2. **Speckle noise inside the tissue ROI** — high-frequency speckle obscures
     the smooth shear-wave fronts even inside the signal region.

This script creates ``velocity_clean_<name>.npy`` files using a three-step
pipeline that fixes both issues:

  1. **Binary ROI mask** — threshold the soft magnitude mask to identify where
     tissue signal exists.
  2. **Normalised convolution** (masked Gaussian smoothing) — Gaussian-smooth
     the velocity using only in-ROI contributions so that wave amplitude is
     *preserved* at mask boundaries (no dilution from the surrounding zeros).
  3. **Background zeroing** — everything outside the ROI is set to exactly 0,
     giving the model a trivially predictable background.

The result is a target with:
  - Clean, smooth shear-wave fronts inside the ROI (easy for the model to learn)
  - Exact zeros outside the ROI (no noise to confuse the model)
  - Full original wave amplitude preserved (no edge dilution)

Usage:
    python scripts/generate_smooth_targets.py
    python scripts/generate_smooth_targets.py --sigma 1.5
    python scripts/generate_smooth_targets.py --splits train val --sigma 1.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def masked_gaussian_smooth(data_2d: np.ndarray, mask_2d: np.ndarray,
                           sigma: float) -> np.ndarray:
    """
    Normalised-convolution Gaussian smoothing that only uses in-mask pixels.

    Standard Gaussian filtering averages signal pixels with surrounding zeros at
    the mask boundary, diluting the wave amplitude.  Normalised convolution
    divides by the smoothed mask so that each output pixel is the weighted
    average of *only* the valid neighbours.

    Args:
        data_2d: (H, W) velocity frame.
        mask_2d: (H, W) binary mask (1 = signal, 0 = background).
        sigma: Gaussian kernel sigma in pixels.

    Returns:
        (H, W) smoothed velocity, valid inside mask, zero outside.
    """
    masked_data = data_2d * mask_2d
    smoothed_numerator = gaussian_filter(masked_data, sigma=sigma)
    smoothed_denominator = gaussian_filter(mask_2d.astype(np.float64), sigma=sigma)
    # Avoid division by zero — pixels with no nearby mask contribute 0
    smoothed_denominator = np.maximum(smoothed_denominator, 1e-10)
    result = (smoothed_numerator / smoothed_denominator) * mask_2d
    return result.astype(data_2d.dtype)


def generate_clean_targets(data_dir: Path, sigma: float = 1.0,
                           mask_threshold: float = 0.1,
                           overwrite: bool = False):
    """
    Generate clean velocity targets for all samples in a directory.

    For each ``velocity_<name>.npy`` + ``magnitude_<name>.npy`` pair, creates
    ``velocity_clean_<name>.npy`` using normalised-convolution smoothing inside
    the ROI and zeroing outside.

    Args:
        data_dir: Directory containing preprocessed files.
        sigma: Gaussian kernel sigma for spatial smoothing.
        mask_threshold: Threshold on soft magnitude mask to produce binary ROI.
        overwrite: Whether to overwrite existing clean files.
    """
    # Collect raw velocity files (exclude derived ones)
    vel_files = sorted([
        f for f in data_dir.glob("velocity_*.npy")
        if not f.name.startswith("velocity_smooth_")
        and not f.name.startswith("velocity_clean_")
    ])

    if not vel_files:
        logger.warning(f"No velocity_*.npy files found in {data_dir}")
        return 0

    logger.info(
        f"Generating clean targets for {len(vel_files)} files in {data_dir} "
        f"(sigma={sigma}, mask_thresh={mask_threshold})"
    )

    created = 0
    for f in tqdm(vel_files, desc=f"  {data_dir.name}"):
        base = f.name.replace("velocity_", "")
        out_path = data_dir / f"velocity_clean_{base}"
        mag_path = data_dir / f"magnitude_{base}"

        if out_path.exists() and not overwrite:
            continue

        if not mag_path.exists():
            logger.warning(f"No magnitude file for {f.name}, skipping")
            continue

        velocity = np.load(f)          # (T, H, W) or (T, 1, H, W)
        magnitude = np.load(mag_path)  # same shape

        velocity_clean = np.zeros_like(velocity)

        if velocity.ndim == 3:  # (T, H, W)
            for t in range(velocity.shape[0]):
                binary_mask = (magnitude[t] > mask_threshold).astype(np.float64)
                velocity_clean[t] = masked_gaussian_smooth(
                    velocity[t], binary_mask, sigma
                )
        elif velocity.ndim == 4:  # (T, 1, H, W)
            for t in range(velocity.shape[0]):
                binary_mask = (magnitude[t, 0] > mask_threshold).astype(np.float64)
                velocity_clean[t, 0] = masked_gaussian_smooth(
                    velocity[t, 0], binary_mask, sigma
                )
        else:
            logger.warning(f"Unexpected ndim={velocity.ndim} for {f.name}, skipping")
            continue

        np.save(out_path, velocity_clean)
        created += 1

    logger.info(f"  Created {created} clean files (skipped {len(vel_files) - created} existing)")
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean silver-standard velocity training targets"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/preprocessed",
        help="Root preprocessed directory containing train/ and val/ subdirs"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which splits to process (default: train val)"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Gaussian sigma for spatial smoothing (default: 1.0)"
    )
    parser.add_argument(
        "--mask_threshold", type=float, default=0.1,
        help="Threshold on magnitude mask for binary ROI (default: 0.1)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing clean files"
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    total = 0

    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} not found, skipping")
            continue
        total += generate_clean_targets(
            split_dir, sigma=args.sigma,
            mask_threshold=args.mask_threshold, overwrite=args.overwrite,
        )

    logger.info(
        f"\nDone — generated {total} clean silver-standard targets "
        f"(sigma={args.sigma}, mask_thresh={args.mask_threshold})"
    )


if __name__ == "__main__":
    main()
