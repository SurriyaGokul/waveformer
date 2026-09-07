#!/usr/bin/env python3
"""
Preprocess data for higher upsampling factors (20x, 25x, etc.).

Since raw IQ files have been modified and no longer match the original 50-frame
windows used for preprocessing, this script takes a different approach:

  - Keep the same 5 sparse IQ frames (T_sparse=5) from data/fullscale_velocity/
  - Increase the model's upsampling_factor (20x or 25x)
  - Temporally interpolate the 49-frame velocity targets to T_dense frames
    where T_dense = 1 + (T_sparse - 1) * factor

This tests the model's ability to achieve higher temporal super-resolution
from the same sparse input.  The interpolated velocity targets provide smooth
ground truth at the finer temporal grid.

Steps per sample:
  1. Copy existing sparse IQ unchanged (T_sparse=5)
  2. Interpolate velocity from 49 → T_dense frames (cubic interpolation)
  3. Copy magnitude mask unchanged
  4. Save to output directory with updated metadata

Usage:
    python scripts/preprocess_higher_upsampling.py --upsampling_factor 20
    python scripts/preprocess_higher_upsampling.py --upsampling_factor 25
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def interpolate_velocity(
    vel: np.ndarray,
    T_target: int,
    method: str = "cubic",
) -> np.ndarray:
    """
    Temporally interpolate velocity from T_source to T_target frames.

    Args:
        vel: (T_source, H, W) float32 velocity in mm/s
        T_target: Number of output temporal frames
        method: Interpolation method ('linear', 'cubic')

    Returns:
        (T_target, H, W) float32 interpolated velocity
    """
    T_src, H, W = vel.shape
    if T_src == T_target:
        return vel.copy()

    # Source and target time grids (normalised to [0, 1])
    t_src = np.linspace(0, 1, T_src)
    t_tgt = np.linspace(0, 1, T_target)

    # Reshape to (T, H*W) for vectorised interpolation
    vel_flat = vel.reshape(T_src, -1)  # (T_src, H*W)

    # Create interpolator and evaluate
    interp_func = interp1d(t_src, vel_flat, axis=0, kind=method, fill_value="extrapolate")
    vel_interp = interp_func(t_tgt)  # (T_target, H*W)

    return vel_interp.reshape(T_target, H, W).astype(np.float32)


def process_sample(
    sample_meta: dict,
    split: str,
    source_data_dir: Path,
    output_dir: Path,
    upsampling_factor: int,
    T_sparse: int = 5,
) -> Optional[dict]:
    """
    Process a single sample: copy sparse IQ, interpolate velocity, copy mask.

    Returns updated metadata dict or None on failure.
    """
    name = sample_meta["name"]
    T_dense = 1 + (T_sparse - 1) * upsampling_factor

    # ---- Copy sparse IQ unchanged ----
    sparse_path = source_data_dir / split / f"sparse_{name}.npy"
    if not sparse_path.exists():
        logger.error(f"  SKIP {name}: sparse file not found: {sparse_path}")
        return None

    sparse_iq = np.load(sparse_path)  # (5, H, W) complex64
    if sparse_iq.shape[0] != T_sparse:
        logger.warning(
            f"  SKIP {name}: sparse has T={sparse_iq.shape[0]}, expected {T_sparse}"
        )
        return None

    # ---- Load and interpolate velocity ----
    vel_path = source_data_dir / split / f"velocity_{name}.npy"
    if not vel_path.exists():
        logger.error(f"  SKIP {name}: velocity file not found: {vel_path}")
        return None

    vel_orig = np.load(vel_path)  # (49, H, W) float32
    vel_interp = interpolate_velocity(vel_orig, T_dense, method="cubic")

    # ---- Copy magnitude mask ----
    mag_path = source_data_dir / split / f"magnitude_{name}.npy"
    if not mag_path.exists():
        logger.error(f"  SKIP {name}: magnitude file not found: {mag_path}")
        return None
    magnitude = np.load(mag_path)  # (H, W)

    # ---- Save outputs ----
    out_split = output_dir / split
    np.save(out_split / f"sparse_{name}.npy", sparse_iq)
    np.save(out_split / f"velocity_{name}.npy", vel_interp)
    np.save(out_split / f"magnitude_{name}.npy", magnitude)

    # ---- Update metadata ----
    updated = dict(sample_meta)
    updated["sparse_shape"] = list(sparse_iq.shape)
    updated["velocity_shape"] = list(vel_interp.shape)
    updated["sparse_step"] = upsampling_factor
    updated["velocity_interpolated_from"] = int(vel_orig.shape[0])

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data for higher upsampling factors"
    )
    parser.add_argument(
        "--upsampling_factor", type=int, required=True,
        help="New upsampling factor (e.g. 20, 25)",
    )
    parser.add_argument(
        "--source_data_dir", type=str, default="data/fullscale_velocity",
        help="Source preprocessed data directory (default: data/fullscale_velocity)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data/fullscale_velocity_{factor}x)",
    )
    args = parser.parse_args()

    factor = args.upsampling_factor
    source_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/fullscale_velocity_{factor}x")

    # T_sparse is fixed at 5 (same as original data)
    T_sparse = 5
    T_dense = 1 + (T_sparse - 1) * factor

    logger.info(f"Preprocessing for {factor}x upsampling")
    logger.info(f"  Source: {source_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  T_sparse={T_sparse} (unchanged), T_dense={T_dense} (interpolated from 49)")

    # Load source metadata
    meta_path = source_dir / "metadata.json"
    if not meta_path.exists():
        logger.error(f"Metadata not found: {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    # Determine splits to process
    splits_to_process = []
    for split_name in ["train", "val", "inference"]:
        if split_name in meta and meta[split_name]:
            splits_to_process.append(split_name)

    # Create output directories
    for split_name in splits_to_process:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Build new metadata
    new_meta = {
        "config": dict(meta.get("config", {})),
        "normalization": dict(meta.get("normalization", {})),
    }
    new_meta["config"]["sparse_step"] = factor
    new_meta["config"]["upsampling_factor"] = factor
    new_meta["config"]["velocity_interpolated"] = True
    new_meta["config"]["velocity_interpolated_from_T"] = 49

    total_processed = 0
    total_skipped = 0

    for split_name in splits_to_process:
        samples = meta[split_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(samples)} {split_name.upper()} samples")
        logger.info(f"{'='*60}")

        processed_samples = []

        for sample in tqdm(samples, desc=split_name):
            result = process_sample(
                sample_meta=sample,
                split=split_name,
                source_data_dir=source_dir,
                output_dir=output_dir,
                upsampling_factor=factor,
                T_sparse=T_sparse,
            )
            if result is not None:
                processed_samples.append(result)
                total_processed += 1
            else:
                total_skipped += 1

        new_meta[split_name] = processed_samples
        logger.info(f"  {split_name}: {len(processed_samples)} processed")

    # Save metadata
    meta_out = output_dir / "metadata.json"
    with open(meta_out, "w") as f:
        json.dump(new_meta, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"PREPROCESSING COMPLETE for {factor}x upsampling")
    logger.info(f"{'='*60}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Processed: {total_processed} samples")
    logger.info(f"  Skipped:   {total_skipped} samples")
    logger.info(f"  T_sparse={T_sparse}, T_dense={T_dense}")
    logger.info(f"  Velocity interpolated: 49 → {T_dense} frames (cubic)")
    logger.info(f"  Metadata: {meta_out}")

    # Verify one sample
    for split_name in splits_to_process:
        split_dir = output_dir / split_name
        sparse_files = sorted(split_dir.glob("sparse_*.npy"))
        if sparse_files:
            s = np.load(sparse_files[0])
            v_files = sorted(split_dir.glob("velocity_*.npy"))
            v = np.load(v_files[0]) if v_files else None
            logger.info(
                f"  Verification [{split_name}]: sparse={s.shape}, "
                f"velocity={v.shape if v is not None else 'N/A'}"
            )
            break


if __name__ == "__main__":
    main()
