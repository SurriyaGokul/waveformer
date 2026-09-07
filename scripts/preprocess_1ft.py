#!/usr/bin/env python3
"""
Preprocess 1FT IQ data → (sparse IQ, Loupas velocity, magnitude mask) triplets.

Replicates the exact naming and metadata schema used by data/fullscale_velocity
so that 1FT samples drop straight into the existing v10b training pipeline.

Directory structure consumed:
    data/1FT/
    └── SampleN/
        └── [10mmfocus|15mmfocus]/
            └── [500Hz|5kHz|10kHz|20kHz]/
                └── Plane N/
                    └── <timestamp>.npy      # (H, W, T) complex128

Output (appended to existing data/fullscale_velocity/):
    data/fullscale_velocity/train/
        sparse_1ft_s<N>_<focus>_<Plane_X>_<freq>_<ts>_w<w>.npy   (T_sparse, H, W) complex64
        velocity_1ft_s<N>_<focus>_<Plane_X>_<freq>_<ts>_w<w>.npy (T_vel, H, W) float32 mm/s
        magnitude_1ft_s<N>_<focus>_<Plane_X>_<freq>_<ts>_w<w>.npy (H, W) float32
    data/fullscale_velocity/val/
        ... (same structure)

Windowing:
    T=50  frames → 1 window  w0=[0,50]
    T=100 frames → 2 windows w0=[0,50], w1=[50,100]

Training frequencies (v10b): 500Hz, 5kHz, 10kHz, 20kHz  (1kHz skipped)

Usage:
    python scripts/preprocess_1ft.py
    python scripts/preprocess_1ft.py --data_dir data/1FT --output_dir data/fullscale_velocity
    python scripts/preprocess_1ft.py --dry_run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.iq_to_velocity import (
    LoupasConfig,
    FREQ_TO_PRF,
    loupas_2d_displacement,
    smooth_displacement,
    compute_velocity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WINDOW_SIZE = 50
TARGET_FREQS = ("500Hz", "5kHz", "10kHz", "20kHz")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_1ft_files(data_dir: Path, frequencies: Tuple[str, ...]) -> List[Dict]:
    """Scan data/1FT/SampleN/focus/freq/Plane N/*.npy and return file records."""
    files = []
    for sample_dir in sorted(data_dir.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("Sample"):
            continue
        sample_num = sample_dir.name.replace("Sample", "")

        for focus_dir in sorted(sample_dir.iterdir()):
            if not focus_dir.is_dir():
                continue
            focus = focus_dir.name  # '10mmfocus' or '15mmfocus'

            for freq_name in frequencies:
                freq_dir = focus_dir / freq_name
                if not freq_dir.exists():
                    continue
                prf = FREQ_TO_PRF.get(freq_name)
                if prf is None:
                    continue

                for plane_dir in sorted(freq_dir.iterdir()):
                    if not plane_dir.is_dir():
                        continue
                    plane = plane_dir.name  # 'Plane 1', 'Plane 2', ...

                    for npy_file in sorted(plane_dir.glob("*.npy")):
                        files.append({
                            "path": npy_file,
                            "sample_num": sample_num,
                            "focus": focus,
                            "freq": freq_name,
                            "prf": prf,
                            "plane": plane,
                            "timestamp": npy_file.stem,
                        })

    logger.info(f"Discovered {len(files)} 1FT .npy files")
    from collections import Counter
    counts = Counter(f["freq"] for f in files)
    for freq, n in sorted(counts.items()):
        logger.info(f"  {freq}: {n} files")
    return files


# ---------------------------------------------------------------------------
# Core: process one 50-frame IQ window
# ---------------------------------------------------------------------------

def process_iq_window(
    iq_window: np.ndarray,   # (H, W, T_window) complex
    freq_name: str,
    config: LoupasConfig,
) -> Optional[Dict]:
    """
    Run the full Loupas pipeline on a 50-frame IQ window.

    Returns dict with sparse_iq, velocity, magnitude_mask, roi_info, metadata;
    or None on failure.
    """
    try:
        H, W, T = iq_window.shape
        prf = FREQ_TO_PRF[freq_name]
        frame_time = 1.0 / prf

        # Step 1: Loupas displacement
        disp, roi_info = loupas_2d_displacement(iq_window, config)

        # Step 2: Median filter
        medfiltdisp = smooth_displacement(
            disp, config.medfilt_kernel_ax, config.medfilt_kernel_lat
        )

        # Step 3: Velocity via finite difference
        vel = compute_velocity(medfiltdisp, frame_time, interp_factor=1)

        # Crop IQ to ROI and clutter-filter
        min_row, max_row = roi_info["min_row"], roi_info["max_row"]
        min_col, max_col = roi_info["min_col"], roi_info["max_col"]
        iq_roi = iq_window[min_row:max_row + 1, min_col:max_col + 1, :]
        iq_filtered = iq_roi - iq_roi.mean(axis=2, keepdims=True)

        # Sparse subsampling
        sparse_indices = np.arange(0, T, config.sparse_step)
        sparse_iq = iq_filtered[:, :, sparse_indices]

        # Align spatial dims to velocity (Loupas kernel trims a few rows)
        vel_H, vel_W, vel_T = vel.shape
        sparse_iq = sparse_iq[:vel_H, :vel_W, :]

        # Magnitude mask (ROI, time-averaged, 1% threshold)
        iq_mag = np.abs(iq_roi[:vel_H, :vel_W, :])
        mag_max = iq_mag.max()
        iq_mag_norm = iq_mag / mag_max if mag_max > 0 else iq_mag
        mag_mask = (iq_mag_norm.mean(axis=2) > 0.01).astype(np.float32)

        # Transpose: (H, W, T) → (T, H, W)
        sparse_out = np.transpose(sparse_iq, (2, 0, 1)).astype(np.complex64)
        vel_out = np.transpose(vel, (2, 0, 1)).astype(np.float32)

        return {
            "sparse_iq": sparse_out,
            "velocity": vel_out,
            "magnitude_mask": mag_mask,
            "roi_info": roi_info,
            "prf": prf,
            "frame_time_s": frame_time,
        }

    except Exception as e:
        logger.error(f"  Loupas failed: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_files(files: List[Dict], train_ratio: float, seed: int) -> Dict[str, List[Dict]]:
    """80/20 random split, stratified by (sample, focus, freq)."""
    rng = np.random.RandomState(seed)
    groups: Dict[Tuple, List] = {}
    for f in files:
        key = (f["sample_num"], f["focus"], f["freq"])
        groups.setdefault(key, []).append(f)

    train_files, val_files = [], []
    for key, group in sorted(groups.items()):
        arr = list(group)
        rng.shuffle(arr)
        n_train = max(1, int(len(arr) * train_ratio))
        train_files.extend(arr[:n_train])
        val_files.extend(arr[n_train:])

    logger.info(f"File-level split: {len(train_files)} train, {len(val_files)} val")
    return {"train": train_files, "val": val_files}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    frequencies: Tuple[str, ...],
    config: LoupasConfig,
    dry_run: bool = False,
):
    # 1. Discover .npy files
    files = discover_1ft_files(data_dir, frequencies)
    if not files:
        logger.error("No .npy files found — run mat-to-npy conversion first.")
        return

    # 2. Split at file level (each file may produce 1 or 2 windows)
    splits = split_files(files, config.train_ratio, config.seed)

    # 3. Load existing metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"train": [], "val": []}

    # 4. Process each split
    new_samples_train, new_samples_val = 0, 0

    for split_name, split_files_list in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        new_entries = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(split_files_list)} {split_name.upper()} files")
        logger.info(f"{'='*60}")

        for file_info in tqdm(split_files_list, desc=split_name):
            path = file_info["path"]

            try:
                iq_full = np.load(path).astype(np.complex64)
            except Exception as e:
                logger.error(f"Cannot load {path}: {e}")
                continue

            if iq_full.ndim != 3:
                logger.warning(f"Skipping {path}: expected (H,W,T), got {iq_full.shape}")
                continue

            H, W, T = iq_full.shape

            # Determine windows from frame count
            if T >= 100:
                windows = [(0, 50, 0), (50, 100, 1)]
            elif T >= WINDOW_SIZE:
                windows = [(0, WINDOW_SIZE, 0)]
            else:
                logger.warning(f"Skipping {path}: T={T} < {WINDOW_SIZE}")
                continue

            sample_tag = f"s{file_info['sample_num']}"
            plane_tag = file_info["plane"].replace(" ", "_")
            base_name = (
                f"1ft_{sample_tag}_{file_info['focus']}"
                f"_{plane_tag}_{file_info['freq']}"
                f"_{file_info['timestamp']}"
            )

            for win_start, win_end, win_idx in windows:
                iq_win = iq_full[:, :, win_start:win_end]
                name = f"{base_name}_w{win_idx}"

                sparse_path = split_dir / f"sparse_{name}.npy"
                vel_path = split_dir / f"velocity_{name}.npy"
                mag_path = split_dir / f"magnitude_{name}.npy"

                if sparse_path.exists():
                    logger.debug(f"Skipping existing: {name}")
                    continue

                if dry_run:
                    logger.info(f"[dry-run] Would write: {name}")
                    continue

                result = process_iq_window(iq_win, file_info["freq"], config)
                if result is None:
                    logger.warning(f"  Loupas failed for {name}, skipping")
                    continue

                np.save(sparse_path, result["sparse_iq"])
                np.save(vel_path, result["velocity"])
                np.save(mag_path, result["magnitude_mask"])

                roi = result["roi_info"]
                entry = {
                    "name": name,
                    "source": "1ft",
                    "tissue_type": "viscoelastic",
                    "frequency": file_info["freq"],
                    "prf_hz": result["prf"],
                    "frame_time_s": result["frame_time_s"],
                    "sample": f"Sample{file_info['sample_num']}",
                    "focus": file_info["focus"],
                    "region": file_info["plane"],
                    "original_iq_shape": [H, W, T],
                    "window": [win_start, win_end],
                    "roi": roi,
                    "sparse_shape": list(result["sparse_iq"].shape),
                    "velocity_shape": list(result["velocity"].shape),
                    "velocity_range_mms": [
                        float(result["velocity"].min()),
                        float(result["velocity"].max()),
                    ],
                    "velocity_std_mms": float(result["velocity"].std()),
                    "sparse_step": config.sparse_step,
                    "interp_factor": 1,
                    "original_path": str(path.resolve()),
                }
                new_entries.append(entry)

                if split_name == "train":
                    new_samples_train += 1
                else:
                    new_samples_val += 1

        # Append to existing metadata
        metadata.setdefault(split_name, []).extend(new_entries)

    if not dry_run:
        # Update sample counts
        counts = metadata.get("sample_counts", {})
        counts["train"] = len(metadata.get("train", []))
        counts["val"] = len(metadata.get("val", []))
        counts["inference"] = counts.get("inference", 0)
        metadata["sample_counts"] = counts

        # Add 1ft tissue properties if not present
        tissue = metadata.setdefault("tissue_properties", {})
        tissue.setdefault("1ft", {
            "material": "viscoelastic",
            "description": "In-vitro tissue phantom ~1 kPa",
            "background_modulus_kpa": 1.0,
            "density_kg_m3": 1000.0,
            "viscosity_pa_s": 0.6,
            "estimated_shear_speed_m_s": 1.0,
        })

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"\nMetadata updated: {meta_path}")

    logger.info(f"\n{'='*60}")
    logger.info("1FT PREPROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  New train samples:  {new_samples_train}")
    logger.info(f"  New val samples:    {new_samples_val}")
    logger.info(f"  Output directory:   {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess 1FT IQ data into data/fullscale_velocity"
    )
    parser.add_argument("--data_dir", default="data/1FT")
    parser.add_argument("--output_dir", default="data/fullscale_velocity")
    parser.add_argument(
        "--frequencies", nargs="+",
        default=list(TARGET_FREQS),
        help="Frequencies to include (default: 500Hz 5kHz 10kHz 20kHz)",
    )
    parser.add_argument("--sparse_step", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Discover files and show plan without writing anything")
    args = parser.parse_args()

    config = LoupasConfig(sparse_step=args.sparse_step, train_ratio=args.train_ratio, seed=args.seed)

    logger.info("1FT Loupas Preprocessing")
    logger.info(f"  Source:      {args.data_dir}")
    logger.info(f"  Output:      {args.output_dir}")
    logger.info(f"  Frequencies: {args.frequencies}")
    logger.info(f"  sparse_step: {args.sparse_step} (→ 5 sparse frames per window)")
    logger.info(f"  dry_run:     {args.dry_run}")

    run_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        frequencies=tuple(args.frequencies),
        config=config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
