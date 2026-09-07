#!/usr/bin/env python3
"""
Convert IQ Data to Velocity Targets using 2D Loupas Autocorrelator.

Faithfully implements the MATLAB IQ2VEl.m pipeline in Python:
    1. Determine ROI from non-zero IQ regions
    2. Compute axial displacement using 2D Loupas autocorrelator (Loupas et al., 1995)
    3. Smooth displacement using 2D median filter [12×6] (~0.6 mm × 0.6 mm)
    4. Compute material velocity via finite difference (mm/s)
    5. (Optional) 10× temporal interpolation before velocity computation

Physics References:
    - Loupas et al., "An axial velocity estimator for ultrasound blood flow imaging,
      based on a full evaluation of the Doppler equation by means of a two-dimensional
      autocorrelation approach," IEEE TUFFC, 1995.
    - Maksuti et al., 2016 (related usage in elastography/ultrasound tracking)

Output Structure:
    data/phantom_velocity/
    ├── train/
    │   ├── sparse_<name>.npy       # Sparse IQ input  (T_sparse, H, W) complex64
    │   ├── velocity_<name>.npy     # Velocity target   (T_vel, H, W) float32 [mm/s]
    │   └── magnitude_<name>.npy    # ROI mask          (H, W) float32
    ├── val/
    │   └── ... (same structure)
    └── metadata.json

Usage:
    python scripts/iq_to_velocity.py
    python scripts/iq_to_velocity.py --data_dir data --output_dir data/phantom_velocity
    python scripts/iq_to_velocity.py --with_interpolation   # include 10× interp
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root on sys.path so src.utils.loupas is importable when script is
# run directly (python scripts/iq_to_velocity.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.loupas import (  # noqa: E402
    LoupasConfig,
    determine_roi,
    loupas_2d_displacement,
    smooth_displacement,
    interpolate_displacement,
    compute_velocity,
    bandpass_filter_velocity,
)


# ============================================================================
# Configuration
# ============================================================================

# Detection PRF mapping (folder name → Hz)
FREQ_TO_PRF = {
    '500Hz': 500.0,
    '1kHz': 1000.0,
    '5kHz': 5000.0,
    '10kHz': 10000.0,
    '20kHz': 20000.0,
}


# ============================================================================
# Full Pipeline
# ============================================================================

def process_single_file(
    file_path: Path,
    freq_name: str,
    config: LoupasConfig,
) -> Optional[Dict]:
    """
    Run the full Loupas pipeline on a single IQ data file.

    Returns:
        Dictionary with output arrays and metadata, or None on failure.
    """
    try:
        # --- Load ---
        iq_data = np.load(file_path)
        if not np.iscomplexobj(iq_data):
            raise ValueError(f"Expected complex IQ data, got dtype={iq_data.dtype}")
        H, W, T = iq_data.shape
        logger.info(f"  Loaded IQ data: shape={iq_data.shape}")

        # --- PRF / frame time ---
        prf = FREQ_TO_PRF.get(freq_name)
        if prf is None:
            raise ValueError(f"Unknown frequency folder: {freq_name}")
        frame_time = 1.0 / prf  # seconds

        # --- Step 1: Loupas 2D displacement ---
        disp, roi_info = loupas_2d_displacement(iq_data, config)
        logger.info(
            f"  Loupas displacement: shape={disp.shape}, "
            f"range=[{disp.min():.2f}, {disp.max():.2f}] μm"
        )

        # --- Step 2: Median filter ---
        medfiltdisp = smooth_displacement(
            disp, config.medfilt_kernel_ax, config.medfilt_kernel_lat
        )

        # --- Step 3 & 4: Velocity computation ---
        if config.do_interpolation:
            interpdisp = interpolate_displacement(medfiltdisp, config.interp_factor)
            vel = compute_velocity(interpdisp, frame_time, config.interp_factor)
            if config.do_bandpass:
                fs = prf * config.interp_factor
                vel = bandpass_filter_velocity(
                    vel, fs, config.bandpass_low, config.bandpass_high
                )
        else:
            vel = compute_velocity(medfiltdisp, frame_time, interp_factor=1)

        logger.info(
            f"  Velocity: shape={vel.shape}, "
            f"range=[{vel.min():.2f}, {vel.max():.2f}] mm/s, "
            f"std={vel.std():.4f}"
        )

        # --- Extract sparse IQ input ---
        min_row = roi_info['min_row']
        max_row = roi_info['max_row']
        min_col = roi_info['min_col']
        max_col = roi_info['max_col']

        # Crop IQ to ROI
        iq_roi = iq_data[min_row:max_row + 1, min_col:max_col + 1, :]

        # Clutter filter: subtract temporal mean
        iq_filtered = iq_roi - iq_roi.mean(axis=2, keepdims=True)

        # Sparse subsampling
        sparse_indices = np.arange(0, T, config.sparse_step)
        sparse_iq = iq_filtered[:, :, sparse_indices]

        # --- Align spatial dimensions ---
        # Velocity spatial dims might be smaller than IQ ROI due to Loupas kernel
        vel_H, vel_W, vel_T = vel.shape
        sparse_iq = sparse_iq[:vel_H, :vel_W, :]

        # --- Compute magnitude mask (from raw IQ ROI) ---
        iq_mag = np.abs(iq_roi[:vel_H, :vel_W, :])
        mag_max = iq_mag.max()
        if mag_max > 0:
            iq_mag_norm = iq_mag / mag_max
        else:
            iq_mag_norm = iq_mag
        # Average over time, threshold at 1%
        mag_mask = (iq_mag_norm.mean(axis=2) > 0.01).astype(np.float32)

        # --- Transpose to (T, H, W) format for saving ---
        sparse_out = np.transpose(sparse_iq, (2, 0, 1)).astype(np.complex64)
        vel_out = np.transpose(vel, (2, 0, 1)).astype(np.float32)

        return {
            'sparse_iq': sparse_out,
            'velocity': vel_out,
            'magnitude_mask': mag_mask,
            'metadata': {
                'frequency': freq_name,
                'prf_hz': prf,
                'frame_time_s': frame_time,
                'original_iq_shape': list(iq_data.shape),
                'roi': roi_info,
                'sparse_shape': list(sparse_out.shape),
                'velocity_shape': list(vel_out.shape),
                'velocity_range_mms': [float(vel_out.min()), float(vel_out.max())],
                'velocity_std_mms': float(vel_out.std()),
                'sparse_step': config.sparse_step,
                'interp_factor': config.interp_factor if config.do_interpolation else 1,
                'original_path': str(file_path),
            },
        }

    except Exception as e:
        logger.error(f"  Error processing {file_path}: {e}", exc_info=True)
        return None


# ============================================================================
# Data Discovery & Orchestration
# ============================================================================

def discover_phantom_files(
    data_dir: Path,
    target_frequencies: Tuple[str, ...] = ('10kHz', '20kHz'),
) -> List[Dict]:
    """
    Discover CIRS Phantom .npy files for the target frequencies.

    Returns:
        List of dicts with keys: path, focus, inclusion, frequency, name
    """
    phantom_root = data_dir / 'CIRS_Phantom_Latest'
    files = []

    if not phantom_root.exists():
        logger.error(f"Phantom data directory not found: {phantom_root}")
        return files

    for dataset_dir in sorted(phantom_root.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith('Data set'):
            continue

        for focus_dir in sorted(dataset_dir.iterdir()):
            if not focus_dir.is_dir():
                continue

            for inclusion_dir in sorted(focus_dir.iterdir()):
                if not inclusion_dir.is_dir():
                    continue

                for freq_name in target_frequencies:
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

    logger.info(f"Discovered {len(files)} phantom IQ files for {target_frequencies}")
    return files


def split_files(
    files: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split files into train/val sets, stratified by inclusion type.
    """
    rng = np.random.RandomState(seed)

    # Group by (focus, inclusion, frequency)
    groups = {}
    for f in files:
        key = (f['focus'], f['inclusion'], f['frequency'])
        groups.setdefault(key, []).append(f)

    train_files, val_files = [], []

    for key, group_files in sorted(groups.items()):
        rng.shuffle(group_files)
        n_train = max(1, int(len(group_files) * train_ratio))
        train_files.extend(group_files[:n_train])
        val_files.extend(group_files[n_train:])

    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")
    return {'train': train_files, 'val': val_files}


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    config: LoupasConfig,
):
    """
    Run the full preprocessing pipeline on all Phantom 10kHz + 20kHz data.
    """
    # Discover files
    files = discover_phantom_files(data_dir, config.target_frequencies)
    if not files:
        logger.error("No files found. Aborting.")
        return

    # Split into train/val
    splits = split_files(files, config.train_ratio, config.seed)

    # Create output directories
    for split_name in ['train', 'val']:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Track global velocity statistics for normalization
    all_vel_abs_max = []
    metadata_entries = {'train': [], 'val': []}

    # Process each split
    for split_name in ['train', 'val']:
        split_files_list = splits[split_name]
        split_dir = output_dir / split_name

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(split_files_list)} {split_name.upper()} files")
        logger.info(f"{'='*60}")

        for file_info in tqdm(split_files_list, desc=f"{split_name}"):
            name = (
                f"{file_info['focus']}_{file_info['inclusion']}"
                f"_{file_info['frequency']}_{file_info['name']}"
            )
            logger.info(f"\nProcessing: {name}")

            result = process_single_file(
                file_info['path'], file_info['frequency'], config
            )
            if result is None:
                continue

            # Save outputs
            np.save(split_dir / f'sparse_{name}.npy', result['sparse_iq'])
            np.save(split_dir / f'velocity_{name}.npy', result['velocity'])
            np.save(split_dir / f'magnitude_{name}.npy', result['magnitude_mask'])

            # Track statistics
            vel_abs_max = float(np.abs(result['velocity']).max())
            all_vel_abs_max.append(vel_abs_max)

            entry = result['metadata'].copy()
            entry['name'] = name
            entry['focus'] = file_info['focus']
            entry['inclusion'] = file_info['inclusion']
            metadata_entries[split_name].append(entry)

    # Compute global normalization scalar
    if all_vel_abs_max:
        global_vel_max = float(np.percentile(all_vel_abs_max, 99))
    else:
        global_vel_max = 1.0

    # Save metadata
    metadata = {
        'config': {
            'speed_of_sound': config.speed_of_sound,
            'center_frequency': config.center_frequency,
            'M': config.M,
            'N': config.N,
            'medfilt_kernel': [config.medfilt_kernel_ax, config.medfilt_kernel_lat],
            'sparse_step': config.sparse_step,
            'interp_factor': config.interp_factor if config.do_interpolation else 1,
            'do_interpolation': config.do_interpolation,
            'target_frequencies': list(config.target_frequencies),
            'train_ratio': config.train_ratio,
        },
        'normalization': {
            'velocity_abs_max_99pct': global_vel_max,
            'velocity_unit': 'mm/s',
        },
        'train': metadata_entries['train'],
        'val': metadata_entries['val'],
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info(f"  Training samples:  {len(metadata_entries['train'])}")
    logger.info(f"  Validation samples: {len(metadata_entries['val'])}")
    logger.info(f"  Velocity norm (99th pct |v|): {global_vel_max:.4f} mm/s")
    logger.info(f"  Metadata saved:    {metadata_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert IQ data to velocity targets using 2D Loupas autocorrelator"
    )
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Root data directory containing CIRS_Phantom_Latest'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/phantom_velocity',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--sparse_step', type=int, default=12,
        help='Take every Nth IQ frame for sparse input (must match upsampling_factor, default: 12)'
    )
    parser.add_argument(
        '--with_interpolation', action='store_true',
        help='Enable 10× temporal interpolation before velocity computation'
    )
    parser.add_argument(
        '--with_bandpass', action='store_true',
        help='Apply bandpass filter (10–800 Hz) to velocity'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Fraction of files for training (default: 0.8)'
    )
    parser.add_argument(
        '--center_freq', type=float, default=7.6e6,
        help='Transducer center frequency in Hz (default: 7.6e6)'
    )
    parser.add_argument(
        '--frequencies', type=str, nargs='+',
        default=['1kHz', '5kHz', '10kHz', '20kHz'],
        help='Target frequencies to process (default: 1kHz 5kHz 10kHz 20kHz)'
    )

    args = parser.parse_args()

    config = LoupasConfig(
        center_frequency=args.center_freq,
        sparse_step=args.sparse_step,
        do_interpolation=args.with_interpolation,
        do_bandpass=args.with_bandpass,
        train_ratio=args.train_ratio,
        target_frequencies=tuple(args.frequencies),
    )

    logger.info("Loupas IQ-to-Velocity Pipeline")
    logger.info(f"  Config: c={config.speed_of_sound} m/s, "
                f"f_c={config.center_frequency/1e6:.1f} MHz, "
                f"M={config.M}, N={config.N}")
    logger.info(f"  Sparse step: {config.sparse_step}")
    logger.info(f"  Interpolation: {'10×' if config.do_interpolation else 'OFF (native rate)'}")
    logger.info(f"  Bandpass: {'ON (10-800 Hz)' if config.do_bandpass else 'OFF'}")

    run_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        config=config,
    )


if __name__ == '__main__':
    main()
