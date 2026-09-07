#!/usr/bin/env python3
"""
Preprocess Non-CIRS IQ Data (2FT/3FT) → Velocity Targets.

Adapts the Loupas IQ→velocity pipeline from iq_to_velocity.py to work
with the non-CIRS phantom data in 2FT(na100) and 3FT(na100) directories.

Directory Structure:
    data/
    ├── 2FT(na100)/
    │   ├── 10mmfocus/
    │   │   ├── 5kHz/
    │   │   │   ├── Plane 1/ → .npy files
    │   │   │   ├── ...
    │   │   ├── 10kHz/
    │   │   └── 20kHz/
    │   └── 15mmfocus/
    └── 3FT(na100)/
        └── ... (same structure)

Output:
    data/non_cirs_velocity/
    ├── train/
    │   ├── sparse_<name>.npy
    │   ├── velocity_<name>.npy
    │   └── magnitude_<name>.npy
    ├── val/
    └── metadata.json

Usage:
    python scripts/preprocess_non_cirs.py
    python scripts/preprocess_non_cirs.py --frequencies 5kHz 10kHz 20kHz
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.iq_to_velocity import (
    LoupasConfig,
    FREQ_TO_PRF,
    process_single_file,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_non_cirs_files(
    data_dir: Path,
    target_frequencies: Tuple[str, ...] = ('5kHz', '10kHz', '20kHz'),
    dataset_dirs: Tuple[str, ...] = ('2FT(na100)', '3FT(na100)'),
) -> List[Dict]:
    """
    Discover non-CIRS .npy IQ files for the target frequencies.

    Scans:
        data_dir/<dataset>/<focus>/<freq>/Plane <n>/*.npy

    Returns:
        List of dicts with keys: path, dataset, focus, frequency, plane, name
    """
    files = []

    for dataset_name in dataset_dirs:
        dataset_root = data_dir / dataset_name
        if not dataset_root.exists():
            logger.warning(f"Dataset directory not found: {dataset_root}")
            continue

        for focus_dir in sorted(dataset_root.iterdir()):
            if not focus_dir.is_dir():
                continue
            focus = focus_dir.name  # e.g. '10mmfocus'

            for freq_name in target_frequencies:
                freq_dir = focus_dir / freq_name
                if not freq_dir.exists():
                    logger.info(
                        f"  Skipping {dataset_name}/{focus}/{freq_name} (not found)"
                    )
                    continue

                for plane_dir in sorted(freq_dir.iterdir()):
                    if not plane_dir.is_dir():
                        continue
                    plane = plane_dir.name  # e.g. 'Plane 1'

                    for npy_file in sorted(plane_dir.glob('*.npy')):
                        files.append({
                            'path': npy_file,
                            'dataset': dataset_name,
                            'focus': focus,
                            'frequency': freq_name,
                            'plane': plane,
                            'name': npy_file.stem,
                        })

    logger.info(
        f"Discovered {len(files)} non-CIRS IQ files for {target_frequencies}"
    )

    # Log per-frequency counts
    freq_counts = {}
    for f in files:
        freq_counts[f['frequency']] = freq_counts.get(f['frequency'], 0) + 1
    for freq, count in sorted(freq_counts.items()):
        logger.info(f"  {freq}: {count} files")

    return files


def split_files(
    files: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Split files into train/val, stratified by (dataset, focus, frequency)."""
    rng = np.random.RandomState(seed)

    groups = {}
    for f in files:
        key = (f['dataset'], f['focus'], f['frequency'])
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
    target_frequencies: Tuple[str, ...] = ('5kHz', '10kHz', '20kHz'),
):
    """Run full Loupas preprocessing on non-CIRS phantom data."""

    # Discover files
    files = discover_non_cirs_files(data_dir, target_frequencies)
    if not files:
        logger.error("No files found. Aborting.")
        return

    # Split
    splits = split_files(files, config.train_ratio, config.seed)

    # Create output directories
    for split_name in ['train', 'val']:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    all_vel_abs_max = []
    metadata_entries = {'train': [], 'val': []}

    for split_name in ['train', 'val']:
        split_files_list = splits[split_name]
        split_dir = output_dir / split_name

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(split_files_list)} {split_name.upper()} files")
        logger.info(f"{'='*60}")

        for file_info in tqdm(split_files_list, desc=f"{split_name}"):
            # Create a unique descriptive name
            name = (
                f"{file_info['dataset']}_{file_info['focus']}"
                f"_{file_info['frequency']}_{file_info['plane']}"
                f"_{file_info['name']}"
            ).replace(' ', '_').replace('(', '').replace(')', '')

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
            entry['dataset'] = file_info['dataset']
            entry['focus'] = file_info['focus']
            entry['plane'] = file_info['plane']
            metadata_entries[split_name].append(entry)

    # Compute global normalization
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
            'target_frequencies': list(target_frequencies),
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

    logger.info(f"\n{'='*60}")
    logger.info("NON-CIRS PREPROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Output directory:   {output_dir}")
    logger.info(f"  Training samples:   {len(metadata_entries['train'])}")
    logger.info(f"  Validation samples: {len(metadata_entries['val'])}")
    logger.info(f"  Velocity norm (99th pct |v|): {global_vel_max:.4f} mm/s")
    logger.info(f"  Metadata saved:     {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess non-CIRS (2FT/3FT) IQ data to velocity targets"
    )
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Root data directory containing 2FT(na100) and 3FT(na100)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/non_cirs_velocity',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--frequencies', type=str, nargs='+',
        default=['5kHz', '10kHz', '20kHz'],
        help='Target frequencies to process (default: 5kHz 10kHz 20kHz)'
    )
    parser.add_argument(
        '--sparse_step', type=int, default=12,
        help='Take every Nth IQ frame (must match upsampling_factor)'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Fraction of files for training (default: 0.8)'
    )

    args = parser.parse_args()

    config = LoupasConfig(
        sparse_step=args.sparse_step,
        target_frequencies=tuple(args.frequencies),
        train_ratio=args.train_ratio,
    )

    logger.info("Non-CIRS Loupas IQ-to-Velocity Pipeline")
    logger.info(f"  Frequencies: {args.frequencies}")
    logger.info(f"  Sparse step: {config.sparse_step}")

    run_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        config=config,
        target_frequencies=tuple(args.frequencies),
    )


if __name__ == '__main__':
    main()
