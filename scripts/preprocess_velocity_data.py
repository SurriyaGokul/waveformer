#!/usr/bin/env python3
"""
Unified Preprocessing Pipeline for Velocity Estimation.

This script preprocesses ALL IQ data from 2FT, 3FT, and CIRS datasets for
direct velocity estimation training with proper physics-based train/val split.

Key Features:
1. Global Normalization: Computes GLOBAL_MAX from 2FT data and applies to all
2. Clutter Filtering: Subtracts temporal mean to remove static tissue signal
3. Kasai Velocity: Computes particle velocity from phase differences
4. Velocity Normalization: Divide by pi to get range [-1, 1], ready for training
5. Magnitude Masking: Creates ROI masks from IQ magnitude to focus on signal regions
6. Physics-Based Split: Train on Planes 1-4 + CIRS (10/60/100 kPa), Validate on Plane 5 + CIRS 50kPa

Output Structure:
    data/preprocessed/
    ├── train/
    │   ├── sparse_<name>.npy    # Sparse IQ input (T_sparse, H, W) complex
    │   ├── velocity_<name>.npy  # Dense velocity target (T_dense-1, H, W)
    │   └── magnitude_<name>.npy # Magnitude mask (T_dense-1, H, W)
    ├── val/
    │   └── ... (same structure)
    └── metadata.json            # Global normalization constants and split info

Usage:
    python scripts/preprocess_velocity_data.py
    python scripts/preprocess_velocity_data.py --visualize
    python scripts/preprocess_velocity_data.py --data_dir data --output_dir data/preprocessed
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _parse_cirs_stiffness(dir_name: str) -> Optional[int]:
    """
    Parse CIRS stiffness (kPa) from directory name.
    
    Expected format: "Data set N(XX)" where XX is stiffness in kPa.
    Examples: "Data set 1(50)" -> 50, "Data set 2(10)" -> 10
    
    Returns:
        Stiffness in kPa as int, or None if parsing fails
    """
    import re
    match = re.search(r'\((\d+)\)', dir_name)
    if match:
        return int(match.group(1))
    return None


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    sparse_step: int = 20  # Take every Nth frame for sparse input
    magnitude_threshold: float = 0.01  # PERMISSIVE: Very low threshold to capture dim tissue
    soft_mask: bool = True  # Use soft (continuous) mask vs hard (binary)
    clutter_filter: bool = True  # Apply temporal mean subtraction
    normalize_velocity: bool = True  # Divide velocity by pi to get range [-1, 1]
    mask_dilation_iterations: int = 2  # Morphological dilation to grow the mask
    
    # Frequencies to include (Hz)
    include_frequencies: Tuple[str, ...] = ('500Hz', '1kHz', '5kHz', '10kHz', '20kHz')
    
    # Planes for training (validation uses the rest)
    train_planes: Tuple[str, ...] = ('Plane 1', 'Plane 2', 'Plane 3', 'Plane 4')
    val_planes: Tuple[str, ...] = ('Plane 5',)
    
    # CIRS stiffness split (kPa): train on 10/60/100, validate on 50
    cirs_train_stiffness: Tuple[int, ...] = (10, 60, 100)
    cirs_val_stiffness: Tuple[int, ...] = (50,)


def discover_data_files(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    Discover all IQ data files organized by split (train/val).
    
    Returns:
        Dictionary with 'train' and 'val' keys, each containing list of file info dicts
    """
    files = {'train': [], 'val': []}
    
    # Discover 2FT and 3FT data
    for ft_type in ['2FT(na100)', '3FT(na100)']:
        ft_dir = data_dir / ft_type
        if not ft_dir.exists():
            logger.warning(f"Directory not found: {ft_dir}")
            continue
        
        for focus_dir in ft_dir.iterdir():
            if not focus_dir.is_dir():
                continue
            
            for freq_dir in focus_dir.iterdir():
                if not freq_dir.is_dir():
                    continue
                
                freq_name = freq_dir.name
                
                for plane_dir in freq_dir.iterdir():
                    if not plane_dir.is_dir():
                        continue
                    
                    plane_name = plane_dir.name
                    is_train = plane_name in PreprocessingConfig.train_planes
                    split = 'train' if is_train else 'val'
                    
                    for npy_file in plane_dir.glob('*.npy'):
                        files[split].append({
                            'path': npy_file,
                            'dataset': ft_type,
                            'focus': focus_dir.name,
                            'frequency': freq_name,
                            'plane': plane_name,
                            'name': npy_file.stem,
                        })
    
    # Discover CIRS data - split by stiffness
    # Train on 10, 60, 100 kPa (ideal Kelvin-Voigt physics)
    # Validate on 50 kPa (out-of-distribution test)
    cirs_dir = data_dir / 'CIRS_Phantom_Latest'
    if cirs_dir.exists():
        for stiffness_dir in cirs_dir.iterdir():
            if not stiffness_dir.is_dir() or not stiffness_dir.name.startswith('Data set'):
                continue
            
            # Parse stiffness from directory name like "Data set 1(50)" -> 50 kPa
            stiffness_kpa = _parse_cirs_stiffness(stiffness_dir.name)
            if stiffness_kpa is None:
                logger.warning(f"Could not parse stiffness from: {stiffness_dir.name}")
                continue
            
            # Determine split based on stiffness
            if stiffness_kpa in PreprocessingConfig.cirs_train_stiffness:
                split = 'train'
            elif stiffness_kpa in PreprocessingConfig.cirs_val_stiffness:
                split = 'val'
            else:
                logger.warning(f"Unknown CIRS stiffness {stiffness_kpa} kPa, skipping")
                continue
            
            for npy_file in stiffness_dir.rglob('*.npy'):
                files[split].append({
                    'path': npy_file,
                    'dataset': 'CIRS',
                    'stiffness': stiffness_dir.name,
                    'stiffness_kpa': stiffness_kpa,
                    'name': npy_file.stem,
                })
    
    logger.info(f"Discovered files: {len(files['train'])} train, {len(files['val'])} val")
    return files


def compute_global_normalization(files: List[Dict], sample_fraction: float = 0.3) -> Dict:
    """
    Compute global normalization statistics from a sample of 2FT/3FT files.
    
    This ensures consistent normalization across all datasets.
    """
    logger.info("Computing global normalization statistics...")
    
    # Filter to 2FT/3FT files only (typically "louder" than CIRS)
    ft_files = [f for f in files if f['dataset'] in ['2FT(na100)', '3FT(na100)']]
    
    # Sample a fraction of files
    num_samples = max(5, int(len(ft_files) * sample_fraction))
    sample_indices = np.random.choice(len(ft_files), size=min(num_samples, len(ft_files)), replace=False)
    sample_files = [ft_files[i] for i in sample_indices]
    
    all_magnitudes = []
    
    for file_info in tqdm(sample_files, desc="Computing global stats"):
        try:
            data = np.load(file_info['path'])
            if not np.iscomplexobj(data):
                continue
            
            # Ensure (T, H, W) format
            if data.ndim == 3 and data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                data = np.transpose(data, (2, 0, 1))
            
            # Compute magnitude
            magnitude = np.abs(data)
            all_magnitudes.append(magnitude.max())
            
        except Exception as e:
            logger.warning(f"Error processing {file_info['path']}: {e}")
    
    global_max = np.percentile(all_magnitudes, 99)  # Use 99th percentile to be robust to outliers
    global_mean = np.mean(all_magnitudes)
    
    stats = {
        'global_max': float(global_max),
        'global_mean': float(global_mean),
        'num_files_sampled': len(sample_files),
    }
    
    logger.info(f"Global normalization: max={global_max:.4e}, mean={global_mean:.4e}")
    return stats


def load_iq_data(file_path: Path) -> np.ndarray:
    """Load IQ data and ensure (T, H, W) complex format."""
    data = np.load(file_path)
    
    # Handle different input formats
    if not np.iscomplexobj(data):
        if data.ndim == 4 and data.shape[1] == 2:
            # (T, 2, H, W) format - convert to complex
            data = data[:, 0, :, :] + 1j * data[:, 1, :, :]
        else:
            raise ValueError(f"Expected complex data, got shape {data.shape}")
    
    # Ensure (T, H, W) format
    if data.ndim == 3:
        if data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
            # (H, W, T) -> (T, H, W)
            data = np.transpose(data, (2, 0, 1))
    
    return data.astype(np.complex64)


def apply_clutter_filter(iq_data: np.ndarray) -> np.ndarray:
    """
    Apply clutter filter by subtracting temporal mean.
    
    This removes static tissue signal, preserving only the moving shear wave.
    """
    temporal_mean = iq_data.mean(axis=0, keepdims=True)
    return iq_data - temporal_mean


def compute_kasai_velocity(iq_data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute Kasai velocity from complex IQ data.
    
    Kasai: R = IQ(t) * conj(IQ(t-1))
    Velocity = angle(R), range [-pi, pi] (or [-1, 1] if normalized)
    
    Args:
        iq_data: Complex IQ data (T, H, W)
        normalize: If True, divide by pi to get range [-1, 1]
    
    Returns: (T-1, H, W) velocity array
    """
    # Consecutive frame pairs
    z_t = iq_data[:-1]   # IQ(t-1)
    z_t1 = iq_data[1:]   # IQ(t)
    
    # Autocorrelation
    R = z_t1 * np.conj(z_t)
    
    # Velocity is the angle
    velocity = np.angle(R)
    
    # Normalize to [-1, 1] for training
    if normalize:
        velocity = velocity / np.pi
    
    return velocity.astype(np.float32)


def compute_magnitude_mask(
    iq_data: np.ndarray, 
    threshold: float = 0.01,  # PERMISSIVE: Low threshold to include dim tissue
    soft: bool = True,
    dilation_iterations: int = 2  # Grow mask to catch neighboring pixels
) -> np.ndarray:
    """
    Compute PERMISSIVE magnitude mask from ORIGINAL (pre-filtered) IQ data.
    
    Philosophy: "Broadly Include Potential Signal"
    - It's safer to train on Signal + A Little Noise than on Half a Signal
    - The neural network can learn to average out zero-mean noise
    - But it cannot learn a signal it is forbidden to see
    
    Key Changes from Original:
    1. VERY LOW threshold (0.01) - only mask absolute electronic noise floor
    2. Morphological DILATION - grow mask by ~2 pixels to catch dim edges
    3. Minimum weight (0.2) for dilated regions - ensure dim wave is not zeroed
    
    Args:
        iq_data: Original complex IQ data (T, H, W)
        threshold: Fraction of sample max below which to mask out (default: 0.01 = 1%)
        soft: Use soft (continuous) mask vs hard (binary)
        dilation_iterations: How many pixels to grow the mask
    
    Returns:
        Magnitude mask (T-1, H, W) - one less than input due to Kasai
    """
    # 1. Compute Magnitude from original IQ (NOT filtered)
    magnitude = np.abs(iq_data)
    
    # 2. Per-Sample Normalization (ensures threshold works across datasets)
    sample_max = magnitude.max()
    if sample_max > 0:
        magnitude_norm = magnitude / sample_max
    else:
        magnitude_norm = np.zeros_like(magnitude)
    
    # 3. Average consecutive frames (to match velocity shape T-1)
    mag_pairs = (magnitude_norm[:-1] + magnitude_norm[1:]) / 2
    
    # 4. Create binary "valid region" and DILATE to catch dim edges
    binary_valid = mag_pairs > threshold
    # Dilate each time frame independently
    dilated_valid = np.zeros_like(binary_valid, dtype=bool)
    for t in range(binary_valid.shape[0]):
        dilated_valid[t] = binary_dilation(binary_valid[t], iterations=dilation_iterations)
    
    if soft:
        # Soft mask with smooth gradients
        mask = np.clip((mag_pairs - threshold) / (1 - threshold), 0, 1)
        
        # KEY: Ensure dilated regions have at least a minimum weight (0.2)
        # This prevents the dim shear wave from being multiplied by zero
        mask[(dilated_valid) & (mask < 0.2)] = 0.2
        
        # Hard zero for background (outside dilated region)
        mask[~dilated_valid] = 0.0
    else:
        # Hard mask: just use the dilated binary mask
        mask = dilated_valid.astype(np.float32)
    
    return mask.astype(np.float32)


def create_sparse_frames(iq_data: np.ndarray, step: int = 20) -> Tuple[np.ndarray, int]:
    """
    Create sparse frame sequence by taking every Nth frame.
    
    Returns:
        sparse_iq: (T_sparse, H, W) complex array
        dense_frames: Number of dense frames that will be generated
    """
    sparse_indices = np.arange(0, iq_data.shape[0], step)
    sparse_iq = iq_data[sparse_indices]
    
    # Dense frames = (T_sparse - 1) * step + 1, but velocity has T-1
    T_sparse = len(sparse_indices)
    dense_frames = (T_sparse - 1) * step + 1
    
    return sparse_iq, dense_frames


def preprocess_single_file(
    file_info: Dict,
    output_dir: Path,
    global_max: float,
    config: PreprocessingConfig,
) -> Optional[Dict]:
    """
    Preprocess a single IQ file.
    
    Returns metadata dict on success, None on failure.
    """
    try:
        # Load original IQ data
        iq_data = load_iq_data(file_info['path'])
        T, H, W = iq_data.shape
        
        # Compute PERMISSIVE magnitude mask from ORIGINAL data (before filtering)
        # Uses low threshold + dilation to broadly include potential signal
        magnitude_mask = compute_magnitude_mask(
            iq_data, 
            threshold=config.magnitude_threshold,
            soft=config.soft_mask,
            dilation_iterations=config.mask_dilation_iterations
        )
        
        # Apply clutter filter
        if config.clutter_filter:
            iq_filtered = apply_clutter_filter(iq_data)
        else:
            iq_filtered = iq_data
        
        # Compute Kasai velocity from filtered data (normalized to [-1, 1])
        velocity = compute_kasai_velocity(iq_filtered, normalize=config.normalize_velocity)
        
        # Create sparse frames from filtered IQ
        sparse_iq, dense_T = create_sparse_frames(iq_filtered, config.sparse_step)
        
        # Ensure velocity and mask match dense_T - 1
        # (Kasai produces T-1 frames from T frames)
        expected_frames = dense_T - 1
        if velocity.shape[0] > expected_frames:
            velocity = velocity[:expected_frames]
            magnitude_mask = magnitude_mask[:expected_frames]
        
        # Generate unique name
        name = file_info['name']
        
        # Save outputs
        np.save(output_dir / f'sparse_{name}.npy', sparse_iq)
        np.save(output_dir / f'velocity_{name}.npy', velocity)
        np.save(output_dir / f'magnitude_{name}.npy', magnitude_mask)
        
        # Return metadata
        return {
            'name': name,
            'original_path': str(file_info['path']),
            'dataset': file_info.get('dataset', 'unknown'),
            'T_sparse': sparse_iq.shape[0],
            'T_dense': velocity.shape[0],
            'H': H,
            'W': W,
            'velocity_range': [float(velocity.min()), float(velocity.max())],
            'signal_fraction': float((magnitude_mask > 0.1).mean()),
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_info['path']}: {e}")
        return None


def preprocess_all_data(
    data_dir: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    visualize: bool = False,
):
    """
    Run full preprocessing pipeline on all data.
    """
    # Discover all files
    files = discover_data_files(data_dir)
    
    if not files['train'] and not files['val']:
        logger.error("No data files found!")
        return
    
    # Compute global normalization from training data
    global_stats = compute_global_normalization(files['train'])
    global_max = global_stats['global_max']
    
    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    metadata = {
        'config': asdict(config),
        'global_stats': global_stats,
        'train': [],
        'val': [],
    }
    
    # Process training data
    logger.info(f"Processing {len(files['train'])} training files...")
    for file_info in tqdm(files['train'], desc="Training data"):
        result = preprocess_single_file(file_info, train_dir, global_max, config)
        if result:
            metadata['train'].append(result)
    
    # Process validation data
    logger.info(f"Processing {len(files['val'])} validation files...")
    for file_info in tqdm(files['val'], desc="Validation data"):
        result = preprocess_single_file(file_info, val_dir, global_max, config)
        if result:
            metadata['val'].append(result)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Training samples: {len(metadata['train'])}")
    logger.info(f"  Validation samples: {len(metadata['val'])}")
    logger.info(f"  Metadata saved to: {metadata_path}")
    
    # Print summary statistics
    if metadata['train']:
        signal_fractions = [m['signal_fraction'] for m in metadata['train']]
        logger.info(f"  Avg signal fraction (train): {np.mean(signal_fractions):.1%}")
    
    if visualize:
        visualize_samples(output_dir, metadata)
    
    return metadata


def visualize_samples(output_dir: Path, metadata: Dict, num_samples: int = 3):
    """Visualize a few preprocessed samples."""
    import matplotlib.pyplot as plt
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val']:
        samples = metadata[split][:num_samples]
        split_dir = output_dir / split
        
        for sample in samples:
            name = sample['name']
            
            # Load data
            sparse = np.load(split_dir / f'sparse_{name}.npy')
            velocity = np.load(split_dir / f'velocity_{name}.npy')
            magnitude = np.load(split_dir / f'magnitude_{name}.npy')
            
            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"{split.upper()}: {name}\nShape: sparse={sparse.shape}, velocity={velocity.shape}", fontsize=12)
            
            # Top row: Sparse IQ magnitude (4 frames)
            T_sparse = sparse.shape[0]
            indices = np.linspace(0, T_sparse - 1, 4, dtype=int)
            for i, idx in enumerate(indices):
                axes[0, i].imshow(np.abs(sparse[idx]), cmap='gray', aspect='auto')
                axes[0, i].set_title(f'Sparse IQ Mag (t={idx})')
                axes[0, i].axis('off')
            
            # Bottom row: Velocity and magnitude mask
            T_vel = velocity.shape[0]
            vel_indices = np.linspace(0, T_vel - 1, 3, dtype=int)
            for i, idx in enumerate(vel_indices):
                # Apply mask to velocity for cleaner visualization
                masked_vel = velocity[idx] * magnitude[idx]
                # Velocity is normalized to [-1, 1] (was divided by pi)
                im = axes[1, i].imshow(masked_vel, cmap='seismic', vmin=-1, vmax=1, aspect='auto')
                axes[1, i].set_title(f'Masked Velocity (t={idx})')
                axes[1, i].axis('off')
            
            # Last panel: Magnitude mask
            mid_idx = T_vel // 2
            axes[1, 3].imshow(magnitude[mid_idx], cmap='hot', vmin=0, vmax=1, aspect='auto')
            axes[1, 3].set_title(f'Magnitude Mask (t={mid_idx})')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f'{split}_{name}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Visualizations saved to: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IQ data for velocity estimation training"
    )
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='Root data directory containing 2FT, 3FT, CIRS folders'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/preprocessed',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--sparse_step', type=int, default=20,
        help='Step size for sparse frame selection'
    )
    parser.add_argument(
        '--magnitude_threshold', type=float, default=0.01,
        help='PERMISSIVE: Threshold for magnitude mask (fraction of sample max). Low value to capture dim tissue.'
    )
    parser.add_argument(
        '--mask_dilation', type=int, default=2,
        help='Morphological dilation iterations to grow the mask'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization of preprocessed samples'
    )
    parser.add_argument(
        '--no_clutter_filter', action='store_true',
        help='Disable clutter filtering'
    )
    
    args = parser.parse_args()
    
    config = PreprocessingConfig(
        sparse_step=args.sparse_step,
        magnitude_threshold=args.magnitude_threshold,
        mask_dilation_iterations=args.mask_dilation,
        clutter_filter=not args.no_clutter_filter,
    )
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    preprocess_all_data(data_dir, output_dir, config, visualize=args.visualize)


if __name__ == '__main__':
    main()
