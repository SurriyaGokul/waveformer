#!/usr/bin/env python3
"""
Visualize Training Targets - Exact Data Fed to the Model.

This script visualizes the EXACT data that the training script uses:
1. Input: Sparse IQ data (2, T_sparse, H, W) - Real/Imag channels
2. Target: Velocity (1, T_dense, H, W) - Normalized to [-1, 1]
3. Mask: Magnitude mask (1, T_dense, H, W) - Values in [0, 1]

It also shows:
- Mask coverage (mean of mask values)
- Weighted target: velocity * mask (what the loss actually focuses on)
- Histogram of mask values

This ensures there is NO DISCREPANCY between what is visualized and what 
the model actually trains on.

Usage:
    python scripts/visualize_training_targets.py
    python scripts/visualize_training_targets.py --num_samples 5
    python scripts/visualize_training_targets.py --output_dir results/training_target_viz
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data_as_model_sees_it(
    preprocessed_dir: Path,
    sample_idx: int = 0,
    split: str = 'train'
):
    """
    Load data EXACTLY as the PreprocessedVelocityDataset does.
    
    This mirrors the code in src/utils/velocity_dataset.py PreprocessedVelocityDataset.__getitem__
    """
    from src.utils.velocity_dataset import PreprocessedVelocityDataset
    
    # Create dataset exactly as training script does
    split_dir = preprocessed_dir / split
    
    # Load metadata for global_max
    metadata_path = preprocessed_dir / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    global_max = metadata['global_stats']['global_max']
    
    dataset = PreprocessedVelocityDataset(
        preprocessed_dir=split_dir,
        global_max=global_max,
        return_mask=True,
    )
    
    # Get sample info
    sample_info = dataset.get_sample_info(sample_idx)
    
    # Load EXACTLY as __getitem__ does
    input_tensor, target_tensor, mask_tensor = dataset[sample_idx]
    
    return input_tensor, target_tensor, mask_tensor, sample_info, metadata


def visualize_single_sample(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor, 
    mask_tensor: torch.Tensor,
    sample_info: dict,
    metadata: dict,
    output_path: Path,
    time_indices: list = None,
):
    """
    Create comprehensive visualization for a single sample.
    
    Shows:
    1. Input IQ magnitude at different sparse time points
    2. Target velocity at dense time points
    3. Magnitude mask at dense time points
    4. Weighted target: velocity * mask
    5. Statistics and histograms
    """
    # Convert to numpy
    input_np = input_tensor.numpy()  # (2, T_sparse, H, W)
    target_np = target_tensor.numpy()  # (1, T_dense, H, W)
    mask_np = mask_tensor.numpy()  # (1, T_dense, H, W)
    
    T_sparse = input_np.shape[1]
    T_dense = target_np.shape[1]
    H, W = target_np.shape[2], target_np.shape[3]
    
    # Compute mask coverage (exactly as in loss function)
    mask_coverage = mask_np.mean()
    
    # Default time indices
    if time_indices is None:
        # Show start, middle, end frames
        time_indices = [0, T_dense // 4, T_dense // 2, 3 * T_dense // 4, min(T_dense - 1, T_dense - 1)]
        time_indices = sorted(list(set(time_indices)))  # Remove duplicates
    
    # Compute weighted target (what loss actually focuses on)
    weighted_target = target_np * mask_np
    
    # Create figure with many subplots
    n_times = len(time_indices)
    fig = plt.figure(figsize=(5 * n_times, 20))
    
    gs = fig.add_gridspec(5, n_times + 1, width_ratios=[1] * n_times + [0.05])
    
    # Row titles
    row_titles = [
        'Input IQ Magnitude (normalized)',
        'Target Velocity [-1, 1]',
        'Magnitude Mask [0, 1]',
        'Weighted Target (vel × mask)',
        'Mask Binary (mask > 0.1)'
    ]
    
    # Colormaps and normalization
    iq_norm = Normalize(vmin=0, vmax=input_np.max())
    vel_norm = Normalize(vmin=-1, vmax=1)
    mask_norm = Normalize(vmin=0, vmax=1)
    
    # Plot each row
    for row in range(5):
        for col, t_idx in enumerate(time_indices):
            ax = fig.add_subplot(gs[row, col])
            
            if row == 0:  # Input IQ magnitude (from sparse frames, need to map)
                sparse_idx = min(t_idx // 20, T_sparse - 1)  # Map dense to sparse
                real = input_np[0, sparse_idx]
                imag = input_np[1, sparse_idx]
                iq_mag = np.sqrt(real**2 + imag**2)
                im = ax.imshow(iq_mag.T, cmap='magma', norm=iq_norm, origin='lower', aspect='auto')
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)
                ax.set_title(f'Sparse t={sparse_idx}', fontsize=9)
            elif row == 1:  # Target velocity
                vel = target_np[0, t_idx]
                im = ax.imshow(vel.T, cmap='RdBu_r', norm=vel_norm, origin='lower', aspect='auto')
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)
                ax.set_title(f'Dense t={t_idx}', fontsize=9)
            elif row == 2:  # Magnitude mask
                mask = mask_np[0, t_idx]
                im = ax.imshow(mask.T, cmap='viridis', norm=mask_norm, origin='lower', aspect='auto')
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)
                ax.set_title(f't={t_idx}, mean={mask.mean():.3f}', fontsize=9)
            elif row == 3:  # Weighted target
                wt = weighted_target[0, t_idx]
                im = ax.imshow(wt.T, cmap='RdBu_r', norm=vel_norm, origin='lower', aspect='auto')
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)
                ax.set_title(f't={t_idx}', fontsize=9)
            elif row == 4:  # Binary mask (> 0.1)
                mask_binary = (mask_np[0, t_idx] > 0.1).astype(float)
                im = ax.imshow(mask_binary.T, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='auto')
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)
                coverage = mask_binary.mean()
                ax.set_title(f't={t_idx}, coverage={coverage:.2%}', fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Colorbar
        cax = fig.add_subplot(gs[row, -1])
        if row == 0:
            plt.colorbar(im, cax=cax, label='Magnitude')
        elif row in [1, 3]:
            plt.colorbar(im, cax=cax, label='Velocity')
        elif row == 2:
            plt.colorbar(im, cax=cax, label='Mask')
        else:
            plt.colorbar(im, cax=cax, label='Binary')
    
    # Add title with statistics
    config = metadata['config']
    title = (
        f"Sample: {sample_info['base_name']}\n"
        f"Shape: Input (2, {T_sparse}, {H}, {W}), Target/Mask (1, {T_dense}, {H}, {W})\n"
        f"Mask Coverage (mean): {mask_coverage:.4f} = {mask_coverage*100:.2f}%\n"
        f"Preprocessing: threshold={config['magnitude_threshold']}, soft_mask={config['soft_mask']}, clutter_filter={config['clutter_filter']}"
    )
    fig.suptitle(title, fontsize=12, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")
    return mask_coverage


def visualize_mask_statistics(
    preprocessed_dir: Path,
    output_path: Path,
    split: str = 'train',
    max_samples: int = 50,
):
    """
    Visualize mask statistics across all samples.
    """
    from src.utils.velocity_dataset import PreprocessedVelocityDataset
    
    split_dir = preprocessed_dir / split
    
    # Load metadata for global_max
    metadata_path = preprocessed_dir / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    global_max = metadata['global_stats']['global_max']
    
    dataset = PreprocessedVelocityDataset(
        preprocessed_dir=split_dir,
        global_max=global_max,
        return_mask=True,
    )
    
    num_samples = min(len(dataset), max_samples)
    
    # Collect statistics
    mask_coverages = []
    mask_means_per_frame = []
    all_mask_values = []
    
    logger.info(f"Analyzing {num_samples} samples...")
    
    for idx in range(num_samples):
        _, target, mask = dataset[idx]
        mask_np = mask.numpy()
        
        mask_coverage = mask_np.mean()
        mask_coverages.append(mask_coverage)
        
        # Per-frame means
        for t in range(mask_np.shape[1]):
            mask_means_per_frame.append(mask_np[0, t].mean())
        
        # Sample of mask values for histogram
        flat = mask_np.flatten()
        indices = np.random.choice(len(flat), min(10000, len(flat)), replace=False)
        all_mask_values.extend(flat[indices])
    
    mask_coverages = np.array(mask_coverages)
    all_mask_values = np.array(all_mask_values)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of mask coverage per sample
    ax = axes[0, 0]
    ax.hist(mask_coverages, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mask_coverages.mean(), color='red', linestyle='--', 
               label=f'Mean: {mask_coverages.mean():.4f}')
    ax.set_xlabel('Mask Coverage (mean mask value)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Mask Coverage Across Samples')
    ax.legend()
    
    # 2. Histogram of all mask values
    ax = axes[0, 1]
    ax.hist(all_mask_values, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mask Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of All Mask Values\n(Sampled from all frames)')
    
    # Count values in different ranges
    zero_count = (all_mask_values < 0.01).sum() / len(all_mask_values)
    low_count = ((all_mask_values >= 0.01) & (all_mask_values < 0.5)).sum() / len(all_mask_values)
    high_count = (all_mask_values >= 0.5).sum() / len(all_mask_values)
    ax.text(0.95, 0.95, 
            f'Near 0: {zero_count:.1%}\n0.01-0.5: {low_count:.1%}\n>0.5: {high_count:.1%}',
            transform=ax.transAxes, va='top', ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Box plot of mask coverage
    ax = axes[1, 0]
    bp = ax.boxplot([mask_coverages], labels=['Mask Coverage'])
    ax.set_ylabel('Mask Coverage')
    ax.set_title(f'Mask Coverage Statistics\nMin: {mask_coverages.min():.4f}, Max: {mask_coverages.max():.4f}')
    
    # 4. Text summary
    ax = axes[1, 1]
    ax.axis('off')
    
    config = metadata['config']
    
    summary = f"""
TRAINING DATA SUMMARY
=====================

Preprocessing Configuration:
  - Magnitude threshold: {config['magnitude_threshold']}
  - Soft mask: {config['soft_mask']}
  - Clutter filter: {config['clutter_filter']}
  - Sparse step: {config['sparse_step']}
  - Velocity normalized: {config['normalize_velocity']}

Mask Statistics ({num_samples} samples, {split} split):
  - Mean mask coverage: {mask_coverages.mean():.4f} ({mask_coverages.mean()*100:.2f}%)
  - Std mask coverage: {mask_coverages.std():.4f}
  - Min mask coverage: {mask_coverages.min():.4f}
  - Max mask coverage: {mask_coverages.max():.4f}

What this means:
  - mask_coverage = mean(mask) = {mask_coverages.mean():.4f}
  - This is the average WEIGHT applied to each pixel
  - Most pixels have mask ≈ 0 (background/noise)
  - Signal regions have mask > 0 (up to 1.0)
  
  - The loss function normalizes by sum(mask),
    so even low coverage works correctly!
  
IMPORTANT: Low mask_coverage is EXPECTED because:
  1. Most of the image is background/noise
  2. The shear wave only occupies a small region
  3. The mask correctly focuses on signal regions
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', ha='left', family='monospace')
    
    plt.suptitle(f'Mask Statistics Analysis - {split.upper()} Split', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved mask statistics to {output_path}")
    
    return {
        'mean_coverage': float(mask_coverages.mean()),
        'std_coverage': float(mask_coverages.std()),
        'min_coverage': float(mask_coverages.min()),
        'max_coverage': float(mask_coverages.max()),
    }


def visualize_mask_on_velocity(
    preprocessed_dir: Path,
    output_dir: Path,
    sample_idx: int = 0,
    split: str = 'train',
):
    """
    Visualize how the mask affects the velocity target.
    
    Shows side by side:
    - Raw velocity (what we're predicting)
    - Mask (what weights the loss)
    - Velocity where mask > threshold (the signal we care about)
    """
    input_tensor, target_tensor, mask_tensor, sample_info, metadata = \
        load_training_data_as_model_sees_it(preprocessed_dir, sample_idx, split)
    
    target_np = target_tensor.numpy()[0]  # (T, H, W)
    mask_np = mask_tensor.numpy()[0]  # (T, H, W)
    
    T = target_np.shape[0]
    
    # Create figure showing time evolution
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    
    time_indices = np.linspace(0, T - 1, 8, dtype=int)
    
    for col, t in enumerate(time_indices):
        # Row 0: Raw velocity
        ax = axes[0, col]
        im = ax.imshow(target_np[t].T, cmap='RdBu_r', vmin=-1, vmax=1, 
                       origin='lower', aspect='auto')
        ax.set_title(f't={t}', fontsize=9)
        if col == 0:
            ax.set_ylabel('Raw Velocity', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row 1: Mask
        ax = axes[1, col]
        im2 = ax.imshow(mask_np[t].T, cmap='viridis', vmin=0, vmax=1,
                        origin='lower', aspect='auto')
        if col == 0:
            ax.set_ylabel('Mask', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row 2: Velocity * Mask (what loss sees)
        ax = axes[2, col]
        weighted = target_np[t] * mask_np[t]
        im3 = ax.imshow(weighted.T, cmap='RdBu_r', vmin=-1, vmax=1,
                        origin='lower', aspect='auto')
        if col == 0:
            ax.set_ylabel('Velocity × Mask', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row 3: Masked velocity (mask > 0.1)
        ax = axes[3, col]
        masked_vel = np.where(mask_np[t] > 0.1, target_np[t], np.nan)
        im4 = ax.imshow(masked_vel.T, cmap='RdBu_r', vmin=-1, vmax=1,
                        origin='lower', aspect='auto')
        if col == 0:
            ax.set_ylabel('Velocity (mask>0.1)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbars
    fig.colorbar(im, ax=axes[0, :], shrink=0.8, label='Velocity')
    fig.colorbar(im2, ax=axes[1, :], shrink=0.8, label='Mask')
    fig.colorbar(im3, ax=axes[2, :], shrink=0.8, label='Weighted Velocity')
    fig.colorbar(im4, ax=axes[3, :], shrink=0.8, label='Masked Velocity')
    
    mask_coverage = mask_np.mean()
    signal_pixels = (mask_np > 0.1).mean()
    
    title = (
        f"Sample: {sample_info['base_name']}\n"
        f"Mask Coverage (mean): {mask_coverage:.4f} | Signal Pixels (mask>0.1): {signal_pixels:.2%}\n"
        f"The loss is weighted by the mask - focusing on signal regions, ignoring noise"
    )
    fig.suptitle(title, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / f'mask_on_velocity_{sample_info["base_name"]}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved mask-on-velocity visualization to {output_path}")
    
    return mask_coverage, signal_pixels


def main():
    parser = argparse.ArgumentParser(description='Visualize exact training targets')
    parser.add_argument('--preprocessed_dir', type=str, default='data/preprocessed',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='results/training_target_viz',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Split to visualize')
    args = parser.parse_args()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TRAINING TARGET VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"This visualizes EXACTLY what the model trains on.")
    logger.info("=" * 60)
    
    # 1. Visualize mask statistics
    logger.info("\n1. Analyzing mask statistics across all samples...")
    stats = visualize_mask_statistics(
        preprocessed_dir, 
        output_dir / 'mask_statistics.png',
        split=args.split,
    )
    
    logger.info(f"\n   MASK COVERAGE SUMMARY:")
    logger.info(f"   Mean: {stats['mean_coverage']:.4f} ({stats['mean_coverage']*100:.2f}%)")
    logger.info(f"   This is the average weight applied per pixel")
    logger.info(f"   Low values are EXPECTED - most of image is background!")
    
    # 2. Visualize individual samples
    logger.info(f"\n2. Visualizing {args.num_samples} individual samples...")
    
    for idx in range(args.num_samples):
        try:
            input_t, target_t, mask_t, info, metadata = \
                load_training_data_as_model_sees_it(preprocessed_dir, idx, args.split)
            
            # Detailed visualization
            output_path = output_dir / f'sample_{idx}_{info["base_name"]}.png'
            coverage = visualize_single_sample(
                input_t, target_t, mask_t, info, metadata, output_path
            )
            
            # Mask-on-velocity visualization
            visualize_mask_on_velocity(preprocessed_dir, output_dir, idx, args.split)
            
            logger.info(f"   Sample {idx}: {info['base_name']}, mask_coverage={coverage:.4f}")
            
        except Exception as e:
            logger.error(f"   Error visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_dir}")
    logger.info("\nKey Files:")
    logger.info(f"  - {output_dir / 'mask_statistics.png'}: Overall mask analysis")
    logger.info(f"  - {output_dir / 'sample_*.png'}: Individual sample views")
    logger.info(f"  - {output_dir / 'mask_on_velocity_*.png'}: Mask effect on velocity")


if __name__ == '__main__':
    main()
