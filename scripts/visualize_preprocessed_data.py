#!/usr/bin/env python
"""
Visualize preprocessed velocity data to verify quality.

This script shows:
1. Sparse IQ magnitude
2. Raw Kasai velocity (before masking)
3. Magnitude mask
4. Masked velocity (training target)

Usage:
    python scripts/visualize_preprocessed_data.py --data_dir data/preprocessed --num_samples 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Fix project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_sample(sparse_path: Path, velocity_path: Path, magnitude_path: Path):
    """Load a single preprocessed sample."""
    sparse = np.load(sparse_path)      # Complex (T, H, W)
    velocity = np.load(velocity_path)  # Real (T, H, W)
    magnitude = np.load(magnitude_path)  # Real (H, W)
    return sparse, velocity, magnitude


def visualize_sample(sparse, velocity, magnitude, sample_name, output_dir, frame_idx=None):
    """
    Visualize a single sample with multiple views.
    
    Note: sparse has T_sparse frames, velocity/magnitude have T_dense frames
    
    Shows:
    - IQ Magnitude (temporal average)
    - Raw velocity (selected frame)
    - Magnitude mask
    - Masked velocity (training target)
    """
    T_sparse = sparse.shape[0]
    T_dense = velocity.shape[0]
    
    if frame_idx is None:
        frame_idx = T_dense // 2  # Middle frame of dense
    
    # Compute IQ magnitude (temporal mean for visibility)
    iq_magnitude = np.abs(sparse).mean(axis=0)
    
    # Get velocity frame
    vel_frame = velocity[frame_idx]
    
    # Handle both 2D and 3D magnitude masks
    if magnitude.ndim == 3:
        mag_frame = magnitude[frame_idx]  # Temporal mask
        mag_mean = magnitude.mean(axis=0)  # Time-averaged mask for display
    else:
        mag_frame = magnitude
        mag_mean = magnitude
    
    # Apply mask for visualization
    masked_velocity = vel_frame * mag_frame
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. IQ Magnitude (log scale for visibility)
    ax = axes[0, 0]
    im = ax.imshow(np.log10(iq_magnitude + 1), cmap='viridis', aspect='auto')
    ax.set_title(f'IQ Magnitude (log10)\nSparse: {sparse.shape}')
    ax.set_xlabel('Lateral (pixels)')
    ax.set_ylabel('Axial (pixels)')
    plt.colorbar(im, ax=ax, label='log10(mag)')
    
    # 2. Raw Kasai velocity
    ax = axes[0, 1]
    vmax = np.abs(vel_frame).max()
    im = ax.imshow(vel_frame, cmap='RdBu_r', aspect='auto', 
                   norm=Normalize(vmin=-vmax, vmax=vmax))
    ax.set_title(f'Raw Kasai Velocity\nDense: {velocity.shape}, frame {frame_idx}')
    ax.set_xlabel('Lateral (pixels)')
    ax.set_ylabel('Axial (pixels)')
    plt.colorbar(im, ax=ax, label='Velocity (rad/sample)')
    
    # 3. Magnitude mask (mean over time if temporal)
    ax = axes[0, 2]
    im = ax.imshow(mag_mean, cmap='gray', aspect='auto')
    ax.set_title(f'Magnitude Mask (mean)\nSignal fraction: {mag_mean.mean():.3f}')
    ax.set_xlabel('Lateral (pixels)')
    ax.set_ylabel('Axial (pixels)')
    plt.colorbar(im, ax=ax, label='Mask')
    
    # 4. Masked velocity (training target)
    ax = axes[1, 0]
    im = ax.imshow(masked_velocity, cmap='RdBu_r', aspect='auto',
                   norm=Normalize(vmin=-vmax, vmax=vmax))
    ax.set_title('Masked Velocity (Training Target)')
    ax.set_xlabel('Lateral (pixels)')
    ax.set_ylabel('Axial (pixels)')
    plt.colorbar(im, ax=ax, label='Velocity (rad/sample)')
    
    # 5. Velocity time profile (center line)
    ax = axes[1, 1]
    center_x = velocity.shape[2] // 2
    # Find a row with signal
    signal_rows = mag_mean[:, center_x] > 0.5
    if signal_rows.any():
        center_y = np.where(signal_rows)[0][len(np.where(signal_rows)[0])//2]
    else:
        center_y = velocity.shape[1] // 2
    
    time_profile = velocity[:, center_y, center_x]
    time = np.arange(T_dense)
    ax.plot(time, time_profile, 'b-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Velocity (rad/sample)')
    ax.set_title(f'Velocity Time Profile\n(y={center_y}, x={center_x})')
    ax.grid(True, alpha=0.3)
    
    # 6. Histogram of masked velocity values
    ax = axes[1, 2]
    masked_vals = vel_frame[mag_frame > 0.5].flatten()
    if len(masked_vals) > 0:
        ax.hist(masked_vals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Velocity (rad/sample)')
    ax.set_ylabel('Count')
    ax.set_title(f'Velocity Distribution (masked region)\nN={len(masked_vals)} pixels')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Sample: {sample_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{sample_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def visualize_split_comparison(train_dir: Path, val_dir: Path, output_dir: Path, metadata: dict):
    """
    Create a comparison figure showing example samples from train and validation sets.
    """
    # Get one sample from each split
    train_files = sorted(train_dir.glob('sparse_*.npy'))[:3]
    val_files = sorted(val_dir.glob('sparse_*.npy'))[:3]
    
    if not train_files or not val_files:
        print("Not enough files for split comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    cmap = 'RdBu_r'
    
    for idx, sparse_file in enumerate(train_files):
        sample_id = sparse_file.stem.replace('sparse_', '')
        velocity = np.load(train_dir / f'velocity_{sample_id}.npy')
        magnitude = np.load(train_dir / f'magnitude_{sample_id}.npy')
        
        frame = velocity.shape[0] // 2
        # Handle both 2D and 3D magnitude
        mag_frame = magnitude[frame] if magnitude.ndim == 3 else magnitude
        masked_vel = velocity[frame] * mag_frame
        
        ax = axes[0, idx]
        vmax = np.abs(velocity[frame]).max()
        im = ax.imshow(masked_vel, cmap=cmap, aspect='auto',
                       norm=Normalize(vmin=-vmax, vmax=vmax))
        ax.set_title(f'Train: {sample_id[:30]}...')
        if idx == 0:
            ax.set_ylabel('TRAIN SET\n(Planes 1-4)')
        plt.colorbar(im, ax=ax)
    
    for idx, sparse_file in enumerate(val_files):
        sample_id = sparse_file.stem.replace('sparse_', '')
        velocity = np.load(val_dir / f'velocity_{sample_id}.npy')
        magnitude = np.load(val_dir / f'magnitude_{sample_id}.npy')
        
        frame = velocity.shape[0] // 2
        # Handle both 2D and 3D magnitude
        mag_frame = magnitude[frame] if magnitude.ndim == 3 else magnitude
        masked_vel = velocity[frame] * mag_frame
        
        ax = axes[1, idx]
        vmax = np.abs(velocity[frame]).max()
        im = ax.imshow(masked_vel, cmap=cmap, aspect='auto',
                       norm=Normalize(vmin=-vmax, vmax=vmax))
        ax.set_title(f'Val: {sample_id[:30]}...')
        if idx == 0:
            ax.set_ylabel('VALIDATION SET\n(Plane 5 + CIRS)')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Physics-Based Train/Val Split\n'
                 f'Train: {metadata.get("train_count", "?")} | Val: {metadata.get("val_count", "?")} | '
                 f'Global Max: {metadata.get("global_max", 0):.2e}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'split_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize preprocessed velocity data')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                       help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='results/preprocessed_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize from each split')
    parser.add_argument('--split', type=str, default='both', choices=['train', 'val', 'both'],
                       help='Which split to visualize')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Loaded metadata:")
        global_stats = metadata.get('global_stats', {})
        global_max = global_stats.get('global_max', 'N/A')
        if isinstance(global_max, (int, float)):
            print(f"  Global max: {global_max:.4e}")
        else:
            print(f"  Global max: {global_max}")
        train_samples = metadata.get('train', [])
        val_samples = metadata.get('val', [])
        print(f"  Train count: {len(train_samples)}")
        print(f"  Val count: {len(val_samples)}")
        # Pass counts for use later
        metadata['train_count'] = len(train_samples)
        metadata['val_count'] = len(val_samples)
        metadata['global_max'] = global_max
    else:
        metadata = {}
        print("Warning: No metadata.json found")
    
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # Visualize train samples
    if args.split in ['train', 'both'] and train_dir.exists():
        train_output = output_dir / 'train'
        train_output.mkdir(parents=True, exist_ok=True)
        
        train_files = sorted(train_dir.glob('sparse_*.npy'))[:args.num_samples]
        print(f"\nVisualizing {len(train_files)} train samples...")
        
        for sparse_file in train_files:
            sample_id = sparse_file.stem.replace('sparse_', '')
            velocity_file = train_dir / f'velocity_{sample_id}.npy'
            magnitude_file = train_dir / f'magnitude_{sample_id}.npy'
            
            if velocity_file.exists() and magnitude_file.exists():
                sparse, velocity, magnitude = load_sample(sparse_file, velocity_file, magnitude_file)
                visualize_sample(sparse, velocity, magnitude, sample_id, train_output)
    
    # Visualize val samples
    if args.split in ['val', 'both'] and val_dir.exists():
        val_output = output_dir / 'val'
        val_output.mkdir(parents=True, exist_ok=True)
        
        val_files = sorted(val_dir.glob('sparse_*.npy'))[:args.num_samples]
        print(f"\nVisualizing {len(val_files)} validation samples...")
        
        for sparse_file in val_files:
            sample_id = sparse_file.stem.replace('sparse_', '')
            velocity_file = val_dir / f'velocity_{sample_id}.npy'
            magnitude_file = val_dir / f'magnitude_{sample_id}.npy'
            
            if velocity_file.exists() and magnitude_file.exists():
                sparse, velocity, magnitude = load_sample(sparse_file, velocity_file, magnitude_file)
                visualize_sample(sparse, velocity, magnitude, sample_id, val_output)
    
    # Create split comparison figure
    if train_dir.exists() and val_dir.exists():
        print("\nCreating split comparison figure...")
        visualize_split_comparison(train_dir, val_dir, output_dir, metadata)
    
    print(f"\nVisualization complete! Check: {output_dir}")


if __name__ == '__main__':
    main()
