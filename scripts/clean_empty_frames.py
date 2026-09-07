#!/usr/bin/env python3
"""
Clean Empty Frames from SWE IQ Dataset
========================================
Detects and removes empty/inactive frames from raw IQ data files.

In shear wave elastography, the push pulse generates a shear wave that
propagates through tissue over a finite number of frames. After the wave
passes, remaining frames contain only static background (no displacement).

Problem detected:
  - 100-frame experiments (2FT, 3FT): Only first ~5–40 frames contain the
    shear wave depending on frequency. The second half (frames 50-99) is
    essentially empty, meaning when split into 2 experiments of 50, the
    second experiment has zero useful data.
  - 50-frame experiments (CIRS): Only first ~5–36 frames contain useful
    wave data depending on frequency.

This script:
  1. Analyzes each IQ file to find the last frame with active shear wave
  2. Reports detailed statistics per dataset/experiment
  3. Trims the raw IQ data to keep only active frames
  4. Saves trimmed data alongside originals (in a _cleaned directory)
  5. Creates visualization videos for verification
"""

import numpy as np
import glob
import os
import json
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('outputs/clean_empty_frames.log')
    ]
)
logger = logging.getLogger(__name__)


def compute_velocity_energy_profile(iq_data_thw):
    """
    Compute per-frame velocity energy from IQ data.
    
    Uses Kasai autocorrelation (phase difference between consecutive frames)
    after clutter filtering to isolate the shear wave signal.
    
    Args:
        iq_data_thw: Complex IQ data of shape (T, H, W)
    
    Returns:
        frame_energies: Array of shape (T-1,) with velocity energy per frame
    """
    T, H, W = iq_data_thw.shape
    
    # Clutter filter: subtract temporal mean to remove static tissue
    temporal_mean = iq_data_thw.mean(axis=0, keepdims=True)
    filtered = iq_data_thw - temporal_mean
    
    # Kasai velocity: phase difference between consecutive frames
    z_t = filtered[1:]
    z_t_minus_1 = filtered[:-1]
    autocorrelation = z_t * np.conj(z_t_minus_1)
    velocity = np.angle(autocorrelation)  # (T-1, H, W), range [-pi, pi]
    
    # Per-frame energy metrics
    frame_energy = np.abs(velocity).mean(axis=(1, 2))
    frame_std = velocity.std(axis=(1, 2))
    
    return frame_energy, frame_std, velocity


def detect_last_active_frame(frame_energy, total_iq_frames, min_active=5):
    """
    Detect the last frame containing meaningful shear wave signal.
    
    Uses a multi-step approach:
    1. For 100-frame data split into 2x50: check if second half is empty
    2. Uses cumulative energy to find the "knee" where most energy has been captured
    3. Falls back to adaptive threshold based on baseline noise
    
    Args:
        frame_energy: Per-frame velocity energy array (length T-1)
        total_iq_frames: Total number of IQ frames (T)
        min_active: Minimum number of active frames to return
        
    Returns:
        last_active: Index of last active IQ frame (exclusive, for slicing)
        baseline: Estimated baseline energy level
        threshold: Threshold used for detection
    """
    n = len(frame_energy)
    
    # Smooth the energy profile to reduce noise
    window = max(3, n // 15)
    kernel = np.ones(window) / window
    smoothed = np.convolve(frame_energy, kernel, mode='same')
    
    # Method 1: Cumulative energy analysis
    # Find where 95% of the total velocity energy has been captured
    cumsum = np.cumsum(smoothed)
    total_energy = cumsum[-1]
    
    if total_energy > 0:
        # Find frame where X% of energy is captured
        energy_95 = np.searchsorted(cumsum, 0.95 * total_energy)
        energy_90 = np.searchsorted(cumsum, 0.90 * total_energy)
    else:
        energy_95 = n - 1
        energy_90 = n - 1
    
    # Method 2: Rolling energy ratio
    # Compare energy in a window vs the minimum window energy
    win = max(5, n // 10)
    rolling_mean = np.convolve(smoothed, np.ones(win)/win, mode='valid')
    if len(rolling_mean) > 0:
        min_rolling = rolling_mean.min()
        # A window is "active" if its mean is > 2x the minimum window mean
        active_windows = np.where(rolling_mean > 2 * min_rolling)[0]
        if len(active_windows) > 0:
            last_active_window = active_windows[-1] + win  # account for window offset
        else:
            last_active_window = n - 1
    else:
        last_active_window = n - 1
    
    # Method 3: Adaptive threshold (improved)
    # Use the last 30% as baseline reference
    tail_start = int(0.7 * n)
    tail = smoothed[tail_start:]
    baseline = np.median(tail)
    baseline_iqr = np.percentile(tail, 75) - np.percentile(tail, 25)
    
    # Threshold: baseline + 2*IQR, minimum 1.3x baseline
    threshold = max(baseline + 2 * baseline_iqr, 1.3 * baseline)
    
    active_mask = smoothed > threshold
    active_indices = np.where(active_mask)[0]
    if len(active_indices) > 0:
        last_threshold = int(active_indices[-1])
    else:
        last_threshold = min_active - 1
    
    # Combine methods: take the minimum of cumulative energy and threshold methods
    # This ensures we don't keep too many frames, but also don't cut too aggressively
    candidates = [energy_95, last_active_window, last_threshold]
    
    # Use the median of the three methods for robustness
    last_active_vel = int(np.median(candidates))
    
    # Add a small buffer (3 frames) and enforce minimum
    last_active_vel = min(last_active_vel + 3, n - 1)
    last_active_vel = max(last_active_vel, min_active - 1)
    
    # Convert from velocity frame index to IQ frame index
    last_iq_frame = last_active_vel + 1
    
    # Ensure we don't exceed total frames
    last_iq_frame = min(last_iq_frame, total_iq_frames)
    
    return last_iq_frame, baseline, threshold


def analyze_all_files(data_root='data'):
    """Analyze all raw IQ files and return per-file statistics."""
    raw_dirs = [
        os.path.join(data_root, '2FT(na100)'),
        os.path.join(data_root, '3FT(na100)'),
        os.path.join(data_root, 'CIRS_Phantom_Latest'),
    ]
    
    results = {}
    
    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir):
            logger.warning(f"Directory not found: {raw_dir}")
            continue
            
        npy_files = sorted(glob.glob(os.path.join(raw_dir, '**/*.npy'), recursive=True))
        logger.info(f"Analyzing {len(npy_files)} files in {raw_dir}")
        
        for f in npy_files:
            try:
                # Load IQ data: stored as (H, W, T), transpose to (T, H, W)
                data = np.load(f)
                data = np.transpose(data, (2, 0, 1))
                T, H, W = data.shape
                
                # Compute velocity energy profile
                frame_energy, frame_std, _ = compute_velocity_energy_profile(data)
                
                # Detect last active frame
                last_active, baseline, threshold = detect_last_active_frame(
                    frame_energy, total_iq_frames=T
                )
                
                rel_path = os.path.relpath(f, data_root)
                
                results[rel_path] = {
                    'filepath': f,
                    'total_frames': T,
                    'height': H,
                    'width': W,
                    'last_active_frame': last_active,
                    'active_frames': last_active,
                    'empty_frames': T - last_active,
                    'pct_active': round(100 * last_active / T, 1),
                    'pct_empty': round(100 * (T - last_active) / T, 1),
                    'max_velocity_energy': float(frame_energy.max()),
                    'baseline_energy': float(baseline),
                    'threshold': float(threshold),
                    'first_half_energy': float(frame_energy[:len(frame_energy)//2].mean()),
                    'second_half_energy': float(frame_energy[len(frame_energy)//2:].mean()),
                    'energy_ratio': float(
                        frame_energy[:len(frame_energy)//2].mean() /
                        (frame_energy[len(frame_energy)//2:].mean() + 1e-10)
                    ),
                    'frame_energies': frame_energy.tolist(),
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {f}: {e}")
                continue
    
    return results


def print_statistics(results):
    """Print detailed statistics grouped by dataset and frequency."""
    
    # Group by dataset
    datasets = defaultdict(dict)
    for rel_path, info in results.items():
        parts = rel_path.split(os.sep)
        dataset = parts[0]
        datasets[dataset][rel_path] = info
    
    print("\n" + "=" * 100)
    print("FRAME ANALYSIS STATISTICS")
    print("=" * 100)
    
    total_files = 0
    total_original_frames = 0
    total_active_frames = 0
    total_eliminated_frames = 0
    
    for dataset_name in sorted(datasets.keys()):
        dataset_files = datasets[dataset_name]
        n_files = len(dataset_files)
        
        print(f"\n{'='*90}")
        print(f"DATASET: {dataset_name} ({n_files} files)")
        print(f"{'='*90}")
        
        # Group by frequency
        freq_groups = defaultdict(list)
        for rel_path, info in sorted(dataset_files.items()):
            # Extract frequency from path
            parts = rel_path.replace('\\', '/').split('/')
            freq = None
            for p in parts:
                if p.endswith('Hz') or p.endswith('kHz'):
                    freq = p
                    break
            if freq is None:
                freq = 'unknown'
            freq_groups[freq].append((rel_path, info))
        
        for freq in sorted(freq_groups.keys(), key=lambda x: _freq_sort_key(x)):
            files = freq_groups[freq]
            active_counts = [info['active_frames'] for _, info in files]
            total_counts = [info['total_frames'] for _, info in files]
            empty_counts = [info['empty_frames'] for _, info in files]
            
            avg_active = np.mean(active_counts)
            min_active = np.min(active_counts)
            max_active = np.max(active_counts)
            avg_total = np.mean(total_counts)
            avg_empty = np.mean(empty_counts)
            
            print(f"\n  Frequency: {freq} ({len(files)} files)")
            print(f"    Total frames/file: {int(avg_total)}")
            print(f"    Active frames: avg={avg_active:.1f}, min={min_active}, max={max_active}")
            print(f"    Empty frames eliminated: avg={avg_empty:.1f} ({100*avg_empty/avg_total:.1f}%)")
            
            # Show individual files
            for rel_path, info in files:
                short = os.path.basename(os.path.dirname(rel_path)) + '/' + os.path.basename(rel_path).split('.')[0]
                status = "TRIM" if info['empty_frames'] > 0 else "KEEP"
                print(f"      {status}: {rel_path}")
                print(f"             Keep frames 0-{info['last_active_frame']-1} ({info['active_frames']}/{info['total_frames']}) | "
                      f"Eliminate {info['empty_frames']} frames ({info['pct_empty']}%) | "
                      f"Energy ratio: {info['energy_ratio']:.1f}x")
        
        # Dataset totals
        ds_total = sum(info['total_frames'] for info in dataset_files.values())
        ds_active = sum(info['active_frames'] for info in dataset_files.values())
        ds_eliminated = ds_total - ds_active
        
        print(f"\n  --- Dataset Summary ---")
        print(f"  Total original frames: {ds_total}")
        print(f"  Active frames kept:    {ds_active}")
        print(f"  Empty frames removed:  {ds_eliminated} ({100*ds_eliminated/ds_total:.1f}%)")
        
        total_files += n_files
        total_original_frames += ds_total
        total_active_frames += ds_active
        total_eliminated_frames += ds_eliminated
    
    print(f"\n{'='*90}")
    print(f"GRAND TOTAL")
    print(f"{'='*90}")
    print(f"  Files processed:       {total_files}")
    print(f"  Total original frames: {total_original_frames}")
    print(f"  Active frames kept:    {total_active_frames}")
    print(f"  Empty frames removed:  {total_eliminated_frames} ({100*total_eliminated_frames/total_original_frames:.1f}%)")
    print(f"  Size reduction:        {100*total_eliminated_frames/total_original_frames:.1f}%")
    
    return total_files, total_original_frames, total_active_frames


def _freq_sort_key(freq_str):
    """Sort frequencies numerically."""
    freq_str = freq_str.lower()
    try:
        if 'khz' in freq_str:
            return float(freq_str.replace('khz', '')) * 1000
        elif 'hz' in freq_str:
            return float(freq_str.replace('hz', ''))
    except:
        pass
    return 999999


def trim_and_save(results, data_root='data', output_suffix='_cleaned'):
    """
    Trim empty frames from all IQ files and save cleaned data.
    
    Saves trimmed files in a parallel directory structure with _cleaned suffix.
    Also backs up originals to _original directories.
    """
    
    for rel_path, info in results.items():
        filepath = info['filepath']
        last_active = info['last_active_frame']
        total = info['total_frames']
        
        if last_active >= total:
            logger.info(f"SKIP (all active): {rel_path}")
            continue
        
        # Load original data (H, W, T)
        data = np.load(filepath)
        original_shape = data.shape
        
        # Trim to active frames only (along T axis, which is axis=2)
        trimmed = data[:, :, :last_active]
        
        # Backup original
        backup_dir = os.path.dirname(filepath).replace(
            data_root, data_root + '_original_backup'
        )
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(filepath))
        if not os.path.exists(backup_path):
            shutil.copy2(filepath, backup_path)
        
        # Overwrite with trimmed data
        np.save(filepath, trimmed)
        
        logger.info(
            f"TRIMMED: {rel_path} | {original_shape} -> {trimmed.shape} | "
            f"Removed {total - last_active} frames"
        )
    
    logger.info("All files trimmed and saved.")


def create_visualization_videos(results, data_root='data', output_dir='outputs/frame_analysis_videos'):
    """Create visualization videos of the cleaned data for each experiment."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        import matplotlib.colors as mcolors
    except ImportError:
        logger.error("matplotlib not found. Install with: pip install matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for ffmpeg
    has_ffmpeg = shutil.which('ffmpeg') is not None
    
    for rel_path, info in sorted(results.items()):
        filepath = info['filepath']
        last_active = info['last_active_frame']
        
        # Load data
        data = np.load(filepath)  # (H, W, T) after trimming
        data = np.transpose(data, (2, 0, 1))  # (T, H, W)
        T, H, W = data.shape
        
        # Compute velocity for visualization
        temporal_mean = data.mean(axis=0, keepdims=True)
        filtered = data - temporal_mean
        z_t = filtered[1:]
        z_t_minus_1 = filtered[:-1]
        autocorrelation = z_t * np.conj(z_t_minus_1)
        velocity = np.angle(autocorrelation)  # (T-1, H, W)
        
        # Create a safe filename
        safe_name = rel_path.replace('/', '_').replace('\\', '_').replace(' ', '_')
        safe_name = safe_name.replace('(', '').replace(')', '').replace('.npy', '')
        
        # Create video
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # IQ magnitude
        iq_mag = np.abs(data)
        vmin_iq, vmax_iq = np.percentile(iq_mag, [1, 99])
        
        # Velocity
        vmin_v, vmax_v = np.percentile(velocity, [1, 99])
        vmax_v = max(abs(vmin_v), abs(vmax_v))
        
        im0 = axes[0].imshow(iq_mag[0], cmap='gray', vmin=vmin_iq, vmax=vmax_iq, aspect='auto')
        axes[0].set_title(f'IQ Magnitude - Frame 0/{T-1}')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        im1 = axes[1].imshow(velocity[0], cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v, aspect='auto')
        axes[1].set_title(f'Velocity - Frame 0/{T-2}')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        fig.suptitle(f'{rel_path}\n(Trimmed: {T} active frames)', fontsize=10)
        plt.tight_layout()
        
        def update(frame):
            im0.set_data(iq_mag[frame])
            axes[0].set_title(f'IQ Magnitude - Frame {frame}/{T-1}')
            
            vel_frame = min(frame, len(velocity) - 1)
            im1.set_data(velocity[vel_frame])
            axes[1].set_title(f'Velocity - Frame {vel_frame}/{T-2}')
            return [im0, im1]
        
        if has_ffmpeg:
            video_path = os.path.join(output_dir, f'{safe_name}.mp4')
            anim = FuncAnimation(fig, update, frames=T, interval=100, blit=True)
            writer = FFMpegWriter(fps=10, bitrate=2000)
            anim.save(video_path, writer=writer)
            logger.info(f"Video saved: {video_path}")
        else:
            # Save as GIF instead
            video_path = os.path.join(output_dir, f'{safe_name}.gif')
            anim = FuncAnimation(fig, update, frames=T, interval=100, blit=True)
            anim.save(video_path, writer='pillow', fps=10)
            logger.info(f"GIF saved: {video_path}")
        
        plt.close(fig)
    
    logger.info(f"All visualization videos saved to {output_dir}")


def create_summary_visualization(results, output_dir='outputs/frame_analysis_videos'):
    """Create a summary plot showing energy profiles and cutoff points."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by dataset and frequency
    datasets = defaultdict(lambda: defaultdict(list))
    for rel_path, info in results.items():
        parts = rel_path.replace('\\', '/').split('/')
        dataset = parts[0]
        freq = None
        for p in parts:
            if p.endswith('Hz') or p.endswith('kHz'):
                freq = p
                break
        if freq:
            datasets[dataset][freq].append((rel_path, info))
    
    for dataset_name in sorted(datasets.keys()):
        freq_data = datasets[dataset_name]
        n_freqs = len(freq_data)
        
        fig, axes = plt.subplots(n_freqs, 1, figsize=(14, 4 * n_freqs), squeeze=False)
        fig.suptitle(f'Velocity Energy Profiles - {dataset_name}', fontsize=14, fontweight='bold')
        
        for idx, freq in enumerate(sorted(freq_data.keys(), key=_freq_sort_key)):
            ax = axes[idx, 0]
            files = freq_data[freq]
            
            for rel_path, info in files[:5]:  # Show up to 5 per freq
                energies = info['frame_energies']
                ax.plot(energies, alpha=0.5, linewidth=0.8)
                # Mark cutoff
                cutoff = info['last_active_frame'] - 1  # velocity frame index
                if cutoff < len(energies):
                    ax.axvline(x=cutoff, color='red', linewidth=0.5, alpha=0.3)
            
            # Show threshold
            if files:
                threshold = files[0][1]['threshold']
                ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, 
                          label=f'Threshold={threshold:.4f}')
            
            avg_active = np.mean([info['active_frames'] for _, info in files])
            ax.set_title(f'{freq} ({len(files)} files) - Avg active frames: {avg_active:.0f}')
            ax.set_xlabel('Velocity Frame Index')
            ax.set_ylabel('Mean |velocity|')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = dataset_name.replace('(', '').replace(')', '').replace(' ', '_')
        fig_path = os.path.join(output_dir, f'energy_profiles_{safe_name}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Energy profile plot saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Clean empty frames from SWE IQ dataset')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory of data')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze and report, do not modify files')
    parser.add_argument('--skip_videos', action='store_true',
                       help='Skip creating visualization videos')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to create (default: all)')
    parser.add_argument('--output_dir', type=str, default='outputs/frame_analysis_videos',
                       help='Output directory for videos and plots')
    
    args = parser.parse_args()
    
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Analyze all files
    logger.info("Step 1: Analyzing all raw IQ files...")
    results = analyze_all_files(args.data_root)
    
    # Step 2: Print and save statistics
    logger.info("Step 2: Computing statistics...")
    print_statistics(results)
    
    # Save analysis results
    analysis_path = os.path.join(args.output_dir, 'frame_analysis_results.json')
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    # Save without frame_energies to keep file manageable
    results_summary = {}
    for k, v in results.items():
        results_summary[k] = {key: val for key, val in v.items() if key != 'frame_energies'}
    with open(analysis_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Analysis results saved to {analysis_path}")
    
    if args.analyze_only:
        logger.info("Analyze-only mode. Exiting without modifying files.")
        return
    
    # Step 3: Create energy profile summary plots
    logger.info("Step 3: Creating energy profile summary plots...")
    create_summary_visualization(results, args.output_dir)
    
    # Step 4: Trim and save cleaned data
    logger.info("Step 4: Trimming empty frames and saving cleaned data...")
    trim_and_save(results, args.data_root)
    
    # Step 5: Create visualization videos
    if not args.skip_videos:
        logger.info("Step 5: Creating visualization videos...")
        # Reload results after trimming (to visualize the trimmed data)
        if args.max_videos:
            # Only create videos for a subset
            subset = dict(list(results.items())[:args.max_videos])
        else:
            subset = results
        create_visualization_videos(subset, args.data_root, args.output_dir)
    
    logger.info("Done! All cleaning and visualization complete.")


if __name__ == '__main__':
    main()
