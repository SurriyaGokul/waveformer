#!/usr/bin/env python3
"""
Visualize Generated Velocity Data as AVI Videos.

Replicates the MATLAB IQ2VEl.m "Make particle velocity movie" section in Python.
Loads velocity .npy files from data/phantom_velocity/{train,val}/ and generates
AVI videos showing particle velocity evolution over slow time.

MATLAB reference (from IQ2VEl.m):
    for lll=1:5:size(vel_filtL,3)
        imagesc(axisWidth, axisDepth, squeeze(vel_filtL(:,:,lll)));
        clim([-3 3]); colorbar; colormap jet;
        ylabel(h, 'Particle velocity (mm/s)')
        title(['Push location at ...', 'Particle velocity (mm/s) at:', ...])

Pixel spacing (from MATLAB comments):
    Axial pixel distance  = 0.0507 mm  (f_s = 5.208 MHz × 4)
    Lateral pixel distance = 0.1013 mm

Usage:
    # Visualize all train and val velocity files:
    python scripts/visualize_velocity_video.py

    # Specific file:
    python scripts/visualize_velocity_video.py --file data/phantom_velocity/train/velocity_10mmfocus_100kPa_inclusion_10kHz_20260128T165124.npy

    # Custom velocity range and frame step:
    python scripts/visualize_velocity_video.py --vmin -5 --vmax 5 --step 1

    # Only train or val split:
    python scripts/visualize_velocity_video.py --split train
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for video generation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pixel spacing from MATLAB IQ2VEl.m comments
# Axial: sampled every 1540 / (5.208 MHz * 4) = 0.074 mm, but after beamforming
# the effective pixel spacing used was:
AXIAL_PIXEL_MM = 0.0507    # mm per axial pixel
LATERAL_PIXEL_MM = 0.1013  # mm per lateral pixel


def load_metadata(metadata_path: Path) -> dict:
    """Load the preprocessing metadata JSON."""
    with open(metadata_path) as f:
        return json.load(f)


def find_entry_for_file(metadata: dict, vel_filename: str) -> dict:
    """Find the metadata entry matching a velocity filename."""
    # Extract name from filename: velocity_<name>.npy -> <name>
    name = vel_filename.replace('velocity_', '').replace('.npy', '')

    for split in ['train', 'val', 'discarded']:
        for entry in metadata.get(split, []):
            if entry.get('name') == name:
                return entry
    return {}


def compute_axes(roi: dict) -> tuple:
    """
    Compute physical axis vectors (mm) from ROI info and pixel spacing.

    Returns:
        axis_depth: (H_roi,) array of axial depths in mm
        axis_width: (W_roi,) array of lateral positions in mm (centered at 0)
    """
    H_roi = roi['H_roi']
    W_roi = roi['W_roi']
    min_row = roi['min_row']
    min_col = roi['min_col']

    # Axial depth axis (mm) — absolute depth from transducer
    axis_depth = (min_row + np.arange(H_roi)) * AXIAL_PIXEL_MM

    # Lateral axis (mm) — centered around middle of full image
    # The full image width is typically 380 columns; center at col 190
    full_width_cols = 380  # typical for this transducer
    center_col = full_width_cols / 2.0
    axis_width = ((min_col + np.arange(W_roi)) - center_col) * LATERAL_PIXEL_MM

    return axis_depth, axis_width


def compute_slow_time_axis(n_frames: int, frame_time_s: float, interp_factor: int = 1) -> np.ndarray:
    """Compute the slow-time axis in seconds."""
    dt = frame_time_s / interp_factor
    return np.arange(n_frames) * dt


def make_velocity_video(
    vel: np.ndarray,
    output_path: Path,
    axis_depth: np.ndarray,
    axis_width: np.ndarray,
    taxis: np.ndarray,
    title_prefix: str = '',
    vmin: float = -3.0,
    vmax: float = 3.0,
    step: int = 1,
    fps: int = 10,
    dpi: int = 100,
):
    """
    Create an AVI video of particle velocity frames — mirrors MATLAB IQ2VEl.m.

    Args:
        vel: (T, H, W) velocity array in mm/s
        output_path: Path for output .avi file
        axis_depth: (H,) axial depth in mm
        axis_width: (W,) lateral position in mm
        taxis: (T,) slow-time axis in seconds
        title_prefix: String to prepend to frame titles
        vmin, vmax: Colorbar limits (mm/s) — MATLAB uses [-3, 3]
        step: Frame step (MATLAB uses 5)
        fps: Video frame rate
        dpi: Figure DPI
    """
    T, H, W = vel.shape
    frame_indices = list(range(0, T, step))

    if len(frame_indices) == 0:
        logger.warning(f"No frames to render for {output_path}")
        return

    logger.info(
        f"  Creating video: {len(frame_indices)} frames, "
        f"step={step}, vel range=[{vel.min():.2f}, {vel.max():.2f}] mm/s"
    )

    # Extent for imagesc-style display: [left, right, top, bottom]
    extent = [axis_width[0], axis_width[-1], axis_depth[-1], axis_depth[0]]

    # Create figure matching MATLAB style
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(
        vel[frame_indices[0]],
        aspect='equal',
        extent=extent,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        origin='upper',
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Particle velocity (mm/s)', fontsize=14, fontweight='bold')

    ax.set_xlabel('Lateral distance from center (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Axial depth (mm)', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    title_text = ax.set_title('', fontsize=13, fontweight='bold')
    fig.tight_layout()

    def update(frame_idx):
        i = frame_indices[frame_idx]
        im.set_data(vel[i])
        t_str = f"{taxis[i] * 1e3:.3f} ms" if i < len(taxis) else f"frame {i}"
        title = f"{title_prefix}Particle velocity (mm/s) at t = {t_str}"
        title_text.set_text(title)
        return [im, title_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=False,
    )

    # Save as AVI using ffmpeg or pillow
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(output_path), writer=writer, dpi=dpi)
        logger.info(f"  Saved video: {output_path}")
    except Exception:
        # Fallback: save as .gif if ffmpeg not available
        gif_path = output_path.with_suffix('.gif')
        logger.warning(f"  FFmpeg not available, saving as GIF: {gif_path}")
        try:
            writer = animation.PillowWriter(fps=fps)
            anim.save(str(gif_path), writer=writer, dpi=dpi)
            logger.info(f"  Saved GIF: {gif_path}")
        except Exception as e2:
            logger.error(f"  Could not save animation: {e2}")
            # Fallback: save individual frames
            frames_dir = output_path.parent / output_path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Saving individual frames to {frames_dir}/")
            for fi, i in enumerate(frame_indices):
                im.set_data(vel[i])
                t_str = f"{taxis[i] * 1e3:.3f} ms" if i < len(taxis) else f"frame {i}"
                title = f"{title_prefix}Particle velocity (mm/s) at t = {t_str}"
                title_text.set_text(title)
                fig.savefig(frames_dir / f"frame_{fi:04d}.png", dpi=dpi, bbox_inches='tight')
            logger.info(f"  Saved {len(frame_indices)} frames to {frames_dir}/")

    plt.close(fig)


def make_summary_figure(
    vel: np.ndarray,
    output_path: Path,
    axis_depth: np.ndarray,
    axis_width: np.ndarray,
    taxis: np.ndarray,
    title_prefix: str = '',
    vmin: float = -3.0,
    vmax: float = 3.0,
    n_snapshots: int = 6,
):
    """
    Save a multi-panel summary figure showing velocity at selected time points.

    Useful for quick visual validation without opening a video.
    """
    T, H, W = vel.shape
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    extent = [axis_width[0], axis_width[-1], axis_depth[-1], axis_depth[0]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        im = ax.imshow(
            vel[idx], aspect='equal', extent=extent,
            cmap='jet', vmin=vmin, vmax=vmax,
            interpolation='nearest', origin='upper',
        )
        t_str = f"{taxis[idx] * 1e3:.3f} ms" if idx < len(taxis) else f"frame {idx}"
        ax.set_title(f"t = {t_str}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Lateral (mm)', fontsize=9)
        ax.set_ylabel('Depth (mm)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title_prefix}Particle Velocity Snapshots", fontsize=14, fontweight='bold')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Velocity (mm/s)')
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved summary figure: {output_path}")


def process_velocity_file(
    vel_path: Path,
    metadata: dict,
    output_dir: Path,
    vmin: float,
    vmax: float,
    step: int,
    fps: int,
    simulate_prf_hz: Optional[float] = None,
    native_prf_hz: Optional[float] = None,
):
    """Process a single velocity file: load, compute axes, generate video + summary."""
    vel = np.load(vel_path)
    logger.info(f"Loaded: {vel_path.name}  shape={vel.shape}  dtype={vel.dtype}")
    logger.info(f"  Velocity range: [{vel.min():.2f}, {vel.max():.2f}] mm/s, std={vel.std():.4f}")

    # Look up metadata entry
    entry = find_entry_for_file(metadata, vel_path.name)

    if entry:
        roi = entry['roi']
        frame_time_s = entry['frame_time_s']
        interp_factor = entry.get('interp_factor', 1)
        freq = entry.get('frequency', '')
        prf_hz = float(entry.get('prf_hz', 0.0) or 0.0)
        focus = entry.get('focus', '')
        inclusion = entry.get('inclusion', '')
        title_prefix = f"{focus}, {inclusion}, {freq}\n"
    else:
        # Fallback: use velocity shape to infer approximate axes
        logger.warning(f"  No metadata entry found for {vel_path.name}, using defaults")
        T, H, W = vel.shape
        roi = {'H_roi': H, 'W_roi': W, 'min_row': 103, 'min_col': 109}
        frame_time_s = 0.0001  # default to 10kHz PRF
        interp_factor = 1
        prf_hz = 0.0
        title_prefix = f"{vel_path.stem}\n"

    # Optionally simulate a different PRF by frame subsampling.
    # Example: native 500 Hz -> simulated 100 Hz => stride = 5.
    simulation_tag = ""
    if simulate_prf_hz is not None:
        if native_prf_hz is None:
            native_prf_hz = prf_hz if prf_hz > 0 else None

        if native_prf_hz is None or native_prf_hz <= 0:
            raise ValueError(
                "--simulate_prf_hz requires --native_prf_hz when metadata doesn't contain prf_hz"
            )
        if simulate_prf_hz <= 0:
            raise ValueError("--simulate_prf_hz must be > 0")

        stride_f = native_prf_hz / simulate_prf_hz
        stride = int(round(stride_f))
        if stride <= 0:
            raise ValueError(f"Invalid stride computed: {stride} (native={native_prf_hz}, simulate={simulate_prf_hz})")
        if abs(stride - stride_f) > 1e-6:
            raise ValueError(
                f"native_prf_hz/simulate_prf_hz must be an integer; got {stride_f:.6f}"
            )

        if step != 1 and step != stride:
            logger.warning(
                f"Overriding --step={step} with stride={stride} due to PRF simulation"
            )
        step = stride
        simulation_tag = f"sim{int(simulate_prf_hz)}Hz_from_{int(native_prf_hz)}Hz_stride{stride}"
        title_prefix = title_prefix + f"Simulated PRF: {simulate_prf_hz:g} Hz from {native_prf_hz:g} Hz (stride={stride})\n"

    axis_depth, axis_width = compute_axes(roi)
    T = vel.shape[0]
    taxis = compute_slow_time_axis(T, frame_time_s, interp_factor)

    # Ensure axis dimensions match data
    H, W = vel.shape[1], vel.shape[2]
    axis_depth = axis_depth[:H]
    axis_width = axis_width[:W]

    # Base name for outputs
    base_name = vel_path.stem.replace('velocity_', '')

    if simulation_tag:
        base_name = f"{base_name}_{simulation_tag}"

    # Generate video
    video_path = output_dir / f"Velocity_movie_{base_name}.avi"
    make_velocity_video(
        vel, video_path, axis_depth, axis_width, taxis,
        title_prefix=title_prefix, vmin=vmin, vmax=vmax,
        step=step, fps=fps,
    )

    # Generate summary figure
    summary_path = output_dir / f"Velocity_summary_{base_name}.png"
    make_summary_figure(
        vel, summary_path, axis_depth, axis_width, taxis,
        title_prefix=title_prefix, vmin=vmin, vmax=vmax,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated velocity data as videos (mirrors MATLAB IQ2VEl.m)"
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/phantom_velocity',
        help='Directory containing velocity .npy files'
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs/velocity_videos',
        help='Output directory for videos and summary figures'
    )
    parser.add_argument(
        '--file', type=str, default=None,
        help='Process a specific velocity .npy file instead of all files'
    )
    parser.add_argument(
        '--split', type=str, default=None, choices=['train', 'val'],
        help='Process only train or val split (default: both)'
    )
    parser.add_argument(
        '--vmin', type=float, default=-3.0,
        help='Minimum velocity for colorbar (mm/s). MATLAB default: -3'
    )
    parser.add_argument(
        '--vmax', type=float, default=3.0,
        help='Maximum velocity for colorbar (mm/s). MATLAB default: 3'
    )
    parser.add_argument(
        '--step', type=int, default=1,
        help='Frame step for video (MATLAB uses 5). Default: 1 (all frames)'
    )
    parser.add_argument(
        '--simulate_prf_hz', type=float, default=None,
        help='Simulate a different PRF by subsampling frames. Example: 100 to simulate 100Hz from native 500Hz.'
    )
    parser.add_argument(
        '--native_prf_hz', type=float, default=None,
        help='Native PRF of the velocity sequence (Hz). Only needed if metadata lacks prf_hz.'
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Video frame rate (default: 10)'
    )
    parser.add_argument(
        '--max_files', type=int, default=None,
        help='Maximum number of files to process (for quick testing)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        metadata = load_metadata(metadata_path)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        logger.warning(f"No metadata found at {metadata_path}, using defaults")
        metadata = {}

    # Collect velocity files
    if args.file:
        vel_files = [Path(args.file)]
    else:
        vel_files = []
        splits = [args.split] if args.split else ['train', 'val']
        for split in splits:
            split_dir = data_dir / split
            if split_dir.exists():
                vel_files.extend(sorted(split_dir.glob('velocity_*.npy')))

    if args.max_files:
        vel_files = vel_files[:args.max_files]

    if not vel_files:
        logger.error("No velocity files found!")
        sys.exit(1)

    logger.info(f"Processing {len(vel_files)} velocity file(s)")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Velocity display range: [{args.vmin}, {args.vmax}] mm/s")
    logger.info(f"Frame step: {args.step}, FPS: {args.fps}")
    if args.simulate_prf_hz is not None:
        logger.info(
            f"Simulating PRF: {args.simulate_prf_hz} Hz "
            f"(native_prf_hz={'metadata' if args.native_prf_hz is None else args.native_prf_hz})"
        )

    for i, vel_path in enumerate(vel_files):
        logger.info(f"\n[{i+1}/{len(vel_files)}] {vel_path.name}")
        try:
            process_velocity_file(
                vel_path, metadata, output_dir,
                vmin=args.vmin, vmax=args.vmax,
                step=args.step, fps=args.fps,
                simulate_prf_hz=args.simulate_prf_hz,
                native_prf_hz=args.native_prf_hz,
            )
        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=True)

    logger.info(f"\nDone! Videos saved to: {output_dir}")


if __name__ == '__main__':
    main()
