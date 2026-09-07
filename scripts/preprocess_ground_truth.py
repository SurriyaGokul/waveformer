#!/usr/bin/env python3
"""
Preprocess Ground Truth Velocity Maps from Dense IQ Data.

This script pre-calculates "Velocity Ground Truth" from existing Dense IQ .npy files
using the Kasai Autocorrelation formula (Lag-1) to extract particle velocity.

Algorithm:
    Autocorrelation: R(t) = z(t) * conj(z(t-1))
    Velocity: v(t) = angle(R(t))

Input Formats Supported:
    - Complex64 with shape (T, H, W)
    - Float32 with shape (T, 2, H, W) where dim=1 is [Real, Imag]

Output:
    - Float32 velocity maps with shape (T-1, 1, H, W)
    - Range: [-pi, pi] radians

Usage:
    python preprocess_ground_truth.py --input_dir /path/to/dense_iq --output_dir /path/to/velocity
    python preprocess_ground_truth.py --input_file /path/to/single_file.npy --output_dir /path/to/velocity
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_iq_data(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load IQ data from .npy file and convert to complex format.
    
    Handles both:
        - Complex64: shape (T, H, W)
        - Float32: shape (T, 2, H, W) where dim=1 is [Real, Imag]
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        Complex64 array of shape (T, H, W)
        
    Raises:
        ValueError: If input format is not recognized
    """
    data = np.load(file_path)
    logger.debug(f"Loaded {file_path}: shape={data.shape}, dtype={data.dtype}")
    
    # Case 1: Already complex
    if np.iscomplexobj(data):
        if data.ndim == 3:
            # Shape: (T, H, W) - expected format
            return data.astype(np.complex64)
        elif data.ndim == 4 and data.shape[1] == 1:
            # Shape: (T, 1, H, W) - squeeze channel dim
            return data.squeeze(1).astype(np.complex64)
        else:
            raise ValueError(
                f"Unexpected complex data shape: {data.shape}. "
                f"Expected (T, H, W) or (T, 1, H, W)."
            )
    
    # Case 2: Real-valued with separate Real/Imag channels
    if data.ndim == 4:
        if data.shape[1] == 2:
            # Shape: (T, 2, H, W) where dim=1 is [Real, Imag]
            real_part = data[:, 0, :, :]
            imag_part = data[:, 1, :, :]
            return (real_part + 1j * imag_part).astype(np.complex64)
        else:
            raise ValueError(
                f"Unexpected 4D real data shape: {data.shape}. "
                f"Expected (T, 2, H, W) with dim=1 as [Real, Imag]."
            )
    
    # Case 3: Real 3D data - cannot convert
    if data.ndim == 3 and not np.iscomplexobj(data):
        raise ValueError(
            f"Cannot convert real 3D array to complex. Shape: {data.shape}. "
            f"Expected either complex (T, H, W) or real (T, 2, H, W)."
        )
    
    raise ValueError(
        f"Unrecognized data format: shape={data.shape}, dtype={data.dtype}"
    )


def detect_active_region(
    iq_data: np.ndarray,
    threshold_ratio: float = 0.1,
    min_active_frames: int = 10,
) -> Tuple[int, int, np.ndarray]:
    """
    Auto-detect the active region where actual ultrasound data exists.
    
    Many ultrasound acquisitions contain zero-padding before/after the actual
    experiment. Including these zero regions in the clutter filter calculation
    dilutes the mean and leaves residual clutter in the output.
    
    This function finds the start and end frames where signal energy exceeds
    a threshold, then crops to only the active region.
    
    Args:
        iq_data: Complex IQ data of shape (T, H, W)
        threshold_ratio: Fraction of max energy to use as threshold (default: 0.1 = 10%)
        min_active_frames: Minimum number of active frames required
        
    Returns:
        Tuple of (start_idx, end_idx, energy_profile):
            - start_idx: First frame index with signal
            - end_idx: Last frame index with signal (exclusive for slicing)
            - energy_profile: Per-frame energy array of shape (T,)
            
    Raises:
        ValueError: If no active region is detected or too few active frames
    """
    # Compute per-frame energy (mean magnitude)
    energy_profile = np.abs(iq_data).mean(axis=(1, 2))  # Shape: (T,)
    
    # Set threshold as fraction of max energy
    max_energy = energy_profile.max()
    threshold = threshold_ratio * max_energy
    
    logger.debug(f"Energy profile: min={energy_profile.min():.2f}, max={max_energy:.2f}, threshold={threshold:.2f}")
    
    # Find frames above threshold
    active_mask = energy_profile > threshold
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) == 0:
        raise ValueError(
            f"No active frames detected! Max energy={max_energy:.2f}, "
            f"threshold={threshold:.2f}. Check if data file is valid."
        )
    
    if len(active_indices) < min_active_frames:
        raise ValueError(
            f"Only {len(active_indices)} active frames detected (need >= {min_active_frames}). "
            f"Data may be corrupted or threshold too high."
        )
    
    # If there are multiple active segments, select the longest contiguous block
    if len(active_indices) > 1:
        split_points = np.where(np.diff(active_indices) > 1)[0]
        segments = []
        seg_start = active_indices[0]
        for split_idx in split_points:
            seg_end = active_indices[split_idx]
            segments.append((seg_start, seg_end))
            seg_start = active_indices[split_idx + 1]
        segments.append((seg_start, active_indices[-1]))

        # Choose the longest segment
        seg_lengths = [(end - start + 1) for start, end in segments]
        longest_idx = int(np.argmax(seg_lengths))
        start_idx, end_idx_inclusive = segments[longest_idx]

        if len(segments) > 1:
            logger.warning(
                f"Multiple active segments detected ({len(segments)}). "
                f"Using longest segment: [{start_idx}, {end_idx_inclusive}] "
                f"({seg_lengths[longest_idx]} frames)."
            )
    else:
        start_idx = active_indices[0]
        end_idx_inclusive = active_indices[0]

    end_idx = end_idx_inclusive + 1  # +1 for exclusive slicing

    # Validate selected segment length
    active_length = end_idx - start_idx
    if active_length < min_active_frames:
        raise ValueError(
            f"Active segment too short: {active_length} frames (need >= {min_active_frames}). "
            f"Threshold may be too strict or data may be corrupted."
        )
    
    # Log the detection results
    total_frames = len(energy_profile)
    active_frames = end_idx - start_idx
    zero_before = start_idx
    zero_after = total_frames - end_idx
    
    logger.info(
        f"Active region detected: frames [{start_idx}, {end_idx}) "
        f"({active_frames}/{total_frames} frames = {100*active_frames/total_frames:.1f}%)"
    )
    if zero_before > 0 or zero_after > 0:
        logger.warning(
            f"Zero-padded regions found: {zero_before} frames before, {zero_after} frames after. "
            f"These will be excluded from processing."
        )
    
    return start_idx, end_idx, energy_profile


def apply_clutter_filter(iq_data: np.ndarray) -> np.ndarray:
    """
    Apply clutter filter by subtracting the temporal mean.
    
    This removes static/slow-moving tissue signal (clutter), which is typically
    100x brighter than the shear wave signal. Without this filter, the Kasai
    algorithm will "lock on" to the stationary tissue and produce near-zero
    velocity estimates.
    
    Args:
        iq_data: Complex IQ data of shape (T, H, W)
        
    Returns:
        Clutter-filtered IQ data of shape (T, H, W)
    """
    # Compute temporal mean (static tissue component)
    temporal_mean = iq_data.mean(axis=0, keepdims=True)
    
    # Subtract to isolate dynamic component (shear wave)
    filtered = iq_data - temporal_mean
    
    logger.debug(
        f"Clutter filter applied: mean magnitude before={np.abs(iq_data).mean():.4f}, "
        f"after={np.abs(filtered).mean():.4f}"
    )
    
    return filtered


def compute_kasai_velocity(
    iq_data: np.ndarray,
    apply_clutter_filtering: bool = True,
    auto_crop_active: bool = True,
    active_threshold_ratio: float = 0.1,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Compute particle velocity using Kasai Autocorrelation (Lag-1).
    
    The Kasai method computes the phase difference between consecutive frames:
        R(t) = z(t) * conj(z(t-1))
        velocity(t) = angle(R(t))
    
    The resulting velocity is in radians, representing the phase shift per frame.
    This is proportional to the axial particle velocity in ultrasound imaging.
    
    IMPORTANT: 
        1. Active region detection removes zero-padded frames that would dilute
           the clutter filter mean.
        2. Clutter filtering (temporal mean subtraction) removes static tissue 
           signal, which is ~100x brighter than the shear wave.
    
    Args:
        iq_data: Complex IQ data of shape (T, H, W)
        apply_clutter_filtering: Whether to apply temporal mean subtraction
            before computing velocity. Default True.
        auto_crop_active: Whether to auto-detect and crop to active region
            (removes zero-padded frames). Default True. HIGHLY RECOMMENDED.
        active_threshold_ratio: Energy threshold for active region detection
            as fraction of max energy (default: 0.1 = 10%)
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (velocity, active_range):
            - velocity: Array of shape (T_active-1, 1, H, W), dtype float32, range [-pi, pi]
            - active_range: Tuple (start_idx, end_idx) of original frame indices, 
                           or None if auto_crop_active=False
    """
    if iq_data.ndim != 3:
        raise ValueError(f"Expected 3D input (T, H, W), got shape {iq_data.shape}")
    
    T, H, W = iq_data.shape
    
    if T < 2:
        raise ValueError(f"Need at least 2 time frames, got T={T}")
    
    logger.debug(f"Computing Kasai velocity: input shape (T={T}, H={H}, W={W})")
    
    active_range = None
    
    # Step 1: Auto-detect and crop to active region (removes zero-padding)
    if auto_crop_active:
        start_idx, end_idx, energy_profile = detect_active_region(
            iq_data, 
            threshold_ratio=active_threshold_ratio
        )
        iq_data = iq_data[start_idx:end_idx]
        active_range = (start_idx, end_idx)
        logger.info(f"Cropped to active region: {iq_data.shape[0]} frames")
    
    # Step 2: Apply clutter filter on CROPPED data only
    if apply_clutter_filtering:
        iq_data = apply_clutter_filter(iq_data)
    
    # Get consecutive frames
    z_t = iq_data[1:, :, :]      # z(t): frames 1 to T-1
    z_t_minus_1 = iq_data[:-1, :, :]  # z(t-1): frames 0 to T-2
    
    # Compute Lag-1 autocorrelation: R = z(t) * conj(z(t-1))
    autocorrelation = z_t * np.conj(z_t_minus_1)
    
    # Extract velocity as the angle (phase) of the autocorrelation
    # np.angle returns values in [-pi, pi]
    velocity = np.angle(autocorrelation).astype(np.float32)
    
    # Add channel dimension: (T-1, H, W) -> (T-1, 1, H, W)
    velocity = velocity[:, np.newaxis, :, :]
    
    logger.debug(f"Output velocity shape: {velocity.shape}")
    logger.debug(f"Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")
    
    return velocity, active_range


def validate_velocity_output(
    velocity: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Validate the computed velocity array.
    
    Args:
        velocity: Velocity array to validate
        expected_shape: Expected shape (optional)
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check dtype
    if velocity.dtype != np.float32:
        raise ValueError(f"Expected float32, got {velocity.dtype}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(velocity)):
        raise ValueError("Velocity contains NaN values")
    if np.any(np.isinf(velocity)):
        raise ValueError("Velocity contains Inf values")
    
    # Check range (should be [-pi, pi])
    if velocity.min() < -np.pi - 1e-6 or velocity.max() > np.pi + 1e-6:
        logger.warning(
            f"Velocity range [{velocity.min():.4f}, {velocity.max():.4f}] "
            f"exceeds expected [-pi, pi]"
        )
    
    # Check shape
    if expected_shape is not None and velocity.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {velocity.shape}")
    
    # Check 4D with channel dim
    if velocity.ndim != 4 or velocity.shape[1] != 1:
        raise ValueError(f"Expected shape (T, 1, H, W), got {velocity.shape}")
    
    return True


def process_single_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = "velocity_",
    validate: bool = True,
    apply_clutter_filter: bool = True,
    auto_crop_active: bool = True,
    active_threshold_ratio: float = 0.1,
) -> Tuple[Path, Optional[dict]]:
    """
    Process a single IQ file and save the velocity ground truth.
    
    Args:
        input_path: Path to input .npy file
        output_dir: Directory to save output
        prefix: Prefix for output filename
        validate: Whether to validate output
        apply_clutter_filter: Whether to apply temporal mean subtraction
            before computing velocity. Should be True for correct results.
        auto_crop_active: Whether to auto-detect and crop to active region
            (removes zero-padded frames). HIGHLY RECOMMENDED.
        active_threshold_ratio: Energy threshold for active region detection
            as fraction of max energy (default: 0.1 = 10%)
        
    Returns:
        Tuple of (output_path, metadata):
            - output_path: Path to the saved velocity file
            - metadata: Dictionary with processing info (active_range, shapes, etc.)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load IQ data
    logger.info(f"Processing: {input_path.name}")
    iq_data = load_iq_data(input_path)
    original_shape = iq_data.shape
    
    # Compute velocity (active region cropping and clutter filtering inside)
    velocity, active_range = compute_kasai_velocity(
        iq_data, 
        apply_clutter_filtering=apply_clutter_filter,
        auto_crop_active=auto_crop_active,
        active_threshold_ratio=active_threshold_ratio,
    )
    
    # Validate
    if validate:
        validate_velocity_output(velocity)
    
    # Generate output filename
    output_name = f"{prefix}{input_path.stem}.npy"
    output_path = output_dir / output_name
    
    # Save velocity
    np.save(output_path, velocity)
    logger.info(f"Saved: {output_path} (shape: {velocity.shape})")
    
    # Prepare metadata
    metadata = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_shape": original_shape,
        "output_shape": velocity.shape,
        "active_range": active_range,
        "clutter_filter": apply_clutter_filter,
        "auto_crop": auto_crop_active,
    }
    
    # Save metadata alongside velocity
    metadata_path = output_dir / f"{prefix}{input_path.stem}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return output_path, metadata


def process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = "velocity_",
    pattern: str = "*.npy",
    validate: bool = True,
    skip_existing: bool = False,
    apply_clutter_filter: bool = True,
    auto_crop_active: bool = True,
    active_threshold_ratio: float = 0.1,
) -> dict:
    """
    Process all IQ files in a directory.
    
    Args:
        input_dir: Directory containing input .npy files
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
        pattern: Glob pattern for input files
        validate: Whether to validate outputs
        skip_existing: Whether to skip files that already have outputs
        apply_clutter_filter: Whether to apply temporal mean subtraction
            before computing velocity. Should be True for correct results.
        auto_crop_active: Whether to auto-detect and crop to active region.
        active_threshold_ratio: Energy threshold for active region detection.
        
    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all input files
    input_files = sorted(input_dir.glob(pattern))
    
    if not input_files:
        logger.warning(f"No files matching '{pattern}' found in {input_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}
    
    logger.info(f"Found {len(input_files)} files to process")
    logger.info(f"Clutter filter enabled: {apply_clutter_filter}")
    logger.info(f"Auto-crop active region: {auto_crop_active} (threshold: {active_threshold_ratio*100:.0f}%)")
    
    stats = {"processed": 0, "skipped": 0, "failed": 0, "failed_files": [], "metadata": []}
    
    for input_path in tqdm(input_files, desc="Processing IQ files"):
        output_name = f"{prefix}{input_path.stem}.npy"
        output_path = output_dir / output_name
        
        # Skip if output exists
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping (exists): {input_path.name}")
            stats["skipped"] += 1
            continue
        
        try:
            _, metadata = process_single_file(
                input_path, output_dir, prefix, validate,
                apply_clutter_filter=apply_clutter_filter,
                auto_crop_active=auto_crop_active,
                active_threshold_ratio=active_threshold_ratio,
            )
            stats["processed"] += 1
            stats["metadata"].append(metadata)
        except Exception as e:
            logger.error(f"Failed to process {input_path.name}: {e}")
            stats["failed"] += 1
            stats["failed_files"].append(str(input_path))
    
    # Summary
    logger.info(
        f"Processing complete: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    
    return stats


def compute_dataset_statistics(
    velocity_dir: Union[str, Path],
    pattern: str = "velocity_*.npy",
) -> dict:
    """
    Compute statistics across all velocity files in a directory.
    
    Useful for understanding the data distribution and setting normalization constants.
    
    Args:
        velocity_dir: Directory containing velocity .npy files
        pattern: Glob pattern for velocity files
        
    Returns:
        Dictionary with statistics
    """
    velocity_dir = Path(velocity_dir)
    files = sorted(velocity_dir.glob(pattern))
    
    if not files:
        logger.warning(f"No files matching '{pattern}' found")
        return {}
    
    all_values = []
    shapes = []
    
    for f in tqdm(files, desc="Computing statistics"):
        data = np.load(f)
        all_values.append(data.flatten())
        shapes.append(data.shape)
    
    all_values = np.concatenate(all_values)
    
    stats = {
        "num_files": len(files),
        "total_samples": len(all_values),
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values)),
        "median": float(np.median(all_values)),
        "percentile_1": float(np.percentile(all_values, 1)),
        "percentile_99": float(np.percentile(all_values, 99)),
        "unique_shapes": list(set(shapes)),
    }
    
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return stats


def main():
    """Main entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess Dense IQ data to Velocity Ground Truth using Kasai Autocorrelation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all files in a directory
    python preprocess_ground_truth.py --input_dir ./data/dense_iq --output_dir ./data/velocity
    
    # Process a single file
    python preprocess_ground_truth.py --input_file ./data/dense_IQ1.npy --output_dir ./data/velocity
    
    # Process directory, skip existing, compute stats
    python preprocess_ground_truth.py --input_dir ./data/dense_iq --output_dir ./data/velocity \\
        --skip_existing --compute_stats
        """,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing Dense IQ .npy files",
    )
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Single Dense IQ .npy file to process",
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save velocity ground truth files",
    )
    
    # Processing options
    parser.add_argument(
        "--prefix",
        type=str,
        default="velocity_",
        help="Prefix for output filenames (default: 'velocity_')",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob pattern for input files when using --input_dir (default: '*.npy')",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if output file already exists",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Disable output validation",
    )
    parser.add_argument(
        "--no_clutter_filter",
        action="store_true",
        help="Disable clutter filtering (temporal mean subtraction). "
             "WARNING: This is NOT recommended! Without clutter filtering, "
             "static tissue signal dominates and Kasai produces near-zero velocity.",
    )
    parser.add_argument(
        "--no_auto_crop",
        action="store_true",
        help="Disable auto-detection and cropping of active region. "
             "WARNING: This is NOT recommended! Zero-padded frames will dilute "
             "the clutter filter mean and leave residual horizontal stripes.",
    )
    parser.add_argument(
        "--active_threshold",
        type=float,
        default=0.1,
        help="Energy threshold ratio for active region detection (default: 0.1 = 10%% of max). "
             "Lower values include more frames, higher values are more strict.",
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Compute and display statistics after processing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Warn if clutter filter is disabled
    if args.no_clutter_filter:
        logger.warning(
            "Clutter filtering is DISABLED. This is NOT recommended!\n"
            "Without clutter filtering, static tissue signal (~100x brighter than shear wave)\n"
            "will dominate, causing Kasai to produce near-zero velocity estimates."
        )
    
    # Warn if auto-crop is disabled
    if args.no_auto_crop:
        logger.warning(
            "Auto-crop of active region is DISABLED. This is NOT recommended!\n"
            "Zero-padded frames at the start/end of the file will dilute the clutter filter mean,\n"
            "leaving ~70%% residual clutter (horizontal stripes) in the velocity output.\n"
            "Your 'Ground Truth' will contain static tissue artifacts, not shear waves."
        )
    else:
        logger.info(f"Active region detection enabled with threshold: {args.active_threshold*100:.0f}%% of max energy")
    
    # Process
    if args.input_file:
        # Single file mode
        process_single_file(
            args.input_file,
            args.output_dir,
            prefix=args.prefix,
            validate=not args.no_validate,
            apply_clutter_filter=not args.no_clutter_filter,
            auto_crop_active=not args.no_auto_crop,
            active_threshold_ratio=args.active_threshold,
        )
    else:
        # Directory mode
        process_directory(
            args.input_dir,
            args.output_dir,
            prefix=args.prefix,
            pattern=args.pattern,
            validate=not args.no_validate,
            skip_existing=args.skip_existing,
            apply_clutter_filter=not args.no_clutter_filter,
            auto_crop_active=not args.no_auto_crop,
            active_threshold_ratio=args.active_threshold,
        )
    
    # Compute statistics if requested
    if args.compute_stats:
        compute_dataset_statistics(
            args.output_dir,
            pattern=f"{args.prefix}*.npy",
        )


if __name__ == "__main__":
    main()
