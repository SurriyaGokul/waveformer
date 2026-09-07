#!/usr/bin/env python3
"""
Convert .mat files to .npy files for the CIRS Phantom Latest dataset.

This script converts the IQ data from MATLAB .mat format to NumPy .npy format,
combining IBuffer and QBuffer into complex arrays (I + jQ).

The script:
1. Recursively finds all .mat files containing IBuffer/QBuffer data
2. Converts them to complex128 .npy files
3. Replaces the original .mat files (deletes them after conversion)

Directory structure:
    CIRS_Phantom_Latest/
    └── Data set 1(50)/
        ├── 10mmfocus/
        │   ├── 10kPa_inclusion/
        │   │   ├── 10kHz/
        │   │   │   ├── 20260128T163048.mat -> 20260128T163048.npy
        │   │   │   └── ...
        │   └── ...
        └── 15mmfocus/
            └── ...

Usage:
    python scripts/convert_mat_to_npy.py [--dry-run] [--keep-mat]

Options:
    --dry-run   Preview conversions without actually performing them
    --keep-mat  Keep original .mat files after conversion (default: delete them)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio


def find_iq_mat_files(root_dir: str) -> List[Path]:
    """
    Find all .mat files that contain IQ data (IBuffer and QBuffer).
    
    Excludes:
    - Root-level configuration .mat files (L11-5v_ARFI_SWEI_AC_*.mat)
    - Zone.Identifier files
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of Path objects for .mat files with IQ data
    """
    mat_files = []
    root_path = Path(root_dir)
    
    for mat_file in root_path.rglob("*.mat"):
        # Skip Zone.Identifier files
        if "Zone.Identifier" in str(mat_file):
            continue
        
        # Skip root-level configuration files (they have different structure)
        if mat_file.parent.name in ["10mmfocus", "15mmfocus"]:
            print(f"  Skipping config file: {mat_file.name}")
            continue
            
        mat_files.append(mat_file)
    
    return sorted(mat_files)


def convert_mat_to_npy(mat_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Convert a single .mat file to .npy format.
    
    The .mat file should contain:
    - IBuffer: (H, W, 1, T) float64 - In-phase component
    - QBuffer: (H, W, 1, T) float64 - Quadrature component
    
    Output .npy will be:
    - Complex128 array of shape (H, W, T) where data = IBuffer + j*QBuffer
    
    Args:
        mat_path: Path to the .mat file
        dry_run: If True, only preview without actual conversion
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    npy_path = mat_path.with_suffix('.npy')
    
    if dry_run:
        return True, f"Would convert: {mat_path} -> {npy_path}"
    
    try:
        # Load .mat file
        data = sio.loadmat(str(mat_path))
        
        # Check for required fields
        if 'IBuffer' not in data or 'QBuffer' not in data:
            return False, f"Missing IBuffer/QBuffer in {mat_path}"
        
        I_buffer = data['IBuffer']  # Shape: (H, W, 1, T)
        Q_buffer = data['QBuffer']  # Shape: (H, W, 1, T)
        
        # Validate shapes match
        if I_buffer.shape != Q_buffer.shape:
            return False, f"Shape mismatch: IBuffer {I_buffer.shape} vs QBuffer {Q_buffer.shape}"
        
        # Squeeze out the singleton dimension and create complex array
        # (H, W, 1, T) -> (H, W, T)
        I_buffer = np.squeeze(I_buffer, axis=2)
        Q_buffer = np.squeeze(Q_buffer, axis=2)
        
        # Combine into complex: IQ = I + jQ
        iq_complex = I_buffer + 1j * Q_buffer
        
        # Save as .npy
        np.save(str(npy_path), iq_complex)
        
        return True, f"Converted: {mat_path.name} -> {npy_path.name} | Shape: {iq_complex.shape} | dtype: {iq_complex.dtype}"
        
    except Exception as e:
        return False, f"Error converting {mat_path}: {str(e)}"


def delete_mat_file(mat_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Delete the original .mat file after successful conversion."""
    if dry_run:
        return True, f"Would delete: {mat_path}"
    
    try:
        mat_path.unlink()
        
        # Also delete Zone.Identifier file if it exists
        zone_id_path = Path(str(mat_path) + ":Zone.Identifier")
        if zone_id_path.exists():
            zone_id_path.unlink()
            
        return True, f"Deleted: {mat_path.name}"
    except Exception as e:
        return False, f"Error deleting {mat_path}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIRS Phantom .mat files to .npy format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversions without performing them"
    )
    parser.add_argument(
        "--keep-mat",
        action="store_true",
        help="Keep original .mat files after conversion"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/CIRS_Phantom_Latest",
        help="Path to the dataset directory"
    )
    
    args = parser.parse_args()
    
    # Determine the root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / args.data_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"{'='*60}")
    print(f"MAT to NPY Converter for CIRS Phantom Dataset")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Keep .mat files: {args.keep_mat}")
    print(f"{'='*60}\n")
    
    # Find all IQ .mat files
    print("Searching for .mat files with IQ data...")
    mat_files = find_iq_mat_files(data_dir)
    print(f"Found {len(mat_files)} .mat files to convert\n")
    
    if not mat_files:
        print("No .mat files found. Exiting.")
        return
    
    # Convert files
    print("Converting files...")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for mat_file in mat_files:
        # Convert
        success, message = convert_mat_to_npy(mat_file, dry_run=args.dry_run)
        print(message)
        
        if success:
            success_count += 1
            
            # Delete original .mat file unless --keep-mat is specified
            if not args.keep_mat:
                del_success, del_message = delete_mat_file(mat_file, dry_run=args.dry_run)
                if not del_success:
                    print(f"  Warning: {del_message}")
        else:
            fail_count += 1
    
    print("-" * 60)
    print(f"\nConversion complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified.")


if __name__ == "__main__":
    main()
