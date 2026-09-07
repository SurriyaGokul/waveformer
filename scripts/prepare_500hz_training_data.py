#!/usr/bin/env python3
"""
Prepare 500 Hz inference data for training.

Copies 500 Hz sparse/velocity/magnitude files from inference/ to train/ and val/,
and updates metadata.json accordingly.

This closes the critical distribution gap: training data only covers 5-20 kHz
(dt_sparse = 0.6-2.4 ms), while 100 Hz inference needs dt = 10 ms.  Adding
500 Hz data (dt_sparse = 24 ms with sparse_step=12) bridges this gap.

Usage:
    python scripts/prepare_500hz_training_data.py
    python scripts/prepare_500hz_training_data.py --data_dir data/fullscale_velocity --val_ratio 0.2
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Copy 500Hz inference data to train/val")
    parser.add_argument("--data_dir", type=str, default="data/fullscale_velocity")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of 500Hz samples for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "metadata.json"

    with open(meta_path) as f:
        meta = json.load(f)

    # Find 500Hz inference samples
    inf_500 = [s for s in meta["inference"] if s.get("frequency") == "500Hz"]
    print(f"Found {len(inf_500)} 500Hz inference samples")

    if not inf_500:
        print("No 500Hz inference samples found. Exiting.")
        sys.exit(1)

    # Check which are already in train/val
    existing_train = {s["name"] for s in meta.get("train", [])}
    existing_val = {s["name"] for s in meta.get("val", [])}
    inf_500 = [s for s in inf_500 if s["name"] not in existing_train and s["name"] not in existing_val]

    if not inf_500:
        print("All 500Hz samples already in train/val. Nothing to do.")
        sys.exit(0)

    print(f"{len(inf_500)} new 500Hz samples to add")

    # Shuffle and split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(inf_500))
    n_val = max(1, int(len(inf_500) * args.val_ratio))
    n_train = len(inf_500) - n_val

    val_samples = [inf_500[i] for i in indices[:n_val]]
    train_samples = [inf_500[i] for i in indices[n_val:]]

    print(f"Split: {n_train} train, {n_val} val")

    # Tissue type distribution
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        tissues = {}
        for s in samples:
            t = s.get("source", "?")
            tissues[t] = tissues.get(t, 0) + 1
        print(f"  {split_name}: {tissues}")

    if args.dry_run:
        print("\n[DRY RUN] Would copy files and update metadata. Exiting.")
        sys.exit(0)

    # Backup metadata
    backup_path = data_dir / "metadata.json.bak2"
    shutil.copy2(meta_path, backup_path)
    print(f"Backed up metadata to {backup_path}")

    # Copy files
    inf_dir = data_dir / "inference"
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        split_dir = data_dir / split_name
        copied = 0
        for sample in samples:
            name = sample["name"]
            for prefix in ("sparse_", "velocity_", "magnitude_"):
                src = inf_dir / f"{prefix}{name}.npy"
                dst = split_dir / f"{prefix}{name}.npy"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                elif not src.exists():
                    print(f"  WARNING: {src} not found!")
        print(f"Copied {copied} files to {split_name}/")

    # Update metadata
    meta["train"].extend(train_samples)
    meta["val"].extend(val_samples)
    # Remove copied samples from inference list
    moved_names = {s["name"] for s in train_samples + val_samples}
    meta["inference"] = [s for s in meta["inference"] if s["name"] not in moved_names]

    # Update counts
    meta["sample_counts"]["train"] = len(meta["train"])
    meta["sample_counts"]["val"] = len(meta["val"])
    meta["sample_counts"]["inference"] = len(meta["inference"])

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nUpdated metadata.json:")
    print(f"  train: {meta['sample_counts']['train']} samples")
    print(f"  val:   {meta['sample_counts']['val']} samples")
    print(f"  inference: {meta['sample_counts']['inference']} samples")
    print("\nDone! 500Hz data is now available for training.")


if __name__ == "__main__":
    main()
