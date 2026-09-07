"""
Clean the data/fullscale_velocity dataset by removing samples that contain no
meaningful shear-wave signal.

Definition of "empty frame":
    velocity_std_mms < MIN_STD_THRESHOLD  (default: 1.0 mm/s)

Such samples contain essentially only noise — the shear-wave pulse either has
not yet arrived or has already passed — and add no training value.

Usage:
    python scripts/clean_fullscale_velocity.py
    python scripts/clean_fullscale_velocity.py --data_dir data/fullscale_velocity
    python scripts/clean_fullscale_velocity.py --threshold 0.5 --dry_run
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Only clean training splits; inference is kept as-is (may legitimately have
# low velocity std at low PRFs such as 500 Hz / 1 kHz).
TRAINING_SPLITS = ("train", "val")
ALL_SPLITS = ("train", "val", "inference")


def clean_split(
    split_dir: Path,
    samples: list,
    threshold: float,
    dry_run: bool,
) -> tuple[list, int]:
    """
    Remove .npy triplets for samples below the velocity-std threshold.

    Returns (kept_samples, n_removed).
    """
    kept = []
    removed = 0

    for sample in samples:
        name = sample.get("name", "")
        vel_std = sample.get("velocity_std_mms", 0.0)

        if vel_std < threshold:
            # Identify the three files for this sample
            sparse = split_dir / f"sparse_{name}.npy"
            velocity = split_dir / f"velocity_{name}.npy"
            magnitude = split_dir / f"magnitude_{name}.npy"

            logger.info(
                f"  [REMOVE] {name}  "
                f"vel_std={vel_std:.3f} mm/s  (< {threshold})"
            )

            if not dry_run:
                for f in (sparse, velocity, magnitude):
                    if f.exists():
                        f.unlink()
                    else:
                        logger.warning(f"    File not found: {f.name}")
            removed += 1
        else:
            kept.append(sample)

    return kept, removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove empty frames from data/fullscale_velocity"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/fullscale_velocity",
        help="Root directory of the fullscale_velocity dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum velocity_std_mms to keep a sample (default: 1.0 mm/s)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be removed without deleting anything",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info(f"=== clean_fullscale_velocity [{mode}] ===")
    logger.info(f"Data dir  : {data_dir.resolve()}")
    logger.info(f"Threshold : {args.threshold} mm/s")

    total_removed = 0

    logger.info(
        "Note: inference split is left untouched (low PRF acquisitions have "
        "naturally low velocity std but are still valid for evaluation)."
    )

    for split in TRAINING_SPLITS:
        if split not in metadata:
            logger.warning(f"Split '{split}' not found in metadata, skipping")
            continue

        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}, skipping")
            continue

        samples = metadata[split]
        n_before = len(samples)

        logger.info(f"\n[{split}]  {n_before} samples before cleaning")
        kept, n_removed = clean_split(
            split_dir, samples, args.threshold, args.dry_run
        )
        n_after = len(kept)
        total_removed += n_removed

        logger.info(
            f"  Removed {n_removed}, kept {n_after} "
            f"({100 * n_after / n_before:.1f}%)"
        )

        if not args.dry_run:
            metadata[split] = kept
            # Update sample_counts
            if "sample_counts" in metadata:
                metadata["sample_counts"][split] = n_after

    logger.info(f"\nTotal removed: {total_removed}")

    if not args.dry_run and total_removed > 0:
        # Back up original metadata before overwriting
        backup_path = metadata_path.with_suffix(".json.bak")
        shutil.copy2(metadata_path, backup_path)
        logger.info(f"Original metadata backed up to {backup_path.name}")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Updated {metadata_path.name} saved.")
    elif args.dry_run:
        logger.info("Dry run complete — no files deleted.")


if __name__ == "__main__":
    main()
