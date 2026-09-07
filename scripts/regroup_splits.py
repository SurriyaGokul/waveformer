#!/usr/bin/env python3
"""
Repair 500 Hz train/val split leakage and carve a frozen test set.

Problem: 12 acquisitions have one window (w0) in train and another (w1) in val,
making val metrics optimistic. Also, no true held-out test split exists.

What this script does:
  1. Groups all 500 Hz samples by acquisition base name (strip `_w<N>` suffix).
  2. Moves whole acquisitions to resolve the 12 straddling train/val cases:
     - Straddling acquisitions → train (keep more data in training).
  3. Carves a frozen test split (~20% of unique acquisitions, stratified by
     phantom source/stiffness, fixed seed=42).
  4. Writes the updated split to data/fullscale_velocity/metadata.json
     (backs up existing file to .bak3).
  5. Copies the test-split npy files into data/fullscale_velocity/test/.

WARNING: This script is DESTRUCTIVE — it modifies metadata.json and moves
data files. Run only when you are ready to retrain (Task 3).

Confirm before running:
    python scripts/regroup_splits.py --dry-run   # Show what would change
    python scripts/regroup_splits.py              # Apply changes

Usage:
    python scripts/regroup_splits.py [--dry-run] [--seed 42] [--test-frac 0.20]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "fullscale_velocity"
META_PATH    = DATA_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def acq_base(sample_name: str) -> str:
    """Strip _w<N> suffix to get acquisition base name."""
    return re.sub(r"_w\d+$", "", sample_name)


def load_meta() -> Dict:
    with open(META_PATH) as f:
        return json.load(f)


def backup_meta(meta_path: Path) -> Path:
    """Find the next available .bak<N> filename."""
    n = 1
    while True:
        bak = meta_path.with_suffix(f".json.bak{n}")
        if not bak.exists():
            shutil.copy2(meta_path, bak)
            return bak
        n += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print changes without writing anything.")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction of unique acquisitions to hold out.")
    args = parser.parse_args()

    meta = load_meta()

    # ---- Collect all 500 Hz samples ----------------------------------------
    all_500hz: Dict[str, Dict] = {}   # sample_name → sample_meta
    for split in ("train", "val", "inference"):
        for s in meta.get(split, []):
            if "500Hz" in s.get("name", ""):
                all_500hz[s["name"]] = {**s, "_split": split}

    # Group by acquisition base
    by_base: Dict[str, List[str]] = defaultdict(list)
    for name in all_500hz:
        by_base[acq_base(name)].append(name)

    # ---- Identify straddling acquisitions -----------------------------------
    train_names = {s["name"] for s in meta.get("train", []) if "500Hz" in s.get("name", "")}
    val_names   = {s["name"] for s in meta.get("val",   []) if "500Hz" in s.get("name", "")}
    train_bases = {acq_base(n) for n in train_names}
    val_bases   = {acq_base(n) for n in val_names}
    straddle_bases = sorted(train_bases & val_bases)

    print(f"Straddling acquisitions ({len(straddle_bases)}): "
          f"moving all their windows to train")
    for b in straddle_bases:
        print(f"  {b}")

    # ---- Build regrouped train/val ------------------------------------------
    # Decision: move straddling acquisitions entirely to train.
    # Non-straddling val acquisitions keep their current split assignment.
    new_train_samples: List[Dict] = []
    new_val_samples:   List[Dict] = []

    for name, s_meta in all_500hz.items():
        base = acq_base(name)
        orig_split = s_meta["_split"]
        if base in straddle_bases:
            new_train_samples.append({k: v for k, v in s_meta.items() if k != "_split"})
        elif orig_split == "train":
            new_train_samples.append({k: v for k, v in s_meta.items() if k != "_split"})
        else:  # val or inference
            new_val_samples.append({k: v for k, v in s_meta.items() if k != "_split"})

    # ---- Carve test split ---------------------------------------------------
    # Use unique acquisitions from the combined pool (not from the val only).
    # Stratify by phantom source (first token of name: "1ft", "2ft", "3ft", "goat").
    all_bases = sorted(by_base.keys())
    rng = np.random.default_rng(args.seed)

    # Group bases by source (phantom distance / type)
    by_source: Dict[str, List[str]] = defaultdict(list)
    for b in all_bases:
        source = b.split("_")[0]
        by_source[source].append(b)

    test_bases = set()
    for source, bases in sorted(by_source.items()):
        n_test = max(1, round(len(bases) * args.test_frac))
        chosen = list(rng.choice(bases, size=n_test, replace=False))
        test_bases.update(chosen)
        print(f"  {source}: {len(bases)} acquisitions → {n_test} test")

    print(f"\nTest acquisitions: {len(test_bases)}")

    # Move test samples out of train/val into test split
    final_train, final_val, final_test = [], [], []
    for s in new_train_samples:
        (final_test if acq_base(s["name"]) in test_bases else final_train).append(s)
    for s in new_val_samples:
        (final_test if acq_base(s["name"]) in test_bases else final_val).append(s)

    print(f"\nSplit sizes (500 Hz samples):")
    print(f"  train: {len(final_train)}")
    print(f"  val:   {len(final_val)}")
    print(f"  test:  {len(final_test)}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ---- Apply changes to metadata.json ------------------------------------
    # Backup
    bak = backup_meta(META_PATH)
    print(f"\nBacked up metadata to {bak}")

    # Rebuild metadata: replace 500 Hz entries, keep all non-500Hz intact.
    def non_500hz(split_key):
        return [s for s in meta.get(split_key, []) if "500Hz" not in s.get("name", "")]

    meta["train"] = non_500hz("train") + final_train
    meta["val"]   = non_500hz("val")   + final_val
    # Merge inference: keep non-500Hz inference entries
    if "inference" not in meta:
        meta["inference"] = []
    meta["inference"] = non_500hz("inference")
    meta["test"] = final_test

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {META_PATH}")

    # ---- Reconcile physical npy files with the new split assignment ---------
    # IQVelocityDataset._discover_samples() globs sparse_*.npy directly from
    # each split directory — it does NOT read metadata.json. So the JSON
    # relabeling above has zero effect on what the trainer actually sees
    # unless the files themselves are moved to match. This step MOVES each
    # 500 Hz sample's npy triplet into its target split dir and deletes any
    # stale copies left in other split dirs (train/val/test/inference), so
    # physical layout == metadata.json (no train/val straddle, no test leak).
    (DATA_DIR / "test").mkdir(exist_ok=True)
    target_split: Dict[str, str] = {}
    for split, samples in (("train", final_train), ("val", final_val), ("test", final_test)):
        for s in samples:
            target_split[s["name"]] = split

    prefixes = ("sparse_", "velocity_", "magnitude_")
    search_splits = ("train", "val", "test", "inference")
    moved = 0
    removed_dupes = 0
    missing = []

    for name, tgt in target_split.items():
        present_in = [sp for sp in search_splits if (DATA_DIR / sp / f"sparse_{name}.npy").exists()]
        if not present_in:
            missing.append(name)
            continue

        if tgt not in present_in:
            src_split = present_in[0]
            for prefix in prefixes:
                src = DATA_DIR / src_split / f"{prefix}{name}.npy"
                if src.exists():
                    shutil.move(str(src), str(DATA_DIR / tgt / f"{prefix}{name}.npy"))
            moved += 1
            present_in = [sp for sp in present_in if sp != src_split] + [tgt]

        # Delete stale duplicates left in any split dir other than the target.
        for sp in present_in:
            if sp == tgt:
                continue
            for prefix in prefixes:
                dup = DATA_DIR / sp / f"{prefix}{name}.npy"
                if dup.exists():
                    dup.unlink()
                    removed_dupes += 1

    if missing:
        print(f"WARNING: {len(missing)} samples in metadata have no physical npy file: {missing[:5]}...")
    print(f"Moved {moved} acquisitions to their target split dir; removed {removed_dupes} stale duplicate files.")
    print("\nDone. Retrain with the new split (Task 3).")


if __name__ == "__main__":
    main()
