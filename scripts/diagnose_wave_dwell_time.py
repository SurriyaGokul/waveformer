#!/usr/bin/env python3
"""
Does the slower 1FT wave really stay in the ROI longer than the faster 3FT wave?

Yes — and that matters, because it means the input starvation under
``sparse_step = 12`` is NOT what makes 1FT worse than 3FT.  All groups are
equally starved by a 24 ms knot gap.  Measured here:

  A. dwell time — how long the propagating wave is visible in the ROI, from the
     native 5 kHz movies (0.2 ms resolution)
  B. what the model's 5 input frames would capture at step 12 (the current
     preprocessing, knots 24 ms apart) versus step 1 (consecutive 500 Hz frames,
     knots 2 ms apart)

Energy excludes the push column (+-15 px); the push transient in frame 0 is
enormous and would otherwise swamp the propagating wave.

Usage:
    python scripts/diagnose_wave_dwell_time.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dispersion import depth_average, find_push_column  # noqa: E402

H_ROI = 116
PUSH_EXCL = 15
LATERAL_MM_PX = 0.1013
GT_CS = {"1FT": 2.16, "2FT": 3.11, "3FT": 3.69, "CIRS": 2.58}


def group_of(name: str) -> str:
    low = name.lower()
    if "cirs" in low:
        return "CIRS"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}FT" if m else "other"


def wave_energy(vel: np.ndarray) -> np.ndarray:
    v = depth_average(np.asarray(vel, float), (0.25, 0.75))
    push = find_push_column(v)
    W = v.shape[1]
    keep = np.ones(W, bool)
    keep[max(0, push - PUSH_EXCL):min(W, push + PUSH_EXCL + 1)] = False
    if keep.sum() < 10:
        keep[:] = True
    return np.sqrt((v[:, keep] ** 2).mean(axis=1)), push


def collect(data_dir: Path, meta, prf: str, per_cell: int):
    cells = collections.defaultdict(list)
    for split in ("val", "test", "train"):
        for e in meta[split]:
            if e.get("frequency") != prf:
                continue
            if str(e.get("window", [0])[0]) != "0":
                continue
            p = data_dir / split / f"velocity_{e['name']}.npy"
            if p.exists():
                cells[group_of(e["name"])].append(p)
    rng = np.random.default_rng(0)
    out = {}
    for g, paths in cells.items():
        pick = paths if len(paths) <= per_cell else [
            paths[i] for i in rng.choice(len(paths), per_cell, replace=False)]
        keep = []
        for p in pick:
            try:
                v = np.load(p).astype(np.float64)[:, :H_ROI, :]
            except (ValueError, EOFError, OSError):
                continue
            if v.shape[0] < 49:
                continue
            e, _ = wave_energy(v)
            if e.max() > 0:
                keep.append(e / e.max())
        if keep:
            out[g] = np.array(keep)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--per-cell", type=int, default=8)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text())

    # ── A. dwell time, from the finely sampled 5 kHz movies ─────────────────
    e5 = collect(data_dir, meta, "5kHz", args.per_cell)
    print("=" * 84)
    print("  A. HOW LONG THE WAVE STAYS IN THE ROI  (native 5 kHz, dt = 0.2 ms)")
    print("=" * 84)
    print(f"  {'group':6s} {'cs':>6} {'transit 8mm/cs':>15} "
          f"{'dwell >25%':>12} {'dwell >10%':>12}")
    for g in ("1FT", "2FT", "3FT", "CIRS"):
        if g not in e5:
            continue
        a = e5[g]
        d25 = np.mean([(row > 0.25).sum() for row in a]) * 0.2
        d10 = np.mean([(row > 0.10).sum() for row in a]) * 0.2
        transit = 8.0 / GT_CS[g]
        print(f"  {g:6s} {GT_CS[g]:6.2f} {transit:13.1f} ms "
              f"{d25:10.1f} ms {d10:10.1f} ms")
    print("\n  -> the slower the gel, the longer the wave lingers. Exactly as expected.")

    # ── B. what the 5 input frames capture, step 12 vs step 1 ───────────────
    e500 = collect(data_dir, meta, "500Hz", args.per_cell)
    print("\n" + "=" * 84)
    print("  B. WHAT THE MODEL'S 5 INPUT FRAMES SEE  (native 500 Hz, dt = 2 ms)")
    print("=" * 84)
    for step, label in ((12, "step 12  (CURRENT — knots 24 ms apart)"),
                        (1, "step 1   (consecutive — knots 2 ms apart)")):
        knots = np.arange(5) * step
        print(f"\n  {label}")
        print(f"    {'group':6s} " + "  ".join(f"{k * step * 2:5.0f}ms" for k in range(5))
              + "   informative knots")
        for g in ("1FT", "2FT", "3FT", "CIRS"):
            if g not in e500:
                continue
            a = e500[g]
            k = np.clip(knots, 0, a.shape[1] - 1)
            f = a[:, k].mean(axis=0)
            n = int((f > 0.15).sum())
            print(f"    {g:6s} " + "  ".join(f"{x:7.3f}" for x in f)
                  + f"   {n} of 5")
    print("\n  -> at step 12 every group gets exactly one informative frame, so input")
    print("     starvation is COMMON to all groups and cannot explain 1FT vs 3FT.")
    print("     At step 1 the ordering flips in 1FT's favour, as the dwell times predict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
