#!/usr/bin/env python3
"""
Why do outputs/comparison_videos look so much better than the abstract figures?

Both use the same checkpoints and the same forward pass.  The difference is what
the five input frames contain, and that is set entirely by the acquisition PRF,
because ``sparse_step = 12`` is counted in NATIVE frames:

    dt_in = 12 x dt_native
      20 kHz -> 0.6 ms   five knots span  2.4 ms
      10 kHz -> 1.2 ms   five knots span  4.8 ms
       5 kHz -> 2.4 ms   five knots span  9.6 ms
     500 Hz  -> 24  ms   five knots span 96   ms

A shear wave crosses the ROI in a few ms, so the kHz protocols sample it several
times and the 500 Hz protocol samples it once.

Energy is measured OUTSIDE the push column (+-15 px), because the push transient
in frame 0 is enormous and swamps the propagating wave in a plain ROI-wide RMS.

Usage:
    python scripts/diagnose_video_vs_abstract.py
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
UPSAMPLE = 12
KNOTS = np.arange(5) * UPSAMPLE
PUSH_EXCL = 15
PRF_DT = {"500Hz": 1 / 500, "5kHz": 1 / 5000,
          "10kHz": 1 / 10000, "20kHz": 1 / 20000}


def group_of(name: str) -> str:
    low = name.lower()
    if "cirs" in low:
        return "CIRS"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}FT" if m else "other"


def wave_energy_profile(vel: np.ndarray) -> np.ndarray:
    """Per-frame RMS of the depth-averaged field, excluding the push column —
    i.e. the energy actually carried by the propagating wave."""
    v = depth_average(np.asarray(vel, float), (0.25, 0.75))     # (T, W)
    push = find_push_column(v)
    W = v.shape[1]
    keep = np.ones(W, bool)
    keep[max(0, push - PUSH_EXCL):min(W, push + PUSH_EXCL + 1)] = False
    if keep.sum() < 10:
        keep[:] = True
    return np.sqrt((v[:, keep] ** 2).mean(axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--per-cell", type=int, default=6)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text())

    cells = collections.defaultdict(list)
    for split in ("val", "test", "train"):
        for e in meta[split]:
            prf = e.get("frequency")
            if prf not in PRF_DT:
                continue
            p = data_dir / split / f"velocity_{e['name']}.npy"
            if p.exists() and str(e.get("window", [0])[0]) == "0":
                cells[(group_of(e["name"]), prf)].append(p)

    rng = np.random.default_rng(0)
    stats = {}
    for (g, prf), paths in sorted(cells.items()):
        pick = paths if len(paths) <= args.per_cell else [
            paths[i] for i in rng.choice(len(paths), args.per_cell, replace=False)]
        fracs, spans = [], []
        for p in pick:
            try:
                v = np.load(p).astype(np.float64)[:, :H_ROI, :]
            except (ValueError, EOFError, OSError):
                continue
            if v.shape[0] < 49:
                continue
            e = wave_energy_profile(v)
            if e.max() <= 0:
                continue
            fracs.append(e[KNOTS] / e.max())
            spans.append(48 * PRF_DT[prf] * UPSAMPLE * 1e3 / UPSAMPLE)
        if fracs:
            stats[(g, prf)] = np.array(fracs).mean(axis=0)

    print("=" * 96)
    print("  PROPAGATING-WAVE ENERGY AT EACH OF THE 5 INPUT KNOTS")
    print("  (depth-averaged RMS outside the push column, normalised to that")
    print("   acquisition's peak frame)")
    print("=" * 96)
    for prf in ("20kHz", "10kHz", "5kHz", "500Hz"):
        dt_in = PRF_DT[prf] * UPSAMPLE
        print(f"\n  {prf}   dt_in = {dt_in * 1e3:6.2f} ms   "
              f"5 knots span {4 * dt_in * 1e3:6.1f} ms")
        print(f"    {'group':6s} " + "  ".join(f"{k * dt_in * 1e3:7.1f}ms"
                                               for k in range(5)))
        for g in ("1FT", "2FT", "3FT", "CIRS"):
            f = stats.get((g, prf))
            if f is None:
                continue
            n_informative = int((f > 0.15).sum())
            print(f"    {g:6s} " + "  ".join(f"{x:9.3f}" for x in f)
                  + f"   -> {n_informative} of 5 knots carry a wave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
