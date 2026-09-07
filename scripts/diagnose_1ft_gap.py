#!/usr/bin/env python3
"""
Why does 1FT do worse than 2FT/3FT/CIRS?

Measures the shear-wave speed the model was actually trained to produce, per
phantom group and per acquisition PRF, using the same time-of-arrival estimator
the evaluation uses.  If the model's test-time predictions cluster near the
training mean regardless of the phantom, the 1FT error is dynamic-range
compression toward that mean rather than anything specific to 1FT.

Usage:
    python scripts/diagnose_1ft_gap.py --per-cell 18
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

from src.analysis.dispersion import estimate_cs_toa  # noqa: E402

H_ROI = 116
PRF_DT = {"5kHz": 1 / 5000, "10kHz": 1 / 10000, "20kHz": 1 / 20000,
          "500Hz": 1 / 500}


def group_of(name: str) -> str:
    low = name.lower()
    if "cirs" in low:
        return "CIRS"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}FT" if m else "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--per-cell", type=int, default=18,
                    help="max acquisitions per (group, PRF) cell")
    ap.add_argument("--out", default="outputs/saus2026/diagnose_1ft.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text())

    # Dense supervision = every acquisition whose native grid is finer than the
    # dense output grid, i.e. the kHz PRFs.  These are what taught the model
    # what a propagating shear wave looks like.
    cells = collections.defaultdict(list)
    for split in ("train", "val"):
        for e in meta[split]:
            prf = e.get("frequency")
            if prf not in ("5kHz", "10kHz", "20kHz"):
                continue
            p = data_dir / split / f"velocity_{e['name']}.npy"
            if p.exists():
                cells[(group_of(e["name"]), prf)].append(p)

    rng = np.random.default_rng(0)
    rows = []
    for (g, prf), paths in sorted(cells.items()):
        pick = paths if len(paths) <= args.per_cell else [
            paths[i] for i in rng.choice(len(paths), args.per_cell, replace=False)]
        for p in pick:
            try:
                v = np.load(p).astype(np.float64)[:, :H_ROI, :]
            except (ValueError, EOFError, OSError) as exc:   # truncated .npy on disk
                print(f"    [skip] {p.name}: {exc}")
                continue
            cs = float(estimate_cs_toa(v, PRF_DT[prf])["cs"])
            rows.append(dict(group=g, prf=prf, file=p.name, cs=cs))
        done = [r["cs"] for r in rows if r["group"] == g and r["prf"] == prf]
        ok = [c for c in done if np.isfinite(c)]
        print(f"  {g:5s} {prf:6s} n={len(done):3d} valid={len(ok):3d} "
              f"cs={np.mean(ok) if ok else float('nan'):.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 78)
    print("  SHEAR WAVE SPEED IN THE DENSE TRAINING DATA (m/s)")
    print("=" * 78)
    print(f"  {'group':6s} {'n':>4} {'valid':>6} {'mean':>7} {'sd':>7} {'median':>7}")
    allc = []
    for g in ["1FT", "2FT", "3FT", "CIRS"]:
        c = np.array([r["cs"] for r in rows if r["group"] == g], float)
        c = c[np.isfinite(c)]
        n = sum(1 for r in rows if r["group"] == g)
        allc.append(c)
        if c.size:
            print(f"  {g:6s} {n:4d} {c.size:6d} {c.mean():7.2f} {c.std():7.2f} "
                  f"{np.median(c):7.2f}")
    pooled = np.concatenate(allc)
    print(f"  {'ALL':6s} {len(rows):4d} {pooled.size:6d} {pooled.mean():7.2f} "
          f"{pooled.std():7.2f} {np.median(pooled):7.2f}")
    print(f"\n  Count-weighted training mean speed: {pooled.mean():.2f} m/s")
    print(f"  Median:                             {np.median(pooled):.2f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
