#!/usr/bin/env python3
"""
Turn the per-window rows written by ``generate_saus2026_values.py`` into the
tables the SAUS 2026 abstract needs.

Two window subsets are reported.  Every 500 Hz acquisition is cut into 50-frame
analysis windows: ``w0`` starts at the push and carries the shear wave, ``w1``
starts 100 ms later and contains no wave at all, so its dispersion fits are
noise.  ``w0`` is therefore the primary subset and ``all`` is kept only for
continuity with the earlier evaluation drivers.

Usage:
    python scripts/summarize_saus2026_values.py --in outputs/saus2026
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

CATEGORIES = ["1FT", "2FT", "3FT", "CIRS"]
METHODS = ["GT_5kHz", "Linear", "Cubic", "WaveFormer"]
PRETTY = {"GT_5kHz": "GT (5 kHz)", "Linear": "Linear",
          "Cubic": "Cubic", "WaveFormer": "WaveFormer"}


def stats(vals) -> Dict:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), median=float("nan"),
                    n=0)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=0)),
                median=float(np.median(a)), n=int(a.size))


def build(rows: List[Dict], subset: str) -> List[Dict]:
    if subset == "w0":
        rows = [r for r in rows if str(r["window"]) == "0"]
    out = []
    for cat in CATEGORIES:
        for meth in METHODS:
            sel = [r for r in rows if r["category"] == cat and r["method"] == meth]
            if not sel:
                continue
            cs = stats([r["cs_m_s"] for r in sel])
            eta = stats([r["eta_Pa_s"] for r in sel])
            gp = stats([r["G_prime_kPa"] for r in sel])
            cse = stats([r["cs_abs_err"] for r in sel])
            ete = stats([r["eta_abs_err"] for r in sel])
            out.append(dict(
                window_subset=subset, category=cat, method=PRETTY[meth],
                n_windows=len(sel),
                cs_mean_m_s=cs["mean"], cs_sd_m_s=cs["sd"],
                cs_median_m_s=cs["median"], cs_n_valid=cs["n"],
                eta_mean_Pa_s=eta["mean"], eta_sd_Pa_s=eta["sd"],
                eta_median_Pa_s=eta["median"], eta_n_valid=eta["n"],
                G_prime_mean_kPa=gp["mean"], G_prime_sd_kPa=gp["sd"],
                G_prime_n_valid=gp["n"],
                cs_MAE_vs_GT=0.0 if meth == "GT_5kHz" else cse["mean"],
                eta_MAE_vs_GT=0.0 if meth == "GT_5kHz" else ete["mean"],
            ))
    return out


FIELDS = ["window_subset", "category", "method", "n_windows",
          "cs_mean_m_s", "cs_sd_m_s", "cs_median_m_s", "cs_n_valid",
          "eta_mean_Pa_s", "eta_sd_Pa_s", "eta_median_Pa_s", "eta_n_valid",
          "G_prime_mean_kPa", "G_prime_sd_kPa", "G_prime_n_valid",
          "cs_MAE_vs_GT", "eta_MAE_vs_GT"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="outputs/saus2026")
    args = ap.parse_args()
    d = Path(args.indir)
    rows = json.loads((d / "saus2026_rows.json").read_text())

    table = build(rows, "w0") + build(rows, "all")
    out = d / "saus2026_barchart_values.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    for subset in ("w0", "all"):
        print("\n" + "=" * 96)
        print(f"  SUBSET: {subset}   "
              f"({'push window only — primary' if subset == 'w0' else 'every window'})")
        print("=" * 96)
        print(f"  {'cat':5s} {'method':11s} {'n':>3} {'cs (m/s)':>18} "
              f"{'eta (Pa.s)':>18} {'G0 (kPa)':>18}")
        for cat in CATEGORIES:
            for r in [r for r in table
                      if r["window_subset"] == subset and r["category"] == cat]:
                print(f"  {r['category']:5s} {r['method']:11s} {r['n_windows']:3d} "
                      f"{r['cs_mean_m_s']:8.2f}+-{r['cs_sd_m_s']:<8.2f} "
                      f"{r['eta_mean_Pa_s']:8.2f}+-{r['eta_sd_Pa_s']:<8.2f} "
                      f"{r['G_prime_mean_kPa']:8.2f}+-{r['G_prime_sd_kPa']:<8.2f}")
            print()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
