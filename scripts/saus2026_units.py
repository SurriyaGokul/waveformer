#!/usr/bin/env python3
"""
Unit of analysis for the SAUS 2026 comparisons.

The per-sample CSV has one row per (500 Hz input window x 5 kHz acquisition).
Several of those rows come from the SAME imaging plane -- the same .mat file
re-acquired, or a second analysis window cut from it.  Treating them as
independent observations is pseudo-replication: it inflates n and shrinks the SD
towards the repeatability of one acquisition rather than the variability of the
phantom.  2FT is the clearest case -- 6 "samples" that are really 2 planes.

unit = "plane"  (default)
    One imaging plane = one .mat file = ONE sample.  Every row belonging to a
    plane is averaged first, so each plane contributes a single number per
    method, and the SD reported across planes is a genuine between-sample SD.

unit = "window"
    The previous behaviour: one row = one observation.  Kept so the two can be
    compared, not because it is the right denominator.
"""
from __future__ import annotations

import collections
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RHO = 1000.0
GROUPS = ["1FT", "2FT", "3FT"]
METHODS = [("GT_5kHz", "GT (5 kHz)"), ("Linear", "Linear"),
           ("Cubic", "Cubic"), ("WaveFormer", "WaveFormer")]
FIELDS = ["gprime", "elastic", "eta", "cs"]
FIELD_LABEL = {
    "gprime": ("Shear modulus G'", "kPa"),
    "elastic": ("Elastic modulus rho*cs^2", "kPa"),
    "eta": ("Viscosity eta", "Pa.s"),
    "cs": ("Shear wave speed", "m/s"),
}


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _value(row: Dict, field: str) -> float:
    if field == "cs":
        return _f(row["cs_m_s"])
    if field == "gprime":
        return _f(row["G_prime_kPa"])
    if field == "eta":
        return _f(row["eta_Pa_s"])
    if field == "elastic":
        cs = _f(row["cs_m_s"])
        return RHO * cs ** 2 / 1000.0 if np.isfinite(cs) else float("nan")
    raise KeyError(field)


def load_rows(path: Path, window: str = "0") -> List[Dict]:
    return [r for r in csv.DictReader(open(path))
            if r["category"] in GROUPS and window in ("all", r["window"])]


def units(rows: List[Dict], group: str, unit: str = "plane"
          ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collapse rows to one observation per unit.

    Returns {unit_id: {method: {field: value}}}, value NaN where the estimator
    did not converge anywhere in that unit.
    """
    if unit not in ("plane", "window"):
        raise ValueError(f"unit must be 'plane' or 'window', got {unit!r}")

    bucket: Dict[str, Dict[str, List[Dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in rows:
        if r["category"] != group:
            continue
        uid = r["plane"] if unit == "plane" else \
            f"{r['input_500hz']} | {r['gt_file']}"
        bucket[uid][r["method"]].append(r)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for uid, by_method in bucket.items():
        entry: Dict[str, Dict[str, float]] = {}
        for key, _label in METHODS:
            vals: Dict[str, float] = {}
            for field in FIELDS:
                v = [_value(r, field) for r in by_method.get(key, [])]
                v = [x for x in v if np.isfinite(x)]
                vals[field] = float(np.mean(v)) if v else float("nan")
            entry[key] = vals
        out[uid] = entry
    return out


def common_units(u: Dict, field: str) -> List[str]:
    """Units on which the 5 kHz reference itself yields a value for `field`.

    A unit with no reference cannot be compared against anything, so it is
    excluded for every method alike -- which is what keeps the denominator
    identical across methods.
    """
    return sorted(k for k, v in u.items()
                  if np.isfinite(v["GT_5kHz"][field]))


def series(u: Dict, ids: List[str], method: str, field: str) -> np.ndarray:
    return np.array([u[id_][method][field] for id_ in ids], float)


def summarise(u: Dict, ids: List[str], method: str, field: str
              ) -> Tuple[float, float, int]:
    """mean, SD (population, as plotted), and how many units converged."""
    a = series(u, ids, method, field)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


def plane_inventory(rows: List[Dict], group: str) -> Dict[str, Dict]:
    """Which acquisitions and windows went into each plane, for the audit CSV."""
    out: Dict[str, Dict] = {}
    for r in rows:
        if r["category"] != group:
            continue
        d = out.setdefault(r["plane"], dict(inputs=set(), gts=set(), windows=set(),
                                            n_rows=0))
        d["inputs"].add(r["input_500hz"])
        d["gts"].add(r["gt_file"])
        d["windows"].add(r["window"])
        d["n_rows"] += 1
    return out
