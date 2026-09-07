#!/usr/bin/env python3
"""
Phase A3 — validate the rebuilt dispersion estimator on ground truth only.

Runs src/analysis/dispersion.py over native high-PRF acquisitions and reports:
  * success rate and failure reasons (old estimator: 37% success on this data)
  * per-phantom G', eta, cs and the conventional elastic modulus rho*cs^2
  * GT-GT repeatability, split into
        WITHIN-acquisition   (windows of one recording)
        ACROSS-acquisition   (repeat pushes at the same plane)
    which is the noise floor every model-vs-GT claim is measured against
  * band-perturbation robustness (f_hi +/- 20%)

Estimator defaults are frozen here, on ground truth only, BEFORE any model
output is evaluated.  Re-running this after changing DispersionConfig is what
justifies (or refuses) such a change.

Usage:
    python scripts/validate_dispersion_gt.py
    python scripts/validate_dispersion_gt.py --glob 'data/eval_v1/velocity_*5kHz*.npy'
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import glob as globmod
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dispersion import (  # noqa: E402
    DEFAULT_CONFIG,
    LATERAL_M_PX,
    estimate_cs_toa,
    estimate_dispersion,
    estimate_dispersion_phase_gradient,
)

NAME_RE = re.compile(r"(.+?)_(\d+k?Hz)_(\d{8}T\d+)(?:_w(\d+))?$")


def parse_name(path: str):
    stem = Path(path).stem.replace("velocity_", "")
    m = NAME_RE.match(stem)
    if not m:
        return dict(plane=stem, prf="?", ts="?", window="0", phantom=phantom_of(stem))
    return dict(plane=m.group(1), prf=m.group(2), ts=m.group(3),
                window=m.group(4) or "0", phantom=phantom_of(stem))


def phantom_of(name: str) -> str:
    """Phantom label.  The old pipeline (evaluate_wavepinn_v10.py:545) labelled
    every 1ft_* sample as '2ft'; this parses the actual prefix."""
    low = name.lower()
    if "cirs" in low:
        return "cirs"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}ft" if m else "other"


def cv_pct(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.mean(x) == 0:
        return float("nan")
    return float(100.0 * np.std(x) / abs(np.mean(x)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/eval_test_split_merged/velocity_*5kHz*.npy")
    ap.add_argument("--dt", type=float, default=2e-4, help="native frame interval [s]")
    ap.add_argument("--out", default="outputs/dispersion_gt_validation")
    ap.add_argument("--phase-gradient", action="store_true",
                    help="also run the phase-gradient cross-check (slower)")
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))
    if not files:
        print(f"No files matched {args.glob}")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("  Phase A3 — dispersion estimator validation on GROUND TRUTH")
    print(f"  {len(files)} acquisitions   dt = {args.dt * 1e3:.3f} ms   "
          f"config = {DEFAULT_CONFIG.version}")
    print("=" * 100)

    recs = []
    for f in files:
        vel = np.load(f).astype(np.float64)
        meta = parse_name(f)
        toa = estimate_cs_toa(vel, args.dt)
        r = estimate_dispersion(vel, args.dt, cs_toa_hint=toa["cs"])
        rec = dict(file=Path(f).name, **meta, cs=toa["cs"], cs_std=toa["cs_std"],
                   toa_step_px=toa["step_px"], **r.as_dict())
        if args.phase_gradient:
            pg = estimate_dispersion_phase_gradient(vel, args.dt, cs_toa_hint=toa["cs"])
            rec["pg_ok"] = pg.ok
            rec["pg_G_prime_kpa"] = pg.G_prime_kpa
            rec["pg_eta"] = pg.eta
        recs.append(rec)

    n_ok = sum(r["ok"] for r in recs)
    print(f"\nSUCCESS RATE: {n_ok}/{len(recs)} = {100 * n_ok / len(recs):.0f}%")
    bad = collections.Counter(r["reason"] for r in recs if not r["ok"])
    if bad:
        print(f"  failures: {dict(bad)}")
    warn = collections.Counter(r["reason"] for r in recs if r["ok"] and r["reason"])
    if warn:
        print(f"  fit warnings: {dict(warn)}")

    # ── Per-phantom summary ──────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("PER-PHANTOM (mean +/- sd over acquisitions)")
    print("-" * 100)
    print(f"{'phantom':10s} {'n':>3} {'cs (m/s)':>14} {'G_KV (kPa)':>16} "
          f"{'eta (Pa.s)':>15} {'rho*cs^2 (kPa)':>15} {'aperture mm':>12}")
    by_ph = collections.defaultdict(list)
    for r in recs:
        if r["ok"]:
            by_ph[r["phantom"]].append(r)
    for ph in sorted(by_ph):
        v = by_ph[ph]
        f = lambda k: np.array([x[k] for x in v], dtype=float)  # noqa: E731
        def ms(k):
            a = f(k); a = a[np.isfinite(a)]
            return f"{np.mean(a):6.2f}+-{np.std(a):<5.2f}" if a.size else "   n/a    "
        print(f"{ph:10s} {len(v):3d} {ms('cs'):>14} {ms('G_prime_kpa'):>16} "
              f"{ms('eta'):>15} {ms('G_elastic_kpa'):>15} "
              f"{np.nanmean(f('aperture_mm')):12.1f}")

    # ── Repeatability ────────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("GT-GT REPEATABILITY  (the noise floor for every model-vs-GT claim)")
    print("-" * 100)

    within = collections.defaultdict(list)
    across = collections.defaultdict(dict)
    for r in recs:
        if not r["ok"]:
            continue
        within[(r["plane"], r["ts"])].append(r)
        across[r["plane"]].setdefault(r["ts"], r)

    def report(groups, title):
        rows = []
        print(f"\n{title}")
        print(f"  {'group':50s} {'n':>2} {'cs CV%':>7} {'G CV%':>7} {'eta CV%':>8}")
        for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
            v = list(members.values()) if isinstance(members, dict) else members
            if len(v) < 2:
                continue
            t = (cv_pct([x["cs"] for x in v]),
                 cv_pct([x["G_prime_kpa"] for x in v]),
                 cv_pct([x["eta"] for x in v]))
            rows.append(t)
            label = key if isinstance(key, str) else " ".join(key)
            print(f"  {label[:50]:50s} {len(v):2d} {t[0]:7.1f} {t[1]:7.1f} {t[2]:8.1f}")
        if rows:
            a = np.array(rows, dtype=float)
            med = [float(np.nanmedian(a[:, i])) for i in range(3)]
            print(f"  {'MEDIAN':50s} {len(rows):2d} {med[0]:7.1f} {med[1]:7.1f} {med[2]:8.1f}")
            return dict(n_groups=len(rows), cs_cv=med[0], G_cv=med[1], eta_cv=med[2])
        print("  (no multi-member groups)")
        return dict(n_groups=0)

    rep_within = report(within, "WITHIN-acquisition (windows of one recording):")
    rep_across = report(across, "ACROSS-acquisition (repeat pushes, same plane):")

    # ── Band-perturbation robustness ─────────────────────────────────────────
    print("\n" + "-" * 100)
    print("BAND ROBUSTNESS  (f_max +/- 20%; median |change| over acquisitions)")
    print("-" * 100)
    pert = {}
    # Perturb the limit that actually binds.  f_hi = min(f_max_abs_hz,
    # f_max_nyq_frac * f_Nyq); at 5 kHz the Nyquist fraction is the binding one,
    # so scaling f_max_abs_hz alone would be a no-op test.
    for tag, scale in [("f_hi -20%", 0.8), ("f_hi +20%", 1.2)]:
        cfg2 = dataclasses.replace(
            DEFAULT_CONFIG,
            f_max_abs_hz=DEFAULT_CONFIG.f_max_abs_hz * scale,
            f_max_nyq_frac=DEFAULT_CONFIG.f_max_nyq_frac * scale,
        )
        dg, de = [], []
        for f, base in zip(files, recs):
            if not base["ok"]:
                continue
            vel = np.load(f).astype(np.float64)
            r2 = estimate_dispersion(vel, args.dt, cfg=cfg2, cs_toa_hint=base["cs"])
            if not r2.ok:
                continue
            if base["G_prime_kpa"]:
                dg.append(abs(r2.G_prime_kpa - base["G_prime_kpa"]) / base["G_prime_kpa"])
            if base["eta"]:
                de.append(abs(r2.eta - base["eta"]) / base["eta"])
        pert[tag] = dict(dG=100 * float(np.median(dg)) if dg else float("nan"),
                         deta=100 * float(np.median(de)) if de else float("nan"))
        print(f"  {tag:12s}  median |dG'| = {pert[tag]['dG']:5.1f}%   "
              f"median |d eta| = {pert[tag]['deta']:5.1f}%")

    # ── Acceptance gates ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PRE-REGISTERED ACCEPTANCE GATES")
    print("=" * 100)
    etas = np.array([r["eta"] for r in recs if r["ok"]], dtype=float)
    etas = etas[np.isfinite(etas)]
    gates = [
        ("success rate >= 90%", 100 * n_ok / len(recs) >= 90,
         f"{100 * n_ok / len(recs):.0f}%"),
        ("median eta in [0.3, 5] Pa.s", 0.3 <= np.median(etas) <= 5.0,
         f"{np.median(etas):.2f} Pa.s"),
        ("across-acq G' CV <= 15%", rep_across.get("G_cv", np.inf) <= 15,
         f"{rep_across.get('G_cv', float('nan')):.1f}%"),
        ("across-acq eta CV <= 25%", rep_across.get("eta_cv", np.inf) <= 25,
         f"{rep_across.get('eta_cv', float('nan')):.1f}%"),
        ("band perturbation |dG'| <= 25%",
         max(pert[t]["dG"] for t in pert) <= 25,
         f"{max(pert[t]['dG'] for t in pert):.1f}%"),
        ("band perturbation |d eta| <= 25%",
         max(pert[t]["deta"] for t in pert) <= 25,
         f"{max(pert[t]['deta'] for t in pert):.1f}%"),
    ]
    all_pass = True
    for name, passed, val in gates:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:36s} -> {val}")
        all_pass &= bool(passed)

    summary = dict(
        config=dataclasses.asdict(DEFAULT_CONFIG),
        n_files=len(recs), n_ok=n_ok,
        failures=dict(bad), warnings=dict(warn),
        repeatability_within=rep_within, repeatability_across=rep_across,
        band_perturbation=pert,
        gates={n: bool(p) for n, p, _ in gates},
        all_gates_pass=bool(all_pass),
    )
    (out_dir / "gt_validation_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "gt_validation_per_file.json", "w") as fh:
        json.dump(recs, fh, indent=2, default=float)
    print(f"\nWrote {out_dir}/gt_validation_summary.json and gt_validation_per_file.json")
    print("=" * 100)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
