#!/usr/bin/env python3
"""
SAUS 2026 abstract — paired statistics against the 5 kHz reference.

Design
------
ONE IMAGING PLANE (one .mat file) = ONE SAMPLE.  Every row belonging to a plane
-- repeat acquisitions of it, and the analysis windows cut from it -- is averaged
first, so a plane contributes exactly one number per method and the SD is a
genuine between-sample SD.  Pooling those rows as if independent is
pseudo-replication: it inflates n and reports the repeatability of one
acquisition instead of the variability of the phantom.

Within that fixed set of planes the methods still differ in how often the
estimator converges.  That is a result, not a sampling choice, so it is reported
as a fit-success count rather than hidden: a paired t-test can only use the
planes where BOTH members of the pair produced a value.

For each method vs GT (5 kHz):
    paired t-test (two-sided) on the complete pairs, with mean difference,
    95% CI, Cohen's dz, and a Wilcoxon signed-rank companion for the small,
    non-normal samples.  p values are Holm-corrected across the three methods
    within each group x quantity family.

Usage:
    python scripts/generate_saus2026_stats.py --in outputs/saus2026_v2
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import saus2026_units as U  # noqa: E402

GROUPS = ["1FT", "2FT", "3FT"]
GROUP_TITLE = {"1FT": "1 freeze-thaw (FT) PVA phantom",
               "2FT": "2 freeze-thaw (FT) PVA phantom",
               "3FT": "3 freeze-thaw (FT) PVA phantom"}
METHODS = [("GT_5kHz", "GT (5 kHz)"), ("Linear", "Linear"),
           ("Cubic", "Cubic"), ("WaveFormer", "WaveFormer")]
TEST_METHODS = ["Linear", "Cubic", "WaveFormer"]
RHO = 1000.0

QUANTITIES = ["gprime", "eta", "cs", "elastic"]

W = 100


def fnum(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def holm(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjustment, NaN-safe."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out: List[Optional[float]] = [None] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


def stars(p: Optional[float]) -> str:
    if p is None:
        return "  -"
    return "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "ns "


def unit_table(rows: List[Dict], group: Optional[str], unit: str) -> Dict:
    """Units for one group, or all three groups pooled with disambiguated ids."""
    if group is not None:
        return U.units(rows, group, unit)
    merged: Dict = {}
    for g in GROUPS:
        for k, v in U.units(rows, g, unit).items():
            merged[f"{g}/{k}"] = v
    return merged


def paired(a: np.ndarray, b: np.ndarray) -> Dict:
    """a = method, b = ground truth, already matched element-wise."""
    d = a - b
    n = d.size
    out = dict(n=n, mean_diff=float(d.mean()) if n else float("nan"),
               sd_diff=float(d.std(ddof=1)) if n > 1 else float("nan"),
               ci=(float("nan"), float("nan")), t=None, df=n - 1,
               p=None, dz=float("nan"), wilcoxon_p=None)
    # A paired t on two values has 1 df: the "test" is then a ratio of two
    # numbers and its t and p are not interpretable, so it is not reported.
    if n < 3 or not np.isfinite(out["sd_diff"]) or out["sd_diff"] == 0:
        if n == 2:
            out["note"] = "n = 2, a paired t on two values is not interpretable"
        elif n >= 1 and out["sd_diff"] == 0:
            out["note"] = "zero variance in the differences"
        return out
    se = out["sd_diff"] / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    out["ci"] = (out["mean_diff"] - tcrit * se, out["mean_diff"] + tcrit * se)
    t, p = stats.ttest_rel(a, b)
    out["t"], out["p"] = float(t), float(p)
    out["dz"] = out["mean_diff"] / out["sd_diff"]
    if n >= 6:
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            pass
    return out


def msd(v: List[float]) -> str:
    if not v:
        return "no fit".rjust(17)
    a = np.array(v, float)
    return f"{a.mean():8.3f} +- {a.std(ddof=0):<6.3f}"


def block(fh, rows: List[Dict], group: Optional[str], field: str,
          unit: str) -> None:
    qname, unit_txt = U.FIELD_LABEL[field]
    u = unit_table(rows, group, unit)
    ids = U.common_units(u, field)
    n_common = len(ids)
    noun = "planes" if unit == "plane" else "windows"
    label = ("POOLED (1FT + 2FT + 3FT)" if group is None
             else f"{group} — {GROUP_TITLE[group]}")

    fh.write(f"\n{'-' * W}\n{label}\n")
    fh.write(f"{qname} [{unit_txt}]     {noun} with a 5 kHz reference: n = {n_common}"
             f"   (of {len(u)} present)\n")
    fh.write(f"{'-' * W}\n")

    if n_common == 0:
        fh.write(f"  no {noun[:-1]} in this group yields a 5 kHz reference fit; "
                 "no comparison possible.\n")
        return

    fh.write(f"  {'method':<12}{'fits':>8}{'mean +- SD':>20}"
             f"{'paired n':>10}{'mean diff':>12}{'95% CI':>22}\n")
    res = {}
    for key, name in U.METHODS:
        m, sd, nfit = U.summarise(u, ids, key, field)
        shown = f"{m:8.3f} +- {sd:<6.3f}" if nfit else "no fit".rjust(17)
        line = f"  {name:<12}{nfit:>3}/{n_common:<4}{shown:>20}"
        if key == "GT_5kHz":
            fh.write(line + f"{'reference':>10}\n")
            continue
        keep = [i for i in ids if np.isfinite(u[i][key][field])]
        r = paired(U.series(u, keep, key, field),
                   U.series(u, keep, "GT_5kHz", field))
        res[key] = r
        ci = ("      —" if not np.isfinite(r["ci"][0])
              else f"[{r['ci'][0]:+7.3f}, {r['ci'][1]:+7.3f}]")
        fh.write(line + f"{r['n']:>10}{r['mean_diff']:>+12.3f}{ci:>22}\n")

    adj = holm([res[k]["p"] for k in TEST_METHODS])
    fh.write(f"\n  Paired t-test vs GT (5 kHz), two-sided\n")
    fh.write(f"  {'method':<12}{'n':>4}{'df':>5}{'t':>10}{'p':>12}"
             f"{'p (Holm)':>12}{'':>5}{'Cohen dz':>10}{'Wilcoxon p':>13}\n")
    for key, pa in zip(TEST_METHODS, adj):
        r = res[key]
        if r["p"] is None:
            why = r.get("note", f"only {r['n']} paired value(s)")
            fh.write(f"  {key:<12}{r['n']:>4}{'':>5}  not testable — {why}\n")
            continue
        wp = "        —" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:9.4f}"
        fh.write(f"  {key:<12}{r['n']:>4}{r['df']:>5}{r['t']:>10.3f}{r['p']:>12.4f}"
                 f"{pa:>12.4f}{'':>1}{stars(pa):>4}{r['dz']:>10.3f}{wp:>13}\n")


def inventory(fh, rows: List[Dict]) -> None:
    fh.write("\n" + "=" * W + "\n")
    fh.write("SAMPLES USED — one imaging plane (.mat file) = one sample\n")
    fh.write("=" * W + "\n")
    for g in GROUPS:
        inv = U.plane_inventory(rows, g)
        fh.write(f"\n  {g}:  {len(inv)} plane(s)\n")
        for plane in sorted(inv):
            d = inv[plane]
            fh.write(f"    {plane:<34} {d['n_rows']:>2} row(s) = "
                     f"{len(d['inputs'])} input x {len(d['gts'])} reference "
                     f"acquisition(s), window(s) {','.join(sorted(d['windows']))}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="outputs/saus2026_v2")
    ap.add_argument("--window", default="0")
    ap.add_argument("--unit", default="plane", choices=["plane", "window"],
                    help="plane = one .mat file is one sample (default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = U.load_rows(in_dir / "saus2026_per_sample.csv", args.window)
    out = (Path(args.out) if args.out else
           in_dir / f"saus2026_paired_stats_{args.unit}.txt")

    with open(out, "w") as fh:
        fh.write("=" * W + "\n")
        fh.write("SAUS 2026 — WaveFormer temporal super-resolution of 500 Hz SWE\n")
        fh.write("Paired comparison of every method against the 5 kHz reference\n")
        fh.write("=" * W + "\n")
        fh.write(f"generated              {datetime.now():%Y-%m-%d %H:%M}\n")
        fh.write(f"source                 {in_dir / 'saus2026_per_sample.csv'}\n")
        fh.write("estimator              src/analysis/dispersion.py, version v2\n")
        fh.write("model                  outputs/wavepinn_v14/best_model.pt, epoch 189\n")
        fh.write("input                  5 consecutive 500 Hz frames (dt_in = 2 ms)\n")
        fh.write(f"windows                {'w0 (push-bearing)' if args.window == '0' else 'all'}"
                 ", frozen test split\n")
        fh.write(f"unit of analysis       {'one imaging plane (.mat file) = one sample' if args.unit == 'plane' else 'one analysis window = one observation'}\n")
        fh.write("\n")
        fh.write("SAMPLE DEFINITION\n")
        fh.write("  One imaging plane -- one .mat file -- is ONE sample.  Repeat acquisitions of a\n")
        fh.write("  plane and the analysis windows cut from it are averaged into that plane's\n")
        fh.write("  single value first, so the SD below is the spread ACROSS planes, not the\n")
        fh.write("  repeatability of one acquisition.\n")
        fh.write("  All four methods are summarised over the SAME planes: those on which the\n")
        fh.write("  5 kHz reference itself yields a valid fit.  A plane with no reference cannot\n")
        fh.write("  be paired against anything and is excluded for every method alike.\n")
        fh.write("  'fits' is how many of those planes each method converged on; a paired t-test\n")
        fh.write("  can only use the planes where both members of the pair produced a value, and\n")
        fh.write("  that n is stated for every test.\n")

        inventory(fh, rows)

        for field in QUANTITIES:
            qname, unit_txt = U.FIELD_LABEL[field]
            fh.write("\n\n" + "=" * W + "\n")
            fh.write(f"{qname.upper()}  [{unit_txt}]\n")
            fh.write("=" * W + "\n")
            for g in GROUPS:
                block(fh, rows, g, field, args.unit)
            block(fh, rows, None, field, args.unit)

        fh.write("\n\n" + "=" * W + "\n")
        fh.write("READING THESE NUMBERS\n")
        fh.write("=" * W + "\n")
        fh.write(
            "  mean diff   method minus ground truth; negative = method reads lower than 5 kHz\n"
            "  95% CI      confidence interval of that mean difference (paired t)\n"
            "  Cohen dz    mean difference divided by the SD of the differences\n"
            "  p (Holm)    Holm-Bonferroni adjusted across the three methods within each\n"
            "              group x quantity family;  *** p<0.001  ** p<0.01  * p<0.05  ns not sig.\n"
            "  Wilcoxon p  signed-rank companion, reported when n >= 6, since these samples are\n"
            "              small and the differences are not demonstrably normal\n"
            "\n"
            "  A non-significant result is NOT evidence of agreement.  With n = 2 to 9 per group\n"
            "  the power to detect a real difference is low, so 'ns' here means 'no difference\n"
            "  was resolved', not 'the methods agree'.  Read it together with the mean difference\n"
            "  and its confidence interval, which show how large a difference remains compatible\n"
            "  with the data.\n"
            "\n"
            "  Linear and Cubic are tested on far fewer pairs than WaveFormer because the\n"
            "  estimator does not converge on their reconstructions.  Their fit-success counts\n"
            "  are the more informative comparison, and the surviving pairs are a favourable\n"
            "  subset for them -- the windows where interpolation happened to work.\n"
            "\n"
            "  Shear modulus G' and viscosity eta come from the same Kelvin-Voigt fit and share\n"
            "  its bias: these acquisitions begin at the push, so the wave onset is missing from\n"
            "  the record and the fit reads apparent dispersion that the medium does not have.\n"
            "  Method-to-reference agreement is meaningful; the absolute values are not.\n"
            "  rho*cs^2 is free of that bias, being conditioned only on the time-of-arrival speed.\n")

    print(open(out).read())
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
