#!/usr/bin/env python3
"""
SAUS 2026 abstract, bottom bar panel — rebuilt in the style of the submitted
figure (panels b/c of SAUS Abstract 2026_20260829.docx), for the three PVA
freeze-thaw groups only.

One block per phantom group; each block is a pair of bar charts, shear modulus
and viscosity, over GT (5 kHz) / Linear / Cubic / WaveFormer.  Bars are
mean +- SD across the w0 (push-bearing) analysis windows of the frozen test
split; a method with no valid fit in a group is marked "Failed" as in the
original figure.

Values come from scripts/generate_saus2026_values.py, i.e. from
src/analysis/dispersion.py at estimator version v2.

The left panel defaults to the Kelvin-Voigt storage modulus G', so each block
is the (G', eta) pair from one fit.  ``--modulus elastic`` plots rho*cs^2
instead (conditioned only on the time-of-arrival speed, independent of the fit);
``--modulus speed`` restores the original shear wave speed panel.

Usage:
    python scripts/generate_saus2026_barpanel.py --in outputs/saus2026_v2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import saus2026_units as U  # noqa: E402

GROUPS = [("1FT", "1 freeze-thaw (FT) PVA phantom"),
          ("2FT", "2 freeze-thaw (FT) PVA phantom"),
          ("3FT", "3 freeze-thaw (FT) PVA phantom")]
PANEL_LETTERS = ["b", "c", "d"]
METHODS = U.METHODS

RHO = 1000.0            # kg/m^3, for rho*cs^2

# What goes in the left-hand panel of each block.
MODULUS = {
    "gprime":  ("gprime",  "Shear modulus (kPa)"),
    "elastic": ("elastic", "Shear modulus (kPa)"),
    "speed":   ("cs",      "Shear wave speed (m/s)"),
}

GREY = "#C9C9C9"
BLUE = "#2E75B6"        # WaveFormer, modulus panel
ORANGE = "#E1701A"      # WaveFormer, viscosity panel
EDGE = "#1a1a1a"


def stats(rows: List[Dict], cat: str, unit: str) -> Dict[str, Dict]:
    """Per-method summaries with ONE IMAGING PLANE as one sample.

    Repeat acquisitions of a plane and the analysis windows cut from it are
    averaged into that plane's single value first, so the error bars are the
    spread across planes rather than the repeatability of one acquisition.
    Every method is summarised over the same planes: those on which the 5 kHz
    reference itself yields a value for that quantity.
    """
    u = U.units(rows, cat, unit)
    out: Dict[str, Dict] = {k: {} for k, _ in U.METHODS}
    out["_n_common"] = {}
    for field in U.FIELDS:
        ids = U.common_units(u, field)
        out["_n_common"][field] = len(ids)
        for key, _label in U.METHODS:
            out[key][field] = U.summarise(u, ids, key, field)
    return out


def draw(ax, st: Dict[str, Dict], field: str, ylabel: str, hi_color: str) -> None:
    x = np.arange(len(METHODS))
    means = [st[k][field][0] for k, _ in METHODS]
    sds = [st[k][field][1] for k, _ in METHODS]
    ns = [st[k][field][2] for k, _ in METHODS]
    colors = [GREY] * (len(METHODS) - 1) + [hi_color]

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.86", lw=0.8)
    ax.xaxis.grid(False)

    plot_m = [0.0 if not np.isfinite(m) else m for m in means]
    plot_s = [0.0 if not np.isfinite(s) else s for s in sds]
    ax.bar(x, plot_m, width=0.68, color=colors, edgecolor=EDGE, linewidth=1.2,
           yerr=plot_s, error_kw=dict(ecolor="black", elinewidth=1.6,
                                      capthick=1.6, capsize=5))

    # Reference line at the 5 kHz ground truth, as in the submitted figure.
    gt = st["GT_5kHz"][field][0]
    top = max([m + s for m, s in zip(plot_m, plot_s)] + [gt if np.isfinite(gt) else 0])
    ax.set_ylim(0, top * 1.18 if top > 0 else 1)
    if np.isfinite(gt):
        ax.axhline(gt, color="0.35", ls="--", lw=1.3, zorder=1)

    for xi, n in zip(x, ns):
        if n == 0:
            ax.text(xi, ax.get_ylim()[1] * 0.06, "Failed", color="#D0021B",
                    fontsize=11, fontweight="bold", rotation=90,
                    ha="center", va="bottom")

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METHODS], rotation=30, ha="right",
                       fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
        sp.set_color(EDGE)


BAR_FIELDS = [
    "panel", "group", "method", "n_windows",
    "G_prime_mean_kPa", "G_prime_sd_kPa", "G_prime_n_valid",
    "rho_cs2_mean_kPa", "rho_cs2_sd_kPa", "rho_cs2_n_valid",
    "eta_mean_Pa_s", "eta_sd_Pa_s", "eta_n_valid",
    "cs_mean_m_s", "cs_sd_m_s", "cs_n_valid",
]

SRC_FIELDS = [
    "group", "method", "plane", "window", "input_500hz", "gt_5kHz_file",
    "cs_m_s", "G_prime_kPa", "rho_cs2_kPa", "eta_Pa_s",
    "fit_r2", "fit_ok", "fit_reason",
]


def r4(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.4f}"


def write_bar_csv(path: Path, blocks: List[tuple]) -> None:
    """The 24 numbers the bars are drawn from, plus the n behind each."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BAR_FIELDS)
        w.writeheader()
        for gi, (cat, _title, st) in enumerate(blocks):
            for key, label in METHODS:
                s_ = st[key]
                w.writerow(dict(
                    panel=f"({PANEL_LETTERS[gi]})", group=cat, method=label,
                    n_windows=st["_n_common"]["gprime"],
                    G_prime_mean_kPa=r4(s_["gprime"][0]),
                    G_prime_sd_kPa=r4(s_["gprime"][1]),
                    G_prime_n_valid=s_["gprime"][2],
                    rho_cs2_mean_kPa=r4(s_["elastic"][0]),
                    rho_cs2_sd_kPa=r4(s_["elastic"][1]),
                    rho_cs2_n_valid=s_["elastic"][2],
                    eta_mean_Pa_s=r4(s_["eta"][0]),
                    eta_sd_Pa_s=r4(s_["eta"][1]),
                    eta_n_valid=s_["eta"][2],
                    cs_mean_m_s=r4(s_["cs"][0]),
                    cs_sd_m_s=r4(s_["cs"][1]),
                    cs_n_valid=s_["cs"][2],
                ))


def write_source_csv(path: Path, rows: List[Dict], unit: str) -> None:
    """One row per sample x method: the plane-level values the bars average."""
    fields = ["group", "sample_plane", "n_rows_averaged", "method",
              "G_prime_kPa", "rho_cs2_kPa", "eta_Pa_s", "cs_m_s"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cat, _title in GROUPS:
            u = U.units(rows, cat, unit)
            inv = U.plane_inventory(rows, cat)
            for uid in sorted(u):
                for key, label in U.METHODS:
                    v = u[uid][key]
                    w.writerow(dict(
                        group=cat, sample_plane=uid,
                        n_rows_averaged=inv.get(uid, {}).get("n_rows", ""),
                        method=label,
                        G_prime_kPa=r4(v["gprime"]), rho_cs2_kPa=r4(v["elastic"]),
                        eta_Pa_s=r4(v["eta"]), cs_m_s=r4(v["cs"]),
                    ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="outputs/saus2026_v2")
    ap.add_argument("--window", default="0",
                    help='"0" for the push-bearing w0 windows, "all" for every window')
    ap.add_argument("--modulus", default="gprime", choices=sorted(MODULUS),
                    help="left panel of each block: gprime = Kelvin-Voigt "
                         "storage modulus (default), elastic = rho*cs^2, "
                         "speed = the original shear wave speed")
    ap.add_argument("--unit", default="plane", choices=["plane", "window"],
                    help="plane = one .mat file is one sample (default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = U.load_rows(in_dir / "saus2026_per_sample.csv", args.window)
    suffix = {"gprime": "", "elastic": "_elastic", "speed": "_speed"}[args.modulus]
    out = (Path(args.out) if args.out
           else in_dir / f"fig_b_bar_panel_pva{suffix}.png")
    mod_field, mod_label = MODULUS[args.modulus]

    fig = plt.figure(figsize=(15.0, 4.3))
    gs = fig.add_gridspec(1, 6, wspace=0.62, left=0.045, right=0.995,
                          top=0.80, bottom=0.235)

    print(f"{'group':6s} {'method':12s} {'n':>3} {'cs (m/s)':>16} "
          f"{"G' KV (kPa)":>16} {'rho.cs2 (kPa)':>16} {'eta (Pa.s)':>16} {'n_fit':>6}")
    blocks = []
    for gi, (cat, title) in enumerate(GROUPS):
        st = stats(rows, cat, args.unit)
        blocks.append((cat, title, st))
        ax_cs = fig.add_subplot(gs[0, 2 * gi])
        ax_et = fig.add_subplot(gs[0, 2 * gi + 1])
        draw(ax_cs, st, mod_field, mod_label, BLUE)
        draw(ax_et, st, "eta", "Viscosity (Pa · s)", ORANGE)

        # Group title centred over the pair, matching the submitted layout.
        b1 = ax_cs.get_position()
        b2 = ax_et.get_position()
        fig.text((b1.x0 + b2.x1) / 2, 0.925,
                 f"({PANEL_LETTERS[gi]}) {title}",
                 ha="center", va="center", fontsize=13, fontweight="bold")

        for key, label in METHODS:
            s = st[key]
            print(f"{cat:6s} {label:12s} {st['_n_common']['gprime']:3d} "
                  f"{s['cs'][0]:7.2f}+-{s['cs'][1]:<7.2f} "
                  f"{s['gprime'][0]:7.2f}+-{s['gprime'][1]:<7.2f} "
                  f"{s['elastic'][0]:7.2f}+-{s['elastic'][1]:<7.2f} "
                  f"{s['eta'][0]:7.2f}+-{s['eta'][1]:<7.2f} {s['eta'][2]:6d}")

    fig.savefig(out, dpi=400, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")

    bar_csv = out.with_name(out.stem + "_bar_values.csv")
    src_csv = out.with_name(out.stem + "_source_data.csv")
    write_bar_csv(bar_csv, blocks)
    write_source_csv(src_csv, rows, args.unit)

    print(f"\nWrote {out}")
    print(f"Wrote {out.with_suffix('.pdf')}")
    print(f"Wrote {bar_csv}   (the plotted mean +- SD)")
    print(f"Wrote {src_csv}   (every measurement behind those means)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
