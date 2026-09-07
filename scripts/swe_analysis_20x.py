#!/usr/bin/env python3
"""
SWE Analysis for 20× Upsampling Model — Publication-Grade Report
=================================================================
Computes shear wave speed (cs), shear modulus (G'), Young's modulus (E),
and viscosity (η) from the 20× model predictions at three virtual frequencies:

  Pred 10 kHz  — from 500 Hz IQ input  (compare with GT 10 kHz)
  Pred  5 kHz  — from 250 Hz IQ input  (compare with GT  5 kHz)
  Pred  2 kHz  — from 100 Hz IQ input  (compare with GT  1 kHz)

Also includes:
  GT 500 Hz    — native 500 Hz (SWE-degraded baseline)
  Pred 6 kHz   — 12× model baseline (previous model)

Matching is done by phantom group (source_focus_plane).

Usage
-----
    python scripts/swe_analysis_20x.py
    python scripts/swe_analysis_20x.py --output_dir outputs/swe_analysis_20x
"""
import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Physical / spatial constants
# ---------------------------------------------------------------------------
RHO_KG_M3         = 1000.0
LATERAL_MM_PER_PX  = 0.1013
AXIAL_MM_PER_PX    = 0.0507
LATERAL_M_PER_PX   = 0.1013e-3
H_ROI, W_ROI       = 116, 160

# Known phantom elastic moduli (CIRS spec)
PHANTOM_PROPS = {
    "2ft": {"E_kPa": 2.0, "G_kPa": 2.0/3, "cs_nominal": 0.816,
            "label": "2FT (E≈2 kPa)"},
    "3ft": {"E_kPa": 3.0, "G_kPa": 1.0,  "cs_nominal": 1.0,
            "label": "3FT (E≈3 kPa)"},
}

# Voigt model parameters
F_PUSH = 400.0
OMEGA_PUSH = 2.0 * np.pi * F_PUSH
G_ELASTIC = {"2ft": 2000.0/3.0, "3ft": 1000.0}  # Pa

# Condition metadata for plotting
COND_META = {
    "10kHz_gt":   {"label": "GT 10 kHz",           "color": "#1565C0", "marker": "o", "ls": "-"},
    "5kHz_gt":    {"label": "GT 5 kHz",             "color": "#2E7D32", "marker": "s", "ls": "-"},
    "1kHz_gt":    {"label": "GT 1 kHz",             "color": "#F57F17", "marker": "^", "ls": "-"},
    "500Hz_gt":   {"label": "GT 500 Hz",            "color": "#BF360C", "marker": "v", "ls": "--"},
    "pred_10kHz": {"label": "Pred 10 kHz (20×)",    "color": "#6A1B9A", "marker": "D", "ls": "-."},
    "pred_5kHz":  {"label": "Pred 5 kHz (20×)",     "color": "#00838F", "marker": "P", "ls": "-."},
    "pred_2kHz":  {"label": "Pred 2 kHz (20×)",     "color": "#AD1457", "marker": "*", "ls": ":"},
    "pred_6kHz":  {"label": "Pred 6 kHz (12×)",     "color": "#E65100", "marker": "X", "ls": ":"},
}

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.framealpha": 0.85,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


# ===========================================================================
# SWE Physics Functions
# ===========================================================================

def _find_push_column(v_abs, early_frames=3):
    T, W = v_abs.shape
    early = v_abs[:min(early_frames, T), :]
    return int(np.argmax(early.max(axis=0)))


def estimate_cs_toa(velocity, dt_s, dx_m,
                    min_amplitude_pct=20.0,
                    depth_frac=(0.2, 0.8),
                    push_exclusion_px=15):
    """Shear wave speed via cross-correlation time-of-arrival."""
    T, H, W = velocity.shape
    row_lo, row_hi = int(H * depth_frac[0]), int(H * depth_frac[1])
    v_abs = np.mean(np.abs(velocity[:, row_lo:row_hi, :]), axis=1)

    peak_amp = v_abs.max(axis=0)
    amp_thresh = np.percentile(peak_amp, min_amplitude_pct)
    amp_valid = peak_amp > amp_thresh

    x_push = _find_push_column(v_abs, early_frames=max(3, int(0.1*T)))
    excl_lo, excl_hi = max(0, x_push-push_exclusion_px), min(W-1, x_push+push_exclusion_px)
    push_excl = np.zeros(W, dtype=bool)
    push_excl[excl_lo:excl_hi+1] = True
    valid_cols = amp_valid & ~push_excl

    min_step = max(1, int(np.ceil(0.15 * dt_s / dx_m)))
    STEP = max(5, min_step)

    v_avg = np.mean(velocity[:, row_lo:row_hi, :], axis=1)
    cs_estimates = []

    for x in range(0, W - STEP):
        x2 = x + STEP
        if not (valid_cols[x] and valid_cols[x2]):
            continue
        sig_a, sig_b = v_avg[:, x], v_avg[:, x2]
        ea, eb = np.sqrt(np.sum(sig_a**2)), np.sqrt(np.sum(sig_b**2))
        if ea < 1e-12 or eb < 1e-12:
            continue
        xcorr = np.correlate(sig_a/ea, sig_b/eb, mode='full')
        lags = np.arange(-(T-1), T)
        pi = int(np.argmax(xcorr))

        if 0 < pi < len(xcorr)-1:
            y0, y1, y2 = xcorr[pi-1], xcorr[pi], xcorr[pi+1]
            denom = y0 - 2*y1 + y2
            frac = 0.5*(y0-y2)/denom if abs(denom) > 1e-12 else 0.0
            peak_lag = float(lags[pi]) + frac
        else:
            peak_lag = float(lags[pi])

        if abs(peak_lag) < 0.05:
            continue
        delay_s = peak_lag * dt_s
        cs_local = abs(STEP * dx_m / delay_s)
        if 0.2 < cs_local < 15.0:
            cs_estimates.append(cs_local)

    t_peak = np.argmax(v_abs, axis=0).astype(float) * dt_s
    x_arr = np.arange(W, dtype=float) * dx_m

    if not cs_estimates:
        return np.nan, np.nan, t_peak, x_arr

    cs_arr = np.array(cs_estimates)
    cs_med = float(np.median(cs_arr))
    cs_iqr = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    cs_std = cs_iqr / 1.349
    if not (0.2 < cs_med < 15.0):
        cs_med, cs_std = np.nan, np.nan

    return cs_med, cs_std, t_peak, x_arr


def voigt_phase_velocity(omega, G_prime_Pa, eta_Pa_s):
    numer = 2 * (G_prime_Pa**2 + (omega * eta_Pa_s)**2)
    denom = RHO_KG_M3 * (np.sqrt(G_prime_Pa**2 + (omega * eta_Pa_s)**2) + G_prime_Pa)
    return np.sqrt(numer / denom)


def voigt_invert(cs_m_s, source):
    """Invert Voigt model for viscosity given cs and known G_e."""
    nan_row = dict(G_prime_kpa=np.nan, eta_Pas=np.nan,
                   G_loss_kpa=np.nan, G_star_kpa=np.nan,
                   tan_delta=np.nan, E_kpa=np.nan)
    if not np.isfinite(cs_m_s) or cs_m_s <= 0:
        return nan_row
    G_e = G_ELASTIC.get(source, np.nan)
    if not np.isfinite(G_e):
        return nan_row

    cs2rho = cs_m_s**2 * RHO_KG_M3
    a, b, c = 2.0, -cs2rho, -cs2rho * G_e
    disc = b**2 - 4*a*c
    if disc < 0:
        return nan_row
    y = (-b + np.sqrt(disc)) / (2*a)
    omega_eta_sq = y**2 - G_e**2
    if omega_eta_sq < 0:
        return nan_row
    omega_eta = np.sqrt(omega_eta_sq)
    eta = omega_eta / OMEGA_PUSH

    G_loss = omega_eta / 1000.0
    G_prime = G_e / 1000.0
    G_star = y / 1000.0
    tan_d = G_loss / G_prime if G_prime > 0 else np.nan
    E_kpa = 3.0 * G_prime

    if not (0.001 <= eta <= 50.0):
        return nan_row
    return dict(G_prime_kpa=G_prime, eta_Pas=eta, G_loss_kpa=G_loss,
                G_star_kpa=G_star, tan_delta=tan_d, E_kpa=E_kpa)


def compute_swe_metrics(velocity, dt_s, label=""):
    """Full SWE pipeline on a single velocity map → dict of metrics."""
    T, H, W = velocity.shape
    cs, cs_std, t_arr, x_arr = estimate_cs_toa(
        velocity, dt_s, LATERAL_M_PER_PX,
        min_amplitude_pct=15.0, depth_frac=(0.2, 0.8))

    if np.isfinite(cs):
        G_kPa = RHO_KG_M3 * cs**2 / 1000
        G_std = 2 * RHO_KG_M3 * cs * cs_std / 1000 if np.isfinite(cs_std) else np.nan
        E_kPa = 3 * G_kPa
        E_std = 3 * G_std if np.isfinite(G_std) else np.nan
    else:
        G_kPa = G_std = E_kPa = E_std = np.nan

    return dict(cs_m_s=cs, cs_std=cs_std,
                G_prime_kPa=G_kPa, G_prime_std=G_std,
                E_kPa=E_kPa, E_std=E_std,
                n_frames=T, dt_s=dt_s,
                temporal_coverage_ms=T*dt_s*1000,
                t_peak=t_arr, x_arr=x_arr)


def parse_group(name):
    """Extract source_focus_Plane_N group key from sample name."""
    m = re.match(r'((?:2ft|3ft|cirs)_\w+focus)_(?:Plane_(\d+))', name.replace(' ', '_'))
    if m:
        return f"{m.group(1)}_Plane_{m.group(2)}"
    # Try with region format "Plane N"
    m2 = re.match(r'((?:2ft|3ft|cirs)_\w+focus)', name)
    return m2.group(1) if m2 else None


def extract_source(name):
    """Extract phantom source (2ft, 3ft, cirs) from name."""
    m = re.match(r'(2ft|3ft|cirs)', name.lower())
    return m.group(1) if m else None


# ===========================================================================
# Data Loading
# ===========================================================================

def load_predictions(pred_dir, freq_tag):
    """Load velocity_<freq_tag>_*.npy from prediction directory."""
    pred_dir = Path(pred_dir)
    result = {}
    for p in sorted(pred_dir.glob(f"velocity_{freq_tag}_*.npy")):
        base = p.stem.replace(f"velocity_{freq_tag}_", "")
        result[base] = np.load(str(p))
    return result


def load_gt_at_frequency(metadata_path, target_prf_hz):
    """Load all GT velocity samples at a specific PRF."""
    meta = json.load(open(metadata_path))
    result = {}
    splits = {"train": meta["train"], "val": meta["val"], "inference": meta["inference"]}

    for split_name, entries in splits.items():
        split_dir = f"data/fullscale_velocity/{split_name}"
        for s in entries:
            if s["prf_hz"] != target_prf_hz:
                continue
            vel_path = Path(split_dir) / f"velocity_{s['name']}.npy"
            if not vel_path.exists():
                continue
            result[s["name"]] = dict(
                velocity=np.load(str(vel_path)),
                dt_s=1.0 / s["prf_hz"],
                source=s["source"],
                focus=s["focus"],
                region=s.get("region", ""),
                group=parse_group(s["name"]),
                name=s["name"],
            )
    return result


# ===========================================================================
# Main Analysis
# ===========================================================================

def run_analysis(output_dir="outputs/swe_analysis_20x",
                 gt_meta_path="data/fullscale_velocity/metadata.json"):

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  SWE ANALYSIS — 20× Upsampling Model (Publication Report)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load predictions (20× model at 3 virtual frequencies)
    # ------------------------------------------------------------------
    print("\n[1/6] Loading 20× model predictions ...")
    pred_10kHz = load_predictions("outputs/10khz_predictions_20x", "10khz")
    pred_5kHz  = load_predictions("outputs/5khz_predictions_20x", "5khz")
    pred_2kHz  = load_predictions("outputs/2khz_predictions_20x", "2khz")
    print(f"      Pred 10 kHz: {len(pred_10kHz)} samples")
    print(f"      Pred  5 kHz: {len(pred_5kHz)} samples")
    print(f"      Pred  2 kHz: {len(pred_2kHz)} samples")

    # Also load 12× baseline if available
    pred_6kHz = load_predictions("outputs/6khz_predictions", "6khz")
    print(f"      Pred  6 kHz (12× baseline): {len(pred_6kHz)} samples")

    # ------------------------------------------------------------------
    # 2. Load GT velocity at comparison frequencies
    # ------------------------------------------------------------------
    print("\n[2/6] Loading ground-truth velocity data ...")
    gt_10kHz = load_gt_at_frequency(gt_meta_path, 10000.0)
    gt_5kHz  = load_gt_at_frequency(gt_meta_path, 5000.0)
    gt_1kHz  = load_gt_at_frequency(gt_meta_path, 1000.0)
    gt_500Hz = load_gt_at_frequency(gt_meta_path, 500.0)
    print(f"      GT 10 kHz: {len(gt_10kHz)} samples")
    print(f"      GT  5 kHz: {len(gt_5kHz)} samples")
    print(f"      GT  1 kHz: {len(gt_1kHz)} samples")
    print(f"      GT 500 Hz: {len(gt_500Hz)} samples")

    # ------------------------------------------------------------------
    # 3. Compute SWE metrics for all conditions
    # ------------------------------------------------------------------
    print("\n[3/6] Computing SWE metrics ...")
    rows = []

    def process_samples(samples, condition, dt_s, source_type="pred"):
        for name, data in samples.items():
            if source_type == "pred":
                vel = data
            else:
                vel = data["velocity"]
            src = extract_source(name)
            grp = parse_group(name)

            m = compute_swe_metrics(vel, dt_s)

            # Voigt inversion
            vi = voigt_invert(m["cs_m_s"], src) if src else {
                "G_prime_kpa": np.nan, "eta_Pas": np.nan,
                "G_loss_kpa": np.nan, "G_star_kpa": np.nan,
                "tan_delta": np.nan, "E_kpa": np.nan}

            rows.append(dict(
                name=name, condition=condition, source=src,
                group=grp,
                cs_m_s=m["cs_m_s"], cs_std=m["cs_std"],
                G_prime_kPa=m["G_prime_kPa"], G_prime_std=m["G_prime_std"],
                E_kPa=m["E_kPa"], E_std=m["E_std"],
                n_frames=m["n_frames"], dt_ms=dt_s*1000,
                temporal_span_ms=m["temporal_coverage_ms"],
                # Voigt
                G_voigt_kPa=vi["G_prime_kpa"],
                eta_Pa_s=vi["eta_Pas"],
                G_loss_kPa=vi["G_loss_kpa"],
                G_star_kPa=vi["G_star_kpa"],
                tan_delta=vi["tan_delta"],
                E_voigt_kPa=vi["E_kpa"],
            ))

    # Predictions
    print("      Processing Pred 10 kHz (20×) ...")
    process_samples(pred_10kHz, "pred_10kHz", dt_s=1/10000)
    print("      Processing Pred 5 kHz (20×) ...")
    process_samples(pred_5kHz,  "pred_5kHz",  dt_s=1/5000)
    print("      Processing Pred 2 kHz (20×) ...")
    process_samples(pred_2kHz,  "pred_2kHz",  dt_s=1/2000)

    # 12× baseline
    if pred_6kHz:
        print("      Processing Pred 6 kHz (12×) ...")
        process_samples(pred_6kHz, "pred_6kHz", dt_s=1/6000)

    # Ground truths
    print("      Processing GT 10 kHz ...")
    process_samples(gt_10kHz, "10kHz_gt", dt_s=1/10000, source_type="gt")
    print("      Processing GT 5 kHz ...")
    process_samples(gt_5kHz,  "5kHz_gt",  dt_s=1/5000,  source_type="gt")
    print("      Processing GT 1 kHz ...")
    process_samples(gt_1kHz,  "1kHz_gt",  dt_s=1/1000,  source_type="gt")
    print("      Processing GT 500 Hz ...")
    process_samples(gt_500Hz, "500Hz_gt", dt_s=1/500,   source_type="gt")

    df = pd.DataFrame(rows)
    df.to_csv(out / "swe_metrics_all.csv", index=False)
    print(f"      Total rows: {len(df)}")

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    print("\n[4/6] Computing summary statistics ...")
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  SWE ANALYSIS — 20× Upsampling Model (Publication Report)")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Temporal coverage table
    report_lines.append("TEMPORAL COVERAGE")
    report_lines.append("-" * 60)
    for cond in ["10kHz_gt", "5kHz_gt", "1kHz_gt", "500Hz_gt",
                 "pred_10kHz", "pred_5kHz", "pred_2kHz", "pred_6kHz"]:
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        n_frames = sub["n_frames"].iloc[0]
        dt = sub["dt_ms"].iloc[0]
        span = sub["temporal_span_ms"].iloc[0]
        meta = COND_META.get(cond, {})
        label = meta.get("label", cond)
        report_lines.append(f"  {label:<25s}  dt={dt:.3f} ms  "
                          f"N={n_frames:3d} frames  span={span:.1f} ms")
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL MEAN ± STD BY CONDITION")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Condition':<25s}{'n':>5s}  {'cs (m/s)':>14s}  "
                       f"{'G′ (kPa)':>14s}  {'E (kPa)':>14s}  {'η (Pa·s)':>12s}")
    report_lines.append("-" * 80)

    cond_order = ["10kHz_gt", "5kHz_gt", "1kHz_gt", "500Hz_gt",
                  "pred_10kHz", "pred_5kHz", "pred_2kHz", "pred_6kHz"]
    summary_rows = []

    for cond in cond_order:
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        cs_valid = sub["cs_m_s"].dropna()
        G_valid = sub["G_prime_kPa"].dropna()
        E_valid = sub["E_kPa"].dropna()
        eta_valid = sub["eta_Pa_s"].dropna()

        n = len(cs_valid)
        meta = COND_META.get(cond, {})
        label = meta.get("label", cond)

        cs_str = f"{cs_valid.mean():.3f}±{cs_valid.std():.3f}" if n > 0 else "N/A"
        G_str = f"{G_valid.mean():.2f}±{G_valid.std():.2f}" if len(G_valid) > 0 else "N/A"
        E_str = f"{E_valid.mean():.2f}±{E_valid.std():.2f}" if len(E_valid) > 0 else "N/A"
        eta_str = f"{eta_valid.mean():.2f}±{eta_valid.std():.2f}" if len(eta_valid) > 0 else "N/A"

        report_lines.append(f"  {label:<23s}{n:>5d}  {cs_str:>14s}  "
                          f"{G_str:>14s}  {E_str:>14s}  {eta_str:>12s}")
        summary_rows.append(dict(condition=cond, label=label, n=n,
                                 cs_mean=cs_valid.mean() if n>0 else np.nan,
                                 cs_std=cs_valid.std() if n>0 else np.nan,
                                 G_mean=G_valid.mean() if len(G_valid)>0 else np.nan,
                                 G_std=G_valid.std() if len(G_valid)>0 else np.nan,
                                 E_mean=E_valid.mean() if len(E_valid)>0 else np.nan,
                                 E_std=E_valid.std() if len(E_valid)>0 else np.nan,
                                 eta_mean=eta_valid.mean() if len(eta_valid)>0 else np.nan,
                                 eta_std=eta_valid.std() if len(eta_valid)>0 else np.nan))

    # Per-phantom breakdown
    for phantom in ["2ft", "3ft"]:
        report_lines.append("")
        report_lines.append(f"PER-PHANTOM: {phantom.upper()} "
                          f"({PHANTOM_PROPS.get(phantom, {}).get('label', '')})")
        report_lines.append("-" * 80)
        report_lines.append(f"  Nominal: cs={PHANTOM_PROPS[phantom]['cs_nominal']:.2f} m/s  "
                          f"G′={PHANTOM_PROPS[phantom]['G_kPa']:.2f} kPa  "
                          f"E={PHANTOM_PROPS[phantom]['E_kPa']:.1f} kPa")
        report_lines.append(f"{'Condition':<25s}{'n':>5s}  {'cs (m/s)':>14s}  "
                          f"{'G′ (kPa)':>14s}  {'E (kPa)':>14s}")
        report_lines.append("-" * 80)

        for cond in cond_order:
            sub = df[(df.condition == cond) & (df.source == phantom)]
            if sub.empty:
                continue
            cs_v = sub["cs_m_s"].dropna()
            G_v = sub["G_prime_kPa"].dropna()
            E_v = sub["E_kPa"].dropna()
            n = len(cs_v)
            label = COND_META.get(cond, {}).get("label", cond)
            cs_str = f"{cs_v.mean():.3f}±{cs_v.std():.3f}" if n > 0 else "N/A"
            G_str = f"{G_v.mean():.2f}±{G_v.std():.2f}" if len(G_v) > 0 else "N/A"
            E_str = f"{E_v.mean():.2f}±{E_v.std():.2f}" if len(E_v) > 0 else "N/A"
            report_lines.append(f"  {label:<23s}{n:>5d}  {cs_str:>14s}  "
                              f"{G_str:>14s}  {E_str:>14s}")

    # Group-level paired comparisons
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("GROUP-LEVEL PAIRED COMPARISONS (matched phantom regions)")
    report_lines.append("=" * 80)

    comparisons = [
        ("pred_10kHz", "10kHz_gt", "Pred 10 kHz vs GT 10 kHz"),
        ("pred_5kHz",  "5kHz_gt",  "Pred  5 kHz vs GT  5 kHz"),
        ("pred_2kHz",  "1kHz_gt",  "Pred  2 kHz vs GT  1 kHz"),
        ("pred_6kHz",  "5kHz_gt",  "Pred  6 kHz (12×) vs GT 5 kHz"),
    ]

    for cond_a, cond_b, desc in comparisons:
        report_lines.append(f"\n  {desc}")
        report_lines.append(f"  {'-'*50}")
        da = df[df.condition == cond_a].groupby("group")["cs_m_s"].mean()
        db = df[df.condition == cond_b].groupby("group")["cs_m_s"].mean()
        common = da.index.intersection(db.index)
        if len(common) == 0:
            report_lines.append("    No matched groups found")
            continue

        cs_a = da[common].values
        cs_b = db[common].values
        valid = np.isfinite(cs_a) & np.isfinite(cs_b)
        if valid.sum() < 2:
            report_lines.append("    Insufficient valid pairs")
            continue

        cs_a, cs_b = cs_a[valid], cs_b[valid]
        bias = np.mean(cs_a - cs_b)
        mae = np.mean(np.abs(cs_a - cs_b))
        rmse = np.sqrt(np.mean((cs_a - cs_b)**2))
        r, p = pearsonr(cs_a, cs_b)
        pct_err = np.mean(np.abs(cs_a - cs_b) / np.abs(cs_b)) * 100

        report_lines.append(f"    Matched groups: {len(cs_a)}")
        report_lines.append(f"    Pearson r = {r:.4f}  (p = {p:.2e})")
        report_lines.append(f"    Bias (pred−GT) = {bias:+.3f} m/s")
        report_lines.append(f"    MAE  = {mae:.3f} m/s")
        report_lines.append(f"    RMSE = {rmse:.3f} m/s")
        report_lines.append(f"    Mean % error = {pct_err:.1f}%")

    report_text = "\n".join(report_lines)
    (out / "swe_report_20x.txt").write_text(report_text)
    print(report_text)

    pd.DataFrame(summary_rows).to_csv(out / "summary_statistics.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Publication figures
    # ------------------------------------------------------------------
    print("\n[5/6] Generating publication figures ...")

    # --- Fig 1: Overall bar chart (cs, G', E by condition) ---
    fig1_bar_chart(df, out)

    # --- Fig 2: Scatter plots (pred vs GT cs, group-level) ---
    fig2_scatter(df, comparisons, out)

    # --- Fig 3: Bland-Altman agreement ---
    fig3_bland_altman(df, comparisons, out)

    # --- Fig 4: Per-phantom bar chart ---
    fig4_phantom_bars(df, out)

    # --- Fig 5: Spatiotemporal velocity maps (best sample) ---
    fig5_spatiotemporal(pred_10kHz, pred_5kHz, pred_2kHz, gt_500Hz, out)

    # --- Fig 6: Arrival time maps ---
    fig6_arrival_time(pred_10kHz, pred_5kHz, pred_2kHz, out)

    # --- Fig 7: Comprehensive comparison with nominal values ---
    fig7_nominal_comparison(df, out)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n[6/6] All outputs saved to: {out}/")
    print(f"  - swe_metrics_all.csv        ({len(df)} rows)")
    print(f"  - swe_report_20x.txt")
    print(f"  - summary_statistics.csv")
    print(f"  - fig1_*.png/pdf through fig7_*.png/pdf")
    print("=" * 72)


# ===========================================================================
# Figure Functions
# ===========================================================================

def _savefig(fig, base, name):
    for ext in ("pdf", "png"):
        fig.savefig(base / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: {name}.pdf / .png")


def fig1_bar_chart(df, out):
    """Overall bar chart: cs, G', E by condition."""
    cond_order = ["10kHz_gt", "5kHz_gt", "1kHz_gt", "500Hz_gt",
                  "pred_10kHz", "pred_5kHz", "pred_2kHz", "pred_6kHz"]
    metrics = [("cs_m_s", "Shear Wave Speed (m/s)", "cs"),
               ("G_prime_kPa", "Shear Modulus G′ (kPa)", "G"),
               ("E_kPa", "Young's Modulus E (kPa)", "E")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (col, ylabel, tag) in zip(axes, metrics):
        means, stds, labels, colors = [], [], [], []
        for cond in cond_order:
            sub = df[df.condition == cond][col].dropna()
            if len(sub) == 0:
                continue
            meta = COND_META.get(cond, {})
            means.append(sub.mean())
            stds.append(sub.std())
            labels.append(meta.get("label", cond))
            colors.append(meta.get("color", "#888"))

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Overall {tag}")

    fig.suptitle("SWE Metrics — 20× Model vs Ground Truth", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, out, "fig1_overall_bars")


def fig2_scatter(df, comparisons, out):
    """Group-level scatter: Pred cs vs GT cs."""
    n_comp = sum(1 for c in comparisons
                 if len(df[df.condition==c[0]]) > 0 and len(df[df.condition==c[1]]) > 0)
    if n_comp == 0:
        return

    fig, axes = plt.subplots(1, n_comp, figsize=(5*n_comp, 5))
    if n_comp == 1:
        axes = [axes]

    idx = 0
    for cond_a, cond_b, desc in comparisons:
        da = df[df.condition == cond_a].groupby("group").agg(
            cs_a=("cs_m_s", "mean"), src=("source", "first")).dropna(subset=["cs_a"])
        db = df[df.condition == cond_b].groupby("group").agg(
            cs_b=("cs_m_s", "mean"), src=("source", "first")).dropna(subset=["cs_b"])
        merged = da.join(db[["cs_b"]], how="inner").dropna()
        if len(merged) < 2:
            continue

        ax = axes[idx]
        idx += 1

        for phantom, marker, color in [("2ft", "o", "#1E88E5"), ("3ft", "s", "#E53935")]:
            sub = merged[merged.src == phantom]
            if len(sub) > 0:
                ax.scatter(sub["cs_b"], sub["cs_a"], c=color, marker=marker,
                          s=60, alpha=0.7, edgecolors="k", linewidth=0.5,
                          label=PHANTOM_PROPS.get(phantom, {}).get("label", phantom))

        lim = [0, max(merged["cs_a"].max(), merged["cs_b"].max()) * 1.15]
        ax.plot(lim, lim, "k--", alpha=0.4, label="Identity")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{COND_META.get(cond_b,{}).get('label',cond_b)} cs (m/s)")
        ax.set_ylabel(f"{COND_META.get(cond_a,{}).get('label',cond_a)} cs (m/s)")

        r, p = pearsonr(merged["cs_a"], merged["cs_b"])
        ax.set_title(f"{desc}\nr = {r:.3f}, n = {len(merged)}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    fig.suptitle("Group-Level Shear Wave Speed: Prediction vs Ground Truth",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    _savefig(fig, out, "fig2_scatter_cs")


def fig3_bland_altman(df, comparisons, out):
    """Bland-Altman agreement plots for cs."""
    n_comp = sum(1 for c in comparisons
                 if len(df[df.condition==c[0]]) > 0 and len(df[df.condition==c[1]]) > 0)
    if n_comp == 0:
        return

    fig, axes = plt.subplots(1, n_comp, figsize=(5*n_comp, 4.5))
    if n_comp == 1:
        axes = [axes]

    idx = 0
    for cond_a, cond_b, desc in comparisons:
        da = df[df.condition==cond_a].groupby("group")["cs_m_s"].mean()
        db = df[df.condition==cond_b].groupby("group")["cs_m_s"].mean()
        common = da.index.intersection(db.index)
        if len(common) < 2:
            continue

        cs_a = da[common].values
        cs_b = db[common].values
        valid = np.isfinite(cs_a) & np.isfinite(cs_b)
        cs_a, cs_b = cs_a[valid], cs_b[valid]
        if len(cs_a) < 2:
            continue

        ax = axes[idx]; idx += 1
        mean_val = (cs_a + cs_b) / 2
        diff = cs_a - cs_b
        bias = np.mean(diff)
        sd = np.std(diff)

        ax.scatter(mean_val, diff, c="#5C6BC0", s=40, alpha=0.7, edgecolors="k", linewidth=0.5)
        ax.axhline(bias, color="r", ls="-", alpha=0.7, label=f"Bias = {bias:+.3f}")
        ax.axhline(bias + 1.96*sd, color="gray", ls="--", alpha=0.6,
                   label=f"+1.96 SD = {bias+1.96*sd:+.3f}")
        ax.axhline(bias - 1.96*sd, color="gray", ls="--", alpha=0.6,
                   label=f"−1.96 SD = {bias-1.96*sd:+.3f}")
        ax.set_xlabel("Mean cs (m/s)")
        ax.set_ylabel("Difference (Pred − GT) (m/s)")
        ax.set_title(desc, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Bland-Altman: Shear Wave Speed Agreement", fontsize=12, y=1.02)
    fig.tight_layout()
    _savefig(fig, out, "fig3_bland_altman")


def fig4_phantom_bars(df, out):
    """Per-phantom grouped bar chart."""
    cond_order = ["10kHz_gt", "5kHz_gt", "1kHz_gt", "500Hz_gt",
                  "pred_10kHz", "pred_5kHz", "pred_2kHz", "pred_6kHz"]
    phantoms = ["2ft", "3ft"]

    fig, axes = plt.subplots(len(phantoms), 3, figsize=(16, 4*len(phantoms)))

    for row, phantom in enumerate(phantoms):
        for col, (metric, ylabel) in enumerate([
                ("cs_m_s", "cs (m/s)"), ("G_prime_kPa", "G′ (kPa)"), ("E_kPa", "E (kPa)")]):
            ax = axes[row, col]
            means, stds, labels, colors = [], [], [], []

            for cond in cond_order:
                sub = df[(df.condition==cond) & (df.source==phantom)][metric].dropna()
                if len(sub) == 0:
                    continue
                meta = COND_META.get(cond, {})
                means.append(sub.mean())
                stds.append(sub.std())
                labels.append(meta.get("label", cond))
                colors.append(meta.get("color", "#888"))

            if not means:
                continue
            x = np.arange(len(means))
            ax.bar(x, means, yerr=stds, capsize=3,
                   color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)

            # Add nominal value line
            if metric == "cs_m_s":
                nom = PHANTOM_PROPS[phantom]["cs_nominal"]
                ax.axhline(nom, color="red", ls=":", alpha=0.6,
                          label=f"Nominal {nom:.2f}")
                ax.legend(fontsize=7)
            elif metric == "E_kPa":
                nom = PHANTOM_PROPS[phantom]["E_kPa"]
                ax.axhline(nom, color="red", ls=":", alpha=0.6,
                          label=f"Nominal {nom:.1f}")
                ax.legend(fontsize=7)

            ax.set_title(f"{PHANTOM_PROPS[phantom]['label']}",
                        fontsize=10 if col == 1 else 9)

    fig.suptitle("Per-Phantom SWE Metrics Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, out, "fig4_phantom_bars")


def fig5_spatiotemporal(pred_10kHz, pred_5kHz, pred_2kHz, gt_500Hz, out):
    """Spatiotemporal velocity maps for a representative sample."""
    # Pick first 2ft sample available in all sets
    common = set(pred_10kHz.keys()) & set(pred_5kHz.keys()) & set(pred_2kHz.keys())
    target = None
    for name in sorted(common):
        if "2ft" in name:
            target = name
            break
    if target is None and common:
        target = sorted(common)[0]
    if target is None:
        print("    Skipping fig5: no common sample found")
        return

    # Average over central depth rows
    def spatiotemporal(vel):
        H = vel.shape[1]
        r0, r1 = int(H*0.3), int(H*0.7)
        return np.mean(vel[:, r0:r1, :], axis=1)  # (T, W)

    v10 = spatiotemporal(pred_10kHz[target])
    v5 = spatiotemporal(pred_5kHz[target])
    v2 = spatiotemporal(pred_2kHz[target])

    # Also get 500Hz GT for this group
    gt_v = None
    grp = parse_group(target)
    for name, data in gt_500Hz.items():
        if parse_group(name) == grp:
            gt_v = spatiotemporal(data["velocity"])
            break

    panels = [("Pred 10 kHz (20×)", v10, 1/10000),
              ("Pred 5 kHz (20×)", v5, 1/5000),
              ("Pred 2 kHz (20×)", v2, 1/2000)]
    if gt_v is not None:
        panels.append(("GT 500 Hz", gt_v, 1/500))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5*len(panels), 4))
    vmax = max(np.percentile(np.abs(p[1]), 99) for p in panels)

    for ax, (title, stv, dt) in zip(axes, panels):
        T, W = stv.shape
        t_axis = np.arange(T) * dt * 1000  # ms
        x_axis = np.arange(W) * LATERAL_MM_PER_PX  # mm

        im = ax.imshow(stv.T, aspect="auto", origin="lower",
                       extent=[t_axis[0], t_axis[-1], x_axis[0], x_axis[-1]],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Lateral position (mm)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="Velocity (mm/s)", shrink=0.8)

    fig.suptitle(f"Spatiotemporal Velocity Maps — {target}", fontsize=11, y=1.03)
    fig.tight_layout()
    _savefig(fig, out, "fig5_spatiotemporal")


def fig6_arrival_time(pred_10kHz, pred_5kHz, pred_2kHz, out):
    """Arrival time (time-to-peak) maps comparison."""
    common = set(pred_10kHz.keys()) & set(pred_5kHz.keys()) & set(pred_2kHz.keys())
    target = None
    for name in sorted(common):
        if "3ft" in name:
            target = name
            break
    if target is None and common:
        target = sorted(common)[0]
    if target is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (label, pred, dt) in zip(axes, [
            ("Pred 10 kHz (20×)", pred_10kHz[target], 1/10000),
            ("Pred 5 kHz (20×)", pred_5kHz[target], 1/5000),
            ("Pred 2 kHz (20×)", pred_2kHz[target], 1/2000)]):

        T, H, W = pred.shape
        v_abs = np.mean(np.abs(pred[:, int(H*0.2):int(H*0.8), :]), axis=1)
        t_peak = np.argmax(v_abs, axis=0) * dt * 1000  # ms

        x_mm = np.arange(W) * LATERAL_MM_PER_PX
        ax.plot(x_mm, t_peak, ".-", markersize=2, alpha=0.7)
        ax.set_xlabel("Lateral position (mm)")
        ax.set_ylabel("Arrival time (ms)")
        ax.set_title(label)

    fig.suptitle(f"Shear Wave Arrival Time — {target}", fontsize=11, y=1.02)
    fig.tight_layout()
    _savefig(fig, out, "fig6_arrival_time")


def fig7_nominal_comparison(df, out):
    """Comprehensive comparison against known phantom nominal values."""
    cond_order = ["10kHz_gt", "5kHz_gt", "1kHz_gt", "500Hz_gt",
                  "pred_10kHz", "pred_5kHz", "pred_2kHz", "pred_6kHz"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: cs by phantom and condition
    ax = axes[0, 0]
    phantoms = ["2ft", "3ft"]
    width = 0.08
    x_pos = np.arange(len(phantoms))

    for i, cond in enumerate(cond_order):
        means, stds = [], []
        for phantom in phantoms:
            sub = df[(df.condition==cond) & (df.source==phantom)]["cs_m_s"].dropna()
            means.append(sub.mean() if len(sub)>0 else 0)
            stds.append(sub.std() if len(sub)>0 else 0)
        if all(m == 0 for m in means):
            continue
        meta = COND_META.get(cond, {})
        offset = (i - len(cond_order)/2) * width
        ax.bar(x_pos + offset, means, width*0.9, yerr=stds, capsize=2,
               color=meta.get("color", "#888"), alpha=0.85,
               label=meta.get("label", cond), edgecolor="k", linewidth=0.3)

    for j, phantom in enumerate(phantoms):
        nom = PHANTOM_PROPS[phantom]["cs_nominal"]
        ax.plot([j-0.4, j+0.4], [nom, nom], "r:", linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([PHANTOM_PROPS[p]["label"] for p in phantoms])
    ax.set_ylabel("Shear Wave Speed (m/s)")
    ax.set_title("(A) Shear Wave Speed by Phantom")
    ax.legend(fontsize=6, ncol=2, loc="upper left")

    # Panel B: E by phantom and condition
    ax = axes[0, 1]
    for i, cond in enumerate(cond_order):
        means, stds = [], []
        for phantom in phantoms:
            sub = df[(df.condition==cond) & (df.source==phantom)]["E_kPa"].dropna()
            means.append(sub.mean() if len(sub)>0 else 0)
            stds.append(sub.std() if len(sub)>0 else 0)
        if all(m == 0 for m in means):
            continue
        meta = COND_META.get(cond, {})
        offset = (i - len(cond_order)/2) * width
        ax.bar(x_pos + offset, means, width*0.9, yerr=stds, capsize=2,
               color=meta.get("color", "#888"), alpha=0.85,
               label=meta.get("label", cond), edgecolor="k", linewidth=0.3)

    for j, phantom in enumerate(phantoms):
        nom = PHANTOM_PROPS[phantom]["E_kPa"]
        ax.plot([j-0.4, j+0.4], [nom, nom], "r:", linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([PHANTOM_PROPS[p]["label"] for p in phantoms])
    ax.set_ylabel("Young's Modulus (kPa)")
    ax.set_title("(B) Young's Modulus by Phantom")
    ax.legend(fontsize=6, ncol=2, loc="upper left")

    # Panel C: % error from nominal cs
    ax = axes[1, 0]
    bar_data = []
    for cond in cond_order:
        errs = []
        for phantom in phantoms:
            sub = df[(df.condition==cond) & (df.source==phantom)]["cs_m_s"].dropna()
            if len(sub) == 0:
                continue
            nom = PHANTOM_PROPS[phantom]["cs_nominal"]
            pct = (sub.mean() - nom) / nom * 100
            errs.append(pct)
        if errs:
            meta = COND_META.get(cond, {})
            bar_data.append((meta.get("label", cond), np.mean(errs),
                           meta.get("color", "#888")))

    if bar_data:
        x = np.arange(len(bar_data))
        ax.bar(x, [d[1] for d in bar_data], color=[d[2] for d in bar_data],
               alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in bar_data], rotation=35, ha="right", fontsize=7)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel("% Error from Nominal cs")
        ax.set_title("(C) Relative Error in Shear Wave Speed")

    # Panel D: Temporal coverage and SWE quality
    ax = axes[1, 1]
    cond_data = []
    for cond in cond_order:
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        cs_v = sub["cs_m_s"].dropna()
        if len(cs_v) == 0:
            continue
        meta = COND_META.get(cond, {})
        span = sub["temporal_span_ms"].iloc[0]
        cs_cv = cs_v.std() / cs_v.mean() * 100  # coefficient of variation
        cond_data.append((span, cs_v.mean(), cs_cv,
                         meta.get("color", "#888"), meta.get("label", cond),
                         meta.get("marker", "o")))

    for span, cs_mean, cs_cv, color, label, marker in cond_data:
        ax.scatter(span, cs_cv, c=color, marker=marker, s=100,
                  edgecolors="k", linewidth=0.5, label=label, zorder=5)

    ax.set_xlabel("Temporal Span (ms)")
    ax.set_ylabel("cs Coefficient of Variation (%)")
    ax.set_title("(D) Temporal Coverage vs Measurement Precision")
    ax.legend(fontsize=6, loc="best")

    fig.suptitle("Comprehensive SWE Comparison — 20× Model vs Ground Truth",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, out, "fig7_comprehensive")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE Analysis for 20× model")
    parser.add_argument("--output_dir", default="outputs/swe_analysis_20x")
    parser.add_argument("--gt_meta", default="data/fullscale_velocity/metadata.json")
    args = parser.parse_args()

    run_analysis(output_dir=args.output_dir, gt_meta_path=args.gt_meta)
