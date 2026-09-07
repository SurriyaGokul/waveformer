#!/usr/bin/env python3
"""
Comprehensive Shear Wave Elastography (SWE) Analysis
=====================================================
Computes shear wave speed (cs), shear modulus (G'), Young's modulus (E),
and viscosity (η via Voigt model) from velocity maps across five conditions:

  1. 20 kHz  – ground‑truth at highest temporal resolution (reference)
  2.  5 kHz  – ground‑truth at medium temporal resolution
  3. 500 Hz  – ground‑truth at lowest temporal resolution
  4.  6 kHz  – v2 model prediction (trained 5/10/20 kHz only)
  5.  6 kHz  – v3 model prediction (trained 500/1/5/10/20 kHz)

Matching strategy
-----------------
Data is matched by (source, focus, plane) – i.e. the same physical tissue
region.  Within each group, multiple acquisition repetitions are averaged.

Physics
-------
  cs   : shear‑wave phase speed (m/s)  — from time‑of‑arrival (ToA) regression
  G'   : shear storage modulus (kPa)   — G' = ρ · cs²  (ρ = 1000 kg/m³)
  E    : Young's modulus (kPa)         — E  = 3 · G'   (incompressible tissue)
  η    : dynamic viscosity (Pa·s)      — Voigt model fit from k‑space dispersion

Spatial calibration
-------------------
  Lateral px spacing : 0.1013 mm/px
  Axial    px spacing : 0.0507 mm/px

Usage
-----
    python scripts/swe_analysis.py
    python scripts/swe_analysis.py --output_dir outputs/swe_analysis
"""
# ---------------------------------------------------------------------------
# Standard library + scientific stack
# ---------------------------------------------------------------------------
import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage, signal
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
RHO_KG_M3           = 1000.0          # tissue phantom density
LATERAL_MM_PER_PX   = 0.1013          # mm per lateral pixel
AXIAL_MM_PER_PX     = 0.0507          # mm per axial pixel
LATERAL_M_PER_PX    = 0.1013e-3       # m  per lateral pixel
AXIAL_M_PER_PX      = 0.0507e-3       # m  per axial pixel
H_ROI               = 116             # ROI height (pixels)
W_ROI               = 160             # ROI width  (pixels)

# PRF → frame time lookup
PRF_TO_DT = {
    "20kHz":  1 / 20000,
    "10kHz":  1 / 10000,
    "5kHz":   1 / 5000,
    "1kHz":   1 / 1000,
    "500Hz":  1 / 500,
    "6kHz":   1 / 6000,    # predicted virtual 6 kHz
}

# Condition display labels and colours
COND_META = {
    "20kHz_gt":   {"label": "GT 20 kHz",         "color": "#2196F3", "ls": "-",  "marker": "o"},
    "5kHz_gt":    {"label": "GT 5 kHz",           "color": "#4CAF50", "ls": "-",  "marker": "s"},
    "500Hz_gt":   {"label": "GT 500 Hz",          "color": "#FF9800", "ls": "--", "marker": "^"},
    "v2_6kHz":    {"label": "Pred 6 kHz (v2)",    "color": "#9C27B0", "ls": "-.", "marker": "D"},
    "v3_6kHz":    {"label": "Pred 6 kHz (v3)",    "color": "#F44336", "ls": ":",  "marker": "*"},
}


# ===========================================================================
# SWE Physics
# ===========================================================================

def voigt_phase_velocity(omega: np.ndarray, G_prime_Pa: float, eta_Pa_s: float) -> np.ndarray:
    """Voigt model phase velocity: cs(ω) = √(2(G'²+ω²η²)/(ρ(√(G'²+ω²η²)+G')))."""
    numer = 2 * (G_prime_Pa**2 + (omega * eta_Pa_s)**2)
    denom = RHO_KG_M3 * (np.sqrt(G_prime_Pa**2 + (omega * eta_Pa_s)**2) + G_prime_Pa)
    return np.sqrt(numer / denom)


def _find_push_column(v_abs: np.ndarray, early_frames: int = 3) -> int:
    """
    Find the dominant push excitation column as the column with highest
    mean absolute velocity in the first few frames.
    """
    T, W = v_abs.shape
    early = v_abs[:min(early_frames, T), :]    # (n_early, W)
    return int(np.argmax(early.max(axis=0)))


def estimate_cs_toa(velocity: np.ndarray,
                    dt_s: float,
                    dx_m: float,
                    min_amplitude_pct: float = 20.0,
                    depth_frac: Tuple[float, float] = (0.2, 0.8),
                    push_exclusion_px: int = 15) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Estimate shear‑wave speed via time‑of‑arrival using cross‑correlation.

    Strategy
    --------
    1. Find the push‑focus column (peak early‑frame energy) and exclude an
       exclusion zone (±push_exclusion_px) to avoid push artefacts.
    2. In the far‑field (left wing and right wing separately), estimate the
       inter‑column time delay via cross‑correlation with a 5‑px step.
    3. Local cs = step_dx / (delay_s).  Median over all valid estimates.

    Parameters
    ----------
    velocity          : (T, H, W) mm/s
    dt_s              : frame interval [s]
    dx_m              : lateral pixel spacing [m]
    min_amplitude_pct : amplitude threshold percentile for noise rejection
    depth_frac        : depth ROI fraction [lo, hi]
    push_exclusion_px : columns excluded on each side of the push centre

    Returns
    -------
    cs_m_s, cs_std, t_peak_arr (NaN=excluded), x_arr_m
    """
    T, H, W = velocity.shape
    row_lo = int(H * depth_frac[0])
    row_hi = int(H * depth_frac[1])

    v_abs = np.mean(np.abs(velocity[:, row_lo:row_hi, :]), axis=1)  # (T, W)

    # amplitude mask
    peak_amp = v_abs.max(axis=0)
    amp_thresh = np.percentile(peak_amp, min_amplitude_pct)
    amp_valid = peak_amp > amp_thresh

    # push column
    x_push = _find_push_column(v_abs, early_frames=max(3, int(0.1 * T)))

    # exclude push zone
    excl_lo = max(0, x_push - push_exclusion_px)
    excl_hi = min(W - 1, x_push + push_exclusion_px)
    push_excl = np.zeros(W, dtype=bool)
    push_excl[excl_lo:excl_hi + 1] = True

    valid_cols = amp_valid & ~push_excl      # (W,)

    x_arr = np.arange(W, dtype=float) * dx_m

    # ------------------------------------------------------------------
    # Cross‑correlation time‑of‑flight between column pairs
    # ------------------------------------------------------------------
    # Adaptive step: ensure lag=1 → cs > 0.15 m/s
    # Required: STEP * dx_m / dt_s > 0.15  →  STEP > 0.15 * dt_s / dx_m
    min_step = max(1, int(np.ceil(0.15 * dt_s / dx_m)))
    STEP = max(5, min_step)     # at least 5 px, more for very slow PRF

    cs_estimates = []
    v_avg = np.mean(velocity[:, row_lo:row_hi, :], axis=1)  # signed (T, W)

    for x in range(0, W - STEP):
        x2 = x + STEP
        if not (valid_cols[x] and valid_cols[x2]):
            continue

        sig_a = v_avg[:, x]
        sig_b = v_avg[:, x2]

        # normalise to unit energy to avoid amplitude bias
        ea = np.sqrt(np.sum(sig_a ** 2))
        eb = np.sqrt(np.sum(sig_b ** 2))
        if ea < 1e-12 or eb < 1e-12:
            continue

        xcorr  = np.correlate(sig_a / ea, sig_b / eb, mode='full')
        lags   = np.arange(-(T - 1), T)
        pi     = int(np.argmax(xcorr))

        # Sub‑pixel parabolic interpolation for fractional lag
        if 0 < pi < len(xcorr) - 1:
            y0, y1, y2 = xcorr[pi - 1], xcorr[pi], xcorr[pi + 1]
            denom = y0 - 2 * y1 + y2
            frac  = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            peak_lag = float(lags[pi]) + frac
        else:
            peak_lag = float(lags[pi])

        if abs(peak_lag) < 0.05:
            continue  # effectively no resolvable delay

        delay_s  = peak_lag * dt_s
        cs_local = abs(STEP * dx_m / delay_s)

        # physical plausibility gate (0.2 – 15 m/s)
        if 0.2 < cs_local < 15.0:
            cs_estimates.append(cs_local)

    if not cs_estimates:
        t_peak = np.argmax(v_abs, axis=0).astype(float) * dt_s
        return np.nan, np.nan, t_peak, x_arr

    cs_arr = np.array(cs_estimates)

    # Robust median ± IQR
    cs_med  = float(np.median(cs_arr))
    cs_iqr  = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    cs_std  = cs_iqr / 1.349    # convert IQR to std‑equivalent

    # Sanity check
    if not (0.2 < cs_med < 15.0):
        cs_med = np.nan
        cs_std = np.nan

    t_peak = np.argmax(v_abs, axis=0).astype(float) * dt_s
    return cs_med, cs_std, t_peak, x_arr


def estimate_viscosity_voigt(velocity: np.ndarray,
                              dt_s: float,
                              dx_m: float,
                              depth_frac: Tuple[float, float] = (0.25, 0.75),
                              n_freq_bins: int = 10
                              ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Estimate shear modulus G' and viscosity η from the k‑space dispersion.

    Method
    ------
    1. Average velocity over central depth range → v(t, x)  shape (T, W)
    2. Apply a right‑propagating directional filter (keep +kx, +ω quadrant)
    3. Extract energy‑weighted dispersion curve cs(ω)
    4. Fit the Voigt model cs(ω) = f(G', η)

    Returns
    -------
    G_prime_kPa, G_prime_std_kPa
    eta_Pa_s,    eta_std_Pa_s
    omega_rad,   cs_disp  — raw dispersion curve used for fit
    """
    T, H, W = velocity.shape
    row_lo = int(H * depth_frac[0])
    row_hi = int(H * depth_frac[1])

    v_avg = np.mean(velocity[:, row_lo:row_hi, :], axis=1)   # (T, W)

    # 2D FFT in (t, x) → (ω, kx)
    V  = np.fft.fftshift(np.fft.fft2(v_avg))                # (T, W)
    P  = np.abs(V)**2

    omegas = np.fft.fftshift(np.fft.fftfreq(T, dt_s)) * 2 * np.pi   # rad/s
    kxs    = np.fft.fftshift(np.fft.fftfreq(W, dx_m))                # 1/m

    # Fold to positive‑kx (handles waves propagating in BOTH directions)
    # Keep ω > 0, fold |kx| by summing symmetric pairs
    omega_pos_mask = omegas > 0
    omega_bins     = omegas[omega_pos_mask]

    kx_pos_mask = kxs > 0
    kx_arr_half = kxs[kx_pos_mask]

    if len(omega_bins) < 5 or len(kx_arr_half) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])

    # Sum positive and negative kx halves (fold) to capture both propagation directions
    P_pos  = np.abs(V[np.ix_(omega_pos_mask, kx_pos_mask)])**2
    kx_neg_mask = kxs < 0
    # Flip negative kx to align with positive kx array (reverse order)
    P_neg_raw = np.abs(V[np.ix_(omega_pos_mask, kx_neg_mask)])**2
    n_kx = min(P_pos.shape[1], P_neg_raw.shape[1])
    P_sub = P_pos[:, :n_kx] + P_neg_raw[:, ::-1][:, :n_kx]
    kx_arr = kx_arr_half[:n_kx]

    # Energy‑weighted kx centroid at each ω
    P_nonneg = np.maximum(P_sub, 0)
    total_weight = P_nonneg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kx_centroid = np.where(
            total_weight > 0,
            (P_nonneg * kx_arr[np.newaxis, :]).sum(axis=1) / total_weight,
            np.nan,
        )                              # (n_ω,)

    # Phase velocity
    with np.errstate(divide="ignore", invalid="ignore"):
        cs_disp = np.where(kx_centroid > 0, omega_bins / kx_centroid, np.nan)

    # Keep physically plausible range (0.3–15 m/s)
    valid_mask = (cs_disp > 0.3) & (cs_disp < 15.0) & np.isfinite(cs_disp)

    # Sub‑sample to n_freq_bins regular bins for stable fitting
    omega_valid = omega_bins[valid_mask]
    cs_valid    = cs_disp[valid_mask]

    if len(omega_valid) < 5:
        return np.nan, np.nan, np.nan, np.nan, omega_bins, cs_disp

    # Fit Voigt model
    try:
        from scipy.optimize import curve_fit
        # Initial guess from cs ≈ √(G'/ρ)
        cs_mean   = np.nanmedian(cs_valid)
        G_init    = RHO_KG_M3 * cs_mean**2        # Pa
        eta_init  = 0.5                            # Pa·s

        popt, pcov = curve_fit(
            lambda w, G, eta: voigt_phase_velocity(w, G, eta),
            omega_valid, cs_valid,
            p0=[G_init, eta_init],
            bounds=([100,     0.001],
                    [500000,  20.0]),
            maxfev=8000
        )
        perr = np.sqrt(np.diag(pcov))
        G_prime_Pa      = popt[0]
        eta_Pa_s        = popt[1]
        G_prime_Pa_std  = perr[0]
        eta_Pa_s_std    = perr[1]
    except Exception:
        # Fall back to simple cs²·ρ estimate
        cs_m = np.nanmedian(cs_valid)
        G_prime_Pa     = RHO_KG_M3 * cs_m**2
        G_prime_Pa_std = 0.0
        eta_Pa_s       = np.nan
        eta_Pa_s_std   = np.nan

    return (G_prime_Pa / 1000, G_prime_Pa_std / 1000,
            eta_Pa_s, eta_Pa_s_std,
            omega_bins, cs_disp)


def compute_swe_metrics(velocity: np.ndarray,
                        dt_s: float,
                        label: str = "") -> Dict:
    """
    Run the full SWE analysis pipeline on a single velocity map.

    Returns a dict with keys:
        cs_m_s, cs_std,
        G_prime_kPa, G_prime_std,
        E_kPa, E_std,
        eta_Pa_s, eta_std,
        n_frames, dt_s
    """
    T, H, W = velocity.shape

    # --- Time‑of‑Arrival ---
    cs, cs_std, t_arr, x_arr = estimate_cs_toa(
        velocity, dt_s, LATERAL_M_PER_PX,
        min_amplitude_pct=15.0,
        depth_frac=(0.2, 0.8)
    )

    # --- Shear modulus from ToA cs ---
    if np.isfinite(cs):
        G_prime_kPa = RHO_KG_M3 * cs**2 / 1000
        G_prime_std = 2 * RHO_KG_M3 * cs * cs_std / 1000 if np.isfinite(cs_std) else np.nan
        E_kPa       = 3 * G_prime_kPa
        E_std       = 3 * G_prime_std if np.isfinite(G_prime_std) else np.nan
    else:
        G_prime_kPa = G_prime_std = E_kPa = E_std = np.nan

    # --- Voigt viscosity from k‑space ---
    G_voigt_kPa, G_voigt_std, eta, eta_std, _, _ = estimate_viscosity_voigt(
        velocity, dt_s, LATERAL_M_PER_PX,
        depth_frac=(0.2, 0.8)
    )

    # Use Voigt G' if ToA is unavailable (fallback)
    if not np.isfinite(G_prime_kPa) and np.isfinite(G_voigt_kPa):
        G_prime_kPa = G_voigt_kPa
        G_prime_std = G_voigt_std if np.isfinite(G_voigt_std) else np.nan
        E_kPa       = 3 * G_prime_kPa
        E_std       = 3 * G_prime_std if np.isfinite(G_prime_std) else np.nan

    return dict(
        cs_m_s      = cs,
        cs_std      = cs_std,
        G_prime_kPa = G_prime_kPa,
        G_prime_std = G_prime_std,
        E_kPa       = E_kPa,
        E_std       = E_std,
        eta_Pa_s    = eta,
        eta_std     = eta_std,
        n_frames    = T,
        dt_s        = dt_s,
        temporal_coverage_ms = T * dt_s * 1000,
    )


# ===========================================================================
# Data loading helpers
# ===========================================================================

def parse_group(name: str) -> Optional[str]:
    """Extract (source, focus, plane) group key from sample name."""
    m = re.match(r'((?:2ft|3ft|cirs)_\w+focus_Plane_\d+)', name)
    return m.group(1) if m else None


def load_all_gt_samples(metadata_path: str = "data/fullscale_velocity/metadata.json"
                        ) -> Dict[str, Dict]:
    """
    Load all GT velocity samples keyed by name.

    Returns dict: name → {velocity, dt_s, prf, source, group, ...}
    """
    meta = json.load(open(metadata_path))
    all_samples = meta["train"] + meta["val"] + meta["inference"]

    by_name = {}
    splits = {
        "train":     ("data/fullscale_velocity/train",     meta["train"]),
        "val":       ("data/fullscale_velocity/val",       meta["val"]),
        "inference": ("data/fullscale_velocity/inference", meta["inference"]),
    }

    for split, (split_dir, entries) in splits.items():
        for s in entries:
            name = s["name"]
            vel_path = Path(split_dir) / f"velocity_{name}.npy"
            if not vel_path.exists():
                continue
            prf_str = s["frequency"]
            dt_s    = 1.0 / s["prf_hz"]
            group   = parse_group(name)
            by_name[name] = dict(
                velocity = np.load(str(vel_path)),
                dt_s     = dt_s,
                prf      = prf_str,
                source   = s["source"],
                focus    = s["focus"],
                group    = group,
                tissue_props = s["tissue_properties"],
                name     = name,
                split    = split,
            )
    return by_name


def load_6khz_predictions(pred_dir: str) -> Dict[str, np.ndarray]:
    """
    Load velocity_6khz_*.npy arrays from a prediction directory.

    Returns dict: base_name → velocity array (49, H, W) in mm/s
    """
    pred_dir = Path(pred_dir)
    result   = {}
    for p in sorted(pred_dir.glob("velocity_6khz_*.npy")):
        # velocity_6khz_2ft_10mmfocus_Plane_1_500Hz_20260129T130048_w0.npy
        base = p.stem.replace("velocity_6khz_", "")  # → 2ft_10mmfocus_...
        result[base] = np.load(str(p))
    return result


# ===========================================================================
# Main analysis
# ===========================================================================

def run_analysis(output_dir: str = "outputs/swe_analysis",
                 gt_meta_path: str = "data/fullscale_velocity/metadata.json",
                 v2_pred_dir:  str = "outputs/6khz_predictions",
                 v3_pred_dir:  str = "outputs/6khz_predictions_v3") -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" SWE ANALYSIS: Elasticity & Viscosity Comparison")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1.  Load all GT velocity samples
    # ------------------------------------------------------------------
    print("\n[1/5] Loading ground-truth velocity data …")
    all_gt = load_all_gt_samples(gt_meta_path)
    print(f"      Loaded {len(all_gt)} GT samples")

    # Separate by frequency
    freq_groups: Dict[str, Dict] = {
        "20kHz_gt": {},
        "5kHz_gt":  {},
        "500Hz_gt": {},
    }
    for name, info in all_gt.items():
        prf = info["prf"]
        if prf == "20kHz":
            freq_groups["20kHz_gt"][name] = info
        elif prf == "5kHz":
            freq_groups["5kHz_gt"][name] = info
        elif prf == "500Hz":
            freq_groups["500Hz_gt"][name] = info

    print(f"      20 kHz: {len(freq_groups['20kHz_gt'])} samples")
    print(f"       5 kHz: {len(freq_groups['5kHz_gt'])} samples")
    print(f"     500 Hz:  {len(freq_groups['500Hz_gt'])} samples")

    # ------------------------------------------------------------------
    # 2.  Load predicted 6 kHz from v2 and v3
    # ------------------------------------------------------------------
    print("\n[2/5] Loading 6 kHz model predictions …")
    v2_preds = load_6khz_predictions(v2_pred_dir)
    v3_preds = load_6khz_predictions(v3_pred_dir)
    print(f"      v2 predictions: {len(v2_preds)} samples")
    print(f"      v3 predictions: {len(v3_preds)} samples")
    DT_6KHZ = 1.0 / 6000

    # For predicted 6kHz, also load the saved GT 500Hz from the pred dir
    # (these are the native 500Hz GT npy saved alongside the predictions)
    gt_500hz_from_pred = {}
    for name, vel in v3_preds.items():
        gt_path = Path(v3_pred_dir) / f"gt_velocity_{name}.npy"
        if gt_path.exists():
            gt_500hz_from_pred[name] = np.load(str(gt_path))

    # ------------------------------------------------------------------
    # 3.  Compute SWE metrics per sample
    # ------------------------------------------------------------------
    print("\n[3/5] Computing SWE metrics (cs, G', E, η) …")
    print("      This may take a minute …")

    records = []   # list of dicts

    # ---------- GT conditions ----------
    for cond_key, samples in freq_groups.items():
        for name, info in samples.items():
            vel = info["velocity"]
            dt  = info["dt_s"]
            m   = compute_swe_metrics(vel, dt, label=name)
            m.update(dict(
                condition = cond_key,
                name      = name,
                group     = info["group"],
                source    = info["source"],
                tissue_props = info["tissue_props"],
            ))
            records.append(m)
        print(f"      {cond_key}: done ({len(samples)} samples)")

    # ---------- Predicted 6 kHz ----------
    for cond_key, preds in [("v2_6kHz", v2_preds), ("v3_6kHz", v3_preds)]:
        for name, vel in preds.items():
            m = compute_swe_metrics(vel, DT_6KHZ, label=name)
            # resolve group from name (contains the 500Hz timestamp)
            group = parse_group(name)
            # look up tissue props from GT 500Hz sample with same name
            tissue_props = {}
            if name in all_gt:
                tissue_props = all_gt[name]["tissue_props"]
            else:
                # best effort: look by group
                for gt_name, info in all_gt.items():
                    if info["group"] == group and info["prf"] == "500Hz":
                        tissue_props = info["tissue_props"]
                        break
            m.update(dict(
                condition    = cond_key,
                name         = name,
                group        = group,
                source       = ("2ft" if "2ft" in name else "3ft"),
                tissue_props = tissue_props,
            ))
            records.append(m)
        print(f"      {cond_key}: done ({len(preds)} samples)")

    df = pd.DataFrame(records)
    df.to_csv(out / "all_sample_metrics.csv", index=False)
    print(f"      Per-sample metrics → {out}/all_sample_metrics.csv")

    # ------------------------------------------------------------------
    # 4.  Group-level aggregation
    # ------------------------------------------------------------------
    print("\n[4/5] Aggregating by (group × condition) …")

    CONDITIONS = ["20kHz_gt", "5kHz_gt", "500Hz_gt", "v2_6kHz", "v3_6kHz"]
    METRICS    = ["cs_m_s", "G_prime_kPa", "E_kPa", "eta_Pa_s"]

    # Groups that have at least v3 6kHz prediction AND 20kHz GT
    pivot = df.pivot_table(
        values=METRICS,
        index=["group", "source"],
        columns="condition",
        aggfunc=["mean", "std"]
    )
    pivot.columns = [f"{met}_{cond}_{agg}" for agg, met, cond in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(out / "group_level_metrics.csv", index=False)
    print(f"      Group metrics → {out}/group_level_metrics.csv")

    # Summary stats per condition
    summary_rows = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        for met in METRICS:
            vals = sub[met].dropna()
            summary_rows.append(dict(
                condition  = cond,
                metric     = met,
                n          = len(vals),
                mean       = vals.mean()  if len(vals) else np.nan,
                std        = vals.std()   if len(vals) else np.nan,
                median     = vals.median() if len(vals) else np.nan,
                q25        = vals.quantile(0.25) if len(vals) else np.nan,
                q75        = vals.quantile(0.75) if len(vals) else np.nan,
            ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "summary_statistics.csv", index=False)

    # ------------------------------------------------------------------
    # 5.  Print comprehensive text report
    # ------------------------------------------------------------------
    print("\n[5/5] Generating report and figures …")

    _print_and_save_text_report(df, summary_df, pivot, out)
    _generate_figures(df, summary_df, out)

    print(f"\n{'=' * 70}")
    print(f"  Analysis complete.  Results in: {out}/")
    print(f"{'=' * 70}")


# ===========================================================================
# Report generation
# ===========================================================================

def _print_and_save_text_report(df, summary_df, pivot, out: Path) -> None:
    """Write a detailed text report to stdout and file."""
    lines = []
    sep   = "=" * 80

    def p(s=""):
        lines.append(s)
        print(s)

    p(sep)
    p("  SHEAR WAVE ELASTOGRAPHY – COMPREHENSIVE RESULTS REPORT")
    p(sep)
    p()
    p("STUDY DESIGN")
    p("-" * 60)
    p("Two viscoelastic tissue phantoms (2FT ~2 kPa, 3FT ~3 kPa) were")
    p("imaged at multiple pulse‑repetition frequencies.  The same physical")
    p("regions were acquired at 20 kHz (reference), 5 kHz, 500 Hz, and two")
    p("deep‑learning models were used to synthesise a virtual 6 kHz signal")
    p("from the 500 Hz raw IQ data.")
    p()
    p("Five conditions evaluated:")
    p("  20kHz_gt : Ground‑truth at 20 kHz  (reference, highest temporal res)")
    p("  5kHz_gt  : Ground‑truth at  5 kHz  (good SWE temporal coverage)")
    p("  500Hz_gt : Ground‑truth at 500 Hz  (low res, wave aliased in time)")
    p("  v2_6kHz  : DL model v2 (5/10/20 kHz training only) → 6 kHz predicted")
    p("  v3_6kHz  : DL model v3 (all freq: 500 Hz/1/5/10/20 kHz training) → 6 kHz predicted")
    p()
    p("SWE PHYSICS BACKGROUND")
    p("-" * 60)
    p("  cs  = shear‑wave phase speed from time‑of‑arrival (ToA) regression")
    p("  G'  = ρ · cs²              [shear storage modulus, kPa]")
    p("  E   = 3 · G'               [Young's modulus, kPa, incompressible limit]")
    p("  η   = Voigt model fit      [dynamic shear viscosity, Pa·s]")
    p("  ρ   = 1000 kg/m³           [phantom density]")
    p("  dx  = 0.1013 mm/px         [lateral pixel spacing]")
    p()
    p("IMPORTANT TEMPORAL COVERAGE NOTE")
    p("-" * 60)
    p("  Each velocity stack = 49 frames.  Temporal span = 49 × dt:")
    for cond, dt_s in [("20kHz_gt", 5e-5), ("5kHz_gt", 2e-4),
                        ("500Hz_gt", 2e-3), ("v2_6kHz/v3_6kHz", 1/6000)]:
        span = 49 * dt_s * 1000
        p(f"    {cond:<20}: dt={dt_s*1000:.3f} ms  →  span={span:.2f} ms")
    p("  For 2FT phantom cs≈1.41 m/s across W=16.2 mm, traversal≈11.5 ms")
    p("  For 3FT phantom cs≈1.73 m/s across W=16.2 mm, traversal≈ 9.4 ms")
    p("  → 20 kHz data covers only ~21% of traversal (estimate less reliable)")
    p("  → 500 Hz data spans ~98 ms; wave crossed FOV ~8× (temporally aliased)")
    p("  → 5 kHz / 6 kHz data cover 71–100% (BEST for ToA estimation)")
    p()

    # ── Overall summary table ──
    p(sep)
    p("  OVERALL MEAN ± STD BY CONDITION")
    p(sep)
    header = f"{'Condition':<18} {'n':>4}  {'cs (m/s)':>12}  {'G\' (kPa)':>12}  {'E (kPa)':>10}  {'η (Pa·s)':>10}"
    p(header)
    p("-" * len(header))

    for cond in ["20kHz_gt", "5kHz_gt", "500Hz_gt", "v2_6kHz", "v3_6kHz"]:
        row = {}
        for met in ["cs_m_s", "G_prime_kPa", "E_kPa", "eta_Pa_s"]:
            r = summary_df[(summary_df.condition == cond) & (summary_df.metric == met)]
            if len(r):
                row[met] = (r.iloc[0]["mean"], r.iloc[0]["std"], r.iloc[0]["n"])
            else:
                row[met] = (np.nan, np.nan, 0)

        def fmt(tup):
            return f"{tup[0]:6.3f}±{tup[1]:5.3f}" if np.isfinite(tup[0]) else "       N/A   "

        n_str = str(int(row["cs_m_s"][2]))
        p(f"{COND_META[cond]['label']:<18} {n_str:>4}  "
          f"{fmt(row['cs_m_s']):>12}  "
          f"{fmt(row['G_prime_kPa']):>12}  "
          f"{fmt(row['E_kPa']):>10}  "
          f"{fmt(row['eta_Pa_s']):>10}")
    p()

    # ── By source phantom ──
    for source in ["2ft", "3ft"]:
        p(sep)
        p(f"  PER‑CONDITION: {source.upper()} PHANTOM only")
        p(sep)
        p(header)
        p("-" * len(header))
        for cond in ["20kHz_gt", "5kHz_gt", "500Hz_gt", "v2_6kHz", "v3_6kHz"]:
            sub = df[(df.condition == cond) & (df.source == source)]
            row = {}
            for met in ["cs_m_s", "G_prime_kPa", "E_kPa", "eta_Pa_s"]:
                vals = sub[met].dropna()
                row[met] = (vals.mean(), vals.std(), len(vals)) if len(vals) else (np.nan, np.nan, 0)

            def fmt(tup):
                return f"{tup[0]:6.3f}±{tup[1]:5.3f}" if np.isfinite(tup[0]) else "       N/A   "

            n_str = str(int(row["cs_m_s"][2]))
            p(f"{COND_META[cond]['label']:<18} {n_str:>4}  "
              f"{fmt(row['cs_m_s']):>12}  "
              f"{fmt(row['G_prime_kPa']):>12}  "
              f"{fmt(row['E_kPa']):>10}  "
              f"{fmt(row['eta_Pa_s']):>10}")
        p()

    # ── Per‑group breakdown ──
    p(sep)
    p("  PER‑GROUP BREAKDOWN  (source_focus_plane)")
    p(sep)
    groups = df["group"].dropna().unique()
    groups = sorted(groups)

    # Only show groups that have 20kHz_gt AND at least one 6kHz prediction
    for g in groups:
        sub = df[df["group"] == g]
        conds_present = sub["condition"].unique()
        has_ref = "20kHz_gt" in conds_present
        has_pred = "v2_6kHz" in conds_present or "v3_6kHz" in conds_present
        if not (has_ref and has_pred):
            continue

        src = sub["source"].iloc[0]
        props = sub["tissue_props"].iloc[0] if sub["tissue_props"].iloc[0] else {}
        nom_cs    = props.get("estimated_shear_speed_m_s", np.nan)
        nom_G     = RHO_KG_M3 * nom_cs**2 / 1000 if np.isfinite(nom_cs) else np.nan
        nom_E     = 3 * nom_G if np.isfinite(nom_G) else np.nan
        nom_eta   = props.get("viscosity_pa_s", np.nan)

        p(f"\n  {g}  [{src} phantom]")
        p(f"    Nominal: cs={nom_cs:.2f} m/s  G'={nom_G:.2f} kPa  "
          f"E={nom_E:.2f} kPa  η={nom_eta} Pa·s")
        p(f"    {'Condition':<18} {'n':>3}  {'cs (m/s)':>10}  {'G\' (kPa)':>10}  {'E (kPa)':>9}  {'η (Pa·s)':>9}")
        p(f"    {'-'*65}")

        for cond in ["20kHz_gt", "5kHz_gt", "500Hz_gt", "v2_6kHz", "v3_6kHz"]:
            csub = sub[sub["condition"] == cond]
            if len(csub) == 0:
                continue
            row = {}
            for met in ["cs_m_s", "G_prime_kPa", "E_kPa", "eta_Pa_s"]:
                vals = csub[met].dropna()
                row[met] = (vals.mean(), vals.std()) if len(vals) else (np.nan, np.nan)

            def fmt2(t):
                if np.isfinite(t[0]):
                    suf = f"±{t[1]:.3f}" if np.isfinite(t[1]) and t[1] > 0 else ""
                    return f"{t[0]:6.3f}{suf}"
                return "   N/A   "

            p(f"    {COND_META[cond]['label']:<18} {len(csub):>3}  "
              f"{fmt2(row['cs_m_s']):>10}  "
              f"{fmt2(row['G_prime_kPa']):>10}  "
              f"{fmt2(row['E_kPa']):>9}  "
              f"{fmt2(row['eta_Pa_s']):>9}")

    p()
    p(sep)
    p("END OF REPORT")
    p(sep)

    # Save to file
    with open(out / "swe_report.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Full text report → {out}/swe_report.txt")


# ===========================================================================
# Figures
# ===========================================================================

def _generate_figures(df: pd.DataFrame, summary_df: pd.DataFrame, out: Path) -> None:
    """Generate publication-quality comparison figures."""

    CONDITIONS = ["20kHz_gt", "5kHz_gt", "500Hz_gt", "v2_6kHz", "v3_6kHz"]
    COLORS     = [COND_META[c]["color"]  for c in CONDITIONS]
    LABELS     = [COND_META[c]["label"]  for c in CONDITIONS]
    METRICS    = [
        ("cs_m_s",      "Shear Wave Speed cs (m/s)",  "cs"),
        ("G_prime_kPa", "Shear Modulus G' (kPa)",     "G_prime"),
        ("E_kPa",       "Young's Modulus E (kPa)",    "E"),
        ("eta_Pa_s",    "Dynamic Viscosity η (Pa·s)", "eta"),
    ]

    # ─────────────────────────────────────────────────────
    # Fig 1: Bar chart — overall mean ± std, split 2ft / 3ft
    # ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("SWE Metrics: Ground Truth vs. DL Predictions\n"
                 "(error bars = 1 standard deviation)", fontsize=14, fontweight="bold")

    for row_idx, source in enumerate(["2ft", "3ft"]):
        for col_idx, (met, ylabel, _) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            means, stds, ns = [], [], []
            for cond in CONDITIONS:
                sub = df[(df.condition == cond) & (df.source == source)]
                vals = sub[met].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std()   if len(vals) else 0)
                ns.append(len(vals))

            x = np.arange(len(CONDITIONS))
            bars = ax.bar(x, means, yerr=stds, capsize=4,
                          color=COLORS, alpha=0.82, edgecolor="k", linewidth=0.6,
                          error_kw=dict(elinewidth=1.0, ecolor="k"))
            ax.set_xticks(x)
            ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{source.upper()} phantom", fontsize=10)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(axis="y", alpha=0.3, which="both")

            # Nominal reference line
            nom = None
            if source == "2ft":
                if met == "cs_m_s":      nom = 1.41
                elif met == "G_prime_kPa": nom = 2.0
                elif met == "E_kPa":     nom = 6.0
                elif met == "eta_Pa_s":  nom = 0.5
            else:
                if met == "cs_m_s":      nom = 1.73
                elif met == "G_prime_kPa": nom = 3.0
                elif met == "E_kPa":     nom = 9.0
                elif met == "eta_Pa_s":  nom = 0.4
            if nom is not None:
                ax.axhline(nom, color="crimson", ls="--", lw=1.5, label=f"Nominal={nom}")
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(out / "fig1_bar_overall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 1 saved: fig1_bar_overall.png")

    # ─────────────────────────────────────────────────────
    # Fig 2: Box plots per condition, for each metric
    # ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Distribution of SWE Metrics by Condition", fontsize=13, fontweight="bold")

    for ax, (met, ylabel, _) in zip(axes, METRICS):
        data = [df[df.condition == c][met].dropna().values for c in CONDITIONS]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2.0),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="x", markersize=4, alpha=0.6))
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(CONDITIONS) + 1))
        ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "fig2_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 2 saved: fig2_boxplot.png")

    # ─────────────────────────────────────────────────────
    # Fig 3: Paired scatter plot — each condition vs 20kHz GT
    #         for groups that have both
    # ─────────────────────────────────────────────────────
    compare_conds = ["5kHz_gt", "500Hz_gt", "v2_6kHz", "v3_6kHz"]
    fig, axes = plt.subplots(2, len(compare_conds), figsize=(18, 9))
    fig.suptitle("Per‑Group Scatter vs. 20 kHz Ground Truth", fontsize=13, fontweight="bold")

    for col_idx, comp in enumerate(compare_conds):
        for row_idx, met in enumerate(["cs_m_s", "G_prime_kPa"]):
            ax = axes[row_idx, col_idx]
            # Collect per-group means
            x_list, y_list, src_list = [], [], []

            for g in df["group"].dropna().unique():
                sub_ref  = df[(df.group == g) & (df.condition == "20kHz_gt")]
                sub_comp = df[(df.group == g) & (df.condition == comp)]
                if len(sub_ref) == 0 or len(sub_comp) == 0:
                    continue
                x_val = sub_ref[met].mean()
                y_val = sub_comp[met].mean()
                if not (np.isfinite(x_val) and np.isfinite(y_val)):
                    continue
                x_list.append(x_val)
                y_list.append(y_val)
                src_list.append(sub_ref["source"].iloc[0])

            if not x_list:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            x_arr = np.array(x_list)
            y_arr = np.array(y_list)
            colors_sc = ["#2196F3" if s == "2ft" else "#F44336" for s in src_list]

            ax.scatter(x_arr, y_arr, c=colors_sc, alpha=0.82, s=45, edgecolors="k", lw=0.4)
            lims  = [min(x_arr.min(), y_arr.min()) * 0.9,
                     max(x_arr.max(), y_arr.max()) * 1.1]
            ax.plot(lims, lims, "k--", lw=1.0, alpha=0.5, label="y = x")

            # Pearson r + RMSE
            if len(x_arr) >= 3:
                r, pval = pearsonr(x_arr, y_arr)
                rmse    = np.sqrt(np.mean((y_arr - x_arr)**2))
                met_unit = "m/s" if met == "cs_m_s" else "kPa"
                ax.set_title(f"{COND_META[comp]['label']}\nr={r:.3f}  RMSE={rmse:.3f} {met_unit}",
                             fontsize=8)

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            metric_names = {"cs_m_s": "cs (m/s)", "G_prime_kPa": "G' (kPa)"}
            ax.set_xlabel(f"20 kHz GT — {metric_names[met]}", fontsize=8)
            ax.set_ylabel(f"{COND_META[comp]['label']} — {metric_names[met]}", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(out / "fig3_scatter_vs_20khz.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 3 saved: fig3_scatter_vs_20khz.png")

    # ─────────────────────────────────────────────────────
    # Fig 4: Grouped bar — per plane, cs comparison
    # ─────────────────────────────────────────────────────
    groups_with_all = []
    for g in sorted(df["group"].dropna().unique()):
        conds = df[df["group"] == g]["condition"].unique()
        if "20kHz_gt" in conds and "v3_6kHz" in conds:
            groups_with_all.append(g)

    if groups_with_all:
        n_groups = len(groups_with_all)
        n_conds  = len(CONDITIONS)
        width    = 0.15
        x        = np.arange(n_groups)

        fig, (ax_cs, ax_G) = plt.subplots(2, 1, figsize=(max(12, n_groups * 1.2), 9),
                                           sharex=True)
        fig.suptitle("Per‑Group cs and G' — All Conditions", fontsize=13, fontweight="bold")

        for ci, cond in enumerate(CONDITIONS):
            cs_means, cs_stds, G_means, G_stds = [], [], [], []
            for g in groups_with_all:
                sub = df[(df.group == g) & (df.condition == cond)]
                cs_v = sub["cs_m_s"].dropna()
                G_v  = sub["G_prime_kPa"].dropna()
                cs_means.append(cs_v.mean() if len(cs_v) else np.nan)
                cs_stds.append(cs_v.std()   if len(cs_v) else 0)
                G_means.append(G_v.mean()   if len(G_v) else np.nan)
                G_stds.append(G_v.std()     if len(G_v) else 0)

            offset = (ci - n_conds / 2 + 0.5) * width
            ax_cs.bar(x + offset, cs_means, width, yerr=cs_stds,
                      label=COND_META[cond]["label"],
                      color=COND_META[cond]["color"],
                      alpha=0.82, capsize=2, error_kw=dict(elinewidth=0.8))
            ax_G.bar(x + offset, G_means, width, yerr=G_stds,
                     color=COND_META[cond]["color"],
                     alpha=0.82, capsize=2, error_kw=dict(elinewidth=0.8))

        ax_cs.set_ylabel("cs (m/s)", fontsize=10)
        ax_G.set_ylabel("G' (kPa)", fontsize=10)
        ax_G.set_xticks(x)
        ax_G.set_xticklabels([g.replace("_", "\n") for g in groups_with_all],
                              fontsize=7, rotation=30, ha="right")
        ax_cs.legend(ncol=3, fontsize=8, loc="upper right")
        ax_cs.grid(axis="y", alpha=0.3)
        ax_G.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "fig4_per_group_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure 4 saved: fig4_per_group_bar.png")

    # ─────────────────────────────────────────────────────
    # Fig 5: Dispersion curve example (best sample)
    # ─────────────────────────────────────────────────────
    _plot_dispersion_examples(df, out)


def _plot_dispersion_examples(df: pd.DataFrame, out: Path) -> None:
    """Show ToA wave propagation map for one good 5kHz and one v3 6kHz sample."""
    import glob
    # Find a 5kHz sample with finite cs
    sub5 = df[(df.condition == "5kHz_gt") & df.cs_m_s.notna()].sort_values("cs_m_s")
    sub3 = df[(df.condition == "v3_6kHz")  & df.cs_m_s.notna()].sort_values("cs_m_s")

    if len(sub5) == 0 or len(sub3) == 0:
        return

    # Pick median‑cs samples
    s5_name = sub5.iloc[len(sub5) // 2]["name"]
    s3_name = sub3.iloc[len(sub3) // 2]["name"]

    vel5_path = sorted(Path("data/fullscale_velocity").rglob(f"velocity_{s5_name}.npy"))
    vel3_path = list(Path("outputs/6khz_predictions_v3").glob(f"velocity_6khz_{s3_name}.npy"))

    if not vel5_path or not vel3_path:
        return

    vel5 = np.load(str(vel5_path[0]))
    vel3 = np.load(str(vel3_path[0]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Shear Wave Propagation: Spatiotemporal Maps\n"
                 "(averaged over central 50% of depth)", fontsize=12, fontweight="bold")

    dt5 = 1 / 5000
    dt3 = 1 / 6000
    T5, H5, W5 = vel5.shape
    T3, H3, W3 = vel3.shape

    for ax_abs, ax_signed, vel, dt, label in [
        (axes[0, 0], axes[0, 1], vel5, dt5, f"GT 5 kHz — {s5_name[:40]}"),
        (axes[1, 0], axes[1, 1], vel3, dt3, f"Pred 6 kHz (v3) — {s3_name[:40]}"),
    ]:
        T, H, W = vel.shape
        v_avg = np.mean(vel[:, H // 4: 3 * H // 4, :], axis=1)  # (T, W)

        t_ax = np.arange(T) * dt * 1000   # ms
        x_ax = np.arange(W) * LATERAL_MM_PER_PX  # mm

        # abs value map
        im = ax_abs.imshow(np.abs(v_avg), aspect="auto", origin="lower",
                           extent=[x_ax[0], x_ax[-1], t_ax[0], t_ax[-1]],
                           cmap="hot", interpolation="bilinear")
        plt.colorbar(im, ax=ax_abs, label="|velocity| (mm/s)", pad=0.02)
        ax_abs.set_xlabel("Lateral (mm)", fontsize=9)
        ax_abs.set_ylabel("Time (ms)",    fontsize=9)
        ax_abs.set_title(f"|v(t,x)| — {label}", fontsize=8)

        # overlay ToA
        peak_row_idx = np.argmax(np.abs(v_avg), axis=0)
        peak_row_t   = peak_row_idx * dt * 1000
        ax_abs.plot(x_ax, peak_row_t, "c--", lw=1.5, label="ToA peak")
        ax_abs.legend(fontsize=7)

        # signed map
        vmax = np.percentile(np.abs(v_avg), 97)
        im2  = ax_signed.imshow(v_avg, aspect="auto", origin="lower",
                                extent=[x_ax[0], x_ax[-1], t_ax[0], t_ax[-1]],
                                cmap="seismic", vmin=-vmax, vmax=vmax,
                                interpolation="bilinear")
        plt.colorbar(im2, ax=ax_signed, label="velocity (mm/s)", pad=0.02)
        ax_signed.set_xlabel("Lateral (mm)", fontsize=9)
        ax_signed.set_ylabel("Time (ms)",    fontsize=9)
        ax_signed.set_title(f"v(t,x) [signed] — {label}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out / "fig5_spatiotemporal_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 5 saved: fig5_spatiotemporal_maps.png")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SWE Elasticity/Viscosity Analysis")
    parser.add_argument("--output_dir",  default="outputs/swe_analysis")
    parser.add_argument("--gt_meta",     default="data/fullscale_velocity/metadata.json")
    parser.add_argument("--v2_pred_dir", default="outputs/6khz_predictions")
    parser.add_argument("--v3_pred_dir", default="outputs/6khz_predictions_v3")
    args = parser.parse_args()

    run_analysis(
        output_dir  = args.output_dir,
        gt_meta_path= args.gt_meta,
        v2_pred_dir = args.v2_pred_dir,
        v3_pred_dir = args.v3_pred_dir,
    )


if __name__ == "__main__":
    main()
