#!/usr/bin/env python3
"""
compute_interpolation_baselines.py
===================================
Computes temporal super-resolution SWE baselines using classical interpolation
methods, then applies the same Kelvin-Voigt inversion as swe_viscoelastic_v10b.py.

Baseline methods (applied per-pixel along the temporal axis):
  1. Linear interpolation   -- piecewise linear between 5 sparse 500 Hz frames
  2. Cubic-spline (RBF)     -- natural cubic spline through 5 sparse 500 Hz frames

Procedure
---------
For each sample:
  1. Load the first 5 frames of the native 500 Hz velocity sequence
     (data/fullscale_velocity/inference/velocity_<name>.npy[:5]).
     These frames are at t = 0, 2, 4, 6, 8 ms (dt = 2 ms, 500 Hz).
  2. Interpolate per-pixel to 49 frames at 6 kHz
     (t = 0, 1/6000, ..., 48/6000 seconds).
  3. Apply the same estimate_cs() cross-correlation ToA method used in
     swe_publication_v10b.py with dt = 1/6000 s.
  4. Derive G_dyn = rho * cs^2 and apply Kelvin-Voigt inversion for
     2FT / 3FT phantoms (CIRS is excluded from KV inversion as in the
     original analysis).

Outputs
-------
  outputs/interp_baselines/interp_swe_metrics.csv   -- raw per-sample metrics
  outputs/interp_baselines/interp_kv_report.txt     -- human-readable report

Usage
-----
  python scripts/compute_interpolation_baselines.py
"""

from __future__ import annotations

import glob
import json
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── constants (mirror swe_publication_v10b.py) ─────────────────────────────────
RHO              = 1000.0          # kg/m³
LATERAL_M_PER_PX = 0.1013e-3      # m/px
DT_6KHZ          = 1.0 / 6000.0   # s  — interpolated output frame interval
DT_500HZ         = 1.0 / 500.0    # s  — native input frame interval

# Kelvin-Voigt quasi-static priors (same as swe_viscoelastic_v10b.py)
G_STATIC_PA = {
    "2ft": 2000.0 / 3.0,   # Pa  (E=2kPa, E=3G ⟹ G=667 Pa)
    "3ft": 3000.0 / 3.0,   # Pa  (E=3kPa)
}
F_PUSH_HZ = 400.0           # assumed SWE push frequency for KV inversion

N_SPARSE  = 5    # number of 500 Hz frames used as interpolation input
N_DENSE   = 49   # number of 6 kHz frames (target), must equal 1+(N_SPARSE-1)*12
assert 1 + (N_SPARSE - 1) * 12 == N_DENSE, "N_DENSE inconsistent with N_SPARSE × 12"

_GROUP_RE = re.compile(r'((?:2ft|3ft|cirs)_\w+focus(?:_\w+inclusion)?_Plane_\d+)')


# ══════════════════════════════════════════════════════════════════════════════
# SWE cross-correlation ToA (copied verbatim from swe_publication_v10b.py)
# ══════════════════════════════════════════════════════════════════════════════

def _push_column(v_abs: np.ndarray, n_early: int = 4) -> int:
    T, Wx = v_abs.shape
    early = v_abs[:min(n_early, T), :]
    return int(np.argmax(early.max(axis=0)))


def estimate_cs(velocity: np.ndarray,
                dt_s: float,
                dx_m: float = LATERAL_M_PER_PX,
                depth_lo: float = 0.20,
                depth_hi: float = 0.80,
                excl_px: int = 10) -> Tuple[float, float, np.ndarray]:
    """
    Estimate shear-wave speed via pairwise cross-correlation ToA.
    Identical to the implementation in swe_publication_v10b.py.
    """
    T, Ht, Wt = velocity.shape
    r0 = int(Ht * depth_lo)
    r1 = int(Ht * depth_hi)

    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)   # (T, W)
    v_abs = np.abs(v_avg)

    amp   = v_abs.max(axis=0)
    thr   = np.percentile(amp[amp > 0], 15) if amp.max() > 0 else 0
    valid = amp > thr

    x_push  = _push_column(v_abs, n_early=max(3, T // 10))
    excl_lo = max(0, x_push - excl_px)
    excl_hi = min(Wt - 1, x_push + excl_px)
    valid[excl_lo: excl_hi + 1] = False

    target_lag_frames = 1.0
    cs_guess = 4.0
    min_step = max(3, int(np.ceil(target_lag_frames * cs_guess * dt_s / dx_m)))
    STEP = min(min_step, 12)

    cs_map = np.full(Wt, np.nan)
    cs_all = []

    for x in range(0, Wt - STEP):
        x2 = x + STEP
        if not (valid[x] and valid[x2]):
            continue
        a = v_avg[:, x]
        b = v_avg[:, x2]
        ea = np.sqrt((a ** 2).sum()); eb = np.sqrt((b ** 2).sum())
        if ea < 1e-12 or eb < 1e-12:
            continue
        xc   = np.correlate(a / ea, b / eb, mode="full")
        lags = np.arange(-(T - 1), T, dtype=float)
        pi   = int(np.argmax(xc))
        if 0 < pi < len(xc) - 1:
            y0, y1, y2 = xc[pi - 1], xc[pi], xc[pi + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            lag_frames = lags[pi] + frac
        else:
            lag_frames = lags[pi]
        if abs(lag_frames) < 0.15:
            continue
        cs_local = abs(STEP * dx_m / (lag_frames * dt_s))
        if 0.3 < cs_local < 12.0:
            cs_all.append(cs_local)
            cs_map[x] = cs_local

    if len(cs_all) < 3:
        return np.nan, np.nan, cs_map

    cs_arr = np.array(cs_all)
    cs_med = float(np.median(cs_arr))
    iqr    = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    cs_std = iqr / 1.349

    if not (0.3 < cs_med < 12.0):
        return np.nan, np.nan, cs_map

    return cs_med, cs_std, cs_map


# ══════════════════════════════════════════════════════════════════════════════
# Kelvin-Voigt inversion (copied verbatim from swe_viscoelastic_v10b.py)
# ══════════════════════════════════════════════════════════════════════════════

def kv_invert(cs_m_s: float, source: str, f_push_hz: float = F_PUSH_HZ):
    out_nan = {k: np.nan for k in
               ["G_prime_kpa", "eta_Pas", "G_loss_kpa", "G_star_kpa",
                "tan_delta", "E_kv_kpa"]}
    if not np.isfinite(cs_m_s) or cs_m_s <= 0:
        return out_nan
    source = str(source).lower()
    if source not in G_STATIC_PA:
        return out_nan
    G_e   = float(G_STATIC_PA[source])
    omega = 2.0 * np.pi * float(f_push_hz)
    cs2rho = cs_m_s ** 2 * RHO
    disc   = cs2rho ** 2 + 8.0 * cs2rho * G_e
    if disc < 0:
        return out_nan
    g_star_pa = (cs2rho + np.sqrt(disc)) / 4.0
    visc_term_sq = g_star_pa ** 2 - G_e ** 2
    if visc_term_sq < 0:
        return out_nan
    eta = np.sqrt(visc_term_sq) / omega
    if not np.isfinite(eta) or eta <= 0 or eta > 100:
        return out_nan
    g_loss_pa   = omega * eta
    g_prime_kpa = G_e / 1000.0
    g_loss_kpa  = g_loss_pa / 1000.0
    g_star_kpa  = g_star_pa / 1000.0
    tan_delta   = g_loss_kpa / g_prime_kpa if g_prime_kpa > 0 else np.nan
    return {
        "G_prime_kpa": g_prime_kpa,
        "eta_Pas":     float(eta),
        "G_loss_kpa":  g_loss_kpa,
        "G_star_kpa":  g_star_kpa,
        "tan_delta":   tan_delta,
        "E_kv_kpa":    3.0 * g_prime_kpa,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Temporal interpolation baselines
# ══════════════════════════════════════════════════════════════════════════════

def temporal_upsample(vel_sparse: np.ndarray,
                      t_sparse: np.ndarray,
                      t_dense: np.ndarray,
                      method: str = "linear") -> np.ndarray:
    """
    Upsample velocity (T_sparse, H, W) → (T_dense, H, W) per-pixel.

    Parameters
    ----------
    vel_sparse : (T_sparse, H, W)
    t_sparse   : (T_sparse,) time axis for sparse frames [s]
    t_dense    : (T_dense,)  target time axis [s]
    method     : "linear" | "cubic"
    """
    T_s, H, W = vel_sparse.shape
    T_d = len(t_dense)

    vel_dense = np.empty((T_d, H, W), dtype=np.float32)
    vel_flat  = vel_sparse.reshape(T_s, -1)   # (T_s, H*W)

    # Clip query times to avoid unstable extrapolation beyond the sparse range
    t_clipped = np.clip(t_dense, t_sparse[0], t_sparse[-1])

    f = interp1d(
        t_sparse, vel_flat, axis=0,
        kind=method,
        bounds_error=False,
        fill_value=(vel_flat[0], vel_flat[-1]),  # constant boundary extension
    )
    vel_dense = f(t_clipped).reshape(T_d, H, W).astype(np.float32)
    return vel_dense


def _source(name: str) -> str:
    if "2ft" in name: return "2ft"
    if "3ft" in name: return "3ft"
    return "cirs"


def _group_key(name: str) -> Optional[str]:
    # Try with inclusion
    m = _GROUP_RE.search(name)
    if m:
        return m.group(1)
    # Fallback: 2ft/3ft_<focus>_Plane_<n>
    m2 = re.search(r'((?:2ft|3ft|cirs)_\w+focus_Plane_\d+)', name)
    return m2.group(1) if m2 else None


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(
    inference_dir: str = "data/fullscale_velocity/inference",
    pred_dir:      str = "outputs/6khz_predictions",
    output_dir:    str = "outputs/interp_baselines",
) -> pd.DataFrame:

    inf_dir  = Path(inference_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── sparse time axis (5 frames at 500 Hz) and dense target axis (6 kHz) ──
    t_sparse = np.arange(N_SPARSE) * DT_500HZ                     # [0, 2, 4, 6, 8] ms
    t_dense  = np.arange(N_DENSE)  * DT_6KHZ                      # [0 … 8.0] ms

    # ── discover all 500 Hz _w0 velocity files that also have predictions ─────
    pred_names = set(
        Path(p).stem.replace("velocity_6khz_", "")
        for p in glob.glob(str(Path(pred_dir) / "velocity_6khz_*.npy"))
    )

    vel_files = sorted(inf_dir.glob("velocity_*500Hz*_w0.npy"))
    # also include CIRS (no _w0 suffix)
    vel_files += sorted(inf_dir.glob("velocity_cirs*500Hz*.npy"))

    rows = []

    for vf in vel_files:
        name = vf.stem.replace("velocity_", "")
        if name not in pred_names:
            continue

        vel_full = np.load(str(vf)).astype(np.float32)  # (49, H, W) at 500 Hz

        if vel_full.shape[0] < N_SPARSE:
            print(f"  [skip] {name}: only {vel_full.shape[0]} frames")
            continue

        vel_sparse = vel_full[:N_SPARSE]   # first 5 frames, t = 0-8 ms at 500 Hz
        src   = _source(name)
        group = _group_key(name)

        for method in ("linear", "cubic"):
            vel_dense = temporal_upsample(vel_sparse, t_sparse, t_dense, method=method)
            cs, cs_std, _ = estimate_cs(vel_dense, DT_6KHZ)
            kv = kv_invert(cs, src)
            row = {
                "condition": f"interp_{method}",
                "name":      name,
                "group":     group,
                "source":    src,
                "cs_m_s":    cs,
                "cs_std":    cs_std,
                "G_dyn_kpa": float(RHO * cs ** 2 / 1e3) if np.isfinite(cs) else np.nan,
                "E_dyn_kpa": float(3.0 * RHO * cs ** 2 / 1e3) if np.isfinite(cs) else np.nan,
            }
            row.update(kv)
            rows.append(row)

        print(f"  {name}: linear cs={rows[-2]['cs_m_s']:.3f}  cubic cs={rows[-1]['cs_m_s']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "interp_swe_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df)} rows)")
    return df


def summarize(df: pd.DataFrame):
    def fmt(s: pd.Series) -> str:
        v = s.dropna().to_numpy(float)
        if v.size == 0: return "N/A"
        if v.size == 1: return f"{v[0]:.3f}"
        return f"{v.mean():.3f} ± {v.std():.3f}"

    print("\n" + "=" * 88)
    print("INTERPOLATION BASELINE REPORT")
    print("=" * 88)
    print(f"{'Condition':<24} {'Source':<8} {'n':>4}  {'cs (m/s)':>14}  {'E_dyn (kPa)':>14}  {'eta (Pa.s)':>12}")
    print("-" * 88)
    for cond in ("interp_linear", "interp_cubic"):
        for src in ("2ft", "3ft", "cirs"):
            sub = df[(df["condition"] == cond) & (df["source"] == src)]
            n = sub["cs_m_s"].dropna().shape[0]
            print(f"  {cond:<22} {src:<8} {n:>4}  {fmt(sub['cs_m_s']):>14}  "
                  f"{fmt(sub['E_dyn_kpa']):>14}  {fmt(sub['eta_Pas']):>12}")
    print("=" * 88)


if __name__ == "__main__":
    df = run()
    summarize(df)
