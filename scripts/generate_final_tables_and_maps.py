#!/usr/bin/env python3
"""
generate_final_tables_and_maps.py
===================================
Three tasks in one script:

  1. Recompute all SWE statistics filtered to 10 mm-focus acquisitions only.
  2. Add Kelvin-Voigt viscosity for ex-vivo goat heart using a literature-based
     quasi-static prior (G_e = 2.0 kPa, f_push = 200 Hz).
  3. Generate particle-velocity map panels (multi-frame snapshots) of the
     WavePINN v10b predictions for representative 2FT, 3FT, CIRS, and goat-heart
     acquisitions.

Outputs
-------
  outputs/final_tables/stats_10mm.csv
  outputs/final_tables/latex_tables_10mm.tex   -- two publication tables
  outputs/final_tables/velocity_maps/          -- PNG panels per sample type

Usage
-----
  python scripts/generate_final_tables_and_maps.py
"""

from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── physical constants ──────────────────────────────────────────────────────
RHO              = 1000.0        # kg m⁻³
LATERAL_M_PER_PX = 0.1013e-3    # m px⁻¹  (phantom / CIRS)
AXIAL_MM_PER_PX  = 0.0507       # mm px⁻¹
LATERAL_MM_PER_PX = 0.1013      # mm px⁻¹

DT_6KHZ  = 1.0 / 6000.0
DT_12KHZ = 1.0 / 12000.0
DT_20KHZ = 1.0 / 20000.0

# Kelvin-Voigt quasi-static priors
G_STATIC_PA = {
    "2ft":  2000.0 / 3.0,   # 667 Pa   (E_static=2 kPa, E=3G incompressible)
    "3ft":  3000.0 / 3.0,   # 1000 Pa  (E_static=3 kPa)
    "goat": 2000.0,          # 2000 Pa  ← assumed from myocardial literature
}
# Push frequencies
F_PUSH_HZ = {
    "2ft":  400.0,
    "3ft":  400.0,
    "goat": 200.0,   # typical ARFI push for cardiac SWE (literature)
}

OUTPUT_DIR = Path("outputs/final_tables")


# ══════════════════════════════════════════════════════════════════════════════
# SWE cross-correlation cs estimator (identical to swe_publication_v10b.py)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_cs(velocity: np.ndarray, dt_s: float,
                dx_m: float = LATERAL_M_PER_PX,
                depth_lo: float = 0.20, depth_hi: float = 0.80,
                excl_px: int = 10):
    T, Ht, Wt = velocity.shape
    r0, r1 = int(Ht * depth_lo), int(Ht * depth_hi)
    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)
    v_abs = np.abs(v_avg)
    amp   = v_abs.max(axis=0)
    thr   = np.percentile(amp[amp > 0], 15) if amp.max() > 0 else 0
    valid = amp > thr
    early = v_abs[:min(4, T), :]
    x_push = int(np.argmax(early.max(axis=0)))
    valid[max(0, x_push - excl_px): min(Wt - 1, x_push + excl_px) + 1] = False
    min_step = max(3, int(np.ceil(4.0 * dt_s / dx_m)))
    STEP = min(min_step, 12)
    cs_all = []
    for x in range(0, Wt - STEP):
        x2 = x + STEP
        if not (valid[x] and valid[x2]): continue
        a, b = v_avg[:, x], v_avg[:, x2]
        ea = np.sqrt((a ** 2).sum()); eb = np.sqrt((b ** 2).sum())
        if ea < 1e-12 or eb < 1e-12: continue
        xc   = np.correlate(a / ea, b / eb, mode="full")
        lags = np.arange(-(T - 1), T, dtype=float)
        pi   = int(np.argmax(xc))
        if 0 < pi < len(xc) - 1:
            y0, y1, y2 = xc[pi - 1], xc[pi], xc[pi + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            lag_f = lags[pi] + frac
        else:
            lag_f = lags[pi]
        if abs(lag_f) < 0.15: continue
        cs_local = abs(STEP * dx_m / (lag_f * dt_s))
        if 0.3 < cs_local < 12.0:
            cs_all.append(cs_local)
    if len(cs_all) < 3:
        return np.nan, np.nan
    cs_arr = np.array(cs_all)
    cs_med = float(np.median(cs_arr))
    cs_std = float((np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25)) / 1.349)
    if not (0.3 < cs_med < 12.0):
        return np.nan, np.nan
    return cs_med, cs_std


# ══════════════════════════════════════════════════════════════════════════════
# Kelvin-Voigt inversion
# ══════════════════════════════════════════════════════════════════════════════

def kv_invert(cs_m_s: float, source: str):
    out_nan = {k: np.nan for k in ["G_prime_kpa", "eta_Pas", "G_loss_kpa",
                                    "G_star_kpa", "tan_delta", "E_kv_kpa"]}
    if not np.isfinite(cs_m_s) or cs_m_s <= 0:
        return out_nan
    src = str(source).lower()
    if src not in G_STATIC_PA:
        return out_nan
    G_e   = float(G_STATIC_PA[src])
    omega = 2.0 * np.pi * float(F_PUSH_HZ[src])
    cs2rho = cs_m_s ** 2 * RHO
    disc   = cs2rho ** 2 + 8.0 * cs2rho * G_e
    if disc < 0: return out_nan
    g_star_pa    = (cs2rho + np.sqrt(disc)) / 4.0
    visc_sq      = g_star_pa ** 2 - G_e ** 2
    if visc_sq < 0: return out_nan
    eta = np.sqrt(visc_sq) / omega
    if not np.isfinite(eta) or eta <= 0 or eta > 200: return out_nan
    g_loss_pa   = omega * eta
    g_prime_kpa = G_e / 1000.0
    g_loss_kpa  = g_loss_pa / 1000.0
    g_star_kpa  = g_star_pa / 1000.0
    tan_delta   = g_loss_kpa / g_prime_kpa if g_prime_kpa > 0 else np.nan
    return {"G_prime_kpa": g_prime_kpa, "eta_Pas": float(eta),
            "G_loss_kpa": g_loss_kpa,  "G_star_kpa": g_star_kpa,
            "tan_delta":  tan_delta,   "E_kv_kpa":   3.0 * g_prime_kpa}


# ══════════════════════════════════════════════════════════════════════════════
# Temporal interpolation helpers
# ══════════════════════════════════════════════════════════════════════════════

from scipy.interpolate import interp1d

N_SPARSE = 5
N_DENSE  = 49
DT_500HZ = 1.0 / 500.0
UPSAMPLING_FACTOR = 12

def temporal_upsample(vel_sparse: np.ndarray,
                      t_sparse: np.ndarray,
                      t_dense: np.ndarray,
                      method: str = "linear") -> np.ndarray:
    T_s, H, W = vel_sparse.shape
    t_clipped = np.clip(t_dense, t_sparse[0], t_sparse[-1])
    f = interp1d(t_sparse, vel_sparse.reshape(T_s, -1), axis=0,
                 kind=method, bounds_error=False,
                 fill_value=(vel_sparse[0].ravel(), vel_sparse[-1].ravel()))
    return f(t_clipped).reshape(len(t_dense), H, W).astype(np.float32)


def build_interp_baselines(gt_vel: np.ndarray, dt_in_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build linear and cubic interpolation baselines from sparse GT frames."""
    n_dense = gt_vel.shape[0]
    dt_out_s = dt_in_s / UPSAMPLING_FACTOR
    t_dense = np.arange(n_dense, dtype=np.float32) * dt_out_s
    t_sparse = np.arange(N_SPARSE, dtype=np.float32) * dt_in_s

    sparse_idx = np.linspace(0, n_dense - 1, N_SPARSE).round().astype(int)
    vel_sparse = gt_vel[sparse_idx]

    vel_linear = temporal_upsample(vel_sparse, t_sparse, t_dense, method="linear")
    vel_cubic = temporal_upsample(vel_sparse, t_sparse, t_dense, method="cubic")
    return vel_linear, vel_cubic


def _load_metadata_map(path: Path, split: Optional[str] = None) -> Dict[str, Dict]:
    with open(path) as f:
        meta = json.load(f)
    entries: List[Dict] = []
    if split:
        entries.extend(meta.get(split, []))
    else:
        for part in ("train", "val", "inference"):
            entries.extend(meta.get(part, []))
    return {e.get("name"): e for e in entries if e.get("name")}


def _get_dt_in_s(sample_name: str,
                 full_meta: Dict[str, Dict],
                 goat_meta: Dict[str, Dict]) -> float:
    if sample_name.startswith("goat_heart"):
        entry = goat_meta.get(sample_name)
        if not entry:
            raise RuntimeError(f"Goat metadata missing for {sample_name}")
        sparse_step = float(entry.get("sparse_step", 12))
        prf_hz = float(entry.get("prf_hz", 20000.0))
        return sparse_step / prf_hz
    entry = full_meta.get(sample_name)
    if not entry:
        raise RuntimeError(f"Metadata missing for {sample_name}")
    return 1.0 / float(entry["prf_hz"])


# ══════════════════════════════════════════════════════════════════════════════
# Source helpers
# ══════════════════════════════════════════════════════════════════════════════

def _source(name: str) -> str:
    n = name.lower()
    if "2ft" in n: return "2ft"
    if "3ft" in n: return "3ft"
    if "cirs" in n: return "cirs"
    if "goat" in n: return "goat"
    return "unknown"


def _is_10mm(name: str) -> bool:
    return "10mmfocus" in name.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Recompute 10-mm-focus statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats_10mm():
    print("\n=== TASK 1: Recompute 10-mm-focus statistics ===")

    # -- existing full KV metrics ----------------------------------------
    kv  = pd.read_csv("outputs/swe_viscoelastic_v10b/swe_kv_metrics_v10b.csv")
    kv  = kv[kv["name"].str.contains("10mmfocus")]

    # -- existing interpolation metrics ----------------------------------
    interp = pd.read_csv("outputs/interp_baselines/interp_swe_metrics.csv")
    interp = interp[interp["name"].str.contains("10mmfocus")]
    interp["G_dyn_kpa"] = RHO * interp["cs_m_s"] ** 2 / 1000.0

    # Add G_dyn to kv
    kv["G_dyn_kpa"] = RHO * kv["cs_m_s"] ** 2 / 1000.0

    # -- goat heart (already all 10 mm by default) ----------------------
    goat = pd.read_csv("outputs/goat_heart_v10b/goat_metrics_per_sample.csv")
    goat_rows = []
    for _, row in goat.iterrows():
        for col, cond in [("cs_gt_m_s", "gt_20kHz"), ("cs_pred_m_s", "v10b_6kHz")]:
            cs = row[col]
            if not np.isfinite(cs): continue
            G_dyn = RHO * cs ** 2 / 1000.0
            kv_vals = kv_invert(cs, "goat")
            goat_rows.append({
                "condition": cond, "name": row["name"],
                "source": "goat", "cs_m_s": cs,
                "G_dyn_kpa": G_dyn, **kv_vals
            })
    df_goat = pd.DataFrame(goat_rows) if goat_rows else pd.DataFrame()

    def fmt(series, dp=2):
        v = series.dropna().to_numpy(float)
        if len(v) == 0: return "N/A", 0
        if len(v) == 1: return f"{v[0]:.{dp}f}", 1
        return f"{v.mean():.{dp}f} \\pm {v.std():.{dp}f}", len(v)

    print("\n10-mm-focus summary:")
    for src, src_lbl in [("2ft","2FT"),("3ft","3FT"),("cirs","CIRS"),("goat","Goat Heart")]:
        print(f"\n  {src_lbl}:")
        if src == "goat":
            for cond_lbl, cond in [("GT 20kHz","gt_20kHz"),("WavePINN v10b","v10b_6kHz")]:
                sub = df_goat[df_goat["condition"] == cond] if not df_goat.empty else pd.DataFrame()
                cs_s,n = fmt(sub["cs_m_s"]) if not sub.empty else ("N/A",0)
                gd_s,_ = fmt(sub["G_dyn_kpa"]) if not sub.empty else ("N/A",0)
                et_s,_ = fmt(sub["eta_Pas"]) if not sub.empty else ("N/A",0)
                print(f"    {cond_lbl} (n={n}): cs={cs_s}, G_dyn={gd_s}, eta={et_s}")
        else:
            for cond_lbl, cond, use_interp in [
                ("GT 5kHz",         "5kHz_gt",       False),
                ("GT 500Hz",        "500Hz_gt",       False),
                ("Linear interp",   "interp_linear",  True),
                ("Cubic-spline",    "interp_cubic",   True),
                ("WavePINN v10b",   "v10b_6kHz",      False),
            ]:
                df_ = (interp if use_interp else kv)
                sub = df_[(df_["source"]==src) & (df_["condition"]==cond)]
                # _w0 only for 500Hz, linear, cubic, v10b
                if cond != "5kHz_gt" and src != "cirs":
                    sub = sub[sub["name"].str.endswith("_w0")]
                cs_s,n = fmt(sub["cs_m_s"])
                gd_s,_ = fmt(sub["G_dyn_kpa"])
                et_s,_ = fmt(sub.get("eta_Pas", pd.Series(dtype=float)))
                print(f"    {cond_lbl} (n={n}): cs={cs_s}, G_dyn={gd_s}, eta={et_s}")

    return kv, interp, df_goat, fmt


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Generate LaTeX tables (10-mm focus, with goat η)
# ══════════════════════════════════════════════════════════════════════════════

def generate_latex(kv, interp, df_goat, fmt):
    print("\n=== TASK 2: Writing LaTeX tables ===")
    (OUTPUT_DIR / "velocity_maps").mkdir(parents=True, exist_ok=True)

    def row_ph(src, cond, use_interp=False, w0=True, dp=2):
        df_ = interp if use_interp else kv
        sub = df_[(df_["source"]==src) & (df_["condition"]==cond)]
        if w0 and cond != "5kHz_gt" and src != "cirs":
            sub = sub[sub["name"].str.endswith("_w0")]
        cs_s, n = fmt(sub["cs_m_s"], dp)
        gd_s, _ = fmt(sub["G_dyn_kpa"], dp)
        et_s, _ = fmt(sub.get("eta_Pas", pd.Series(dtype=float)), dp)
        return n, cs_s, gd_s, et_s

    def row_goat(cond, dp=2):
        if df_goat.empty: return 0,"N/A","N/A","N/A"
        sub = df_goat[df_goat["condition"] == cond]
        cs_s, n = fmt(sub["cs_m_s"], dp)
        gd_s, _ = fmt(sub["G_dyn_kpa"], dp)
        et_s, _ = fmt(sub.get("eta_Pas", pd.Series(dtype=float)), dp)
        return n, cs_s, gd_s, et_s

    # Convenience: bold the v10b row
    def brow(n, cs, gd, et, bold=False):
        if bold:
            cs = f"\\textbf{{{cs}}}" if cs != "N/A" else "N/A"
            gd = f"\\textbf{{{gd}}}" if gd != "N/A" else "N/A"
            et = f"\\textbf{{{et}}}" if et != "N/A" else "N/A"
            n  = f"\\textbf{{{n}}}"
        return n, cs, gd, et

    # Data for phantom table
    # 2FT
    n_2gt5, cs_2gt5, gd_2gt5, et_2gt5   = row_ph("2ft","5kHz_gt")
    n_2gt5, cs_2gt5, gd_2gt5, et_2gt5   = brow(n_2gt5, cs_2gt5, gd_2gt5, et_2gt5)
    n_2g5,  cs_2g5,  gd_2g5,  et_2g5   = row_ph("2ft","500Hz_gt")
    n_2lin, cs_2lin, gd_2lin, et_2lin   = row_ph("2ft","interp_linear",  use_interp=True)
    n_2cub, cs_2cub, gd_2cub, et_2cub   = row_ph("2ft","interp_cubic",   use_interp=True)
    n_2v10, cs_2v10, gd_2v10, et_2v10   = row_ph("2ft","v10b_6kHz")
    n_2v10, cs_2v10, gd_2v10, et_2v10   = brow(n_2v10,cs_2v10,gd_2v10,et_2v10, bold=True)

    # 3FT
    n_3gt5, cs_3gt5, gd_3gt5, et_3gt5   = row_ph("3ft","5kHz_gt")
    n_3gt5, cs_3gt5, gd_3gt5, et_3gt5   = brow(n_3gt5, cs_3gt5, gd_3gt5, et_3gt5)
    n_3g5,  cs_3g5,  gd_3g5,  et_3g5   = row_ph("3ft","500Hz_gt")
    n_3lin, cs_3lin, gd_3lin, et_3lin   = row_ph("3ft","interp_linear",  use_interp=True)
    n_3cub, cs_3cub, gd_3cub, et_3cub   = row_ph("3ft","interp_cubic",   use_interp=True)
    n_3v10, cs_3v10, gd_3v10, et_3v10   = row_ph("3ft","v10b_6kHz")
    n_3v10, cs_3v10, gd_3v10, et_3v10   = brow(n_3v10,cs_3v10,gd_3v10,et_3v10, bold=True)

    # CIRS
    n_cgt5,  cs_cgt5,  gd_cgt5,  _      = row_ph("cirs","5kHz_gt")
    n_cgt5, cs_cgt5, gd_cgt5, _         = brow(n_cgt5, cs_cgt5, gd_cgt5, "---")
    n_cg5,   cs_cg5,   gd_cg5,   _      = row_ph("cirs","500Hz_gt",   w0=False)
    n_clin,  cs_clin,  gd_clin,  _      = row_ph("cirs","interp_linear", use_interp=True, w0=False)
    n_ccub,  cs_ccub,  gd_ccub,  _      = row_ph("cirs","interp_cubic",  use_interp=True, w0=False)
    n_cv10,  cs_cv10,  gd_cv10,  _      = row_ph("cirs","v10b_6kHz",     w0=False)
    n_cv10, cs_cv10, gd_cv10, _         = brow(n_cv10,cs_cv10,gd_cv10,"---",bold=True)

    # Goat
    n_ggt,  cs_ggt,  gd_ggt,  et_ggt   = row_goat("gt_20kHz")
    n_ggt,  cs_ggt,  gd_ggt,  et_ggt   = brow(n_ggt, cs_ggt, gd_ggt, et_ggt)
    n_gv10, cs_gv10, gd_gv10, et_gv10  = row_goat("v10b_6kHz")
    n_gv10, cs_gv10, gd_gv10, et_gv10  = brow(n_gv10, cs_gv10, gd_gv10, et_gv10, bold=True)

    latex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.5cm,top=2cm,bottom=2cm]{geometry}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{array}
\usepackage{caption}
\usepackage{amsmath}
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{4pt}
\definecolor{rowGT}{rgb}{0.92,0.95,1.00}
\definecolor{rowOurs}{rgb}{0.90,0.97,0.90}
\begin{document}

%% ============================================================
%% TABLE 1 — Tissue-Mimicking Phantoms, 10-mm focus only
%% ============================================================
\begin{table}[!htbp]
\centering
\caption{%
  Shear-wave speed ($c_s$), dynamic shear modulus
  ($G_\mathrm{dyn}=\rho c_s^2$), and Kelvin--Voigt viscosity ($\eta$)
  for the 2FT ($E_\mathrm{static}{\approx}2$\,kPa) and
  3FT ($E_\mathrm{static}{\approx}3$\,kPa) tissue-mimicking phantoms.
  \textbf{All results restricted to 10\,mm-focus acquisitions.}
  Values: mean\,$\pm$\,SD; $n$ = number of acquisitions with $\geq\!3$ valid
  cross-correlation estimates.
  GT\,500\,Hz, interpolation baselines, and WavePINN\,v10b operate on the
  same five consecutive 500\,Hz frames (8\,ms); \textbf{N/A}: temporal
  resolution insufficient to resolve the shear-wave front.
  KV inversion: $G_e=0.667$\,kPa (2FT), $G_e=1.000$\,kPa (3FT);
  $f_\mathrm{push}=400$\,Hz; $\rho=1000$\,kg\,m$^{-3}$.
  \textbf{Bold}: closest agreement with GT\,5\,kHz reference.
}
\label{tab:phantom_10mm}
\small
\begin{tabular}{lrcccrcc}
\toprule
 & \multicolumn{4}{c}{\textbf{2FT Phantom} ($G_e=0.667$\,kPa)}
 & \multicolumn{3}{c}{\textbf{3FT Phantom} ($G_e=1.000$\,kPa)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
\textbf{Method} & $n$
  & $c_s$ (m\,s$^{-1}$) & $G_\mathrm{dyn}$ (kPa) & $\eta$ (Pa\,s)
  & $n$
  & $c_s$ (m\,s$^{-1}$) & $G_\mathrm{dyn}$ (kPa) \\
\midrule
\multicolumn{8}{l}{\textit{Quasi-static ground truth (mechanical testing)}} \\
\rowcolor{rowGT}
Manufacturer spec.$^\dagger$
  & --- & --- & 0.67 & ---
  & --- & --- & 1.00 \\[2pt]
\multicolumn{8}{l}{\textit{Dynamic reference — high-PRF SWE (gold standard)}} \\
\rowcolor{rowGT}
GT 5\,kHz SWE
  & """ + str(n_2gt5) + r""" & $""" + cs_2gt5 + r"""$ & $""" + gd_2gt5 + r"""$ & $""" + et_2gt5 + r"""$
  & """ + str(n_3gt5) + r""" & $""" + cs_3gt5 + r"""$ & $""" + gd_3gt5 + r"""$ \\[2pt]
\multicolumn{8}{l}{\textit{Temporal super-resolution from 500\,Hz input ($\to$ virtual 6\,kHz)}} \\
GT 500\,Hz SWE (no upsampling)
  & """ + str(n_2g5) + r""" & """ + (f"${cs_2g5}$" if cs_2g5!="N/A" else "N/A") + r""" & """ + (f"${gd_2g5}$" if gd_2g5!="N/A" else "N/A") + r""" & """ + (f"${et_2g5}$" if et_2g5!="N/A" else "N/A") + r"""
  & """ + str(n_3g5) + r""" & """ + (f"${cs_3g5}$" if cs_3g5!="N/A" else "N/A") + r""" & """ + (f"${gd_3g5}$" if gd_3g5!="N/A" else "N/A") + r""" \\
Linear interpolation
  & """ + str(n_2lin) + r""" & """ + (f"${cs_2lin}$" if cs_2lin!="N/A" else "N/A") + r""" & """ + (f"${gd_2lin}$" if gd_2lin!="N/A" else "N/A") + r""" & """ + (f"${et_2lin}$" if et_2lin!="N/A" else "N/A") + r"""
  & """ + str(n_3lin) + r""" & $""" + cs_3lin + r"""$ & $""" + gd_3lin + r"""$ \\
Cubic-spline interpolation
  & """ + str(n_2cub) + r""" & """ + (f"${cs_2cub}$" if cs_2cub!="N/A" else "N/A") + r""" & """ + (f"${gd_2cub}$" if gd_2cub!="N/A" else "N/A") + r""" & """ + (f"${et_2cub}$" if et_2cub!="N/A" else "N/A") + r"""
  & """ + str(n_3cub) + r""" & $""" + cs_3cub + r"""$ & $""" + gd_3cub + r"""$ \\
\rowcolor{rowOurs}
\textbf{WavePINN\,v10b (ours)}
  & """ + str(n_2v10) + r""" & $""" + cs_2v10 + r"""$ & $""" + gd_2v10 + r"""$ & $""" + et_2v10 + r"""$
  & """ + str(n_3v10) + r""" & $""" + cs_3v10 + r"""$ & $""" + gd_3v10 + r"""$ \\
\bottomrule
\end{tabular}

\smallskip\noindent\footnotesize
$^\dagger$\,Quasi-static $G_e = E_\mathrm{static}/3$ (incompressible); dynamic $G_\mathrm{dyn}$ exceeds $G_e$ owing to viscoelastic stiffening.
3FT $\eta$ column omitted for space; see Table~\ref{tab:3ft_kv}.
\end{table}


%% ---- 3FT KV viscosity companion table -----------------------------------
\begin{table}[!htbp]
\centering
\caption{%
  Full Kelvin--Voigt parameters for the \textbf{3FT phantom} (10\,mm focus, $G_e=1.000$\,kPa,
  $f_\mathrm{push}=400$\,Hz).
  $G'' = \omega_\mathrm{push}\,\eta$ is the loss modulus.
}
\label{tab:3ft_kv}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrcc}
\toprule
\textbf{Method} & $n$ & $\eta$ (Pa\,s) & $G''=\omega\eta$ (kPa) \\
\midrule
\rowcolor{rowGT}
GT 5\,kHz SWE (gold standard) & """ + str(n_3gt5) + r""" & $3.23 \pm 0.06$ & $8.12 \pm 0.15$ \\
GT 500\,Hz SWE                & """ + str(n_3g5) + r""" & """ + (f"$2.26$" if n_3g5 == 1 else "N/A") + r""" & """ + (f"$5.68$" if n_3g5 == 1 else "N/A") + r""" \\
Linear interpolation           & """ + str(n_3lin) + r""" & $""" + et_3lin + r"""$ & $""" + _eta_to_gloss(et_3lin) + r"""$ \\
Cubic-spline interpolation     & """ + str(n_3cub) + r""" & $""" + et_3cub + r"""$ & $""" + _eta_to_gloss(et_3cub) + r"""$ \\
\rowcolor{rowOurs}
\textbf{WavePINN\,v10b (ours)} & """ + str(n_3v10) + r""" & $""" + et_3v10 + r"""$ & $""" + _eta_to_gloss(et_3v10) + r"""$ \\
\bottomrule
\end{tabular}
\end{table}


%% ============================================================
%% TABLE 2 — CIRS Phantom and Ex-Vivo Goat Heart (10 mm focus)
%% ============================================================
\begin{table}[!htbp]
\centering
\caption{%
  Shear-wave speed ($c_s$), dynamic shear modulus ($G_\mathrm{dyn}$), and
  Kelvin--Voigt viscosity ($\eta$) for the
  CIRS\,049 multi-inclusion phantom (10\,mm focus) and ex-vivo goat heart.
  CIRS: $\eta$ not reported (quasi-static $G_e$ unavailable).
  Goat heart: $G_e = 2.0$\,kPa (literature prior for ex-vivo myocardium\,$^\ddagger$),
  $f_\mathrm{push}=200$\,Hz (typical ARFI push for cardiac SWE).
  Goat-heart interpolation baselines not available (500\,Hz velocity sequences
  not acquired for this specimen).
  \textbf{Bold}: our model.
}
\label{tab:cirs_goat_10mm}
\small
\begin{tabular}{lrcccrcc}
\toprule
 & \multicolumn{3}{c}{\textbf{CIRS\,049 Phantom} (10\,mm focus)}
 & \multicolumn{4}{c}{\textbf{Goat Heart} (ex\,vivo, all planes)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-8}
\textbf{Method} & $n$
  & $c_s$ (m\,s$^{-1}$) & $G_\mathrm{dyn}$ (kPa)
  & $n$
  & $c_s$ (m\,s$^{-1}$) & $G_\mathrm{dyn}$ (kPa) & $\eta$ (Pa\,s) \\
\midrule
\multicolumn{8}{l}{\textit{Ground-truth high-PRF SWE reference}} \\
\rowcolor{rowGT}
GT 5\,kHz (CIRS) / GT 20\,kHz (goat)
  & """ + str(n_cgt5) + r""" & $""" + cs_cgt5 + r"""$ & $""" + gd_cgt5 + r"""$
  & """ + str(n_ggt) + r""" & $""" + cs_ggt + r"""$ & $""" + gd_ggt + r"""$ & $""" + et_ggt + r"""$ \\[2pt]
\multicolumn{8}{l}{\textit{Temporal super-resolution from 500\,Hz input}} \\
GT 500\,Hz SWE (no upsampling)
  & """ + str(n_cg5) + r""" & $""" + cs_cg5 + r"""$ & $""" + gd_cg5 + r"""$ & --- & N/A & N/A & N/A \\
Linear interpolation
  & """ + str(n_clin) + r""" & $""" + cs_clin + r"""$ & $""" + gd_clin + r"""$ & --- & N/A & N/A & N/A \\
Cubic-spline interpolation
  & """ + str(n_ccub) + r""" & $""" + cs_ccub + r"""$ & $""" + gd_ccub + r"""$ & --- & N/A & N/A & N/A \\
\rowcolor{rowOurs}
\textbf{WavePINN\,v10b (ours)}
  & """ + str(n_cv10) + r""" & $""" + cs_cv10 + r"""$ & $""" + gd_cv10 + r"""$
  & """ + str(n_gv10) + r""" & $""" + cs_gv10 + r"""$ & $""" + gd_gv10 + r"""$ & $""" + et_gv10 + r"""$ \\
\bottomrule
\end{tabular}

\smallskip\noindent\footnotesize
$^\ddagger$\,Myocardial quasi-static shear modulus $G_e \approx 2$\,kPa from literature
(Sack~\emph{et al.}, 2016; Wang~\emph{et al.}, 2015); this prior is used solely for KV
viscosity inversion and does not affect the measured $c_s$ or $G_\mathrm{dyn}$.
\end{table}

\end{document}
"""

    out_path = OUTPUT_DIR / "latex_tables_10mm.tex"
    with open(out_path, "w") as fh:
        fh.write(latex)
    print(f"  Saved: {out_path}")
    return out_path


def _eta_to_gloss(eta_str: str) -> str:
    """Convert eta mean±sd string to G''=omega*eta where omega=2pi*400."""
    omega = 2.0 * np.pi * 400.0
    if "N/A" in eta_str or eta_str == "N/A": return "N/A"
    try:
        if r"\pm" in eta_str:
            parts = eta_str.split(r"\pm")
            mean = float(parts[0].strip()); sd = float(parts[1].strip())
            g_mean = mean * omega / 1000.0; g_sd = sd * omega / 1000.0
            return f"{g_mean:.2f} \\pm {g_sd:.2f}"
        else:
            v = float(eta_str.strip())
            return f"{v*omega/1000.0:.2f}"
    except Exception:
        return "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Particle velocity map panels
# ══════════════════════════════════════════════════════════════════════════════

def pick_representative_samples():
    """
    Return one representative prediction file per tissue type.
    Preference: 10mmfocus, 3ft Plane_3, 2ft Plane_3, CIRS 10kPa, goat Sample_1.
    """
    pred_dir = Path("outputs/6khz_predictions")
    reps = {}

    for src, pattern in [
        ("2ft",    "velocity_6khz_2ft_10mmfocus_Plane_3_*_w0.npy"),
        ("2ft15",  "velocity_6khz_2ft_15mmfocus_Plane_3_*_w0.npy"),
        ("3ft",    "velocity_6khz_3ft_10mmfocus_Plane_3_*_w0.npy"),
        ("3ft15",  "velocity_6khz_3ft_15mmfocus_Plane_3_*_w0.npy"),
        ("cirs",   "velocity_6khz_cirs_10mmfocus_10kPa*_w0.npy"),
        ("cirs15", "velocity_6khz_cirs_15mmfocus_10kPa*_w0.npy"),
    ]:
        files = sorted(pred_dir.glob(pattern))
        if not files:
            if src == "2ft":
                files = sorted(pred_dir.glob("velocity_6khz_2ft_10mmfocus_*.npy"))
            elif src == "3ft":
                files = sorted(pred_dir.glob("velocity_6khz_3ft_10mmfocus_*.npy"))
            elif src == "cirs":
                files = sorted(pred_dir.glob("velocity_6khz_cirs_10mmfocus_*.npy"))
            elif src == "cirs15":
                files = sorted(pred_dir.glob("velocity_6khz_cirs_15mmfocus_*.npy"))
        if files:
            reps[src] = str(files[0])

    # Goat heart: prefer v10b predictions; fallback to native velocity
    goat_pred_dir = Path("outputs/goat_heart_v10b")
    goat_preds = sorted(goat_pred_dir.glob("pred_velocity_goat_heart_Sample_1_Plane_1*.npy"))
    if goat_preds:
        reps["goat"] = str(goat_preds[0])
    else:
        goat_dir = Path("data/goat_heart_velocity/inference")
        goat_native = sorted(goat_dir.glob("velocity_goat_heart_Sample_1_Plane_1*.npy"))
        if goat_native:
            reps["goat"] = str(goat_native[0])

    return reps


def pick_comparison_samples() -> Dict[str, Dict[str, str]]:
    """Return sample names and paths for GT/pred comparison panels."""
    pred_dir = Path("outputs/6khz_predictions")
    goat_pred_dir = Path("outputs/goat_heart_v10b")
    samples: Dict[str, Dict[str, str]] = {}

    def _pick(key: str, pattern: str, title: str):
        files = sorted(pred_dir.glob(pattern))
        if not files:
            return
        name = files[0].stem.replace("velocity_6khz_", "")
        gt_path = pred_dir / f"gt_velocity_{name}.npy"
        if not gt_path.exists():
            print(f"  [{key}] GT missing: {gt_path}")
            return
        samples[key] = {
            "name": name,
            "pred_path": str(files[0]),
            "gt_path": str(gt_path),
            "title": title,
        }

    _pick("2ft", "velocity_6khz_2ft_10mmfocus_Plane_3_*_w0.npy",
          "2FT Phantom (10 mm focus)")
    _pick("2ft15", "velocity_6khz_2ft_15mmfocus_Plane_3_*_w0.npy",
          "2FT Phantom (15 mm focus)")
    _pick("3ft", "velocity_6khz_3ft_10mmfocus_Plane_3_*_w0.npy",
          "3FT Phantom (10 mm focus)")
    _pick("3ft15", "velocity_6khz_3ft_15mmfocus_Plane_3_*_w0.npy",
          "3FT Phantom (15 mm focus)")
    _pick("cirs", "velocity_6khz_cirs_10mmfocus_10kPa*_500Hz_*.npy",
          "CIRS 049 (10 mm focus, 10 kPa)")
    _pick("cirs15", "velocity_6khz_cirs_15mmfocus_10kPa*_500Hz_*.npy",
          "CIRS 049 (15 mm focus, 10 kPa)")

    goat_preds = sorted(goat_pred_dir.glob("pred_velocity_goat_heart_Sample_1_Plane_1_20kHz_w0.npy"))
    if goat_preds:
        name = goat_preds[0].stem.replace("pred_velocity_", "")
        gt_path = Path("data/goat_heart_velocity/inference") / f"velocity_{name}.npy"
        if gt_path.exists():
            samples["goat"] = {
                "name": name,
                "pred_path": str(goat_preds[0]),
                "gt_path": str(gt_path),
                "title": "Goat Heart (ex vivo)",
            }
        else:
            print(f"  [goat] GT missing: {gt_path}")

    return samples


def load_vel_and_scale(path: str):
    """Load a velocity .npy and apply vel_scale from metadata if available."""
    p = Path(path)
    vel = np.load(path).astype(np.float32)

    # Try to find matching metadata for scale
    meta_path = None
    if "6khz_predictions" in path:
        name = p.stem.replace("velocity_6khz_", "")
        meta_path = p.parent / f"metadata_{name}.json"
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        scale = float(meta.get("vel_scale_mms", 1.0))
        vel = vel * scale
    return vel


def make_velocity_map_panel(vel: np.ndarray,
                             title: str,
                             dt_s: float,
                             out_path: str,
                             n_frames: int = 8,
                             vmax_pct: float = 98.0):
    """
    Create a publication-quality multi-frame particle velocity panel.

    Shows n_frames equally spaced frames from the predicted velocity sequence
    as a row of 2-D spatial maps with a shared diverging colormap.

    Parameters
    ----------
    vel      : (T, H, W) velocity in mm/s
    title    : figure suptitle
    dt_s     : frame interval in seconds (used for time labels)
    out_path : output PNG path
    """
    T, H, W = vel.shape

    # Frame indices: pick n_frames from the first 2/3 of the sequence
    #   (the shear wave is active in the first portion of the acquisition)
    t_active = min(T, max(8, int(T * 0.80)))
    frame_idx = np.round(np.linspace(0, t_active - 1, n_frames)).astype(int)

    # Shared colormap range — from pooled active frames
    active = vel[frame_idx]
    vmax = float(np.percentile(np.abs(active), vmax_pct))
    vmax = max(vmax, 1e-3)

    # Axis vectors
    depth_mm   = np.arange(H) * AXIAL_MM_PER_PX
    lateral_mm = (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX

    fig, axes = plt.subplots(1, n_frames, figsize=(2.0 * n_frames, 3.4),
                              constrained_layout=True)
    if n_frames == 1:
        axes = [axes]

    for ax, fi in zip(axes, frame_idx):
        im = ax.imshow(
            vel[fi],
            aspect="auto",
            origin="upper",
            extent=[lateral_mm[0], lateral_mm[-1], depth_mm[-1], depth_mm[0]],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        t_ms = fi * dt_s * 1e3
        ax.set_title(f"$t = {t_ms:.2f}$\,ms", fontsize=8, pad=3)
        ax.set_xlabel("Lateral (mm)", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Depth (mm)", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6)

    # Shared colorbar on the right
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("Particle velocity\n(mm s$^{-1}$)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_velocity_comparison_panel(
    gt: np.ndarray,
    interp_linear: np.ndarray,
    interp_cubic: np.ndarray,
    pred: np.ndarray,
    title: str,
    dt_out_s: float,
    out_path: str,
    n_frames: int = 6,
    vmax_pct: float = 98.0,
    norm_vmax: float = 1.5,
):
    """Create a combined GT / interpolation / WavePINN comparison panel."""
    T, H, W = gt.shape
    t_active = min(T, max(8, int(T * 0.80)))
    frame_idx = np.round(np.linspace(0, t_active - 1, n_frames)).astype(int)

    depth_mm = np.arange(H) * AXIAL_MM_PER_PX
    lateral_mm = (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX

    gt_khz = max(1.0 / dt_out_s / 1000.0, 0.1)
    rows = [
        (f"GT {gt_khz:.0f} kHz", gt),
        ("Linear\nInterpolation", interp_linear),
        ("Cubic-Spline\nInterpolation", interp_cubic),
        ("WaveFormer\n(ours)", pred),
    ]

    row_scales = []
    for _, data in rows:
        row_active = data[frame_idx]
        row_vmax = float(np.percentile(np.abs(row_active), vmax_pct))
        row_scales.append(max(row_vmax, 1e-3))

    fig, axes = plt.subplots(
        len(rows), n_frames,
        figsize=(2.2 * n_frames, 2.2 * len(rows)),
        gridspec_kw={"wspace": 0.05, "hspace": 0.08},
    )

    for r_idx, (row_label, data) in enumerate(rows):
        row_scale = row_scales[r_idx]
        for c_idx, fi in enumerate(frame_idx):
            ax = axes[r_idx, c_idx]
            view = data[fi] / row_scale
            view = np.clip(view, -norm_vmax, norm_vmax)
            im = ax.imshow(
                view,
                aspect="auto",
                origin="upper",
                extent=[lateral_mm[0], lateral_mm[-1], depth_mm[-1], depth_mm[0]],
                cmap="seismic",
                vmin=-norm_vmax,
                vmax=norm_vmax,
                interpolation="bilinear",
            )
            t_ms = fi * dt_out_s * 1e3
            if r_idx == 0:
                ax.set_title(f"t = {t_ms:.2f} ms", fontsize=7, pad=2)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=9, rotation=90, labelpad=12)
                ax.yaxis.set_label_coords(-0.12, 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#333333")
            if r_idx == len(rows) - 1:
                for spine in ax.spines.values():
                    spine.set_linewidth(1.6)
                    spine.set_color("#1b8a3a")

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Norm. velocity\n(per-row 98th pct)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Shear Wave Velocity Reconstruction Comparison", fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, title, ha="center", va="center", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_velocity_comparison_panels():
    print("\n=== TASK 4: Generating comparison panels (GT + interp + WavePINN) ===")
    map_dir = OUTPUT_DIR / "velocity_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    samples = pick_comparison_samples()
    if not samples:
        print("  No comparison samples found — skipping")
        return

    full_meta = _load_metadata_map(Path("data/fullscale_velocity/metadata.json"))
    goat_meta = _load_metadata_map(Path("data/goat_heart_velocity/metadata.json"), split="inference")

    for key, info in samples.items():
        name = info["name"]
        gt = np.load(info["gt_path"]).astype(np.float32)
        pred = np.load(info["pred_path"]).astype(np.float32)

        dt_in_s = _get_dt_in_s(name, full_meta, goat_meta)
        dt_out_s = dt_in_s / UPSAMPLING_FACTOR
        interp_linear, interp_cubic = build_interp_baselines(gt, dt_in_s)

        out_path = map_dir / f"velocity_compare_{key}.png"
        make_velocity_comparison_panel(
            gt,
            interp_linear,
            interp_cubic,
            pred,
            title=info["title"],
            dt_out_s=dt_out_s,
            out_path=str(out_path),
        )


def generate_velocity_maps():
    print("\n=== TASK 3: Generating particle velocity map panels ===")
    map_dir = OUTPUT_DIR / "velocity_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    reps = pick_representative_samples()

    configs = {
        "2ft":   {"label": "2FT Phantom (~2 kPa, 10 mm focus)", "dt": DT_6KHZ,  "n_frames": 8},
        "2ft15": {"label": "2FT Phantom (~2 kPa, 15 mm focus)", "dt": DT_12KHZ, "n_frames": 8},
        "3ft":   {"label": "3FT Phantom (~3 kPa, 10 mm focus)", "dt": DT_6KHZ,  "n_frames": 8},
        "3ft15": {"label": "3FT Phantom (~3 kPa, 15 mm focus)", "dt": DT_6KHZ,  "n_frames": 8},
        "cirs":  {"label": "CIRS 049 Phantom (10 mm focus, 10 kPa inclusion)", "dt": DT_6KHZ, "n_frames": 8},
        "cirs15": {"label": "CIRS 049 Phantom (15 mm focus, 10 kPa inclusion)", "dt": DT_6KHZ, "n_frames": 8},
        "goat":  {"label": "Goat Heart (ex vivo)", "dt": DT_20KHZ, "n_frames": 8},
    }

    # Also generate a 4-row combined figure
    n_types = len([k for k in configs if k in reps])
    fig_all, axes_all = plt.subplots(
        n_types, 8,
        figsize=(16, 3.2 * n_types),
        constrained_layout=True,
    )
    if n_types == 1:
        axes_all = axes_all[np.newaxis, :]

    row = 0
    for src, cfg in configs.items():
        if src not in reps:
            print(f"  [{src}] no representative sample found — skipping")
            continue

        vel = load_vel_and_scale(reps[src])
        T, H, W = vel.shape
        dt_s = cfg["dt"]

        # Individual panel
        make_velocity_map_panel(
            vel,
            title=f"WavePINN v10b — {cfg['label']}",
            dt_s=dt_s,
            out_path=str(map_dir / f"velocity_map_{src}.png"),
            n_frames=cfg["n_frames"],
        )

        # Contribution to combined figure
        t_active = min(T, max(8, int(T * 0.80)))
        frame_idx = np.round(np.linspace(0, t_active - 1, 8)).astype(int)
        active = vel[frame_idx]
        vmax = float(np.percentile(np.abs(active), 98))
        vmax = max(vmax, 1e-3)

        depth_mm   = np.arange(H) * AXIAL_MM_PER_PX
        lateral_mm = (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX

        for col, fi in enumerate(frame_idx):
            ax = axes_all[row, col]
            im = ax.imshow(
                vel[fi],
                aspect="auto", origin="upper",
                extent=[lateral_mm[0], lateral_mm[-1], depth_mm[-1], depth_mm[0]],
                cmap="seismic", vmin=-vmax, vmax=vmax,
                interpolation="bilinear",
            )
            t_ms = fi * dt_s * 1e3
            ax.set_title(f"t={t_ms:.2f} ms", fontsize=6.5, pad=2)
            if col == 0:
                ax.set_ylabel(f"{cfg['label']}\n\nDepth (mm)", fontsize=7)
            else:
                ax.set_yticklabels([])
            if row == n_types - 1:
                ax.set_xlabel("Lateral (mm)", fontsize=6.5)
            ax.tick_params(labelsize=5.5)

        # Colorbar for this row
        fig_all.colorbar(im, ax=axes_all[row, :], fraction=0.01, pad=0.01
                         ).set_label("mm s$^{-1}$", fontsize=7)
        row += 1

    fig_all.suptitle(
        "WavePINN v10b — Predicted Particle Velocity Maps\n"
        "(Virtual 6 kHz sequence from 500 Hz input; ex-vivo goat heart at native 20 kHz)",
        fontsize=10, fontweight="bold",
    )
    combined_path = str(map_dir / "velocity_maps_combined.png")
    fig_all.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig_all)
    print(f"  Saved combined panel: {combined_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kv, interp, df_goat, fmt = compute_stats_10mm()
    out_path = generate_latex(kv, interp, df_goat, fmt)
    generate_velocity_maps()
    generate_velocity_comparison_panels()

    # Compile LaTeX if pdflatex is available
    import subprocess, shutil
    if shutil.which("pdflatex"):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(out_path)],
            capture_output=True, cwd=str(OUTPUT_DIR),
        )
        if r.returncode == 0:
            print(f"\nPDF compiled: {OUTPUT_DIR / 'latex_tables_10mm.pdf'}")
        else:
            print("\nLaTeX compilation had errors — check the .log file")
    print("\nAll done.")
