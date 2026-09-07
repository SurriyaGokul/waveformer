#!/usr/bin/env python3
"""
compare_model_maps.py
=====================
Publication-grade comparison of spatial elasticity (E) and viscosity (η) maps
produced by all available model weights on 500 Hz → 6 kHz SWE upsampling.

Models evaluated
----------------
  V2       — ComplexToRealUNet (production, 9.4 M params)
  V9-ep60  — WavePINNv9  epoch 60  checkpoint  (16.4 M params)
  V9-best  — WavePINNv9  best model (epoch 138)
  V8-best  — WavePINNv7  best model (epoch 11, poor training)
  V10-best  — WavePINNv10 best model (epoch 116, no 500Hz training)
  V10b-best — WavePINNv10 best model (epoch 166, 500Hz-inclusive training)

References
----------
  GT-5kHz  — native 5 kHz Loupas velocity (gold standard)
  GT-20kHz — native 20 kHz Loupas velocity (near-perfect temporal resolution)

Physics
-------
  cs(x,y)   : shear-wave speed via depth-windowed cross-correlation ToA
  E(x,y)    : Young's modulus  = 3 * ρ * cs²  (incompressible media)  [kPa]
  η(x,y)    : shear viscosity via Voigt model inversion               [Pa·s]
               Voigt:  G*(ω) = G_e + iωη
               Inversion:  2|G*|² − cs²ρ·|G*| − cs²ρ·G_e = 0
               η = sqrt(|G*|² − G_e²) / ω

Spatial calibration
-------------------
  Lateral : 0.1013 mm/px   (W = 160 px → 16.21 mm)
  Axial   : 0.0507 mm/px   (H = 116 px →  5.88 mm)

Usage
-----
  python scripts/compare_model_maps.py
  python scripts/compare_model_maps.py --output_dir outputs/model_comparison
  python scripts/compare_model_maps.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

# Ensure project root is on sys.path so that `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════════════
# Physical constants
# ══════════════════════════════════════════════════════════════════════════════
RHO              = 1000.0          # kg/m³
LATERAL_M_PER_PX = 0.1013e-3      # m/px
AXIAL_M_PER_PX   = 0.0507e-3      # m/px
LATERAL_MM_PER_PX = 0.1013        # mm/px
AXIAL_MM_PER_PX   = 0.0507        # mm/px
H_ROI, W_ROI     = 116, 160       # pixels
F_PUSH            = 400.0          # Hz — SWE ARF push frequency
OMEGA             = 2.0 * np.pi * F_PUSH

# Quasi-static shear moduli from CIRS phantom specs (E_static = 3 G_e,
# incompressible assumption)
G_ELASTIC = {
    "2ft":  2000.0 / 3.0,   # ≈  667 Pa  (E=2 kPa spec)
    "3ft":  3000.0 / 3.0,   # = 1000 Pa  (E=3 kPa spec)
}

# Frame intervals
DT_5KHZ  = 1.0 / 5000.0            # s  (0.200 ms) — GT 5 kHz
DT_20KHZ = 1.0 / 20000.0           # s  (0.050 ms) — GT 20 kHz
DT_500HZ = 1.0 / 500.0             # s  (2.000 ms) — GT 500 Hz (49-frame, 96ms GT vel file)

# For inference we load 5 CONSECUTIVE 500 Hz frames (step=1, dt=2 ms).
# Models upsample × 12, so output dt = 2ms / 12 = 0.1667 ms (virtual 6 kHz).
# This matches the existing V2 production prediction pipeline exactly.
DT_INPUT_500HZ = 1.0 / 500.0       # s  (2 ms)    — WavePINN dt_seconds argument
DT_MODEL       = DT_INPUT_500HZ / 12.0   # s  (≈ 0.167 ms = 6 kHz)  — model output frame dt
T_SPARSE       = 5                  # input sparse frames
UPSAMPLING_FACTOR = 12

# Matplotlib publication style
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.alpha":         0.35,
    "grid.linewidth":     0.5,
})

# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "V2": {
        "path":  "outputs/fullscale_velocity_v2/Final_Publication_Model_Weights.pt",
        "arch":  "ComplexToRealUNet",
        "label": "V2 (Production)",
        "color": "#2166AC",
        "dt":    DT_MODEL,
    },
    "V9-ep60": {
        "path":  "SWE_PINN_outputs_wavepinn_v9_checkpoint_epoch_60.pt",
        "arch":  "WavePINNv9",
        "label": "V9 epoch-60",
        "color": "#4DAC26",
        "dt":    DT_MODEL,
    },
    "V9-best": {
        "path":  "SWE_PINN_outputs_wavepinn_v9_best_model.pt",
        "arch":  "WavePINNv9",
        "label": "V9 best (ep138)",
        "color": "#74C476",
        "dt":    DT_MODEL,
    },
    "V8-best": {
        "path":  "SWE_PINN_outputs_wavepinn_v8_best_model.pt",
        "arch":  "WavePINNv7",
        "label": "V8 best (ep11)",
        "color": "#D6604D",
        "dt":    DT_MODEL,
    },
    "V10-best": {
        "path":  "SWE_PINN_outputs_wavepinn_v10_best_model.pt",
        "arch":  "WavePINNv10",
        "label": "V10 best (ep116)",
        "color": "#762A83",
        "dt":    DT_MODEL,
    },
    "V10b-best": {
        "path":  "SWE_PINN_outputs_wavepinn_v10b_best_model.pt",
        "arch":  "WavePINNv10",
        "label": "V10b (500Hz-trained, ep166)",
        "color": "#D01C8B",
        "dt":    DT_MODEL,
    },
}

# Reference conditions
REFERENCE_CONDITIONS = {
    "GT-5kHz":  {"dt": DT_5KHZ,  "label": "GT 5 kHz",  "color": "#F4A736", "ls": "-"},
    "GT-20kHz": {"dt": DT_20KHZ, "label": "GT 20 kHz", "color": "#B15928", "ls": "--"},
    "GT-500Hz": {"dt": DT_500HZ, "label": "GT 500 Hz", "color": "#999999", "ls": ":"},
}


# ══════════════════════════════════════════════════════════════════════════════
# SWE physics — 2D spatial cross-correlation ToA
# ══════════════════════════════════════════════════════════════════════════════

def _push_column(v_abs: np.ndarray, n_early: int = 4) -> int:
    """Identify push-focus lateral column from peak energy in first frames."""
    T, W = v_abs.shape
    early = v_abs[:min(n_early, T), :]
    return int(np.argmax(early.max(axis=0)))


def estimate_cs_scalar(velocity: np.ndarray,
                       dt_s: float,
                       dx_m: float = LATERAL_M_PER_PX,
                       depth_lo: float = 0.20,
                       depth_hi: float = 0.80,
                       excl_px: int = 10) -> Tuple[float, float, np.ndarray]:
    """
    Scalar cs estimate via depth-averaged cross-correlation ToA.

    Parameters
    ----------
    velocity : (T, H, W)  arbitrary amplitude units (only timing used)
    dt_s     : frame interval [s]
    dx_m     : lateral pixel pitch [m]

    Returns
    -------
    cs_med : median shear-wave speed [m/s]  (NaN on failure)
    cs_std : robust IQR-based std  [m/s]
    cs_col : (W,) per-column cs estimates
    """
    T, H, W = velocity.shape
    r0, r1  = int(H * depth_lo), int(H * depth_hi)

    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)  # (T, W)
    v_abs = np.abs(v_avg)
    amp   = v_abs.max(axis=0)
    thr   = np.percentile(amp[amp > 0], 15) if amp.max() > 0 else 0.0
    valid = amp > thr

    x_push   = _push_column(v_abs, n_early=max(3, T // 10))
    excl_lo  = max(0, x_push - excl_px)
    excl_hi  = min(W - 1, x_push + excl_px)
    valid[excl_lo: excl_hi + 1] = False

    cs_guess  = 4.0
    min_step  = max(3, int(np.ceil(1.0 * cs_guess * dt_s / dx_m)))
    STEP      = min(min_step, 12)

    cs_col = np.full(W, np.nan)
    cs_all = []

    for x in range(0, W - STEP):
        x2 = x + STEP
        if not (valid[x] and valid[x2]):
            continue
        a = v_avg[:, x];  b = v_avg[:, x2]
        ea = np.sqrt((a**2).sum());  eb = np.sqrt((b**2).sum())
        if ea < 1e-12 or eb < 1e-12:
            continue

        xc   = np.correlate(a / ea, b / eb, mode="full")
        lags = np.arange(-(T - 1), T, dtype=float)
        pi   = int(np.argmax(xc))

        if 0 < pi < len(xc) - 1:
            y0, y1, y2 = xc[pi - 1], xc[pi], xc[pi + 1]
            den        = y0 - 2 * y1 + y2
            frac       = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            lag_frames = lags[pi] + frac
        else:
            lag_frames = lags[pi]

        if abs(lag_frames) < 0.15:
            continue

        cs_local = abs(STEP * dx_m / (lag_frames * dt_s))
        if 0.3 < cs_local < 12.0:
            cs_all.append(cs_local)
            cs_col[x] = cs_local

    if len(cs_all) < 3:
        return np.nan, np.nan, cs_col

    cs_arr = np.array(cs_all)
    cs_med = float(np.median(cs_arr))
    iqr    = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    cs_std = iqr / 1.349
    if not (0.3 < cs_med < 12.0):
        return np.nan, np.nan, cs_col
    return cs_med, cs_std, cs_col


def estimate_cs_map_2d(velocity: np.ndarray,
                       dt_s: float,
                       dx_m: float = LATERAL_M_PER_PX,
                       dy_m: float = AXIAL_M_PER_PX,
                       depth_window_px: int = 20,
                       depth_step_px: int   = 4,
                       excl_px: int         = 10,
                       smooth_sigma: float  = 1.5,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D spatial cs map by sliding a depth window across the image
    and running cross-correlation ToA within each window.

    Parameters
    ----------
    velocity        : (T, H, W)  — velocity (any units; only timing matters)
    dt_s            : frame interval [s]
    dx_m            : lateral pixel pitch [m]
    dy_m            : axial pixel pitch [m]
    depth_window_px : axial averaging window width [px]
    depth_step_px   : step between consecutive windows [px]
    excl_px         : columns to exclude around ARF push column
    smooth_sigma    : Gaussian smoothing sigma applied to the final cs map

    Returns
    -------
    cs_map      : (n_depths, W)   shear-wave speed  [m/s], NaN where invalid
    y_mm        : (n_depths,)     axial centres      [mm]
    x_mm        : (W,)            lateral positions   [mm]
    """
    T, H, W = velocity.shape

    # Choose lateral STEP so that expected lag ≥ 0.5 frames
    cs_guess = 4.0  # m/s (conservative upper bound)
    min_step = max(3, int(np.ceil(0.5 * cs_guess * dt_s / dx_m)))
    STEP     = min(min_step, 14)

    # ARF push column — exclude ±excl_px to avoid near-field artefacts
    v_abs_full = np.abs(velocity).mean(axis=1)  # (T, W)
    x_push     = _push_column(v_abs_full, n_early=max(3, T // 10))
    excl_lo    = max(0, x_push - excl_px)
    excl_hi    = min(W - 1, x_push + excl_px)

    # Depth window centres
    y_starts    = list(range(0, H - depth_window_px + 1, depth_step_px))
    y_centers   = [r + depth_window_px // 2 for r in y_starts]
    n_depths    = len(y_centers)

    cs_map = np.full((n_depths, W), np.nan)

    for d_idx, y_c in enumerate(y_centers):
        r0  = max(0, y_c - depth_window_px // 2)
        r1  = min(H, r0 + depth_window_px)
        v_d = np.mean(velocity[:, r0:r1, :], axis=1)  # (T, W) depth-averaged
        v_abs = np.abs(v_d)

        amp   = v_abs.max(axis=0)
        thr   = np.percentile(amp[amp > 0], 10) if amp.max() > 0 else 0.0
        valid = amp > thr
        valid[excl_lo: excl_hi + 1] = False

        for x in range(0, W - STEP):
            x2 = x + STEP
            if not (valid[x] and valid[x2]):
                continue

            a = v_d[:, x];  b = v_d[:, x2]
            ea = np.sqrt((a**2).sum());  eb = np.sqrt((b**2).sum())
            if ea < 1e-12 or eb < 1e-12:
                continue

            xc   = np.correlate(a / ea, b / eb, mode="full")
            lags = np.arange(-(T - 1), T, dtype=float)
            pi   = int(np.argmax(xc))

            if 0 < pi < len(xc) - 1:
                y0, y1, y2 = xc[pi - 1], xc[pi], xc[pi + 1]
                den        = y0 - 2 * y1 + y2
                frac       = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                lag_f      = lags[pi] + frac
            else:
                lag_f = lags[pi]

            if abs(lag_f) < 0.15:
                continue

            cs_local = abs(STEP * dx_m / (lag_f * dt_s))
            if 0.3 < cs_local < 12.0:
                cs_map[d_idx, x] = cs_local

    # Smooth the map with a Gaussian filter (NaN-aware)
    if smooth_sigma > 0:
        cs_map = _gaussian_smooth_nan(cs_map, sigma=smooth_sigma)

    y_mm = np.array(y_centers, dtype=float) * dy_m * 1e3
    x_mm = np.arange(W, dtype=float) * dx_m * 1e3

    return cs_map, y_mm, x_mm


def _gaussian_smooth_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware 2D Gaussian smoothing."""
    mask  = np.isfinite(arr).astype(float)
    arr_f = np.where(mask, arr, 0.0)
    sm_a  = gaussian_filter(arr_f, sigma=sigma)
    sm_m  = gaussian_filter(mask,  sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(sm_m > 0.05, sm_a / sm_m, np.nan)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Voigt model inversion (vectorised)
# ══════════════════════════════════════════════════════════════════════════════

def elasticity_from_cs(cs_map: np.ndarray) -> np.ndarray:
    """E(x,y) = 3·ρ·cs²  [kPa].  Direct elastic approximation."""
    with np.errstate(invalid="ignore"):
        return 3.0 * RHO * cs_map**2 / 1e3   # kPa


def voigt_invert_map(cs_map: np.ndarray,
                     source: str,
                     f_push: float = F_PUSH,
                     rho: float    = RHO
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Voigt inversion:  given cs(x,y) and quasi-static G_e,
    recover η(x,y)  [Pa·s].

    Parameters
    ----------
    cs_map : (H_d, W)  shear-wave speed map [m/s]
    source : phantom label  ("2ft" | "3ft")
    f_push : SWE push frequency [Hz]

    Returns
    -------
    E_kpa  : (H_d, W)  dynamic Young's modulus  [kPa]  = 3·|G*|/1000
    eta    : (H_d, W)  shear viscosity           [Pa·s]
    """
    G_e   = G_ELASTIC.get(source, np.nan)
    omega = 2.0 * np.pi * f_push

    E_kpa = elasticity_from_cs(cs_map)   # simple SWE E map

    if not np.isfinite(G_e):
        # No Voigt inversion possible (e.g. CIRS phantom, unknown G_e)
        return E_kpa, np.full_like(cs_map, np.nan)

    cs2rho = cs_map**2 * rho              # |G*|-related term, Pa
    a      = 2.0
    b      = -cs2rho
    c      = -cs2rho * G_e
    disc   = b**2 - 4.0 * a * c

    G_star = np.full_like(cs_map, np.nan)
    ok     = np.isfinite(disc) & (disc >= 0) & np.isfinite(cs_map) & (cs_map > 0)
    G_star[ok] = (-b[ok] + np.sqrt(disc[ok])) / (2.0 * a)

    omega_eta_sq = G_star**2 - G_e**2
    eta          = np.full_like(cs_map, np.nan)
    ok2          = ok & np.isfinite(G_star) & (omega_eta_sq >= 0)
    eta[ok2]     = np.sqrt(omega_eta_sq[ok2]) / omega

    # Plausibility gate
    eta[(eta < 0.001) | (eta > 50.0)] = np.nan

    return E_kpa, eta


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_v2(ckpt: dict) -> torch.nn.Module:
    from src.architectures.velocity_unet import ComplexToRealUNet
    cfg = ckpt["config"]
    model = ComplexToRealUNet(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("base_channels", 32),
        encoder_depths=list(cfg.get("encoder_depths", [32, 64, 128, 256])),
        upsampling_factor=cfg.get("upsampling_factor", 12),
        use_motion_interpolation=cfg.get("use_motion_interpolation", True),
        complex_to_real_mode=cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn=cfg.get("use_naive_bn", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _load_wavepinn_v7(ckpt: dict) -> torch.nn.Module:
    from src.architectures.wavepinn_v7 import WavePINNv7
    cfg = ckpt["config"]["model"]
    model = WavePINNv7(
        in_channels=cfg.get("in_channels", 1),
        encoder_depths=cfg.get("encoder_depths", [32, 64, 128, 256]),
        upsampling_factor=cfg.get("upsampling_factor", 12),
        complex_to_real_mode=cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn=cfg.get("use_naive_bn", True),
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        n_self_layers=cfg.get("n_self_layers", 2),
        n_freq=cfg.get("n_freq", 32),
        n_refine_iters=cfg.get("n_refine_iters", 2),
        dropout=cfg.get("dropout", 0.1),
        use_attention_skips=cfg.get("use_attention_skips", True),
        skip_attn_d_model=cfg.get("skip_attn_d_model", 128),
        skip_attn_heads=cfg.get("skip_attn_heads", 4),
        skip_attn_n_freq=cfg.get("skip_attn_n_freq", 32),
        skip_attn_dropout=cfg.get("skip_attn_dropout", 0.1),
        cs_head_mid_channels=cfg.get("cs_head_mid_channels", 128),
        cs_min=cfg.get("cs_min", 0.3),
        cs_max=cfg.get("cs_max", 8.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _load_wavepinn_v9(ckpt: dict) -> torch.nn.Module:
    from src.architectures.wavepinn_v9 import WavePINNv9
    cfg = ckpt["config"]["model"]
    model = WavePINNv9(
        in_channels=cfg.get("in_channels", 1),
        encoder_depths=cfg.get("encoder_depths", [32, 64, 128, 256]),
        upsampling_factor=cfg.get("upsampling_factor", 12),
        complex_to_real_mode=cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn=cfg.get("use_naive_bn", True),
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        n_self_layers=cfg.get("n_self_layers", 2),
        n_freq=cfg.get("n_freq", 32),
        n_refine_iters=cfg.get("n_refine_iters", 2),
        dropout=cfg.get("dropout", 0.1),
        use_attention_skips=cfg.get("use_attention_skips", True),
        skip_attn_d_model=cfg.get("skip_attn_d_model", 128),
        skip_attn_heads=cfg.get("skip_attn_heads", 4),
        skip_attn_n_freq=cfg.get("skip_attn_n_freq", 32),
        skip_attn_dropout=cfg.get("skip_attn_dropout", 0.1),
        cs_min=cfg.get("cs_min", 0.3),
        cs_max=cfg.get("cs_max", 8.0),
        cs_init=cfg.get("cs_init", 2.54),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _load_wavepinn_v10(ckpt: dict) -> torch.nn.Module:
    from src.architectures.wavepinn_v10 import WavePINNv10
    cfg = ckpt["config"]["model"]
    model = WavePINNv10(
        in_channels=cfg.get("in_channels", 1),
        encoder_depths=cfg.get("encoder_depths", [32, 64, 128, 256]),
        upsampling_factor=cfg.get("upsampling_factor", 12),
        complex_to_real_mode=cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn=cfg.get("use_naive_bn", True),
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        n_self_layers=cfg.get("n_self_layers", 2),
        n_freq=cfg.get("n_freq", 32),
        n_refine_iters=cfg.get("n_refine_iters", 2),
        dropout=cfg.get("dropout", 0.1),
        use_attention_skips=cfg.get("use_attention_skips", True),
        skip_attn_d_model=cfg.get("skip_attn_d_model", 128),
        skip_attn_heads=cfg.get("skip_attn_heads", 4),
        skip_attn_n_freq=cfg.get("skip_attn_n_freq", 32),
        skip_attn_dropout=cfg.get("skip_attn_dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


_ARCH_LOADER = {
    "ComplexToRealUNet": _load_v2,
    "WavePINNv7":        _load_wavepinn_v7,
    "WavePINNv9":        _load_wavepinn_v9,
    "WavePINNv10":       _load_wavepinn_v10,
}


def load_model(model_key: str, device: torch.device) -> Optional[torch.nn.Module]:
    """Load and return a model in eval mode. Returns None if weight file missing."""
    info = MODEL_REGISTRY[model_key]
    path = Path(info["path"])
    if not path.exists():
        print(f"  [skip] {model_key}: weight file not found at {path}")
        return None
    print(f"  Loading {model_key} ({info['arch']}) from {path.name} …", end=" ", flush=True)
    ckpt  = torch.load(str(path), map_location="cpu")
    loader = _ARCH_LOADER.get(info["arch"])
    if loader is None:
        print(f"UNKNOWN ARCH: {info['arch']}")
        return None
    model = loader(ckpt)
    model.to(device).eval()
    print(f"OK  (epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('best_val_loss', float('nan')):.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ══════════════════════════════════════════════════════════════════════════════

def _load_consecutive_iq(sample: dict, start_frame: int = 0) -> np.ndarray:
    """
    Load T_SPARSE consecutive 500 Hz frames from original raw IQ.

    Replicates the predict_6khz_from_500hz_v4.py preprocessing:
      1. Load original_path
      2. Crop ROI and time window
      3. Subtract temporal mean (clutter/DC filter)
      4. Extract T_SPARSE consecutive frames from start_frame

    Falls back to the preprocessed sparse_*.npy (step=12) if original missing.
    """
    orig_path = sample.get("original_path")
    if orig_path and Path(orig_path).exists():
        raw = np.load(orig_path)    # shape: (H, W, T) or (T, H, W)
        roi = sample["roi"]
        win = sample["window"]

        # Detect layout: if last axis is smallest → (H, W, T) format
        # otherwise → (T, H, W) format
        if raw.shape[2] < min(raw.shape[0], raw.shape[1]):
            # (H_full, W_full, T_full) layout
            iq_win = raw[
                roi["min_row"]: roi["max_row"] + 1,
                roi["min_col"]: roi["max_col"] + 1,
                win[0]: win[1],
            ].astype(np.complex64)
            iq_win -= iq_win.mean(axis=2, keepdims=True)   # clutter filter
            T_win = iq_win.shape[2]
            sf = min(start_frame, max(0, T_win - T_SPARSE))
            frames = np.stack(
                [iq_win[:, :, sf + i] for i in range(min(T_SPARSE, T_win - sf))],
                axis=0,
            )
        else:
            # (T_full, H_full, W_full) layout
            iq_win = raw[
                win[0]: win[1],
                roi["min_row"]: roi["max_row"] + 1,
                roi["min_col"]: roi["max_col"] + 1,
            ].astype(np.complex64)
            iq_win -= iq_win.mean(axis=0, keepdims=True)   # clutter filter
            T_win = iq_win.shape[0]
            sf = min(start_frame, max(0, T_win - T_SPARSE))
            frames = iq_win[sf: sf + T_SPARSE]

        return frames.astype(np.complex64)

    # Fallback: preprocessed step-12 sparse file (dt=24ms — out-of-distribution)
    return np.load(sample["sparse_path"]).astype(np.complex64)


def run_inference(model_key: str,
                  model: torch.nn.Module,
                  sample: dict,
                  device: torch.device) -> np.ndarray:
    """
    Run model inference on a 500 Hz sample.

    Loads 5 CONSECUTIVE 500 Hz IQ frames (step=1, dt=2 ms) from original
    raw IQ data — identical to the production V2 prediction pipeline.

    Parameters
    ----------
    model_key : MODEL_REGISTRY key
    model     : loaded model in eval mode
    sample    : dict with original_path, roi, window, sparse_path

    Returns
    -------
    velocity  : (T_dense=49, H, W) float32 — normalised output in [-1, 1]
    """
    arch = MODEL_REGISTRY[model_key]["arch"]

    iq_consec = _load_consecutive_iq(sample)   # (T_SPARSE, H, W) complex64

    # Per-sample IQ normalisation (99th percentile of magnitude)
    iq_scale = float(np.percentile(np.abs(iq_consec), 99))
    iq_scale  = max(iq_scale, 1.0)
    iq_n      = iq_consec / iq_scale            # (T, H, W) complex64

    # Stack real + imaginary → (2, T, H, W) float32
    x = np.stack([iq_n.real.astype(np.float32),
                  iq_n.imag.astype(np.float32)], axis=0)   # (2, T, H, W)

    # Pad H, W to multiples of 16 (UNet stride requirement)
    _, H, W = x.shape[1], x.shape[2], x.shape[3]
    H_pad = ((H + 15) // 16) * 16
    W_pad = ((W + 15) // 16) * 16
    if H_pad != H or W_pad != W:
        xp = np.zeros((2, x.shape[1], H_pad, W_pad), dtype=np.float32)
        xp[:, :, :H, :W] = x
        x = xp

    x_t = torch.from_numpy(x[np.newaxis]).to(device)   # (1, 2, T, H_pad, W_pad)

    with torch.no_grad():
        if arch == "ComplexToRealUNet":
            # V2: forward(x) — dt is implicit in model's ConvLSTM
            out = model(x_t)
        elif arch in ("WavePINNv7", "WavePINNv9"):
            # V7 returns (velocity, cs_map); V9 returns (velocity, cs_scalar)
            out, _ = model(x_t, dt_seconds=DT_INPUT_500HZ)
        else:
            # WavePINNv10 — velocity only
            out = model(x_t, dt_seconds=DT_INPUT_500HZ)

    vel = out[0, 0].cpu().numpy()       # (T_dense, H_pad, W_pad)
    return vel[:, :H, :W]               # crop to original ROI size


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _group_key(sample_meta: dict) -> str:
    """Construct group key from metadata: source_focus_region."""
    return f"{sample_meta['source']}_{sample_meta['focus']}_{sample_meta['region']}"


def load_500hz_val_samples(data_dir: str = "data/fullscale_velocity") -> List[dict]:
    """
    Load all 500 Hz val samples with metadata.

    Returns list of dicts with keys:
        name, group, source, sparse_path, velocity_path, mask_path, dt_s
    """
    meta_path = Path(data_dir) / "metadata.json"
    with open(str(meta_path)) as f:
        meta = json.load(f)

    val_dir = Path(data_dir) / "val"
    samples = []
    for s in meta["val"]:
        if abs(s.get("prf_hz", 0) - 500.0) > 1.0:
            continue
        name  = s["name"]
        sp    = val_dir / f"sparse_{name}.npy"
        vel_p = val_dir / f"velocity_{name}.npy"
        mag_p = val_dir / f"magnitude_{name}.npy"
        if not sp.exists() or not vel_p.exists():
            continue
        samples.append({
            "name":          name,
            "group":         _group_key(s),
            "source":        s["source"],
            "focus":         s["focus"],
            "region":        s["region"],
            "sparse_path":   str(sp),
            "velocity_path": str(vel_p),
            "mask_path":     str(mag_p) if mag_p.exists() else None,
            "dt_s":          1.0 / s["prf_hz"],
            "tissue_props":  s.get("tissue_properties", {}),
            # For loading consecutive raw IQ frames (production pipeline)
            "original_path": s.get("original_path"),
            "roi":           s.get("roi", {}),
            "window":        s.get("window", [0, 50]),
        })
    return samples


def load_reference_velocities(data_dir: str = "data/fullscale_velocity"
                              ) -> Dict[str, Dict[str, dict]]:
    """
    Load all available 5 kHz and 20 kHz velocity files from train + val.
    Returns {freq_key: {group_key: {"velocity": ndarray, "dt_s": float}}}
    """
    refs: Dict[str, Dict[str, dict]] = {"5kHz": {}, "20kHz": {}}
    meta_path = Path(data_dir) / "metadata.json"
    with open(str(meta_path)) as f:
        meta = json.load(f)

    for split in ("train", "val"):
        split_dir = Path(data_dir) / split
        for s in meta.get(split, []):
            prf = s.get("prf_hz", 0)
            name = s.get("name", "")
            vel_p = split_dir / f"velocity_{name}.npy"
            if not vel_p.exists():
                continue
            if abs(prf - 5000) < 1:
                fkey = "5kHz"
            elif abs(prf - 20000) < 1:
                fkey = "20kHz"
            else:
                continue
            gk = _group_key(s)
            if gk not in refs[fkey]:  # keep first encounter (val preferred, but iteration order is train/val)
                refs[fkey][gk] = {
                    "velocity":  np.load(str(vel_p)),  # (T, H, W) mm/s
                    "dt_s":      1.0 / prf,
                    "name":      name,
                    "source":    s["source"],
                }
    return refs


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_sample(sample: dict,
                   models: Dict[str, Optional[torch.nn.Module]],
                   refs: Dict[str, Dict[str, dict]],
                   device: torch.device,
                   ) -> dict:
    """
    Run all models on one 500 Hz sample.  Return cs / E / η scalars and maps.

    Returns dict with keys:
        name, group, source,
        gt_500hz: {cs_med, cs_std, cs_col, E_med, eta_med}
        refs:     {"5kHz": {...}, "20kHz": {...}}   — if available
        models:   {model_key: {cs_med, cs_std, cs_col, E_med, eta_med, vel_map}}
    """
    name   = sample["name"]
    group  = sample["group"]
    source = sample["source"]

    # Load 500 Hz GT velocity (raw mm/s)
    vel_500_raw = np.load(sample["velocity_path"])   # (T, H, W) float32

    result = {
        "name":   name,
        "group":  group,
        "source": source,
        "gt_500hz": {},
        "refs":   {},
        "models": {},
    }

    # ── GT 500 Hz ────────────────────────────────────────────────────────────
    cs_m, cs_s, cs_col = estimate_cs_scalar(vel_500_raw, DT_500HZ)
    E_med   = float(3.0 * RHO * cs_m**2 / 1e3) if np.isfinite(cs_m) else np.nan
    eta_med = float(_voigt_scalar(cs_m, source))
    result["gt_500hz"] = {"cs_med": cs_m, "cs_std": cs_s,
                          "E_med": E_med, "eta_med": eta_med}

    # ── Reference GTs ─────────────────────────────────────────────────────────
    for freq_tag in ("5kHz", "20kHz"):
        if group in refs[freq_tag]:
            ref = refs[freq_tag][group]
            vel_ref = ref["velocity"]        # raw mm/s
            dt_ref  = ref["dt_s"]
            cs_m_r, cs_s_r, cs_col_r = estimate_cs_scalar(vel_ref, dt_ref)
            E_med_r   = float(3.0 * RHO * cs_m_r**2 / 1e3) if np.isfinite(cs_m_r) else np.nan
            eta_med_r = float(_voigt_scalar(cs_m_r, source))
            result["refs"][freq_tag] = {
                "cs_med": cs_m_r, "cs_std": cs_s_r,
                "E_med": E_med_r, "eta_med": eta_med_r,
                "velocity": vel_ref, "dt_s": dt_ref,
            }

    # ── Model predictions ─────────────────────────────────────────────────────
    for mkey, model in models.items():
        if model is None:
            result["models"][mkey] = None
            continue
        try:
            vel_pred = run_inference(mkey, model, sample, device)  # (T_dense, H, W)
        except Exception as e:
            print(f"    [warn] {mkey} inference failed on {name}: {e}")
            result["models"][mkey] = None
            continue

        dt_pred = MODEL_REGISTRY[mkey]["dt"]
        cs_m_p, cs_s_p, cs_col_p = estimate_cs_scalar(vel_pred, dt_pred)
        E_med_p   = float(3.0 * RHO * cs_m_p**2 / 1e3) if np.isfinite(cs_m_p) else np.nan
        eta_med_p = float(_voigt_scalar(cs_m_p, source))
        result["models"][mkey] = {
            "cs_med":   cs_m_p,
            "cs_std":   cs_s_p,
            "E_med":    E_med_p,
            "eta_med":  eta_med_p,
            "velocity": vel_pred,
            "dt_s":     dt_pred,
        }

    return result


def _voigt_scalar(cs: float, source: str) -> float:
    """Scalar Voigt η inversion."""
    if not np.isfinite(cs) or cs <= 0:
        return np.nan
    G_e = G_ELASTIC.get(source, np.nan)
    if not np.isfinite(G_e):
        return np.nan
    cs2rho = cs**2 * RHO
    a, b, c = 2.0, -cs2rho, -cs2rho * G_e
    disc = b**2 - 4 * a * c
    if disc < 0:
        return np.nan
    y = (-b + np.sqrt(disc)) / (2 * a)
    omega_eta_sq = y**2 - G_e**2
    if omega_eta_sq < 0:
        return np.nan
    eta = np.sqrt(omega_eta_sq) / OMEGA
    if not (0.001 <= eta <= 50.0):
        return np.nan
    return float(eta)


# ══════════════════════════════════════════════════════════════════════════════
# Publication figures
# ══════════════════════════════════════════════════════════════════════════════

def _pick_representative_samples(results: List[dict]) -> List[dict]:
    """One representative per phantom type (2ft, 3ft, cirs) with ≥1 ref."""
    seen = {}
    for r in results:
        src = r["source"]
        if src not in seen and r["refs"]:
            seen[src] = r
    return list(seen.values())


def _cs_map_for_display(velocity: np.ndarray, dt_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D cs map for display."""
    return estimate_cs_map_2d(velocity, dt_s,
                              depth_window_px=16, depth_step_px=4,
                              excl_px=10, smooth_sigma=1.0)


def fig_spatial_maps(representative_results: List[dict],
                     out: Path) -> None:
    """
    Figure 1: Spatial E(x,y) maps for each representative phantom.

    Layout: rows = phantoms, cols = [GT-5kHz | GT-20kHz | V2 | V9-ep60 | V10-best | V10b-best]
    with shared colorscale per row.
    """
    model_cols = ["V2", "V9-ep60", "V9-best", "V10-best", "V10b-best"]
    ref_cols   = ["5kHz", "20kHz"]
    all_cols   = [(f"ref_{t}", f"GT {t}", REFERENCE_CONDITIONS[f"GT-{t}"]["color"])
                  for t in ref_cols] + \
                 [(f"model_{k}", MODEL_REGISTRY[k]["label"], MODEL_REGISTRY[k]["color"])
                  for k in model_cols]
    n_cols = len(all_cols)

    for rep in representative_results:
        name   = rep["name"]
        source = rep["source"]
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(3.2 * n_cols, 4.0),
                                 squeeze=False)
        axes = axes[0]

        E_all = []

        # Collect E maps
        col_maps = []
        for col_key, col_label, col_color in all_cols:
            if col_key.startswith("ref_"):
                freq_tag = col_key[4:]   # "5kHz" or "20kHz"
                ref_data = rep["refs"].get(freq_tag)
                if ref_data is None:
                    col_maps.append((None, None, None, col_label, col_color))
                    continue
                vel = ref_data["velocity"]
                dt  = ref_data["dt_s"]
            else:
                mkey     = col_key[6:]   # model key
                mdata    = rep["models"].get(mkey)
                if mdata is None:
                    col_maps.append((None, None, None, col_label, col_color))
                    continue
                vel = mdata["velocity"]
                dt  = mdata["dt_s"]

            cs_m, y_mm, x_mm = _cs_map_for_display(vel, dt)
            E_map = elasticity_from_cs(cs_m)
            E_all.append(E_map[np.isfinite(E_map)])
            col_maps.append((E_map, y_mm, x_mm, col_label, col_color))

        if not E_all:
            plt.close(fig)
            continue

        E_cat  = np.concatenate(E_all)
        vmin   = float(np.nanpercentile(E_cat, 2))
        vmax   = float(np.nanpercentile(E_cat, 98))

        # Plot each column
        for ax, (E_map, y_mm, x_mm, col_label, col_color) in zip(axes, col_maps):
            if E_map is None:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="white", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                # Create display array with NaN → facecolor
                E_disp = np.where(np.isfinite(E_map), E_map, np.nan)
                im = ax.imshow(E_disp, origin="upper", aspect="auto",
                               extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
                               cmap="viridis", vmin=vmin, vmax=vmax,
                               interpolation="bicubic")
                ax.set_xlabel("Lateral (mm)", fontsize=8)
                ax.set_ylabel("Depth (mm)", fontsize=8)
                ax.tick_params(labelsize=7)

            ax.set_title(col_label, fontsize=9, fontweight="bold",
                         color=col_color, pad=3)

            # Colour spine border
            for spine in ax.spines.values():
                spine.set_edgecolor(col_color)
                spine.set_linewidth(1.8)
                spine.set_visible(True)

        # Add shared colorbar on last axis
        last_E = next(E for E, *_ in col_maps if E is not None)
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[-1], pad=0.03, fraction=0.05)
        cbar.set_label("E  (kPa)", fontsize=8, labelpad=2)
        cbar.ax.tick_params(labelsize=7)

        phantom_label = {"2ft": "2FT  (E≈2 kPa)", "3ft": "3FT  (E≈3 kPa)",
                         "cirs": "CIRS  (elastic phantom)"}.get(source, source)
        fig.suptitle(f"Spatial Young's Modulus Map  |  {phantom_label}\n"
                     f"E(x,y) = 3·ρ·cs²  [kPa]",
                     fontsize=11, fontweight="bold", y=1.02)

        fp = out / f"fig_E_map_{source}_{rep['name'][:30]}.png"
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fp.name}")


def fig_eta_maps(representative_results: List[dict], out: Path) -> None:
    """
    Figure 2: Spatial η(x,y) maps.  Only for 2ft / 3ft phantoms (known G_e).
    """
    model_cols = ["V2", "V9-ep60", "V9-best", "V10-best", "V10b-best"]
    ref_cols   = ["5kHz", "20kHz"]

    for rep in representative_results:
        source = rep["source"]
        if source not in G_ELASTIC:
            continue   # no Voigt inversion for CIRS (unknown per-inclusion G_e)

        all_cols = [(f"ref_{t}", f"GT {t}", REFERENCE_CONDITIONS[f"GT-{t}"]["color"])
                    for t in ref_cols] + \
                   [(f"model_{k}", MODEL_REGISTRY[k]["label"], MODEL_REGISTRY[k]["color"])
                    for k in model_cols]
        n_cols = len(all_cols)

        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(3.2 * n_cols, 4.0),
                                 squeeze=False)
        axes = axes[0]
        eta_all = []

        col_maps = []
        for col_key, col_label, col_color in all_cols:
            if col_key.startswith("ref_"):
                freq_tag = col_key[4:]
                ref_data = rep["refs"].get(freq_tag)
                if ref_data is None:
                    col_maps.append((None, None, None, col_label, col_color))
                    continue
                vel = ref_data["velocity"]
                dt  = ref_data["dt_s"]
            else:
                mkey  = col_key[6:]
                mdata = rep["models"].get(mkey)
                if mdata is None:
                    col_maps.append((None, None, None, col_label, col_color))
                    continue
                vel = mdata["velocity"]
                dt  = mdata["dt_s"]

            cs_m, y_mm, x_mm = _cs_map_for_display(vel, dt)
            _, eta_map = voigt_invert_map(cs_m, source)
            eta_all.append(eta_map[np.isfinite(eta_map)])
            col_maps.append((eta_map, y_mm, x_mm, col_label, col_color))

        if not eta_all:
            plt.close(fig)
            continue

        eta_cat = np.concatenate(eta_all)
        vmin    = max(0.0, float(np.nanpercentile(eta_cat, 2)))
        vmax    = float(np.nanpercentile(eta_cat, 98))

        for ax, (eta_map, y_mm, x_mm, col_label, col_color) in zip(axes, col_maps):
            if eta_map is None:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="white", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                im = ax.imshow(eta_map, origin="upper", aspect="auto",
                               extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
                               cmap="viridis", vmin=vmin, vmax=vmax,
                               interpolation="bicubic")
                ax.set_xlabel("Lateral (mm)", fontsize=8)
                ax.set_ylabel("Depth (mm)", fontsize=8)
                ax.tick_params(labelsize=7)

            ax.set_title(col_label, fontsize=9, fontweight="bold",
                         color=col_color, pad=3)
            for spine in ax.spines.values():
                spine.set_edgecolor(col_color)
                spine.set_linewidth(1.8)
                spine.set_visible(True)

        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[-1], pad=0.03, fraction=0.05)
        cbar.set_label("η  (Pa·s)", fontsize=8, labelpad=2)
        cbar.ax.tick_params(labelsize=7)

        phantom_label = {"2ft": "2FT  (E≈2 kPa)", "3ft": "3FT  (E≈3 kPa)"}.get(source, source)
        fig.suptitle(f"Spatial Viscosity Map  |  {phantom_label}\n"
                     f"Voigt inversion  [G*(ω) = G_e + iωη]  |  f_push = {F_PUSH:.0f} Hz",
                     fontsize=11, fontweight="bold", y=1.02)

        fp = out / f"fig_eta_map_{source}_{rep['name'][:30]}.png"
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fp.name}")


def fig_metrics_summary(df: pd.DataFrame, out: Path) -> None:
    """
    Figure 3: Bar charts of cs MAE, E MAE, η MAE for each model vs GT-5kHz.
    """
    model_keys = [k for k in MODEL_REGISTRY if k in df["model"].unique()]
    if not model_keys:
        return

    metrics = [
        ("cs_mae",  "cs MAE  (m/s)",  "cs_bias", "cs Bias (m/s)"),
        ("E_mae",   "E MAE  (kPa)",   "E_bias",  "E Bias  (kPa)"),
    ]

    # Only include η if at least some values are finite
    if df["eta_mae"].notna().any():
        metrics.append(("eta_mae", "η MAE  (Pa·s)", "eta_bias", "η Bias  (Pa·s)"))

    n_rows = 2  # MAE row and Bias row
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_rows, n_metrics,
                             figsize=(4.0 * n_metrics, 3.5 * n_rows),
                             squeeze=False)

    for c_idx, (mae_col, mae_label, bias_col, bias_label) in enumerate(metrics):
        ax_mae  = axes[0, c_idx]
        ax_bias = axes[1, c_idx]

        sub = df[df["model"].isin(model_keys)].dropna(subset=[mae_col])
        if sub.empty:
            continue

        vals_mae  = []
        vals_bias = []
        colors    = []
        labels    = []
        for mkey in model_keys:
            mdf = sub[sub["model"] == mkey]
            if mdf.empty:
                continue
            vals_mae.append(float(mdf[mae_col].mean()))
            vals_bias.append(float(mdf[bias_col].mean()) if bias_col in mdf else np.nan)
            colors.append(MODEL_REGISTRY[mkey]["color"])
            labels.append(MODEL_REGISTRY[mkey]["label"])

        if not vals_mae:
            continue

        x = np.arange(len(labels))
        w = 0.55

        bars = ax_mae.bar(x, vals_mae, width=w, color=colors, edgecolor="black",
                          linewidth=0.6, alpha=0.88)
        ax_mae.set_ylabel(mae_label, fontsize=9)
        ax_mae.set_xticks(x)
        ax_mae.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax_mae.set_title(mae_label, fontsize=9, pad=4)
        ax_mae.yaxis.grid(True, alpha=0.4); ax_mae.set_axisbelow(True)
        for bar, v in zip(bars, vals_mae):
            ax_mae.text(bar.get_x() + bar.get_width() / 2, v * 1.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        bars_b = ax_bias.bar(x, vals_bias, width=w, color=colors,
                             edgecolor="black", linewidth=0.6, alpha=0.88)
        ax_bias.axhline(0, color="black", lw=0.8, ls="--")
        ax_bias.set_ylabel(bias_label, fontsize=9)
        ax_bias.set_xticks(x)
        ax_bias.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax_bias.yaxis.grid(True, alpha=0.4); ax_bias.set_axisbelow(True)
        for bar, v in zip(bars_b, vals_bias):
            if np.isfinite(v):
                ax_bias.text(bar.get_x() + bar.get_width() / 2,
                             v + (0.005 if v >= 0 else -0.01),
                             f"{v:+.3f}", ha="center",
                             va="bottom" if v >= 0 else "top", fontsize=7)

    fig.suptitle("Model Accuracy vs GT 5 kHz  |  500 Hz → 6 kHz SWE Upsampling",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    fp = out / "fig_metrics_summary.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fp.name}")


def fig_cs_scatter(all_results: List[dict], out: Path) -> None:
    """
    Figure 4: Scatter plot of model cs vs GT-5kHz cs.
    One sub-plot per model, with identity line and Pearson r.
    """
    model_keys = list(MODEL_REGISTRY.keys())
    n = len(model_keys)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.8), squeeze=False)
    axes = axes[0]

    for ax, mkey in zip(axes, model_keys):
        xs, ys = [], []
        for r in all_results:
            ref5 = r["refs"].get("5kHz")
            mdat = r["models"].get(mkey)
            if ref5 is None or mdat is None:
                continue
            cs_ref = ref5["cs_med"]
            cs_mod = mdat["cs_med"]
            if np.isfinite(cs_ref) and np.isfinite(cs_mod):
                xs.append(cs_ref)
                ys.append(cs_mod)
        xs, ys = np.array(xs), np.array(ys)
        color  = MODEL_REGISTRY[mkey]["color"]
        label  = MODEL_REGISTRY[mkey]["label"]

        if len(xs) >= 2:
            r_val, _ = pearsonr(xs, ys)
            mae      = float(np.mean(np.abs(ys - xs)))
            bias     = float(np.mean(ys - xs))
            ax.scatter(xs, ys, c=color, s=45, alpha=0.85, edgecolors="none",
                       zorder=3)
            lo = min(xs.min(), ys.min()) * 0.9
            hi = max(xs.max(), ys.max()) * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="Identity")
            ax.text(0.05, 0.92, f"r = {r_val:.3f}\nMAE = {mae:.3f} m/s\nbias = {bias:+.3f} m/s",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_xlabel("cs  GT 5 kHz  (m/s)", fontsize=9)
        ax.set_ylabel("cs  Model  (m/s)", fontsize=9)
        ax.set_title(label, fontsize=9, fontweight="bold", color=color, pad=4)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)

    fig.suptitle("Shear-Wave Speed Comparison: Model vs GT 5 kHz",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    fp = out / "fig_cs_scatter.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fp.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics table
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_df(all_results: List[dict]) -> pd.DataFrame:
    """
    Build a tidy DataFrame with one row per (sample, model) pair,
    containing error metrics vs GT-5kHz reference.
    """
    rows = []
    for r in all_results:
        ref5 = r["refs"].get("5kHz")
        for mkey in MODEL_REGISTRY:
            mdat = r["models"].get(mkey)
            if mdat is None:
                continue
            cs_p  = mdat["cs_med"]
            E_p   = mdat["E_med"]
            eta_p = mdat["eta_med"]

            if ref5 is not None:
                cs_r  = ref5["cs_med"]
                E_r   = ref5["E_med"]
                eta_r = ref5["eta_med"]
            else:
                cs_r = E_r = eta_r = np.nan

            rows.append({
                "model":    mkey,
                "label":    MODEL_REGISTRY[mkey]["label"],
                "name":     r["name"],
                "group":    r["group"],
                "source":   r["source"],
                "cs_pred":  cs_p,
                "cs_ref5":  cs_r,
                "cs_mae":   abs(cs_p - cs_r) if np.isfinite(cs_p) and np.isfinite(cs_r) else np.nan,
                "cs_bias":  (cs_p - cs_r)    if np.isfinite(cs_p) and np.isfinite(cs_r) else np.nan,
                "E_pred":   E_p,
                "E_ref5":   E_r,
                "E_mae":    abs(E_p - E_r)   if np.isfinite(E_p)  and np.isfinite(E_r)  else np.nan,
                "E_bias":   (E_p - E_r)      if np.isfinite(E_p)  and np.isfinite(E_r)  else np.nan,
                "eta_pred": eta_p,
                "eta_ref5": eta_r,
                "eta_mae":  abs(eta_p - eta_r) if np.isfinite(eta_p) and np.isfinite(eta_r) else np.nan,
                "eta_bias": (eta_p - eta_r)    if np.isfinite(eta_p) and np.isfinite(eta_r) else np.nan,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, all_results: List[dict], out: Path) -> None:
    """Print and save the summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  MODEL COMPARISON REPORT — 500 Hz → 6 kHz SWE Upsampling")
    lines.append("  Spatial E(x,y) and η(x,y) accuracy vs GT 5 kHz reference")
    lines.append("=" * 80)
    lines.append("")
    lines.append("EXPERIMENTAL SETUP")
    lines.append("-" * 60)
    lines.append(f"  Input:     500 Hz IQ — 5 consecutive frames (step=1, dt=2 ms)")
    lines.append(f"  Model out: virtual 6 kHz velocity (T_dense=49, dt={DT_MODEL*1e3:.3f} ms ≈ 6 kHz)")
    lines.append(f"  Reference: GT 5 kHz and GT 20 kHz native Loupas velocity")
    lines.append(f"  Physics:   cs via cross-correlation ToA | E = 3ρcs² | Voigt η")
    lines.append(f"  f_push   : {F_PUSH:.0f} Hz  (SWE ARF push frequency)")
    lines.append(f"  ρ        : {RHO:.0f} kg/m³")
    lines.append(f"  Samples  : {df['name'].nunique()} × 500 Hz val samples")
    lines.append("")

    # Overall summary table
    lines.append("OVERALL MEAN ACCURACY  (vs GT 5 kHz)  |  all phantom types")
    lines.append("-" * 80)
    lines.append(f"{'Model':<22} {'N':>4}  {'cs MAE (m/s)':>14}  {'cs Bias':>10}  "
                 f"{'E MAE (kPa)':>12}  {'η MAE (Pa·s)':>13}  {'Pearson r':>10}")
    lines.append("-" * 80)

    for mkey in MODEL_REGISTRY:
        sub = df[df["model"] == mkey]
        sub_cs = sub.dropna(subset=["cs_mae"])
        n = len(sub_cs)
        if n == 0:
            lines.append(f"  {MODEL_REGISTRY[mkey]['label']:<20} {'N/A':>4}")
            continue
        cs_mae  = sub_cs["cs_mae"].mean()
        cs_bias = sub_cs["cs_bias"].mean()
        E_mae   = sub_cs["E_mae"].mean()
        eta_mae = sub_cs["eta_mae"].mean() if sub_cs["eta_mae"].notna().any() else np.nan

        # Pearson r on valid pairs
        valid_r = sub_cs.dropna(subset=["cs_pred", "cs_ref5"])
        if len(valid_r) >= 2:
            r_val, _ = pearsonr(valid_r["cs_pred"], valid_r["cs_ref5"])
        else:
            r_val = np.nan

        eta_str = f"{eta_mae:.3f}" if np.isfinite(eta_mae) else "  N/A"
        r_str   = f"{r_val:.4f}"   if np.isfinite(r_val)  else "  N/A"
        lines.append(f"  {MODEL_REGISTRY[mkey]['label']:<20} {n:>4}  {cs_mae:>14.4f}  "
                     f"{cs_bias:>+10.4f}  {E_mae:>12.3f}  {eta_str:>13}  {r_str:>10}")
    lines.append("")

    # Per-phantom breakdown
    for src in ["2ft", "3ft", "cirs"]:
        sub_src = df[df["source"] == src]
        if sub_src.empty:
            continue
        lines.append(f"PHANTOM: {src.upper()}")
        lines.append("-" * 60)
        lines.append(f"{'Model':<22} {'N':>4}  {'cs MAE':>10}  {'cs Bias':>9}  "
                     f"{'E MAE':>9}  {'η MAE':>10}")
        lines.append("-" * 60)
        for mkey in MODEL_REGISTRY:
            sub = sub_src[sub_src["model"] == mkey].dropna(subset=["cs_mae"])
            n = len(sub)
            if n == 0:
                continue
            eta_s = (f"{sub['eta_mae'].mean():.3f}"
                     if sub["eta_mae"].notna().any() else "N/A")
            lines.append(f"  {MODEL_REGISTRY[mkey]['label']:<20} {n:>4}  "
                         f"{sub['cs_mae'].mean():>10.4f}  {sub['cs_bias'].mean():>+9.4f}  "
                         f"{sub['E_mae'].mean():>9.3f}  {eta_s:>10}")
        lines.append("")

    # Verdict
    # Rank models by cs MAE (main clinical metric)
    ranking = []
    for mkey in MODEL_REGISTRY:
        sub = df[df["model"] == mkey].dropna(subset=["cs_mae"])
        if not sub.empty:
            ranking.append((mkey, sub["cs_mae"].mean(), sub["cs_bias"].mean()))
    ranking.sort(key=lambda x: x[1])

    lines.append("RANKING  (by cs MAE vs GT 5 kHz — primary SWE metric)")
    lines.append("-" * 60)
    for rank, (mkey, mae, bias) in enumerate(ranking, 1):
        lines.append(f"  #{rank}  {MODEL_REGISTRY[mkey]['label']:<20}  "
                     f"cs MAE = {mae:.4f} m/s  |  bias = {bias:+.4f} m/s")
    lines.append("")
    if ranking:
        best_key = ranking[0][0]
        lines.append(f"BEST MODEL: {MODEL_REGISTRY[best_key]['label']}")
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)
    fp = out / "comparison_report.txt"
    fp.write_text(report_text)
    print(f"\n  Report saved: {fp}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/fullscale_velocity")
    p.add_argument("--output_dir", default="outputs/model_comparison")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of 500Hz val samples (for debugging)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  WavePINN Model Comparison — Spatial SWE Maps")
    print(f"  Device : {device}")
    print(f"  Output : {out}")
    print(f"{'='*60}\n")

    # ── Load all models ────────────────────────────────────────────────────────
    print("Loading models …")
    models: Dict[str, Optional[torch.nn.Module]] = {}
    for mkey in MODEL_REGISTRY:
        models[mkey] = load_model(mkey, device)
    n_loaded = sum(m is not None for m in models.values())
    print(f"\n  {n_loaded}/{len(MODEL_REGISTRY)} models loaded.\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading 500 Hz val samples …")
    samples_500 = load_500hz_val_samples(args.data_dir)
    print(f"  Found {len(samples_500)} samples.")

    if args.max_samples:
        samples_500 = samples_500[:args.max_samples]
        print(f"  [debug] Limited to {args.max_samples} samples.")

    print("\nLoading 5 kHz / 20 kHz reference velocities …")
    refs = load_reference_velocities(args.data_dir)
    print(f"  5 kHz : {len(refs['5kHz'])} groups")
    print(f"  20 kHz: {len(refs['20kHz'])} groups\n")

    # ── Run analysis ───────────────────────────────────────────────────────────
    print("Running inference and SWE analysis …")
    all_results = []
    for i, sample in enumerate(samples_500):
        print(f"  [{i+1:>3}/{len(samples_500)}] {sample['name'][:60]} …", flush=True)
        result = analyse_sample(sample, models, refs, device)
        all_results.append(result)

    print(f"\n  Analysis complete: {len(all_results)} samples processed.\n")

    # ── Build metrics table ────────────────────────────────────────────────────
    print("Building metrics table …")
    df = build_metrics_df(all_results)
    csv_path = out / "metrics_per_sample.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Per-sample metrics saved: {csv_path}\n")

    # ── Generate figures ───────────────────────────────────────────────────────
    print("Generating publication figures …")
    reps = _pick_representative_samples(all_results)

    print(f"  E-map figures ({len(reps)} representative samples) …")
    fig_spatial_maps(reps, out)

    print(f"  η-map figures …")
    fig_eta_maps(reps, out)

    print(f"  Metrics summary bar chart …")
    fig_metrics_summary(df, out)

    print(f"  cs scatter plots …")
    fig_cs_scatter(all_results, out)

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print_report(df, all_results, out)

    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    main()
