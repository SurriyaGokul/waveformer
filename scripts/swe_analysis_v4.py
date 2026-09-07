#!/usr/bin/env python3
"""
swe_analysis_v4.py
==================
SWE viscoelastic analysis and visualization for WavePINN v4 predictions.

Computes:
  - Shear wave speed (cs) via cross-correlation ToA
  - Voigt model viscoelastic parameters (G', η, G'', |G*|, E, tan δ)
  - Comparison across: GT 500Hz, GT 5kHz, v4 Pred 1.2kHz

Also generates side-by-side velocity comparison videos (GT vs Pred).

Usage:
    python scripts/swe_analysis_v4.py
    python scripts/swe_analysis_v4.py --no_video
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import pandas as pd

# ─── Physical constants ──────────────────────────────────────────────────────
RHO              = 1000.0       # kg/m³
F_PUSH           = 400.0        # Hz (SWE push frequency)
OMEGA            = 2.0 * np.pi * F_PUSH
LATERAL_M_PER_PX = 0.1013e-3   # m/px
AXIAL_MM_PER_PX  = 0.0507      # mm/px
LATERAL_MM_PER_PX = 0.1013     # mm/px
H_ROI, W_ROI     = 116, 160

# CIRS phantom quasi-static elastic moduli (E → G_e = E/3)
G_ELASTIC = {"2ft": 2000.0 / 3.0, "3ft": 3000.0 / 3.0}  # Pa
NOMINAL_CS = {"2ft": 1.41, "3ft": 1.73}  # quasi-static cs (m/s)

# Timing
DT_500HZ  = 1.0 / 500.0    # 2 ms
DT_5KHZ   = 1.0 / 5000.0   # 0.2 ms
DT_1200HZ = 1.0 / 1200.0   # 0.833 ms

UPSAMPLING_FACTOR = 12

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "500Hz_gt":   {"color": "#D6604D", "label": "GT 500 Hz",           "marker": "o"},
    "5kHz_gt":    {"color": "#2166AC", "label": "GT 5 kHz (ref)",      "marker": "^"},
    "v4_1.2kHz":  {"color": "#4DAC26", "label": "v4 Pred 1.2 kHz",    "marker": "s"},
}
ORDER = ["500Hz_gt", "5kHz_gt", "v4_1.2kHz"]
PHANTOM_LABEL = {"2ft": "2FT (E≈2 kPa)", "3ft": "3FT (E≈3 kPa)"}

plt.rcParams.update({
    "figure.dpi":         150,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "legend.framealpha":  0.85,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-correlation ToA cs estimation (from swe_publication.py)
# ═══════════════════════════════════════════════════════════════════════════════

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

    Returns: (cs_median_m_s, cs_std_m_s, cs_map)
    """
    T, Ht, Wt = velocity.shape
    r0 = int(Ht * depth_lo)
    r1 = int(Ht * depth_hi)

    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)  # (T, W)
    v_abs = np.abs(v_avg)

    amp = v_abs.max(axis=0)
    thr = np.percentile(amp[amp > 0], 15) if amp.max() > 0 else 0
    valid = amp > thr

    x_push = _push_column(v_abs, n_early=max(3, T // 10))
    excl_lo = max(0, x_push - excl_px)
    excl_hi = min(Wt - 1, x_push + excl_px)
    valid[excl_lo:excl_hi + 1] = False

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
        ea = np.sqrt((a ** 2).sum())
        eb = np.sqrt((b ** 2).sum())
        if ea < 1e-12 or eb < 1e-12:
            continue

        xc = np.correlate(a / ea, b / eb, mode="full")
        lags = np.arange(-(T - 1), T, dtype=float)
        pi = int(np.argmax(xc))

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
    iqr = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    cs_std = iqr / 1.349

    if not (0.3 < cs_med < 12.0):
        return np.nan, np.nan, cs_map

    return cs_med, cs_std, cs_map


# ═══════════════════════════════════════════════════════════════════════════════
# Voigt model inversion
# ═══════════════════════════════════════════════════════════════════════════════

def voigt_invert(cs_m_s: float, source: str,
                 f_push: float = F_PUSH,
                 rho: float = RHO) -> dict:
    """
    Invert Voigt model: given measured cs and known quasi-static G_e,
    recover η, G'', |G*|, tan δ.
    """
    nan_row = dict(G_prime_kpa=np.nan, eta_Pas=np.nan,
                   G_loss_kpa=np.nan, G_star_kpa=np.nan,
                   tan_delta=np.nan, E_kpa=np.nan)

    if not np.isfinite(cs_m_s) or cs_m_s <= 0:
        return nan_row

    G_e = G_ELASTIC.get(source, np.nan)
    if not np.isfinite(G_e):
        return nan_row

    omega = 2.0 * np.pi * f_push
    cs2rho = cs_m_s ** 2 * rho

    # Solve: 2y² − cs²ρ·y − cs²ρ·G_e = 0
    a, b, c = 2.0, -cs2rho, -cs2rho * G_e
    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        return nan_row
    y = (-b + np.sqrt(disc)) / (2.0 * a)  # |G*| in Pa

    omega_eta_sq = y ** 2 - G_e ** 2
    if omega_eta_sq < 0:
        return nan_row
    omega_eta = np.sqrt(omega_eta_sq)
    eta = omega_eta / omega

    G_loss = omega_eta / 1000.0  # kPa
    G_prime = G_e / 1000.0       # kPa
    G_star = y / 1000.0          # kPa
    tan_d = G_loss / G_prime if G_prime > 0 else np.nan
    E_kpa = 3.0 * G_prime

    if not (0.001 <= eta <= 50.0):
        return nan_row

    return dict(G_prime_kpa=G_prime, eta_Pas=eta,
                G_loss_kpa=G_loss, G_star_kpa=G_star,
                tan_delta=tan_d, E_kpa=E_kpa)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _source(name: str) -> str:
    if "2ft" in name:
        return "2ft"
    if "3ft" in name:
        return "3ft"
    return "cirs"


def _plane_key(name: str) -> Optional[str]:
    m = re.search(r'(\d+ft_\d+mmfocus_Plane_\d+)', name)
    return m.group(1) if m else None


def load_v4_data(eval_dir: str) -> List[dict]:
    """Load v4 predictions and matched GT from the eval output directory."""
    eval_dir = Path(eval_dir)
    results = []
    for pred_path in sorted(eval_dir.glob("velocity_1200hz_v4_*.npy")):
        name = pred_path.stem.replace("velocity_1200hz_v4_", "")
        gt_path = eval_dir / f"gt_velocity_{name}.npy"
        if not gt_path.exists():
            continue
        results.append({
            "name": name,
            "pred": np.load(str(pred_path)),
            "gt500": np.load(str(gt_path)),
            "source": _source(name),
            "plane_key": _plane_key(name),
        })
    return results


def load_5khz_gt(data_dir: str = "data/fullscale_velocity") -> Dict[str, dict]:
    """Load all 5kHz GT velocity arrays, keyed by plane_key."""
    result = {}
    data_dir = Path(data_dir)
    for split in ("train", "val", "inference"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.glob("velocity_*5kHz*.npy")):
            name = p.stem.replace("velocity_", "")
            pk = _plane_key(name)
            if pk and pk not in result:
                result[pk] = {
                    "velocity": np.load(str(p)),
                    "name": name,
                    "path": str(p),
                    "source": _source(name),
                }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Video generation
# ═══════════════════════════════════════════════════════════════════════════════

def _try_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") is not None


def write_comparison_video(pred, gt500, sample_name, out_path, fps=15):
    """Side-by-side GT (500Hz) vs Pred (1.2kHz) video with error panel."""
    T = min(pred.shape[0], gt500.shape[0])
    pred = pred[:T]
    gt500 = gt500[:T]

    dt_pred_ms = 1000.0 / 1200.0
    dt_gt_ms = 1000.0 / 500.0

    clim = max(float(np.abs(gt500).max()), float(np.abs(pred).max()), 0.2)

    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(20, 6), facecolor="black")
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.04],
                           wspace=0.07, left=0.05, right=0.97, top=0.86, bottom=0.12)
    ax_gt = fig.add_subplot(gs[0])
    ax_pr = fig.add_subplot(gs[1])
    ax_err = fig.add_subplot(gs[2])
    ax_cb = fig.add_subplot(gs[3])

    titles = [
        f"GT 500 Hz\n(dt={dt_gt_ms:.1f} ms)",
        f"v4 Pred 1.2 kHz\n(dt={dt_pred_ms:.3f} ms)",
        "|Error|",
    ]
    for ax, ttl in zip([ax_gt, ax_pr, ax_err], titles):
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(ttl, fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel("Lateral (mm)", fontsize=8)
    ax_gt.set_ylabel("Axial depth (mm)", fontsize=8)

    kw = dict(cmap="jet", vmin=-clim, vmax=clim, aspect="auto",
              origin="upper", extent=extent)

    im_gt = ax_gt.imshow(gt500[0], **kw)
    im_pr = ax_pr.imshow(pred[0], **kw)
    im_err = ax_err.imshow(np.abs(pred[0] - gt500[0]),
                           cmap="hot", vmin=0, vmax=clim,
                           aspect="auto", origin="upper", extent=extent)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Particle velocity (mm/s)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    sup = fig.suptitle("", color="white", fontsize=10, fontweight="bold")

    def _update(fi):
        im_gt.set_data(gt500[fi])
        im_pr.set_data(pred[fi])
        im_err.set_data(np.abs(pred[fi] - gt500[fi]))
        t_pred = fi * dt_pred_ms
        t_gt = fi * dt_gt_ms
        sup.set_text(
            f"WavePINN v4 — {sample_name}  |  "
            f"frame {fi + 1}/{T}  |  "
            f"GT t={t_gt:.2f} ms  |  Pred t={t_pred:.3f} ms"
        )
        return im_gt, im_pr, im_err, sup

    anim = FuncAnimation(fig, _update, frames=T, blit=True, interval=1000 // fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _try_ffmpeg():
        w = FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
        sp = out_path.with_suffix(".mp4")
        anim.save(str(sp), writer=w, dpi=120, savefig_kwargs={"facecolor": "black"})
    else:
        w = PillowWriter(fps=fps)
        sp = out_path.with_suffix(".gif")
        anim.save(str(sp), writer=w, savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    return sp


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rheology_bars(df: pd.DataFrame, out: Path) -> None:
    """4-panel bar chart: G_dyn, E, η, tan δ  per phantom × condition."""
    sources = ["2ft", "3ft"]
    params = [
        ("G_dyn_kpa", r"$G_0 = \rho c_s^2$ (kPa)", "Dynamic Shear Modulus"),
        ("E_kpa", r"Young's Modulus $E = 3G_e$ (kPa)", "Young's Modulus"),
        ("eta_Pas", r"Shear Viscosity $\eta$ (Pa·s)", "Shear Viscosity"),
        ("tan_delta", r"Loss Tangent $\tan\delta$", "Loss Tangent"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "Viscoelastic Tissue Characterisation — WavePINN v4\n"
        r"Voigt model: $G^*(\omega)=G_e + i\omega\eta$  |  "
        f"$f_{{push}}$ = {F_PUSH:.0f} Hz",
        fontsize=12, fontweight="bold")

    for ax, (col, ylabel, title) in zip(axes.flat, params):
        x_tick_pos = []
        x_tick_labs = []
        cluster_gap = 1.2
        bar_w = 0.28
        offsets = [-bar_w, 0, bar_w]

        for si, src in enumerate(sources):
            cx = si * cluster_gap
            for oi, cond in enumerate(ORDER):
                vals = df[(df.condition == cond) & (df.source == src)][col].dropna()
                if len(vals) == 0:
                    ax.text(cx + offsets[oi], 0.02, "N/A",
                            ha="center", va="bottom", fontsize=7,
                            color=PALETTE[cond]["color"], rotation=90)
                    continue
                m, s = vals.mean(), vals.sem()
                ax.bar(cx + offsets[oi], m,
                       width=bar_w * 0.9,
                       color=PALETTE[cond]["color"],
                       alpha=0.85, edgecolor="white",
                       yerr=s, capsize=4,
                       error_kw=dict(linewidth=1.5, ecolor="black"))
                ax.text(cx + offsets[oi], m + s + 0.01 * max(ax.get_ylim()[1], 0.1),
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7)

            x_tick_pos.append(cx)
            x_tick_labs.append(PHANTOM_LABEL[src])

        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_labs, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        if col == "G_dyn_kpa":
            patches = [mpatches.Patch(fc=PALETTE[c]["color"],
                       label=PALETTE[c]["label"]) for c in ORDER]
            ax.legend(handles=patches, fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_rheology_bars.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: fig_rheology_bars.pdf / .png")


def fig_cs_comparison(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of cs per sample, grouped by condition."""
    fig, ax = plt.subplots(figsize=(14, 6))

    planes = sorted(df["plane_key"].dropna().unique())
    n_planes = len(planes)
    x = np.arange(n_planes)
    bar_w = 0.25
    offsets = {"500Hz_gt": -bar_w, "5kHz_gt": 0, "v4_1.2kHz": bar_w}

    for cond in ORDER:
        cs_vals = []
        cs_errs = []
        for pk in planes:
            sub = df[(df.condition == cond) & (df.plane_key == pk)]
            if len(sub) > 0:
                cs_vals.append(sub["cs_m_s"].mean())
                cs_errs.append(sub["cs_m_s"].std() if len(sub) > 1 else 0)
            else:
                cs_vals.append(0)
                cs_errs.append(0)

        bars = ax.bar(x + offsets[cond], cs_vals,
                      width=bar_w * 0.9,
                      color=PALETTE[cond]["color"],
                      alpha=0.85, label=PALETTE[cond]["label"],
                      yerr=cs_errs, capsize=3,
                      error_kw=dict(linewidth=1))

        for xi, v in enumerate(cs_vals):
            if v > 0:
                ax.text(xi + offsets[cond], v + 0.05,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Nominal cs lines
    for src, cs_nom in NOMINAL_CS.items():
        src_planes = [i for i, pk in enumerate(planes) if src in pk]
        if src_planes:
            ax.hlines(cs_nom, min(src_planes) - 0.4, max(src_planes) + 0.4,
                      linestyle="--", color="gray", alpha=0.7, linewidth=1.5)
            ax.text(max(src_planes) + 0.45, cs_nom,
                    f"Nominal {src.upper()} cs={cs_nom:.2f}",
                    fontsize=7, color="gray", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([pk.replace("_", "\n") for pk in planes],
                       fontsize=8, rotation=0)
    ax.set_ylabel("Shear wave speed cs (m/s)", fontsize=11)
    ax.set_title("Shear Wave Speed Comparison: GT 500Hz vs GT 5kHz vs v4 Pred 1.2kHz",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_cs_comparison.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: fig_cs_comparison.pdf / .png")


def fig_youngs_modulus(df: pd.DataFrame, out: Path) -> None:
    """G_dyn (dynamic shear modulus) per phantom compared to nominal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Dynamic Shear Modulus from SWE\n"
                 r"$G_0 = \rho \cdot c_s^2$ (kPa)",
                 fontsize=12, fontweight="bold")

    for ax, src in zip(axes, ["2ft", "3ft"]):
        for cond in ORDER:
            vals = df[(df.condition == cond) & (df.source == src)]["G_dyn_kpa"].dropna()
            if len(vals) > 0:
                ax.bar(cond, vals.mean(),
                       yerr=vals.sem(), capsize=5,
                       color=PALETTE[cond]["color"],
                       alpha=0.85, width=0.5)
                ax.text(cond, vals.mean() + vals.sem() + 0.05,
                        f"{vals.mean():.2f}", ha="center", va="bottom", fontsize=9)

        nom_G = NOMINAL_CS[src] ** 2 * RHO / 1000.0
        ax.axhline(nom_G, ls="--", color="gray", alpha=0.6)
        ax.text(0.95, nom_G + 0.05, f"Nominal={nom_G:.2f} kPa",
                transform=ax.get_yaxis_transform(), fontsize=8, color="gray",
                ha="right", va="bottom")

        ax.set_title(PHANTOM_LABEL[src], fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$G_0$ (kPa)", fontsize=10)
        ax.set_xticklabels([PALETTE[c]["label"] for c in ORDER],
                           fontsize=8, rotation=15, ha="right")

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_youngs_modulus.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: fig_youngs_modulus.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def fig_learned_cs_analysis(df: pd.DataFrame, learned_cs: float, out: Path) -> None:
    """Comparison of model learned cs vs 5kHz xcorr cs per phantom."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        "Shear Wave Speed: Model Learned cs vs 5kHz Reference\n"
        r"$c_s$ from wave equation regularizer $\partial^2 v/\partial t^2 = c_s^2 \nabla^2 v$",
        fontsize=12, fontweight="bold")

    for ax, src in zip(axes, ["2ft", "3ft"]):
        # 5kHz reference
        sub_5k = df[(df.condition == "5kHz_gt") & (df.source == src)]["cs_m_s"].dropna()
        # Nominal
        cs_nom = NOMINAL_CS[src]

        bar_x = [0, 1, 2]
        bar_labels = ["Nominal\n(quasi-static)", "GT 5kHz\n(xcorr ToA)", "v4 Learned\n(wave eq reg)"]
        bar_vals = [cs_nom,
                    sub_5k.mean() if len(sub_5k) > 0 else 0,
                    learned_cs]
        bar_errs = [0,
                    sub_5k.std() if len(sub_5k) > 1 else 0,
                    0]
        bar_colors = ["#888888", PALETTE["5kHz_gt"]["color"], PALETTE["v4_1.2kHz"]["color"]]

        bars = ax.bar(bar_x, bar_vals, yerr=bar_errs, capsize=5,
                      color=bar_colors, alpha=0.85, width=0.5, edgecolor="white")

        for xi, (v, e) in enumerate(zip(bar_vals, bar_errs)):
            if v > 0:
                ax.text(xi, v + e + 0.05, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        ax.set_xticks(bar_x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel("cs (m/s)", fontsize=11)
        ax.set_title(PHANTOM_LABEL[src], fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(bar_vals) * 1.25)

        # Dynamic G annotation
        G_learned = RHO * learned_cs ** 2 / 1000.0
        G_5k = RHO * sub_5k.mean() ** 2 / 1000.0 if len(sub_5k) > 0 else 0
        ax.text(0.98, 0.95,
                f"G (learned) = {G_learned:.1f} kPa\n"
                f"G (5kHz) = {G_5k:.1f} kPa\n"
                f"G (nominal) = {cs_nom**2 * RHO / 1000:.1f} kPa",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_learned_cs_analysis.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: fig_learned_cs_analysis.pdf / .png")


def load_learned_cs(ckpt_path: str = "outputs/wavepinn_v4/best_model.pt") -> float:
    """Load the learned cs from the v4 model checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return float(ckpt.get("learned_cs", np.nan))


def main():
    parser = argparse.ArgumentParser(
        description="SWE viscoelastic analysis for WavePINN v4"
    )
    parser.add_argument("--eval_dir", default="outputs/1200hz_predictions_v4_eval")
    parser.add_argument("--output_dir", default="outputs/swe_analysis_v4")
    parser.add_argument("--model_path", default="outputs/wavepinn_v4/best_model.pt")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model's learned cs ───────────────────────────────────────────────
    learned_cs = load_learned_cs(args.model_path)
    print(f"Model learned cs = {learned_cs:.4f} m/s")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading v4 predictions and GT...")
    v4_data = load_v4_data(args.eval_dir)
    print(f"  Loaded {len(v4_data)} samples")

    print("Loading 5kHz GT reference...")
    gt5k = load_5khz_gt()
    print(f"  Loaded {len(gt5k)} 5kHz reference samples")

    # ── Compute cs for all conditions ─────────────────────────────────────────
    # NOTE: Cross-correlation ToA cs estimation ONLY works at sufficiently high
    # temporal resolution. At 500Hz (dt=2ms), a 3 m/s shear wave traverses
    # ~60 pixels/frame — the wavefront is unresolvable by xcorr.
    # Similarly, v4 predictions at 1.2kHz have low amplitude and lack the
    # spatial propagation structure needed for ToA estimation.
    #
    # Strategy:
    # - GT 5kHz: Use standard xcorr ToA (works well at dt=0.2ms)
    # - GT 500Hz: Attempt xcorr ToA (expected to fail for most samples)
    # - v4 Pred: Use the model's LEARNED cs (from wave equation regularizer)
    #            as a global estimate. This is a regularizer compromise across
    #            phantom types, not a per-sample measurement.
    rows = []

    for sample in v4_data:
        name = sample["name"]
        src = sample["source"]
        pk = sample["plane_key"]
        print(f"\n[{name}]")

        # --- v4 prediction: use model learned cs (global regularizer) ---
        pred = sample["pred"]
        # Also attempt xcorr (for completeness)
        cs_pred_xcorr, _, _ = estimate_cs(pred, DT_1200HZ)
        # Use learned cs as the primary estimate
        cs_pred = learned_cs
        G_dyn_pred = RHO * cs_pred ** 2 / 1000.0
        vp = voigt_invert(cs_pred, src)
        xcorr_note = f"(xcorr={cs_pred_xcorr:.2f})" if np.isfinite(cs_pred_xcorr) else "(xcorr=N/A)"
        print(f"  v4 Pred 1.2kHz:  cs={cs_pred:.3f} m/s {xcorr_note}  G_dyn={G_dyn_pred:.2f} kPa")
        rows.append({
            "name": name, "condition": "v4_1.2kHz", "source": src,
            "plane_key": pk, "cs_m_s": cs_pred, "cs_std": 0.0,
            "cs_method": "learned", "G_dyn_kpa": G_dyn_pred, **vp,
        })

        # --- GT 500Hz ---
        gt500 = sample["gt500"]
        cs_gt500, cs_gt500_std, _ = estimate_cs(gt500, DT_500HZ)
        G_dyn_gt500 = RHO * cs_gt500 ** 2 / 1000.0 if np.isfinite(cs_gt500) else np.nan
        vp500 = voigt_invert(cs_gt500, src)
        cs_str = f"{cs_gt500:.3f}" if np.isfinite(cs_gt500) else "N/A"
        G_str = f"{G_dyn_gt500:.2f}" if np.isfinite(G_dyn_gt500) else "N/A"
        print(f"  GT 500Hz:        cs={cs_str} m/s  G_dyn={G_str} kPa  (xcorr fails: dt too coarse)")
        rows.append({
            "name": name, "condition": "500Hz_gt", "source": src,
            "plane_key": pk, "cs_m_s": cs_gt500, "cs_std": cs_gt500_std,
            "cs_method": "xcorr", "G_dyn_kpa": G_dyn_gt500, **vp500,
        })

        # --- GT 5kHz (if available) ---
        if pk and pk in gt5k:
            gt5 = gt5k[pk]["velocity"]
            cs_5k, cs_5k_std, _ = estimate_cs(gt5, DT_5KHZ)
            G_dyn_5k = RHO * cs_5k ** 2 / 1000.0 if np.isfinite(cs_5k) else np.nan
            vp5k = voigt_invert(cs_5k, src)
            print(f"  GT 5kHz:         cs={cs_5k:.3f} m/s  G_dyn={G_dyn_5k:.2f} kPa  (xcorr)")
            rows.append({
                "name": gt5k[pk]["name"], "condition": "5kHz_gt", "source": src,
                "plane_key": pk, "cs_m_s": cs_5k, "cs_std": cs_5k_std,
                "cs_method": "xcorr", "G_dyn_kpa": G_dyn_5k, **vp5k,
            })

        # --- Video ---
        if not args.no_video:
            vid_path = out_dir / f"video_{name}"
            print(f"  Creating video...")
            try:
                saved = write_comparison_video(
                    pred, gt500, name, vid_path, fps=args.fps)
                print(f"  Video -> {saved}")
            except Exception as e:
                print(f"  Video FAILED: {e}")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = out_dir / "swe_metrics_v4.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nMetrics CSV -> {csv_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SWE VISCOELASTIC ANALYSIS — WavePINN v4")
    print("=" * 80)
    print(f"\nModel learned cs = {learned_cs:.4f} m/s "
          f"(G_dyn = {RHO * learned_cs**2 / 1000:.2f} kPa)")
    print("\nNote: 500Hz GT xcorr ToA fails (dt=2ms too coarse for wavefront).")
    print("v4 uses the model's learned cs from wave equation regularizer.\n")

    for src in ["2ft", "3ft"]:
        print(f"--- {PHANTOM_LABEL[src]} ---")
        for cond in ORDER:
            sub = df[(df.condition == cond) & (df.source == src)]
            if len(sub) == 0:
                continue
            cs = sub["cs_m_s"].dropna()
            G = sub["G_dyn_kpa"].dropna()
            E = sub["E_kpa"].dropna()
            eta = sub["eta_Pas"].dropna()
            td = sub["tan_delta"].dropna()

            label = PALETTE[cond]["label"]
            if len(cs) == 0:
                print(f"  {label:25s}  cs=N/A  (xcorr failed)")
                continue

            parts = [f"cs={cs.mean():.3f}+/-{cs.std():.3f} m/s",
                     f"G_dyn={G.mean():.2f} kPa"]
            if len(E) > 0:
                parts.append(f"E={E.mean():.2f} kPa")
            if len(eta) > 0:
                parts.append(f"eta={eta.mean():.3f} Pa.s")
            if len(td) > 0:
                parts.append(f"tan_d={td.mean():.3f}")
            print(f"  {label:25s}  {'  '.join(parts)}")
        print()

    print(f"Nominal: 2FT cs~1.41 m/s (G~2.0 kPa), 3FT cs~1.73 m/s (G~3.0 kPa)")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_cs_comparison(df, out_dir)
    fig_rheology_bars(df, out_dir)
    fig_youngs_modulus(df, out_dir)

    # ── Learned cs analysis figure ────────────────────────────────────────────
    fig_learned_cs_analysis(df, learned_cs, out_dir)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "description": "SWE viscoelastic analysis — WavePINN v4 (1.2kHz from 100Hz)",
        "model_learned_cs_m_s": learned_cs,
        "model_G_dyn_kpa": RHO * learned_cs ** 2 / 1000.0,
        "conditions": ORDER,
        "notes": {
            "500Hz_gt": "xcorr ToA fails at 500Hz — dt=2ms too coarse",
            "v4_pred": "Uses model learned cs (global wave eq regularizer)",
            "5kHz_gt": "xcorr ToA reference standard",
        },
        "per_phantom": {},
    }
    for src in ["2ft", "3ft"]:
        summary["per_phantom"][src] = {}
        for cond in ORDER:
            sub = df[(df.condition == cond) & (df.source == src)]
            if len(sub) == 0:
                continue
            entry = {"n_samples": len(sub)}
            for col in ["cs_m_s", "G_dyn_kpa", "E_kpa", "eta_Pas", "tan_delta"]:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    entry[f"{col}_mean"] = float(vals.mean())
                    entry[f"{col}_std"] = float(vals.std())
            summary["per_phantom"][src][cond] = entry

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out_dir}/summary.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
