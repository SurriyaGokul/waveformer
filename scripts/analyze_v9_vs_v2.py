#!/usr/bin/env python3
"""
V9 (best & final) vs V2 — Comprehensive Analysis
==================================================

Generates:
  1. Comparison videos  (GT | v9_best | v9_final | v2 | |Err_best| | |Err_v2|)
  2. Pixel-level metrics  (MAE, RMSE, Pearson r — full & active-frame)
  3. SWE-specific metrics (cs, G', E, η) — using 5 kHz GT velocity as reference
  4. Text report + CSV + figures

Physical context
----------------
  - 5 kHz IQ: 5 sparse frames at sparse_step=12 → dt_sparse = 12*(1/5000)=2.4 ms
  - Model outputs 49 dense frames at dt_output = 2.4ms/12 = 0.2ms (≡ 5 kHz)
  - 5 kHz GT: 49 frames at 0.2 ms intervals  ← reference for ALL SWE metrics
  - v9 passes dt_seconds to model; v2 does not use temporal conditioning

Usage
-----
    python scripts/analyze_v9_vs_v2.py \\
        --v9_best   SWE_PINN_outputs_wavepinn_v9_best_model.pt \\
        --v9_final  SWE_PINN_outputs_wavepinn_v9_final_model.pt \\
        --v2_model  saved_models/fullscale_velocity_v2_best.pt \\
        --data_dir  data/fullscale_velocity/val \\
        --output_dir outputs/v9_vs_v2_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.architectures.velocity_unet import ComplexToRealUNet
from src.architectures.wavepinn_v9 import WavePINNv9, create_wavepinn_v9

# ---------------------------------------------------------------------------
# Physical / spatial constants
# ---------------------------------------------------------------------------
RHO_KG_M3         = 1000.0
LATERAL_M_PER_PX  = 0.1013e-3   # m/px
AXIAL_M_PER_PX    = 0.0507e-3   # m/px
LATERAL_MM_PER_PX = 0.1013      # mm/px
AXIAL_MM_PER_PX   = 0.0507      # mm/px
H_ROI             = 116
W_ROI             = 160

DT_5KHZ = 1.0 / 5000.0          # 0.2 ms — native 5 kHz frame time
SPARSE_STEP_5KHZ = 12            # preprocessing sparse_step for 5 kHz
DT_SPARSE_5KHZ = SPARSE_STEP_5KHZ * DT_5KHZ   # 2.4 ms between sparse inputs
DT_OUTPUT_5KHZ = DT_SPARSE_5KHZ / 12          # 0.2 ms per output frame

N_ZOOM = 20   # frames for zoom video


# ===========================================================================
# Utilities
# ===========================================================================

def _pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    *leading, H, W = arr.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH == 0 and pW == 0:
        return arr
    return np.pad(arr, [(0, 0)] * len(leading) + [(0, pH), (0, pW)])


def _try_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def make_axes(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    dep_ax = np.arange(H) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX
    return lat_ax, dep_ax


def parse_group(name: str) -> Optional[str]:
    """Extract (source, focus, plane) group key."""
    m = re.match(r"((?:2ft|3ft|cirs)_\w+focus_(?:Plane_\d+|\w+))", name)
    if m:
        return m.group(1)
    return None


def _mean_std(vals):
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


# ===========================================================================
# Model loading
# ===========================================================================

def load_v9(path: str, device: torch.device, label: str = "v9") -> WavePINNv9:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.pop("name", None)
    model = create_wavepinn_v9(config=model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}/WavePINNv9] epoch={epoch}  best_val_loss={val_loss:.6f}  params={n_params:,}")
    return model


def load_v2(path: str, device: torch.device) -> ComplexToRealUNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = ComplexToRealUNet(
        in_channels              = cfg.get("in_channels", 1),
        base_channels            = cfg.get("base_channels", 32),
        encoder_depths           = list(cfg.get("encoder_depths", [32, 64, 128, 256])),
        upsampling_factor        = cfg.get("upsampling_factor", 12),
        use_motion_interpolation = cfg.get("use_motion_interpolation", True),
        complex_to_real_mode     = cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn             = cfg.get("use_naive_bn", True),
        flow_hidden_channels     = cfg.get("flow_hidden_channels", 32),
        flow_scale               = cfg.get("flow_scale", 0.1),
        use_prf_conditioning     = cfg.get("use_prf_conditioning", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [v2/ComplexToRealUNet] epoch={epoch}  best_val_loss={val_loss:.6f}  params={n_params:,}")
    return model


# ===========================================================================
# Inference
# ===========================================================================

@torch.no_grad()
def infer_all(
    v9b_model: WavePINNv9,
    v9f_model: WavePINNv9,
    v2_model:  ComplexToRealUNet,
    iq_path:   Path,
    vel_path:  Path,
    device:    torch.device,
    dt_sparse: float = DT_SPARSE_5KHZ,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
        gt_mm      : (49, H_ROI, W)  5 kHz GT velocity [mm/s]
        v9b_mm     : (49, H_ROI, W)  v9_best prediction [mm/s]
        v9f_mm     : (49, H_ROI, W)  v9_final prediction [mm/s]
        v2_mm      : (49, H_ROI, W)  v2 prediction [mm/s]
        cs_best    : scalar cs estimate from v9_best head [m/s]
        cs_final   : scalar cs estimate from v9_final head [m/s]
    """
    iq_data  = np.load(iq_path).astype(np.complex64)    # (T_sp, H, W)
    vel_data = np.load(vel_path).astype(np.float32)      # (49, H, W)

    # Per-sample IQ normalisation
    iq_scale = max(float(np.percentile(np.abs(iq_data), 99)), 1.0)
    iq_norm  = iq_data / iq_scale
    inp_arr  = np.stack([iq_norm.real, iq_norm.imag], axis=0)   # (2, T_sp, H, W)

    # Per-sample velocity normalisation
    vel_scale = max(float(np.percentile(np.abs(vel_data), 99)), 1.0)

    # Padding
    inp_pad = _pad_to_multiple(inp_arr)
    inp_t   = torch.from_numpy(inp_pad).unsqueeze(0).to(device)   # (1, 2, T_sp, Hp, Wp)

    dt_t = torch.tensor([dt_sparse], dtype=torch.float32, device=device)

    # ── v9_best ────────────────────────────────────────────────────────────
    v9b_vel, v9b_cs_t = v9b_model(inp_t, dt_seconds=dt_t, detach_head=False)
    v9b_norm = v9b_vel[0, 0].cpu().numpy()[:, :H_ROI, :]   # (T, H_ROI, W)
    cs_best  = float(v9b_cs_t[0].item())

    # ── v9_final ───────────────────────────────────────────────────────────
    v9f_vel, v9f_cs_t = v9f_model(inp_t, dt_seconds=dt_t, detach_head=False)
    v9f_norm = v9f_vel[0, 0].cpu().numpy()[:, :H_ROI, :]
    cs_final = float(v9f_cs_t[0].item())

    # ── v2 ─────────────────────────────────────────────────────────────────
    v2_out  = v2_model(inp_t)
    if isinstance(v2_out, dict):
        v2_out = v2_out["velocity"]
    v2_norm = v2_out[0, 0].cpu().numpy()[:, :H_ROI, :]

    # ── GT ─────────────────────────────────────────────────────────────────
    vel_norm = np.clip(vel_data / vel_scale, -1.0, 1.0)
    gt_mm    = vel_norm[:, :H_ROI, :] * vel_scale

    v9b_mm = v9b_norm * vel_scale
    v9f_mm = v9f_norm * vel_scale
    v2_mm  = v2_norm  * vel_scale

    return gt_mm, v9b_mm, v9f_mm, v2_mm, cs_best, cs_final


# ===========================================================================
# Pixel metrics
# ===========================================================================

def compute_metrics(gt: np.ndarray, pred: np.ndarray,
                    active_std_thresh: float = 0.1) -> Dict:
    diff = pred - gt
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    g, p = gt.flatten(), pred.flatten()
    corr = float(np.corrcoef(g, p)[0, 1]) if g.std() > 1e-8 else float("nan")

    frame_stds = gt.std(axis=(1, 2))
    active     = frame_stds > active_std_thresh

    if active.any():
        ga, pa    = gt[active], pred[active]
        mae_act   = float(np.abs(pa - ga).mean())
        corr_act  = float(np.corrcoef(ga.flatten(), pa.flatten())[0, 1])
        rmse_act  = float(np.sqrt(((pa - ga) ** 2).mean()))
    else:
        mae_act = rmse_act = corr_act = float("nan")

    return dict(
        mae=mae, rmse=rmse, corr=corr,
        mae_active=mae_act, rmse_active=rmse_act, corr_active=corr_act,
        n_active=int(active.sum()),
    )


# ===========================================================================
# SWE physics
# ===========================================================================

def voigt_phase_velocity(omega, G_Pa, eta):
    numer = 2 * (G_Pa**2 + (omega * eta)**2)
    denom = RHO_KG_M3 * (np.sqrt(G_Pa**2 + (omega * eta)**2) + G_Pa)
    return np.sqrt(numer / denom)


def _find_push_col(v_abs: np.ndarray, early: int = 3) -> int:
    return int(np.argmax(v_abs[:min(early, v_abs.shape[0])].max(axis=0)))


def estimate_cs_toa(velocity: np.ndarray, dt_s: float,
                    min_amp_pct: float = 15.0,
                    depth_frac: Tuple[float, float] = (0.2, 0.8),
                    push_excl_px: int = 15,
                    step: int = 5) -> Dict:
    T, H, W = velocity.shape
    r0, r1  = int(H * depth_frac[0]), int(H * depth_frac[1])
    v_abs   = np.mean(np.abs(velocity[:, r0:r1, :]), axis=1)

    amp_thresh = np.percentile(v_abs.max(axis=0), min_amp_pct)
    amp_ok     = v_abs.max(axis=0) > amp_thresh

    xp = _find_push_col(v_abs, early=max(3, int(0.1 * T)))
    push_mask = np.zeros(W, bool)
    push_mask[max(0, xp - push_excl_px): min(W, xp + push_excl_px + 1)] = True

    valid = amp_ok & ~push_mask
    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)

    min_step = max(1, int(np.ceil(0.15 * dt_s / LATERAL_M_PER_PX)))
    STEP     = max(step, min_step)

    cs_list = []
    for x in range(0, W - STEP):
        x2 = x + STEP
        if not (valid[x] and valid[x2]):
            continue
        sa, sb = v_avg[:, x], v_avg[:, x2]
        ea, eb = np.sqrt(np.sum(sa**2)), np.sqrt(np.sum(sb**2))
        if ea < 1e-12 or eb < 1e-12:
            continue
        xcorr = np.correlate(sa / ea, sb / eb, mode="full")
        lags  = np.arange(-(T - 1), T)
        pi    = int(np.argmax(xcorr))
        if 0 < pi < len(xcorr) - 1:
            y0, y1, y2 = xcorr[pi - 1], xcorr[pi], xcorr[pi + 1]
            dn = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / dn if abs(dn) > 1e-12 else 0.0
            lag = float(lags[pi]) + frac
        else:
            lag = float(lags[pi])
        if abs(lag) < 0.05:
            continue
        cs_local = abs(STEP * LATERAL_M_PER_PX / (lag * dt_s))
        if 0.2 < cs_local < 15.0:
            cs_list.append(cs_local)

    if not cs_list:
        return dict(cs_m_s=float("nan"), cs_std=float("nan"))

    cs_arr = np.array(cs_list)
    cs_med = float(np.median(cs_arr))
    cs_iqr = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    cs_std = cs_iqr / 1.349
    if not (0.2 < cs_med < 15.0):
        return dict(cs_m_s=float("nan"), cs_std=float("nan"))
    return dict(cs_m_s=cs_med, cs_std=cs_std)


def estimate_viscosity_voigt(velocity: np.ndarray, dt_s: float,
                              depth_frac: Tuple[float, float] = (0.25, 0.75)) -> Dict:
    T, H, W = velocity.shape
    r0, r1  = int(H * depth_frac[0]), int(H * depth_frac[1])
    v_avg   = np.mean(velocity[:, r0:r1, :], axis=1)   # (T, W)

    V      = np.fft.fftshift(np.fft.fft2(v_avg))
    omegas = np.fft.fftshift(np.fft.fftfreq(T, dt_s)) * 2 * np.pi
    kxs    = np.fft.fftshift(np.fft.fftfreq(W, LATERAL_M_PER_PX))

    omega_ok = omegas > 0
    kx_pos   = kxs > 0
    kx_neg   = kxs < 0

    omega_bins = omegas[omega_ok]
    kx_half    = kxs[kx_pos]

    P_pos = np.abs(V[np.ix_(omega_ok, kx_pos)])**2
    P_neg = np.abs(V[np.ix_(omega_ok, kx_neg)])**2
    n_kx  = min(P_pos.shape[1], P_neg.shape[1])
    P_sub = P_pos[:, :n_kx] + P_neg[:, ::-1][:, :n_kx]
    kx    = kx_half[:n_kx]

    total = P_sub.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kx_c = np.where(total > 0, (P_sub * kx[None, :]).sum(axis=1) / total, float("nan"))

    with np.errstate(divide="ignore", invalid="ignore"):
        cs_d = np.where(kx_c > 0, omega_bins / kx_c, float("nan"))

    valid = (cs_d > 0.3) & (cs_d < 15.0) & np.isfinite(cs_d)
    ow, cw = omega_bins[valid], cs_d[valid]

    if len(ow) < 5:
        return dict(G_prime_kPa=float("nan"), G_prime_std_kPa=float("nan"),
                    eta_Pa_s=float("nan"), eta_std_Pa_s=float("nan"))

    cs_m = float(np.nanmedian(cw))
    try:
        popt, pcov = curve_fit(
            voigt_phase_velocity,
            ow, cw,
            p0=[RHO_KG_M3 * cs_m**2, 0.5],
            bounds=([100, 0.001], [500000, 20.0]),
            maxfev=8000,
        )
        perr = np.sqrt(np.diag(pcov))
        return dict(
            G_prime_kPa=popt[0] / 1000,
            G_prime_std_kPa=perr[0] / 1000,
            eta_Pa_s=popt[1],
            eta_std_Pa_s=perr[1],
        )
    except Exception:
        G = RHO_KG_M3 * cs_m**2
        return dict(G_prime_kPa=G / 1000, G_prime_std_kPa=float("nan"),
                    eta_Pa_s=float("nan"), eta_std_Pa_s=float("nan"))


def compute_swe_metrics(velocity: np.ndarray, dt_s: float) -> Dict:
    """Full SWE pipeline: ToA cs → G' → E → Voigt η."""
    toa  = estimate_cs_toa(velocity, dt_s)
    visc = estimate_viscosity_voigt(velocity, dt_s)

    cs     = toa["cs_m_s"]
    cs_std = toa["cs_std"]

    if np.isfinite(cs):
        G_prime = RHO_KG_M3 * cs**2 / 1000
        G_std   = 2 * RHO_KG_M3 * cs * cs_std / 1000 if np.isfinite(cs_std) else float("nan")
        E_kPa   = 3.0 * G_prime
        E_std   = 3.0 * G_std if np.isfinite(G_std) else float("nan")
    else:
        G_prime = G_std = E_kPa = E_std = float("nan")

    # Fallback to Voigt G' if ToA failed
    if not np.isfinite(G_prime) and np.isfinite(visc["G_prime_kPa"]):
        G_prime = visc["G_prime_kPa"]
        G_std   = visc["G_prime_std_kPa"]
        E_kPa   = 3.0 * G_prime
        E_std   = 3.0 * G_std if np.isfinite(G_std) else float("nan")

    return dict(
        cs_m_s=cs,
        cs_std=cs_std,
        G_prime_kPa=G_prime,
        G_prime_std=G_std,
        E_kPa=E_kPa,
        E_std=E_std,
        eta_Pa_s=visc["eta_Pa_s"],
        eta_std=visc["eta_std_Pa_s"],
    )


# ===========================================================================
# Video writer
# ===========================================================================

def write_video(
    gt: np.ndarray,
    v9b: np.ndarray,
    v9f: np.ndarray,
    v2: np.ndarray,
    lat_ax: np.ndarray,
    dep_ax: np.ndarray,
    t_ms: np.ndarray,
    clim: float,
    title: str,
    out_path: Path,
    fps: int = 8,
) -> Path:
    """6-panel: GT | v9_best | v9_final | v2 | |Err_best| | |Err_v2|."""
    T      = len(t_ms)
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(30, 5), facecolor="black")
    gs  = gridspec.GridSpec(
        1, 7, figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.04],
        wspace=0.06, left=0.03, right=0.97, top=0.85, bottom=0.12,
    )

    panels = [fig.add_subplot(gs[i]) for i in range(6)]
    ax_cb  = fig.add_subplot(gs[6])

    for ax in panels:
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    kw_vel = dict(cmap="jet", vmin=-clim, vmax=clim, aspect="auto",
                  origin="upper", extent=extent)
    kw_err = dict(cmap="hot", vmin=0,    vmax=clim, aspect="auto",
                  origin="upper", extent=extent)

    ims = [
        panels[0].imshow(gt[0],              **kw_vel),
        panels[1].imshow(v9b[0],             **kw_vel),
        panels[2].imshow(v9f[0],             **kw_vel),
        panels[3].imshow(v2[0],              **kw_vel),
        panels[4].imshow(np.abs(gt[0]-v9b[0]), **kw_err),
        panels[5].imshow(np.abs(gt[0]-v2[0]),  **kw_err),
    ]

    labels = ["GT 5 kHz", "v9 Best", "v9 Final", "v2 UNet",
              "|Err| v9 Best", "|Err| v2"]
    for ax, lbl in zip(panels, labels):
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xlabel("Lateral (mm)", fontsize=7)
        ax.set_ylabel("Depth (mm)",   fontsize=7)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Velocity (mm/s)", color="white", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    sup = fig.suptitle("", color="white", fontsize=9, fontweight="bold")

    def _upd(fi: int):
        sup.set_text(f"{title}  |  t = {t_ms[fi]:.2f} ms  (frame {fi+1}/{T})")
        ims[0].set_data(gt[fi])
        ims[1].set_data(v9b[fi])
        ims[2].set_data(v9f[fi])
        ims[3].set_data(v2[fi])
        ims[4].set_data(np.abs(gt[fi] - v9b[fi]))
        ims[5].set_data(np.abs(gt[fi] - v2[fi]))
        return ims + [sup]

    anim = FuncAnimation(fig, _upd, frames=T, blit=True, interval=1000 // fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _try_ffmpeg():
        writer    = FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
        save_path = out_path.with_suffix(".mp4")
        anim.save(str(save_path), writer=writer, dpi=100,
                  savefig_kwargs={"facecolor": "black"})
    else:
        writer    = PillowWriter(fps=fps)
        save_path = out_path.with_suffix(".gif")
        anim.save(str(save_path), writer=writer,
                  savefig_kwargs={"facecolor": "black"})

    plt.close(fig)
    return save_path


# ===========================================================================
# Figure helpers
# ===========================================================================

def save_pixel_metrics_figure(records: List[Dict], out_path: Path):
    """Bar charts: MAE and Pearson r for v9_best, v9_final, v2."""
    names    = [r["name"][-32:] for r in records]
    v9b_mae  = [r["v9b_mae_active"]  for r in records]
    v9f_mae  = [r["v9f_mae_active"]  for r in records]
    v2_mae   = [r["v2_mae_active"]   for r in records]
    v9b_corr = [r["v9b_corr_active"] for r in records]
    v9f_corr = [r["v9f_corr_active"] for r in records]
    v2_corr  = [r["v2_corr_active"]  for r in records]

    x = np.arange(len(records))
    w = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(records) * 0.8), 10))
    fig.suptitle("V9 Best / V9 Final / V2 — Active-frame Pixel Metrics (5 kHz val)",
                 fontsize=13, fontweight="bold")

    colors = ["steelblue", "darkorange", "coral"]

    for ax, (vals_a, vals_b, vals_c), ylabel, better in [
        (axes[0], (v9b_mae,  v9f_mae,  v2_mae),  "Active-frame MAE (mm/s)",  "lower"),
        (axes[1], (v9b_corr, v9f_corr, v2_corr), "Active-frame Pearson r",   "higher"),
    ]:
        ax.bar(x - w, vals_a, width=w, label="v9 Best",  color=colors[0], alpha=0.85)
        ax.bar(x,     vals_b, width=w, label="v9 Final", color=colors[1], alpha=0.85)
        ax.bar(x + w, vals_c, width=w, label="v2 UNet",  color=colors[2], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} (↓ better)" if better == "lower" else f"{ylabel} (↑ better)",
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)

        for vals, color, lbl in zip([vals_a, vals_b, vals_c], colors,
                                     ["v9 Best", "v9 Final", "v2"]):
            fin = [v for v in vals if np.isfinite(v)]
            if fin:
                mu = float(np.mean(fin))
                ax.axhline(mu, color=color, lw=1.5, ls="--",
                           label=f"{lbl} mean={mu:.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pixel metrics figure → {out_path}")


def save_swe_comparison_figure(swe_by_name: Dict[str, Dict], out_path: Path):
    """
    Scatter plots: model predictions vs 5 kHz GT SWE metrics.
    Each point = one 5 kHz val sample.
    """
    gt5_cs,  v9b_cs,  v9f_cs,  v2_cs  = [], [], [], []
    gt5_G,   v9b_G,   v9f_G,   v2_G   = [], [], [], []
    gt5_E,   v9b_E,   v9f_E,   v2_E   = [], [], [], []
    gt5_eta, v9b_eta, v9f_eta, v2_eta = [], [], [], []

    for name, d in swe_by_name.items():
        gt_s  = d.get("gt_swe",      {})
        v9b_s = d.get("v9b_swe",     {})
        v9f_s = d.get("v9f_swe",     {})
        v2_s  = d.get("v2_swe",      {})

        gt5_cs.append(gt_s.get("cs_m_s",    float("nan")))
        v9b_cs.append(v9b_s.get("cs_m_s",   float("nan")))
        v9f_cs.append(v9f_s.get("cs_m_s",   float("nan")))
        v2_cs.append(v2_s.get("cs_m_s",     float("nan")))

        gt5_G.append(gt_s.get("G_prime_kPa",  float("nan")))
        v9b_G.append(v9b_s.get("G_prime_kPa", float("nan")))
        v9f_G.append(v9f_s.get("G_prime_kPa", float("nan")))
        v2_G.append(v2_s.get("G_prime_kPa",   float("nan")))

        gt5_E.append(gt_s.get("E_kPa",  float("nan")))
        v9b_E.append(v9b_s.get("E_kPa", float("nan")))
        v9f_E.append(v9f_s.get("E_kPa", float("nan")))
        v2_E.append(v2_s.get("E_kPa",   float("nan")))

        gt5_eta.append(gt_s.get("eta_Pa_s",  float("nan")))
        v9b_eta.append(v9b_s.get("eta_Pa_s", float("nan")))
        v9f_eta.append(v9f_s.get("eta_Pa_s", float("nan")))
        v2_eta.append(v2_s.get("eta_Pa_s",   float("nan")))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("SWE Metrics: v9 Best, v9 Final & v2 Predictions vs 5 kHz Ground Truth",
                 fontsize=13, fontweight="bold")

    metric_sets = [
        (gt5_cs,  v9b_cs,  v9f_cs,  v2_cs,  "cs (m/s)",  axes[0, 0]),
        (gt5_G,   v9b_G,   v9f_G,   v2_G,   "G' (kPa)",  axes[0, 1]),
        (gt5_E,   v9b_E,   v9f_E,   v2_E,   "E (kPa)",   axes[1, 0]),
        (gt5_eta, v9b_eta, v9f_eta, v2_eta, "η (Pa·s)",  axes[1, 1]),
    ]

    for gt5, v9b_v, v9f_v, v2_v, ylabel, ax in metric_sets:
        all_vals = []
        for pred_v, color, lbl, marker in [
            (v9b_v, "steelblue",  "v9 Best",  "o"),
            (v9f_v, "darkorange", "v9 Final", "s"),
            (v2_v,  "coral",      "v2 UNet",  "^"),
        ]:
            pairs = [(g, p) for g, p in zip(gt5, pred_v)
                     if np.isfinite(g) and np.isfinite(p)]
            if pairs:
                gv, pv = zip(*pairs)
                ax.scatter(gv, pv, c=color, alpha=0.75, label=lbl, marker=marker, s=60)
                all_vals.extend(list(gv) + list(pv))

        if all_vals:
            lo, hi = min(all_vals) * 0.85, max(all_vals) * 1.15
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y=x", zorder=0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_xlabel(f"5 kHz GT {ylabel}", fontsize=10)
        ax.set_ylabel(f"Predicted {ylabel}", fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  SWE comparison figure → {out_path}")


def save_cs_head_figure(records: List[Dict], out_path: Path):
    """Compare v9 ScalarCsHead estimates vs ToA estimates vs 5kHz GT SWE cs."""
    names = [r["name"][-30:] for r in records]
    cs_head_best  = [r["cs_head_best"]  for r in records]
    cs_head_final = [r["cs_head_final"] for r in records]
    cs_toa_best   = [r.get("v9b_swe_cs", float("nan")) for r in records]
    cs_toa_gt     = [r.get("gt_swe_cs",  float("nan")) for r in records]

    x = np.arange(len(records))
    w = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(records) * 0.7), 6))
    fig.suptitle("v9 ScalarCsHead vs ToA cs vs 5 kHz GT — per sample", fontsize=12)

    ax.bar(x - 1.5*w, cs_head_best,  width=w, label="v9 Best (CsHead)",  color="steelblue", alpha=0.85)
    ax.bar(x - 0.5*w, cs_head_final, width=w, label="v9 Final (CsHead)", color="darkorange", alpha=0.85)
    ax.bar(x + 0.5*w, cs_toa_best,   width=w, label="v9 Best (ToA)",     color="royalblue",  alpha=0.6)
    ax.bar(x + 1.5*w, cs_toa_gt,     width=w, label="5 kHz GT (ToA)",    color="green",       alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("cs (m/s)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cs-head figure → {out_path}")


def save_swe_bar_figure(swe_by_name: Dict, out_path: Path):
    """
    Bar chart of SWE metrics (cs, G', E, η) for all conditions per sample.
    """
    names = list(swe_by_name.keys())
    short = [n[-28:] for n in names]
    x = np.arange(len(names))
    w = 0.2

    conditions = [
        ("gt_swe",  "GT 5 kHz",  "green"),
        ("v9b_swe", "v9 Best",   "steelblue"),
        ("v9f_swe", "v9 Final",  "darkorange"),
        ("v2_swe",  "v2 UNet",   "coral"),
    ]

    metrics_def = [
        ("cs_m_s",     "cs (m/s)"),
        ("G_prime_kPa", "G' (kPa)"),
        ("E_kPa",       "E (kPa)"),
        ("eta_Pa_s",    "η (Pa·s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(max(16, len(names)*0.9), 12))
    fig.suptitle("SWE Metrics per Sample — GT 5 kHz vs v9 Best vs v9 Final vs v2",
                 fontsize=13, fontweight="bold")

    for (mkey, mlabel), ax in zip(metrics_def, axes.flatten()):
        for i, (cond_key, cond_label, color) in enumerate(conditions):
            vals = [swe_by_name[n].get(cond_key, {}).get(mkey, float("nan"))
                    for n in names]
            offset = (i - 1.5) * w
            ax.bar(x + offset, vals, width=w, label=cond_label, color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  SWE bar chart → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V9 Best / V9 Final / V2 — Visualisation + SWE analysis"
    )
    parser.add_argument("--v9_best",    default="SWE_PINN_outputs_wavepinn_v9_best_model.pt")
    parser.add_argument("--v9_final",   default="SWE_PINN_outputs_wavepinn_v9_final_model.pt")
    parser.add_argument("--v2_model",   default="saved_models/fullscale_velocity_v2_best.pt")
    parser.add_argument("--data_dir",   default="data/fullscale_velocity/val")
    parser.add_argument("--output_dir", default="outputs/v9_vs_v2_analysis")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fps",        type=int, default=8)
    parser.add_argument("--no_video",   action="store_true")
    parser.add_argument("--skip_swe",   action="store_true")
    args = parser.parse_args()

    out_dir  = PROJECT_ROOT / args.output_dir
    data_dir = PROJECT_ROOT / args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print("  V9 Best / V9 Final / V2 UNet — Comprehensive SWE Analysis")
    print(f"{'='*72}")
    print(f"  device      : {device}")
    print(f"  data_dir    : {data_dir}")
    print(f"  output_dir  : {out_dir}")
    print(f"  5 kHz timing: dt_sparse={DT_SPARSE_5KHZ*1e3:.2f} ms  "
          f"dt_output={DT_OUTPUT_5KHZ*1e3:.3f} ms  T=49 frames")
    print()

    # ── 1. Load models ─────────────────────────────────────────────────────
    print("[1/5] Loading models …")
    v9b_model = load_v9(str(PROJECT_ROOT / args.v9_best),  device, label="v9_best")
    v9f_model = load_v9(str(PROJECT_ROOT / args.v9_final), device, label="v9_final")
    v2_model  = load_v2(str(PROJECT_ROOT / args.v2_model), device)
    print()

    # ── 2. Discover 5 kHz val samples ─────────────────────────────────────
    print("[2/5] Discovering 5 kHz val samples …")
    sparse_files = sorted(data_dir.glob("sparse_*5kHz*.npy"))
    if args.max_samples:
        sparse_files = sparse_files[:args.max_samples]
    print(f"  Found {len(sparse_files)} samples\n")

    lat_ax, dep_ax = make_axes(H_ROI, W_ROI)

    all_pixel_records: List[Dict] = []
    cs_head_records:   List[Dict] = []

    # ── 3. Inference + pixel metrics + videos ─────────────────────────────
    print("[3/5] Inference, pixel metrics, and videos …")
    video_dir = out_dir / "videos"

    for sp_path in sparse_files:
        name     = sp_path.stem[len("sparse_"):]
        vel_path = sp_path.parent / f"velocity_{name}.npy"
        if not vel_path.exists():
            print(f"  SKIP (no velocity): {name}")
            continue

        try:
            gt_mm, v9b_mm, v9f_mm, v2_mm, cs_best, cs_final = infer_all(
                v9b_model, v9f_model, v2_model, sp_path, vel_path, device,
                dt_sparse=DT_SPARSE_5KHZ,
            )
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Pixel metrics vs 5 kHz GT
        T  = gt_mm.shape[0]
        m  = {"name": name, "group": parse_group(name)}
        mv = compute_metrics(gt_mm, v9b_mm)
        m.update({"v9b_" + k: v for k, v in mv.items()})
        mv = compute_metrics(gt_mm, v9f_mm)
        m.update({"v9f_" + k: v for k, v in mv.items()})
        mv = compute_metrics(gt_mm, v2_mm)
        m.update({"v2_" + k: v for k, v in mv.items()})
        m["cs_head_best"]  = cs_best
        m["cs_head_final"] = cs_final
        all_pixel_records.append(m)

        t_ms  = np.arange(T) * DT_OUTPUT_5KHZ * 1e3
        title = f"5kHz | {name[-38:]}"

        # clim from GT active frames
        frame_stds  = gt_mm.std(axis=(1, 2))
        active_mask = frame_stds > 0.05
        clim = float(np.abs(gt_mm[active_mask] if active_mask.any() else gt_mm).max())
        clim = max(clim, 0.2)

        if not args.no_video:
            vid_path = write_video(
                gt_mm, v9b_mm, v9f_mm, v2_mm,
                lat_ax, dep_ax, t_ms, clim, title,
                video_dir / name,
                fps=args.fps,
            )
            # Zoom video (first N_ZOOM active frames)
            n_zoom = min(N_ZOOM, T)
            write_video(
                gt_mm[:n_zoom], v9b_mm[:n_zoom], v9f_mm[:n_zoom], v2_mm[:n_zoom],
                lat_ax, dep_ax, t_ms[:n_zoom], clim,
                title + f" [zoom {n_zoom} frames]",
                video_dir / (name + "_zoom"),
                fps=max(2, args.fps // 2),
            )

        n_act = m.get("v9b_n_active", 0)
        print(
            f"  {name[-40:]:40s} "
            f"active={n_act} "
            f"v9b_MAE={m.get('v9b_mae_active', float('nan')):.3f}  "
            f"v9f_MAE={m.get('v9f_mae_active', float('nan')):.3f}  "
            f"v2_MAE={m.get('v2_mae_active', float('nan')):.3f}  "
            f"cs_best={cs_best:.2f}  cs_final={cs_final:.2f}"
        )

    # ── Save pixel metrics CSV ─────────────────────────────────────────────
    pix_csv = out_dir / "pixel_metrics_5khz.csv"
    if all_pixel_records:
        keys = list(all_pixel_records[0].keys())
        with pix_csv.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(all_pixel_records)
        print(f"\n  Pixel metrics CSV → {pix_csv}")

    # ── 4. SWE analysis ────────────────────────────────────────────────────
    swe_by_name: Dict[str, Dict] = {}

    if not args.skip_swe:
        print("\n[4/5] SWE analysis (cs, G', E, η) vs 5 kHz GT …")

        for sp_path in sparse_files:
            name     = sp_path.stem[len("sparse_"):]
            vel_path = sp_path.parent / f"velocity_{name}.npy"
            if not vel_path.exists():
                continue

            try:
                gt_mm, v9b_mm, v9f_mm, v2_mm, _, _ = infer_all(
                    v9b_model, v9f_model, v2_model, sp_path, vel_path, device,
                    dt_sparse=DT_SPARSE_5KHZ,
                )
            except Exception as e:
                print(f"  SWE ERROR {name}: {e}")
                continue

            # SWE from 5 kHz GT (use native 5 kHz dt)
            gt_swe  = compute_swe_metrics(gt_mm,  DT_OUTPUT_5KHZ)
            # SWE from model predictions (same output dt)
            v9b_swe = compute_swe_metrics(v9b_mm, DT_OUTPUT_5KHZ)
            v9f_swe = compute_swe_metrics(v9f_mm, DT_OUTPUT_5KHZ)
            v2_swe  = compute_swe_metrics(v2_mm,  DT_OUTPUT_5KHZ)

            swe_by_name[name] = {
                "gt_swe":  gt_swe,
                "v9b_swe": v9b_swe,
                "v9f_swe": v9f_swe,
                "v2_swe":  v2_swe,
                "group":   parse_group(name),
            }

            # Also attach cs_head values to records for the cs-head figure
            for rec in all_pixel_records:
                if rec["name"] == name:
                    rec["gt_swe_cs"]  = gt_swe["cs_m_s"]
                    rec["v9b_swe_cs"] = v9b_swe["cs_m_s"]
                    rec["v9f_swe_cs"] = v9f_swe["cs_m_s"]
                    rec["v2_swe_cs"]  = v2_swe["cs_m_s"]

            print(
                f"  {name[-42:]:42s} "
                f"GT_cs={gt_swe['cs_m_s']:.3f}  "
                f"v9b_cs={v9b_swe['cs_m_s']:.3f}  "
                f"v9f_cs={v9f_swe['cs_m_s']:.3f}  "
                f"v2_cs={v2_swe['cs_m_s']:.3f}"
            )

    # ── 5. Report + Figures ────────────────────────────────────────────────
    print("\n[5/5] Generating report and figures …")

    if all_pixel_records:
        save_pixel_metrics_figure(all_pixel_records, out_dir / "pixel_metrics.png")
        save_cs_head_figure(all_pixel_records, out_dir / "cs_head_comparison.png")

    if swe_by_name:
        save_swe_comparison_figure(swe_by_name, out_dir / "swe_scatter.png")
        save_swe_bar_figure(swe_by_name, out_dir / "swe_bars.png")

        # Save SWE CSV
        swe_csv = out_dir / "swe_metrics_5khz.csv"
        swe_rows = []
        for name, d in swe_by_name.items():
            row = {"name": name, "group": d.get("group", "")}
            for cond in ["gt_swe", "v9b_swe", "v9f_swe", "v2_swe"]:
                for k, v in d.get(cond, {}).items():
                    row[f"{cond}_{k}"] = v
            swe_rows.append(row)
        if swe_rows:
            with swe_csv.open("w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(swe_rows[0].keys()))
                wr.writeheader()
                wr.writerows(swe_rows)
            print(f"  SWE metrics CSV → {swe_csv}")

    # ── Text report ────────────────────────────────────────────────────────
    report_lines = []

    def row(text=""):
        report_lines.append(text)
        print(text)

    sep  = "=" * 80
    sep2 = "-" * 60

    row(); row(sep)
    row("  V9 BEST / V9 FINAL / V2 UNET — EVALUATION REPORT")
    row(f"  5 kHz val split  |  GT = native 5 kHz velocity (dt = {DT_OUTPUT_5KHZ*1e3:.3f} ms)")
    row(sep)

    # ── Pixel metrics ──
    row(); row("PIXEL-LEVEL METRICS (vs 5 kHz GT velocity)"); row(sep2)
    row(f"  {'Model':<14}  {'MAE_full':>10}  {'RMSE_full':>10}  {'r_full':>8}  "
        f"{'MAE_act':>10}  {'RMSE_act':>10}  {'r_act':>8}  {'N_act':>6}")
    row("  " + "-" * 78)

    for mkey, label in [("v9b", "v9 Best"), ("v9f", "v9 Final"), ("v2", "v2 UNet")]:
        maes  = [r[f"{mkey}_mae"]         for r in all_pixel_records if np.isfinite(r.get(f"{mkey}_mae", float("nan")))]
        rmses = [r[f"{mkey}_rmse"]        for r in all_pixel_records if np.isfinite(r.get(f"{mkey}_rmse", float("nan")))]
        corrs = [r[f"{mkey}_corr"]        for r in all_pixel_records if np.isfinite(r.get(f"{mkey}_corr", float("nan")))]
        mact  = [r[f"{mkey}_mae_active"]  for r in all_pixel_records if np.isfinite(r.get(f"{mkey}_mae_active", float("nan")))]
        ract  = [r[f"{mkey}_rmse_active"] for r in all_pixel_records if np.isfinite(r.get(f"{mkey}_rmse_active", float("nan")))]
        cact  = [r[f"{mkey}_corr_active"] for r in all_pixel_records if np.isfinite(r.get(f"{mkey}_corr_active", float("nan")))]
        nact  = [r[f"{mkey}_n_active"]    for r in all_pixel_records]

        def mu(lst): return float(np.mean(lst)) if lst else float("nan")
        row(f"  {label:<14}  {mu(maes):>10.4f}  {mu(rmses):>10.4f}  {mu(corrs):>8.4f}  "
            f"{mu(mact):>10.4f}  {mu(ract):>10.4f}  {mu(cact):>8.4f}  {int(mu(nact)):>6d}")

    # ── SWE metrics ──
    if swe_by_name:
        row(); row(); row("SWE METRICS (ToA cs + Voigt η)  |  Ground truth = 5 kHz GT velocity")
        row(sep2)
        row(f"  {'Model':<14}  {'cs (m/s)':>12}  {'G\' (kPa)':>12}  {'E (kPa)':>12}  "
            f"{'η (Pa·s)':>12}  {'cs MAE':>10}  {'G\' MAE':>10}")
        row("  " + "-" * 86)

        cond_keys = [
            ("gt_swe",  "GT 5 kHz"),
            ("v9b_swe", "v9 Best"),
            ("v9f_swe", "v9 Final"),
            ("v2_swe",  "v2 UNet"),
        ]

        # Per-phantom breakdown
        phantom_groups = defaultdict(list)
        for name, d in swe_by_name.items():
            grp = d.get("group") or name
            phantom_groups[grp].append(d)

        for cond_key, cond_label in cond_keys:
            cs_vals  = [d[cond_key]["cs_m_s"]     for d in swe_by_name.values()
                        if np.isfinite(d[cond_key].get("cs_m_s",  float("nan")))]
            G_vals   = [d[cond_key]["G_prime_kPa"] for d in swe_by_name.values()
                        if np.isfinite(d[cond_key].get("G_prime_kPa", float("nan")))]
            E_vals   = [d[cond_key]["E_kPa"]       for d in swe_by_name.values()
                        if np.isfinite(d[cond_key].get("E_kPa",      float("nan")))]
            eta_vals = [d[cond_key]["eta_Pa_s"]    for d in swe_by_name.values()
                        if np.isfinite(d[cond_key].get("eta_Pa_s",   float("nan")))]

            mu_cs  = float(np.mean(cs_vals))  if cs_vals  else float("nan")
            mu_G   = float(np.mean(G_vals))   if G_vals   else float("nan")
            mu_E   = float(np.mean(E_vals))   if E_vals   else float("nan")
            mu_eta = float(np.mean(eta_vals)) if eta_vals else float("nan")

            # MAE vs 5 kHz GT
            if cond_key != "gt_swe":
                cs_maes = []
                G_maes  = []
                for d in swe_by_name.values():
                    gt_cs = d["gt_swe"].get("cs_m_s",     float("nan"))
                    pd_cs = d[cond_key].get("cs_m_s",     float("nan"))
                    gt_G  = d["gt_swe"].get("G_prime_kPa", float("nan"))
                    pd_G  = d[cond_key].get("G_prime_kPa", float("nan"))
                    if np.isfinite(gt_cs) and np.isfinite(pd_cs):
                        cs_maes.append(abs(pd_cs - gt_cs))
                    if np.isfinite(gt_G) and np.isfinite(pd_G):
                        G_maes.append(abs(pd_G - gt_G))
                cs_mae = float(np.mean(cs_maes)) if cs_maes else float("nan")
                G_mae  = float(np.mean(G_maes))  if G_maes  else float("nan")
            else:
                cs_mae = G_mae = float("nan")

            row(f"  {cond_label:<14}  {mu_cs:>12.4f}  {mu_G:>12.4f}  {mu_E:>12.4f}  "
                f"{mu_eta:>12.4f}  {cs_mae:>10.4f}  {G_mae:>10.4f}")

        # ── ScalarCsHead vs ToA vs GT ──
        row(); row(); row("v9 SCALAR CS HEAD  vs  ToA Cs  vs  5 kHz GT Cs"); row(sep2)
        row(f"  {'Model':<14}  {'CsHead mean':>13}  {'CsHead MAE':>12}  "
            f"{'ToA cs mean':>13}  {'ToA MAE':>10}")
        row("  " + "-" * 68)

        for hkey, tkey, label in [
            ("cs_head_best",  "v9b_swe_cs", "v9 Best"),
            ("cs_head_final", "v9f_swe_cs", "v9 Final"),
        ]:
            head_vals = [r[hkey] for r in all_pixel_records if np.isfinite(r.get(hkey, float("nan")))]
            toa_vals  = [r[tkey] for r in all_pixel_records if np.isfinite(r.get(tkey, float("nan")))]
            gt_vals   = [r.get("gt_swe_cs", float("nan")) for r in all_pixel_records]

            head_maes = [abs(h - g) for h, g in zip(
                         [r[hkey] for r in all_pixel_records],
                         gt_vals) if np.isfinite(h) and np.isfinite(g)]
            toa_maes  = [abs(t - g) for t, g in zip(
                         [r.get(tkey, float("nan")) for r in all_pixel_records],
                         gt_vals) if np.isfinite(t) and np.isfinite(g)]

            mu_head     = float(np.mean(head_vals)) if head_vals else float("nan")
            mu_toa      = float(np.mean(toa_vals))  if toa_vals  else float("nan")
            mu_head_mae = float(np.mean(head_maes)) if head_maes else float("nan")
            mu_toa_mae  = float(np.mean(toa_maes))  if toa_maes  else float("nan")

            row(f"  {label:<14}  {mu_head:>13.4f}  {mu_head_mae:>12.4f}  "
                f"{mu_toa:>13.4f}  {mu_toa_mae:>10.4f}")

        # ── Per-phantom breakdown ──
        row(); row(); row("PER-PHANTOM SWE BREAKDOWN"); row(sep2)
        G_prime_col = "G'"
        row(f"  {'Phantom':<38}  {'Model':<14}  {'cs':>6}  {G_prime_col:>8}  {'E':>8}  {'eta':>8}  {'cs_MAE':>8}")
        row("  " + "-" * 96)

        for grp in sorted(phantom_groups.keys()):
            entries = phantom_groups[grp]
            for cond_key, cond_label in cond_keys:
                cs_l  = [e[cond_key].get("cs_m_s",     float("nan")) for e in entries]
                G_l   = [e[cond_key].get("G_prime_kPa", float("nan")) for e in entries]
                E_l   = [e[cond_key].get("E_kPa",       float("nan")) for e in entries]
                eta_l = [e[cond_key].get("eta_Pa_s",    float("nan")) for e in entries]

                mu = lambda lst: float(np.nanmean([v for v in lst if np.isfinite(v)])) if any(np.isfinite(v) for v in lst) else float("nan")

                cs_gt_l = [e["gt_swe"].get("cs_m_s", float("nan")) for e in entries]
                if cond_key != "gt_swe":
                    mae_cs = float(np.nanmean([abs(p - g) for p, g in zip(cs_l, cs_gt_l)
                                               if np.isfinite(p) and np.isfinite(g)]) or float("nan"))
                else:
                    mae_cs = float("nan")

                row(f"  {grp[:38]:<38}  {cond_label:<14}  "
                    f"{mu(cs_l):>6.3f}  {mu(G_l):>8.3f}  {mu(E_l):>8.3f}  "
                    f"{mu(eta_l):>8.3f}  {mae_cs:>8.3f}")
            row()

    row(); row(sep)
    row(f"  Output dir: {out_dir}")
    row(sep)

    report_path = out_dir / "evaluation_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n  Report → {report_path}")
    print(f"\nDone.  All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
