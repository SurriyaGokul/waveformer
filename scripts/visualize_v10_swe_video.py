#!/usr/bin/env python3
"""
WavePINN V10 — Publication-Quality SWE Visualisation Video
===========================================================
Generates a multi-panel, publication-ready animation comparing:

  Panel 1  — Ground Truth (Loupas velocity, 500 Hz rate, shown as step function)
  Panel 2  — WavePINN V10 Prediction (virtual 6 kHz or sparse-fill depending on mode)
  Panel 3  — Absolute Error |GT − Pred|
  Panel 4  — Lateral waveform profile at focal depth (GT vs Pred vs Error)

Two output artefacts per sample:
  (a) Full temporal sequence video (.mp4, ~8–15 fps)
  (b) Spatiotemporal (x–t) diagram: lateral position × time image (static PNG)

Usage
-----
    python scripts/visualize_v10_swe_video.py \
        --model_path SWE_PINN_outputs_wavepinn_v10_best_model.pt \
        --output_dir outputs/v10_videos \
        --mode deployment          # 'deployment' (dt=2ms) or 'sparse' (dt=24ms)

    # Single sample
    python scripts/visualize_v10_swe_video.py --sample cirs_10mmfocus_60kPa_inclusion_500Hz_20260128T163842

Physical parameters (from IQ2VEl.m)
-------------------------------------
    Axial pixel spacing  : 0.0507 mm  (λ/4 at 7.6 MHz)
    Lateral pixel spacing: 0.1013 mm
    ROI                  : 116 × 160 px (axial × lateral)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v10 import create_wavepinn_v10

# ── Physical constants ────────────────────────────────────────────────────────
AXIAL_MM_PER_PX   = 0.0507
LATERAL_MM_PER_PX = 0.1013
H_ROI, W_ROI      = 116, 160
UPSAMPLING_FACTOR  = 12
DATA_ROOT          = PROJECT_ROOT / "data" / "fullscale_velocity"
META_PATH          = DATA_ROOT / "metadata.json"

# Publication style
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 9,
    "axes.linewidth"   : 0.8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
    "figure.dpi"       : 150,
})


# =============================================================================
# Helpers (shared with evaluate_v10_swe_report.py — duplicated for independence)
# =============================================================================

def _pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    *lead, H, W = arr.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH == 0 and pW == 0:
        return arr
    pad = [(0, 0)] * len(lead) + [(0, pH), (0, pW)]
    return np.pad(arr, pad, mode="constant")


def load_model(ckpt_path: str, device: torch.device):
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = dict(ckpt["config"]["model"])
    model_cfg.pop("name", None)
    model = create_wavepinn_v10(config=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("best_val_loss", float("nan"))
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  V10 loaded — epoch={epoch}  best_5kHz_L1={loss:.5f}  params={n_p:,}")
    return model


def load_sparse_iq(sample_meta: Dict, sparse_step: int, start_frame: int = 0) -> np.ndarray:
    raw = np.load(sample_meta["original_path"]).astype(np.complex128)
    roi = sample_meta["roi"]
    win = sample_meta["window"]
    iq  = raw[roi["min_row"]:roi["max_row"]+1,
               roi["min_col"]:roi["max_col"]+1,
               win[0]:win[1]].astype(np.complex64)
    iq -= iq.mean(axis=2, keepdims=True)   # clutter subtraction
    T_win    = iq.shape[2]
    max_step = (T_win - start_frame - 1) // 4
    sparse_step = min(sparse_step, max(max_step, 1))
    indices = [start_frame + i * sparse_step for i in range(5)]
    return np.stack([iq[:, :, idx] for idx in indices], axis=0)


def load_gt(sample_meta: Dict) -> np.ndarray:
    name = sample_meta["name"]
    for split in ("val", "train", "inference"):
        p = DATA_ROOT / split / f"velocity_{name}.npy"
        if p.exists():
            return np.load(str(p)).astype(np.float32)
    raise FileNotFoundError(f"GT not found: {name}")


@torch.no_grad()
def infer(
    model: torch.nn.Module,
    sparse_iq: np.ndarray,
    dt_seconds: float,
    vel_scale: float,
    device: torch.device,
) -> np.ndarray:
    scale = float(np.percentile(np.abs(sparse_iq), 99))
    scale = max(scale, 1e-6)
    iq_n  = (sparse_iq / scale).astype(np.complex64)
    inp   = np.stack([iq_n.real, iq_n.imag], axis=0)   # (2, 5, H, W)
    inp   = _pad_to_multiple(inp)
    tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
    pred  = model(tensor, dt_seconds=dt_seconds)         # (1, 1, T, H_pad, W_pad)
    pred_np = pred[0, 0].cpu().numpy()[:, :H_ROI, :W_ROI]
    return (pred_np * vel_scale).astype(np.float32)


def _physical_axes(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    lat = (np.arange(W) - W / 2) * LATERAL_MM_PER_PX
    dep = np.arange(H) * AXIAL_MM_PER_PX
    return lat, dep


def _try_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


# =============================================================================
# Spatiotemporal (x–t) diagram
# =============================================================================

def compute_xt_diagram(
    vel_map: np.ndarray,
    depth_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Extract velocity(t, lateral) at a fixed depth row.

    Returns (T, W) float32.
    """
    if depth_idx is None:
        depth_idx = vel_map.shape[1] // 3   # upper-third focal zone
    return vel_map[:, depth_idx, :].astype(np.float32)   # (T, W)


def save_xt_diagram(
    gt_xt: np.ndarray,
    pred_xt: np.ndarray,
    sample_name: str,
    out_path: Path,
    dt_gt_ms: float,
    dt_pred_ms: float,
    focal_depth_mm: float,
    metrics: Optional[Dict] = None,
    gt_freq_label: str = "500 Hz",
):
    """
    Publication-ready x–t (spatiotemporal) diagram.
    Top row: GT. Bottom row: Prediction. Right panel: Lateral waveform at t_peak.
    """
    T_gt, W   = gt_xt.shape
    T_pred, _ = pred_xt.shape
    lat_mm    = (np.arange(W) - W / 2) * LATERAL_MM_PER_PX

    t_gt   = np.arange(T_gt)   * dt_gt_ms
    t_pred = np.arange(T_pred) * dt_pred_ms

    clim = max(float(np.abs(gt_xt).max()), 0.5)
    ext_gt   = [lat_mm[0], lat_mm[-1], t_gt[-1],   0]
    ext_pred = [lat_mm[0], lat_mm[-1], t_pred[-1], 0]

    fig = plt.figure(figsize=(10, 8), dpi=200)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.35],
        height_ratios=[1, 1],
        wspace=0.35, hspace=0.30,
        left=0.09, right=0.95, top=0.90, bottom=0.09,
    )

    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    ax_wave = fig.add_subplot(gs[:, 2])
    ax_diff = fig.add_subplot(gs[1, 1])

    kw_vel = dict(cmap="RdBu_r", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", interpolation="nearest")
    kw_err = dict(cmap="hot", vmin=0, vmax=clim,
                  aspect="auto", origin="upper", interpolation="nearest")

    im_gt   = ax_gt.imshow(gt_xt,  extent=ext_gt,   **kw_vel)
    im_pred = ax_pred.imshow(pred_xt, extent=ext_pred, **kw_vel)

    # Error: resample prediction to GT time grid for direct comparison
    n_match = min(T_gt, T_pred)
    gt_crop   = gt_xt[:n_match]
    pred_crop = pred_xt[:n_match]
    err_xt    = np.abs(gt_crop - pred_crop)
    ext_crop  = [lat_mm[0], lat_mm[-1], t_gt[n_match-1], 0]
    im_err  = ax_err.imshow(err_xt,  extent=ext_crop, **kw_err)
    # Difference (signed)
    diff_xt = pred_crop - gt_crop
    ax_diff.imshow(diff_xt, extent=ext_crop,
                   cmap="RdBu_r", vmin=-clim, vmax=clim,
                   aspect="auto", origin="upper", interpolation="nearest")

    # Labels
    for ax, title in [
        (ax_gt,   f"GT — {dt_gt_ms:.2f} ms/frame (Loupas {gt_freq_label})"),
        (ax_pred, f"V10 Pred — {dt_pred_ms:.3f} ms/frame (virtual {1/(dt_pred_ms*1e-3)/1000:.0f} kHz)"),
        (ax_err,  "|Error|  GT − Pred"),
        (ax_diff, "Signed Difference  Pred − GT"),
    ]:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel("Lateral (mm)", fontsize=8)

    ax_gt.set_ylabel("Time (ms)", fontsize=8)
    ax_pred.set_ylabel("Time (ms)", fontsize=8)

    # Colorbars
    divider = make_axes_locatable(ax_gt)
    cax1    = divider.append_axes("right", size="5%", pad=0.04)
    cb1 = fig.colorbar(im_gt, cax=cax1)
    cb1.set_label("v (mm/s)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    divider2 = make_axes_locatable(ax_err)
    cax2     = divider2.append_axes("right", size="5%", pad=0.04)
    cb2 = fig.colorbar(im_err, cax=cax2)
    cb2.set_label("|Error| (mm/s)", fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    # Lateral waveform at the time of maximum GT amplitude
    t_peak_gt   = int(np.argmax(np.abs(gt_xt).max(axis=1)))
    t_peak_pred = min(t_peak_gt, T_pred - 1)
    ax_wave.plot(gt_xt[t_peak_gt], lat_mm, color="steelblue", lw=2,
                 label=f"GT  t={t_gt[t_peak_gt]:.2f} ms")
    ax_wave.plot(pred_xt[t_peak_pred], lat_mm, color="crimson", lw=1.5, ls="--",
                 label=f"Pred  t={t_pred[t_peak_pred]:.2f} ms")
    ax_wave.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax_wave.set_xlabel("Velocity (mm/s)", fontsize=8)
    ax_wave.set_ylabel("Lateral (mm)", fontsize=8)
    ax_wave.set_title("Lateral profile\nat peak GT frame", fontsize=8, fontweight="bold")
    ax_wave.legend(fontsize=7, loc="lower right")
    ax_wave.yaxis.set_label_position("right")
    ax_wave.yaxis.tick_right()
    ax_wave.grid(alpha=0.3)

    # Overall title
    metrics_str = ""
    if metrics:
        r_a  = metrics.get("r_active", float("nan"))
        mae  = metrics.get("mae", float("nan"))
        ssim_v = metrics.get("ssim", float("nan"))
        metrics_str = f"  MAE={mae:.3f} mm/s  |  SSIM={ssim_v:.3f}  |  r_active={r_a:.3f}"

    fig.suptitle(
        f"WavePINN V10 — Spatiotemporal Velocity (x–t Diagram)\n"
        f"{sample_name}  |  Focal depth: {focal_depth_mm:.1f} mm{metrics_str}",
        fontsize=10, fontweight="bold", y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    x–t diagram → {out_path}")


# =============================================================================
# Video
# =============================================================================

def make_video(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_name: str,
    out_path: Path,
    dt_gt_ms: float,
    dt_pred_ms: float,
    fps: int = 12,
    dpi: int = 180,
    gt_freq_label: str = "500 Hz",
):
    """
    Four-panel publication video:
      [GT]  [Prediction]  [|Error|]  [Lateral waveform at focal depth]

    The GT is shown as a step function (value held constant for upsampling_factor frames).

    Parameters
    ----------
    gt       : (T_gt, H, W) mm/s — native 500 Hz GT
    pred     : (T_pred, H, W) mm/s — dense V10 prediction
    dt_gt_ms : time step of GT frames (ms)
    dt_pred_ms : time step of dense prediction frames (ms)
    """
    T_gt,   H, W = gt.shape
    T_pred, _, _ = pred.shape
    lat_mm, dep_mm = _physical_axes(H, W)
    extent = [lat_mm[0], lat_mm[-1], dep_mm[-1], dep_mm[0]]

    clim = max(float(np.abs(gt[:min(8, T_gt)]).max()), 0.5)

    focal_row = H // 3
    focal_mm  = focal_row * AXIAL_MM_PER_PX

    # Precompute GT at prediction time grid (step function)
    def gt_at_pred_frame(fi: int) -> np.ndarray:
        gi = min(fi // UPSAMPLING_FACTOR, T_gt - 1)
        return gt[gi]

    # Build figure layout
    fig = plt.figure(figsize=(14, 6.5), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        1, 5,
        width_ratios=[1, 1, 1, 0.035, 0.9],
        wspace=0.05,
        left=0.06, right=0.97, top=0.88, bottom=0.10,
    )
    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_err  = fig.add_subplot(gs[0, 2])
    ax_cb   = fig.add_subplot(gs[0, 3])
    ax_wave = fig.add_subplot(gs[0, 4])

    kw_img = dict(cmap="RdBu_r", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", extent=extent, interpolation="bilinear")
    kw_err = dict(cmap="hot", vmin=0, vmax=clim,
                  aspect="auto", origin="upper", extent=extent, interpolation="bilinear")

    # Initial frames
    gt_f0   = gt_at_pred_frame(0)
    pred_f0 = pred[0]
    err_f0  = np.abs(pred_f0 - gt_f0)

    im_gt   = ax_gt.imshow(gt_f0,   **kw_img)
    im_pred = ax_pred.imshow(pred_f0, **kw_img)
    im_err  = ax_err.imshow(err_f0,  **kw_err)

    for ax, title in [
        (ax_gt,   f"Ground Truth\n(Loupas @ {gt_freq_label})"),
        (ax_pred, f"WavePINN V10\n(virtual {1/(dt_pred_ms*1e-3)/1000:.0f} kHz)"),
        (ax_err,  "|Error|\n(mm/s)"),
    ]:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Lateral (mm)", fontsize=8)
        ax.tick_params(labelsize=7)

    ax_gt.set_ylabel("Depth (mm)", fontsize=8)
    for ax in [ax_pred, ax_err]:
        ax.set_yticks([])

    # Horizontal dashed line at focal depth
    for ax in [ax_gt, ax_pred, ax_err]:
        ax.axhline(focal_mm, color="lime", lw=0.8, ls="--", alpha=0.6)

    # Colorbar for velocity
    sm_vel = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=mcolors.Normalize(vmin=-clim, vmax=clim)
    )
    sm_vel.set_array([])
    cb = fig.colorbar(sm_vel, cax=ax_cb)
    cb.set_label("Particle velocity (mm/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Waveform panel (lateral profile at focal depth)
    ax_wave.set_facecolor("white")
    ax_wave.set_xlim(-clim * 1.1, clim * 1.1)
    ax_wave.set_ylim(lat_mm[0], lat_mm[-1])
    ax_wave.set_xlabel("Velocity (mm/s)", fontsize=8)
    ax_wave.set_ylabel("Lateral (mm)", fontsize=8)
    ax_wave.yaxis.tick_right()
    ax_wave.yaxis.set_label_position("right")
    ax_wave.tick_params(labelsize=7)
    ax_wave.grid(alpha=0.3)
    ax_wave.set_title("Lateral profile\n@ focal depth", fontsize=9, fontweight="bold", pad=4)
    ax_wave.axvline(0, color="gray", lw=0.5, alpha=0.4)

    line_gt,   = ax_wave.plot(gt_f0[focal_row],   lat_mm, "steelblue", lw=2,   label="GT")
    line_pred, = ax_wave.plot(pred_f0[focal_row],  lat_mm, "crimson",   lw=1.5, ls="--", label="V10")
    ax_wave.legend(fontsize=8, loc="lower right", handlelength=1.5)

    sup = fig.suptitle("", fontsize=10, fontweight="bold", y=0.97)

    def _update(fi: int):
        gt_f   = gt_at_pred_frame(fi)
        pred_f = pred[min(fi, T_pred - 1)]
        err_f  = np.abs(pred_f - gt_f)

        im_gt.set_data(gt_f)
        im_pred.set_data(pred_f)
        im_err.set_data(err_f)
        line_gt.set_xdata(gt_f[focal_row])
        line_pred.set_xdata(pred_f[focal_row])

        t_ms = fi * dt_pred_ms
        gt_frame_idx = min(fi // UPSAMPLING_FACTOR, T_gt - 1)
        sup.set_text(
            f"WavePINN V10 — {sample_name}\n"
            f"t = {t_ms:.3f} ms  |  dense frame {fi+1}/{T_pred}  |  "
            f"GT frame {gt_frame_idx+1}/{T_gt}"
        )
        return im_gt, im_pred, im_err, line_gt, line_pred, sup

    anim = FuncAnimation(fig, _update, frames=T_pred, blit=False, interval=max(1000 // fps, 40))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _try_ffmpeg():
        save_path = out_path.with_suffix(".mp4")
        writer = FFMpegWriter(
            fps=fps,
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "slow",
                "-movflags", "+faststart",
            ],
        )
        anim.save(str(save_path), writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": "white"})
        print(f"    Video (MP4) → {save_path}")
    else:
        save_path = out_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer,
                  savefig_kwargs={"facecolor": "white"})
        print(f"    Video (GIF) → {save_path}")

    plt.close(fig)
    return save_path


# =============================================================================
# Multi-panel comparison poster (static, publication-ready)
# =============================================================================

def save_poster_figure(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_name: str,
    out_path: Path,
    dt_gt_ms: float,
    dt_pred_ms: float,
    ref_cs: Optional[float] = None,
    tissue_label: str = "",
    gt_freq_label: str = "500 Hz",
):
    """
    Comprehensive single-figure poster:
      Row 1 : Selected GT frames (4 timepoints)
      Row 2 : Corresponding V10 frames (4 timepoints — can be sub-frames)
      Row 3 : |Error| at same timepoints
      Row 4 : x–t diagrams (GT and Pred side-by-side)
      Row 5 : Lateral waveform at 4 timepoints (GT vs Pred overlay)
    """
    T_gt,   H, W = gt.shape
    T_pred, _, _ = pred.shape
    lat_mm, dep_mm = _physical_axes(H, W)
    extent = [lat_mm[0], lat_mm[-1], dep_mm[-1], dep_mm[0]]

    clim = max(float(np.abs(gt).max()), 0.5)

    # Select 4 representative timepoints in the active GT window
    frame_stds = gt.std(axis=(1, 2))
    active_ids = np.where(frame_stds > 0.05)[0]
    if len(active_ids) >= 4:
        show_gt_frames = active_ids[np.linspace(0, len(active_ids) - 1, 4, dtype=int)]
    else:
        show_gt_frames = np.arange(min(4, T_gt))
    show_gt_frames = list(show_gt_frames)

    # Corresponding prediction frames (map GT timepoint → dense index)
    show_pred_frames = [min(gf * UPSAMPLING_FACTOR, T_pred - 1) for gf in show_gt_frames]

    focal_row = H // 3

    # x–t at focal depth
    xt_gt   = gt[:, focal_row, :].astype(np.float32)   # (T_gt, W)
    xt_pred = pred[:, focal_row, :].astype(np.float32) # (T_pred, W)
    t_gt    = np.arange(T_gt)   * dt_gt_ms
    t_pred  = np.arange(T_pred) * dt_pred_ms
    xt_ext_gt   = [lat_mm[0], lat_mm[-1], t_gt[-1],   0]
    xt_ext_pred = [lat_mm[0], lat_mm[-1], t_pred[-1], 0]

    n_cols = 4
    fig = plt.figure(figsize=(4 * n_cols + 0.8, 20), dpi=200)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(
        5, n_cols + 1,
        width_ratios=[1] * n_cols + [0.06],
        height_ratios=[1, 1, 1, 1.4, 1],
        wspace=0.04, hspace=0.22,
        left=0.07, right=0.97, top=0.95, bottom=0.04,
    )

    rows_data = [
        (f"Ground Truth (Loupas {gt_freq_label})", show_gt_frames, gt,   dt_gt_ms),
        (f"WavePINN V10 (virtual {1/(dt_pred_ms*1e-3)/1000:.0f} kHz)", show_pred_frames, pred, dt_pred_ms),
        ("|Error| (mm/s)", show_gt_frames, None, dt_gt_ms),
    ]

    kw_vel = dict(cmap="RdBu_r", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", extent=extent, interpolation="bilinear")
    kw_err = dict(cmap="hot", vmin=0, vmax=clim,
                  aspect="auto", origin="upper", extent=extent, interpolation="bilinear")

    # ── Rows 0–2: spatial frames ──────────────────────────────────────────────
    im_vel = None
    im_err_ax = None
    for row_idx, (row_title, frame_ids, vol, dt_ms_row) in enumerate(rows_data):
        for col, fi in enumerate(frame_ids):
            ax = fig.add_subplot(outer[row_idx, col])
            t_ms = fi * dt_ms_row

            if row_idx == 2:      # Error row
                gt_fi  = gt[fi]   if fi < T_gt   else gt[-1]
                pr_fi  = pred[fi] if fi < T_pred else pred[-1]
                img    = np.abs(gt_fi - pr_fi)
                im_err_ax = ax.imshow(img, **kw_err)
                ax.set_title(f"t = {t_ms:.2f} ms", fontsize=8, pad=2)
            else:
                fr = vol[fi] if fi < vol.shape[0] else vol[-1]
                im_vel = ax.imshow(fr, **kw_vel)
                ax.set_title(f"t = {t_ms:.2f} ms", fontsize=8, pad=2)

            ax.set_xlabel("Lateral (mm)", fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel("Depth (mm)", fontsize=7)
                ax.annotate(row_title, xy=(-0.22, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=8, fontweight="bold")
            else:
                ax.set_yticks([])
            ax.axhline(focal_row * AXIAL_MM_PER_PX, color="lime",
                       lw=0.7, ls="--", alpha=0.5)

    # Colorbars (right of col n_cols)
    ax_cb_vel = fig.add_subplot(outer[0:2, n_cols])
    ax_cb_err = fig.add_subplot(outer[2, n_cols])

    if im_vel:
        sm_vel = plt.cm.ScalarMappable(cmap="RdBu_r",
                                        norm=mcolors.Normalize(-clim, clim))
        sm_vel.set_array([])
        cb1 = fig.colorbar(sm_vel, cax=ax_cb_vel)
        cb1.set_label("v (mm/s)", fontsize=7)
        cb1.ax.tick_params(labelsize=6)

    if im_err_ax:
        sm_err = plt.cm.ScalarMappable(cmap="hot",
                                        norm=mcolors.Normalize(0, clim))
        sm_err.set_array([])
        cb2 = fig.colorbar(sm_err, cax=ax_cb_err)
        cb2.set_label("|Error| (mm/s)", fontsize=7)
        cb2.ax.tick_params(labelsize=6)

    # ── Row 3: x–t diagrams ───────────────────────────────────────────────────
    ax_xt_gt   = fig.add_subplot(outer[3, 0:2])
    ax_xt_pred = fig.add_subplot(outer[3, 2:4])
    ax_xt_cb   = fig.add_subplot(outer[3, n_cols])

    im_xt_gt   = ax_xt_gt.imshow(xt_gt, cmap="RdBu_r", vmin=-clim, vmax=clim,
                                   aspect="auto", origin="upper", extent=xt_ext_gt,
                                   interpolation="nearest")
    im_xt_pred = ax_xt_pred.imshow(xt_pred, cmap="RdBu_r", vmin=-clim, vmax=clim,
                                    aspect="auto", origin="upper", extent=xt_ext_pred,
                                    interpolation="nearest")

    ax_xt_gt.set_title(f"Spatiotemporal (x–t) — GT @ depth {focal_row * AXIAL_MM_PER_PX:.1f} mm",
                        fontsize=8, fontweight="bold")
    ax_xt_pred.set_title(f"Spatiotemporal (x–t) — V10 @ depth {focal_row * AXIAL_MM_PER_PX:.1f} mm",
                          fontsize=8, fontweight="bold")

    for ax in [ax_xt_gt, ax_xt_pred]:
        ax.set_xlabel("Lateral (mm)", fontsize=8)
        ax.set_ylabel("Time (ms)", fontsize=8)
        ax.tick_params(labelsize=7)

    sm_xt = plt.cm.ScalarMappable(cmap="RdBu_r",
                                   norm=mcolors.Normalize(-clim, clim))
    sm_xt.set_array([])
    cb_xt = fig.colorbar(sm_xt, cax=ax_xt_cb)
    cb_xt.set_label("v (mm/s)", fontsize=7)
    cb_xt.ax.tick_params(labelsize=6)

    # ── Row 4: Lateral waveform at 4 timepoints ───────────────────────────────
    for col, (gf, pf) in enumerate(zip(show_gt_frames, show_pred_frames)):
        ax = fig.add_subplot(outer[4, col])
        gt_row  = gt[gf, focal_row, :]   if gf < T_gt   else gt[-1, focal_row, :]
        pr_row  = pred[pf, focal_row, :] if pf < T_pred else pred[-1, focal_row, :]
        ax.plot(lat_mm, gt_row,  color="steelblue", lw=1.8, label="GT")
        ax.plot(lat_mm, pr_row,  color="crimson",   lw=1.2, ls="--", label="V10")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
        ax.set_xlim(lat_mm[0], lat_mm[-1])
        ax.set_ylim(-clim * 1.1, clim * 1.1)
        ax.set_xlabel("Lateral (mm)", fontsize=7)
        ax.set_title(f"t = {gf * dt_gt_ms:.2f} ms", fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("v (mm/s)", fontsize=7)
            ax.annotate("Lateral waveform\n@ focal depth",
                        xy=(-0.22, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center",
                        fontsize=8, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right", handlelength=1.2)

    # Overall title
    ref_str = f"  |  Reference cs = {ref_cs:.2f} m/s" if ref_cs else ""
    fig.suptitle(
        f"WavePINN V10 — Comprehensive SWE Evaluation\n"
        f"{sample_name}  {tissue_label}{ref_str}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Poster figure → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V10 SWE publication visualisation")
    parser.add_argument("--model_path", default="SWE_PINN_outputs_wavepinn_v10_best_model.pt")
    parser.add_argument("--output_dir", default="outputs/v10_videos")
    parser.add_argument("--mode", choices=["deployment", "sparse"], default="deployment",
                        help="'deployment' → dt=2ms (6kHz)  |  'sparse' → dt=24ms")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample", type=str, default=None,
                        help="Process only this sample name")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--no_poster", action="store_true")
    parser.add_argument("--freq_filter", type=str, default=None,
                        help="Filter val samples by frequency (e.g. '5kHz', '500Hz'). Default: all val samples.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print("WavePINN V10 — Publication-Quality SWE Visualisation")
    print(f"{'='*72}")
    print(f"  Device : {device}  |  Mode : {args.mode}  |  FPS : {args.fps}")

    model = load_model(args.model_path, device)

    with open(META_PATH) as f:
        meta = json.load(f)

    all_val = meta.get("val", [])
    if args.freq_filter:
        val_samples = [s for s in all_val if s.get("frequency") == args.freq_filter]
        print(f"  Frequency filter: {args.freq_filter}  ({len(val_samples)} samples)")
    else:
        val_samples = all_val
        print(f"  Frequency filter: none  ({len(val_samples)} samples)")

    if args.sample:
        val_samples = [s for s in val_samples if s["name"] == args.sample]
        if not val_samples:
            raise ValueError(f"Sample '{args.sample}' not found in filtered val set")

    if args.max_samples:
        val_samples = val_samples[:args.max_samples]

    print(f"  Processing: {len(val_samples)} samples")
    print_separator = lambda: print("─" * 72)

    for i, smeta in enumerate(val_samples):
        name  = smeta["name"]
        print(f"\n[{i+1}/{len(val_samples)}] {name}")

        # Load GT
        try:
            gt = load_gt(smeta)[:, :H_ROI, :W_ROI]   # (T, H, W) mm/s
        except Exception as e:
            print(f"  SKIP (no GT): {e}")
            continue

        vel_scale = float(np.percentile(np.abs(gt), 99))
        vel_scale = max(vel_scale, 1.0)
        dt_gt_ms     = smeta["frame_time_s"] * 1e3
        freq_str     = smeta.get("frequency", "unknown")
        gt_freq_label = freq_str  # e.g. "500Hz" or "5kHz"
        prf_khz      = 1.0 / smeta["frame_time_s"] / 1000.0  # e.g. 0.5 for 500Hz, 5.0 for 5kHz

        # Load IQ & infer
        try:
            if args.mode == "deployment":
                sparse_iq = load_sparse_iq(smeta, sparse_step=1)
                dt_in     = smeta["frame_time_s"]
                dt_out_ms = dt_in * 1e3 / UPSAMPLING_FACTOR
                virtual_khz = 1.0 / (dt_out_ms * 1e-3) / 1000.0
                mode_label = f"Deployment (dt={dt_in*1e3:.2f} ms → virtual {virtual_khz:.0f} kHz)"
            else:
                sp = smeta.get("sparse_step", 12)
                sparse_iq = load_sparse_iq(smeta, sparse_step=sp)
                dt_in     = sp * smeta["frame_time_s"]
                dt_out_ms = dt_in * 1e3 / UPSAMPLING_FACTOR
                virtual_khz = 1.0 / (dt_out_ms * 1e-3) / 1000.0
                mode_label = f"Sparse (dt={dt_in*1e3:.1f} ms → virtual {virtual_khz:.0f} kHz)"

            print(f"  Mode: {mode_label}")
            pred = infer(model, sparse_iq, dt_in, vel_scale, device)
        except Exception as e:
            print(f"  SKIP (inference failed): {e}")
            continue

        tissue = smeta.get("tissue_properties", {})
        ref_cs = tissue.get("estimated_shear_speed_m_s", None)
        tissue_label = f"{tissue.get('material', '')} {smeta.get('source', '')}".strip()

        # x–t diagram
        focal_row = H_ROI // 3
        focal_mm  = focal_row * AXIAL_MM_PER_PX
        xt_gt     = compute_xt_diagram(gt, focal_row)
        xt_pred   = compute_xt_diagram(pred, focal_row)

        xt_path = out_dir / "xt_diagrams" / f"xt_{args.mode}_{name}.png"
        save_xt_diagram(
            gt_xt          = xt_gt,
            pred_xt        = xt_pred,
            sample_name    = name,
            out_path       = xt_path,
            dt_gt_ms       = dt_gt_ms,
            dt_pred_ms     = dt_out_ms,
            focal_depth_mm = focal_mm,
            gt_freq_label  = gt_freq_label,
        )

        # Poster
        if not args.no_poster:
            poster_path = out_dir / "posters" / f"poster_{args.mode}_{name}.png"
            save_poster_figure(
                gt            = gt,
                pred          = pred,
                sample_name   = name,
                out_path      = poster_path,
                dt_gt_ms      = dt_gt_ms,
                dt_pred_ms    = dt_out_ms,
                ref_cs        = ref_cs,
                tissue_label  = tissue_label,
                gt_freq_label = gt_freq_label,
            )

        # Video
        if not args.no_video:
            vid_path = out_dir / "videos" / f"v10_{args.mode}_{name}"
            make_video(
                gt            = gt,
                pred          = pred,
                sample_name   = name,
                out_path      = vid_path,
                dt_gt_ms      = dt_gt_ms,
                dt_pred_ms    = dt_out_ms,
                fps           = args.fps,
                gt_freq_label = gt_freq_label,
            )

    print(f"\n{'='*72}")
    print(f"Done. Outputs → {out_dir}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
