#!/usr/bin/env python3
"""
WavePINN V10 — SWE Performance Evaluation Report
=================================================
Inferences the V10 best model on the 500 Hz validation set and produces a
scientifically rigorous, SWE-focused performance report.

Evaluation strategy
-------------------
Two complementary inference modes are run on every val-500Hz sample:

  Mode A — "Sparse reconstruction" (dt = 24 ms)
      Input : 5 sparse IQ frames at sparse_step=12 (every 12th 500 Hz frame)
              dt_seconds = 12 × 0.002 = 0.024 s
      Output: 49 dense frames at 2 ms intervals (virtual 500 Hz, fill-in)
      GT    : pre-computed 49-frame Loupas velocity (2 ms spacing)
      Note  : dt is 10× outside training range — OOD evaluation.

  Mode B — "Deployment" (dt = 2 ms)
      Input : 5 CONSECUTIVE 500 Hz IQ frames
              dt_seconds = 0.002 s   (matches training step=10 from 5kHz data)
      Output: 49 dense frames at 0.167 ms intervals (virtual ~6 kHz)
      GT    : block-averaged Loupas from the same 5-frame window
      Note  : This is the intended deployment scenario for clinical SWE.

SWE-specific metrics computed
------------------------------
  • MAE, RMSE  (mm/s, in physical units after denormalisation)
  • SSIM       (structural similarity, full-volume)
  • PSNR       (dB)
  • Pearson r  — full-volume and active-frames only
  • ToA accuracy — time-of-arrival curve via cross-correlation (Mode B)
  • Shear wave speed cs (m/s) — linear regression on ToA vs lateral position
  • cs_MAE (m/s) vs reference cs from sample metadata

Usage
-----
    python scripts/evaluate_v10_swe_report.py \
        --model_path SWE_PINN_outputs_wavepinn_v10_best_model.pt \
        --output_dir outputs/v10_swe_report

    # Skip the expensive Mode-B deployment evaluation
    python scripts/evaluate_v10_swe_report.py --skip_deployment
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v10 import create_wavepinn_v10

# ── Physical constants ────────────────────────────────────────────────────────
AXIAL_MM_PER_PX   = 0.0507   # mm / axial pixel  (λ/4 @ 7.6 MHz)
LATERAL_MM_PER_PX = 0.1013   # mm / lateral pixel
H_ROI, W_ROI      = 116, 160
UPSAMPLING_FACTOR  = 12
DATA_ROOT          = PROJECT_ROOT / "data" / "fullscale_velocity"
META_PATH          = DATA_ROOT / "metadata.json"


# =============================================================================
# Model loading
# =============================================================================

def load_v10_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    epoch     = ckpt.get("epoch", "?")
    best_loss = ckpt.get("best_val_loss", float("nan"))

    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model = create_wavepinn_v10(config=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  WavePINNv10 loaded  epoch={epoch}  best_5kHz_L1={best_loss:.5f}  "
          f"params={n_params:,}")
    return model, cfg


# =============================================================================
# Data helpers
# =============================================================================

def _pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    *lead, H, W = arr.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH == 0 and pW == 0:
        return arr
    pad = [(0, 0)] * len(lead) + [(0, pH), (0, pW)]
    return np.pad(arr, pad, mode="constant")


def load_sparse_iq(
    sample_meta: Dict,
    sparse_step: int,
    start_frame: int = 0,
) -> np.ndarray:
    """
    Load 5 sparse IQ frames from the original file.

    Returns (5, H_ROI, W_ROI) complex64 after clutter subtraction.
    """
    raw  = np.load(sample_meta["original_path"])          # (H_full, W_full, T)
    roi  = sample_meta["roi"]
    win  = sample_meta["window"]

    iq_win = raw[
        roi["min_row"] : roi["max_row"] + 1,
        roi["min_col"] : roi["max_col"] + 1,
        win[0] : win[1],
    ].astype(np.complex64)                                 # (H, W, T_win)

    # Simple clutter subtraction (mean over time window)
    iq_win = iq_win - iq_win.mean(axis=2, keepdims=True)

    T_win = iq_win.shape[2]
    frames_needed = start_frame + (4 * sparse_step) + 1

    # If the configured sparse_step doesn't fit, auto-reduce to the largest
    # step that fits within the available window.
    if frames_needed > T_win:
        max_step = (T_win - start_frame - 1) // 4
        if max_step < 1:
            raise ValueError(
                f"Window has only {T_win} frames from start_frame={start_frame}; "
                f"too short for even step=1."
            )
        sparse_step = max_step

    indices = [start_frame + i * sparse_step for i in range(5)]
    sparse  = np.stack([iq_win[:, :, idx] for idx in indices], axis=0)  # (5, H, W)
    return sparse, sparse_step  # return actual step used


def iq_to_model_input(
    sparse_iq: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Normalise sparse IQ and return (1, 2, 5, H_pad, W_pad) tensor + iq_scale.
    """
    iq_scale = float(np.percentile(np.abs(sparse_iq), 99))
    iq_scale = max(iq_scale, 1e-6)
    iq_norm  = (sparse_iq / iq_scale).astype(np.complex64)  # (5, H, W)

    real = iq_norm.real  # (5, H, W)
    imag = iq_norm.imag
    inp  = np.stack([real, imag], axis=0)          # (2, 5, H, W)
    inp  = _pad_to_multiple(inp)                   # (2, 5, H_pad, W_pad)

    tensor = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, 2, 5, H, W)
    return tensor, iq_scale


def load_gt_velocity(sample_meta: Dict) -> np.ndarray:
    """Load pre-computed Loupas velocity (T, H, W) float32, mm/s."""
    name = sample_meta["name"]
    for split in ("val", "train", "inference"):
        p = DATA_ROOT / split / f"velocity_{name}.npy"
        if p.exists():
            return np.load(str(p)).astype(np.float32)  # (T, H, W)
    raise FileNotFoundError(f"GT velocity not found for {name}")


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def infer(
    model: torch.nn.Module,
    sparse_iq: np.ndarray,
    dt_seconds: float,
    vel_scale: float,
    device: torch.device,
) -> np.ndarray:
    """
    Returns dense velocity prediction in mm/s, shape (T_dense, H_ROI, W).
    """
    inp_tensor, _iq_scale = iq_to_model_input(sparse_iq, device)

    with torch.no_grad():
        pred = model(inp_tensor, dt_seconds=dt_seconds)   # (1, 1, T, H_pad, W_pad)

    pred_np = pred[0, 0].cpu().numpy()   # (T, H_pad, W_pad)
    pred_mm = pred_np[:, :H_ROI, :W_ROI] * vel_scale
    return pred_mm.astype(np.float32)


# =============================================================================
# SWE metrics
# =============================================================================

def compute_pixel_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """
    All pixel-level metrics in physical units (mm/s) where applicable.

    Parameters
    ----------
    gt, pred : (T, H, W) float32, mm/s

    Returns
    -------
    dict with keys: mae, rmse, psnr, ssim, r_full, r_active, n_active_frames
    """
    assert gt.shape == pred.shape, f"Shape mismatch: gt={gt.shape} pred={pred.shape}"

    diff  = pred - gt
    mae   = float(np.abs(diff).mean())
    rmse  = float(np.sqrt((diff ** 2).mean()))

    # PSNR relative to GT peak-to-peak range
    gt_range = float(gt.max() - gt.min()) + 1e-12
    psnr = float(20 * np.log10(gt_range / (rmse + 1e-12)))

    # SSIM: average over frames, computed in normalised space [-1, 1]
    gt_range_ssim = max(float(np.abs(gt).max()) * 2, 1e-6)
    ssim_vals = []
    for t in range(min(gt.shape[0], pred.shape[0])):
        g = gt[t] / gt_range_ssim
        p = pred[t] / gt_range_ssim
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = ssim(g, p, data_range=1.0)
            ssim_vals.append(float(s))
        except Exception:
            pass
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else float("nan")

    # Pearson r — full volume
    gf, pf = gt.flatten(), pred.flatten()
    if gf.std() > 1e-8 and pf.std() > 1e-8:
        r_full = float(pearsonr(gf, pf)[0])
    else:
        r_full = float("nan")

    # Pearson r — active frames (|GT| std > 0.1 mm/s)
    frame_stds  = gt.std(axis=(1, 2))
    active_mask = frame_stds > 0.1
    n_active    = int(active_mask.sum())
    if n_active > 0:
        ga = gt[active_mask].flatten()
        pa = pred[active_mask].flatten()
        if ga.std() > 1e-8 and pa.std() > 1e-8:
            r_active = float(pearsonr(ga, pa)[0])
        else:
            r_active = float("nan")
    else:
        r_active = float("nan")

    return dict(
        mae=mae,
        rmse=rmse,
        psnr=psnr,
        ssim=mean_ssim,
        r_full=r_full,
        r_active=r_active,
        n_active_frames=n_active,
    )


def estimate_shear_wave_speed(
    vel_map: np.ndarray,
    dt_s: float,
    dx_mm: float = LATERAL_MM_PER_PX,
    min_snr: float = 0.15,
    depth_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Estimate shear wave speed via lateral time-of-arrival (ToA) analysis.

    Method: for each depth row in [depth_range], find the lateral position
    of maximum |velocity| as a function of time (ToA via argmax).  Linear
    regression of (lateral_mm, ToA_ms) gives cs = 1 / slope.

    Parameters
    ----------
    vel_map : (T, H, W) mm/s
    dt_s    : time step between output frames in seconds
    dx_mm   : lateral pixel spacing in mm
    min_snr : minimum per-frame std (mm/s) required to include the frame

    Returns
    -------
    dict with cs_ms (m/s), r2, n_points, toa_ms (W,), lateral_mm (W,)
    """
    T, H, W = vel_map.shape
    dt_ms = dt_s * 1e3

    if depth_range is None:
        d0, d1 = H // 4, 3 * H // 4
    else:
        d0, d1 = depth_range

    lateral_mm = np.arange(W) * dx_mm

    # Collect ToA estimates per lateral column (averaged over depth range)
    toa_list    = []
    lateral_list = []
    amplitude_list = []

    for w in range(W):
        # Depth-averaged waveform for this lateral column
        waveform = vel_map[:, d0:d1, w].mean(axis=1)  # (T,)
        if np.std(waveform) < min_snr:
            continue
        toa_idx = int(np.argmax(np.abs(waveform)))
        toa_list.append(toa_idx * dt_ms)
        lateral_list.append(lateral_mm[w])
        amplitude_list.append(float(np.abs(waveform).max()))

    if len(toa_list) < 5:
        return dict(cs_ms=float("nan"), r2=float("nan"), n_points=len(toa_list))

    toa_arr = np.array(toa_list)
    lat_arr = np.array(lateral_list)
    amp_arr = np.array(amplitude_list)

    # Amplitude-weighted linear regression: ToA = offset + (1/cs) * lateral
    w_arr = amp_arr / amp_arr.sum()
    lat_c = np.average(lat_arr, weights=w_arr)
    toa_c = np.average(toa_arr, weights=w_arr)

    num = np.sum(w_arr * (lat_arr - lat_c) * (toa_arr - toa_c))
    den = np.sum(w_arr * (lat_arr - lat_c) ** 2)

    if abs(den) < 1e-12:
        return dict(cs_ms=float("nan"), r2=float("nan"), n_points=len(toa_list))

    slope = num / den                    # ms/mm
    cs_ms = abs(1.0 / (slope * 1e-3))   # m/s  (slope: ms/mm → 1/slope: mm/ms → ×1e-3 → m/s)

    # R² of the linear fit
    toa_pred = toa_c + slope * (lat_arr - lat_c)
    ss_res   = np.sum(w_arr * (toa_arr - toa_pred) ** 2)
    ss_tot   = np.sum(w_arr * (toa_arr - toa_c) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    return dict(
        cs_ms   = float(cs_ms),
        r2      = float(r2),
        n_points = len(toa_list),
        toa_ms  = toa_arr,
        lateral_mm = lat_arr,
        amplitude  = amp_arr,
        slope_ms_mm = float(slope),
    )


def compute_block_averaged_metrics(
    gt_500hz: np.ndarray,
    pred_dense: np.ndarray,
    upsampling_factor: int = UPSAMPLING_FACTOR,
) -> Dict[str, float]:
    """
    Temporal block-averaging for comparing 6kHz predictions to 500Hz GT.

    For each 500Hz GT frame i, average the upsampling_factor corresponding
    dense prediction frames and compare.
    """
    T_gt   = gt_500hz.shape[0]
    T_pred = pred_dense.shape[0]
    n_blocks = min(T_gt, T_pred // upsampling_factor)

    pred_blocks = []
    for k in range(n_blocks):
        block = pred_dense[k * upsampling_factor : (k + 1) * upsampling_factor]
        pred_blocks.append(block.mean(axis=0))
    pred_blocks = np.stack(pred_blocks, axis=0)  # (n_blocks, H, W)

    gt_crop = gt_500hz[:n_blocks]
    return compute_pixel_metrics(gt_crop, pred_blocks)


# =============================================================================
# Reporting
# =============================================================================

def print_separator(char: str = "─", width: int = 72):
    print(char * width)


def format_metric(name: str, val: float, unit: str = "") -> str:
    if np.isnan(val):
        return f"  {name:<30s} : n/a"
    if unit:
        return f"  {name:<30s} : {val:.4f} {unit}"
    return f"  {name:<30s} : {val:.4f}"


def print_sample_report(
    name: str,
    mode_a: Optional[Dict],
    mode_b: Optional[Dict],
    tissue_props: Optional[Dict],
):
    print(f"\n  Sample : {name}")
    if tissue_props:
        ref_cs = tissue_props.get("estimated_shear_speed_m_s", float("nan"))
        mat    = tissue_props.get("material", "?")
        print(f"  Tissue : {mat}  |  Reference cs = {ref_cs:.2f} m/s")

    if mode_a:
        print("  ── Mode A (Sparse reconstruction, dt=24 ms, in-window GT) ──")
        for k in ["mae", "rmse", "psnr", "ssim", "r_full", "r_active"]:
            units = {"mae": "mm/s", "rmse": "mm/s", "psnr": "dB"}.get(k, "")
            print(format_metric(k, mode_a.get(k, float("nan")), units))

    if mode_b:
        print("  ── Mode B (Deployment, dt=2 ms, virtual 6 kHz) ─────────────")
        for k in ["mae", "rmse", "psnr", "ssim", "r_full", "r_active"]:
            units = {"mae": "mm/s", "rmse": "mm/s", "psnr": "dB"}.get(k, "")
            print(format_metric(k, mode_b.get(k, float("nan")), units))
        cs = mode_b.get("cs_ms", float("nan"))
        cs_r2 = mode_b.get("cs_r2", float("nan"))
        print(format_metric("cs (ToA, predicted)", cs, "m/s"))
        print(format_metric("cs R² of ToA fit", cs_r2))
        if tissue_props:
            ref_cs = tissue_props.get("estimated_shear_speed_m_s", float("nan"))
            cs_err = abs(cs - ref_cs) if not np.isnan(cs) else float("nan")
            print(format_metric("cs_error vs reference", cs_err, "m/s"))


# =============================================================================
# Publication figures
# =============================================================================

CMAP_VEL = "RdBu_r"
CMAP_ERR = "hot"


def _physical_extent(H: int, W: int) -> List[float]:
    lat = (np.arange(W) - W / 2) * LATERAL_MM_PER_PX
    dep = np.arange(H) * AXIAL_MM_PER_PX
    return [lat[0], lat[-1], dep[-1], dep[0]]


def save_static_comparison_figure(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_name: str,
    out_path: Path,
    dt_out_ms: float,
    mode_label: str,
    metrics: Dict[str, float],
    frame_indices: Optional[List[int]] = None,
):
    """
    Publication-ready static grid: GT (top row) vs Prediction (bottom row)
    for selected frames, plus an error row.
    """
    T, H, W = gt.shape
    if frame_indices is None:
        # Show 6 evenly spaced active frames
        frame_stds = gt.std(axis=(1, 2))
        active_ids = np.where(frame_stds > 0.05)[0]
        if len(active_ids) >= 6:
            frame_indices = list(active_ids[np.linspace(0, len(active_ids)-1, 6, dtype=int)])
        else:
            frame_indices = list(range(min(6, T)))

    n_frames = len(frame_indices)
    extent   = _physical_extent(H, W)
    clim     = max(float(np.abs(gt[frame_indices]).max()), 0.5)

    fig = plt.figure(figsize=(3.2 * n_frames + 0.6, 10), dpi=150)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(
        3, n_frames + 1,
        width_ratios=[1.0] * n_frames + [0.05],
        wspace=0.04, hspace=0.08,
        left=0.07, right=0.96, top=0.91, bottom=0.06,
    )

    row_labels = [
        "Ground Truth (Loupas velocity)",
        f"WavePINN V10 Prediction ({mode_label})",
        "Absolute Error |GT − Pred|",
    ]
    axes_rows = []
    for row in range(3):
        row_axes = [fig.add_subplot(outer[row, col]) for col in range(n_frames)]
        axes_rows.append(row_axes)

    ax_cb_vel = fig.add_subplot(outer[0:2, n_frames])
    ax_cb_err = fig.add_subplot(outer[2, n_frames])

    kw_vel = dict(cmap=CMAP_VEL, vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", extent=extent, interpolation="bilinear")
    kw_err = dict(cmap=CMAP_ERR, vmin=0, vmax=clim,
                  aspect="auto", origin="upper", extent=extent, interpolation="bilinear")

    for col, fi in enumerate(frame_indices):
        t_ms = fi * dt_out_ms
        gt_f   = gt[fi]
        pred_f = pred[fi]
        err_f  = np.abs(gt_f - pred_f)

        # GT row
        ax = axes_rows[0][col]
        im_v = ax.imshow(gt_f, **kw_vel)
        ax.set_title(f"t = {t_ms:.2f} ms", fontsize=9, fontweight="bold", pad=3)
        if col == 0:
            ax.set_ylabel("Depth (mm)", fontsize=8)
        ax.set_xticks([])
        if col > 0:
            ax.set_yticks([])

        # Prediction row
        ax2 = axes_rows[1][col]
        ax2.imshow(pred_f, **kw_vel)
        ax2.set_xticks([])
        if col == 0:
            ax2.set_ylabel("Depth (mm)", fontsize=8)
        if col > 0:
            ax2.set_yticks([])

        # Error row
        ax3 = axes_rows[2][col]
        ax3.imshow(err_f, **kw_err)
        ax3.set_xlabel("Lateral (mm)", fontsize=8)
        if col == 0:
            ax3.set_ylabel("Depth (mm)", fontsize=8)
        if col > 0:
            ax3.set_yticks([])

    # Row labels (rotated text on left)
    for row, label in enumerate(row_labels):
        axes_rows[row][0].annotate(
            label, xy=(-0.18, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=8, fontweight="bold",
        )

    # Colorbars
    sm_vel = plt.cm.ScalarMappable(cmap=CMAP_VEL,
                                   norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm_vel.set_array([])
    cb_vel = fig.colorbar(sm_vel, cax=ax_cb_vel)
    cb_vel.set_label("Particle velocity (mm/s)", fontsize=8)
    cb_vel.ax.tick_params(labelsize=7)

    sm_err = plt.cm.ScalarMappable(cmap=CMAP_ERR,
                                   norm=plt.Normalize(vmin=0, vmax=clim))
    sm_err.set_array([])
    cb_err = fig.colorbar(sm_err, cax=ax_cb_err)
    cb_err.set_label("|Error| (mm/s)", fontsize=8)
    cb_err.ax.tick_params(labelsize=7)

    # Title with metrics
    r_a  = metrics.get("r_active", float("nan"))
    mae  = metrics.get("mae", float("nan"))
    ssim_v = metrics.get("ssim", float("nan"))
    fig.suptitle(
        f"WavePINN V10 — {mode_label}\n"
        f"{sample_name}\n"
        f"MAE = {mae:.3f} mm/s | SSIM = {ssim_v:.3f} | r_active = {r_a:.3f}",
        fontsize=10, fontweight="bold", y=0.99,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved figure → {out_path}")


def save_toa_cs_figure(
    toa_result: Dict,
    sample_name: str,
    out_path: Path,
    ref_cs: Optional[float] = None,
):
    """Publication-ready ToA plot with cs regression line."""
    if "toa_ms" not in toa_result or np.isnan(toa_result.get("cs_ms", float("nan"))):
        return

    lat  = toa_result["lateral_mm"]
    toa  = toa_result["toa_ms"]
    amp  = toa_result.get("amplitude", np.ones_like(lat))
    cs   = toa_result["cs_ms"]
    r2   = toa_result["r2"]
    slope = toa_result["slope_ms_mm"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor("white")

    sc = ax.scatter(lat, toa, c=amp, cmap="plasma", s=20, alpha=0.7,
                    zorder=3, label="ToA (argmax |v|)")

    # Regression line
    lat_lin = np.linspace(lat.min(), lat.max(), 200)
    toa_lin = toa.mean() + slope * (lat_lin - lat.mean())
    ax.plot(lat_lin, toa_lin, color="crimson", lw=2, zorder=4,
            label=f"ToA fit  cs = {cs:.2f} m/s  R² = {r2:.3f}")

    if ref_cs is not None and not np.isnan(ref_cs):
        ref_slope = 1.0 / (ref_cs * 1e3 / 1e0)   # ms/mm
        toa_ref   = toa.mean() + ref_slope * (lat_lin - lat.mean())
        ax.plot(lat_lin, toa_ref, color="steelblue", lw=1.5, ls="--",
                label=f"Reference  cs = {ref_cs:.2f} m/s")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("|v| amplitude (mm/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_xlabel("Lateral position (mm)", fontsize=10)
    ax.set_ylabel("Time of arrival (ms)", fontsize=10)
    ax.set_title(
        f"Shear Wave ToA Analysis — WavePINN V10\n{sample_name}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved ToA figure → {out_path}")


def save_summary_figure(all_results: List[Dict], out_path: Path):
    """Box-plot / violin summary across all samples for the report."""
    mode_a_results = [r for r in all_results if r.get("mode_a")]
    mode_b_results = [r for r in all_results if r.get("mode_b")]

    metrics_to_plot = [
        ("mae",      "MAE (mm/s)",     None),
        ("rmse",     "RMSE (mm/s)",    None),
        ("ssim",     "SSIM",           (0, 1)),
        ("r_active", "Pearson r (active)", (-1, 1)),
    ]

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, (key, label, ylim) in zip(axes, metrics_to_plot):
        vals_a = [r["mode_a"][key] for r in mode_a_results
                  if r.get("mode_a") and not np.isnan(r["mode_a"].get(key, float("nan")))]
        vals_b = [r["mode_b"][key] for r in mode_b_results
                  if r.get("mode_b") and not np.isnan(r["mode_b"].get(key, float("nan")))]

        data   = []
        labels = []
        if vals_a:
            data.append(vals_a)
            labels.append(f"Mode A\n(dt=24 ms)\nN={len(vals_a)}")
        if vals_b:
            data.append(vals_b)
            labels.append(f"Mode B\n(dt=2 ms)\nN={len(vals_b)}")

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        colors = ["#4C72B0", "#DD8452"]
        for patch, col in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)

        # Overlay individual points
        for i, (vals, col) in enumerate(zip(data, colors[:len(data)]), start=1):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter([i + j for j in jitter], vals, color=col, s=20,
                       alpha=0.8, zorder=5)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "WavePINN V10 — Performance Summary (500 Hz Val Set)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Summary figure → {out_path}")


def save_cs_comparison_figure(all_results: List[Dict], out_path: Path):
    """Predicted vs reference shear wave speed scatter plot."""
    cs_pred = []
    cs_ref  = []
    sources = []

    for r in all_results:
        if not r.get("mode_b"):
            continue
        cs_p = r["mode_b"].get("cs_ms", float("nan"))
        cs_r = r.get("ref_cs", float("nan"))
        if not np.isnan(cs_p) and not np.isinf(cs_p) and cs_p < 50.0 \
                and not np.isnan(cs_r) and not np.isinf(cs_r):
            cs_pred.append(cs_p)
            cs_ref.append(cs_r)
            sources.append(r.get("source", "?"))

    if len(cs_pred) < 2:
        print("  Insufficient cs data for comparison figure.")
        return

    cs_pred = np.array(cs_pred)
    cs_ref  = np.array(cs_ref)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # Colour by source phantom
    source_set = sorted(set(sources))
    palette    = plt.cm.Set1(np.linspace(0, 0.8, len(source_set)))
    for src, col in zip(source_set, palette):
        mask = [s == src for s in sources]
        ax.scatter(
            cs_ref[mask], cs_pred[mask],
            color=col, s=60, alpha=0.8, label=src, zorder=3,
        )

    # Identity line
    lim = [0, max(cs_ref.max(), cs_pred.max()) * 1.15]
    ax.plot(lim, lim, "k--", lw=1.5, alpha=0.5, label="Identity")
    ax.set_xlim(lim); ax.set_ylim(lim)

    # Regression
    from scipy.stats import linregress
    slope, intercept, r_val, _, _ = linregress(cs_ref, cs_pred)
    xlin = np.linspace(*lim, 100)
    ax.plot(xlin, slope * xlin + intercept, "r-", lw=1.5,
            label=f"Regression  r={r_val:.3f}")

    mae_cs = float(np.abs(cs_pred - cs_ref).mean())
    ax.set_xlabel("Reference cs (m/s)", fontsize=11)
    ax.set_ylabel("Predicted cs (m/s)", fontsize=11)
    ax.set_title(
        f"Shear Wave Speed Estimation\nMAE = {mae_cs:.3f} m/s  |  N = {len(cs_pred)}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  cs comparison figure → {out_path}")


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_evaluation(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print("WavePINN V10 — SWE Performance Report")
    print(f"{'='*72}")
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model_path}")
    print(f"  Output dir : {out_dir}")

    # --- Load model -------------------------------------------------------
    print("\n[1/4] Loading model ...")
    model, cfg = load_v10_model(args.model_path, device)

    # --- Load metadata ----------------------------------------------------
    print("\n[2/4] Loading metadata ...")
    with open(META_PATH) as f:
        meta = json.load(f)

    freq_filter = args.freq_filter
    val_samples = [
        s for s in meta.get("val", [])
        if freq_filter is None or s.get("frequency") == freq_filter
    ]
    if args.max_samples:
        val_samples = val_samples[:args.max_samples]
    print(f"  Found {len(val_samples)} val samples "
          f"(freq_filter={freq_filter or 'all'}).")

    # --- Evaluation loop --------------------------------------------------
    print("\n[3/4] Running inference and computing metrics ...")
    print_separator()

    all_results = []
    n_mode_a_ok = 0
    n_mode_b_ok = 0

    for i, smeta in enumerate(val_samples):
        name  = smeta["name"]
        print(f"\n[{i+1:2d}/{len(val_samples)}] {name}  [{smeta.get('frequency')}]")

        # Physical scale from pre-existing GT velocity (99th-percentile)
        try:
            gt_native = load_gt_velocity(smeta)  # (49, H, W)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        vel_scale = float(np.percentile(np.abs(gt_native), 99))
        vel_scale = max(vel_scale, 1.0)

        tissue = smeta.get("tissue_properties", {})
        ref_cs = tissue.get("estimated_shear_speed_m_s", float("nan"))

        result = dict(
            name   = name,
            source = smeta.get("source", "unknown"),
            ref_cs = ref_cs,
        )

        # ── Mode A: sparse reconstruction (dt=24ms) ────────────────────
        mode_a_metrics = None
        try:
            iq_sparse, actual_step_a = load_sparse_iq(smeta, sparse_step=smeta["sparse_step"])
            dt_a      = actual_step_a * smeta["frame_time_s"]
            if actual_step_a != smeta["sparse_step"]:
                print(f"  [WARN] sparse_step auto-reduced {smeta['sparse_step']}→{actual_step_a}")
            pred_a    = infer(model, iq_sparse, dt_a, vel_scale, device)
            # GT is already at the same 2ms output grid
            gt_a = gt_native[:pred_a.shape[0], :H_ROI, :W_ROI]
            mode_a_metrics = compute_pixel_metrics(gt_a, pred_a)
            result["mode_a"] = mode_a_metrics
            result["pred_a"] = pred_a
            result["gt_a"]   = gt_a
            result["dt_a_ms"]   = dt_a * 1e3 / UPSAMPLING_FACTOR  # output frame spacing
            result["dt_in_ms"]  = dt_a * 1e3
            n_mode_a_ok += 1
            print(f"  Mode A  dt_in={dt_a*1e3:.1f}ms  MAE={mode_a_metrics['mae']:.3f} mm/s  "
                  f"r_active={mode_a_metrics['r_active']:.3f}  "
                  f"SSIM={mode_a_metrics['ssim']:.3f}")
        except Exception as exc:
            print(f"  Mode A FAILED: {exc}")

        # ── Mode B: deployment (dt=2ms, consecutive 500Hz) ─────────────
        mode_b_metrics = None
        if not args.skip_deployment:
            try:
                iq_consec, _ = load_sparse_iq(smeta, sparse_step=1)  # consecutive
                dt_b         = smeta["frame_time_s"]               # 0.002 s
                pred_b    = infer(model, iq_consec, dt_b, vel_scale, device)
                dt_out_b  = dt_b / UPSAMPLING_FACTOR                # 0.000167 s

                # Block-average prediction to 500Hz for pixel comparison
                mode_b_metrics = compute_block_averaged_metrics(
                    gt_native[:, :H_ROI, :W_ROI], pred_b, UPSAMPLING_FACTOR
                )

                # ToA / cs estimation from dense 6kHz output
                toa_result = estimate_shear_wave_speed(pred_b, dt_out_b)
                mode_b_metrics["cs_ms"] = toa_result.get("cs_ms", float("nan"))
                mode_b_metrics["cs_r2"] = toa_result.get("r2",    float("nan"))
                result["mode_b"]     = mode_b_metrics
                result["pred_b"]     = pred_b
                result["toa_result"] = toa_result
                result["dt_b_ms"]    = dt_out_b * 1e3
                n_mode_b_ok += 1

                cs_pred = mode_b_metrics["cs_ms"]
                cs_err  = abs(cs_pred - ref_cs) if not np.isnan(cs_pred) and not np.isnan(ref_cs) else float("nan")
                print(f"  Mode B  MAE={mode_b_metrics['mae']:.3f} mm/s  "
                      f"r_active={mode_b_metrics['r_active']:.3f}  "
                      f"cs={cs_pred:.2f} m/s  "
                      f"(ref={ref_cs:.2f}  err={cs_err:.2f})")

                # ToA figure
                toa_fig_path = out_dir / "toa_figures" / f"toa_{name}.png"
                save_toa_cs_figure(toa_result, name, toa_fig_path, ref_cs)

            except Exception as exc:
                print(f"  Mode B FAILED: {exc}")

        # Static comparison figures
        if mode_a_metrics and not args.no_figures:
            fig_a_path = out_dir / "static_figures" / f"modeA_{name}.png"
            save_static_comparison_figure(
                gt        = result["gt_a"],
                pred      = result["pred_a"],
                sample_name = name,
                out_path  = fig_a_path,
                dt_out_ms = result["dt_a_ms"],
                mode_label = "Mode A — Sparse recon, dt=24 ms",
                metrics   = mode_a_metrics,
            )

        all_results.append(result)

    # --- Aggregate report -------------------------------------------------
    print(f"\n{'='*72}")
    print("WavePINN V10 — AGGREGATE PERFORMANCE REPORT")
    print(f"{'='*72}")
    print(f"\nTotal samples processed : {len(all_results)}")
    print(f"Mode A (sparse, dt=24ms): {n_mode_a_ok} successful")
    print(f"Mode B (deploy, dt=2ms) : {n_mode_b_ok} successful")

    for mode_key, mode_label in [("mode_a", "Mode A — Sparse recon (dt=24 ms, OOD)"),
                                  ("mode_b", "Mode B — Deployment (dt=2 ms, virtual 6 kHz)")]:
        mode_results = [r[mode_key] for r in all_results if r.get(mode_key)]
        if not mode_results:
            continue

        print(f"\n{'─'*72}")
        print(f"  {mode_label}")
        print(f"{'─'*72}")

        metric_keys = ["mae", "rmse", "psnr", "ssim", "r_full", "r_active"]
        units_map   = {"mae": "mm/s", "rmse": "mm/s", "psnr": "dB"}

        for key in metric_keys:
            vals = [r.get(key, float("nan")) for r in mode_results]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            unit = units_map.get(key, "")
            print(f"  {key:<20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} {unit}")

        if mode_key == "mode_b":
            cs_vals = [r.get("cs_ms", float("nan")) for r in mode_results]
            ref_vals = [r.get("ref_cs", float("nan")) for r in all_results if r.get(mode_key)]
            cs_pairs = [(c, r) for c, r in zip(cs_vals, ref_vals)
                        if not np.isnan(c) and not np.isnan(r)]
            if cs_pairs:
                cs_preds, cs_refs = zip(*cs_pairs)
                cs_mae = float(np.abs(np.array(cs_preds) - np.array(cs_refs)).mean())
                print(f"  {'cs_MAE (m/s)':<20s}: {cs_mae:.4f}  "
                      f"(N={len(cs_pairs)}, mean_pred={np.mean(cs_preds):.3f}, "
                      f"mean_ref={np.mean(cs_refs):.3f})")

    # --- Save JSON summary ------------------------------------------------
    summary = []
    for r in all_results:
        entry = dict(
            name   = r["name"],
            source = r.get("source"),
            ref_cs = r.get("ref_cs"),
        )
        for mk in ("mode_a", "mode_b"):
            if r.get(mk):
                entry[mk] = {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in r[mk].items()
                    if isinstance(v, (int, float, np.floating, np.ndarray)) and np.ndim(v) == 0
                }
        summary.append(entry)

    summary_path = out_dir / "v10_swe_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON report → {summary_path}")

    # --- Summary figures --------------------------------------------------
    if not args.no_figures and all_results:
        save_summary_figure(all_results, out_dir / "summary_metrics.png")
        save_cs_comparison_figure(all_results, out_dir / "cs_comparison.png")

    print(f"\n{'='*72}")
    print("Evaluation complete.")
    print(f"{'='*72}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="WavePINN V10 SWE performance report")
    p.add_argument(
        "--model_path",
        default="SWE_PINN_outputs_wavepinn_v10_best_model.pt",
        help="Path to V10 checkpoint (.pt)",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/v10_swe_report",
        help="Directory for figures and JSON report",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of samples (useful for testing)",
    )
    p.add_argument(
        "--skip_deployment", action="store_true",
        help="Skip Mode B (deployment, dt=2ms) — faster but less complete",
    )
    p.add_argument(
        "--no_figures", action="store_true",
        help="Skip saving static comparison figures",
    )
    p.add_argument(
        "--freq_filter", type=str, default="500Hz",
        help="Frequency tag to filter val samples (e.g. '500Hz', '5kHz', None for all). "
             "Default: 500Hz",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
