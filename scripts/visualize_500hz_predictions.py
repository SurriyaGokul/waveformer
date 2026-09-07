#!/usr/bin/env python3
"""
500 Hz SWE — Ground Truth vs Model Prediction Visualisation
============================================================
Generates comparison videos from the 500 Hz inference set.

Physical context (from IQ2VEl.m):
  • Detection PRF = 500 Hz  →  frame_time = 2.0 ms
    • This repo’s `data/fullscale_velocity/*/velocity_*.npy` targets are stored at
        the native detection PRF (no 10× slow-time interpolation applied), i.e.
        dt = 2.0 ms per velocity frame.
  • 5 sparse IQ frames → 49 dense velocity frames (upsampling_factor = 12)
  • At 500 Hz the shear wave passes through in the FIRST 2–3 velocity frames
        (~4–6 ms); the remaining frames are near-zero.

Strategy:
  1. Full video  (49 frames, 8 fps)  – correct temporal representation
  2. Zoom video  (first N_zoom frames, 4 fps) – slowed down to show the wave
  3. Metrics summary PNG across all processed samples

Usage
-----
  python scripts/visualize_500hz_predictions.py \
      --model_path outputs/fullscale_velocity_v2/best_model.pt \
      --output_dir outputs/500hz_viz

  # Only first 6 samples
  python scripts/visualize_500hz_predictions.py \
      --model_path outputs/fullscale_velocity_v2/best_model.pt \
      --max_samples 6
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from architectures.velocity_unet import ComplexToRealUNet
from src.utils.deployment_scale import estimate_velocity_scale_from_iq

# ---------------------------------------------------------------------------
# Physical constants (from IQ2VEl.m)
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX   = 0.0507   # mm per axial pixel  (λ/4 @ 7.6 MHz)
LATERAL_MM_PER_PX = 0.1013   # mm per lateral pixel
H_ROI = 116                   # original ROI height (before UNet padding to 128)

# 500 Hz target timing (native-rate velocity targets)
DETECT_PRF_HZ  = 500.0        # detection pulse repetition frequency
DT_FRAME_S     = 1.0 / DETECT_PRF_HZ   # 0.002 s = 2.0 ms
N_ZOOM_FRAMES  = 10           # frames to show in the zoomed active-wave video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_axes(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    depth_ax   = np.arange(H) * AXIAL_MM_PER_PX
    lateral_ax = (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX
    return lateral_ax, depth_ax


def _pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Pad H and W dims to nearest multiple of 16 for UNet."""
    *leading, H, W = arr.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH == 0 and pW == 0:
        return arr
    pad_widths = [(0, 0)] * len(leading) + [(0, pH), (0, pW)]
    return np.pad(arr, pad_widths, mode='constant', constant_values=0)


def _try_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
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
    print(f"  Loaded epoch={epoch}  best_val_loss={val_loss:.6f}  "
          f"params={n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sample(
    model: torch.nn.Module,
    iq_path: Path,
    vel_path: Path,
    device: torch.device,
    scale_mode: str = "loupas",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns:
        gt_mm            : (T, H_ROI, W) float32
        pred_mm          : (T, H_ROI, W) float32
        vel_scale_gt     : float — 99th-pct of |GT velocity| (mm/s)
        vel_scale_loupas : float — Loupas proxy scale (mm/s)
    """
    # --- Load raw data ---
    iq_data  = np.load(iq_path).astype(np.complex64)   # (T_sp, H, W)
    vel_data = np.load(vel_path).astype(np.float32)     # (T_vel, H, W)

    # --- Per-sample IQ normalisation (same as v2 training) ---
    iq_scale = float(np.percentile(np.abs(iq_data), 99))
    iq_scale = max(iq_scale, 1.0)
    iq_norm  = iq_data / iq_scale

    # Split real/imag → (2, T_sp, H, W)
    real = iq_norm.real
    imag = iq_norm.imag
    inp_arr = np.stack([real, imag], axis=0)   # (2, T_sp, H, W)

    # --- Velocity normalisation scale ---
    vel_scale_gt = float(np.percentile(np.abs(vel_data), 99))
    vel_scale_gt = max(vel_scale_gt, 1.0)

    # Loupas proxy: sparse IQ is already clutter-filtered; transpose to (H, W, T_sp)
    iq_hw_t = np.transpose(iq_data, (1, 2, 0))  # (H, W, T_sp)
    vel_scale_loupas = estimate_velocity_scale_from_iq(iq_hw_t, DT_FRAME_S)

    vel_scale = vel_scale_loupas if scale_mode == "loupas" else vel_scale_gt

    vel_norm  = np.clip(vel_data / vel_scale, -1.0, 1.0)

    # --- Spatial padding to multiple of 16 ---
    inp_arr  = _pad_to_multiple(inp_arr)    # (2, T_sp, H_pad, W_pad)
    vel_norm = _pad_to_multiple(vel_norm)   # (T_vel, H_pad, W_pad)

    # --- Tensor conversion ---
    inp_t = torch.from_numpy(inp_arr).unsqueeze(0).to(device)   # (1, 2, T_sp, H_pad, W_pad)

    # --- Model forward ---
    out = model(inp_t)
    pred_norm = out["velocity"] if isinstance(out, dict) else out  # (1, 1, T_vel, H_pad, W_pad)

    # --- Back to mm/s (reverse normalisation) ---
    gt_mm   = vel_norm[:, :H_ROI, :] * vel_scale
    pred_mm = pred_norm[0, 0].cpu().numpy()[:, :H_ROI, :] * vel_scale

    return gt_mm, pred_mm, vel_scale_gt, vel_scale_loupas


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    g, p = gt.flatten(), pred.flatten()
    corr = float(np.corrcoef(g, p)[0, 1]) if g.std() > 1e-8 else float("nan")

    # Active-frame metrics (only frames where GT std > 0.1 mm/s)
    frame_stds = gt.std(axis=(1, 2))
    active_mask = frame_stds > 0.1
    if active_mask.any():
        gt_a, pr_a = gt[active_mask], pred[active_mask]
        mae_active  = float(np.abs(pr_a - gt_a).mean())
        corr_active = float(np.corrcoef(gt_a.flatten(), pr_a.flatten())[0, 1])
    else:
        mae_active  = float("nan")
        corr_active = float("nan")

    return dict(mae=mae, rmse=rmse, corr=corr,
                mae_active=mae_active, corr_active=corr_active,
                n_active=int(active_mask.sum()))


def compute_frame0_similarity(gt: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute how much each frame resembles frame 0.

    Returns dict with:
      - energy_ratio: (T,) mean(gt[t]^2) / mean(gt[0]^2)
      - corr_signed:  (T,) Pearson corr(frame0, frame_t)
      - corr_abs:     (T,) abs(Pearson corr)
      - frame_stds:   (T,) spatial std per frame
    """
    if gt.ndim != 3:
        raise ValueError(f"Expected gt shape (T,H,W), got {gt.shape}")

    T = gt.shape[0]

    # Energy ratio (mean square)
    energy = np.mean(gt ** 2, axis=(1, 2))
    e0 = float(energy[0])
    energy_ratio = energy / max(e0, 1e-12)

    # Per-frame std
    frame_stds = gt.std(axis=(1, 2))

    # Pearson correlation vs frame 0 (mean-subtracted, so ddof cancels)
    X = gt.reshape(T, -1).astype(np.float64, copy=False)
    x0 = X[0]
    x0c = x0 - x0.mean()
    x0n = float(np.linalg.norm(x0c))

    corr_signed = np.full((T,), np.nan, dtype=np.float64)
    corr_abs = np.full((T,), np.nan, dtype=np.float64)
    corr_signed[0] = 1.0
    corr_abs[0] = 1.0

    if x0n < 1e-12:
        return {
            "energy_ratio": energy_ratio.astype(np.float32),
            "corr_signed": corr_signed.astype(np.float32),
            "corr_abs": corr_abs.astype(np.float32),
            "frame_stds": frame_stds.astype(np.float32),
        }

    for t in range(1, T):
        xt = X[t]
        xtc = xt - xt.mean()
        denom = x0n * float(np.linalg.norm(xtc))
        if denom < 1e-12:
            continue
        r = float(np.dot(x0c, xtc) / denom)
        # Numerical guard
        r = max(min(r, 1.0), -1.0)
        corr_signed[t] = r
        corr_abs[t] = abs(r)

    return {
        "energy_ratio": energy_ratio.astype(np.float32),
        "corr_signed": corr_signed.astype(np.float32),
        "corr_abs": corr_abs.astype(np.float32),
        "frame_stds": frame_stds.astype(np.float32),
    }


def save_frame0_similarity_summary(
    names: List[str],
    time_ms: np.ndarray,
    sim_list: List[Dict[str, np.ndarray]],
    out_png: Path,
    out_csv: Path,
    active_std_thresh: float = 0.1,
) -> None:
    if not sim_list:
        return

    energy_ratios = np.stack([d["energy_ratio"] for d in sim_list], axis=0)  # (N,T)
    corr_abs = np.stack([d["corr_abs"] for d in sim_list], axis=0)  # (N,T)
    frame_stds = np.stack([d["frame_stds"] for d in sim_list], axis=0)  # (N,T)

    active = frame_stds > active_std_thresh

    # Aggregate statistics
    mean_energy = np.nanmean(energy_ratios, axis=0)
    p10_energy = np.nanpercentile(energy_ratios, 10, axis=0)
    p90_energy = np.nanpercentile(energy_ratios, 90, axis=0)

    corr_abs_active = np.where(active, corr_abs, np.nan)
    # Avoid warnings when a time index has no active frames (all-NaN slice)
    valid = np.isfinite(corr_abs_active)
    valid_counts = valid.sum(axis=0)
    sum_vals = np.where(valid, corr_abs_active, 0.0).sum(axis=0)
    mean_abs_corr_active = np.full(sum_vals.shape, np.nan, dtype=np.float64)
    np.divide(sum_vals, valid_counts, out=mean_abs_corr_active, where=valid_counts > 0)

    frac_active = active.mean(axis=0)

    # --- Save per-sample CSV summary ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            f"n_active(std>{active_std_thresh})",
            "abs_corr_f1",
            "corr_f1",
            "energy_ratio_f1",
            "energy_ratio_f2",
        ])
        for name, d in zip(names, sim_list):
            stds = d["frame_stds"]
            n_active = int(np.sum(stds > active_std_thresh))
            corr_f1 = float(d["corr_signed"][1]) if len(d["corr_signed"]) > 1 else float("nan")
            abs_corr_f1 = float(d["corr_abs"][1]) if len(d["corr_abs"]) > 1 else float("nan")
            e1 = float(d["energy_ratio"][1]) if len(d["energy_ratio"]) > 1 else float("nan")
            e2 = float(d["energy_ratio"][2]) if len(d["energy_ratio"]) > 2 else float("nan")
            w.writerow([name, n_active, abs_corr_f1, corr_f1, e1, e2])

    # --- Plot summary PNG ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        "500 Hz: how informative is frame 0 for later motion?",
        fontsize=14,
        fontweight="bold",
    )

    # 1) Energy decay
    eps = 1e-6
    axes[0].semilogy(time_ms, np.clip(mean_energy, eps, None), label="mean")
    axes[0].fill_between(
        time_ms,
        np.clip(p10_energy, eps, None),
        np.clip(p90_energy, eps, None),
        alpha=0.2,
        label="10–90%",
    )
    axes[0].set_ylabel(r"Energy ratio $E_t/E_0$ (mean square)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # 2) Similarity of frames to frame 0
    axes[1].plot(time_ms, mean_abs_corr_active, linewidth=2)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel(r"Mean $|corr(f_0, f_t)|$ (active only)")
    axes[1].grid(True, alpha=0.3)

    # 3) How often the wave is still present
    axes[2].plot(time_ms, frac_active, linewidth=2)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Active fraction")
    axes[2].set_xlabel("Time (ms)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Frame-0 similarity summary → {out_png}")
    print(f"  Frame-0 similarity CSV     → {out_csv}")


# ---------------------------------------------------------------------------
# Video writer (core)
# ---------------------------------------------------------------------------

def _write_video(
    gt:         np.ndarray,
    pred:       np.ndarray,
    lat_ax:     np.ndarray,
    dep_ax:     np.ndarray,
    t_ax_ms:    np.ndarray,
    clim:       float,
    title_base: str,
    out_path:   Path,
    fps:        int = 8,
    active_mask: Optional[np.ndarray] = None,
) -> Path:
    """Write a 3-panel (GT | Prediction | |Error|) video."""
    T = len(t_ax_ms)
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(18, 6), facecolor="black")
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                            width_ratios=[1, 1, 1, 0.04],
                            wspace=0.08, left=0.06, right=0.96,
                            top=0.88, bottom=0.12)

    ax_gt  = fig.add_subplot(gs[0])
    ax_pr  = fig.add_subplot(gs[1])
    ax_err = fig.add_subplot(gs[2])
    ax_cb  = fig.add_subplot(gs[3])

    for ax in (ax_gt, ax_pr, ax_err):
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    im_gt  = ax_gt.imshow(gt[0], cmap="jet", vmin=-clim, vmax=clim,
                          aspect="auto", origin="upper", extent=extent)
    im_pr  = ax_pr.imshow(pred[0], cmap="jet", vmin=-clim, vmax=clim,
                          aspect="auto", origin="upper", extent=extent)
    im_err = ax_err.imshow(np.abs(gt[0] - pred[0]), cmap="hot",
                           vmin=0, vmax=clim,
                           aspect="auto", origin="upper", extent=extent)

    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Particle velocity (mm/s)", color="white",
                   fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for ax, ttl in ((ax_gt, "Ground Truth (Loupas)"),
                    (ax_pr, "Model Prediction"),
                    (ax_err, "|Error| (mm/s)")):
        ax.set_xlabel("Lateral distance (mm)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Axial depth (mm)",      fontsize=11, fontweight="bold")
        ax.set_title(ttl, fontsize=13, fontweight="bold")

    sup = fig.suptitle("", color="white", fontsize=11, fontweight="bold")

    def _update(fi: int):
        active_str = " ◀ WAVE ACTIVE" if (active_mask is not None and active_mask[fi]) else ""
        sup.set_text(
            f"{title_base}  |  t = {t_ax_ms[fi]:.2f} ms  "
            f"(frame {fi+1}/{T}){active_str}"
        )
        im_gt.set_data(gt[fi])
        im_pr.set_data(pred[fi])
        im_err.set_data(np.abs(gt[fi] - pred[fi]))
        return im_gt, im_pr, im_err, sup

    anim = FuncAnimation(fig, _update, frames=T, blit=True, interval=1000 // fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _try_ffmpeg():
        writer    = FFMpegWriter(fps=fps,
                                 metadata={"title": title_base},
                                 extra_args=["-pix_fmt", "yuv420p"])
        save_path = out_path.with_suffix(".mp4")
        anim.save(str(save_path), writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": "black"})
    else:
        writer    = PillowWriter(fps=fps)
        save_path = out_path.with_suffix(".gif")
        anim.save(str(save_path), writer=writer,
                  savefig_kwargs={"facecolor": "black"})

    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def save_metrics_summary(metrics_list: List[Dict], names: List[str], out_path: Path):
    if not metrics_list:
        return
    maes  = [m["mae_active"] for m in metrics_list]
    corrs = [m["corr_active"] for m in metrics_list]
    labels = [n.split("_500Hz")[0][-28:] for n in names]

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(names) * 0.5), 8))
    fig.suptitle("500 Hz Validation — Active-frame metrics", fontsize=14, fontweight="bold")

    for ax, vals, ylabel, color in zip(
        axes,
        [maes, corrs],
        ["Active-frame MAE (mm/s)", "Active-frame Pearson r"],
        ["steelblue", "seagreen"],
    ):
        non_nan = [v for v in vals if not np.isnan(v)]
        ax.bar(range(len(vals)), vals, color=color, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if non_nan:
            mu = float(np.mean(non_nan))
            ax.axhline(mu, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean = {mu:.3f}")
            ax.legend(fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics summary  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="500 Hz SWE: GT vs Prediction visualisation"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="outputs/fullscale_velocity_v2/best_model.pt",
        help="Path to v2 model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="data/fullscale_velocity/inference",
        help="Directory containing sparse_*500Hz* / velocity_*500Hz* files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/500hz_viz",
        help="Output directory for videos and plots",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to first N _w0 samples (default: all)",
    )
    parser.add_argument(
        "--clim", type=float, default=None,
        help="Colour limit mm/s (default: per-sample 99th pct of GT active frames)",
    )
    parser.add_argument(
        "--fps_full", type=int, default=8,
        help="FPS for the full 49-frame video (default: 8)",
    )
    parser.add_argument(
        "--fps_zoom", type=int, default=3,
        help="FPS for the zoomed active-wave video (default: 3)",
    )
    parser.add_argument(
        "--n_zoom", type=int, default=10,
        help="Number of frames in the zoomed video (default: 10)",
    )
    parser.add_argument(
        "--analysis_only", action="store_true",
        help="Skip model inference and video generation; only analyze GT frame-0 similarity.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["loupas", "gt"],
        default="loupas",
        dest="scale_mode",
        help="'loupas' (GT-free, default) or 'gt' (evaluation mode, requires GT velocity).",
    )
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = PROJECT_ROOT / args.model_path
    data_dir   = PROJECT_ROOT / args.data_dir
    out_dir    = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  500 Hz SWE Visualisation")
    print(f"  model  : {model_path.name}")
    print(f"  data   : {data_dir}")
    print(f"  output : {out_dir}")
    print(f"  device : {device}")
    print(
        f"  Physical: PRF=500Hz (native targets) → dt={DT_FRAME_S*1e3:.2f}ms/frame, "
        f"49 frames → {48*DT_FRAME_S*1e3:.1f}ms total"
    )
    print(f"{'='*65}\n")

    # Physical axes + time axis (49 frames, 0.2ms apart)
    lat_ax, dep_ax = make_axes(H_ROI, 160)
    t_ax_ms_full = np.arange(49) * DT_FRAME_S * 1e3

    frame0_sim_list: List[Dict[str, np.ndarray]] = []
    frame0_names: List[str] = []

    if args.analysis_only:
        print("[1/1] Analysis-only mode (GT frame-0 similarity) …")
        vel_files = sorted(data_dir.glob("velocity_*500Hz*_w0.npy"))
        if not vel_files:
            vel_files = sorted(data_dir.glob("velocity_*500Hz*.npy"))
        if not vel_files:
            raise FileNotFoundError(f"No velocity_*500Hz*.npy files found in {data_dir}")

        if args.max_samples is not None:
            vel_files = vel_files[: args.max_samples]

        print(f"  Found {len(vel_files)} velocity files\n")
        for vel_path in tqdm(vel_files, desc="GT similarity", unit="sample"):
            name = vel_path.stem[len("velocity_") :]
            gt_mm = np.load(vel_path).astype(np.float32)
            frame0_sim_list.append(compute_frame0_similarity(gt_mm))
            frame0_names.append(name)

        save_frame0_similarity_summary(
            frame0_names,
            t_ax_ms_full[: gt_mm.shape[0]],
            frame0_sim_list,
            out_dir / "frame0_similarity_500hz.png",
            out_dir / "frame0_similarity_500hz.csv",
            active_std_thresh=0.1,
        )
        return

    # ── 1. Load model ─────────────────────────────────────────────────────
    print("[1/3] Loading model …")
    model = load_model(str(model_path), device)
    print()

    # ── 2. Discover _w0 500Hz samples ─────────────────────────────────────
    print("[2/3] Discovering 500Hz _w0 samples …")
    sparse_files = sorted(data_dir.glob("sparse_*500Hz*_w0.npy"))
    if not sparse_files:
        # Fallback: check any 500Hz file
        sparse_files = sorted(data_dir.glob("sparse_*500Hz*.npy"))
    if not sparse_files:
        raise FileNotFoundError(
            f"No sparse_*500Hz*_w0.npy files found in {data_dir}"
        )

    if args.max_samples is not None:
        sparse_files = sparse_files[: args.max_samples]

    print(f"  Found {len(sparse_files)} samples\n")

    all_metrics: List[Dict] = []
    all_names: List[str] = []

    # ── 3. Inference + video ───────────────────────────────────────────────
    print(f"[3/3] Running inference and generating videos …\n")

    for sp_path in tqdm(sparse_files, desc="500Hz samples", unit="sample"):
        name = sp_path.stem[len("sparse_") :]
        vel_path = sp_path.parent / f"velocity_{name}.npy"
        if not vel_path.exists():
            print(f"  WARNING: missing velocity file for {name}, skipping")
            continue

        try:
            gt_mm, pred_mm, vel_scale_gt, vel_scale_loupas = infer_sample(
                model, sp_path, vel_path, device, scale_mode=args.scale_mode,
            )
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            continue

        T, _, _ = gt_mm.shape
        t_ax_ms = np.arange(T) * DT_FRAME_S * 1e3   # 0, 0.2, 0.4 … ms

        # Frame-0 similarity analysis (GT only)
        frame0_sim_list.append(compute_frame0_similarity(gt_mm))
        frame0_names.append(name)

        # Active mask — frames where wave is present (visual marker)
        frame_stds = gt_mm.std(axis=(1, 2))
        active_mask = frame_stds > 0.05   # 0.05 mm/s threshold

        # Colour limit — use actual peak of active frames
        if args.clim is not None:
            clim = args.clim
        else:
            active_data = gt_mm[active_mask] if active_mask.any() else gt_mm
            clim = float(np.abs(active_data).max())
            clim = max(clim, 0.2)

        # Plane / freq info for title
        p1 = name.split("_Plane_")[1][:8] if "_Plane_" in name else ""
        source = name.split("_")[0]
        title_base = f"500 Hz SWE | {source} | Plane {p1} | {name[-22:]}"

        metrics = compute_metrics(gt_mm, pred_mm)
        metrics["vel_scale_gt_mms"]     = vel_scale_gt
        metrics["vel_scale_loupas_mms"] = vel_scale_loupas
        metrics["scale_mode"]           = args.scale_mode
        all_metrics.append(metrics)
        all_names.append(name)

        # ── Full video (49 frames) ──────────────────────────────────────
        full_path = _write_video(
            gt=gt_mm,
            pred=pred_mm,
            lat_ax=lat_ax,
            dep_ax=dep_ax,
            t_ax_ms=t_ax_ms,
            clim=clim,
            title_base=title_base + "  [full 49 frames]",
            out_path=out_dir / f"{name}_full",
            fps=args.fps_full,
            active_mask=active_mask,
        )

        # ── Zoom video (first N_zoom frames — the active wave window) ──
        n_zoom = min(args.n_zoom, T)
        _write_video(
            gt=gt_mm[:n_zoom],
            pred=pred_mm[:n_zoom],
            lat_ax=lat_ax,
            dep_ax=dep_ax,
            t_ax_ms=t_ax_ms[:n_zoom],
            clim=clim,
            title_base=title_base + f"  [zoom: first {n_zoom} frames = {t_ax_ms[n_zoom-1]:.1f}ms]",
            out_path=out_dir / f"{name}_zoom",
            fps=args.fps_zoom,
            active_mask=active_mask[:n_zoom],
        )

        n_active = metrics["n_active"]
        tqdm.write(
            f"  {name[-40:]:40s}  clim=±{clim:.1f}mm/s  "
            f"active_frames={n_active}  "
            f"MAE={metrics['mae_active']:.3f}mm/s  r={metrics['corr_active']:.3f}  "
            f"→ {full_path.name}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Processed {len(all_metrics)} samples")
    if all_metrics:
        valid_corrs = [m["corr_active"] for m in all_metrics
                       if not np.isnan(m["corr_active"])]
        valid_maes  = [m["mae_active"]  for m in all_metrics
                       if not np.isnan(m["mae_active"])]
        if valid_corrs:
            print(f"  Active-frame Pearson r : mean={np.mean(valid_corrs):.4f}  "
                  f"median={np.median(valid_corrs):.4f}")
        if valid_maes:
            print(f"  Active-frame MAE       : mean={np.mean(valid_maes):.3f} mm/s  "
                  f"median={np.median(valid_maes):.3f} mm/s")
    print(f"  Videos saved to: {out_dir}")
    print(f"{'='*65}\n")

    if all_metrics:
        save_metrics_summary(
            all_metrics, all_names,
            out_dir / "metrics_summary_500hz.png"
        )

    if frame0_sim_list:
        # Use the first sample's T for time axis (all should be 49)
        T0 = int(frame0_sim_list[0]["energy_ratio"].shape[0])
        save_frame0_similarity_summary(
            frame0_names,
            t_ax_ms_full[:T0],
            frame0_sim_list,
            out_dir / "frame0_similarity_500hz.png",
            out_dir / "frame0_similarity_500hz.csv",
            active_std_thresh=0.1,
        )


if __name__ == "__main__":
    main()
