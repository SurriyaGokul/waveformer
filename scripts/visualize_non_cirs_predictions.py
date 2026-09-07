#!/usr/bin/env python3
"""
Non-CIRS Velocity Prediction Visualisation (5 kHz / 10 kHz / 20 kHz).

Loads a trained ComplexToRealUNet checkpoint, runs inference on the
non-CIRS validation set and generates one side-by-side comparison video
per PRF (plus per-sample summary PNGs and a global metrics CSV).

Layout of each video frame
~~~~~~~~~~~~~~~~~~~~~~~~~~
┌────────────────┬────────────────┬────────────────┐
│  Ground Truth  │   Prediction   │  Absolute Err  │
│   (Jet cmap)   │   (Jet cmap)   │  (Hot cmap)    │
└────────────────┴────────────────┴────────────────┘

Usage
~~~~~
  python scripts/visualize_non_cirs_predictions.py \
      --model_path outputs/non_cirs_5_10_20kHz/best_model.pt \
      --data_dir   data/non_cirs_velocity \
      --output_dir outputs/non_cirs_viz \
      --frequencies 5kHz 10kHz 20kHz \
      --samples_per_freq 2         # videos per frequency (0 = all)
      --fps 15
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
from tqdm import tqdm

# ── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.velocity_unet import create_velocity_model
from src.utils.iq_velocity_dataset import IQVelocityDataset, normalize_prf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Physical pixel spacing (from MATLAB IQ2VEl.m comments)
AXIAL_MM   = 0.0507   # mm per axial pixel
LATERAL_MM = 0.1013   # mm per lateral pixel


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def load_checkpoint(model_path: str, device: torch.device):
    """Return (model, config_dict, data_info_dict)."""
    logger.info(f"Loading checkpoint: {model_path}")
    ck = torch.load(model_path, map_location=device, weights_only=False)

    cfg = ck.get("config", {})
    data_info = ck.get("data_info", {})
    epoch = ck.get("epoch", "?")
    val_loss = ck.get("best_val_loss", float("nan"))
    logger.info(f"  Checkpoint epoch={epoch}  best_val_loss={val_loss:.6f}")

    enc_depths = cfg.get("encoder_depths", [32, 64, 128, 256])
    if isinstance(enc_depths, tuple):
        enc_depths = list(enc_depths)

    model = create_velocity_model(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("base_channels", 32),
        encoder_depths=enc_depths,
        upsampling_factor=cfg.get("upsampling_factor", 14),
        use_motion_interpolation=cfg.get("use_motion_interpolation", True),
        complex_to_real_mode=cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn=cfg.get("use_naive_bn", True),
        flow_hidden_channels=cfg.get("flow_hidden_channels", 32),
        flow_scale=cfg.get("flow_scale", 0.1),
        use_prf_conditioning=cfg.get("use_prf_conditioning", True),
        prf_hidden_dim=cfg.get("prf_hidden_dim", 64),
    ).to(device)

    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {n_params:,} parameters")
    return model, cfg, data_info


def axes_mm(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return physical depth / lateral axes in mm."""
    depth_ax = np.arange(H) * AXIAL_MM
    # centre the lateral axis around zero
    lat_ax = (np.arange(W) - W / 2.0) * LATERAL_MM
    return depth_ax, lat_ax


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    diff = pred - gt
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    if pred.std() > 1e-8 and gt.std() > 1e-8:
        corr = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1])
    else:
        corr = float("nan")
    metrics = dict(mae=mae, rmse=rmse, corr=corr)
    if mask is not None:
        m = mask.ravel() > 0.1
        if m.sum() > 0:
            metrics["masked_mae"] = float(np.abs(diff.ravel()[m]).mean())
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: IQVelocityDataset,
    device: torch.device,
    vel_norm: float,
    use_prf_conditioning: bool,
) -> List[Dict]:
    """Run model inference on every sample in *dataset*.

    Returns list of dicts with keys:
        name, prf_hz, prediction_mms, ground_truth_mms, mask
    """
    results = []
    for idx in tqdm(range(len(dataset)), desc="Inference"):
        sample = dataset[idx]

        inp   = sample["input"].unsqueeze(0).to(device)       # (1,2,T,H,W)
        tgt   = sample["target"].unsqueeze(0).to(device)      # (1,1,T,H,W)
        mask  = sample["mask"][0, 0].cpu().numpy()             # (H,W)
        prf_hz = float(sample["prf_hz"])

        log_prf = None
        if use_prf_conditioning:
            log_prf = sample["log_prf"].unsqueeze(0).to(device)

        pred = model(inp, log_prf=log_prf)                    # (1,1,T,H,W)

        # Align temporal dim in case of rounding
        T_t = tgt.shape[2]
        T_p = pred.shape[2]
        if T_p > T_t:
            pred = pred[:, :, :T_t]
        elif T_p < T_t:
            tgt = tgt[:, :, :T_p]

        # Denormalise to mm/s
        pred_np = pred[0, 0].cpu().numpy() * vel_norm      # (T,H,W)
        gt_np   = tgt[0, 0].cpu().numpy()  * vel_norm      # (T,H,W)

        results.append({
            "name":             sample["name"],
            "prf_hz":           prf_hz,
            "prediction_mms":   pred_np,
            "ground_truth_mms": gt_np,
            "mask":             mask,
        })

    logger.info(f"Inference complete: {len(results)} samples")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Per-sample summary PNG
# ════════════════════════════════════════════════════════════════════════════

def save_summary_png(
    pred: np.ndarray,
    gt:   np.ndarray,
    mask: np.ndarray,
    name: str,
    prf_hz: float,
    output_path: Path,
    n_cols: int = 6,
    vmax_mms: float = 30.0,
):
    """Save an (n_rows × n_cols) panel of GT / Prediction / Error snapshots."""
    T = pred.shape[0]
    indices = np.linspace(0, T - 1, n_cols * 2, dtype=int)
    H, W = pred.shape[1], pred.shape[2]
    depth_ax, lat_ax = axes_mm(H, W)
    ext = [lat_ax[0], lat_ax[-1], depth_ax[-1], depth_ax[0]]

    n_rows = 3   # GT / Pred / |Error|
    fig, axes = plt.subplots(n_rows, len(indices),
                             figsize=(2.2 * len(indices), 5.5),
                             constrained_layout=True)

    kw_vel = dict(aspect="auto", extent=ext, cmap="jet",
                  vmin=-vmax_mms, vmax=vmax_mms, origin="upper")
    kw_err = dict(aspect="auto", extent=ext, cmap="hot",
                  vmin=0, vmax=vmax_mms * 0.5, origin="upper")

    for col, ti in enumerate(indices):
        axes[0, col].imshow(gt[ti],             **kw_vel)
        axes[1, col].imshow(pred[ti],            **kw_vel)
        im_e = axes[2, col].imshow(np.abs(pred[ti] - gt[ti]), **kw_err)

        for row in range(n_rows):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_title(f"t={ti}", fontsize=7)

    # Row labels
    for row, lbl in enumerate(["Ground Truth", "Prediction", "|Error|"]):
        axes[row, 0].set_ylabel(lbl, fontsize=9, fontweight="bold")

    # Colorbars
    fig.colorbar(
        axes[0, -1].images[0], ax=axes[0:2, :], shrink=0.6,
        label="Velocity (mm/s)", pad=0.01, location="right",
    )
    fig.colorbar(
        im_e, ax=axes[2, :], shrink=0.6,
        label="|Error| (mm/s)", pad=0.01, location="right",
    )

    metrics = compute_metrics(pred, gt, mask[np.newaxis, :, :].repeat(T, axis=0))
    prf_khz = prf_hz / 1000
    fig.suptitle(
        f"{name}  |  PRF={prf_khz:.0f} kHz  |  "
        f"MAE={metrics['mae']:.2f} mm/s  RMSE={metrics['rmse']:.2f} mm/s  "
        f"Corr={metrics['corr']:.3f}",
        fontsize=9, fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved summary PNG: {output_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# Side-by-side video
# ════════════════════════════════════════════════════════════════════════════

def save_comparison_video(
    pred: np.ndarray,
    gt:   np.ndarray,
    mask: np.ndarray,
    name: str,
    prf_hz: float,
    output_path: Path,
    vmax_mms: float = 30.0,
    fps: int = 15,
    step: int = 1,
    dpi: int = 120,
):
    """Produce a 3-panel (GT | Pred | |Error|) AVI/GIF animation."""
    T, H, W = pred.shape
    frames_idx = list(range(0, T, step))
    if not frames_idx:
        return

    depth_ax, lat_ax = axes_mm(H, W)
    ext = [lat_ax[0], lat_ax[-1], depth_ax[-1], depth_ax[0]]

    prf_khz = prf_hz / 1000.0
    dt_s = 1.0 / prf_hz                         # time between velocity frames

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], figure=fig,
                            left=0.05, right=0.93, bottom=0.12, top=0.86,
                            wspace=0.04, hspace=0.0)

    ax_gt  = fig.add_subplot(gs[0, 0])
    ax_pr  = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[0, 2])
    ax_cb  = fig.add_subplot(gs[0, 3])

    kw_vel = dict(aspect="auto", extent=ext, cmap="jet",
                  vmin=-vmax_mms, vmax=vmax_mms, origin="upper",
                  interpolation="bilinear")
    kw_err = dict(aspect="auto", extent=ext, cmap="hot",
                  vmin=0, vmax=vmax_mms * 0.4, origin="upper",
                  interpolation="bilinear")

    # Initial frame
    ti0 = frames_idx[0]
    im_gt  = ax_gt.imshow(gt[ti0],              **kw_vel)
    im_pr  = ax_pr.imshow(pred[ti0],             **kw_vel)
    im_err = ax_err.imshow(np.abs(pred[ti0] - gt[ti0]), **kw_err)

    # Shared colorbar for GT + Pred
    cb_vel = fig.colorbar(im_gt, cax=ax_cb)
    cb_vel.set_label("Velocity (mm/s)", fontsize=10)

    for ax, title in zip([ax_gt, ax_pr, ax_err],
                         ["Ground Truth", "Prediction", "|Error|"]):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Lateral (mm)", fontsize=9)

    ax_gt.set_ylabel("Depth (mm)", fontsize=9)

    sup = fig.suptitle("", fontsize=11, fontweight="bold")

    # Per-frame metrics
    all_mae  = [np.abs(pred[i] - gt[i]).mean() for i in range(T)]
    all_rmse = [np.sqrt(((pred[i] - gt[i]) ** 2).mean()) for i in range(T)]

    def update(frame_no: int):
        ti = frames_idx[frame_no]
        t_ms = ti * dt_s * 1e3
        im_gt.set_data(gt[ti])
        im_pr.set_data(pred[ti])
        im_err.set_data(np.abs(pred[ti] - gt[ti]))
        sup.set_text(
            f"{name}  |  PRF={prf_khz:.0f} kHz  |  "
            f"t = {t_ms:.2f} ms  (frame {ti}/{T-1})  |  "
            f"MAE={all_mae[ti]:.2f}  RMSE={all_rmse[ti]:.2f} mm/s"
        )
        return [im_gt, im_pr, im_err, sup]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_idx),
        interval=max(1, 1000 // fps), blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try FFmpeg first, fall back to Pillow GIF
    try:
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=3000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        anim.save(str(output_path), writer=writer, dpi=dpi)
        logger.info(f"  Saved video: {output_path.name}")
    except Exception as e_ff:
        logger.warning(f"  FFmpeg failed ({e_ff}), falling back to GIF …")
        gif_path = output_path.with_suffix(".gif")
        try:
            anim.save(str(gif_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
            logger.info(f"  Saved GIF: {gif_path.name}")
        except Exception as e_gif:
            logger.error(f"  Could not save animation: {e_gif}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Per-frequency aggregated video (best N samples stitched)
# ════════════════════════════════════════════════════════════════════════════

def save_freq_overview_video(
    freq_results: List[Dict],
    freq_label: str,
    output_path: Path,
    vmax_mms: float = 30.0,
    fps: int = 15,
    step: int = 1,
    dpi: int = 100,
    max_samples: int = 3,
):
    """One video showing up to *max_samples* samples for a given PRF, stacked vertically."""
    results = freq_results[:max_samples]
    n = len(results)
    if n == 0:
        return

    T = results[0]["prediction_mms"].shape[0]
    H = results[0]["prediction_mms"].shape[1]
    W = results[0]["prediction_mms"].shape[2]
    frames_idx = list(range(0, T, step))
    depth_ax, lat_ax = axes_mm(H, W)
    ext = [lat_ax[0], lat_ax[-1], depth_ax[-1], depth_ax[0]]

    # Figure: n rows × 3 cols (GT | Pred | Err) + colorbar strip
    fig, axes = plt.subplots(
        n, 3, figsize=(13, 4.0 * n),
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    kw_v = dict(aspect="auto", extent=ext, cmap="jet",
                vmin=-vmax_mms, vmax=vmax_mms, origin="upper",
                interpolation="bilinear")
    kw_e = dict(aspect="auto", extent=ext, cmap="hot",
                vmin=0, vmax=vmax_mms * 0.4, origin="upper",
                interpolation="bilinear")

    ims_gt, ims_pr, ims_err = [], [], []
    t0 = frames_idx[0]
    for row, res in enumerate(results):
        gt   = res["ground_truth_mms"]
        pred = res["prediction_mms"]
        ims_gt.append(axes[row, 0].imshow(gt[t0],   **kw_v))
        ims_pr.append(axes[row, 1].imshow(pred[t0],  **kw_v))
        ims_err.append(axes[row, 2].imshow(np.abs(pred[t0]-gt[t0]), **kw_e))

        short = res["name"][:50]
        for col, lbl in enumerate(["GT", "Pred", "|Err|"]):
            axes[row, col].set_title(f"{lbl}  [{short}]", fontsize=8)
            axes[row, col].axis("off")

    # Colorbars
    fig.colorbar(ims_gt[0], ax=axes[:, 0:2], shrink=0.6,
                 label="Velocity (mm/s)", location="right")
    fig.colorbar(ims_err[0], ax=axes[:, 2], shrink=0.6,
                 label="|Error| (mm/s)", location="right")

    sup = fig.suptitle("", fontsize=11, fontweight="bold")

    dt_s = 1.0 / results[0]["prf_hz"]

    def update(frame_no: int):
        ti = frames_idx[frame_no]
        t_ms = ti * dt_s * 1e3
        for row, res in enumerate(results):
            gt   = res["ground_truth_mms"]
            pred = res["prediction_mms"]
            ims_gt[row].set_data(gt[ti])
            ims_pr[row].set_data(pred[ti])
            ims_err[row].set_data(np.abs(pred[ti] - gt[ti]))
        sup.set_text(f"PRF = {freq_label}  |  t = {t_ms:.2f} ms  (frame {ti}/{T-1})")
        return ims_gt + ims_pr + ims_err + [sup]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_idx),
        interval=max(1, 1000 // fps), blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=3000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        anim.save(str(output_path), writer=writer, dpi=dpi)
        logger.info(f"  Saved overview video: {output_path.name}")
    except Exception as e:
        logger.warning(f"  FFmpeg failed ({e}), trying GIF …")
        gif_path = output_path.with_suffix(".gif")
        try:
            anim.save(str(gif_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
            logger.info(f"  Saved GIF: {gif_path.name}")
        except Exception as e2:
            logger.error(f"  Could not save: {e2}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualise non-CIRS velocity predictions vs ground truth"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="outputs/non_cirs_5_10_20kHz/best_model.pt",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="data/non_cirs_velocity",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/non_cirs_viz",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val"],
        help="Dataset split to visualise (default: val)",
    )
    parser.add_argument(
        "--frequencies", nargs="+",
        default=["5kHz", "10kHz", "20kHz"],
        help="PRF labels to include",
    )
    parser.add_argument(
        "--samples_per_freq", type=int, default=0,
        help="Max individual-sample videos per frequency (0 = all)",
    )
    parser.add_argument(
        "--overview_samples", type=int, default=3,
        help="Samples per frequency in the overview video (default: 3)",
    )
    parser.add_argument(
        "--vmax", type=float, default=30.0,
        help="Velocity colour scale ±vmax mm/s (default 30)",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Video frames per second (default: 15)",
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Only render every Nth temporal frame (default: 1)",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="cuda / cpu (default: auto-detect)",
    )
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ──────────────────────────────────────────────────────────
    model, cfg, data_info = load_checkpoint(args.model_path, device)
    use_prf = cfg.get("use_prf_conditioning", True)
    vel_norm = data_info.get("vel_norm_scalar", 225.07)
    iq_norm  = data_info.get("iq_norm_scalar",  22533408.0)

    # ── load dataset ────────────────────────────────────────────────────────
    dataset = IQVelocityDataset(
        data_dir=args.data_dir,
        split=args.split,
        iq_norm_scalar=iq_norm,
        vel_norm_scalar=vel_norm,
    )
    logger.info(f"Loaded {len(dataset)} {args.split} samples")

    # ── run inference on all samples ────────────────────────────────────────
    results = run_inference(model, dataset, device, vel_norm, use_prf)

    # ── group by frequency ───────────────────────────────────────────────────
    freq_map: Dict[str, List[Dict]] = {}
    # Build PRF-Hz → label lookup
    prf_to_label = {
        500.0: "500Hz", 1000.0: "1kHz", 5000.0: "5kHz",
        10000.0: "10kHz", 20000.0: "20kHz",
    }
    for res in results:
        lbl = prf_to_label.get(res["prf_hz"], f"{int(res['prf_hz'])}Hz")
        freq_map.setdefault(lbl, []).append(res)

    # ── metrics CSV ─────────────────────────────────────────────────────────
    csv_path = output_dir / "metrics.csv"
    csv_rows = []

    all_freqs_requested = [f for f in args.frequencies if f in freq_map]
    if not all_freqs_requested:
        logger.warning(
            f"No results found for {args.frequencies}. "
            f"Available: {list(freq_map.keys())}"
        )
        all_freqs_requested = list(freq_map.keys())

    for freq_label in all_freqs_requested:
        freq_results = freq_map.get(freq_label, [])
        if not freq_results:
            logger.warning(f"No samples found for {freq_label}, skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Frequency: {freq_label}  ({len(freq_results)} samples)")
        logger.info(f"{'='*60}")

        # ── overview video (multiple samples stacked) ─────────────────────
        overview_path = output_dir / f"overview_{freq_label}.avi"
        save_freq_overview_video(
            freq_results,
            freq_label=freq_label,
            output_path=overview_path,
            vmax_mms=args.vmax,
            fps=args.fps,
            step=args.step,
            max_samples=args.overview_samples,
        )

        # ── per-sample individual videos and summary PNGs ─────────────────
        n_to_render = len(freq_results) if args.samples_per_freq == 0 \
                      else args.samples_per_freq

        for si, res in enumerate(freq_results[:n_to_render]):
            name  = res["name"]
            pred  = res["prediction_mms"]
            gt    = res["ground_truth_mms"]
            mask  = res["mask"]

            metrics = compute_metrics(
                pred, gt,
                mask[np.newaxis].repeat(pred.shape[0], axis=0),
            )
            logger.info(
                f"  [{si+1}/{n_to_render}] {name[:60]}"
                f"  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}"
                f"  Corr={metrics['corr']:.3f}"
            )
            csv_rows.append({
                "freq": freq_label,
                "name": name,
                **{k: f"{v:.4f}" for k, v in metrics.items()},
            })

            # Summary PNG
            png_path = output_dir / freq_label / f"summary_{name[:80]}.png"
            save_summary_png(pred, gt, mask, name, res["prf_hz"], png_path,
                             vmax_mms=args.vmax)

            # Individual comparison video
            vid_path = output_dir / freq_label / f"compare_{name[:80]}.avi"
            save_comparison_video(
                pred, gt, mask, name, res["prf_hz"], vid_path,
                vmax_mms=args.vmax, fps=args.fps, step=args.step,
            )

    # ── write CSV ────────────────────────────────────────────────────────────
    if csv_rows:
        fieldnames = ["freq", "name", "mae", "rmse", "corr", "masked_mae"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"\nMetrics CSV saved: {csv_path}")

    # ── print aggregate summary ───────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("AGGREGATE METRICS BY FREQUENCY")
    logger.info("="*60)
    for freq_label in all_freqs_requested:
        rows = [r for r in csv_rows if r["freq"] == freq_label]
        if rows:
            maes  = [float(r["mae"]) for r in rows]
            rmses = [float(r["rmse"]) for r in rows]
            corrs = [float(r["corr"]) for r in rows if r["corr"] != "nan"]
            logger.info(
                f"  {freq_label:8s}  "
                f"MAE={np.mean(maes):.3f}±{np.std(maes):.3f}  "
                f"RMSE={np.mean(rmses):.3f}±{np.std(rmses):.3f}  "
                f"Corr={np.mean(corrs) if corrs else float('nan'):.3f}"
            )

    logger.info(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
