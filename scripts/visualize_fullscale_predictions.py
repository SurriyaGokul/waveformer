#!/usr/bin/env python3
"""
Velocity Prediction Visualisation — Fullscale Dataset
======================================================
Generates side-by-side MP4 comparison videos (Ground Truth | Prediction | Error)
for every validation sample, mirroring the style of the reference MATLAB IQ2VEl.m
script (jet colormap, particle-velocity colour bar, lateral/axial axes in mm).

Physical axis calibration (from IQ2VEl.m medfilt2 kernel comment):
    axial pixel spacing   = 0.0507 mm  (~λ/4 at f_c = 7.6 MHz, c = 1540 m/s)
    lateral pixel spacing = 0.1013 mm  (2× axial spacing)

Usage
-----
    python scripts/visualize_fullscale_predictions.py \
        --model_path SWE_PINN_outputs_fullscale_velocity_v1_best_model.pt \
        --output_dir outputs/fullscale_velocity_viz

    # Limit to first N samples (quick check)
    python scripts/visualize_fullscale_predictions.py \
        --model_path SWE_PINN_outputs_fullscale_velocity_v1_best_model.pt \
        --max_samples 5

    # Override colour limits (mm/s)
    python scripts/visualize_fullscale_predictions.py \
        --model_path SWE_PINN_outputs_fullscale_velocity_v1_best_model.pt \
        --clim 80
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                     # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from architectures.velocity_unet import ComplexToRealUNet
from utils.iq_velocity_dataset import IQVelocityDataset

# ---------------------------------------------------------------------------
# Physical constants (from IQ2VEl.m)
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX   = 0.0507   # mm per axial pixel  (~λ/4 at 7.6 MHz)
LATERAL_MM_PER_PX = 0.1013   # mm per lateral pixel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict, Dict]:
    """Load ComplexToRealUNet from a training checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    di   = ckpt["data_info"]

    model = ComplexToRealUNet(
        in_channels            = cfg.get("in_channels", 1),
        base_channels          = cfg.get("base_channels", 32),
        encoder_depths         = list(cfg.get("encoder_depths", [32, 64, 128, 256])),
        upsampling_factor      = cfg.get("upsampling_factor", 12),
        use_motion_interpolation = cfg.get("use_motion_interpolation", True),
        complex_to_real_mode   = cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn           = cfg.get("use_naive_bn", True),
        flow_hidden_channels   = cfg.get("flow_hidden_channels", 32),
        flow_scale             = cfg.get("flow_scale", 0.1),
        use_prf_conditioning   = cfg.get("use_prf_conditioning", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch     = ckpt.get("epoch", "?")
    val_loss  = ckpt.get("best_val_loss", float("nan"))
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Loaded checkpoint  epoch={epoch}  best_val_loss={val_loss:.6f}  "
          f"params={n_params:,}")
    return model, cfg, di


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def make_axes(H_roi: int, W_roi: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return lateral (x) and axial-depth (y) axes in mm, centred laterally."""
    depth_axis   = np.arange(H_roi) * AXIAL_MM_PER_PX
    lateral_axis = (np.arange(W_roi) - W_roi / 2.0) * LATERAL_MM_PER_PX
    return lateral_axis, depth_axis


def make_time_axis_ms(n_frames: int, frame_time_s: float) -> np.ndarray:
    """Return slow-time axis in milliseconds for the velocity frames."""
    return np.arange(n_frames) * frame_time_s * 1e3   # ms


# ---------------------------------------------------------------------------
# Per-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sample(
    model: torch.nn.Module,
    batch:  Dict,
    device: torch.device,
    vel_norm: float,
    H_roi:  int,
    per_sample_vel_norm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run forward pass and return (gt_mm_s, pred_mm_s, mask) all as numpy.

    Shapes:  (T, H_roi, W)  for velocity,  (H_roi, W) for mask.
    The dataset pads H from 116 → 128; we crop back to H_roi.
    """
    inputs  = batch["input"].to(device)   # (1, 2, T_sp, H_pad, W)
    targets = batch["target"].to(device)  # (1, 1, T_vel, H_pad, W)
    mask    = batch["mask"].to(device)    # (1, 1, 1, H_pad, W)

    preds = model(inputs)                 # (1, 1, T_vel, H_pad, W)

    # Denormalise → mm/s, squeeze batch/channel dims
    if per_sample_vel_norm:
        scale = batch["vel_scale"].item() if batch["vel_scale"].dim() == 0 else batch["vel_scale"][0].item()
    else:
        scale = vel_norm
    gt_np   = targets[0, 0].cpu().numpy() * scale   # (T, H_pad, W)
    pr_np   = preds[0, 0].cpu().numpy()   * scale   # (T, H_pad, W)
    mask_np = mask[0, 0, 0].cpu().numpy()            # (H_pad, W)

    # Crop padding back to original ROI height
    gt_np   = gt_np[:,   :H_roi, :]
    pr_np   = pr_np[:,   :H_roi, :]
    mask_np = mask_np[:H_roi, :]

    return gt_np, pr_np, mask_np


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    g, p = gt.flatten(), pred.flatten()
    corr = float(np.corrcoef(g, p)[0, 1]) if g.std() > 1e-8 else float("nan")
    return dict(mae=mae, rmse=rmse, corr=corr)


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def _try_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def make_video(
    gt:        np.ndarray,   # (T, H, W) mm/s
    pred:      np.ndarray,   # (T, H, W) mm/s
    lat_ax:    np.ndarray,   # mm
    dep_ax:    np.ndarray,   # mm
    t_ax:      np.ndarray,   # ms
    clim:      float,        # colour saturation mm/s
    title_base:str,
    out_path:  Path,
    fps:       int = 12,
):
    """
    Write a 3-panel (GT | Prediction | |Error|) MP4 or GIF video.

    Layout mirrors the MATLAB IQ2VEl.m movie:
      - jet colormap, clim ±clim for velocity panels
      - 0..clim for the error panel
      - lateral distance (mm) x-axis, axial depth (mm) y-axis
    """
    T, H, W = gt.shape
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]   # (left,right,bottom,top) for imshow

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("black")
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.04],
                           wspace=0.08, left=0.06, right=0.96, top=0.88, bottom=0.12)

    ax_gt  = fig.add_subplot(gs[0])
    ax_pr  = fig.add_subplot(gs[1])
    ax_err = fig.add_subplot(gs[2])
    ax_cb  = fig.add_subplot(gs[3])

    for ax in (ax_gt, ax_pr, ax_err):
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Initial images
    im_gt  = ax_gt.imshow(gt[0],            cmap="jet", vmin=-clim, vmax=clim,
                          aspect="auto", origin="upper", extent=extent)
    im_pr  = ax_pr.imshow(pred[0],           cmap="jet", vmin=-clim, vmax=clim,
                          aspect="auto", origin="upper", extent=extent)
    im_err = ax_err.imshow(np.abs(gt[0] - pred[0]), cmap="hot", vmin=0, vmax=clim,
                           aspect="auto", origin="upper", extent=extent)

    # Shared velocity colourbar (right panel)
    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Particle velocity (mm/s)", color="white", fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Axis labels
    for ax, ttl in ((ax_gt, "Ground Truth"), (ax_pr, "Model Prediction"), (ax_err, "|Error|")):
        ax.set_xlabel("Lateral distance (mm)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Axial depth (mm)",      fontsize=11, fontweight="bold")
        ax.set_title(ttl, fontsize=13, fontweight="bold")

    sup = fig.suptitle("", color="white", fontsize=12, fontweight="bold")

    def _update(frame_idx: int):
        t_ms = t_ax[frame_idx]
        im_gt.set_data(gt[frame_idx])
        im_pr.set_data(pred[frame_idx])
        im_err.set_data(np.abs(gt[frame_idx] - pred[frame_idx]))
        sup.set_text(f"{title_base}  |  t = {t_ms:.2f} ms  (frame {frame_idx + 1}/{T})")
        return im_gt, im_pr, im_err, sup

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, _update, frames=T, blit=True, interval=1000 // fps)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _try_ffmpeg():
        writer = FFMpegWriter(fps=fps, metadata={"title": title_base},
                              extra_args=["-pix_fmt", "yuv420p"])
        save_path = out_path.with_suffix(".mp4")
        anim.save(str(save_path), writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": "black"})
    else:
        # Fall back to GIF if ffmpeg is unavailable
        writer = PillowWriter(fps=fps)
        save_path = out_path.with_suffix(".gif")
        anim.save(str(save_path), writer=writer,
                  savefig_kwargs={"facecolor": "black"})

    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Summary metrics plot
# ---------------------------------------------------------------------------

def save_metrics_summary(
    metrics_list: List[Dict],
    names:        List[str],
    out_path:     Path,
):
    """Bar chart of MAE per validation sample plus a global summary."""
    maes  = [m["mae"]  for m in metrics_list]
    rmses = [m["rmse"] for m in metrics_list]
    corrs = [m["corr"] for m in metrics_list]

    # Short label
    labels = [n.split("_w")[0][-28:] for n in names]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(names) * 0.4), 10))
    fig.suptitle("Validation — per-sample metrics", fontsize=14, fontweight="bold")

    for ax, vals, ylabel, color in zip(
        axes,
        [maes, rmses, corrs],
        ["MAE (mm/s)", "RMSE (mm/s)", "Pearson r"],
        ["steelblue", "coral", "seagreen"],
    ):
        ax.bar(range(len(vals)), vals, color=color, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        mu = np.nanmean(vals)
        ax.axhline(mu, color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {mu:.3f}")
        ax.legend(fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics summary → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate GT vs Prediction velocity videos")
    parser.add_argument(
        "--model_path", type=str,
        default="SWE_PINN_outputs_fullscale_velocity_v1_best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/fullscale_velocity",
        help="Root data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/fullscale_velocity_viz",
        help="Directory to save videos and plots",
    )
    parser.add_argument(
        "--clim", type=float, default=None,
        help="Colour saturation limit in mm/s (default: 99th pct of GT magnitude)",
    )
    parser.add_argument(
        "--fps", type=int, default=12,
        help="Frames per second for output video",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Process at most this many val samples (default: all)",
    )
    parser.add_argument(
        "--per_sample_iq_norm", action="store_true",
        help="Use per-sample IQ normalisation (v2 models)",
    )
    parser.add_argument(
        "--per_sample_vel_norm", action="store_true",
        help="Use per-sample velocity normalisation (v2 models)",
    )
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir    = PROJECT_ROOT / args.output_dir
    model_path = PROJECT_ROOT / args.model_path
    data_dir   = PROJECT_ROOT / args.data_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Velocity Prediction Visualisation")
    print(f"  model  : {model_path.name}")
    print(f"  data   : {data_dir}")
    print(f"  output : {out_dir}")
    print(f"  device : {device}")
    print(f"{'='*60}\n")

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("[1/4] Loading model …")
    model, cfg, data_info = load_model(str(model_path), device)

    iq_norm  = data_info["iq_norm_scalar"]
    vel_norm = data_info["vel_norm_scalar"]
    print(f"  IQ norm = {iq_norm:.2e}   vel norm = {vel_norm:.3f} mm/s\n")

    # ── 2. Load per-sample metadata (for frame_time_s, prf, source) ───────
    print("[2/4] Loading dataset metadata …")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    meta_by_name: Dict[str, Dict] = {s["name"]: s for s in meta["val"]}
    H_ROI = 116   # original ROI height before padding

    # ── 3. Build val dataset ───────────────────────────────────────────────
    print("[3/4] Building validation dataset …")
    val_ds = IQVelocityDataset(
        data_dir         = data_dir,
        split            = "val",
        iq_norm_scalar   = iq_norm if not args.per_sample_iq_norm else None,
        vel_norm_scalar  = vel_norm if not args.per_sample_vel_norm else None,
        per_sample_iq_norm  = args.per_sample_iq_norm,
        per_sample_vel_norm = args.per_sample_vel_norm,
    )
    print(f"  {len(val_ds)} validation samples")
    if args.per_sample_iq_norm:
        print("  Using per-sample IQ normalisation")
    if args.per_sample_vel_norm:
        print("  Using per-sample velocity normalisation")
    print()

    n_samples = len(val_ds) if args.max_samples is None else min(args.max_samples, len(val_ds))

    # ── 4. Inference + video ───────────────────────────────────────────────
    print(f"[4/4] Generating {n_samples} videos …\n")

    all_metrics: List[Dict]  = []
    all_names:   List[str]   = []

    for idx in tqdm(range(n_samples), desc="Samples", unit="sample"):
        batch     = val_ds[idx]
        name      = batch["name"]
        sample_meta = meta_by_name.get(name, {})

        # Wrap scalars in batch dim for the model
        batch_tensors = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items() if k != "name"
        }
        batch_tensors["name"] = name

        # Run inference
        gt_mm, pr_mm, mask = infer_sample(
            model, batch_tensors, device, vel_norm, H_ROI,
            per_sample_vel_norm=args.per_sample_vel_norm,
        )

        T, H, W = gt_mm.shape

        # Physical axes
        dep_ax = make_axes(H, W)[1]
        lat_ax = make_axes(H, W)[0]

        frame_time_s = sample_meta.get("frame_time_s", 1.0 / sample_meta.get("prf_hz", 10000.0))
        t_ax = np.arange(T) * frame_time_s * 1e3   # ms

        # Colour saturation
        if args.clim is not None:
            clim = args.clim
        else:
            clim = float(np.percentile(np.abs(gt_mm), 99))
            clim = max(clim, 5.0)   # floor so constant-zero targets still render

        # Source / frequency info for title
        src   = sample_meta.get("source", "")
        freq  = sample_meta.get("frequency", "")
        prf   = sample_meta.get("prf_hz", "")
        focus = sample_meta.get("focus", "")
        title_base = f"{src} | {focus} | {freq} (PRF={prf:.0f} Hz) | {name[-18:]}"

        # Save video
        video_path = out_dir / f"{name}.mp4"
        saved_path = make_video(
            gt        = gt_mm,
            pred      = pr_mm,
            lat_ax    = lat_ax,
            dep_ax    = dep_ax,
            t_ax      = t_ax,
            clim      = clim,
            title_base = title_base,
            out_path  = video_path,
            fps       = args.fps,
        )

        # Metrics
        m = compute_metrics(gt_mm, pr_mm)
        all_metrics.append(m)
        all_names.append(name)

        tqdm.write(
            f"  {name[-40:]:<40}  "
            f"MAE={m['mae']:6.2f}  RMSE={m['rmse']:6.2f}  r={m['corr']:.3f}  "
            f"clim=±{clim:.0f}  → {saved_path.name}"
        )

    # ── 5. Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Processed {n_samples} samples")
    print(f"  Mean MAE  : {np.nanmean([m['mae']  for m in all_metrics]):.3f} mm/s")
    print(f"  Mean RMSE : {np.nanmean([m['rmse'] for m in all_metrics]):.3f} mm/s")
    print(f"  Mean r    : {np.nanmean([m['corr'] for m in all_metrics]):.4f}")
    print("="*60 + "\n")

    # Metrics bar chart
    if len(all_metrics) > 1:
        save_metrics_summary(all_metrics, all_names, out_dir / "metrics_summary.png")

    # JSON log
    log = [{"name": n, **m} for n, m in zip(all_names, all_metrics)]
    with open(out_dir / "metrics_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Metrics log → {out_dir / 'metrics_log.json'}")
    print(f"\nAll files saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
