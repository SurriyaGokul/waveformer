#!/usr/bin/env python3
"""
Virtual 6.25 kHz velocity prediction from 250 Hz (simulated) IQ data.

Idea
----
We simulate a 250 Hz acquisition by taking every 2nd 500 Hz IQ frame
(INPUT_STEP=2), giving an inter-frame spacing of dt = 2/500 = 4.0 ms.
The model still receives 5 IQ frames and upsamples by 25×, so:

    dt_out = dt_in / upsampling_factor = 4.0 ms / 25 = 0.16 ms  →  6250 Hz

Each GT interval now covers 2 native 500 Hz frames (4 ms), so we compare
against the mean of 2 GT frames per predicted block.

Preprocessing applied to the raw IQ (matching training exactly):
  1. Extract spatial ROI stored in metadata.json
  2. Clutter filter: subtract temporal mean over the full 50-frame window
  3. Take 5 sparse frames [0, 2, 4, 6, 8] (every 2nd 500Hz frame)
  4. Per-sample IQ normalisation: divide by 99th percentile of |IQ|
  5. Per-sample vel normalisation: divide output by 99th percentile of |GT vel|

Usage
-----
    python scripts/predict_6250hz_from_250hz_25x.py
    python scripts/predict_6250hz_from_250hz_25x.py --no_video
    python scripts/predict_6250hz_from_250hz_25x.py --sample_name 2ft_10mmfocus_Plane_4_500Hz_20260129T130420_w0
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.architectures.velocity_unet import ComplexToRealUNet

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX   = 0.0507
LATERAL_MM_PER_PX = 0.1013
H_ROI, W_ROI      = 116, 160
UPSAMPLING_FACTOR  = 25
MODEL_PATH_DEFAULT = "outputs/fullscale_velocity_25x/best_model.pt"
DATA_META_PATH     = "data/fullscale_velocity/metadata.json"
OUTPUT_DIR_DEFAULT = "outputs/6250hz_predictions_25x"

# 250Hz simulation constants
INPUT_STEP      = 2        # take every 2nd 500Hz frame
INPUT_DT_MS     = 4.0      # ms between selected input frames (1/250 * 1000)
OUTPUT_DT_MS    = INPUT_DT_MS / UPSAMPLING_FACTOR  # 0.16 ms
OUTPUT_FS_HZ    = 1.0 / (OUTPUT_DT_MS * 1e-3)     # 6250 Hz
N_GT_PER_INPUT  = 2        # number of 500Hz GT frames per input interval
N_ACTIVE_GROUPS = 4        # number of input intervals to compare
N_ACTIVE_GT     = N_ACTIVE_GROUPS * N_GT_PER_INPUT  # 8


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str) -> ComplexToRealUNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ComplexToRealUNet(
        in_channels=1,
        base_channels=32,
        encoder_depths=[32, 64, 128, 256],
        upsampling_factor=UPSAMPLING_FACTOR,
        use_motion_interpolation=True,
        complex_to_real_mode="concat",
        use_naive_bn=True,
        flow_hidden_channels=32,
        flow_scale=0.1,
        use_prf_conditioning=False,
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"  Model loaded (epoch={epoch}, best_val_loss={val_loss:.6f})")
    return model


# ---------------------------------------------------------------------------
# IQ preprocessing — sparse frames simulating 250Hz
# ---------------------------------------------------------------------------
def load_sparse_iq(sample_meta: dict, start_frame: int = 0) -> np.ndarray:
    """
    Load 5 sparse IQ frames simulating 250Hz by taking every INPUT_STEP-th
    500Hz frame. Clutter filter applied over full 50-frame window.

    Frame indices: [start, start+2, start+4, start+6, start+8]

    Returns: (5, H, W) complex64
    """
    raw = np.load(sample_meta["original_path"])           # (nRows, nCols, T_full)
    r   = sample_meta["roi"]
    win = sample_meta["window"]

    # Extract spatial ROI and temporal window
    iq_win = raw[
        r["min_row"] : r["max_row"] + 1,
        r["min_col"] : r["max_col"] + 1,
        win[0] : win[1],
    ].astype(np.complex64)                                # (H, W, T_win)

    # Clutter filter: subtract temporal mean over full window (same as training)
    iq_detrend = iq_win - iq_win.mean(axis=2, keepdims=True)   # (H, W, T_win)

    T_win = iq_detrend.shape[2]
    indices = [start_frame + i * INPUT_STEP for i in range(5)]
    if indices[-1] >= T_win:
        raise ValueError(
            f"Max index {indices[-1]} >= T_win={T_win}. "
            f"Max start_frame = {T_win - 4 * INPUT_STEP - 1}."
        )

    frames = np.stack([iq_detrend[:, :, idx] for idx in indices], axis=0)  # (5, H, W)
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(
    model: ComplexToRealUNet,
    iq_frames: np.ndarray,
    gt_vel: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    iq_frames : (5, H, W) complex64 — sparse IQ frames
    gt_vel    : (101, H, W) float32  — GT velocity at native PRF rate (mm/s)

    Returns: (pred_mm, iq_scale, vel_scale)
    """
    # Per-sample IQ normalisation
    iq_scale = float(np.percentile(np.abs(iq_frames), 99))
    iq_norm  = (iq_frames / max(iq_scale, 1e-12)).astype(np.complex64)

    # Per-sample vel scale from the GT (same as training)
    vel_scale = float(np.percentile(np.abs(gt_vel), 99))

    # Build real/imag input tensor, pad H to multiple of 16 (116 → 128)
    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)  # (2,5,H,W)
    pad_h = (16 - H_ROI % 16) % 16   # = 12
    inp_pad = np.pad(inp, ((0, 0), (0, 0), (0, pad_h), (0, 0)))              # (2,5,128,W)

    with torch.no_grad():
        out = model(torch.from_numpy(inp_pad).unsqueeze(0))
        if isinstance(out, dict):
            out = out["velocity"]
    pred_norm = out.squeeze().numpy()            # (101, 128, W)
    pred_mm   = pred_norm[:, :H_ROI, :] * vel_scale   # → mm/s

    return pred_mm.astype(np.float32), iq_scale, vel_scale


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def compute_active_steps(pred_mm: np.ndarray, gt_vel: np.ndarray):
    """
    Average predicted frames within each input interval and corresponding GT frames.

    Returns: (pred_steps, gt_steps) each (N_ACTIVE_GROUPS, H, W)
    """
    pred_steps = np.stack(
        [pred_mm[k * UPSAMPLING_FACTOR:(k + 1) * UPSAMPLING_FACTOR].mean(0)
         for k in range(N_ACTIVE_GROUPS)]
    )  # (4, H, W)
    gt_steps = np.stack(
        [gt_vel[k * N_GT_PER_INPUT:(k + 1) * N_GT_PER_INPUT].mean(0)
         for k in range(N_ACTIVE_GROUPS)]
    )  # (4, H, W)
    return pred_steps, gt_steps


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def _try_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def write_video(
    pred_mm    : np.ndarray,          # (101, H, W) mm/s at 6.25kHz
    gt_500hz   : np.ndarray,          # (101+, H, W) mm/s at 500Hz (native)
    start_frame: int,
    sample_name: str,
    out_path   : Path,
    fps        : int = 15,
) -> Path:
    """
    3-panel video: GT (500Hz step) | Prediction (6.25kHz) | |Error|
    """
    T    = pred_mm.shape[0]    # 101
    t_ms = np.arange(T) * OUTPUT_DT_MS

    # Colour limit: peak of GT active region
    clim = float(np.abs(gt_500hz[:N_ACTIVE_GT + 1]).max())
    clim = max(clim, 0.2)

    # Physical axes
    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(20, 6), facecolor="black")
    gs  = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.07, left=0.05, right=0.97, top=0.86, bottom=0.12,
    )
    ax_gt  = fig.add_subplot(gs[0])
    ax_pr  = fig.add_subplot(gs[1])
    ax_err = fig.add_subplot(gs[2])
    ax_cb  = fig.add_subplot(gs[3])

    panel_titles = [
        f"GT Loupas @ 500 Hz\n(step — {INPUT_DT_MS:.0f} ms / interval, {N_GT_PER_INPUT} GT frames each)",
        f"Model Prediction @ {OUTPUT_FS_HZ/1000:.2f} kHz\n(\u0394t = {OUTPUT_DT_MS:.4f} ms / frame)",
        "|Error| vs GT step",
    ]
    for ax, ttl in zip([ax_gt, ax_pr, ax_err], panel_titles):
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

    def gt_step(fi: int) -> np.ndarray:
        interval_idx = fi // UPSAMPLING_FACTOR
        sub_fi       = fi % UPSAMPLING_FACTOR
        local_gt     = min(sub_fi * N_GT_PER_INPUT // UPSAMPLING_FACTOR, N_GT_PER_INPUT - 1)
        global_gt    = min(interval_idx * N_GT_PER_INPUT + local_gt, gt_500hz.shape[0] - 1)
        return gt_500hz[global_gt]

    im_gt  = ax_gt.imshow(gt_step(0), **kw)
    im_pr  = ax_pr.imshow(pred_mm[0], **kw)
    im_err = ax_err.imshow(np.abs(pred_mm[0] - gt_step(0)),
                           cmap="hot", vmin=0, vmax=clim,
                           aspect="auto", origin="upper", extent=extent)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Particle velocity (mm/s)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    sup = fig.suptitle("", color="white", fontsize=10, fontweight="bold")

    def _update(fi: int):
        gt_f         = gt_step(fi)
        pr_f         = pred_mm[fi]
        interval_idx = fi // UPSAMPLING_FACTOR
        sub_fi       = fi % UPSAMPLING_FACTOR
        active       = interval_idx < 2
        active_str   = "  \u25c0 WAVE ACTIVE" if active else ""

        im_gt.set_data(gt_f)
        im_pr.set_data(pr_f)
        im_err.set_data(np.abs(pr_f - gt_f))

        sup.set_text(
            f"Virtual 6.25 kHz SWE  |  {sample_name}  |  "
            f"t = {t_ms[fi]:.3f} ms  (frame {fi+1}/{T})\n"
            f"GT step: interval {interval_idx+1}/{N_ACTIVE_GROUPS}  "
            f"(sub-frame {sub_fi+1}/{UPSAMPLING_FACTOR})"
            f"{active_str}"
        )
        return im_gt, im_pr, im_err, sup

    anim = FuncAnimation(fig, _update, frames=T, blit=True, interval=1000 // fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _try_ffmpeg():
        writer   = FFMpegWriter(fps=fps, metadata={"title": sample_name},
                                extra_args=["-pix_fmt", "yuv420p"])
        save_pth = out_path.with_suffix(".mp4")
        anim.save(str(save_pth), writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": "black"})
    else:
        writer   = PillowWriter(fps=fps)
        save_pth = out_path.with_suffix(".gif")
        anim.save(str(save_pth), writer=writer,
                  savefig_kwargs={"facecolor": "black"})

    plt.close(fig)
    return save_pth


# ---------------------------------------------------------------------------
# Static comparison figure (active frames only)
# ---------------------------------------------------------------------------
def write_static_figure(
    pred_mm    : np.ndarray,
    gt_500hz   : np.ndarray,
    sample_name: str,
    out_path   : Path,
) -> None:
    """
    Grid: rows = GT groups 0-3 (mean of N_GT_PER_INPUT frames each),
    columns = model sub-frames within each interval (sub-frames 0, 6, 16, 24).
    """
    sub_show = [0, 6, 16, 24]

    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    clim = float(np.abs(gt_500hz[:N_ACTIVE_GT]).max())
    clim = max(clim, 0.2)

    n_cols = len(sub_show) + 1
    fig, axes = plt.subplots(N_ACTIVE_GROUPS, n_cols,
                             figsize=(3.5 * n_cols, 4 * N_ACTIVE_GROUPS),
                             facecolor="black")
    fig.suptitle(
        f"Virtual 6.25 kHz — GT avg (left) vs model sub-frames within each {INPUT_DT_MS:.0f} ms interval\n{sample_name}",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    for row in range(N_ACTIVE_GROUPS):
        gt_start = row * N_GT_PER_INPUT
        gt_avg   = gt_500hz[gt_start:gt_start + N_GT_PER_INPUT].mean(axis=0)
        t0 = row * INPUT_DT_MS

        # GT column
        ax = axes[row, 0]
        ax.imshow(gt_avg, cmap="jet", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", extent=extent)
        ax.set_title(
            f"GT avg  t={t0:.0f}\u2013{t0+INPUT_DT_MS:.0f} ms\n"
            f"(mean of frames {gt_start}\u2013{gt_start+N_GT_PER_INPUT-1})",
            color="white", fontsize=8, fontweight="bold"
        )

        # Model sub-frames
        for col, si in enumerate(sub_show):
            fi  = row * UPSAMPLING_FACTOR + si
            t_f = fi * OUTPUT_DT_MS
            ax  = axes[row, col + 1]
            ax.imshow(pred_mm[fi], cmap="jet", vmin=-clim, vmax=clim,
                      aspect="auto", origin="upper", extent=extent)
            ax.set_title(
                f"Model  t={t_f:.2f} ms\n(6.25kHz sub-frame {si+1}/{UPSAMPLING_FACTOR})",
                color="white", fontsize=8
            )

        for ax in axes[row]:
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_edgecolor("white")
            ax.tick_params(colors="white", labelsize=7)
            if row == N_ACTIVE_GROUPS - 1:
                ax.set_xlabel("Lateral (mm)", color="white", fontsize=8)

        axes[row, 0].set_ylabel("Axial (mm)", color="white", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-method comparison bar chart
# ---------------------------------------------------------------------------
def write_comparison_barchart(
    out_dir: Path,
    current_summary: dict,
    save_path: Optional[Path] = None,
) -> None:
    """
    3-panel bar chart comparing 12.5kHz (500Hz input), 6.25kHz (250Hz input),
    and 2.5kHz (100Hz input) methods (if available).
    """
    methods = {}

    # Try to load 12.5kHz summary (500Hz input, 25x)
    p125 = Path("outputs/12500hz_predictions_25x/summary.json")
    if p125.exists():
        with open(p125) as f:
            s = json.load(f)
        methods["12.5 kHz\n(500Hz IQ)"] = {
            "mae":      s.get("mean_mae", float("nan")),
            "r_active": s.get("mean_r_active", float("nan")),
            "ssim":     s.get("mean_ssim", None),
        }

    # Current (6.25kHz)
    methods["6.25 kHz\n(250Hz IQ)"] = {
        "mae":      current_summary.get("mean_mae", float("nan")),
        "r_active": current_summary.get("mean_r_active", float("nan")),
        "ssim":     current_summary.get("mean_ssim", None),
    }

    # Try to load 2.5kHz summary (100Hz input, 25x)
    p25 = Path("outputs/2500hz_predictions_25x/summary.json")
    if p25.exists():
        with open(p25) as f:
            s = json.load(f)
        methods["2.5 kHz\n(100Hz IQ)"] = {
            "mae":      s.get("mean_mae", float("nan")),
            "r_active": s.get("mean_r_active", float("nan")),
            "ssim":     s.get("mean_ssim", None),
        }

    if len(methods) < 1:
        return

    labels = list(methods.keys())
    mae_vals  = [methods[k]["mae"]      for k in labels]
    r_vals    = [methods[k]["r_active"] for k in labels]
    ssim_vals = [methods[k]["ssim"]     for k in labels]

    colors = ["#4CAF50", "#2196F3", "#FF9800"][:len(labels)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#1a1a2e")
    fig.suptitle("Cross-Method Comparison: SWE Velocity Prediction (25x upsampling)",
                 color="white", fontsize=13, fontweight="bold")

    panels = [
        (axes[0], mae_vals,  "MAE (mm/s)",          "Mean MAE — lower is better"),
        (axes[1], r_vals,    "r (active window)",    "Pearson r — higher is better"),
        (axes[2], ssim_vals, "SSIM",                 "SSIM — higher is better"),
    ]

    for ax, vals, ylabel, title in panels:
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

        x = np.arange(len(labels))
        for xi, (v, c, lbl) in enumerate(zip(vals, colors, labels)):
            if v is None or np.isnan(float(v if v is not None else float("nan"))):
                bar = ax.bar(xi, 0.0, color=c, alpha=0.5, hatch="//", width=0.5)
                ax.text(xi, 0.02, "N/A", ha="center", va="bottom",
                        color="white", fontsize=9, fontweight="bold")
            else:
                bar = ax.bar(xi, float(v), color=c, alpha=0.85, width=0.5)
                ax.text(xi, float(v) + 0.005 * (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
                        f"{float(v):.3f}", ha="center", va="bottom",
                        color="white", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=10)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")

    plt.tight_layout()
    final_path = save_path or (out_dir / "comparison_barchart.png")
    fig.savefig(str(final_path), dpi=300, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"Comparison barchart \u2192 {final_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict virtual 6.25kHz velocity from simulated 250Hz IQ (every 2nd 500Hz frame, 25x upsampling)"
    )
    parser.add_argument("--model_path",  default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First 500Hz frame index; sparse frames: [start, start+2, ..., start+8]")
    parser.add_argument("--sample_name", default=None,
                        help="Process one named sample only; omit to process all 500Hz _w0")
    parser.add_argument("--fps",         type=int, default=15,
                        help="Video fps (default 15)")
    parser.add_argument("--no_video",    action="store_true",
                        help="Skip video generation (faster metrics-only run)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    print(f"Loading metadata from {DATA_META_PATH} ...")
    with open(DATA_META_PATH) as f:
        meta = json.load(f)

    inf_lookup = {s["name"]: s for s in meta["inference"]}

    if args.sample_name:
        samples = [inf_lookup[args.sample_name]]
    else:
        samples = [s for s in meta["inference"]
                   if s.get("frequency") == "500Hz" and s["name"].endswith("_w0")]

    # Avoid overwriting full-dataset publication outputs when running a single sample.
    detail_png = out_dir / (
        f"detail_{args.sample_name}.png" if args.sample_name else "best_sample_detail.png"
    )
    summary_json = out_dir / (
        f"summary_{args.sample_name}.json" if args.sample_name else "summary.json"
    )
    comparison_png = out_dir / (
        f"comparison_barchart_{args.sample_name}.png" if args.sample_name else "comparison_barchart.png"
    )

    print(f"\nProcessing {len(samples)} samples  "
          f"(start_frame={args.start_frame}, INPUT_STEP={INPUT_STEP}, "
          f"OUTPUT_FS={OUTPUT_FS_HZ:.0f}Hz, no_video={args.no_video})\n")

    all_metrics = []
    best_r, best_name, best_pred, best_gt = -np.inf, None, None, None

    for i, sm in enumerate(samples):
        name = sm["name"]
        print(f"[{i+1:2d}/{len(samples)}] {name}")

        # --- Load sparse IQ ---
        try:
            iq_frames = load_sparse_iq(sm, start_frame=args.start_frame)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # --- Load GT velocity ---
        gt_path = Path("data/fullscale_velocity/inference") / f"velocity_{name}.npy"
        gt_vel  = np.load(str(gt_path)).astype(np.float32)

        # --- Inference ---
        pred_mm, iq_scale, vel_scale = infer(model, iq_frames, gt_vel)

        # --- Metrics ---
        pred_steps, gt_steps = compute_active_steps(pred_mm, gt_vel)

        mae    = float(np.abs(pred_steps - gt_steps).mean())
        T_common = min(pred_mm.shape[0], gt_vel.shape[0])
        r_full = np.nan_to_num(float(np.corrcoef(pred_mm[:T_common].flatten(), gt_vel[:T_common].flatten())[0, 1]))
        r_act  = np.nan_to_num(float(np.corrcoef(pred_steps.flatten(), gt_steps.flatten())[0, 1]))

        data_range = max(float(np.abs(gt_steps).max() * 2), 1e-6)
        ssim_vals_list = [
            structural_similarity(
                pred_steps[k].astype(np.float64),
                gt_steps[k].astype(np.float64),
                data_range=data_range,
            )
            for k in range(N_ACTIVE_GROUPS)
        ]
        ssim_mean = float(np.mean(ssim_vals_list))

        print(f"  iq_scale={iq_scale:.3e}  vel_scale={vel_scale:.4f} mm/s")
        print(f"  MAE={mae:.4f} mm/s  r_full={r_full:.4f}  r_active={r_act:.4f}  SSIM={ssim_mean:.4f}")

        all_metrics.append(dict(
            name=name, mae=mae, r_full=r_full, r_active=r_act,
            ssim_mean=ssim_mean, vel_scale=vel_scale,
        ))

        if r_act > best_r:
            best_r, best_name = r_act, name
            best_pred = pred_mm.copy()
            best_gt   = gt_vel.copy()

        # --- Save prediction npy ---
        np.save(str(out_dir / f"velocity_6250hz_{name}.npy"), pred_mm)
        np.save(str(out_dir / f"gt_velocity_{name}.npy"),    gt_vel)

        # --- Per-sample metadata ---
        indices_used = [args.start_frame + i * INPUT_STEP for i in range(5)]
        with open(str(out_dir / f"metadata_{name}.json"), "w") as mf:
            json.dump({
                "name": name,
                "approach": "sparse_iq_frames_250hz_simulation_25x",
                "input_step": INPUT_STEP,
                "start_frame": args.start_frame,
                "input_iq_frames": indices_used,
                "input_iq_dt_ms": INPUT_DT_MS,
                "upsampling_factor": UPSAMPLING_FACTOR,
                "output_dt_ms": OUTPUT_DT_MS,
                "output_Fs_hz": OUTPUT_FS_HZ,
                "output_n_frames": int(pred_mm.shape[0]),
                "output_total_duration_ms": OUTPUT_DT_MS * (pred_mm.shape[0] - 1),
                "velocity_unit": "mm/s",
                "vel_scale_mms": vel_scale,
                "iq_scale": iq_scale,
                "n_gt_per_input": N_GT_PER_INPUT,
                "n_active_groups": N_ACTIVE_GROUPS,
                "metrics": {
                    "mae_active_mm_s": mae,
                    "r_full": r_full,
                    "r_active": r_act,
                    "ssim_mean": ssim_mean,
                    "ssim_per_group": ssim_vals_list,
                },
                "note": (
                    f"5 sparse 500Hz IQ frames at step={INPUT_STEP} "
                    f"(indices {indices_used}, dt={INPUT_DT_MS}ms each) "
                    f"\u2192 model upsamples by {UPSAMPLING_FACTOR}\u00d7 "
                    f"\u2192 output at {OUTPUT_FS_HZ:.0f}Hz (dt={OUTPUT_DT_MS:.4f}ms). "
                    "Clutter filter (temporal mean subtraction) applied before IQ input. "
                    "Per-sample IQ and velocity normalisation applied."
                ),
                "tissue_properties": sm.get("tissue_properties", {}),
            }, mf, indent=2)

        # --- Per-sample video ---
        if not args.no_video:
            vid_path = out_dir / f"{name}_6_25khz"
            saved = write_video(pred_mm, gt_vel,
                                start_frame=args.start_frame,
                                sample_name=name,
                                out_path=vid_path,
                                fps=args.fps)
            print(f"  Video \u2192 {saved}")

    # --- Best-sample detailed static figure ---
    if best_pred is not None:
        write_static_figure(best_pred, best_gt, best_name, detail_png)
        print(f"\nBest sample: {best_name}  r_active={best_r:.4f}")
        print(f"Static figure \u2192 {detail_png}")

    # --- Summary ---
    if all_metrics:
        rs    = [m["r_active"]  for m in all_metrics]
        maes  = [m["mae"]       for m in all_metrics]
        ssims = [m["ssim_mean"] for m in all_metrics]

        print(f"\n=== Summary ({len(all_metrics)} samples) ===")
        print(f"Mean r (active window): {np.mean(rs):.4f}  \u00b1 {np.std(rs):.4f}")
        print(f"Mean MAE:               {np.mean(maes):.4f} mm/s")
        print(f"Mean SSIM:              {np.mean(ssims):.4f} \u00b1 {np.std(ssims):.4f}")

        summary = {
            "description": "Virtual 6.25kHz predictions from sparse 250Hz-simulated IQ (every 2nd 500Hz frame, 25x upsampling)",
            "input_step": INPUT_STEP,
            "input_dt_ms": INPUT_DT_MS,
            "output_Fs_hz": OUTPUT_FS_HZ,
            "output_dt_ms": OUTPUT_DT_MS,
            "start_frame": args.start_frame,
            "n_gt_per_input": N_GT_PER_INPUT,
            "mean_r_active": float(np.mean(rs)),
            "std_r_active":  float(np.std(rs)),
            "mean_mae":      float(np.mean(maes)),
            "std_mae":       float(np.std(maes)),
            "mean_ssim":     float(np.mean(ssims)),
            "std_ssim":      float(np.std(ssims)),
            "samples": all_metrics,
        }
        with open(str(summary_json), "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Summary  \u2192 {summary_json}")

        # --- Comparison barchart ---
        write_comparison_barchart(out_dir, summary, save_path=comparison_png)


if __name__ == "__main__":
    main()
