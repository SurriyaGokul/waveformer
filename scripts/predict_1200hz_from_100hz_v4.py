#!/usr/bin/env python3
"""
Virtual 1.2 kHz velocity prediction from 100 Hz (simulated) IQ — WavePINN v4.

Uses WavePINNv4 (Conditional Wave Propagation Network) with dt_seconds
parameter for PRF-aware inference via learned wave dynamics.

100 Hz is simulated by taking every 5th 500 Hz frame (dt = 10 ms).

Usage
-----
    python scripts/predict_1200hz_from_100hz_v4.py
    python scripts/predict_1200hz_from_100hz_v4.py --no_video
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.architectures.wavepinn_v4 import WavePINNv4, create_wavepinn_v4

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX   = 0.0507
LATERAL_MM_PER_PX = 0.1013
H_ROI, W_ROI      = 116, 160
UPSAMPLING_FACTOR  = 12
MODEL_PATH_DEFAULT = "outputs/wavepinn_v4/best_model.pt"
DATA_META_PATH     = "data/fullscale_velocity/metadata.json"
OUTPUT_DIR_DEFAULT = "outputs/1200hz_predictions_v4"

# 100Hz simulation
INPUT_STEP      = 5
INPUT_DT_S      = INPUT_STEP / 500.0      # 10 ms
INPUT_DT_MS     = INPUT_DT_S * 1e3
OUTPUT_DT_MS    = INPUT_DT_MS / UPSAMPLING_FACTOR  # 0.8333 ms
OUTPUT_FS_HZ    = 1.0 / (OUTPUT_DT_MS * 1e-3)     # 1200 Hz
N_GT_PER_INPUT  = 5
N_ACTIVE_GROUPS = 4
N_ACTIVE_GT     = N_ACTIVE_GROUPS * N_GT_PER_INPUT


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str) -> WavePINNv4:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_wavepinn_v4().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    cs = ckpt.get("learned_cs", "?")
    print(f"  WavePINNv4 loaded (epoch={epoch}, best_val_loss={val_loss:.6f}, cs={cs})")
    return model


# ---------------------------------------------------------------------------
# IQ preprocessing — sparse frames simulating 100Hz
# ---------------------------------------------------------------------------
def load_sparse_iq(sample_meta: dict, start_frame: int = 0) -> np.ndarray:
    """Load 5 sparse IQ frames simulating 100Hz. Returns (5, H, W) complex64."""
    raw = np.load(sample_meta["original_path"])
    r   = sample_meta["roi"]
    win = sample_meta["window"]

    iq_win = raw[
        r["min_row"] : r["max_row"] + 1,
        r["min_col"] : r["max_col"] + 1,
        win[0] : win[1],
    ].astype(np.complex64)

    iq_detrend = iq_win - iq_win.mean(axis=2, keepdims=True)

    T_win = iq_detrend.shape[2]
    indices = [start_frame + i * INPUT_STEP for i in range(5)]
    if indices[-1] >= T_win:
        raise ValueError(f"Max index {indices[-1]} >= T_win={T_win}")

    frames = np.stack([iq_detrend[:, :, idx] for idx in indices], axis=0)
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(model: WavePINNv4, iq_frames: np.ndarray, gt_vel: np.ndarray):
    """Returns (pred_mm, iq_scale, vel_scale)."""
    iq_scale = float(np.percentile(np.abs(iq_frames), 99))
    iq_norm  = (iq_frames / max(iq_scale, 1e-12)).astype(np.complex64)
    vel_scale = float(np.percentile(np.abs(gt_vel), 99))

    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)
    pad_h = (16 - H_ROI % 16) % 16
    inp_pad = np.pad(inp, ((0, 0), (0, 0), (0, pad_h), (0, 0)))

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(
            torch.from_numpy(inp_pad).unsqueeze(0).to(device),
            dt_seconds=INPUT_DT_S,
        )
        if isinstance(out, dict):
            out = out["velocity"]
    pred_norm = out.squeeze().cpu().numpy()
    pred_mm   = pred_norm[:, :H_ROI, :] * vel_scale

    return pred_mm.astype(np.float32), iq_scale, vel_scale


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def compute_active_steps(pred_mm, gt_vel):
    pred_steps = np.stack(
        [pred_mm[k * UPSAMPLING_FACTOR:(k+1) * UPSAMPLING_FACTOR].mean(0)
         for k in range(N_ACTIVE_GROUPS)]
    )
    gt_steps = np.stack(
        [gt_vel[k * N_GT_PER_INPUT:(k+1) * N_GT_PER_INPUT].mean(0)
         for k in range(N_ACTIVE_GROUPS)]
    )
    return pred_steps, gt_steps


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def _try_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") is not None


def write_video(pred, gt, sample_name, out_path, fps=15):
    T    = pred.shape[0]
    t_ms = np.arange(T) * OUTPUT_DT_MS
    clim = max(float(np.abs(gt[:N_ACTIVE_GT + 1]).max()), 0.2)

    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(20, 6), facecolor="black")
    gs  = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.04],
                            wspace=0.07, left=0.05, right=0.97, top=0.86, bottom=0.12)
    ax_gt  = fig.add_subplot(gs[0])
    ax_pr  = fig.add_subplot(gs[1])
    ax_err = fig.add_subplot(gs[2])
    ax_cb  = fig.add_subplot(gs[3])

    titles = [
        f"GT Loupas @ 500 Hz\n({INPUT_DT_MS:.0f} ms / interval)",
        f"WavePINN v4 @ {OUTPUT_FS_HZ/1000:.1f} kHz\n(dt={OUTPUT_DT_MS:.4f} ms)",
        "|Error|",
    ]
    for ax, ttl in zip([ax_gt, ax_pr, ax_err], titles):
        ax.set_facecolor("black")
        for sp in ax.spines.values(): sp.set_edgecolor("white")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(ttl, fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel("Lateral (mm)", fontsize=8)
    ax_gt.set_ylabel("Axial depth (mm)", fontsize=8)

    kw = dict(cmap="jet", vmin=-clim, vmax=clim, aspect="auto", origin="upper", extent=extent)

    def gt_step(fi):
        interval_idx = fi // UPSAMPLING_FACTOR
        sub_fi       = fi % UPSAMPLING_FACTOR
        local_gt     = min(sub_fi * N_GT_PER_INPUT // UPSAMPLING_FACTOR, N_GT_PER_INPUT - 1)
        global_gt    = min(interval_idx * N_GT_PER_INPUT + local_gt, gt.shape[0] - 1)
        return gt[global_gt]

    im_gt  = ax_gt.imshow(gt_step(0), **kw)
    im_pr  = ax_pr.imshow(pred[0], **kw)
    im_err = ax_err.imshow(np.abs(pred[0] - gt_step(0)),
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
        gt_f = gt_step(fi)
        im_gt.set_data(gt_f)
        im_pr.set_data(pred[fi])
        im_err.set_data(np.abs(pred[fi] - gt_f))
        interval_idx = fi // UPSAMPLING_FACTOR
        sub_fi       = fi % UPSAMPLING_FACTOR
        sup.set_text(
            f"WavePINN v4 — Virtual 1.2 kHz SWE  |  {sample_name}  |  "
            f"t = {t_ms[fi]:.3f} ms  (frame {fi+1}/{T})\n"
            f"GT interval {interval_idx+1}/{N_ACTIVE_GROUPS}  "
            f"(sub {sub_fi+1}/{UPSAMPLING_FACTOR})"
        )
        return im_gt, im_pr, im_err, sup

    anim = FuncAnimation(fig, _update, frames=T, blit=True, interval=1000 // fps)
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


# ---------------------------------------------------------------------------
# Static comparison figure
# ---------------------------------------------------------------------------
def write_static_figure(pred, gt, sample_name, out_path):
    sub_show = [0, 3, 7, 11]
    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]
    clim = max(float(np.abs(gt[:N_ACTIVE_GT]).max()), 0.2)

    n_cols = len(sub_show) + 1
    fig, axes = plt.subplots(N_ACTIVE_GROUPS, n_cols,
                             figsize=(3.5 * n_cols, 4 * N_ACTIVE_GROUPS),
                             facecolor="black")
    fig.suptitle(
        f"WavePINN v4 — Virtual 1.2 kHz: GT avg vs sub-frames ({INPUT_DT_MS:.0f} ms intervals)\n{sample_name}",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    for row in range(N_ACTIVE_GROUPS):
        gt_start = row * N_GT_PER_INPUT
        gt_avg   = gt[gt_start:gt_start + N_GT_PER_INPUT].mean(axis=0)
        t0 = row * INPUT_DT_MS

        ax = axes[row, 0]
        ax.imshow(gt_avg, cmap="jet", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", extent=extent)
        ax.set_title(f"GT avg  t={t0:.0f}-{t0+INPUT_DT_MS:.0f} ms",
                     color="white", fontsize=8, fontweight="bold")

        for col, si in enumerate(sub_show):
            fi = row * UPSAMPLING_FACTOR + si
            t_f = fi * OUTPUT_DT_MS
            ax = axes[row, col + 1]
            ax.imshow(pred[fi], cmap="jet", vmin=-clim, vmax=clim,
                      aspect="auto", origin="upper", extent=extent)
            ax.set_title(f"v3  t={t_f:.2f} ms  (sub {si+1}/{UPSAMPLING_FACTOR})",
                         color="white", fontsize=8)

        for ax in axes[row]:
            ax.set_facecolor("black")
            for sp in ax.spines.values(): sp.set_edgecolor("white")
            ax.tick_params(colors="white", labelsize=7)
            if row == N_ACTIVE_GROUPS - 1:
                ax.set_xlabel("Lateral (mm)", color="white", fontsize=8)
        axes[row, 0].set_ylabel("Axial (mm)", color="white", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# v2 vs v3 comparison bar chart
# ---------------------------------------------------------------------------
def write_comparison_barchart(v3_summary, out_dir):
    methods = {}

    # v2 baselines
    for label, path in [
        ("v2 6kHz\n(500Hz)", "outputs/6khz_predictions/summary.json"),
        ("v2 3kHz\n(250Hz)", "outputs/3khz_predictions/summary.json"),
        ("v2 1.2kHz\n(100Hz)", "outputs/1200hz_predictions/summary.json"),
    ]:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            methods[label] = {
                "mae": s.get("mean_mae", float("nan")),
                "r_active": s.get("mean_r_active", float("nan")),
                "ssim": s.get("mean_ssim"),
            }

    # v3 results
    v3_6k = Path("outputs/6khz_predictions_v3/summary.json")
    if v3_6k.exists():
        with open(v3_6k) as f:
            s = json.load(f)
        methods["v3 6kHz\n(500Hz)"] = {
            "mae": s.get("mean_mae", float("nan")),
            "r_active": s.get("mean_r_active", float("nan")),
            "ssim": s.get("mean_ssim"),
        }

    methods["v3 1.2kHz\n(100Hz)"] = {
        "mae": v3_summary.get("mean_mae", float("nan")),
        "r_active": v3_summary.get("mean_r_active", float("nan")),
        "ssim": v3_summary.get("mean_ssim"),
    }

    labels = list(methods.keys())
    mae_vals  = [methods[k]["mae"] for k in labels]
    r_vals    = [methods[k]["r_active"] for k in labels]
    ssim_vals = [methods[k]["ssim"] for k in labels]

    # Color: greens for v2, blues for v3
    colors = []
    for l in labels:
        colors.append("#4CAF50" if "v2" in l else "#2196F3")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#1a1a2e")
    fig.suptitle("v2 vs v3: SWE Velocity Prediction Comparison",
                 color="white", fontsize=13, fontweight="bold")

    panels = [
        (axes[0], mae_vals,  "MAE (mm/s)",       "MAE (lower is better)"),
        (axes[1], r_vals,    "r (active window)", "Pearson r (higher is better)"),
        (axes[2], ssim_vals, "SSIM",              "SSIM (higher is better)"),
    ]

    for ax, vals, ylabel, title in panels:
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

        x = np.arange(len(labels))
        for xi, (v, c) in enumerate(zip(vals, colors)):
            is_nan = (v is None) or np.isnan(float(v)) if isinstance(v, (int, float)) else True
            if is_nan:
                ax.bar(xi, 0, color=c, alpha=0.5, hatch="//", width=0.5)
                ax.text(xi, 0.02, "N/A", ha="center", va="bottom",
                        color="white", fontsize=9, fontweight="bold")
            else:
                ax.bar(xi, float(v), color=c, alpha=0.85, width=0.5)

        ax.autoscale(axis="y")
        ymax = ax.get_ylim()[1]
        for xi, (v, c) in enumerate(zip(vals, colors)):
            is_nan = (v is None) or (not isinstance(v, (int, float)))
            if not is_nan and not np.isnan(float(v)):
                ax.text(xi, float(v) + ymax * 0.02, f"{float(v):.3f}",
                        ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=10)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white")

    plt.tight_layout()
    save_path = out_dir / "v2_vs_v3_comparison.png"
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"Comparison chart -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WavePINN v4: 1.2kHz from simulated 100Hz IQ"
    )
    parser.add_argument("--model_path",  default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--sample_name", default=None)
    parser.add_argument("--fps",         type=int, default=15)
    parser.add_argument("--no_video",    action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading WavePINNv4 from {args.model_path} ...")
    model = load_model(args.model_path).to(device)

    with open(DATA_META_PATH) as f:
        meta = json.load(f)

    # Search across all splits (500Hz data was moved to train/val for training)
    all_samples = {}
    for split_key in ("inference", "val", "train"):
        for s in meta.get(split_key, []):
            all_samples[s["name"]] = s

    if args.sample_name:
        samples = [all_samples[args.sample_name]]
    else:
        # Use val samples for evaluation (not augmented during training)
        samples = [s for s in meta.get("val", [])
                   if s.get("frequency") == "500Hz" and s["name"].endswith("_w0")]
        if not samples:
            samples = [s for s in all_samples.values()
                       if s.get("frequency") == "500Hz" and s["name"].endswith("_w0")]

    print(f"\nProcessing {len(samples)} samples (1.2kHz from 100Hz, v3)\n")

    all_metrics = []
    best_r, best_name, best_pred, best_gt = -np.inf, None, None, None

    for i, sm in enumerate(samples):
        name = sm["name"]
        print(f"[{i+1:2d}/{len(samples)}] {name}")

        try:
            iq_frames = load_sparse_iq(sm, start_frame=args.start_frame)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        gt_path = None
        for split_dir in ("inference", "val", "train"):
            p = Path(f"data/fullscale_velocity/{split_dir}") / f"velocity_{name}.npy"
            if p.exists():
                gt_path = p
                break
        if gt_path is None:
            print(f"  SKIP: velocity GT not found")
            continue
        gt_vel = np.load(str(gt_path)).astype(np.float32)

        pred_mm, iq_scale, vel_scale = infer(model, iq_frames, gt_vel)

        pred_steps, gt_steps = compute_active_steps(pred_mm, gt_vel)

        mae    = float(np.abs(pred_steps - gt_steps).mean())
        r_full = np.nan_to_num(float(np.corrcoef(pred_mm.flatten(), gt_vel.flatten())[0, 1]))
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

        print(f"  MAE={mae:.4f}  r_full={r_full:.4f}  r_active={r_act:.4f}  SSIM={ssim_mean:.4f}")

        all_metrics.append(dict(
            name=name, mae=mae, r_full=r_full, r_active=r_act,
            ssim_mean=ssim_mean, vel_scale=vel_scale,
        ))

        if r_act > best_r:
            best_r, best_name = r_act, name
            best_pred = pred_mm.copy()
            best_gt   = gt_vel.copy()

        np.save(str(out_dir / f"velocity_1200hz_v4_{name}.npy"), pred_mm)
        np.save(str(out_dir / f"gt_velocity_{name}.npy"), gt_vel)

        if not args.no_video:
            vid_path = out_dir / f"{name}_1200hz_v4"
            saved = write_video(pred_mm, gt_vel, name, vid_path, args.fps)
            print(f"  Video -> {saved}")

    if best_pred is not None:
        fig_path = out_dir / "best_sample_1200hz_v4_detail.png"
        write_static_figure(best_pred, best_gt, best_name, fig_path)
        print(f"\nBest sample: {best_name}  r_active={best_r:.4f}")
        print(f"Static figure -> {fig_path}")

    if all_metrics:
        rs    = [m["r_active"]  for m in all_metrics]
        maes  = [m["mae"]       for m in all_metrics]
        ssims = [m["ssim_mean"] for m in all_metrics]

        print(f"\n=== v3 Summary ({len(all_metrics)} samples, 1.2kHz from 100Hz) ===")
        print(f"Mean r (active): {np.mean(rs):.4f} +/- {np.std(rs):.4f}")
        print(f"Mean MAE:        {np.mean(maes):.4f} mm/s")
        print(f"Mean SSIM:       {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")

        # v2 baseline comparison
        v2_path = Path("outputs/1200hz_predictions/summary.json")
        if v2_path.exists():
            with open(v2_path) as f:
                v2 = json.load(f)
            print(f"\n--- v2 baseline (1.2kHz from 100Hz) ---")
            print(f"Mean r (active): {v2.get('mean_r_active', 'N/A')}")
            print(f"Mean MAE:        {v2.get('mean_mae', 'N/A')} mm/s")
            print(f"Mean SSIM:       {v2.get('mean_ssim', 'N/A')}")

        summary = {
            "description": "WavePINNv4: 1.2kHz from sparse 100Hz-simulated IQ",
            "model": "WavePINNv4",
            "input_step": INPUT_STEP,
            "input_dt_ms": INPUT_DT_MS,
            "output_Fs_hz": OUTPUT_FS_HZ,
            "output_dt_ms": OUTPUT_DT_MS,
            "mean_r_active": float(np.mean(rs)),
            "std_r_active": float(np.std(rs)),
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "mean_ssim": float(np.mean(ssims)),
            "std_ssim": float(np.std(ssims)),
            "samples": all_metrics,
        }
        with open(str(out_dir / "summary.json"), "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Summary -> {out_dir}/summary.json")

        write_comparison_barchart(summary, out_dir)


if __name__ == "__main__":
    main()
