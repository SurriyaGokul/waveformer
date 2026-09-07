#!/usr/bin/env python3
"""
Virtual 6 kHz velocity prediction from 500 Hz IQ data - WavePINN v5.

Uses ComplexToRealUNetV5 with temporal skip refinement.

Usage
-----
    python scripts/predict_6khz_from_500hz_v5.py
    python scripts/predict_6khz_from_500hz_v5.py --sample_name 2ft_10mmfocus_Plane_4_500Hz_20260129T130420_w0
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.architectures.wavepinn_v5 import ComplexToRealUNetV5, create_wavepinn_v5

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX = 0.0507
LATERAL_MM_PER_PX = 0.1013
H_ROI, W_ROI = 116, 160
UPSAMPLING_FACTOR = 12
MODEL_PATH_DEFAULT = "outputs/wavepinn_v5/best_model.pt"
DATA_META_PATH = "data/fullscale_velocity/metadata.json"
OUTPUT_DIR_DEFAULT = "outputs/6khz_predictions_v5"

# 500 Hz consecutive -> 6 kHz
INPUT_DT_S = 1.0 / 500.0
OUTPUT_DT_MS = INPUT_DT_S * 1e3 / UPSAMPLING_FACTOR
OUTPUT_FS_HZ = 1.0 / (OUTPUT_DT_MS * 1e-3)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str) -> ComplexToRealUNetV5:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_wavepinn_v5().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    learned_cs = ckpt.get("learned_cs", None)
    if learned_cs is None:
        print(f"  WavePINNv5 loaded (epoch={epoch}, best_val_loss={val_loss:.6f})")
    else:
        print(
            f"  WavePINNv5 loaded (epoch={epoch}, best_val_loss={val_loss:.6f}, "
            f"learned_cs={learned_cs:.4f})"
        )
    return model


# ---------------------------------------------------------------------------
# IQ preprocessing
# ---------------------------------------------------------------------------
def load_consecutive_iq(sample_meta: dict, start_frame: int = 0) -> np.ndarray:
    """Load 5 consecutive IQ frames with clutter filtering. Returns (5, H, W) complex64."""
    raw = np.load(sample_meta["original_path"])
    roi = sample_meta["roi"]
    win = sample_meta["window"]

    iq_win = raw[
        roi["min_row"] : roi["max_row"] + 1,
        roi["min_col"] : roi["max_col"] + 1,
        win[0] : win[1],
    ].astype(np.complex64)

    iq_detrend = iq_win - iq_win.mean(axis=2, keepdims=True)

    t_win = iq_detrend.shape[2]
    end = start_frame + 5
    if end > t_win:
        raise ValueError(f"start_frame={start_frame} + 5 > T_window={t_win}")

    frames = np.stack([iq_detrend[:, :, start_frame + i] for i in range(5)], axis=0)
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(model: ComplexToRealUNetV5, iq_consec: np.ndarray, gt_vel: np.ndarray):
    """Returns (pred_mm, iq_scale, vel_scale)."""
    iq_scale = float(np.percentile(np.abs(iq_consec), 99))
    iq_norm = (iq_consec / max(iq_scale, 1e-12)).astype(np.complex64)
    vel_scale = float(np.percentile(np.abs(gt_vel), 99))

    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)
    pad_h = (16 - H_ROI % 16) % 16
    inp_pad = np.pad(inp, ((0, 0), (0, 0), (0, pad_h), (0, 0)))

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.from_numpy(inp_pad).unsqueeze(0).to(device))
        if isinstance(out, dict):
            out = out["velocity"]

    pred_norm = out.squeeze().cpu().numpy()
    pred_mm = pred_norm[:, :H_ROI, :] * vel_scale

    return pred_mm.astype(np.float32), iq_scale, vel_scale


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def _try_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


def write_video(pred, gt, sample_name, out_path, fps=15):
    t_dense = pred.shape[0]
    t_ms = np.arange(t_dense) * OUTPUT_DT_MS
    clim = max(float(np.abs(gt[:5]).max()), 0.2)

    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(20, 6), facecolor="black")
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.07,
        left=0.05,
        right=0.97,
        top=0.86,
        bottom=0.12,
    )
    ax_gt = fig.add_subplot(gs[0])
    ax_pr = fig.add_subplot(gs[1])
    ax_err = fig.add_subplot(gs[2])
    ax_cb = fig.add_subplot(gs[3])

    titles = [
        "GT Loupas @ 500 Hz\n(step function)",
        f"WavePINN v5 @ {OUTPUT_FS_HZ/1000:.0f} kHz\n(dt={OUTPUT_DT_MS:.3f} ms)",
        "|Error|",
    ]
    for ax, ttl in zip([ax_gt, ax_pr, ax_err], titles):
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(ttl, fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel("Lateral (mm)", fontsize=8)
    ax_gt.set_ylabel("Axial depth (mm)", fontsize=8)

    kw = dict(cmap="jet", vmin=-clim, vmax=clim, aspect="auto", origin="upper", extent=extent)

    def gt_step(fi):
        gi = min(fi // UPSAMPLING_FACTOR, gt.shape[0] - 1)
        return gt[gi]

    im_gt = ax_gt.imshow(gt_step(0), **kw)
    im_pr = ax_pr.imshow(pred[0], **kw)
    im_err = ax_err.imshow(
        np.abs(pred[0] - gt_step(0)),
        cmap="hot",
        vmin=0,
        vmax=clim,
        aspect="auto",
        origin="upper",
        extent=extent,
    )

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
        sup.set_text(
            f"WavePINN v5 - Virtual 6 kHz SWE  |  {sample_name}  |  "
            f"t = {t_ms[fi]:.3f} ms  (frame {fi+1}/{t_dense})"
        )
        return im_gt, im_pr, im_err, sup

    anim = FuncAnimation(fig, _update, frames=t_dense, blit=True, interval=1000 // fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _try_ffmpeg():
        writer = FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
        save_path = out_path.with_suffix(".mp4")
        anim.save(str(save_path), writer=writer, dpi=120, savefig_kwargs={"facecolor": "black"})
    else:
        writer = PillowWriter(fps=fps)
        save_path = out_path.with_suffix(".gif")
        anim.save(str(save_path), writer=writer, savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Static comparison figure
# ---------------------------------------------------------------------------
def write_static_figure(pred, gt, sample_name, out_path):
    sub_show = [0, 3, 7, 11]
    n_gt = 4

    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]
    clim = max(float(np.abs(gt[:n_gt]).max()), 0.2)

    n_cols = len(sub_show) + 1
    fig, axes = plt.subplots(n_gt, n_cols, figsize=(3.5 * n_cols, 4 * n_gt), facecolor="black")
    fig.suptitle(
        f"WavePINN v5 - Virtual 6 kHz: GT (left) vs sub-frames\n{sample_name}",
        color="white",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    for row in range(n_gt):
        gt_f = gt[row]
        t0 = row * 2.0

        ax = axes[row, 0]
        ax.imshow(gt_f, cmap="jet", vmin=-clim, vmax=clim, aspect="auto", origin="upper", extent=extent)
        ax.set_title(
            f"GT  t={t0:.0f}-{t0+2:.0f} ms\n(Loupas @ 500Hz)",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

        for col, si in enumerate(sub_show):
            fi = row * UPSAMPLING_FACTOR + si
            t_f = fi * OUTPUT_DT_MS
            ax = axes[row, col + 1]
            ax.imshow(pred[fi], cmap="jet", vmin=-clim, vmax=clim, aspect="auto", origin="upper", extent=extent)
            ax.set_title(
                f"v5  t={t_f:.2f} ms\n(sub {si+1}/{UPSAMPLING_FACTOR})",
                color="white",
                fontsize=8,
            )

        for ax in axes[row]:
            ax.set_facecolor("black")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
            ax.tick_params(colors="white", labelsize=7)
            if row == n_gt - 1:
                ax.set_xlabel("Lateral (mm)", color="white", fontsize=8)
        axes[row, 0].set_ylabel("Axial (mm)", color="white", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WavePINN v5: 6kHz from 500Hz IQ")
    parser.add_argument("--model_path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--sample_name", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading WavePINNv5 from {args.model_path} ...")
    model = load_model(args.model_path).to(device)

    with open(DATA_META_PATH) as f:
        meta = json.load(f)

    all_samples = {}
    for split_key in ("inference", "val", "train"):
        for sample in meta.get(split_key, []):
            all_samples[sample["name"]] = sample

    if args.sample_name:
        if args.sample_name not in all_samples:
            raise ValueError(f"Sample not found in metadata: {args.sample_name}")
        samples = [all_samples[args.sample_name]]
    else:
        samples = [
            sample
            for sample in meta.get("val", [])
            if sample.get("frequency") == "500Hz" and sample["name"].endswith("_w0")
        ]
        if not samples:
            samples = [
                sample
                for sample in all_samples.values()
                if sample.get("frequency") == "500Hz" and sample["name"].endswith("_w0")
            ]

    print(f"\nProcessing {len(samples)} samples (6kHz from 500Hz, v5)\n")

    all_metrics = []
    best_r, best_name, best_pred, best_gt = -np.inf, None, None, None

    for i, sample_meta in enumerate(samples):
        name = sample_meta["name"]
        print(f"[{i+1:2d}/{len(samples)}] {name}")

        try:
            iq_consec = load_consecutive_iq(sample_meta, start_frame=args.start_frame)
        except Exception as exc:
            print(f"  SKIP: {exc}")
            continue

        gt_path = None
        for split_dir in ("inference", "val", "train"):
            candidate = Path(f"data/fullscale_velocity/{split_dir}") / f"velocity_{name}.npy"
            if candidate.exists():
                gt_path = candidate
                break
        if gt_path is None:
            print("  SKIP: velocity GT not found")
            continue

        gt_vel = np.load(str(gt_path)).astype(np.float32)
        pred_mm, _, vel_scale = infer(model, iq_consec, gt_vel)

        gt_active = gt_vel[:4]
        pred_steps = np.stack(
            [pred_mm[k * UPSAMPLING_FACTOR : (k + 1) * UPSAMPLING_FACTOR].mean(0) for k in range(4)],
            axis=0,
        )

        mae = float(np.abs(pred_steps - gt_active).mean())
        r_full = float(np.corrcoef(pred_mm.flatten(), gt_vel.flatten())[0, 1])
        r_active = float(np.corrcoef(pred_steps.flatten(), gt_active.flatten())[0, 1])

        print(f"  MAE={mae:.4f} mm/s  r_full={r_full:.4f}  r_active={r_active:.4f}")

        all_metrics.append(
            {
                "name": name,
                "mae": mae,
                "r_full": r_full,
                "r_active": r_active,
                "vel_scale": vel_scale,
            }
        )

        if r_active > best_r:
            best_r, best_name = r_active, name
            best_pred = pred_mm.copy()
            best_gt = gt_vel.copy()

        np.save(str(out_dir / f"velocity_6khz_v5_{name}.npy"), pred_mm)
        np.save(str(out_dir / f"gt_velocity_{name}.npy"), gt_vel)

        if not args.no_video:
            vid_path = out_dir / f"{name}_6khz_v5"
            saved = write_video(pred_mm, gt_vel, name, vid_path, args.fps)
            print(f"  Video -> {saved}")

    if best_pred is not None:
        fig_path = out_dir / "best_sample_6khz_v5_detail.png"
        write_static_figure(best_pred, best_gt, best_name, fig_path)
        print(f"\nBest sample: {best_name}  r_active={best_r:.4f}")
        print(f"Static figure -> {fig_path}")

    if all_metrics:
        rs = [m["r_active"] for m in all_metrics]
        maes = [m["mae"] for m in all_metrics]
        print(f"\n=== v5 Summary ({len(all_metrics)} samples, 6kHz from 500Hz) ===")
        print(f"Mean r (active): {np.mean(rs):.4f} +/- {np.std(rs):.4f}")
        print(f"Mean MAE:        {np.mean(maes):.4f} mm/s")

        v2_path = Path("outputs/6khz_predictions/summary.json")
        if v2_path.exists():
            with open(v2_path) as f:
                v2 = json.load(f)
            print("\n--- v2 baseline ---")
            print(f"Mean r (active): {v2.get('mean_r_active', 'N/A')}")
            print(f"Mean MAE:        {v2.get('mean_mae', 'N/A')} mm/s")

        summary = {
            "description": "WavePINNv5: 6kHz from consecutive 500Hz IQ",
            "model": "WavePINNv5",
            "output_Fs_hz": OUTPUT_FS_HZ,
            "output_dt_ms": OUTPUT_DT_MS,
            "mean_r_active": float(np.mean(rs)),
            "std_r_active": float(np.std(rs)),
            "mean_mae": float(np.mean(maes)),
            "samples": all_metrics,
        }
        with open(str(out_dir / "summary.json"), "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Summary -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
