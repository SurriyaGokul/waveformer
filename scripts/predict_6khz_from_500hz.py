#!/usr/bin/env python3
"""
Virtual 6 kHz velocity prediction from 500 Hz IQ data.

Idea
----
During training the model saw 5 IQ frames subsampled every 12th frame at the
native PRF.  For 5 kHz training data that means inter-frame dt = 12/5000 = 2.4 ms,
and the 49 output velocity frames are at dt = 2.4/12 = 0.2 ms.

We now pass 5 *consecutive* 500 Hz IQ frames (spacing = 1/500 = 2.0 ms each).
The model treats those exactly as it was trained — 5 IQ frames, map to 49 velocity
frames — so the output velocity frames are at:

    dt_out = dt_in / upsampling_factor = 2.0 ms / 12 = 0.1667 ms  →  6000 Hz

The input spacing (2.0 ms) is very close to what the model saw in training for
5 kHz data (2.4 ms), so this is *in-distribution* unlike the previous sparse
500 Hz test (which had 24 ms spacing — 10× out of range).

Preprocessing applied to the raw IQ (matching training exactly):
  1. Extract spatial ROI stored in metadata.json
  2. Clutter filter: subtract temporal mean over the full 50-frame window
  3. Take 5 consecutive frames [0,1,2,3,4]
  4. Per-sample IQ normalisation: divide by 99th percentile of |IQ|
  5. Per-sample vel normalisation: divide output by 99th percentile of |GT vel|

Usage
-----
    python scripts/predict_6khz_from_500hz.py
    python scripts/predict_6khz_from_500hz.py --sample_name 2ft_10mmfocus_Plane_4_500Hz_20260129T130420_w0
    python scripts/predict_6khz_from_500hz.py --start_frame 0   # choose which consecutive window
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.architectures.velocity_unet import ComplexToRealUNet

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX   = 0.0507
LATERAL_MM_PER_PX = 0.1013
H_ROI, W_ROI      = 116, 160
UPSAMPLING_FACTOR  = 12
MODEL_PATH_DEFAULT = "outputs/fullscale_velocity_v2/best_model_final.pt"
DATA_META_PATH     = "data/fullscale_velocity/metadata.json"
OUTPUT_DIR_DEFAULT = "outputs/6khz_predictions"


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
# IQ preprocessing — same as training pipeline
# ---------------------------------------------------------------------------
def load_consecutive_iq(sample_meta: dict, start_frame: int = 0) -> np.ndarray:
    """
    Load 5 consecutive IQ frames from the raw source file, apply the same
    clutter filter as training (temporal mean subtraction over the full window).

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

    # Clutter filter: subtract temporal mean (same as training preprocessing)
    iq_detrend = iq_win - iq_win.mean(axis=2, keepdims=True)   # (H, W, T_win)

    T_win = iq_detrend.shape[2]
    end   = start_frame + 5
    if end > T_win:
        raise ValueError(
            f"start_frame={start_frame} + 5 > T_win={T_win}. "
            f"Max start_frame = {T_win - 5}."
        )

    frames = np.stack(
        [iq_detrend[:, :, start_frame + i] for i in range(5)], axis=0
    )  # (5, H, W)
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(
    model: ComplexToRealUNet,
    iq_consec: np.ndarray,
    gt_vel: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    iq_consec : (5, H, W) complex64 — consecutive IQ frames
    gt_vel    : (49, H, W) float32  — GT velocity at native PRF rate (mm/s)

    Returns: (pred_mm, iq_scale, vel_scale)
    """
    # Per-sample IQ normalisation
    iq_scale = float(np.percentile(np.abs(iq_consec), 99))
    iq_norm  = (iq_consec / max(iq_scale, 1e-12)).astype(np.complex64)

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
    pred_norm = out.squeeze().numpy()            # (49, 128, W)
    pred_mm   = pred_norm[:, :H_ROI, :] * vel_scale   # → mm/s

    return pred_mm.astype(np.float32), iq_scale, vel_scale


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def _try_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def write_6khz_video(
    pred_6khz : np.ndarray,          # (49, H, W) mm/s at 6kHz
    gt_500hz  : np.ndarray,          # (49, H, W) mm/s at 500Hz (native)
    start_frame: int,
    sample_name: str,
    out_path  : Path,
    fps       : int = 15,
) -> Path:
    """
    3-panel video: GT (500Hz step function) | Prediction (6kHz) | |Error|

    The GT panel shows a step function: for each of the 49 model output frames,
    we display the GT velocity frame whose index = (output_frame // UPSAMPLING_FACTOR).
    This shows exactly which GT frame the model is expanding between, making the
    temporal super-resolution visible.

    Only the first 4 GT frames (frames 0-3) cover the same 8ms window as the
    6kHz prediction.  Frames 4-48 of the 6kHz output are extrapolation territory.
    """
    T      = pred_6khz.shape[0]    # 49
    dt_out = (1.0 / 500.0 * 1e3) / UPSAMPLING_FACTOR   # 0.1667 ms
    t_ms   = np.arange(T) * dt_out

    # Colour limit: peak of the GT active region (first 5 GT frames)
    clim = float(np.abs(gt_500hz[:5]).max())
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
        "GT Loupas @ 500 Hz\n(step function — 2 ms / step)",
        f"Model Prediction @ 6 kHz\n(Δt = {dt_out:.3f} ms / frame)",
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

    # GT step: which GT frame does output frame fi map to?
    def gt_step(fi: int) -> np.ndarray:
        gi = min(fi // UPSAMPLING_FACTOR, gt_500hz.shape[0] - 1)
        return gt_500hz[gi]

    im_gt  = ax_gt.imshow(gt_step(0), **kw)
    im_pr  = ax_pr.imshow(pred_6khz[0], **kw)
    im_err = ax_err.imshow(np.abs(pred_6khz[0] - gt_step(0)),
                           cmap="hot", vmin=0, vmax=clim,
                           aspect="auto", origin="upper", extent=extent)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Particle velocity (mm/s)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    prf_hz = 500
    sup = fig.suptitle("", color="white", fontsize=10, fontweight="bold")

    def _update(fi: int):
        gt_f   = gt_step(fi)
        pr_f   = pred_6khz[fi]
        gi     = min(fi // UPSAMPLING_FACTOR, gt_500hz.shape[0] - 1)
        sub_fi = fi % UPSAMPLING_FACTOR  # which sub-frame within the GT interval

        # Active annotation: only first few GT intervals have wave
        active = gi < 2
        active_str = "  ◀ WAVE ACTIVE" if active else ""

        im_gt.set_data(gt_f)
        im_pr.set_data(pr_f)
        im_err.set_data(np.abs(pr_f - gt_f))

        sup.set_text(
            f"Virtual 6 kHz SWE  |  {sample_name}  |  "
            f"t = {t_ms[fi]:.3f} ms  (frame {fi+1}/{T})\n"
            f"GT step: interval {gi+1}/4  (sub-frame {sub_fi+1}/{UPSAMPLING_FACTOR})"
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
    pred_6khz  : np.ndarray,
    gt_500hz   : np.ndarray,
    sample_name: str,
    out_path   : Path,
) -> None:
    """
    Grid: rows = GT frames 0-3 (one per 500Hz interval), columns = model sub-frames
    within each GT interval (sub-frames 0, 4, 8, 11 → 0ms, 0.67ms, 1.33ms, 1.83ms
    within each 2ms GT interval).
    """
    sub_show = [0, 3, 7, 11]   # which sub-frame indices to show within each GT interval
    n_gt     = 4                # first 4 GT intervals cover 0-8ms
    dt_out   = (1.0 / 500.0 * 1e3) / UPSAMPLING_FACTOR

    dep_ax = np.arange(H_ROI) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W_ROI) - W_ROI / 2) * LATERAL_MM_PER_PX
    extent = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    clim = float(np.abs(gt_500hz[:n_gt]).max())
    clim = max(clim, 0.2)

    n_cols = len(sub_show) + 1   # +1 for GT column
    fig, axes = plt.subplots(n_gt, n_cols, figsize=(3.5 * n_cols, 4 * n_gt),
                             facecolor="black")
    fig.suptitle(
        f"Virtual 6 kHz — GT (left) vs model sub-frames within each 2 ms interval\n{sample_name}",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    for row in range(n_gt):
        gt_f = gt_500hz[row]
        t0   = row * 2.0   # ms

        # GT column
        ax = axes[row, 0]
        ax.imshow(gt_f, cmap="jet", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper", extent=extent)
        ax.set_title(f"GT  t={t0:.0f}–{t0+2:.0f} ms\n(Loupas @ 500Hz)",
                     color="white", fontsize=8, fontweight="bold")

        # Model sub-frames
        for col, si in enumerate(sub_show):
            fi  = row * UPSAMPLING_FACTOR + si
            t_f = fi * dt_out
            ax  = axes[row, col + 1]
            ax.imshow(pred_6khz[fi], cmap="jet", vmin=-clim, vmax=clim,
                      aspect="auto", origin="upper", extent=extent)
            ax.set_title(f"Model  t={t_f:.2f} ms\n(6kHz sub-frame {si+1}/{UPSAMPLING_FACTOR})",
                         color="white", fontsize=8)

        for ax in axes[row]:
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_edgecolor("white")
            ax.tick_params(colors="white", labelsize=7)
            if row == n_gt - 1:
                ax.set_xlabel("Lateral (mm)", color="white", fontsize=8)

        axes[row, 0].set_ylabel("Axial (mm)", color="white", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict virtual 6kHz velocity from 500Hz IQ")
    parser.add_argument("--model_path",  default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First of 5 consecutive IQ frames to use (default: 0)")
    parser.add_argument("--sample_name", default=None,
                        help="Process one named sample only; omit to process all 500Hz _w0")
    parser.add_argument("--fps",         type=int, default=15,
                        help="Video fps (default 15 — gives smooth 6kHz animation)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    print(f"Loading metadata from {DATA_META_PATH} ...")
    with open(DATA_META_PATH) as f:
        meta = json.load(f)

    # Build lookup: name → sample dict
    inf_lookup = {s["name"]: s for s in meta["inference"]}

    # Select samples
    if args.sample_name:
        samples = [inf_lookup[args.sample_name]]
    else:
        samples = [s for s in meta["inference"]
                   if s.get("frequency") == "500Hz" and s["name"].endswith("_w0")]

    print(f"\nProcessing {len(samples)} samples  "
          f"(start_frame={args.start_frame}, fps={args.fps})\n")

    dt_out_ms = (1.0 / 500.0 * 1e3) / UPSAMPLING_FACTOR   # 0.1667 ms
    Fs_out    = 1.0 / (dt_out_ms * 1e-3)                   # 6000 Hz

    all_metrics = []
    best_r, best_name, best_pred, best_gt = -np.inf, None, None, None

    for i, sm in enumerate(samples):
        name = sm["name"]
        print(f"[{i+1:2d}/{len(samples)}] {name}")

        # --- Load raw consecutive IQ ---
        try:
            iq_consec = load_consecutive_iq(sm, start_frame=args.start_frame)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # --- Load GT velocity ---
        gt_path = Path("data/fullscale_velocity/inference") / f"velocity_{name}.npy"
        gt_vel  = np.load(str(gt_path)).astype(np.float32)

        # --- Inference ---
        pred_mm, iq_scale, vel_scale = infer(model, iq_consec, gt_vel)

        # --- Metrics vs first 4 GT intervals (same 8ms window) ---
        # GT frames 0-3 correspond to model frames 0-47 (four 2ms intervals)
        gt_active  = gt_vel[:4]           # (4, H, W) — 0 to 8 ms at 2ms/frame
        pred_steps = np.stack(
            [pred_mm[k * UPSAMPLING_FACTOR : (k + 1) * UPSAMPLING_FACTOR].mean(axis=0)
             for k in range(4)],
            axis=0,
        )  # (4, H, W) — model averaged within each 2ms GT interval

        mae  = float(np.abs(pred_steps - gt_active).mean())
        r_full = float(np.corrcoef(pred_mm.flatten(), gt_vel.flatten())[0, 1])
        r_act  = float(np.corrcoef(pred_steps.flatten(), gt_active.flatten())[0, 1])

        print(f"  iq_scale={iq_scale:.3e}  vel_scale={vel_scale:.4f} mm/s")
        print(f"  MAE(active window)={mae:.4f} mm/s  r(full)={r_full:.4f}  r(active)={r_act:.4f}")

        all_metrics.append(dict(name=name, mae=mae, r_full=r_full, r_active=r_act,
                                vel_scale=vel_scale))

        # Track best sample for detailed visualisation
        if r_act > best_r:
            best_r, best_name = r_act, name
            best_pred = pred_mm.copy()
            best_gt   = gt_vel.copy()

        # --- Save prediction npy ---
        np.save(str(out_dir / f"velocity_6khz_{name}.npy"), pred_mm)
        np.save(str(out_dir / f"gt_velocity_{name}.npy"),    gt_vel)

        # --- Per-sample metadata ---
        with open(str(out_dir / f"metadata_{name}.json"), "w") as mf:
            json.dump({
                "name": name,
                "approach": "consecutive_iq_frames",
                "start_frame": args.start_frame,
                "input_iq_frames": list(range(args.start_frame, args.start_frame + 5)),
                "input_iq_dt_ms": 1.0 / sm["prf_hz"] * 1e3,
                "upsampling_factor": UPSAMPLING_FACTOR,
                "output_dt_ms": dt_out_ms,
                "output_Fs_hz": Fs_out,
                "output_n_frames": int(pred_mm.shape[0]),
                "output_total_duration_ms": dt_out_ms * (pred_mm.shape[0] - 1),
                "velocity_unit": "mm/s",
                "vel_scale_mms": vel_scale,
                "iq_scale": iq_scale,
                "metrics": {"mae_active_mm_s": mae, "r_full": r_full, "r_active": r_act},
                "note": (
                    f"5 consecutive 500Hz IQ frames (dt={1/sm['prf_hz']*1e3:.1f}ms each) "
                    f"→ model upsamples by {UPSAMPLING_FACTOR}× "
                    f"→ output at {Fs_out:.0f}Hz (dt={dt_out_ms:.4f}ms). "
                    "Clutter filter (temporal mean subtraction) applied before IQ input. "
                    "Per-sample IQ and velocity normalisation applied."
                ),
                "tissue_properties": sm.get("tissue_properties", {}),
            }, mf, indent=2)

        # --- Per-sample video ---
        vid_path = out_dir / f"{name}_6khz"
        saved = write_6khz_video(pred_mm, gt_vel,
                                 start_frame=args.start_frame,
                                 sample_name=name,
                                 out_path=vid_path,
                                 fps=args.fps)
        print(f"  Video → {saved}")

    # --- Best-sample detailed static figure ---
    if best_pred is not None:
        fig_path = out_dir / "best_sample_6khz_detail.png"
        write_static_figure(best_pred, best_gt, best_name, fig_path)
        print(f"\nBest sample: {best_name}  r_active={best_r:.4f}")
        print(f"Static figure → {fig_path}")

    # --- Summary ---
    if all_metrics:
        rs  = [m["r_active"] for m in all_metrics]
        mse = [m["mae"]      for m in all_metrics]
        print(f"\n=== Summary ({len(all_metrics)} samples) ===")
        print(f"Mean r (active window): {np.mean(rs):.4f}  ± {np.std(rs):.4f}")
        print(f"Mean MAE:               {np.mean(mse):.4f} mm/s")
        with open(str(out_dir / "summary.json"), "w") as sf:
            json.dump({
                "description": "Virtual 6kHz predictions from consecutive 500Hz IQ",
                "output_Fs_hz": Fs_out,
                "output_dt_ms": dt_out_ms,
                "start_frame": args.start_frame,
                "mean_r_active": float(np.mean(rs)),
                "std_r_active":  float(np.std(rs)),
                "mean_mae":      float(np.mean(mse)),
                "samples": all_metrics,
            }, sf, indent=2)
        print(f"Summary  → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
