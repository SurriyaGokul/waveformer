#!/usr/bin/env python3
"""generate_poster_wave_montage_500hz_vs_10khz.py

Creates a publication-grade 2-row montage for posters:
  - Row 1: 500 Hz shear-wave velocity (sample-and-hold / step function)
    - Row 2: 10 kHz shear-wave velocity (model prediction)

The figure is designed to visually communicate that at 500 Hz the wave front
passes through the ROI within ~2 frames, while at 10 kHz the motion can be
observed smoothly across intermediate time steps.

Assumes `scripts/predict_10khz_from_500hz_20x.py` has already produced:
  - outputs/10khz_predictions_20x/gt_velocity_<sample>.npy
  - outputs/10khz_predictions_20x/velocity_10khz_<sample>.npy

Usage
-----
  python scripts/generate_poster_wave_montage_500hz_vs_10khz.py
    python scripts/generate_poster_wave_montage_500hz_vs_10khz.py --sample_name <SAMPLE_NAME>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# Spatial calibration (from repo constants / IQ2VEl.m)
AXIAL_MM_PER_PX = 0.0507
LATERAL_MM_PER_PX = 0.1013

# Timing
GT_500HZ_DT_MS = 2.0
PRED_10KHZ_DT_MS = 0.1


def _extent_mm(H: int, W: int):
    dep_ax = np.arange(H) * AXIAL_MM_PER_PX
    lat_ax = (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX
    return [float(lat_ax[0]), float(lat_ax[-1]), float(dep_ax[-1]), float(dep_ax[0])]


def _load_arrays(input_dir: Path, sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    gt_path = input_dir / f"gt_velocity_{sample_name}.npy"
    pr_path = input_dir / f"velocity_10khz_{sample_name}.npy"

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Missing GT array: {gt_path}. Run scripts/predict_10khz_from_500hz_20x.py first."
        )
    if not pr_path.exists():
        raise FileNotFoundError(
            f"Missing 10kHz prediction array: {pr_path}. Run scripts/predict_10khz_from_500hz_20x.py first."
        )

    gt_500 = np.load(gt_path).astype(np.float32)
    pred_10k = np.load(pr_path).astype(np.float32)
    if gt_500.ndim != 3 or pred_10k.ndim != 3:
        raise ValueError(
            f"Expected arrays shaped (T,H,W). Got gt={gt_500.shape}, pred={pred_10k.shape}."
        )
    return gt_500, pred_10k


def _frame_idx_from_time_ms(t_ms: float, dt_ms: float, n_frames: int, mode: str) -> int:
    if mode == "hold":
        idx = int(np.floor(t_ms / dt_ms + 1e-12))
    elif mode == "nearest":
        idx = int(np.round(t_ms / dt_ms))
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'hold' or 'nearest'.")
    return int(np.clip(idx, 0, n_frames - 1))


def make_montage(
    gt_500hz: np.ndarray,
    pred_10khz: np.ndarray,
    sample_name: str,
    out_png: Path,
    out_pdf: Path | None,
    t_start_ms: float,
    t_end_ms: float,
    dt_show_ms: float,
    clim_percentile: float,
    cmap: str,
    gt_time_mode: str,
):
    times = np.arange(t_start_ms, t_end_ms + 1e-9, dt_show_ms, dtype=np.float32)
    n_cols = int(times.size)

    H, W = int(gt_500hz.shape[1]), int(gt_500hz.shape[2])
    extent = _extent_mm(H, W)

    gt_idxs = [
        _frame_idx_from_time_ms(float(t), GT_500HZ_DT_MS, gt_500hz.shape[0], gt_time_mode)
        for t in times
    ]
    pr_idxs = [
        _frame_idx_from_time_ms(float(t), PRED_10KHZ_DT_MS, pred_10khz.shape[0], "nearest")
        for t in times
    ]

    # Shared symmetric colour scale (robust percentile over selected frames)
    abs_samples = []
    for gi, pi in zip(gt_idxs, pr_idxs):
        abs_samples.append(np.abs(gt_500hz[gi]).ravel())
        abs_samples.append(np.abs(pred_10khz[pi]).ravel())
    abs_all = np.concatenate(abs_samples, axis=0)
    clim = float(np.percentile(abs_all, clim_percentile))
    clim = max(clim, 1e-3)

    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    # Figure sizing: tuned for posters (wide, compact height)
    fig_w = max(12.0, 1.55 * n_cols)
    fig_h = 5.0
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.04, "hspace": 0.08},
    )

    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    im_ref = None
    for col, (t_ms, gi, pi) in enumerate(zip(times, gt_idxs, pr_idxs)):
        ax0 = axes[0, col]
        ax1 = axes[1, col]

        im0 = ax0.imshow(
            gt_500hz[gi],
            cmap=cmap,
            norm=norm,
            origin="upper",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
        )
        im1 = ax1.imshow(
            pred_10khz[pi],
            cmap=cmap,
            norm=norm,
            origin="upper",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
        )
        im_ref = im_ref or im0

        ax0.set_title(f"t = {float(t_ms):.1f} ms", fontsize=10, fontweight="bold")

        # Axes styling (publication white background)
        for ax in (ax0, ax1):
            ax.tick_params(labelsize=8)
            # Keep frames clean: y tick labels only on first column
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)

    # Axis labels
    axes[0, 0].set_ylabel("500 Hz\nAxial depth (mm)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("10 kHz\nAxial depth (mm)", fontsize=10, fontweight="bold")
    for ax in axes[1, :]:
        ax.set_xlabel("Lateral (mm)", fontsize=10)

    # Single colorbar
    cbar = fig.colorbar(
        im_ref,
        ax=axes.ravel().tolist(),
        fraction=0.018,
        pad=0.01,
    )
    cbar.set_label("Particle velocity (mm/s)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("500 Hz  vs  10 kHz", fontsize=22, fontweight="bold", y=1.04,
                 color="black", ha="center")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2-row poster montage: 500Hz vs 10kHz shear-wave velocity."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/10khz_predictions_20x",
        help="Directory containing gt_velocity_*.npy and velocity_10khz_*.npy",
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        default="2ft_10mmfocus_Plane_4_500Hz_20260129T130420_w0",
        help="Sample name (without prefix/suffix).",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="outputs/swe_publication_v10b/figures/fig12_wave_motion_500hz_vs_10khz.png",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default="outputs/swe_publication_v10b/figures/fig12_wave_motion_500hz_vs_10khz.pdf",
    )
    parser.add_argument("--t_start_ms", type=float, default=0.0)
    parser.add_argument("--t_end_ms", type=float, default=4.0)
    parser.add_argument("--dt_show_ms", type=float, default=0.5)
    parser.add_argument(
        "--clim_percentile",
        type=float,
        default=99.0,
        help="Percentile for symmetric colour limits over selected frames.",
    )
    parser.add_argument("--cmap", type=str, default="seismic")
    parser.add_argument(
        "--gt_time_mode",
        type=str,
        default="hold",
        choices=["hold", "nearest"],
        help="Mapping from continuous time to 500Hz frames: 'hold' (sample-and-hold) or 'nearest'.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gt_500, pred_10k = _load_arrays(input_dir, args.sample_name)

    out_png = Path(args.out_png)
    out_pdf = Path(args.out_pdf) if args.out_pdf else None

    make_montage(
        gt_500hz=gt_500,
        pred_10khz=pred_10k,
        sample_name=args.sample_name,
        out_png=out_png,
        out_pdf=out_pdf,
        t_start_ms=float(args.t_start_ms),
        t_end_ms=float(args.t_end_ms),
        dt_show_ms=float(args.dt_show_ms),
        clim_percentile=float(args.clim_percentile),
        cmap=str(args.cmap),
        gt_time_mode=str(args.gt_time_mode),
    )

    print(f"Saved poster montage -> {out_png}")
    if out_pdf is not None:
        print(f"Saved poster montage -> {out_pdf}")


if __name__ == "__main__":
    main()
