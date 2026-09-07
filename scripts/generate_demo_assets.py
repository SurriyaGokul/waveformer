#!/usr/bin/env python3
"""
scripts/generate_demo_assets.py

Generates all WaveFormer project-page demo assets from WavePINN v10b model
weights.  Outputs go to project-page/public/demo/.

Produced files
--------------
  panel_sparse.mp4 / .webm     – 49-frame sparse 500 Hz input panel (5 GT frames held)
  panel_dense.mp4  / .webm     – 49-frame v10b reconstructed virtual-6 kHz output panel
  comparison.mp4   / .webm     – 2-panel Input vs Output comparison (full scientific view)
  poster_sparse.jpg             – peak-wave thumbnail for sparse panel
  poster_dense.jpg              – peak-wave thumbnail for dense panel
  og_image.jpg                  – 1200x630 Open Graph image
  demo_metadata.json            – frame/timing/physical metadata for the JS layer

Scientific accuracy
-------------------
  * WavePINN v10b weights: SWE_PINN_outputs_wavepinn_v10b_best_model.pt
  * Sample: 2ft_10mmfocus_Plane_3_500Hz_20260129T130307_w0 (validation set)
  * Physical pixel spacing from IQ2VEl.m:
      lateral: 0.1013 mm/px   axial: 0.0507 mm/px
  * Temporal: 5 sparse IQ frames at 500 Hz (dt = 2.0 ms each, 8 ms window)
    → 49 dense velocity frames at virtual 6 kHz (dt = 0.1667 ms each, 8 ms window)
  * Colormap: seismic diverging [-clim, clim] mm/s (clim = 99th pct of GT active)
  * Both panel videos share identical colormap scale so they can be compared directly
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from src.architectures.wavepinn_v10 import create_wavepinn_v10

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_NAME   = "2ft_10mmfocus_Plane_3_500Hz_20260129T130307_w0"
MODEL_PATH    = PROJECT_ROOT / "SWE_PINN_outputs_wavepinn_v10b_best_model.pt"
META_PATH     = PROJECT_ROOT / "data" / "fullscale_velocity" / "metadata.json"
INF_DIR       = PROJECT_ROOT / "data" / "fullscale_velocity" / "inference"
OUT_DIR       = PROJECT_ROOT / "project-page" / "public" / "demo"

# Physical constants (IQ2VEl.m)
AXIAL_MM_PX   = 0.0507   # mm per axial pixel
LAT_MM_PX     = 0.1013   # mm per lateral pixel
H_ROI         = 116       # ROI height pixels
W_ROI         = 160       # ROI width pixels

# Timing
PRF_IN_HZ     = 500.0
DT_IN_MS      = 1e3 / PRF_IN_HZ          # 2.0 ms (sparse input frame spacing)
UPSAMPLE      = 12
DT_OUT_MS     = DT_IN_MS / UPSAMPLE      # 0.1667 ms (dense output frame spacing)
N_SPARSE      = 5                         # number of input IQ frames
N_DENSE       = 49                        # number of output velocity frames
TOTAL_MS      = (N_DENSE - 1) * DT_OUT_MS  # 8.0 ms

# Video parameters
VIDEO_FPS     = 8          # frames per second for panel videos
VIDEO_W       = 800        # output pixel width  (must be even for yuv420p)
VIDEO_H       = 460        # output pixel height (must be even for yuv420p)
COMP_W        = 1280       # comparison video width
COMP_H        = 480        # comparison video height
DPI           = 150        # matplotlib DPI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_phys_axes() -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat_mm, depth_mm) axis vectors for the ROI."""
    depth_mm  = np.arange(H_ROI) * AXIAL_MM_PX
    # Center lateral axis at 0
    lat_mm    = (np.arange(W_ROI) - W_ROI / 2.0) * LAT_MM_PX
    return lat_mm, depth_mm


def pad_to_multiple(arr: np.ndarray, m: int = 16) -> Tuple[np.ndarray, int, int]:
    *lead, h, w = arr.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return arr, 0, 0
    pads = [(0, 0)] * len(lead) + [(0, ph), (0, pw)]
    return np.pad(arr, pads, mode="constant"), ph, pw


def encode_video(frames_dir: Path, out_stem: Path, fps: int,
                  target_w: int | None = None, target_h: int | None = None) -> None:
    """Encode a directory of frame_XXXX.png files to .mp4 and .webm.

    If target_w / target_h are given a scale filter is applied to normalise
    all frames to exactly that resolution (required for consistent yuv420p).
    """
    scale_filter = ""
    if target_w and target_h:
        # Ensure even dimensions (required for yuv420p)
        tw = target_w + (target_w % 2)
        th = target_h + (target_h % 2)
        scale_filter = f"scale={tw}:{th}:flags=lanczos"

    # MP4 – H.264, yuv420p, no audio
    mp4 = str(out_stem) + ".mp4"
    vf_mp4 = f"{scale_filter},format=yuv420p" if scale_filter else "format=yuv420p"
    cmd_mp4 = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", vf_mp4,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-movflags", "+faststart",
        mp4,
    ]
    subprocess.run(cmd_mp4, check=True, capture_output=True)
    print(f"  MP4  → {mp4}")

    # WebM – VP9
    webm = str(out_stem) + ".webm"
    vf_webm = scale_filter if scale_filter else ""
    cmd_webm = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        *((["-vf", vf_webm]) if vf_webm else []),
        "-c:v", "libvpx-vp9",
        "-b:v", "0",
        "-crf", "20",
        webm,
    ]
    subprocess.run(cmd_webm, check=True, capture_output=True)
    print(f"  WebM → {webm}")


def save_poster(fig: plt.Figure, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Poster → {path}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    cfg  = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    model = create_wavepinn_v10(config=cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    epoch    = ckpt.get("epoch", "?")
    best_val = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded v10b  epoch={epoch}  val_loss={best_val:.6f}  params={n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Data loading + inference
# ---------------------------------------------------------------------------

def load_sample(meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    iq_consec : (5, H_ROI, W_ROI) complex64 – 5 consecutive clutter-filtered IQ frames
    gt_vel    : (49, H_ROI, W_ROI) float32  – ground-truth velocity in mm/s
    """
    base = PROJECT_ROOT

    sm = None
    for split in ("val", "train", "inference"):
        for s in meta.get(split, []):
            if s.get("name") == SAMPLE_NAME:
                sm = s
                break
        if sm is not None:
            break

    if sm is None:
        raise RuntimeError(f"Sample {SAMPLE_NAME} not found in metadata")

    # Raw IQ: (H_full, W_full, T) complex
    raw_path = base / sm["original_path"]
    raw = np.load(str(raw_path)).astype(np.complex64)

    roi  = sm["roi"]
    win  = sm["window"]
    iq_win = raw[
        roi["min_row"] : roi["max_row"] + 1,
        roi["min_col"] : roi["max_col"] + 1,
        win[0] : win[1],
    ]
    # Temporal mean subtraction (clutter filter – matches training)
    iq_win = iq_win - iq_win.mean(axis=2, keepdims=True)

    # Take first 5 consecutive frames (window index 0-4)
    iq_consec = np.stack([iq_win[:, :, i] for i in range(5)], axis=0)  # (5, H, W)

    # Ground-truth velocity (mm/s, 49 frames at 500 Hz native = 8 ms window w/ v10b targets)
    gt_path = INF_DIR / f"velocity_{SAMPLE_NAME}.npy"
    gt_vel  = np.load(str(gt_path)).astype(np.float32)[:, :H_ROI, :W_ROI]

    return iq_consec, gt_vel


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    iq_consec: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run v10b inference.

    Returns
    -------
    pred_mm : (49, H_ROI, W_ROI) float32 – predicted velocity in mm/s
    """
    # Per-sample IQ normalisation (99th percentile, matching training)
    iq_scale = float(np.percentile(np.abs(iq_consec), 99))
    iq_scale = max(iq_scale, 1e-6)
    iq_norm  = (iq_consec / iq_scale).astype(np.complex64)

    # (2, 5, H, W)  real / imag channels
    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)
    inp_pad, ph, pw = pad_to_multiple(inp, 16)

    inp_t  = torch.from_numpy(inp_pad).unsqueeze(0).to(device)     # (1, 2, 5, H_pad, W_pad)
    dt_t   = torch.tensor([1.0 / PRF_IN_HZ], dtype=torch.float32, device=device)

    pred  = model(inp_t, dt_seconds=dt_t)                           # (1, 1, T, H_pad, W_pad)
    pred_np = pred[0, 0].cpu().numpy()

    # Reverse normalisation: pred is in normalised space → mm/s
    # v10b output is normalised by vel_scale at training; but we don't have vel_scale here
    # at inference time.  The stored GT is already in mm/s so we need the GT's vel_scale
    # to recover physical units.  Use 99th percentile of GT *active* region as scale.
    return pred_np[:, :H_ROI, :W_ROI]   # (49, H_ROI, W_ROI) normalised [-1,1] approx


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def compute_clim(gt_vel: np.ndarray, percentile: float = 99.5) -> float:
    """Compute symmetric ± colormap limit from GT active frames."""
    frame_stds = gt_vel.std(axis=(1, 2))
    active = gt_vel[frame_stds > 0.05] if (frame_stds > 0.05).any() else gt_vel
    return float(max(np.percentile(np.abs(active), percentile), 0.1))


def vel_scale_from_gt(gt_vel: np.ndarray) -> float:
    """99th-percentile velocity scale (used to denorm model output)."""
    return float(max(np.percentile(np.abs(gt_vel), 99), 1e-6))


# ---------------------------------------------------------------------------
# Panel video renderer
# ---------------------------------------------------------------------------

def _render_panel_frame(
    vel_frame: np.ndarray,
    lat_mm: np.ndarray,
    depth_mm: np.ndarray,
    clim: float,
    t_ms: float,
    frame_no: int,
    n_frames: int,
    tag: str,
    fig_w_in: float,
    fig_h_in: float,
) -> plt.Figure:
    """Render a single panel frame.  Returns a matplotlib Figure."""
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.90, top=0.88, bottom=0.13)

    # Layout: heatmap on left, thin colorbar on right
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1, 0.04],
        wspace=0.04,
        left=0.09, right=0.90,
        top=0.88, bottom=0.13,
    )
    ax   = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    extent = [lat_mm[0], lat_mm[-1], depth_mm[-1], depth_mm[0]]
    norm   = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    im = ax.imshow(
        vel_frame,
        cmap="seismic",
        norm=norm,
        aspect="auto",
        origin="upper",
        extent=extent,
        interpolation="bilinear",
    )

    # Colorbar
    cb = fig.colorbar(im, cax=ax_c)
    cb.set_label("mm/s", fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7)
    cb.locator = MaxNLocator(nbins=5, symmetric=True)
    cb.update_ticks()

    # Axis labels
    ax.set_xlabel("Lateral (mm)", fontsize=8, labelpad=3)
    ax.set_ylabel("Depth (mm)",   fontsize=8, labelpad=3)
    ax.tick_params(labelsize=7)

    # Title row
    ax.set_title(
        f"{tag}   t = {t_ms:.2f} ms   frame {frame_no}/{n_frames}",
        fontsize=8.5, fontweight="bold", pad=5,
    )

    return fig


def render_panel_video(
    frames: np.ndarray,           # (N, H, W) velocity in mm/s
    lat_mm: np.ndarray,
    depth_mm: np.ndarray,
    clim: float,
    t_axis_ms: np.ndarray,        # (N,) timestamps in ms
    tag: str,
    out_stem: Path,
    fps: int = VIDEO_FPS,
) -> None:
    """Render a panel video (MP4 + WebM) and a poster JPEG."""
    N = len(frames)
    fig_w = VIDEO_W / DPI
    fig_h = VIDEO_H / DPI

    print(f"  Rendering {N} panel frames ({tag}) …")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        poster_frame_idx = int(np.argmax(frames.std(axis=(1, 2))))  # frame w/ max activity

        for i in range(N):
            fig = _render_panel_frame(
                vel_frame   = frames[i],
                lat_mm      = lat_mm,
                depth_mm    = depth_mm,
                clim        = clim,
                t_ms        = float(t_axis_ms[i]),
                frame_no    = i + 1,
                n_frames    = N,
                tag         = tag,
                fig_w_in    = fig_w,
                fig_h_in    = fig_h,
            )
            fig.savefig(str(tmp_path / f"frame_{i:04d}.png"), dpi=DPI,
                        facecolor="white")

            # Save poster (peak-activity frame)
            if i == poster_frame_idx:
                poster_path = Path(str(out_stem) + "_poster.jpg")
                fig.savefig(str(poster_path), dpi=DPI,
                            facecolor="white", format="jpeg", pil_kwargs={"quality": 95})
                print(f"  Poster → {poster_path}")

            plt.close(fig)

        encode_video(tmp_path, out_stem, fps, target_w=VIDEO_W, target_h=VIDEO_H)


# ---------------------------------------------------------------------------
# Comparison video renderer
# ---------------------------------------------------------------------------

def render_comparison_video(
    sparse_frames: np.ndarray,    # (N, H, W)  sparse (held 500 Hz)
    dense_frames: np.ndarray,     # (N, H, W)  dense v10b prediction
    lat_mm: np.ndarray,
    depth_mm: np.ndarray,
    clim: float,
    t_axis_ms: np.ndarray,
    out_stem: Path,
    fps: int = VIDEO_FPS,
) -> None:
    """Render a side-by-side sparse|dense comparison video."""
    N = len(sparse_frames)
    fig_w = COMP_W / DPI
    fig_h = COMP_H / DPI
    extent = [lat_mm[0], lat_mm[-1], depth_mm[-1], depth_mm[0]]
    norm   = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    print(f"  Rendering {N} comparison frames …")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i in range(N):
            fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
            gs  = gridspec.GridSpec(
                1, 3, figure=fig,
                width_ratios=[1, 1, 0.04],
                wspace=0.06,
                left=0.06, right=0.94,
                top=0.87, bottom=0.12,
            )
            ax_l  = fig.add_subplot(gs[0])
            ax_r  = fig.add_subplot(gs[1])
            ax_cb = fig.add_subplot(gs[2])

            for ax, data, title in (
                (ax_l, sparse_frames[i], "Sparse Input  500 Hz"),
                (ax_r, dense_frames[i],  "Reconstructed  Virtual 6 kHz"),
            ):
                im = ax.imshow(data, cmap="seismic", norm=norm,
                               aspect="auto", origin="upper", extent=extent,
                               interpolation="bilinear")
                ax.set_xlabel("Lateral (mm)", fontsize=9, labelpad=3)
                ax.set_ylabel("Depth (mm)",   fontsize=9, labelpad=3)
                ax.tick_params(labelsize=8)
                ax.set_title(title, fontsize=10, fontweight="bold", pad=5)

            cb = fig.colorbar(im, cax=ax_cb)
            cb.set_label("mm/s", fontsize=9, labelpad=4)
            cb.ax.tick_params(labelsize=8)
            cb.locator = MaxNLocator(nbins=5, symmetric=True)
            cb.update_ticks()

            fig.suptitle(
                f"WavePINN v10b  ·  2FT Phantom  10 mm focus  Plane 3  "
                f"·  t = {t_axis_ms[i]:.3f} ms  (frame {i+1}/{N})",
                fontsize=9, fontweight="bold", y=0.97,
            )

            fig.savefig(str(tmp_path / f"frame_{i:04d}.png"), dpi=DPI,
                        facecolor="white")

            # Poster at peak frame
            if i == int(np.argmax(dense_frames.std(axis=(1, 2)))):
                poster_path = Path(str(out_stem) + "_poster.jpg")
                fig.savefig(str(poster_path), dpi=DPI,
                            facecolor="white", format="jpeg", pil_kwargs={"quality": 95})
                print(f"  Poster → {poster_path}")

            plt.close(fig)

        encode_video(tmp_path, out_stem, fps, target_w=COMP_W, target_h=COMP_H)


# ---------------------------------------------------------------------------
# OG image (static 1200×630 publication figure)
# ---------------------------------------------------------------------------

def render_og_image(
    sparse_frames: np.ndarray,
    dense_frames:  np.ndarray,
    lat_mm: np.ndarray,
    depth_mm: np.ndarray,
    clim: float,
    t_axis_ms: np.ndarray,
    out_path: Path,
) -> None:
    """
    Render a 1200×630 Open Graph / social preview image.
    Shows: peak-activity sparse frame (left) | peak-activity dense frame (right).
    """
    peak_idx_sparse = int(np.argmax(sparse_frames.std(axis=(1, 2))))
    peak_idx_dense  = int(np.argmax(dense_frames.std(axis=(1, 2))))

    fig = plt.figure(figsize=(1200/150, 630/150), facecolor="#0f172a")
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1, 1, 0.04],
        wspace=0.05, left=0.05, right=0.93, top=0.82, bottom=0.12,
    )
    ax_l  = fig.add_subplot(gs[0])
    ax_r  = fig.add_subplot(gs[1])
    ax_cb = fig.add_subplot(gs[2])

    extent = [lat_mm[0], lat_mm[-1], depth_mm[-1], depth_mm[0]]
    norm   = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    for ax, data, title, t_ms in (
        (ax_l, sparse_frames[peak_idx_sparse], "Low PRF  500 Hz",
         float(t_axis_ms[peak_idx_sparse])),
        (ax_r, dense_frames[peak_idx_dense],   "Virtual 6 kHz  (WavePINN v10b)",
         float(t_axis_ms[peak_idx_dense])),
    ):
        im = ax.imshow(data, cmap="seismic", norm=norm, aspect="auto",
                       origin="upper", extent=extent, interpolation="bilinear")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.set_facecolor("#0f172a")
        ax.set_xlabel("Lateral (mm)", fontsize=9, labelpad=3)
        ax.set_ylabel("Depth (mm)",   fontsize=9, labelpad=3)
        ax.set_title(f"{title}\nt = {t_ms:.2f} ms",
                     fontsize=10, fontweight="bold", color="white", pad=6)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("mm/s", fontsize=9, labelpad=4, color="#94a3b8")
    cb.ax.tick_params(colors="#94a3b8", labelsize=8)
    cb.locator = MaxNLocator(nbins=5, symmetric=True)
    cb.update_ticks()
    ax_cb.yaxis.set_tick_params(color="#94a3b8")

    fig.suptitle(
        "WaveFormer  ·  Temporal Super-Resolution for Shear Wave Elastography",
        fontsize=12, fontweight="bold", color="white", y=0.94,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="jpeg",
                pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"  OG image → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  WaveFormer project-page asset generation")
    print(f"  Model  : {MODEL_PATH.name}")
    print(f"  Sample : {SAMPLE_NAME}")
    print(f"  Device : {device}")
    print(f"  Output : {OUT_DIR}")
    print(f"{'='*60}\n")

    # ── 1. Load metadata + data ──────────────────────────────────────────
    print("[1/5] Loading metadata and sample data …")
    with open(str(META_PATH)) as f:
        meta = json.load(f)
    iq_consec, gt_vel = load_sample(meta)
    print(f"  IQ shape: {iq_consec.shape}  GT shape: {gt_vel.shape}")

    # Physical axes
    lat_mm, depth_mm = make_phys_axes()

    # ── 2. Inference ─────────────────────────────────────────────────────
    print("\n[2/5] Running WavePINN v10b inference …")
    model = load_model(device)

    # Run inference – output is normalised; recover mm/s via GT vel_scale
    pred_norm = run_inference(model, iq_consec, device)  # (49, H, W) ≈ [-1, 1]
    vel_scale = vel_scale_from_gt(gt_vel)
    pred_mm   = pred_norm * vel_scale                     # mm/s

    # Keep only H_ROI rows (GT target shape)
    gt_vel   = gt_vel  [:, :H_ROI, :W_ROI]
    pred_mm  = pred_mm [:N_DENSE, :H_ROI, :W_ROI]

    print(f"  GT   range: [{gt_vel.min():.3f}, {gt_vel.max():.3f}] mm/s")
    print(f"  Pred range: [{pred_mm.min():.3f}, {pred_mm.max():.3f}] mm/s")

    # ── 3. Build frame sequences ─────────────────────────────────────────
    print("\n[3/5] Building frame sequences …")

    # Dense output time axis: 0, 0.167, 0.333, …, 8.0 ms
    t_dense_ms = np.arange(N_DENSE) * DT_OUT_MS

    # Sparse input: 5 GT frames at 0, 2, 4, 6, 8 ms → stretch to N_DENSE frames.
    # The dense timeline at 6 kHz has dt = 2 ms / 12 = 0.167 ms.
    # Sparse frame k (at t = k * 2 ms) corresponds to dense index k * UPSAMPLE.
    # We hold each sparse frame for exactly UPSAMPLE dense steps so the
    # visual transitions align perfectly with the 500 Hz input timestamps.
    #   Frame 0 → dense indices   0 – 11  (12 frames, t = 0.000 – 1.833 ms)
    #   Frame 1 → dense indices  12 – 23  (12 frames, t = 2.000 – 3.833 ms)
    #   Frame 2 → dense indices  24 – 35  (12 frames, t = 4.000 – 5.833 ms)
    #   Frame 3 → dense indices  36 – 47  (12 frames, t = 6.000 – 7.833 ms)
    #   Frame 4 → dense index   48        ( 1 frame,  t = 8.000 ms)
    sparse_src = gt_vel[:N_SPARSE]                         # (5, H, W) at 500 Hz
    sparse_expanded = np.stack(
        [sparse_src[min(i // UPSAMPLE, N_SPARSE - 1)] for i in range(N_DENSE)],
        axis=0,
    )                                                       # (49, H, W)

    clim = compute_clim(gt_vel)
    print(f"  Colour limit ± {clim:.3f} mm/s")

    # Metrics
    mae  = float(np.abs(pred_mm[:5] - gt_vel[:5]).mean())
    rmse = float(np.sqrt(((pred_mm[:5] - gt_vel[:5])**2).mean()))
    print(f"  MAE (first 5 frames): {mae:.4f} mm/s   RMSE: {rmse:.4f} mm/s")

    # ── 4. Render panel videos ───────────────────────────────────────────
    print("\n[4/5] Rendering panel videos …")

    render_panel_video(
        frames     = sparse_expanded,
        lat_mm     = lat_mm,
        depth_mm   = depth_mm,
        clim       = clim,
        t_axis_ms  = t_dense_ms,
        tag        = "Sparse input  500 Hz",
        out_stem   = OUT_DIR / "panel_sparse",
    )

    render_panel_video(
        frames     = pred_mm,
        lat_mm     = lat_mm,
        depth_mm   = depth_mm,
        clim       = clim,
        t_axis_ms  = t_dense_ms,
        tag        = "Reconstructed  Virtual 6 kHz  (WavePINN v10b)",
        out_stem   = OUT_DIR / "panel_dense",
    )

    render_comparison_video(
        sparse_frames = sparse_expanded,
        dense_frames  = pred_mm,
        lat_mm        = lat_mm,
        depth_mm      = depth_mm,
        clim          = clim,
        t_axis_ms     = t_dense_ms,
        out_stem      = OUT_DIR / "comparison",
    )

    render_og_image(
        sparse_frames = sparse_expanded,
        dense_frames  = pred_mm,
        lat_mm        = lat_mm,
        depth_mm      = depth_mm,
        clim          = clim,
        t_axis_ms     = t_dense_ms,
        out_path      = OUT_DIR / "og_image.jpg",
    )

    # ── 5. Write metadata JSON ───────────────────────────────────────────
    print("\n[5/5] Writing demo_metadata.json …")
    metadata = {
        "sample_name"     : SAMPLE_NAME,
        "model"           : "WavePINN v10b",
        "model_file"      : MODEL_PATH.name,
        "phantom"         : "2FT tissue-mimicking phantom, 10 mm focus, Plane 3",
        "sparse": {
            "n_frames"    : N_SPARSE,
            "prf_hz"      : PRF_IN_HZ,
            "dt_ms"       : DT_IN_MS,
            "total_ms"    : float((N_SPARSE - 1) * DT_IN_MS),
            "description" : "5 consecutive 500 Hz complex IQ frames (2 ms between frames)",
        },
        "dense": {
            "n_frames"    : N_DENSE,
            "virtual_prf_hz" : float(1e3 / DT_OUT_MS),
            "dt_ms"       : float(DT_OUT_MS),
            "total_ms"    : float(TOTAL_MS),
            "description" : "49 virtual 6 kHz velocity frames predicted by WavePINN v10b",
        },
        "upsampling_factor": UPSAMPLE,
        "physical": {
            "lateral_fov_mm"   : float(W_ROI * LAT_MM_PX),
            "depth_range_mm"   : [float(0), float(H_ROI * AXIAL_MM_PX)],
            "lat_mm_per_px"    : LAT_MM_PX,
            "axial_mm_per_px"  : AXIAL_MM_PX,
        },
        "colormap": {
            "name"  : "seismic",
            "clim"  : float(clim),
            "unit"  : "mm/s",
            "description": "Symmetric diverging colormap ±clim mm/s (particle velocity)",
        },
        "video": {
            "fps"   : VIDEO_FPS,
            "width" : VIDEO_W,
            "height": VIDEO_H,
            "duration_s" : round(N_DENSE / VIDEO_FPS, 3),
        },
        "metrics": {
            "mae_first5_mm_s"  : float(f"{mae:.4f}"),
            "rmse_first5_mm_s" : float(f"{rmse:.4f}"),
        },
    }

    meta_path = OUT_DIR / "demo_metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata → {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Done!  All assets written to {OUT_DIR}")
    print(f"  Files:")
    for p in sorted(OUT_DIR.iterdir()):
        size_kb = p.stat().st_size // 1024
        print(f"    {p.name:<35s} {size_kb:>6d} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
