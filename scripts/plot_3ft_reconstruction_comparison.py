#!/usr/bin/env python3
"""
plot_3ft_reconstruction_comparison.py
======================================
Generates a publication-quality comparison panel of shear-wave velocity
reconstructions for a representative 3FT phantom (10-mm focus) sample.

Rows (5 methods):
  1. GT 5 kHz SWE           — gold standard, 49 frames at 5 kHz
  2. GT 500 Hz SWE          — sparse input, 5 frames at 500 Hz
  3. Linear interpolation   — 5 sparse frames → 49 frames at 6 kHz
  4. Cubic-spline interp.   — 5 sparse frames → 49 frames at 6 kHz
  5. WavePINN v10b (ours)   — 49 frames at 6 kHz

Columns: 5 equally-spaced physical time points shared across all methods:
  t = 0, 2, 4, 6, 8 ms  (i.e. the five 500-Hz sparse frame times)
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import interp1d

# ── configuration ──────────────────────────────────────────────────────────
SAMPLE_STEM_500   = "3ft_10mmfocus_Plane_3_500Hz_20260129T142653_w0"
SAMPLE_STEM_PRED  = SAMPLE_STEM_500                               # same name
SAMPLE_STEM_5KHZ  = "3ft_10mmfocus_Plane_3_5kHz_20260129T144124_w0"

PATH_500  = f"data/fullscale_velocity/inference/velocity_{SAMPLE_STEM_500}.npy"
PATH_PRED = f"outputs/6khz_predictions/velocity_6khz_{SAMPLE_STEM_PRED}.npy"
PATH_5KHZ = f"data/fullscale_velocity/val/velocity_{SAMPLE_STEM_5KHZ}.npy"

DT_6KHZ  = 1.0 / 6000.0   # s per dense frame
DT_5KHZ  = 1.0 / 5000.0   # s per 5-kHz frame
DT_500HZ = 1.0 / 500.0    # s per sparse frame

N_DENSE   = 49
N_SPARSE  = 5
SPARSE_STEP_6K = 12        # sparse frame i → 6-kHz frame i*12

# Seven physical display times (ms) — sub-ms resolution to capture wave propagation.
# The 3FT shear wave (cs ~ 3.84 m/s, ROI ~ 16 mm) crosses the field in ~1.5 ms.
# Using 6 kHz dt = 0.167 ms; choose times that are exact multiples for clean frame indices.
# t = 0, 2, 4, 6, 8, 10, 12 × (1/6000) s = 0, 0.33, 0.67, 1.00, 1.33, 1.67, 2.00 ms
DISPLAY_TIMES_MS = [round(i * 1000 / 6000, 3) for i in [0, 2, 4, 6, 8, 10]]
N_COLS = len(DISPLAY_TIMES_MS)

LATERAL_MM_PER_PX = 0.1013   # mm / pixel
AXIAL_MM_PER_PX   = 0.0507   # mm / pixel (half of lateral, typical for SWE depth)

OUT_DIR = Path("outputs/final_tables/velocity_maps")
OUT_PATH = OUT_DIR / "3ft_reconstruction_comparison.png"

# ── helpers ────────────────────────────────────────────────────────────────

def norm99(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise by 99th-percentile of absolute value; return (normed, scale)."""
    scale = float(np.percentile(np.abs(arr), 99))
    scale = max(scale, 1e-9)
    return arr / scale, scale


def interp_sparse_to_dense(vel_sparse: np.ndarray, method: str) -> np.ndarray:
    """Up-sample 5 sparse 500-Hz frames → 49 frames at 6 kHz."""
    T_s, H, W = vel_sparse.shape
    t_sparse = np.arange(T_s) * DT_500HZ
    t_dense  = np.arange(N_DENSE) * DT_6KHZ
    t_clip   = np.clip(t_dense, t_sparse[0], t_sparse[-1])
    vel_flat = vel_sparse.reshape(T_s, -1)
    f = interp1d(
        t_sparse, vel_flat, axis=0,
        kind=method,
        bounds_error=False,
        fill_value=(vel_flat[0], vel_flat[-1]),
    )
    return f(t_clip).reshape(N_DENSE, H, W).astype(np.float32)


def get_frame_index(method: str, t_ms: float) -> int | None:
    """Return the array frame index closest to display time t_ms.
    Returns None if the method cannot represent this time point."""
    if method == "gt5k":
        idx = round(t_ms * 1e-3 / DT_5KHZ)
    elif method == "gt500":
        # Sparse 500 Hz frames are at t = 0, 2, 4, 6, 8 ms only.
        # Only return a frame if t_ms is within 0.1 ms of a valid sparse time.
        sparse_times_ms = [i * DT_500HZ * 1e3 for i in range(N_SPARSE)]  # [0,2,4,6,8]
        closest = min(sparse_times_ms, key=lambda st: abs(st - t_ms))
        if abs(closest - t_ms) > 0.1:
            return None  # no sparse frame near this display time
        idx = sparse_times_ms.index(closest)
    else:  # 6kHz outputs (pred, linear, cubic)
        idx = round(t_ms * 1e-3 / DT_6KHZ)
    return int(min(idx, N_DENSE - 1))


# ── load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
vel_500  = np.load(PATH_500).astype(np.float32)   # (49, H, W) normalised ±1
vel_pred = np.load(PATH_PRED).astype(np.float32)  # (49, H, W) normalised ±1
vel_5khz = np.load(PATH_5KHZ).astype(np.float32)  # (49, H, W) mm/s

sparse_frames = vel_500[:N_SPARSE]                # first 5 frames
vel_linear = interp_sparse_to_dense(sparse_frames, "linear")
vel_cubic  = interp_sparse_to_dense(sparse_frames, "cubic")

# Normalise 5-kHz GT to ±1 for display; record scale
vel_5khz_norm, scale_5khz = norm99(vel_5khz)

# Normalise each sequence independently by its own 99th-percentile amplitude.
# This makes the wavefront pattern visible in every row regardless of absolute
# amplitude differences between methods. Amplitude magnitudes are reported
# in the tables; the maps show spatial propagation patterns.
def row_norm(arr: np.ndarray) -> np.ndarray:
    p = float(np.percentile(np.abs(arr), 99))
    p = max(p, 1e-9)
    return arr / p

vel_5khz_norm  = row_norm(vel_5khz)
vel_500_norm   = row_norm(vel_500)
vel_linear_n   = row_norm(vel_linear)
vel_cubic_n    = row_norm(vel_cubic)
vel_pred_n     = row_norm(vel_pred)

H, W = vel_500.shape[1], vel_500.shape[2]
x_mm = np.arange(W) * LATERAL_MM_PER_PX
z_mm = np.arange(H) * AXIAL_MM_PER_PX

# ── layout ─────────────────────────────────────────────────────────────────
ROW_LABELS = [
    "GT 5 kHz",
    "Linear\nInterpolation",
    "Cubic-Spline\nInterpolation",
    "WaveFormer\n(ours)",
]

METHOD_DATA = [
    ("gt5k",   vel_5khz_norm),
    ("6khz",   vel_linear_n),
    ("6khz",   vel_cubic_n),
    ("6khz",   vel_pred_n),
]

N_ROWS = len(ROW_LABELS)
FIG_W  = 2.6 * N_COLS + 2.0   # extra for row labels + colorbar
FIG_H  = 2.0 * N_ROWS + 0.9   # extra for col titles + title

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=180)
gs = gridspec.GridSpec(
    N_ROWS, N_COLS + 1,
    width_ratios=[1.0] * N_COLS + [0.06],
    hspace=0.12,
    wspace=0.08,
    left=0.20, right=0.97,
    top=0.93,  bottom=0.04,
)

CMAP = "seismic"
VMAX = 1.5

axes = [[fig.add_subplot(gs[r, c]) for c in range(N_COLS)] for r in range(N_ROWS)]
cax  = fig.add_subplot(gs[:, N_COLS])

im_ref = None  # to attach colorbar

for r, (mtype, data) in enumerate(METHOD_DATA):
    for c, t_ms in enumerate(DISPLAY_TIMES_MS):
        ax = axes[r][c]
        fidx = get_frame_index(mtype, t_ms)

        if fidx is None:
            # 500 Hz GT: no frame available at this sub-ms time — show "no data" panel
            ax.set_facecolor("#d8d8d8")
            ax.text(0.5, 0.5, "no data\n(500 Hz)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color="#555555",
                    style="italic")
        else:
            frame = data[fidx]
            im = ax.imshow(
                frame,
                cmap=CMAP,
                vmin=-VMAX, vmax=VMAX,
                aspect="auto",
                extent=[0, x_mm[-1], z_mm[-1], 0],
                interpolation="none",
            )
            if im_ref is None:
                im_ref = im

        # Axis decoration
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Column title (top row only)
        if r == 0:
            ax.set_title(f"t = {t_ms:.2f} ms", fontsize=9.5, fontweight="semibold", pad=5)

    # Row label: placed in figure coords at the vertical midpoint of this row's axes
    ax0 = axes[r][0]
    # Get the bounding box of the first axes cell in figure fraction coordinates
    bbox = ax0.get_position()          # Bbox in figure fractions
    y_mid = bbox.y0 + bbox.height / 2  # vertical centre of the row
    x_left = bbox.x0 - 0.012           # just to the left of the image
    fig.text(
        x_left, y_mid,
        ROW_LABELS[r],
        fontsize=14,
        fontweight="bold",
        color="#111111",
        rotation=90,
        va="center",
        ha="center",
    )

# Highlight our method row (last row) with a coloured border
for c in range(N_COLS):
    for spine in axes[N_ROWS - 1][c].spines.values():
        spine.set_edgecolor("#1a7a32")
        spine.set_linewidth(1.8)

# Colorbar
if im_ref is not None:
    cb = plt.colorbar(im_ref, cax=cax, orientation="vertical")
    cb.set_label("Norm. velocity\n(per-row 99th pct)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

fig.suptitle(
    "Shear Wave Velocity Reconstruction Comparison",
    fontsize=20, fontweight="bold",
    x=0.585, y=0.99, ha="center",
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PATH), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
