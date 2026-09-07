#!/usr/bin/env python3
"""
Matched-ground-truth comparison video — v12 / v14 / v14d vs real GT.
=====================================================================
For a native high-PRF acquisition (5/10/20 kHz, sparse_step=12), feeding 5
sparse input frames at step=12 gives dt_in ~= 2.4 ms (~417 Hz effective input
rate) -- the deployment scenario (500 Hz consecutive-frame input) -- while the
model's dense output lands frame-for-frame on the acquisition's OWN native
grid. That means the "ground truth" shown here is the real acquisition at its
native (upsampled) PRF, never a temporal interpolation of the 500 Hz input.
See scripts/evaluate_matched_gt.py for the numeric version of this protocol.

For each output video:
  Top row     — spatial velocity maps: GT | v12 | v14 | v14d  (animated)
  Bottom      — velocity profile: lateral waveform at the focal depth,
                GT vs each model's RECONSTRUCTED waveform overlaid (animated)

Usage:
    python scripts/visualize_matched_gt_comparison_video.py
    python scripts/visualize_matched_gt_comparison_video.py --categories 1ft 3ft
    python scripts/visualize_matched_gt_comparison_video.py --sample <name>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402

AXIAL_MM_PER_PX = 0.0507
LATERAL_MM_PER_PX = 0.1013
H_ROI, W_ROI = 116, 160
UPSAMPLE = 12
DATA_DIR = PROJECT_ROOT / "data" / "fullscale_velocity"

MODELS_DEFAULT = {
    "v12": PROJECT_ROOT / "saved_models" / "wavepinn_v12_best.pt",
    "v14": PROJECT_ROOT / "saved_models" / "wavepinn_v14_best.pt",
    "v14d": PROJECT_ROOT / "saved_models" / "wavepinn_v14d_best.pt",
}
MODEL_COLORS = {"v12": "#e67e22", "v14": "#2ca02c", "v14d": "#9b59b6"}
GT_COLOR = "#1f6fb4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 140,
})


def phantom_of(name: str) -> str:
    low = name.lower()
    if "cirs" in low:
        return "cirs"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}ft" if m else "other"


def pad16(a: np.ndarray) -> np.ndarray:
    *lead, H, W = a.shape
    pH, pW = (16 - H % 16) % 16, (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return a
    return np.pad(a, [(0, 0)] * len(lead) + [(0, pH), (0, pW)])


def load_model(path: Path, device: torch.device, label: str) -> torch.nn.Module:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [{label}] WARNING missing keys: {list(missing)[:5]}")
    print(f"  [{label}] loaded from {path.name} "
          f"(epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val_loss', float('nan')):.5f})")
    return m


def select_samples(meta: List[Dict], categories: List[str],
                    prfs: List[str]) -> List[Tuple[str, str, str]]:
    """One sample per (category, prf): the one with the largest velocity_std_mms
    (a proxy for a clearly visible propagating wave), so the video is legible."""
    by_key: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for e in meta:
        name = e["name"]
        freq = e.get("frequency")
        cat = phantom_of(name)
        if freq in prfs and cat in categories:
            by_key.setdefault((cat, freq), []).append(
                (name, float(e.get("velocity_std_mms", 0.0)))
            )
    chosen = []
    for cat in categories:
        for prf in prfs:
            items = by_key.get((cat, prf))
            if not items:
                continue
            items.sort(key=lambda x: -x[1])
            chosen.append((cat, prf, items[0][0]))
    return chosen


@torch.no_grad()
def run_sample(name: str, entry: Dict, models: Dict[str, torch.nn.Module],
               device: torch.device, split_dir: Path) -> Optional[Dict]:
    sp = split_dir / f"sparse_{name}.npy"
    gp = split_dir / f"velocity_{name}.npy"
    if not (sp.exists() and gp.exists()):
        return None

    step = int(entry.get("sparse_step", UPSAMPLE))
    dt_native = float(entry["frame_time_s"])
    dt_in = step * dt_native
    dt_out = dt_in / UPSAMPLE  # == dt_native for native-step acquisitions

    gt = np.load(gp).astype(np.float64)[:, :H_ROI, :W_ROI]
    T_dense = 1 + (5 - 1) * UPSAMPLE
    gt = gt[:T_dense]

    iq = np.load(sp).astype(np.complex64)
    iq_scale = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    vel_scale = max(float(np.percentile(np.abs(gt), 99)), 1.0)

    inp = pad16(np.stack([(iq / iq_scale).real, (iq / iq_scale).imag], axis=0))
    t = torch.from_numpy(inp).unsqueeze(0).to(device)

    preds: Dict[str, np.ndarray] = {}
    for label, model in models.items():
        out = model(t, dt_seconds=torch.tensor([dt_in], device=device))
        preds[label] = (out[0, 0].float().cpu().numpy()[:, :H_ROI, :W_ROI] * vel_scale)

    return dict(gt=gt.astype(np.float32), preds=preds, dt_ms=dt_out * 1e3,
                dt_in_ms=dt_in * 1e3, prf_label=entry.get("frequency", "?"))


def make_video(name: str, cat: str, gt: np.ndarray, preds: Dict[str, np.ndarray],
               dt_ms: float, dt_in_ms: float, prf_label: str, out_path: Path,
               fps: int = 10, dpi: int = 130) -> Path:
    T, H, W = gt.shape
    lat_mm = (np.arange(W) - W / 2) * LATERAL_MM_PER_PX
    dep_mm = np.arange(H) * AXIAL_MM_PER_PX
    extent = [lat_mm[0], lat_mm[-1], dep_mm[-1], dep_mm[0]]
    focal_row = H // 3
    focal_mm = focal_row * AXIAL_MM_PER_PX

    clim = max(float(np.abs(gt).max()), 0.5)
    model_labels = list(preds.keys())

    fig = plt.figure(figsize=(16, 7.2), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2, len(model_labels) + 2,
        width_ratios=[1] * (len(model_labels) + 1) + [0.035],
        height_ratios=[1, 0.8],
        wspace=0.08, hspace=0.35,
        left=0.05, right=0.96, top=0.86, bottom=0.09,
    )

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_models = [fig.add_subplot(gs[0, i + 1]) for i in range(len(model_labels))]
    ax_cb = fig.add_subplot(gs[0, -1])
    ax_wave = fig.add_subplot(gs[1, :])

    kw_img = dict(cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto",
                  origin="upper", extent=extent, interpolation="bilinear")

    im_gt = ax_gt.imshow(gt[0], **kw_img)
    ax_gt.set_title(f"Ground Truth\n(native {prf_label})", fontsize=9, fontweight="bold")
    ax_gt.set_ylabel("Depth (mm)", fontsize=8)
    ax_gt.set_xlabel("Lateral (mm)", fontsize=8)
    ax_gt.axhline(focal_mm, color="lime", lw=0.8, ls="--", alpha=0.6)

    im_models = []
    for ax, lbl in zip(ax_models, model_labels):
        im = ax.imshow(preds[lbl][0], **kw_img)
        im_models.append(im)
        ax.set_title(f"WavePINN {lbl}\n(reconstructed)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Lateral (mm)", fontsize=8)
        ax.set_yticks([])
        ax.axhline(focal_mm, color="lime", lw=0.8, ls="--", alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(-clim, clim))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("Particle velocity (mm/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Velocity profile panel: lateral waveform at focal depth.
    ax_wave.set_xlim(lat_mm[0], lat_mm[-1])
    ax_wave.set_ylim(-clim * 1.15, clim * 1.15)
    ax_wave.set_xlabel("Lateral position (mm)", fontsize=9)
    ax_wave.set_ylabel("Velocity (mm/s)", fontsize=9)
    ax_wave.axhline(0, color="gray", lw=0.6, alpha=0.5)
    ax_wave.grid(alpha=0.3)
    ax_wave.set_title(f"Velocity profile @ focal depth ({focal_mm:.1f} mm) — "
                       "reconstructed waveform vs ground truth", fontsize=9, fontweight="bold")

    line_gt, = ax_wave.plot(lat_mm, gt[0, focal_row], color=GT_COLOR, lw=2.2, label="GT (native)")
    lines_pred = []
    for lbl in model_labels:
        ln, = ax_wave.plot(lat_mm, preds[lbl][0, focal_row], color=MODEL_COLORS.get(lbl, "k"),
                            lw=1.6, ls="--", label=f"{lbl} (reconstructed)")
        lines_pred.append(ln)
    ax_wave.legend(fontsize=8, loc="upper right", ncol=len(model_labels) + 1, handlelength=1.6)

    sup = fig.suptitle("", fontsize=11, fontweight="bold", y=0.98)

    def _update(fi: int):
        im_gt.set_data(gt[fi])
        line_gt.set_ydata(gt[fi, focal_row])
        for im, ln, lbl in zip(im_models, lines_pred, model_labels):
            im.set_data(preds[lbl][fi])
            ln.set_ydata(preds[lbl][fi, focal_row])
        sup.set_text(f"{name}  ({cat.upper()}, native {prf_label})\n"
                     f"t = {fi * dt_ms:.3f} ms  |  frame {fi + 1}/{T}  |  "
                     f"input dt = {dt_in_ms:.2f} ms (500 Hz-class deployment)")
        return (im_gt, *im_models, line_gt, *lines_pred, sup)

    anim = FuncAnimation(fig, _update, frames=T, blit=False, interval=max(1000 // fps, 40))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p", "-crf", "20",
                                                "-preset", "medium", "-movflags", "+faststart"])
    anim.save(str(out_path), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "white"})
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--split", default="val")
    ap.add_argument("--categories", nargs="+", default=["1ft", "2ft", "3ft", "cirs"])
    ap.add_argument("--prfs", nargs="+", default=["5kHz", "10kHz", "20kHz"])
    ap.add_argument("--sample", default=None, help="Run a single named sample instead of auto-selecting.")
    ap.add_argument("--out", default="outputs/comparison_videos")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text())
    split_dir = data_dir / args.split

    print("Loading models...")
    models = {label: load_model(path, device, label) for label, path in MODELS_DEFAULT.items()}

    if args.sample:
        entry = next((e for e in meta[args.split] if e["name"] == args.sample), None)
        if entry is None:
            print(f"ERROR: sample {args.sample!r} not found in {args.split} split")
            return 1
        selections = [(phantom_of(args.sample), entry.get("frequency", "?"), args.sample)]
    else:
        selections = select_samples(meta[args.split], args.categories, args.prfs)

    if not selections:
        print("No matching samples found.")
        return 1

    print(f"\nSelected {len(selections)} sample(s):")
    for cat, prf, name in selections:
        print(f"  {cat:6s} {prf:6s} {name}")

    entries_by_name = {e["name"]: e for e in meta[args.split]}
    out_root = Path(args.out)
    written = []
    for i, (cat, prf, name) in enumerate(selections, 1):
        print(f"\n[{i}/{len(selections)}] {name}")
        entry = entries_by_name[name]
        result = run_sample(name, entry, models, device, split_dir)
        if result is None:
            print("  SKIP: sparse/velocity npy not found")
            continue
        out_path = out_root / cat / prf / f"{name}.mp4"
        make_video(name, cat, result["gt"], result["preds"], result["dt_ms"],
                   result["dt_in_ms"], result["prf_label"], out_path, fps=args.fps)
        print(f"  -> {out_path}")
        written.append(str(out_path))

    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps(written, indent=2))
    print(f"\nWrote {len(written)} video(s). Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
