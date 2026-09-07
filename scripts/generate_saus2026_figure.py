#!/usr/bin/env python3
"""
SAUS 2026 abstract — panel (a): shear-wave propagation in a 1FT PVA phantom.

Four rows, one shared time axis:
    GT 5 kHz            native 5 kHz acquisition of the same imaging plane
    Linear              linear temporal interpolation of the 5 sparse knots
    Cubic-spline        cubic-spline interpolation of the same 5 knots
    WaveFormer (ours)   v14 checkpoint, 5 sparse 500 Hz IQ frames -> 49 frames

Every panel uses the SAME lateral extent, the SAME axial extent, a true 1:1
mm aspect ratio (axial pitch 0.0507 mm/px, lateral 0.1013 mm/px), and ONE
shared colour scale.

Two stages:
    --scan    run every paired 1FT test window, cache the fields, print a
              legibility score so the nicest sequence can be picked
    --sample  render the figure for one cached window

Usage:
    python scripts/generate_saus2026_figure.py --scan
    python scripts/generate_saus2026_figure.py --sample <stem>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dispersion import find_push_column, depth_average  # noqa: E402
from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402
from src.utils.deployment_input import build_sparse_iq  # noqa: E402

DT_500HZ = 1.0 / 500.0
UPSAMPLE = 12
DT_OUT = DT_500HZ / UPSAMPLE          # 6 kHz equivalent
DT_5KHZ = 1.0 / 5000.0
H_ROI = 116
N_DENSE = 49
N_SPARSE = 5

AXIAL_MM_PX = 0.0507
LATERAL_MM_PX = 0.1013

CACHE = Path("outputs/saus2026/fig_cache")
INPUT_STEP = 1   # set from --input-step in main()


def pad16(a: np.ndarray) -> np.ndarray:
    *lead, H, W = a.shape
    pH, pW = (16 - H % 16) % 16, (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return a
    return np.pad(a, [(0, 0)] * len(lead) + [(0, pH), (0, pW)])


def load_model(path: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).eval()
    m.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  checkpoint epoch={ckpt.get('epoch','?')} "
          f"val={ckpt.get('best_val_loss', float('nan')):.5f}")
    return m


def find_velocity_500hz(stem: str, data_dir: Path) -> Optional[Path]:
    for split in ("test", "inference", "val", "train"):
        p = data_dir / split / f"velocity_{stem}.npy"
        if p.exists():
            return p
    return None


_META: Optional[Dict[str, Dict]] = None


def entry_of(stem: str, data_dir: Path) -> Optional[Dict]:
    global _META
    if _META is None:
        meta = json.loads((data_dir / "metadata.json").read_text())
        _META = {e["name"]: e
                 for split in ("test", "inference", "val", "train")
                 for e in meta.get(split, [])}
    return _META.get(stem)


def roi_of(stem: str, data_dir: Path) -> Dict:
    e = entry_of(stem, data_dir)
    return e["roi"] if e else dict(min_row=0, max_row=H_ROI - 1)


def interp_baseline(v: np.ndarray, step: int, kind: str) -> np.ndarray:
    """Interpolate exactly the frames the model receives (native indices
    0, step, 2*step ...) onto the dense output grid."""
    src = np.arange(N_SPARSE) * step
    dst = np.arange(N_SPARSE) * UPSAMPLE
    f = interp1d(dst.astype(float), v[src], axis=0, kind=kind,
                 bounds_error=False, fill_value="extrapolate")
    return f(np.arange(N_DENSE, dtype=float))


def pair_group(eval_dir: Path, group: str) -> List[Dict]:
    sp_re = re.compile(r"sparse_(.+?)_500Hz_(\d{8}T\d+)(?:_w(\d+))?\.npy$")
    gt_re = re.compile(r"velocity_(.+?)_5kHz_(\d{8}T\d+)(?:_w(\d+))?\.npy$")
    gts: Dict[str, List[Path]] = {}
    for f in sorted(eval_dir.glob("velocity_*5kHz*.npy")):
        m = gt_re.match(f.name)
        if m:
            gts.setdefault(m.group(1), []).append(f)
    out = []
    for f in sorted(eval_dir.glob(f"sparse_{group}*500Hz*.npy")):
        m = sp_re.match(f.name)
        if m and m.group(1) in gts:
            out.append(dict(plane=m.group(1), sparse=f,
                            stem=f.name[len("sparse_"):-len(".npy")],
                            gts=gts[m.group(1)]))
    return out


def wave_score(field: np.ndarray) -> Dict:
    """How legible is the propagating wavefront?  A clean sequence has (i) a
    front that actually travels and (ii) energy spread over many frames rather
    than dumped into frame 0."""
    v = depth_average(np.asarray(field, float), (0.25, 0.75))   # (T, W)
    e = np.sqrt((v ** 2).mean(axis=1))
    if e.max() <= 0:
        return dict(travel_mm=0.0, spread=0.0, score=0.0)
    push = find_push_column(v)
    W = v.shape[1]
    # position of the peak |v| on each side of the push, per frame
    trav = []
    for t in range(v.shape[0]):
        row = np.abs(v[t])
        if row.max() < 0.05 * np.abs(v).max():
            continue
        right = row[push + 15:]
        if right.size and right.max() > 0.15 * row.max():
            trav.append((push + 15 + int(np.argmax(right))) * LATERAL_MM_PX)
    travel_mm = (max(trav) - min(trav)) if len(trav) > 2 else 0.0
    spread = float((e > 0.25 * e.max()).sum())
    return dict(travel_mm=float(travel_mm), spread=spread,
                score=float(travel_mm * np.sqrt(spread)))


def build_fields(stem: str, sparse: Path, gt_path: Path, model,
                 data_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    entry = entry_of(stem, data_dir)
    if entry is None:
        return None
    iq = build_sparse_iq(entry, step=INPUT_STEP)
    if iq is None:
        return None

    gt = np.load(gt_path).astype(np.float64)[:N_DENSE, :H_ROI, :]
    vel_scale = float(max(np.percentile(np.abs(gt), 99), 1.0))

    fields = {"GT 5 kHz": gt}
    v500 = find_velocity_500hz(stem, data_dir)
    if v500 is not None:
        v = np.load(v500).astype(np.float64)[:N_DENSE, :H_ROI, :]
        fields["Linear\ninterp."] = interp_baseline(v, INPUT_STEP, "linear")
        fields["Cubic\nspline"] = interp_baseline(v, INPUT_STEP, "cubic")

    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    inp = pad16(np.stack([(iq / iq_s).real, (iq / iq_s).imag], axis=0))
    with torch.no_grad():
        out = model(torch.from_numpy(inp).unsqueeze(0),
                    dt_seconds=torch.tensor([DT_500HZ]))
    fields["WaveFormer\n(ours)"] = out[0, 0].float().numpy()[:, :H_ROI, :] * vel_scale
    return fields


ROW_ORDER = ["GT 5 kHz", "Linear\ninterp.", "Cubic\nspline", "WaveFormer\n(ours)"]


def render(stem: str, gt_name: str, fields: Dict[str, np.ndarray],
           roi: Dict, times_ms: List[float], out_path: Path,
           vmax: float, title: str) -> None:
    H, W = fields["GT 5 kHz"].shape[1:]
    push = find_push_column(depth_average(fields["GT 5 kHz"], (0.25, 0.75)))

    x0 = -push * LATERAL_MM_PX
    x1 = (W - 1 - push) * LATERAL_MM_PX
    z0 = roi["min_row"] * AXIAL_MM_PX
    z1 = (roi["min_row"] + H - 1) * AXIAL_MM_PX

    # Every row is shown on ONE shared colour scale.  Rows are first put on a
    # common amplitude footing by their own 99th-percentile particle velocity,
    # because the 5 kHz ground truth is a separate acquisition with a different
    # push amplitude; the shape of the wavefront, not the push strength, is what
    # the panel compares.
    norm: Dict[str, np.ndarray] = {}
    for k, v in fields.items():
        s = max(float(np.percentile(np.abs(v), 99)), 1e-9)
        norm[k] = v / s

    rows = [r for r in ROW_ORDER if r in norm]
    n_rows, n_cols = len(rows), len(times_ms)

    panel_w_mm, panel_h_mm = (x1 - x0), (z1 - z0)
    panel_w_in = 1.55
    panel_h_in = panel_w_in * panel_h_mm / panel_w_mm      # true 1:1 mm aspect

    fig_w = panel_w_in * n_cols + 1.90
    fig_h = panel_h_in * n_rows + 1.15
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.05],
        hspace=0.10, wspace=0.06,
        left=(1.30 / fig_w), right=1 - (0.60 / fig_w),
        top=1 - (0.62 / fig_h), bottom=(0.53 / fig_h),
    )

    im_ref = None
    axes: List[List[plt.Axes]] = []
    for r, key in enumerate(rows):
        axes.append([])
        for c, t_ms in enumerate(times_ms):
            ax = fig.add_subplot(gs[r, c])
            axes[r].append(ax)
            dt = DT_5KHZ if key == "GT 5 kHz" else DT_OUT
            idx = int(min(round(t_ms * 1e-3 / dt), N_DENSE - 1))
            im = ax.imshow(norm[key][idx], cmap="seismic",
                           vmin=-vmax, vmax=vmax, aspect="equal",
                           extent=[x0, x1, z1, z0], interpolation="bilinear")
            im_ref = im_ref or im
            ax.set_xlim(x0, x1)
            ax.set_ylim(z1, z0)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                if key.startswith("WaveFormer"):
                    sp.set_edgecolor("#1a7a32")
                    sp.set_linewidth(1.4)
            if r == 0:
                ax.set_title(f"t = {t_ms:g} ms", fontsize=9, fontweight="semibold",
                             pad=4)
            if c == 0:
                ax.set_yticks([np.ceil(z0), np.floor(z1)])
                ax.tick_params(axis="y", labelsize=7, length=2, pad=1)
            else:
                ax.set_yticks([])
            if r == n_rows - 1:
                ax.set_xticks([-5, 0, 5])
                ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
            else:
                ax.set_xticks([])

    # Row labels in figure coordinates, at each row's own vertical midpoint —
    # set_ylabel stacks them all at the same height once the panels are short.
    for r, key in enumerate(rows):
        bb = axes[r][0].get_position()
        fig.text(bb.x0 - 0.012, bb.y0 + bb.height / 2, key,
                 fontsize=9, fontweight="bold", rotation=0,
                 ha="right", va="center", linespacing=1.25,
                 color="#1a7a32" if key.startswith("WaveFormer") else "#111111")

    cax = fig.add_subplot(gs[:, n_cols])
    cb = fig.colorbar(im_ref, cax=cax)
    cb.set_label("Normalized particle velocity (a.u.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])

    fig.text(0.5, 0.012, "Lateral distance from push (mm)", ha="center", fontsize=9)
    fig.text(0.010, 0.5, "Axial depth (mm)", va="center", rotation=90, fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}   ({fig_w:.1f} x {fig_h:.1f} in)")
    print(f"  lateral {x0:.2f} .. {x1:.2f} mm   axial {z0:.2f} .. {z1:.2f} mm   "
          f"1:1 aspect, shared colour scale +-{vmax}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/wavepinn_v14/best_model.pt")
    ap.add_argument("--eval-dir", default="outputs/eval_test_split_merged")
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--input-step", type=int, default=1,
                    help="1 = consecutive 500 Hz frames (matches v14 training); 12 = legacy")
    ap.add_argument("--group", default="1ft",
                    help="phantom prefix to scan: 1ft, 2ft, 3ft, cirs")
    ap.add_argument("--sample", default="")
    ap.add_argument("--times", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/saus2026/fig_a_1ft_propagation.png")
    ap.add_argument("--title",
                    default="Shear wave propagation, PVA phantom (held-out test set)")
    args = ap.parse_args()

    global INPUT_STEP, CACHE
    INPUT_STEP = args.input_step
    CACHE = Path(f"{CACHE}_step{INPUT_STEP}")
    data_dir = Path(args.data_dir)
    CACHE.mkdir(parents=True, exist_ok=True)

    if args.scan:
        model = load_model(args.model)
        print(f"\nscanning paired {args.group.upper()} test windows\n")
        for p in pair_group(Path(args.eval_dir), args.group):
            for g in p["gts"]:
                fields = build_fields(p["stem"], p["sparse"], g, model, data_dir)
                if fields is None:
                    print(f"  {p['stem'][:46]:48s} [skip] raw IQ unavailable")
                    continue
                tag = f"{p['stem']}__{g.stem}"
                np.savez_compressed(CACHE / f"{tag}.npz",
                                    **{k.replace("\n", " "): v
                                       for k, v in fields.items()})
                s_gt = wave_score(fields["GT 5 kHz"])
                s_wf = wave_score(fields["WaveFormer\n(ours)"])
                print(f"  {p['stem'][:46]:48s} GT[trav {s_gt['travel_mm']:5.1f}mm "
                      f"spread {s_gt['spread']:4.0f}]  "
                      f"WF[trav {s_wf['travel_mm']:5.1f}mm spread {s_wf['spread']:4.0f}] "
                      f"score {s_wf['score']:6.1f}")
        print(f"\ncached to {CACHE}")
        return 0

    if not args.sample:
        ap.error("give --scan or --sample <stem>")

    hits = sorted(CACHE.glob(f"{args.sample}*.npz"))
    if not hits:
        ap.error(f"no cached fields matching {args.sample} in {CACHE}")
    z = np.load(hits[0])
    key_map = {k.replace("\n", " "): k for k in ROW_ORDER}
    # caches written before the row labels were shortened
    key_map.update({"Linear interpolation": "Linear\ninterp.",
                    "Cubic-spline interpolation": "Cubic\nspline"})
    fields = {key_map[k]: z[k] for k in z.files if k in key_map}
    stem, gt_name = hits[0].stem.split("__", 1)

    times = [float(t) for t in args.times.split(",")]
    render(stem, gt_name, fields, roi_of(stem, data_dir), times,
           Path(args.out), args.vmax,
           args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
