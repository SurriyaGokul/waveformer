#!/usr/bin/env python3
"""
True 500 Hz deployment: FIVE CONSECUTIVE 500 Hz frames -> 49 frames at 6 kHz.

Every other driver in this repo builds the model's input with
``sparse_step = 12`` counted in NATIVE frames.  On a 500 Hz acquisition that
puts the five input frames at 0, 24, 48, 72 and 96 ms — but the shear wave is
over within ~3 ms, so only the first frame carries it and the model has nothing
to interpolate between.

This script rebuilds the input from the RAW 500 Hz IQ at ``step = 1``:

    frames 0,1,2,3,4  ->  dt_in = 2 ms  (a real 500 Hz probe)
    dt_out = 2 ms / 12 = 1/6000 s       (6 kHz reconstruction)

replicating ``scripts/preprocess_1ft.py:process_iq_window`` exactly — same ROI
crop, same clutter filter (subtract the temporal mean over the FULL 50-frame
window, not over the 5 input frames), same 99th-percentile IQ normalisation.

Ground truth is the native 5 kHz acquisition of the same phantom sample and
imaging plane.  It cannot be the same *acquisition*: 500 Hz and 5 kHz are
separate captures, so no simultaneous dual-PRF recording exists.  The two are
usually minutes apart; the pairing gap is printed for every sample.

Outputs a 4-row figure (GT 5 kHz | linear | cubic | WaveFormer) and, with
--video, an mp4 in the style of outputs/comparison_videos.

Usage:
    python scripts/predict_from_consecutive_500hz.py --group 1ft --video
    python scripts/predict_from_consecutive_500hz.py --sample <stem>
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dispersion import (  # noqa: E402
    depth_average, estimate_cs_toa, estimate_dispersion, find_push_column,
)
from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402

H_ROI, W_ROI = 116, 160
UPSAMPLE = 12
N_SPARSE = 5
N_DENSE = 1 + (N_SPARSE - 1) * UPSAMPLE          # 49
DT_500HZ = 1.0 / 500.0                            # native 500 Hz frame period
DT_IN_CONSEC = DT_500HZ                           # step 1 -> 2 ms
DT_OUT = DT_IN_CONSEC / UPSAMPLE                  # 1/6000 s
DT_IN_STEP12 = DT_500HZ * UPSAMPLE                # 24 ms, the old path
DT_5KHZ = 1.0 / 5000.0

AXIAL_MM_PX = 0.0507
LATERAL_MM_PX = 0.1013

ROW_ORDER = ["GT 5 kHz", "Linear\ninterp.", "Cubic\nspline", "WaveFormer\n(ours)"]


def pad16(a: np.ndarray) -> np.ndarray:
    *lead, H, W = a.shape
    pH, pW = (16 - H % 16) % 16, (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return a
    return np.pad(a, [(0, 0)] * len(lead) + [(0, pH), (0, pW)])


def group_of(name: str) -> str:
    low = name.lower()
    if "cirs" in low:
        return "cirs"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}ft" if m else "other"


def load_model(path: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).eval()
    m.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  checkpoint epoch={ckpt.get('epoch','?')} "
          f"val={ckpt.get('best_val_loss', float('nan')):.5f}")
    return m


RAW_ROOT = Path("500hz_raw_transfer")
_RAW_INDEX: Optional[Dict[str, List[Path]]] = None


def resolve_raw(entry: Dict) -> Optional[Path]:
    """Locate the raw acquisition.  Some metadata entries carry an
    ``original_path`` from the machine the preprocessing was first run on, so
    fall back to indexing the local raw tree by timestamp."""
    global _RAW_INDEX
    p = Path(entry.get("original_path", ""))
    if p.exists():
        return p
    if _RAW_INDEX is None:
        _RAW_INDEX = collections.defaultdict(list)
        for f in RAW_ROOT.rglob("*.npy"):
            _RAW_INDEX[f.stem].append(f)
    cands = _RAW_INDEX.get(p.stem or "", [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Disambiguate by the tail of the recorded path (focus / plane folders).
    want = [x.lower() for x in p.parts[-4:-1]]
    for c in cands:
        low = str(c).lower()
        if all(w in low for w in want):
            return c
    return cands[0]


def build_input_iq(entry: Dict, step: int) -> Optional[np.ndarray]:
    """Rebuild the model's 5-frame IQ input from the raw acquisition, following
    preprocess_1ft.py:process_iq_window exactly but with a configurable step."""
    raw = resolve_raw(entry)
    if raw is None:
        return None
    iq = np.load(raw, mmap_mode="r")
    w0, w1 = entry["window"]
    r = entry["roi"]
    iq_win = np.asarray(iq[r["min_row"]:r["max_row"] + 1,
                           r["min_col"]:r["max_col"] + 1, w0:w1])
    # Clutter filter over the FULL window, as in preprocessing.
    iq_filt = iq_win - iq_win.mean(axis=2, keepdims=True)
    idx = np.arange(N_SPARSE) * step
    if idx[-1] >= iq_filt.shape[2]:
        return None
    sparse = iq_filt[:H_ROI, :W_ROI, idx]
    return np.transpose(sparse, (2, 0, 1)).astype(np.complex64)


@torch.no_grad()
def infer(model, iq: np.ndarray, dt_in: float, vel_scale: float) -> np.ndarray:
    s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    inp = pad16(np.stack([(iq / s).real, (iq / s).imag], axis=0))
    out = model(torch.from_numpy(inp).unsqueeze(0),
                dt_seconds=torch.tensor([dt_in]))
    return out[0, 0].float().numpy()[:, :H_ROI, :W_ROI] * vel_scale


def interp_baseline(vel: np.ndarray, kind: str) -> np.ndarray:
    """Interpolate the five CONSECUTIVE 500 Hz velocity frames onto the dense
    grid — the information a 500 Hz probe actually hands a naive method."""
    knots = np.arange(N_SPARSE) * UPSAMPLE
    f = interp1d(knots.astype(float), vel[:N_SPARSE], axis=0, kind=kind,
                 bounds_error=False, fill_value="extrapolate")
    return f(np.arange(N_DENSE, dtype=float))


def parse_ts(s: str) -> Optional[dt.datetime]:
    m = re.search(r"(\d{8}T\d{6})", s)
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m else None


def find_gt(plane: str, eval_dir: Path) -> List[Path]:
    gt_re = re.compile(r"velocity_(.+?)_5kHz_(\d{8}T\d+)(?:_w(\d+))?\.npy$")
    out = []
    for f in sorted(eval_dir.glob("velocity_*5kHz*.npy")):
        m = gt_re.match(f.name)
        if m and m.group(1) == plane:
            out.append(f)
    return out


def build_fields(entry: Dict, gt_path: Path, model, data_dir: Path) -> Optional[Dict]:
    gt = np.load(gt_path).astype(np.float64)[:N_DENSE, :H_ROI, :W_ROI]
    vel_scale = float(max(np.percentile(np.abs(gt), 99), 1.0))

    iq1 = build_input_iq(entry, step=1)
    if iq1 is None:
        return None
    iq12 = build_input_iq(entry, step=UPSAMPLE)

    fields = {"GT 5 kHz": gt}

    v500 = None
    for split in ("test", "inference", "val", "train"):
        p = data_dir / split / f"velocity_{entry['name']}.npy"
        if p.exists():
            v500 = np.load(p).astype(np.float64)[:, :H_ROI, :W_ROI]
            break
    if v500 is not None:
        fields["Linear\ninterp."] = interp_baseline(v500, "linear")
        fields["Cubic\nspline"] = interp_baseline(v500, "cubic")

    fields["WaveFormer\n(ours)"] = infer(model, iq1, DT_IN_CONSEC, vel_scale)
    extra = {}
    if iq12 is not None:
        extra["step12"] = infer(model, iq12, DT_IN_STEP12 / UPSAMPLE, vel_scale)
    return dict(fields=fields, extra=extra, vel_scale=vel_scale)


def measure(field: np.ndarray, dt_s: float) -> Dict:
    toa = estimate_cs_toa(field, dt_s)
    disp = estimate_dispersion(field, dt_s, cs_toa_hint=toa["cs"])
    return dict(cs=float(toa["cs"]), eta=float(disp.eta),
                G=float(disp.G_prime_kpa), ok=bool(disp.ok))


# ── static figure ───────────────────────────────────────────────────────────
def render_figure(fields: Dict[str, np.ndarray], roi: Dict, times_ms: List[float],
                  out_path: Path, vmax: float, title: str) -> None:
    H, W = fields["GT 5 kHz"].shape[1:]
    push = find_push_column(depth_average(fields["GT 5 kHz"], (0.25, 0.75)))
    x0, x1 = -push * LATERAL_MM_PX, (W - 1 - push) * LATERAL_MM_PX
    z0 = roi["min_row"] * AXIAL_MM_PX
    z1 = (roi["min_row"] + H - 1) * AXIAL_MM_PX

    norm = {k: v / max(float(np.percentile(np.abs(v), 99)), 1e-9)
            for k, v in fields.items()}
    rows = [r for r in ROW_ORDER if r in norm]
    n_rows, n_cols = len(rows), len(times_ms)

    panel_w_in = 1.55
    panel_h_in = panel_w_in * (z1 - z0) / (x1 - x0)
    fig_w = panel_w_in * n_cols + 1.90
    fig_h = panel_h_in * n_rows + 1.15
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = gridspec.GridSpec(n_rows, n_cols + 1,
                           width_ratios=[1.0] * n_cols + [0.05],
                           hspace=0.10, wspace=0.06,
                           left=1.30 / fig_w, right=1 - 0.60 / fig_w,
                           top=1 - 0.62 / fig_h, bottom=0.53 / fig_h)

    im_ref, axes = None, []
    for r, key in enumerate(rows):
        axes.append([])
        for c, t_ms in enumerate(times_ms):
            ax = fig.add_subplot(gs[r, c])
            axes[r].append(ax)
            dtv = DT_5KHZ if key == "GT 5 kHz" else DT_OUT
            idx = int(min(round(t_ms * 1e-3 / dtv), N_DENSE - 1))
            im = ax.imshow(norm[key][idx], cmap="seismic", vmin=-vmax, vmax=vmax,
                           aspect="equal", extent=[x0, x1, z1, z0],
                           interpolation="bilinear")
            im_ref = im_ref or im
            ax.set_xlim(x0, x1); ax.set_ylim(z1, z0)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                if key.startswith("WaveFormer"):
                    sp.set_edgecolor("#1a7a32"); sp.set_linewidth(1.4)
            if r == 0:
                ax.set_title(f"t = {t_ms:g} ms", fontsize=9,
                             fontweight="semibold", pad=4)
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

    for r, key in enumerate(rows):
        bb = axes[r][0].get_position()
        fig.text(bb.x0 - 0.012, bb.y0 + bb.height / 2, key, fontsize=9,
                 fontweight="bold", ha="right", va="center", linespacing=1.25,
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
    print(f"    figure -> {out_path}")


# ── video ───────────────────────────────────────────────────────────────────
def render_video(fields: Dict[str, np.ndarray], roi: Dict, out_path: Path,
                 title: str, fps: int = 10) -> None:
    keys = [k for k in ROW_ORDER if k in fields]
    H, W = fields["GT 5 kHz"].shape[1:]
    push = find_push_column(depth_average(fields["GT 5 kHz"], (0.25, 0.75)))
    lat = (np.arange(W) - push) * LATERAL_MM_PX
    z0 = roi["min_row"] * AXIAL_MM_PX
    extent = [lat[0], lat[-1], z0 + (H - 1) * AXIAL_MM_PX, z0]
    focal_row = H // 3
    focal_mm = z0 + focal_row * AXIAL_MM_PX

    norm = {k: v / max(float(np.percentile(np.abs(v), 99)), 1e-9)
            for k, v in fields.items()}
    clim = 1.5
    colors = {"GT 5 kHz": "#1f6fb4", "Linear\ninterp.": "#e67e22",
              "Cubic\nspline": "#9b59b6", "WaveFormer\n(ours)": "#2ca02c"}

    fig = plt.figure(figsize=(15, 6.6), dpi=130)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, len(keys) + 1,
                           width_ratios=[1] * len(keys) + [0.035],
                           height_ratios=[1, 0.85], wspace=0.09, hspace=0.36,
                           left=0.05, right=0.95, top=0.86, bottom=0.09)
    axs = [fig.add_subplot(gs[0, i]) for i in range(len(keys))]
    cax = fig.add_subplot(gs[0, -1])
    axw = fig.add_subplot(gs[1, :])

    ims = []
    for ax, k in zip(axs, keys):
        im = ax.imshow(norm[k][0], cmap="seismic", vmin=-clim, vmax=clim,
                       aspect="equal", extent=extent, interpolation="bilinear")
        ims.append(im)
        ax.set_title(k.replace("\n", " "), fontsize=10, fontweight="bold",
                     color=colors[k])
        ax.set_xlabel("Lateral (mm)", fontsize=8)
        ax.axhline(focal_mm, color="lime", lw=0.8, ls="--", alpha=0.6)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("Depth (mm)", fontsize=8)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("Norm. particle velocity", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    lines = {}
    for k in keys:
        (ln,) = axw.plot(lat, norm[k][0, focal_row], lw=1.6, color=colors[k],
                         label=k.replace("\n", " "),
                         alpha=1.0 if k in ("GT 5 kHz", "WaveFormer\n(ours)") else 0.75)
        lines[k] = ln
    axw.set_xlim(lat[0], lat[-1])
    axw.set_ylim(-clim, clim)
    axw.set_xlabel("Lateral distance from push (mm)", fontsize=9)
    axw.set_ylabel(f"Norm. particle velocity\nat {focal_mm:.1f} mm depth", fontsize=9)
    axw.grid(alpha=0.25, lw=0.5)
    axw.legend(fontsize=8, ncol=len(keys), loc="upper right", framealpha=0.9)
    axw.axvline(0, color="0.4", lw=0.8, ls=":")

    sup = fig.suptitle("", fontsize=12, fontweight="bold", y=0.955)

    def update(f):
        t_ms = f * DT_OUT * 1e3
        gt_idx = int(min(round(t_ms * 1e-3 / DT_5KHZ), N_DENSE - 1))
        for k, im, ln in zip(keys, ims, [lines[k] for k in keys]):
            i = gt_idx if k == "GT 5 kHz" else f
            im.set_data(norm[k][i])
            ln.set_ydata(norm[k][i, focal_row])
        sup.set_text(f"{title}    t = {t_ms:5.2f} ms")
        return ims + list(lines.values()) + [sup]

    anim = FuncAnimation(fig, update, frames=N_DENSE, interval=1000 / fps,
                         blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    print(f"    video  -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/wavepinn_v14/best_model.pt")
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--eval-dir", default="outputs/eval_test_split_merged")
    ap.add_argument("--group", default="", help="1ft | 2ft | 3ft | cirs")
    ap.add_argument("--sample", default="")
    ap.add_argument("--times", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="outputs/saus2026/consecutive500")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    meta = json.loads((data_dir / "metadata.json").read_text())
    by_name = {e["name"]: e for e in meta["test"]}

    model = load_model(args.model)

    sp_re = re.compile(r"sparse_(.+?)_500Hz_(\d{8}T\d+)(?:_w(\d+))?\.npy$")
    cands = []
    for f in sorted(Path(args.eval_dir).glob("sparse_*500Hz*.npy")):
        m = sp_re.match(f.name)
        if not m:
            continue
        stem = f.name[len("sparse_"):-len(".npy")]
        if stem not in by_name or (m.group(3) or "0") != "0":
            continue
        if args.sample and stem != args.sample:
            continue
        if args.group and group_of(stem) != args.group:
            continue
        gts = find_gt(m.group(1), Path(args.eval_dir))
        if gts:
            cands.append((stem, by_name[stem], gts))

    if not cands:
        print("no candidates matched")
        return 1

    times = [float(t) for t in args.times.split(",")]
    summary = []

    for stem, entry, gts in cands:
        gt_path = gts[0]
        gap = None
        a, b = parse_ts(stem), parse_ts(gt_path.name)
        if a and b:
            gap = abs((b - a).total_seconds()) / 60.0
        print(f"\n  {stem}")
        print(f"    GT: {gt_path.name}   pairing gap {gap:.1f} min"
              if gap is not None else f"    GT: {gt_path.name}")

        built = build_fields(entry, gt_path, model, data_dir)
        if built is None:
            print("    [skip] raw IQ unavailable")
            continue
        fields, extra = built["fields"], built["extra"]

        m_gt = measure(fields["GT 5 kHz"], DT_5KHZ)
        m_wf = measure(fields["WaveFormer\n(ours)"], DT_OUT)
        m_12 = measure(extra["step12"], DT_OUT) if "step12" in extra else None
        print(f"    cs  GT {m_gt['cs']:.2f}   "
              f"WaveFormer[step 1, dt_in=2ms] {m_wf['cs']:.2f}"
              + (f"   [step 12, old] {m_12['cs']:.2f}" if m_12 else ""))
        print(f"    eta GT {m_gt['eta']:.2f}   "
              f"WaveFormer[step 1] {m_wf['eta']:.2f}"
              + (f"   [step 12] {m_12['eta']:.2f}" if m_12 else ""))
        summary.append(dict(sample=stem, group=group_of(stem),
                            gt_file=gt_path.name, gap_min=gap,
                            cs_gt=m_gt["cs"], eta_gt=m_gt["eta"], G_gt=m_gt["G"],
                            cs_step1=m_wf["cs"], eta_step1=m_wf["eta"], G_step1=m_wf["G"],
                            cs_step12=m_12["cs"] if m_12 else None,
                            eta_step12=m_12["eta"] if m_12 else None,
                            G_step12=m_12["G"] if m_12 else None))

        g = group_of(stem).upper()
        title = (f"Shear wave propagation, {g} phantom — "
                 f"5 consecutive 500 Hz frames -> 6 kHz")
        render_figure(fields, entry["roi"], times,
                      out_dir / f"{stem}_consecutive500.png", args.vmax, title)
        if args.video:
            render_video(fields, entry["roi"],
                         out_dir / f"{stem}_consecutive500.mp4", title)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nWrote {out_dir}/summary.json  ({len(summary)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
