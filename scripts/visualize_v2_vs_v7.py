#!/usr/bin/env python3
"""
V2 vs V7 WavePINN — Side-by-Side Comparison on 500 Hz SWE data
===============================================================
5-panel video per sample: GT | V2 | V7 | V2 error | V7 error
Plus SWE-specific metrics: ToA-based Cs estimation, active-frame Pearson r.

Usage
-----
  python scripts/visualize_v2_vs_v7.py \
      --v2_model  saved_models/fullscale_velocity_v2_best.pt \
      --v7_model  "SWE_PINN_outputs_wavepinn_v7_best_model (1).pt" \
      --data_dir  data/fullscale_velocity/inference \
      --output_dir outputs/v2_vs_v7 \
      --max_samples 10
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from architectures.velocity_unet import ComplexToRealUNet
from architectures.wavepinn_v7 import create_wavepinn_v7

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AXIAL_MM_PER_PX   = 0.0507
LATERAL_MM_PER_PX = 0.1013
LATERAL_M_PER_PX  = LATERAL_MM_PER_PX * 1e-3
H_ROI             = 116
DETECT_PRF_HZ     = 500.0
DT_S              = 1.0 / DETECT_PRF_HZ    # 0.002 s per dense frame

# SWE estimation params (mirror swe_analysis.py)
DEPTH_FRAC_LO     = 0.2
DEPTH_FRAC_HI     = 0.8
TOA_STEP          = 5       # lateral stride for ToA pairs
PUSH_EXCL_PX      = 15     # columns each side of push to exclude
CS_LO, CS_HI      = 0.2, 15.0  # m/s plausible bounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad16(arr: np.ndarray) -> np.ndarray:
    *lead, H, W = arr.shape
    pH = (16 - H % 16) % 16
    pW = (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return arr
    return np.pad(arr, [(0,0)]*len(lead) + [(0,pH),(0,pW)], mode='constant')


def _ffmpeg_ok() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def make_axes(H: int, W: int):
    return (
        (np.arange(W) - W / 2.0) * LATERAL_MM_PER_PX,   # lateral_ax
        np.arange(H) * AXIAL_MM_PER_PX,                  # depth_ax
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_v2(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = ComplexToRealUNet(
        in_channels              = cfg.get("in_channels", 1),
        base_channels            = cfg.get("base_channels", 32),
        encoder_depths           = list(cfg.get("encoder_depths", [32,64,128,256])),
        upsampling_factor        = cfg.get("upsampling_factor", 12),
        use_motion_interpolation = cfg.get("use_motion_interpolation", True),
        complex_to_real_mode     = cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn             = cfg.get("use_naive_bn", True),
        flow_hidden_channels     = cfg.get("flow_hidden_channels", 32),
        flow_scale               = cfg.get("flow_scale", 0.1),
        use_prf_conditioning     = cfg.get("use_prf_conditioning", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ep  = ckpt.get("epoch", "?")
    vl  = ckpt.get("best_val_loss", float("nan"))
    print(f"  V2 loaded  epoch={ep}  best_val_loss={vl:.5f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


def load_v7(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    mcfg = dict(cfg.get("model", {}))
    mcfg.pop("name", None)
    # Defaults if config absent
    mcfg.setdefault("in_channels", 1)
    mcfg.setdefault("encoder_depths", [32,64,128,256])
    mcfg.setdefault("upsampling_factor", 12)
    mcfg.setdefault("cs_min", 0.3)
    mcfg.setdefault("cs_max", 8.0)
    model = create_wavepinn_v7(config=mcfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  V7 missing keys: {missing}")
    if unexpected:
        print(f"  V7 unexpected keys: {unexpected}")
    model.eval()
    ep  = ckpt.get("epoch", "?")
    vl  = ckpt.get("best_val_loss", float("nan"))
    print(f"  V7 loaded  epoch={ep}  best_val_loss={vl:.5f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_v2(model, iq_path: Path, vel_path: Path, device) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (gt_mm, pred_mm, vel_scale)."""
    iq   = np.load(iq_path).astype(np.complex64)
    vel  = np.load(vel_path).astype(np.float32)
    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    iq_n = iq / iq_s
    v_s  = max(float(np.percentile(np.abs(vel), 99)), 1.0)
    v_n  = np.clip(vel / v_s, -1, 1)
    inp  = _pad16(np.stack([iq_n.real, iq_n.imag], axis=0))
    v_n  = _pad16(v_n)
    t    = torch.from_numpy(inp).unsqueeze(0).to(device)
    out  = model(t)
    pred = out["velocity"] if isinstance(out, dict) else out
    gt_mm   = v_n[:, :H_ROI, :] * v_s
    pred_mm = pred[0, 0].cpu().numpy()[:, :H_ROI, :] * v_s
    return gt_mm, pred_mm, v_s


@torch.no_grad()
def infer_v7(model, iq_path: Path, vel_path: Path, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (gt_mm, pred_mm, cs_map, vel_scale)."""
    iq   = np.load(iq_path).astype(np.complex64)
    vel  = np.load(vel_path).astype(np.float32)
    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    iq_n = iq / iq_s
    v_s  = max(float(np.percentile(np.abs(vel), 99)), 1.0)
    v_n  = np.clip(vel / v_s, -1, 1)
    inp  = _pad16(np.stack([iq_n.real, iq_n.imag], axis=0))
    v_n  = _pad16(v_n)
    t    = torch.from_numpy(inp).unsqueeze(0).to(device)
    dt   = torch.tensor([DT_S], dtype=torch.float32, device=device)
    vel_out, cs_map = model(t, dt_seconds=dt)
    gt_mm   = v_n[:, :H_ROI, :] * v_s
    pred_mm = vel_out[0, 0].cpu().numpy()[:, :H_ROI, :] * v_s
    cs_np   = cs_map[0, 0].cpu().numpy()[:H_ROI, :]       # (H, W)
    return gt_mm, pred_mm, cs_np, v_s


# ---------------------------------------------------------------------------
# SWE: ToA-based Cs estimation
# ---------------------------------------------------------------------------

def estimate_cs_toa(vel_mm: np.ndarray, dt_s: float = DT_S,
                    push_col: int = None) -> Optional[float]:
    """
    Simplified ToA pipeline (matches swe_analysis.py logic):
    1. Depth-average (20-80% depth)
    2. Find peak-abs-value time per column (ToA)
    3. Linear fit ToA vs column → slope = dt/pixel → cs = dx/slope
    Returns cs in m/s, or None if insufficient data.
    """
    T, H, W = vel_mm.shape
    r0, r1  = int(DEPTH_FRAC_LO * H), int(DEPTH_FRAC_HI * H)
    v_avg   = vel_mm[:, r0:r1, :].mean(axis=1)      # (T, W)

    # Peak amplitude per column (filter low-energy cols)
    peak_amp = np.abs(v_avg).max(axis=0)             # (W,)
    amp_thresh = peak_amp.mean() * 0.3
    valid_cols = np.where(peak_amp > amp_thresh)[0]

    # Exclude push zone
    if push_col is not None:
        valid_cols = valid_cols[np.abs(valid_cols - push_col) > PUSH_EXCL_PX]

    if len(valid_cols) < 10:
        return None

    # ToA = time of peak magnitude per column
    toa_idx = np.argmax(np.abs(v_avg[:, valid_cols]), axis=0).astype(float)  # (n_valid,)
    toa_s   = toa_idx * dt_s  # (n_valid,)
    col_m   = valid_cols.astype(float) * LATERAL_M_PER_PX

    # Linear fit: col_m = cs * toa_s  → slope = col/toa = cs
    # Only use stride-5 pairs to match DiffToaCsLoss
    cs_vals = []
    for i in range(0, len(valid_cols) - TOA_STEP, TOA_STEP):
        dt_toa = toa_s[i + TOA_STEP] - toa_s[i]
        dx     = col_m[i + TOA_STEP] - col_m[i]
        if abs(dt_toa) < 1e-9:
            continue
        cs = abs(dx / dt_toa)
        if CS_LO < cs < CS_HI:
            cs_vals.append(cs)

    return float(np.median(cs_vals)) if cs_vals else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray,
                    vel_scale: float = 1.0) -> Dict[str, float]:
    """Full + active-frame metrics."""
    diff        = pred - gt
    mae_full    = float(np.abs(diff).mean())
    rmse_full   = float(np.sqrt((diff**2).mean()))
    g, p        = gt.flatten(), pred.flatten()
    r_full      = float(np.corrcoef(g, p)[0,1]) if g.std() > 1e-8 else float("nan")

    frame_stds  = gt.std(axis=(1,2))
    act         = frame_stds > 0.05 * vel_scale / 100.0  # ~0.05 normalised unit
    # fallback: at least 1st 5 frames if nothing active
    if not act.any():
        act = np.zeros(len(frame_stds), bool)
        act[:min(5, len(act))] = True

    gt_a, pr_a  = gt[act], pred[act]
    mae_active  = float(np.abs(pr_a - gt_a).mean())
    rmse_active = float(np.sqrt(((pr_a - gt_a)**2).mean()))
    r_active    = float(np.corrcoef(gt_a.flatten(), pr_a.flatten())[0,1]) \
                  if gt_a.flatten().std() > 1e-8 else float("nan")

    cs_toa      = estimate_cs_toa(gt if vel_scale == 1.0 else gt)   # pass real mm/s

    return dict(mae_full=mae_full, rmse_full=rmse_full, r_full=r_full,
                mae_active=mae_active, rmse_active=rmse_active, r_active=r_active,
                n_active=int(act.sum()), cs_toa_gt=cs_toa if cs_toa else float("nan"))


# ---------------------------------------------------------------------------
# Video writer — 5 panels
# ---------------------------------------------------------------------------

def _write_5panel(
    gt: np.ndarray, pred_v2: np.ndarray, pred_v7: np.ndarray,
    lat_ax, dep_ax, t_ax_ms, clim: float,
    title_base: str, out_path: Path, fps: int = 6,
) -> Path:
    T       = len(t_ax_ms)
    extent  = [lat_ax[0], lat_ax[-1], dep_ax[-1], dep_ax[0]]

    fig = plt.figure(figsize=(26, 5.2), facecolor="black")
    gs  = gridspec.GridSpec(
        1, 6, figure=fig,
        width_ratios=[1,1,1,1,1,0.04],
        wspace=0.07, left=0.04, right=0.97, top=0.85, bottom=0.13
    )

    axes  = [fig.add_subplot(gs[i]) for i in range(5)]
    ax_cb = fig.add_subplot(gs[5])

    for ax in axes:
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_edgecolor("gray")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    kw = dict(cmap="jet", vmin=-clim, vmax=clim, aspect="auto",
              origin="upper", extent=extent)
    kw_err = dict(cmap="hot", vmin=0, vmax=clim*0.6, aspect="auto",
                  origin="upper", extent=extent)

    ims = [
        axes[0].imshow(gt[0], **kw),
        axes[1].imshow(pred_v2[0], **kw),
        axes[2].imshow(pred_v7[0], **kw),
        axes[3].imshow(np.abs(gt[0]-pred_v2[0]), **kw_err),
        axes[4].imshow(np.abs(gt[0]-pred_v7[0]), **kw_err),
    ]

    titles = ["Ground Truth", "V2 Prediction", "V7 Prediction",
              "|Error| V2", "|Error| V7"]
    for ax, tt in zip(axes, titles):
        ax.set_title(tt, fontsize=10, fontweight="bold")
        ax.set_xlabel("Lateral (mm)", fontsize=8)
        ax.set_ylabel("Depth (mm)", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("mm/s", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    sup = fig.suptitle("", color="white", fontsize=10, fontweight="bold")

    def _upd(fi):
        sup.set_text(f"{title_base}  |  t={t_ax_ms[fi]:.2f}ms  (frame {fi+1}/{T})")
        ims[0].set_data(gt[fi])
        ims[1].set_data(pred_v2[fi])
        ims[2].set_data(pred_v7[fi])
        ims[3].set_data(np.abs(gt[fi] - pred_v2[fi]))
        ims[4].set_data(np.abs(gt[fi] - pred_v7[fi]))
        return ims + [sup]

    anim = FuncAnimation(fig, _upd, frames=T, blit=True, interval=1000//fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _ffmpeg_ok():
        wr   = FFMpegWriter(fps=fps, extra_args=["-pix_fmt","yuv420p"])
        save = out_path.with_suffix(".mp4")
        anim.save(str(save), writer=wr, dpi=110,
                  savefig_kwargs={"facecolor":"black"})
    else:
        wr   = PillowWriter(fps=fps)
        save = out_path.with_suffix(".gif")
        anim.save(str(save), writer=wr,
                  savefig_kwargs={"facecolor":"black"})

    plt.close(fig)
    return save


# ---------------------------------------------------------------------------
# Summary plots & CSV
# ---------------------------------------------------------------------------

def save_comparison_summary(
    names: List[str],
    metrics_v2: List[Dict], metrics_v7: List[Dict],
    out_dir: Path,
):
    """Bar-chart comparisons: r_active, mae_active, cs_toa."""
    if not metrics_v2:
        return

    keys   = ["r_active", "mae_active", "rmse_active", "cs_toa_gt"]
    labels = [n[-30:] for n in names]
    N      = len(names)

    fig, axes = plt.subplots(len(keys), 1,
                             figsize=(max(14, N*0.55), 4*len(keys)))
    fig.suptitle("V2 vs V7 — SWE Comparison", fontsize=14, fontweight="bold")

    ylabels = {
        "r_active":   "Pearson r (active frames)",
        "mae_active": "MAE active (mm/s)",
        "rmse_active":"RMSE active (mm/s)",
        "cs_toa_gt":  "GT Cs estimate (m/s)",
    }

    x = np.arange(N)
    w = 0.35

    for ax, key in zip(axes, keys):
        v2v = [m.get(key, float("nan")) for m in metrics_v2]
        v7v = [m.get(key, float("nan")) for m in metrics_v7]

        ax.bar(x - w/2, v2v, w, label="V2", color="steelblue", alpha=0.85)
        ax.bar(x + w/2, v7v, w, label="V7", color="tomato",    alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel(ylabels.get(key, key), fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

        # Add mean lines
        v2m = np.nanmean(v2v) if any(not np.isnan(v) for v in v2v) else None
        v7m = np.nanmean(v7v) if any(not np.isnan(v) for v in v7v) else None
        if v2m is not None:
            ax.axhline(v2m, color="steelblue", ls="--", lw=1.5,
                       label=f"V2 mean={v2m:.3f}")
        if v7m is not None:
            ax.axhline(v7m, color="tomato", ls="--", lw=1.5,
                       label=f"V7 mean={v7m:.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "comparison_summary.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot → {out_png}")

    # CSV
    out_csv = out_dir / "comparison_metrics.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name",
                    "v2_r_active","v7_r_active","delta_r",
                    "v2_mae_active","v7_mae_active",
                    "v2_rmse_active","v7_rmse_active",
                    "v2_r_full","v7_r_full",
                    "cs_toa_gt"])
        for name, m2, m7 in zip(names, metrics_v2, metrics_v7):
            r2  = m2.get("r_active", float("nan"))
            r7  = m7.get("r_active", float("nan"))
            dr  = r7 - r2 if not (np.isnan(r2) or np.isnan(r7)) else float("nan")
            w.writerow([
                name,
                f"{r2:.4f}", f"{r7:.4f}", f"{dr:.4f}",
                f"{m2.get('mae_active', float('nan')):.4f}",
                f"{m7.get('mae_active', float('nan')):.4f}",
                f"{m2.get('rmse_active', float('nan')):.4f}",
                f"{m7.get('rmse_active', float('nan')):.4f}",
                f"{m2.get('r_full', float('nan')):.4f}",
                f"{m7.get('r_full', float('nan')):.4f}",
                f"{m2.get('cs_toa_gt', float('nan')):.4f}",
            ])
    print(f"  Metrics CSV   → {out_csv}")

    # Print aggregate summary
    print("\n" + "="*60)
    print("  AGGREGATE COMPARISON (mean ± std)")
    print("="*60)
    for key, lbl in ylabels.items():
        v2v = [m.get(key, float("nan")) for m in metrics_v2]
        v7v = [m.get(key, float("nan")) for m in metrics_v7]
        v2v = [v for v in v2v if not np.isnan(v)]
        v7v = [v for v in v7v if not np.isnan(v)]
        if v2v and v7v:
            print(f"  {lbl:<35} V2={np.mean(v2v):.4f}±{np.std(v2v):.4f}  "
                  f"V7={np.mean(v7v):.4f}±{np.std(v7v):.4f}  "
                  f"Δ={np.mean(v7v)-np.mean(v2v):+.4f}")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="V2 vs V7 comparison on 500Hz SWE data")
    ap.add_argument("--v2_model",   default="saved_models/fullscale_velocity_v2_best.pt")
    ap.add_argument("--v7_model",   default="SWE_PINN_outputs_wavepinn_v7_best_model (1).pt")
    ap.add_argument("--data_dir",   default="data/fullscale_velocity/inference")
    ap.add_argument("--output_dir", default="outputs/v2_vs_v7")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--fps_full",   type=int, default=6)
    ap.add_argument("--fps_zoom",   type=int, default=3)
    ap.add_argument("--n_zoom",     type=int, default=12)
    ap.add_argument("--no_video",   action="store_true", help="Skip video, metrics only")
    ap.add_argument("--clim",       type=float, default=None)
    args = ap.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / args.data_dir
    out_dir  = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_path = PROJECT_ROOT / args.v2_model
    v7_path = PROJECT_ROOT / args.v7_model

    print(f"\n{'='*65}")
    print("  V2 vs V7 Comparison — 500 Hz SWE")
    print(f"  V2 : {v2_path.name}")
    print(f"  V7 : {v7_path.name}")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # Load models
    print("Loading V2 model...")
    v2 = load_v2(str(v2_path), device)
    print("Loading V7 model...")
    v7 = load_v7(str(v7_path), device)
    print()

    # Discover 500Hz _w0 samples
    sp_files = sorted(data_dir.glob("sparse_*500Hz*_w0.npy"))
    if not sp_files:
        sp_files = sorted(data_dir.glob("sparse_*500Hz*.npy"))
    if not sp_files:
        raise FileNotFoundError(f"No sparse_*500Hz*.npy in {data_dir}")
    if args.max_samples:
        sp_files = sp_files[:args.max_samples]
    print(f"  Found {len(sp_files)} 500Hz _w0 samples\n")

    all_names, metrics_v2_list, metrics_v7_list = [], [], []

    for sp_path in tqdm(sp_files, desc="Samples"):
        name     = sp_path.stem[len("sparse_"):]
        vel_path = sp_path.parent / f"velocity_{name}.npy"
        if not vel_path.exists():
            print(f"  SKIP (no velocity): {name}")
            continue

        try:
            gt_mm,  p2_mm, vel_s   = infer_v2(v2, sp_path, vel_path, device)
            gt_mm2, p7_mm, cs_np, _= infer_v7(v7, sp_path, vel_path, device)
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            continue

        # Align T dims (V7 may produce same or +1 frames vs V2)
        T = min(gt_mm.shape[0], p2_mm.shape[0], p7_mm.shape[0])
        gt_mm  = gt_mm[:T]
        p2_mm  = p2_mm[:T]
        p7_mm  = p7_mm[:T]

        m2 = compute_metrics(gt_mm, p2_mm, vel_scale=vel_s)
        m7 = compute_metrics(gt_mm, p7_mm, vel_scale=vel_s)

        # Attach cs_map stats to V7 metrics
        m7["cs_map_mean"] = float(cs_np.mean())
        m7["cs_map_std"]  = float(cs_np.std())

        all_names.append(name)
        metrics_v2_list.append(m2)
        metrics_v7_list.append(m7)

        tqdm.write(
            f"  {name[-38:]:38s}  "
            f"r_act V2={m2['r_active']:.3f} V7={m7['r_active']:.3f}  "
            f"MAE V2={m2['mae_active']:.2f} V7={m7['mae_active']:.2f} mm/s  "
            f"Cs_map={m7['cs_map_mean']:.2f}m/s"
        )

        if args.no_video:
            continue

        # Physical axes
        _, H, W = gt_mm.shape
        lat_ax, dep_ax = make_axes(H_ROI, W)
        t_ax_ms = np.arange(T) * DT_S * 1e3

        clim = args.clim
        if clim is None:
            # use GT active frames for colour range
            frame_stds = gt_mm.std(axis=(1,2))
            act = frame_stds > 0.05 * vel_s / 100
            src = gt_mm[act] if act.any() else gt_mm
            clim = max(float(np.abs(src).max()), 0.2)

        # --- Full video ---
        _write_5panel(
            gt_mm, p2_mm, p7_mm,
            lat_ax, dep_ax, t_ax_ms, clim,
            f"V2vsV7 | 500Hz | {name[-26:]}  [full {T}f]",
            out_dir / f"{name}_full", fps=args.fps_full,
        )

        # --- Zoom video ---
        nz = min(args.n_zoom, T)
        _write_5panel(
            gt_mm[:nz], p2_mm[:nz], p7_mm[:nz],
            lat_ax, dep_ax, t_ax_ms[:nz], clim,
            f"V2vsV7 | 500Hz | {name[-26:]}  [zoom {nz}f]",
            out_dir / f"{name}_zoom", fps=args.fps_zoom,
        )

    # Summary
    if all_names:
        save_comparison_summary(all_names, metrics_v2_list, metrics_v7_list, out_dir)
    else:
        print("No samples processed.")


if __name__ == "__main__":
    main()
