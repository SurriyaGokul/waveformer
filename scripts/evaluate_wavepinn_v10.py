#!/usr/bin/env python3
"""
WavePINN V10 Deployment Evaluation
====================================
Runs the 500Hz → 6kHz deployment evaluation and compares V10 against
V9-Ep60 (current best) and V2.

Usage:
    python scripts/evaluate_wavepinn_v10.py
    python scripts/evaluate_wavepinn_v10.py --v10 path/to/v10_best.pt
    python scripts/evaluate_wavepinn_v10.py --no-video

Evaluation:
  - Input:  5 consecutive 500Hz IQ frames (dt_input=2ms)
  - Output: 49 virtual frames at dt_out≈0.167ms (6kHz equivalent)
  - GT SWE: derived from 5kHz native velocity (dt_gt=0.2ms)
  - Metrics: cs (m/s), G' (kPa), E (kPa), eta (Pa·s) per phantom

Ground truth = SWE analysis on 5kHz native velocity.
Targets to beat: V9-Ep60 cs_MAE=1.98, V2 cs_MAE=2.63 m/s.
"""
import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.architectures.velocity_unet import ComplexToRealUNet
from src.architectures.wavepinn_v9 import create_wavepinn_v9
from src.architectures.wavepinn_v10 import create_wavepinn_v10

# ── Physical constants ────────────────────────────────────────────────────────
DT_500HZ     = 1.0 / 500.0        # 2 ms between consecutive 500Hz frames
UPSAMPLE     = 12
DT_OUT       = DT_500HZ / UPSAMPLE  # ~0.1667 ms (6kHz virtual)
DT_5KHZ_GT   = 1.0 / 5000.0        # 0.2 ms (5kHz reference)
LATERAL_M_PX = 0.1013e-3
RHO          = 1000.0
H_ROI        = 116


def pad16(a: np.ndarray) -> np.ndarray:
    *lead, H, W = a.shape
    pH = (16 - H % 16) % 16
    pW = (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return a
    return np.pad(a, [(0, 0)] * len(lead) + [(0, pH), (0, pW)])


# ── SWE: Time-of-Arrival ──────────────────────────────────────────────────────
def estimate_cs_toa(
    velocity: np.ndarray,
    dt_s: float,
    step: int = 5,
    depth_frac: Tuple[float, float] = (0.2, 0.8),
    min_amp_pct: float = 15.0,
    push_excl_px: int = 15,
) -> Dict:
    T, H, W = velocity.shape
    r0, r1 = int(H * depth_frac[0]), int(H * depth_frac[1])
    v_abs = np.mean(np.abs(velocity[:, r0:r1, :]), axis=1)
    amp_thresh = np.percentile(v_abs.max(axis=0), min_amp_pct)
    amp_ok = v_abs.max(axis=0) > amp_thresh
    xp = int(np.argmax(v_abs[:max(3, int(0.1 * T))].max(axis=0)))
    push_mask = np.zeros(W, bool)
    push_mask[max(0, xp - push_excl_px):min(W, xp + push_excl_px + 1)] = True
    valid = amp_ok & ~push_mask
    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)
    min_step = max(1, int(np.ceil(0.15 * dt_s / LATERAL_M_PX)))
    STEP = max(step, min_step)
    cs_list = []
    for x in range(0, W - STEP):
        x2 = x + STEP
        if not (valid[x] and valid[x2]):
            continue
        sa, sb = v_avg[:, x], v_avg[:, x2]
        ea = np.sqrt(np.sum(sa ** 2))
        eb = np.sqrt(np.sum(sb ** 2))
        if ea < 1e-12 or eb < 1e-12:
            continue
        xcorr = np.correlate(sa / ea, sb / eb, mode="full")
        lags = np.arange(-(T - 1), T)
        pi = int(np.argmax(xcorr))
        if 0 < pi < len(xcorr) - 1:
            y0, y1, y2 = xcorr[pi - 1], xcorr[pi], xcorr[pi + 1]
            dn = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / dn if abs(dn) > 1e-12 else 0.0
            lag = float(lags[pi]) + frac
        else:
            lag = float(lags[pi])
        if abs(lag) < 0.05:
            continue
        cs_local = abs(STEP * LATERAL_M_PX / (lag * dt_s))
        if 0.2 < cs_local < 15.0:
            cs_list.append(cs_local)
    if not cs_list:
        return dict(cs=float("nan"), cs_std=float("nan"))
    cs_arr = np.array(cs_list)
    cs_med = float(np.median(cs_arr))
    cs_iqr = float(np.percentile(cs_arr, 75) - np.percentile(cs_arr, 25))
    if not (0.2 < cs_med < 15.0):
        return dict(cs=float("nan"), cs_std=float("nan"))
    return dict(cs=cs_med, cs_std=cs_iqr / 1.349)


# ── SWE: Voigt viscosity ──────────────────────────────────────────────────────
def estimate_voigt(
    velocity: np.ndarray,
    dt_s: float,
    depth_frac: Tuple[float, float] = (0.25, 0.75),
) -> Dict:
    T, H, W = velocity.shape
    r0, r1 = int(H * depth_frac[0]), int(H * depth_frac[1])
    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)
    V = np.fft.fftshift(np.fft.fft2(v_avg))
    omegas = np.fft.fftshift(np.fft.fftfreq(T, dt_s)) * 2 * np.pi
    kxs = np.fft.fftshift(np.fft.fftfreq(W, LATERAL_M_PX))
    omega_ok = omegas > 0
    kx_pos = kxs > 0
    kx_neg = kxs < 0
    kx_half = kxs[kx_pos]
    P_pos = np.abs(V[np.ix_(omega_ok, kx_pos)]) ** 2
    P_neg = np.abs(V[np.ix_(omega_ok, kx_neg)]) ** 2
    n_kx = min(P_pos.shape[1], P_neg.shape[1])
    P_sub = P_pos[:, :n_kx] + P_neg[:, ::-1][:, :n_kx]
    kx = kx_half[:n_kx]
    total = P_sub.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kx_c = np.where(total > 0, (P_sub * kx[None, :]).sum(axis=1) / total, float("nan"))
    omega_bins = omegas[omega_ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        cs_disp = np.where(kx_c > 0, omega_bins / kx_c, float("nan"))
    ok = np.isfinite(cs_disp) & (cs_disp > 0.2) & (cs_disp < 12.0) & (omega_bins > 0)
    if ok.sum() < 3:
        return dict(G_prime=float("nan"), E=float("nan"), eta=float("nan"))

    def voigt_cv(om, G, eta):
        num = 2 * (G ** 2 + (om * eta) ** 2)
        den = RHO * (np.sqrt(G ** 2 + (om * eta) ** 2) + G)
        return np.sqrt(np.maximum(num / den, 0))

    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            voigt_cv, omega_bins[ok], cs_disp[ok],
            p0=[3000.0, 1.0], bounds=([0, 0], [1e7, 1e3]), maxfev=2000,
        )
        G_Pa, eta = popt
        G_kPa = G_Pa / 1000.0
        return dict(G_prime=G_kPa, E=3 * G_kPa, eta=eta)
    except Exception:
        return dict(G_prime=float("nan"), E=float("nan"), eta=float("nan"))


def swe_full(velocity_mm_s: np.ndarray, dt_s: float) -> Dict:
    toa = estimate_cs_toa(velocity_mm_s, dt_s)
    voigt = estimate_voigt(velocity_mm_s, dt_s)
    return dict(
        cs=toa["cs"],
        cs_std=toa.get("cs_std", float("nan")),
        G_prime=voigt["G_prime"],
        E=voigt["E"],
        eta=voigt["eta"],
    )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_v10(path: str, device: torch.device, label: str = "V10") -> torch.nn.Module:
    """Load a V10 (= V6 backbone) checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [{label}] WARNING missing keys: {missing[:5]}")
    ep = ckpt.get("epoch", "?")
    vl = ckpt.get("best_val_loss", float("nan"))
    print(f"  [{label}] epoch={ep}  val_loss={vl:.5f}")
    return m


def load_v9(path: str, device: torch.device, label: str = "V9") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v9(config=cfg).to(device).eval()
    m.load_state_dict(ckpt["model_state_dict"], strict=True)
    ep = ckpt.get("epoch", "?")
    vl = ckpt.get("best_val_loss", float("nan"))
    print(f"  [{label}] epoch={ep}  val_loss={vl:.5f}")
    return m


def load_v2(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    m = ComplexToRealUNet(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("base_channels", 32),
        encoder_depths=list(cfg.get("encoder_depths", [32, 64, 128, 256])),
        upsampling_factor=cfg.get("upsampling_factor", 12),
        use_motion_interpolation=cfg.get("use_motion_interpolation", True),
        complex_to_real_mode=cfg.get("complex_to_real_mode", "concat"),
        use_naive_bn=cfg.get("use_naive_bn", True),
        flow_hidden_channels=cfg.get("flow_hidden_channels", 32),
        flow_scale=cfg.get("flow_scale", 0.1),
        use_prf_conditioning=cfg.get("use_prf_conditioning", False),
    ).to(device).eval()
    m.load_state_dict(ckpt["model_state_dict"])
    ep = ckpt.get("epoch", "?")
    vl = ckpt.get("best_val_loss", float("nan"))
    print(f"  [V2] epoch={ep}  val_loss={vl:.5f}")
    return m


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(
    models_dict: Dict,
    iq_path: str,
    gt_vel_path: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Returns dict: model_name -> velocity_mm_s (49, H_ROI, W)"""
    iq = np.load(iq_path).astype(np.complex64)        # (5, H, W)
    vel_gt = np.load(gt_vel_path).astype(np.float32)  # (49, H, W) in mm/s

    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    vel_s = max(float(np.percentile(np.abs(vel_gt), 99)), 1.0)

    iq_norm = iq / iq_s
    inp = pad16(np.stack([iq_norm.real, iq_norm.imag], axis=0))
    inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, 2, 5, H_pad, W_pad)
    dt = torch.tensor([DT_500HZ], device=device)

    results: Dict[str, np.ndarray] = {}
    for name, (model, model_type) in models_dict.items():
        if model_type == "v10":
            # V10 = V6: returns single velocity tensor
            out = model(inp_t, dt_seconds=dt)
        elif model_type == "v9":
            # V9: returns (velocity, cs_scalar)
            out, _ = model(inp_t, dt_seconds=dt, detach_head=False)
        else:
            # V2: no dt_seconds
            out = model(inp_t)
            if isinstance(out, dict):
                out = out["velocity"]
        pred_norm = out[0, 0].float().cpu().numpy()[:, :H_ROI, :]  # (49, H_ROI, W)
        results[name] = pred_norm * vel_s

    results["5kHz_GT"] = vel_gt[:, :H_ROI, :]
    return results


# ── Report ────────────────────────────────────────────────────────────────────
def build_report(
    all_records: List[Dict],
    model_names: List[str],
    out_dir: Path,
) -> str:
    sep = "=" * 100
    sep2 = "-" * 100
    lines = []

    def row(s: str = ""):
        lines.append(s)

    row(sep)
    row("  V10 DEPLOYMENT EVALUATION REPORT")
    row("  Input: 500 Hz IQ (5 consecutive frames, dt=2ms)")
    row("  Output: 49 virtual frames at dt=0.167ms (6kHz equivalent)")
    row("  Ground truth SWE: derived from 5 kHz native velocity (dt=0.2ms)")
    row(sep)

    for ph in ["2ft", "3ft", "cirs", "ALL"]:
        subset = [r for r in all_records if ph == "ALL" or r["phantom"] == ph]
        if not subset:
            continue
        row()
        row(f"PHANTOM: {ph.upper()}  ({len(subset)} samples)")
        row(sep2)
        G_prime_col = "G' (kPa)"
        G_prime_mae_col = "G'_MAE"
        row(
            f"  {'Model':<14} {'cs (m/s)':>10} {'cs_MAE':>10} "
            f"{G_prime_col:>10} {G_prime_mae_col:>10} {'E (kPa)':>10} {'eta (Pa.s)':>12}"
        )
        row("  " + "-" * 78)

        gt_cs = [r["5kHz_GT"]["cs"] for r in subset if np.isfinite(r["5kHz_GT"]["cs"])]
        gt_G = [r["5kHz_GT"]["G_prime"] for r in subset if np.isfinite(r["5kHz_GT"]["G_prime"])]
        gt_E = [r["5kHz_GT"]["E"] for r in subset if np.isfinite(r["5kHz_GT"]["E"])]
        gt_eta = [r["5kHz_GT"]["eta"] for r in subset if np.isfinite(r["5kHz_GT"]["eta"])]
        row(
            f"  {'5kHz GT':<14} "
            f"{np.nanmean(gt_cs) if gt_cs else float('nan'):>10.4f} {'---':>10} "
            f"{np.nanmean(gt_G) if gt_G else float('nan'):>10.4f} {'---':>10} "
            f"{np.nanmean(gt_E) if gt_E else float('nan'):>10.4f} "
            f"{np.nanmean(gt_eta) if gt_eta else float('nan'):>12.4f}"
        )

        for mdl in model_names:
            cs_v = [r[mdl]["cs"] for r in subset if np.isfinite(r[mdl]["cs"])]
            G_v = [r[mdl]["G_prime"] for r in subset if np.isfinite(r[mdl]["G_prime"])]
            E_v = [r[mdl]["E"] for r in subset if np.isfinite(r[mdl]["E"])]
            eta_v = [r[mdl]["eta"] for r in subset if np.isfinite(r[mdl]["eta"])]
            cs_mae = [
                abs(r[mdl]["cs"] - r["5kHz_GT"]["cs"])
                for r in subset
                if np.isfinite(r[mdl]["cs"]) and np.isfinite(r["5kHz_GT"]["cs"])
            ]
            G_mae = [
                abs(r[mdl]["G_prime"] - r["5kHz_GT"]["G_prime"])
                for r in subset
                if np.isfinite(r[mdl]["G_prime"]) and np.isfinite(r["5kHz_GT"]["G_prime"])
            ]

            def mn(lst):
                return float(np.mean(lst)) if lst else float("nan")

            row(
                f"  {mdl:<14} "
                f"{mn(cs_v):>10.4f} {mn(cs_mae):>10.4f} "
                f"{mn(G_v):>10.4f} {mn(G_mae):>10.4f} "
                f"{mn(E_v):>10.4f} {mn(eta_v):>12.4f}"
            )

    # Per-sample detail
    row()
    row()
    row("PER-SAMPLE DETAIL")
    row(sep2)
    G_prime_hdr = "G'"
    row(
        f"  {'Sample':<45} {'Model':<14} {'cs':>8} {'cs_MAE':>8} "
        f"{G_prime_hdr:>8} {'E':>8} {'eta':>10}"
    )
    row("  " + "-" * 105)
    for rec in all_records:
        gt_swe = rec["5kHz_GT"]
        row(
            f"  {rec['key'][:45]:<45} {'5kHz GT':<14} "
            f"{gt_swe['cs']:>8.3f} {'---':>8} "
            f"{gt_swe['G_prime']:>8.3f} {gt_swe['E']:>8.3f} {gt_swe['eta']:>10.3f}"
        )
        for mdl in model_names:
            swe = rec[mdl]
            mae = (
                abs(swe["cs"] - gt_swe["cs"])
                if np.isfinite(swe["cs"]) and np.isfinite(gt_swe["cs"])
                else float("nan")
            )
            row(
                f"  {'':45} {mdl:<14} "
                f"{swe['cs']:>8.3f} {mae:>8.3f} "
                f"{swe['G_prime']:>8.3f} {swe['E']:>8.3f} {swe['eta']:>10.3f}"
            )
        row()

    row(sep)
    row(f"  Output: {out_dir}")
    row(sep)
    return "\n".join(lines)


# ── Video (optional) ──────────────────────────────────────────────────────────
def write_deployment_video(sample_key, gt_5khz, pred_v10, pred_v2, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        import shutil
    except ImportError:
        print("  matplotlib not available — skipping video")
        return None

    T, H, W = gt_5khz.shape
    AXIAL_M_PX = 0.0507e-3
    lat_mm = (np.arange(W) - W / 2.0) * (LATERAL_M_PX * 1e3)
    dep_mm = np.arange(H) * (AXIAL_M_PX * 1e3)
    extent = [lat_mm[0], lat_mm[-1], dep_mm[-1], dep_mm[0]]

    clim = float(np.percentile(np.abs(gt_5khz), 99.5))
    clim = max(clim, 0.2)

    fig = plt.figure(figsize=(24, 5), facecolor="black")
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.04], wspace=0.06)
    axs = [fig.add_subplot(gs[i]) for i in range(5)]
    ax_cb = fig.add_subplot(gs[5])

    for ax in axs:
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=7)

    vel_kw = dict(cmap="jet", vmin=-clim, vmax=clim, aspect="auto", origin="upper", extent=extent)
    err_kw = dict(cmap="hot", vmin=0, vmax=clim, aspect="auto", origin="upper", extent=extent)

    ims = [
        axs[0].imshow(gt_5khz[0], **vel_kw),
        axs[1].imshow(pred_v10[0], **vel_kw),
        axs[2].imshow(pred_v2[0], **vel_kw),
        axs[3].imshow(np.abs(pred_v10[0] - gt_5khz[0]), **err_kw),
        axs[4].imshow(np.abs(pred_v2[0] - gt_5khz[0]), **err_kw),
    ]
    for ax, title in zip(axs, [
        "GT 5kHz (dt=0.2ms)", "V10 (virtual 6kHz)", "V2 (virtual 6kHz)",
        "|Error| V10 vs GT", "|Error| V2 vs GT",
    ]):
        ax.set_title(title, color="white", fontsize=9, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=-clim, vmax=clim))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Velocity (mm/s)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    sup = fig.suptitle("", color="white", fontsize=10, fontweight="bold")

    def _update(fi):
        ims[0].set_data(gt_5khz[fi])
        ims[1].set_data(pred_v10[fi])
        ims[2].set_data(pred_v2[fi])
        ims[3].set_data(np.abs(pred_v10[fi] - gt_5khz[fi]))
        ims[4].set_data(np.abs(pred_v2[fi] - gt_5khz[fi]))
        sup.set_text(
            f"Deployment 500Hz→6kHz vs 5kHz GT  |  {sample_key}  |  frame {fi+1}/{T}"
        )
        return ims + [sup]

    anim = FuncAnimation(fig, _update, frames=T, blit=True, interval=100)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg"):
        writer = FFMpegWriter(fps=10, extra_args=["-pix_fmt", "yuv420p"])
        save_path = out_path.with_suffix(".mp4")
        anim.save(str(save_path), writer=writer, dpi=120, savefig_kwargs={"facecolor": "black"})
    else:
        writer = PillowWriter(fps=10)
        save_path = out_path.with_suffix(".gif")
        anim.save(str(save_path), writer=writer, savefig_kwargs={"facecolor": "black"})

    plt.close(fig)
    print(f"  Saved video: {save_path}")
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="V10 deployment evaluation")
    parser.add_argument(
        "--v10", type=str,
        default=str(ROOT / "saved_models/wavepinn_v10_best.pt"),
        help="Path to V10 checkpoint",
    )
    parser.add_argument(
        "--v9-ep60", type=str,
        default=str(ROOT / "SWE_PINN_outputs_wavepinn_v9_checkpoint_epoch_60.pt"),
        help="Path to V9-Ep60 checkpoint",
    )
    parser.add_argument(
        "--v2", type=str,
        default=str(ROOT / "saved_models/fullscale_velocity_v2_best.pt"),
        help="Path to V2 checkpoint",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(ROOT / "data/fullscale_velocity/val"),
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(ROOT / "outputs/deployment_eval_v10"),
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print("  V10 DEPLOYMENT EVALUATION: 500Hz → 6kHz vs 5kHz GT SWE")
    print(f"  dt_input=2ms  dt_output={DT_OUT*1e3:.4f}ms  dt_gt={DT_5KHZ_GT*1e3:.2f}ms")
    print(f"{'='*72}\n")

    print("[1/4] Loading models...")
    models: Dict = {}

    if os.path.exists(args.v10):
        models["V10"] = (load_v10(args.v10, device, "V10"), "v10")
    else:
        print(f"  V10 checkpoint not found: {args.v10}")
        print("  Run training first: python -m src.training.train_wavepinn_v10 --config configs/config_wavepinn_v10.json")

    if os.path.exists(args.v9_ep60):
        models["V9-Ep60"] = (load_v9(args.v9_ep60, device, "V9-Ep60"), "v9")
    else:
        print(f"  V9-Ep60 not found: {args.v9_ep60} (skipping)")

    if os.path.exists(args.v2):
        models["V2"] = (load_v2(args.v2, device), "v2")
    else:
        print(f"  V2 not found: {args.v2} (skipping)")

    if not models:
        print("ERROR: No models loaded. Exiting.")
        return

    print(f"\n[2/4] Building 500Hz ↔ 5kHz GT matched pairs...")
    files_500 = sorted(f for f in os.listdir(data_dir) if "500Hz" in f and f.startswith("sparse_"))
    vel5k_by_key: Dict[str, List[str]] = {}
    for f in os.listdir(data_dir):
        m = re.match(r"velocity_(.+?)_5kHz", f)
        if m:
            vel5k_by_key.setdefault(m.group(1), []).append(f)

    pairs = []
    for f in files_500:
        m = re.match(r"sparse_(.+?)_500Hz", f)
        if not m:
            continue
        k = m.group(1)
        gts = vel5k_by_key.get(k, [])
        if not gts:
            continue
        phantom = "cirs" if "cirs" in k else ("3ft" if "3ft" in k else "2ft")
        pairs.append((f, gts[0], phantom, k))

    print(f"  Found {len(pairs)} matched pairs")
    if not pairs:
        print("ERROR: No matched pairs found. Check data directory.")
        return

    print(f"\n[3/4] Inference + SWE analysis...")
    all_records: List[Dict] = []
    video_payload = None

    for sp_file, gt_file, phantom, key_id in pairs:
        sp_path = data_dir / sp_file
        gt_path = data_dir / gt_file
        print(f"  {key_id[:55]}")

        try:
            preds = infer(models, str(sp_path), str(gt_path), device)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if (
            video_payload is None
            and "V10" in preds
            and "V2" in preds
            and "5kHz_GT" in preds
            and not args.no_video
        ):
            video_payload = {
                "sample_key": key_id,
                "gt_5khz": preds["5kHz_GT"].copy(),
                "v10": preds["V10"].copy(),
                "v2": preds["V2"].copy(),
            }

        row = {"key": key_id, "phantom": phantom}
        for name, vel_mm in preds.items():
            dt = DT_5KHZ_GT if name == "5kHz_GT" else DT_OUT
            swe = swe_full(vel_mm, dt)
            row[name] = swe
            print(
                f"    {name:<12}: cs={swe['cs']:.3f}  "
                f"G'={swe['G_prime']:.3f}  E={swe['E']:.3f}  eta={swe['eta']:.3f}"
            )
        all_records.append(row)

    if not all_records:
        print("ERROR: No samples processed.")
        return

    print(f"\n[4/4] Building report...")
    model_names = list(models.keys())
    report = build_report(all_records, model_names, out_dir)

    report_path = out_dir / "deployment_report_v10.txt"
    report_path.write_text(report)
    print("\n" + report)

    # Video for first matched sample
    if video_payload and not args.no_video:
        print("\nGenerating deployment comparison video...")
        write_deployment_video(
            sample_key=video_payload["sample_key"],
            gt_5khz=video_payload["gt_5khz"],
            pred_v10=video_payload["v10"],
            pred_v2=video_payload["v2"],
            out_path=out_dir / "deployment_comparison_v10",
        )

    print(f"\nOutput saved to: {out_dir}")


if __name__ == "__main__":
    main()
