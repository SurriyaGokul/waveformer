#!/usr/bin/env python3
"""Evaluate WavePINN v10b on goat-heart data and generate paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v10 import create_wavepinn_v10
# V14: the original estimate_voigt had a 2*pi wavenumber unit error that made
# every G'/eta value it produced an artefact (see src/analysis/dispersion.py).
# Route through the rebuilt estimator, keeping this call signature.
from src.analysis.dispersion import estimate_cs_toa as _v14_toa
from src.analysis.dispersion import estimate_dispersion as _v14_disp


def estimate_voigt(velocity, dt_s, **_kwargs):
    toa = _v14_toa(velocity, dt_s)
    r = _v14_disp(velocity, dt_s, cs_toa_hint=toa["cs"])
    return dict(G_prime=r.G_prime_kpa, E=r.E_kpa, eta=r.eta)


def pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    *lead, h, w = arr.shape
    p_h = (multiple - h % multiple) % multiple
    p_w = (multiple - w % multiple) % multiple
    pad = [(0, 0)] * len(lead) + [(0, p_h), (0, p_w)]
    return np.pad(arr, pad, mode="constant")


def estimate_cs(velocity: np.ndarray, dt_s: float, dx_m: float = 0.1013e-3) -> float:
    """Robust cross-correlation ToA cs estimate."""
    t, h, w = velocity.shape
    r0, r1 = int(h * 0.2), int(h * 0.8)
    v_avg = np.mean(velocity[:, r0:r1, :], axis=1)
    v_abs = np.abs(v_avg)

    amp = v_abs.max(axis=0)
    valid = amp > (np.percentile(amp[amp > 0], 15) if amp.max() > 0 else 0)

    x_push = int(np.argmax(v_abs[: max(3, t // 10)].max(axis=0)))
    excl = 10
    valid[max(0, x_push - excl) : min(w, x_push + excl + 1)] = False

    step = min(max(3, int(np.ceil(4.0 * dt_s / dx_m))), 12)
    cs_all = []

    for x in range(0, w - step):
        x2 = x + step
        if not (valid[x] and valid[x2]):
            continue
        a = v_avg[:, x]
        b = v_avg[:, x2]
        ea = np.sqrt((a ** 2).sum())
        eb = np.sqrt((b ** 2).sum())
        if ea < 1e-12 or eb < 1e-12:
            continue

        xc = np.correlate(a / ea, b / eb, mode="full")
        lags = np.arange(-(t - 1), t, dtype=float)
        pi = int(np.argmax(xc))
        if 0 < pi < len(xc) - 1:
            y0, y1, y2 = xc[pi - 1], xc[pi], xc[pi + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            lag = lags[pi] + frac
        else:
            lag = lags[pi]

        if abs(lag) < 0.15:
            continue

        cs_local = abs(step * dx_m / (lag * dt_s))
        if 0.3 < cs_local < 12.0:
            cs_all.append(cs_local)

    if len(cs_all) < 3:
        return float("nan")

    return float(np.median(np.array(cs_all)))


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    model = create_wavepinn_v10(config=cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


@torch.no_grad()
def infer_one(model, sparse_iq: np.ndarray, gt: np.ndarray, dt_in_s: float, device: torch.device) -> np.ndarray:
    h, w = sparse_iq.shape[1], sparse_iq.shape[2]

    iq_scale = float(np.percentile(np.abs(sparse_iq), 99))
    iq_scale = max(iq_scale, 1e-6)
    iq_norm = (sparse_iq / iq_scale).astype(np.complex64)

    vel_scale = float(np.percentile(np.abs(gt), 99))
    vel_scale = max(vel_scale, 1e-6)

    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)
    inp = pad_to_multiple(inp, multiple=16)

    inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
    dt_t = torch.tensor([dt_in_s], dtype=torch.float32, device=device)

    pred = model(inp_t, dt_seconds=dt_t)
    pred_np = pred[0, 0].detach().cpu().numpy()[:, :h, :w]
    return (pred_np * vel_scale).astype(np.float32)


def compute_pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))

    data_range = float(gt.max() - gt.min())
    if data_range < 1e-8:
        return {"mae": mae, "rmse": rmse, "ssim": float("nan"), "psnr": float("nan")}

    psnr = float(20 * np.log10(data_range / (rmse + 1e-12)))

    ssim_vals = []
    for t in range(min(gt.shape[0], pred.shape[0])):
        try:
            ssim_vals.append(float(ssim(gt[t], pred[t], data_range=data_range)))
        except Exception:
            pass

    return {
        "mae": mae,
        "rmse": rmse,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
        "psnr": psnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WavePINN v10b goat-heart evaluation")
    parser.add_argument("--model_path", default=str(PROJECT_ROOT / "SWE_PINN_outputs_wavepinn_v10b_best_model.pt"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data" / "goat_heart_velocity"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "goat_heart_v10b"))
    parser.add_argument("--sample_name", default=None, help="Only evaluate a single sample name")
    parser.add_argument("--save_predictions", action="store_true", help="Save predicted velocity arrays")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_root / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    entries = meta.get("inference", [])
    if not entries:
        raise RuntimeError("No goat-heart inference entries found")

    if args.sample_name:
        entries = [e for e in entries if e.get("name") == args.sample_name]
        if not entries:
            raise RuntimeError(f"Sample not found in metadata: {args.sample_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path), device)

    rows: List[Dict] = []
    for i, entry in enumerate(entries, start=1):
        name = entry["name"]
        sparse_path = data_root / "inference" / f"sparse_{name}.npy"
        gt_path = data_root / "inference" / f"velocity_{name}.npy"
        if not sparse_path.exists() or not gt_path.exists():
            continue

        sparse_iq = np.load(str(sparse_path)).astype(np.complex64)
        gt = np.load(str(gt_path)).astype(np.float32)

        dt_in_s = float(entry.get("sparse_step", 12)) / float(entry.get("prf_hz", 20000.0))
        dt_out_s = dt_in_s / 12.0

        pred = infer_one(model, sparse_iq, gt, dt_in_s, device)
        if args.save_predictions:
            np.save(out_dir / f"pred_velocity_{name}.npy", pred)
        pix = compute_pixel_metrics(gt, pred)

        cs_gt = estimate_cs(gt, dt_out_s)
        cs_pred = estimate_cs(pred, dt_out_s)
        cs_mae = abs(cs_pred - cs_gt) if np.isfinite(cs_gt) and np.isfinite(cs_pred) else float("nan")

        voigt_gt = estimate_voigt(gt, dt_out_s)
        voigt_pred = estimate_voigt(pred, dt_out_s)

        rows.append(
            {
                "name": name,
                "sample_id": entry.get("sample_id", "unknown"),
                "region": entry.get("region", "unknown"),
                "frequency": entry.get("frequency", "unknown"),
                "mae_mm_s": pix["mae"],
                "rmse_mm_s": pix["rmse"],
                "ssim": pix["ssim"],
                "psnr_db": pix["psnr"],
                "cs_gt_m_s": cs_gt,
                "cs_pred_m_s": cs_pred,
                "cs_mae_m_s": cs_mae,
                "G_prime_gt_kPa": voigt_gt["G_prime"],
                "G_prime_pred_kPa": voigt_pred["G_prime"],
                "E_gt_kPa": voigt_gt["E"],
                "E_pred_kPa": voigt_pred["E"],
                "eta_gt_Pa_s": voigt_gt["eta"],
                "eta_pred_Pa_s": voigt_pred["eta"],
            }
        )

        print(f"[{i:02d}/{len(entries):02d}] {name}: RMSE={pix['rmse']:.3f}, cs_MAE={cs_mae:.3f}")

    if not rows:
        raise RuntimeError("No goat-heart rows computed")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "goat_metrics_per_sample.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    summary = {
        "n_samples": int(len(df)),
        "rmse_mean_mm_s": float(df["rmse_mm_s"].mean()),
        "rmse_std_mm_s": float(df["rmse_mm_s"].std()),
        "ssim_mean": float(df["ssim"].mean()),
        "psnr_mean_db": float(df["psnr_db"].mean()),
        "cs_mae_mean_m_s": float(df["cs_mae_m_s"].mean()),
        "cs_mae_std_m_s": float(df["cs_mae_m_s"].std()),
        "G_prime_gt_kPa_mean": float(df["G_prime_gt_kPa"].mean(skipna=True)),
        "G_prime_pred_kPa_mean": float(df["G_prime_pred_kPa"].mean(skipna=True)),
        "E_gt_kPa_mean": float(df["E_gt_kPa"].mean(skipna=True)),
        "E_pred_kPa_mean": float(df["E_pred_kPa"].mean(skipna=True)),
        "eta_gt_Pa_s_mean": float(df["eta_gt_Pa_s"].mean(skipna=True)),
        "eta_pred_Pa_s_mean": float(df["eta_pred_Pa_s"].mean(skipna=True)),
    }

    paired = df.dropna(subset=["cs_gt_m_s", "cs_pred_m_s"])
    if len(paired) >= 3:
        r, p = pearsonr(paired["cs_gt_m_s"], paired["cs_pred_m_s"])
        summary["cs_r"] = float(r)
        summary["cs_r_p"] = float(p)

    with open(out_dir / "goat_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figure 1: cs scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(paired["cs_gt_m_s"], paired["cs_pred_m_s"], c="#2E75B6", edgecolors="white", s=80)
    if len(paired) >= 2:
        lo = min(float(paired["cs_gt_m_s"].min()), float(paired["cs_pred_m_s"].min())) - 0.2
        hi = max(float(paired["cs_gt_m_s"].max()), float(paired["cs_pred_m_s"].max())) + 0.2
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Goat GT $c_s$ (m/s)")
    ax.set_ylabel("Goat v10b Pred $c_s$ (m/s)")
    ax.set_title("Goat-Heart Shear-Wave Speed Agreement")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "goat_cs_scatter.png", dpi=220)
    plt.close(fig)

    # Figure 2: RMSE by sample
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    ax.bar(x, df["rmse_mm_s"], color="#4DAC26", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("RMSE (mm/s)")
    ax.set_title("Goat-Heart RMSE per Sample")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "goat_rmse_per_sample.png", dpi=220)
    plt.close(fig)

    print("\nGoat-heart evaluation complete")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {out_dir / 'goat_summary.json'}")


if __name__ == "__main__":
    main()
