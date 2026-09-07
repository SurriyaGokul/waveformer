#!/usr/bin/env python3
"""Evaluate WavePINN checkpoints (v10b / v12) on the previously-unused PVA phantom dataset.

Adapted from scripts/evaluate_goat_heart_v10b.py (model loading + cs estimation via
normalized-cross-correlation ToA) plus the r_active/L1 convention from
compute_validation_metrics() in src/training/train_wavepinn_v10.py (normalized-space
Pearson r on frames where |target_norm| > 0.1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

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

# dx_m not present in PVA metadata; reuse the same default pixel spacing
# scripts/evaluate_goat_heart_v10b.py assumes for this imaging platform.
DX_M = 0.1013e-3


def pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    *lead, h, w = arr.shape
    p_h = (multiple - h % multiple) % multiple
    p_w = (multiple - w % multiple) % multiple
    pad = [(0, 0)] * len(lead) + [(0, p_h), (0, p_w)]
    return np.pad(arr, pad, mode="constant")


def estimate_cs(velocity: np.ndarray, dt_s: float, dx_m: float = DX_M) -> float:
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
def infer_one(model, sparse_iq: np.ndarray, gt: np.ndarray, dt_in_s: float, device: torch.device):
    h, w = sparse_iq.shape[1], sparse_iq.shape[2]

    iq_scale = max(float(np.percentile(np.abs(sparse_iq), 99)), 1e-6)
    iq_norm = (sparse_iq / iq_scale).astype(np.complex64)

    vel_scale = max(float(np.percentile(np.abs(gt), 99)), 1e-6)

    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)
    inp = pad_to_multiple(inp, multiple=16)
    inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
    dt_t = torch.tensor([dt_in_s], dtype=torch.float32, device=device)

    pred = model(inp_t, dt_seconds=dt_t)
    pred_norm_full = pred[0, 0].detach().cpu().numpy()  # normalized model output, padded
    pred_norm = pred_norm_full[:, :h, :w]
    pred_denorm = (pred_norm * vel_scale).astype(np.float32)
    gt_norm = (gt / vel_scale).astype(np.float32)
    return pred_denorm, pred_norm.astype(np.float32), gt_norm


def compute_r_active_l1(pred_norm: np.ndarray, gt_norm: np.ndarray) -> Dict[str, float]:
    """Mirrors compute_validation_metrics() in train_wavepinn_v10.py (normalized space)."""
    l1 = float(np.abs(pred_norm - gt_norm).mean())
    mse = float(((pred_norm - gt_norm) ** 2).mean())

    n_active = min(pred_norm.shape[0], 12)
    active_mask = np.abs(gt_norm[:n_active]) > 0.1
    p_active = pred_norm[:n_active][active_mask]
    t_active = gt_norm[:n_active][active_mask]
    r_active = float("nan")
    if len(p_active) > 10 and np.std(p_active) > 1e-8 and np.std(t_active) > 1e-8:
        r_active = float(np.corrcoef(p_active, t_active)[0, 1])

    return {"l1": l1, "mse": mse, "r_active": r_active}


def run(model_path: str, output_dir: str, data_root: str = "data/pva_velocity") -> pd.DataFrame:
    data_root_p = Path(data_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root_p / "metadata.json") as f:
        meta = json.load(f)
    entries = meta.get("val", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(model_path), device)

    rows: List[Dict] = []
    for i, entry in enumerate(entries, start=1):
        name = entry["name"]
        sparse_path = data_root_p / "val" / f"sparse_{name}.npy"
        gt_path = data_root_p / "val" / f"velocity_{name}.npy"
        if not sparse_path.exists() or not gt_path.exists():
            print(f"  MISSING: {name}")
            continue

        sparse_iq = np.load(str(sparse_path)).astype(np.complex64)
        gt = np.load(str(gt_path)).astype(np.float32)

        dt_in_s = float(entry["sparse_step"]) / float(entry["prf_hz"])
        dt_out_s = dt_in_s / 12.0  # upsampling_factor=12

        pred_denorm, pred_norm, gt_norm = infer_one(model, sparse_iq, gt, dt_in_s, device)
        m = compute_r_active_l1(pred_norm, gt_norm)

        cs_gt = estimate_cs(gt, dt_out_s)
        cs_pred = estimate_cs(pred_denorm, dt_out_s)
        cs_mae = abs(cs_pred - cs_gt) if np.isfinite(cs_gt) and np.isfinite(cs_pred) else float("nan")

        voigt_gt = estimate_voigt(gt, dt_out_s)
        voigt_pred = estimate_voigt(pred_denorm, dt_out_s)

        rows.append({
            "name": name,
            "source": entry.get("source", "unknown"),
            "l1": m["l1"],
            "mse": m["mse"],
            "r_active": m["r_active"],
            "cs_gt_m_s": cs_gt,
            "cs_pred_m_s": cs_pred,
            "cs_mae_m_s": cs_mae,
            "G_prime_gt_kPa": voigt_gt["G_prime"],
            "G_prime_pred_kPa": voigt_pred["G_prime"],
            "E_gt_kPa": voigt_gt["E"],
            "E_pred_kPa": voigt_pred["E"],
            "eta_gt_Pa_s": voigt_gt["eta"],
            "eta_pred_Pa_s": voigt_pred["eta"],
        })
        print(f"[{i:02d}/{len(entries):02d}] {name}: L1={m['l1']:.4f} r_active={m['r_active']:.3f} "
              f"cs_gt={cs_gt:.3f} cs_pred={cs_pred:.3f} cs_MAE={cs_mae:.3f} "
              f"G_gt={voigt_gt['G_prime']:.2f} G_pred={voigt_pred['G_prime']:.2f} "
              f"eta_gt={voigt_gt['eta']:.2f} eta_pred={voigt_pred['eta']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pva_metrics_per_sample.csv", index=False)

    summary = {
        "n_samples": int(len(df)),
        "l1_mean": float(df["l1"].mean()),
        "mse_mean": float(df["mse"].mean()),
        "r_active_mean": float(df["r_active"].mean(skipna=True)),
        "cs_mae_mean_m_s": float(df["cs_mae_m_s"].mean(skipna=True)),
        "cs_mae_std_m_s": float(df["cs_mae_m_s"].std(skipna=True)),
        "G_prime_gt_kPa_mean": float(df["G_prime_gt_kPa"].mean(skipna=True)),
        "G_prime_pred_kPa_mean": float(df["G_prime_pred_kPa"].mean(skipna=True)),
        "E_gt_kPa_mean": float(df["E_gt_kPa"].mean(skipna=True)),
        "E_pred_kPa_mean": float(df["E_pred_kPa"].mean(skipna=True)),
        "eta_gt_Pa_s_mean": float(df["eta_gt_Pa_s"].mean(skipna=True)),
        "eta_pred_Pa_s_mean": float(df["eta_pred_Pa_s"].mean(skipna=True)),
    }
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        summary[f"cs_mae_{src}_m_s"] = float(sub["cs_mae_m_s"].mean(skipna=True))
        summary[f"n_{src}"] = int(len(sub))
        summary[f"G_prime_gt_{src}_kPa"] = float(sub["G_prime_gt_kPa"].mean(skipna=True))
        summary[f"G_prime_pred_{src}_kPa"] = float(sub["G_prime_pred_kPa"].mean(skipna=True))
        summary[f"eta_gt_{src}_Pa_s"] = float(sub["eta_gt_Pa_s"].mean(skipna=True))
        summary[f"eta_pred_{src}_Pa_s"] = float(sub["eta_pred_Pa_s"].mean(skipna=True))

    with open(out_dir / "pva_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="data/pva_velocity")
    args = parser.parse_args()
    run(args.model_path, args.output_dir, args.data_root)
