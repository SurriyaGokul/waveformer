#!/usr/bin/env python3
"""Simulated 500Hz-vs-5kHz comparison on the PVA phantom dataset.

PVA has no raw multi-frame complex IQ available in this environment (original_path
files for the raw acquisitions are missing) -- only a native 10kHz dense GT velocity
field (49 frames, dt=0.1ms) and an already-decimated 5-frame complex sparse input
(dt=1.2ms, PRF~833Hz) exist per acquisition. Because the model always maps 5 sparse
frames -> 49 dense frames via upsampling_factor=12, and here dt_in/12 == native dt,
the model's predicted dense output is time-aligned with the native GT grid. So instead
of decimating raw IQ (impossible -- data doesn't exist), this script decimates the
finished dense velocity fields -- both GT and the model's own dense prediction --
identically in time, to simulate what a 5kHz (every 2nd frame) or 500Hz (every 20th
frame) acquisition's cs estimate would look like.

Reuses model loading / inference / cs estimation from evaluate_pva_v10b_v12.py so the
cs numbers stay directly comparable to the un-decimated PVA cs MAE already reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_pva_v10b_v12 import estimate_cs, load_model, infer_one

DATA_ROOT = PROJECT_ROOT / "data" / "pva_velocity"


def decimate(arr: np.ndarray, step: int) -> np.ndarray:
    return arr[::step]


def run(model_path: str, tag: str) -> Dict[str, pd.DataFrame]:
    with open(DATA_ROOT / "metadata.json") as f:
        meta = json.load(f)
    entries = meta.get("val", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(model_path), device)

    rows_5k: List[Dict] = []
    rows_500: List[Dict] = []

    for i, entry in enumerate(entries, start=1):
        name = entry["name"]
        sparse_path = DATA_ROOT / "val" / f"sparse_{name}.npy"
        gt_path = DATA_ROOT / "val" / f"velocity_{name}.npy"
        if not sparse_path.exists() or not gt_path.exists():
            print(f"  MISSING: {name}")
            continue

        sparse_iq = np.load(str(sparse_path)).astype(np.complex64)
        gt = np.load(str(gt_path)).astype(np.float32)

        dt_in_s = float(entry["sparse_step"]) / float(entry["prf_hz"])
        dt_out_s = dt_in_s / 12.0  # native rate, ~0.1ms

        pred_denorm, _, _ = infer_one(model, sparse_iq, gt, dt_in_s, device)

        # --- 5kHz-equivalent: every 2nd frame, dt=0.0002s (=1/5000) ---
        gt_5k = decimate(gt, 2)
        pred_5k = decimate(pred_denorm, 2)
        dt_5k = dt_out_s * 2
        cs_gt_5k = estimate_cs(gt_5k, dt_5k)
        cs_pred_5k = estimate_cs(pred_5k, dt_5k)
        cs_mae_5k = abs(cs_pred_5k - cs_gt_5k) if np.isfinite(cs_gt_5k) and np.isfinite(cs_pred_5k) else float("nan")
        rows_5k.append({
            "name": name, "source": entry.get("source", "unknown"),
            "n_frames": gt_5k.shape[0], "dt_s": dt_5k,
            "cs_gt_m_s": cs_gt_5k, "cs_pred_m_s": cs_pred_5k, "cs_mae_m_s": cs_mae_5k,
        })

        # --- 500Hz-equivalent: every 20th frame, dt=0.002s (=1/500) ---
        gt_500 = decimate(gt, 20)
        pred_500 = decimate(pred_denorm, 20)
        dt_500 = dt_out_s * 20
        cs_gt_500 = estimate_cs(gt_500, dt_500)
        cs_pred_500 = estimate_cs(pred_500, dt_500)
        cs_mae_500 = abs(cs_pred_500 - cs_gt_500) if np.isfinite(cs_gt_500) and np.isfinite(cs_pred_500) else float("nan")
        rows_500.append({
            "name": name, "source": entry.get("source", "unknown"),
            "n_frames": gt_500.shape[0], "dt_s": dt_500,
            "cs_gt_m_s": cs_gt_500, "cs_pred_m_s": cs_pred_500, "cs_mae_m_s": cs_mae_500,
        })

        print(f"[{i:02d}/{len(entries):02d}] {name}: "
              f"5kHz-eq(n={gt_5k.shape[0]}) cs_gt={cs_gt_5k:.3f} cs_pred={cs_pred_5k:.3f} MAE={cs_mae_5k:.3f} | "
              f"500Hz-eq(n={gt_500.shape[0]}) cs_gt={cs_gt_500:.3f} cs_pred={cs_pred_500:.3f} MAE={cs_mae_500:.3f}")

    df_5k = pd.DataFrame(rows_5k)
    df_500 = pd.DataFrame(rows_500)

    out_dir = PROJECT_ROOT / "outputs" / f"pva_simulated_rates_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_5k.to_csv(out_dir / "sim_5khz_per_sample.csv", index=False)
    df_500.to_csv(out_dir / "sim_500hz_per_sample.csv", index=False)

    summary = {
        "5khz_equiv": {
            "n_samples": int(df_5k["cs_mae_m_s"].notna().sum()),
            "cs_mae_mean_m_s": float(df_5k["cs_mae_m_s"].mean(skipna=True)),
        },
        "500hz_equiv": {
            "n_samples": int(df_500["cs_mae_m_s"].notna().sum()),
            "cs_mae_mean_m_s": float(df_500["cs_mae_m_s"].mean(skipna=True)),
        },
    }
    for src in df_5k["source"].unique():
        summary["5khz_equiv"][f"cs_mae_{src}_m_s"] = float(df_5k[df_5k.source == src]["cs_mae_m_s"].mean(skipna=True))
    for src in df_500["source"].unique():
        summary["500hz_equiv"][f"cs_mae_{src}_m_s"] = float(df_500[df_500.source == src]["cs_mae_m_s"].mean(skipna=True))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{tag} summary: {json.dumps(summary, indent=2)}")
    return {"5khz": df_5k, "500hz": df_500}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    run(args.model_path, args.tag)
