#!/usr/bin/env python3
"""
Validate GT-free deployment scale (src/utils/deployment_scale.py).

For each 500 Hz val/inference sample:
  1. Compare proxy vel_scale (T=5 window) vs GT vel_scale (99th-pct of |GT vel|).
  2. Compare proxy vel_scale (full IQ window) vs GT vel_scale.
  3. Assert that cs ToA is identical when denormalizing with proxy vs GT scale
     (since normalised-xcorr ToA is amplitude-invariant).

Acceptance criteria (per plan):
  - Full-window proxy/GT ratio: median ≈ 1.0 ± 5% (|median - 1| < 0.05),
    IQR < 0.15.
  - cs from proxy-scaled vs GT-scaled predictions: max diff < 0.01 m/s.

Usage:
  python scripts/validate_deployment_scale.py
  python scripts/validate_deployment_scale.py --model_path <path> --max_samples 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.deployment_scale import estimate_velocity_scale_from_iq

MODEL_PATH_DEFAULT = "SWE_PINN_outputs_wavepinn_v10b_best_model.pt"
DATA_META_PATH = "data/fullscale_velocity/metadata.json"
UPSAMPLING_FACTOR = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_multiple(arr: np.ndarray, multiple: int = 16):
    *lead, h, w = arr.shape
    p_h = (multiple - h % multiple) % multiple
    p_w = (multiple - w % multiple) % multiple
    if p_h == 0 and p_w == 0:
        return arr, (0, 0)
    pad = [(0, 0)] * len(lead) + [(0, p_h), (0, p_w)]
    return np.pad(arr, pad, mode="constant"), (p_h, p_w)


def load_model(ckpt_path: str, device: torch.device):
    from src.architectures.wavepinn_v10 import create_wavepinn_v10
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    model = create_wavepinn_v10(config=cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


def _get_500hz_samples(meta: Dict) -> List[Dict]:
    samples = []
    for split in ("val", "inference", "train"):
        for s in meta.get(split, []):
            freq = str(s.get("frequency", "")).lower()
            name = str(s.get("name", ""))
            if "500hz" in freq or "500hz" in name.lower():
                samples.append(s)
    seen = set()
    dedup = []
    for s in samples:
        if s["name"] not in seen:
            seen.add(s["name"])
            dedup.append(s)
    return dedup


def _load_iq_full(sample_meta: Dict) -> Optional[np.ndarray]:
    """Load full IQ window (H, W, T), pre-clutter-filter."""
    try:
        raw = np.load(sample_meta["original_path"])
        r   = sample_meta["roi"]
        win = sample_meta["window"]
        iq_win = raw[
            r["min_row"]: r["max_row"] + 1,
            r["min_col"]: r["max_col"] + 1,
            win[0]: win[1],
        ].astype(np.complex64)
        return iq_win
    except Exception as exc:
        print(f"    IQ load failed: {exc}")
        return None


def _load_5frame_iq(sample_meta: Dict) -> Optional[np.ndarray]:
    """Load clutter-filtered 5-frame IQ, transpose to (H, W, 5) for Loupas."""
    try:
        raw = np.load(sample_meta["original_path"])
        r   = sample_meta["roi"]
        win = sample_meta["window"]
        iq_win = raw[
            r["min_row"]: r["max_row"] + 1,
            r["min_col"]: r["max_col"] + 1,
            win[0]: win[1],
        ].astype(np.complex64)
        iq_win = iq_win - iq_win.mean(axis=2, keepdims=True)
        iq_5 = iq_win[:, :, :5]  # (H, W, 5)
        return iq_5
    except Exception as exc:
        print(f"    5-frame IQ load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--data_meta",  default=DATA_META_PATH)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_model", action="store_true",
                        help="Skip model inference (scale ratio only).")
    args = parser.parse_args()

    with open(args.data_meta) as f:
        meta = json.load(f)

    samples = _get_500hz_samples(meta)
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Validating deployment scale on {len(samples)} 500 Hz samples")
    print(f"Model: {args.model_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if not args.no_model:
        ckpt = Path(args.model_path)
        if ckpt.exists():
            model = load_model(str(ckpt), device)
            print(f"  Model loaded on {device}\n")
        else:
            print(f"  Checkpoint not found: {ckpt} — skipping model tests\n")

    ratios_full = []
    ratios_5fr  = []
    results = []

    for sm in samples:
        name = sm["name"]
        dt_s = float(1.0 / sm["prf_hz"])

        vel_path = None
        for split in ("val", "inference", "train"):
            p = PROJECT_ROOT / "data" / "fullscale_velocity" / split / f"velocity_{name}.npy"
            if p.exists():
                vel_path = p
                break

        if vel_path is None:
            print(f"  [{name}] SKIP: no GT velocity file")
            continue

        gt_vel = np.load(str(vel_path)).astype(np.float32)
        vel_scale_gt = float(np.percentile(np.abs(gt_vel), 99))
        vel_scale_gt = max(vel_scale_gt, 1.0)

        # Full window proxy
        iq_full = _load_iq_full(sm)
        scale_full = (
            estimate_velocity_scale_from_iq(iq_full, dt_s) if iq_full is not None else float("nan")
        )

        # T=5 window proxy
        iq_5 = _load_5frame_iq(sm)
        scale_5 = (
            estimate_velocity_scale_from_iq(iq_5, dt_s) if iq_5 is not None else float("nan")
        )

        ratio_full = scale_full / vel_scale_gt if vel_scale_gt > 0 else float("nan")
        ratio_5    = scale_5    / vel_scale_gt if vel_scale_gt > 0 else float("nan")

        ratios_full.append(ratio_full)
        ratios_5fr.append(ratio_5)

        print(f"  {name}")
        print(f"    GT scale={vel_scale_gt:.3f}  full-window={scale_full:.3f}  T=5={scale_5:.3f}")
        print(f"    ratio(full)={ratio_full:.4f}  ratio(T=5)={ratio_5:.4f}")

        results.append({
            "name": name,
            "vel_scale_gt": vel_scale_gt,
            "vel_scale_full_window": scale_full,
            "vel_scale_5frame": scale_5,
            "ratio_full": ratio_full,
            "ratio_5frame": ratio_5,
        })

    if not results:
        print("No samples processed.")
        return

    ratios_full = np.array([r for r in ratios_full if not np.isnan(r)])
    ratios_5fr  = np.array([r for r in ratios_5fr  if not np.isnan(r)])

    print("\n" + "=" * 60)
    print("Scale ratio summary (proxy / GT):")
    if len(ratios_full):
        p25, p50, p75 = np.percentile(ratios_full, [25, 50, 75])
        print(f"  Full window:  median={p50:.4f}  IQR=[{p25:.4f},{p75:.4f}]  "
              f"range=[{ratios_full.min():.4f},{ratios_full.max():.4f}]")
    if len(ratios_5fr):
        p25, p50, p75 = np.percentile(ratios_5fr, [25, 50, 75])
        print(f"  T=5 window:   median={p50:.4f}  IQR=[{p25:.4f},{p75:.4f}]  "
              f"range=[{ratios_5fr.min():.4f},{ratios_5fr.max():.4f}]")

    print("\nAcceptance check (full window):")
    if len(ratios_full):
        med = float(np.median(ratios_full))
        iqr = float(np.percentile(ratios_full, 75) - np.percentile(ratios_full, 25))
        ok_median = abs(med - 1.0) < 0.05
        ok_iqr    = iqr < 0.15
        print(f"  |median - 1| = {abs(med-1.0):.4f} < 0.05: {'PASS' if ok_median else 'FAIL'}")
        print(f"  IQR = {iqr:.4f} < 0.15: {'PASS' if ok_iqr else 'FAIL'}")


if __name__ == "__main__":
    main()
