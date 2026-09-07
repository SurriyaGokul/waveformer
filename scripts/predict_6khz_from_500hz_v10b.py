#!/usr/bin/env python3
"""
Virtual 6 kHz velocity prediction from consecutive 500 Hz IQ using WavePINN v10b.

This script mirrors the legacy publication export format so downstream analysis
scripts can read it directly:
  - velocity_6khz_<sample>.npy
  - gt_velocity_<sample>.npy
  - metadata_<sample>.json

Usage:
  /home/surriya_gokul/SWE_PINN/SWE_PINN/.venv/bin/python scripts/predict_6khz_from_500hz_v10b.py
  /home/surriya_gokul/SWE_PINN/SWE_PINN/.venv/bin/python scripts/predict_6khz_from_500hz_v10b.py --sample_name <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v10 import create_wavepinn_v10
from src.utils.deployment_scale import estimate_velocity_scale_from_iq

MODEL_PATH_DEFAULT = "SWE_PINN_outputs_wavepinn_v10b_best_model.pt"
DATA_META_PATH = "data/fullscale_velocity/metadata.json"
OUTPUT_DIR_DEFAULT = "outputs/v10b_predictions"
UPSAMPLING_FACTOR = 12


def _pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad spatial dims (H, W) to a multiple of `multiple` for UNet pooling."""
    *lead, h, w = arr.shape
    p_h = (multiple - h % multiple) % multiple
    p_w = (multiple - w % multiple) % multiple
    if p_h == 0 and p_w == 0:
        return arr, (0, 0)
    pad = [(0, 0)] * len(lead) + [(0, p_h), (0, p_w)]
    return np.pad(arr, pad, mode="constant"), (p_h, p_w)


def load_v10b_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = dict(ckpt.get("config", {}).get("model", {}))
    model_cfg.pop("name", None)
    model = create_wavepinn_v10(config=model_cfg).to(device).eval()

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    epoch = ckpt.get("epoch", "?")
    best_val = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  WavePINN v10b loaded: epoch={epoch}, best_val={best_val:.6f}, "
        f"params={n_params:,}"
    )
    return model


def load_consecutive_iq(
    sample_meta: Dict,
    start_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load 5 consecutive IQ frames and return (iq_consec, iq_full_window).

    Returns
    -------
    iq_consec : ndarray (5, H, W) complex64
        Clutter-filtered 5-frame window used as model input.
    iq_full_window : ndarray (H, W, T) complex64
        Full raw IQ window (pre-clutter-filter) for Loupas proxy scale.
    """
    raw = np.load(sample_meta["original_path"])  # (H_full, W_full, T)
    roi = sample_meta["roi"]
    win = sample_meta["window"]

    iq_win = raw[
        roi["min_row"] : roi["max_row"] + 1,
        roi["min_col"] : roi["max_col"] + 1,
        win[0] : win[1],
    ].astype(np.complex64)

    # Keep full window (pre-clutter-filter) for deployment scale estimation
    iq_full_window = iq_win.copy()  # (H, W, T)

    # Match training preprocessing: temporal mean subtraction (clutter filter)
    iq_win = iq_win - iq_win.mean(axis=2, keepdims=True)

    t_win = iq_win.shape[2]
    end = start_frame + 5
    if end > t_win:
        raise ValueError(f"start_frame={start_frame} + 5 > T_window={t_win}")

    iq_consec = np.stack([iq_win[:, :, start_frame + i] for i in range(5)], axis=0)
    return iq_consec, iq_full_window


@torch.no_grad()
def infer_one(
    model: torch.nn.Module,
    iq_consec: np.ndarray,
    gt_vel: np.ndarray,
    dt_seconds: float,
    device: torch.device,
    scale_mode: str = "loupas",
    iq_full_window: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Run inference and return (pred_mm_s, iq_scale, vel_scale_gt, vel_scale_loupas).

    Parameters
    ----------
    scale_mode : {"loupas", "gt"}
        Which scale to use for denormalising the model output:
        - "loupas": GT-free proxy from Loupas on the IQ window (deployment mode).
        - "gt": 99th-pct of |gt_vel| (evaluation mode, requires GT velocity).
    iq_full_window : ndarray (H, W, T_full) or None
        Full raw IQ window for computing the Loupas proxy scale.  If None,
        falls back to the 5-frame iq_consec window (shape (5, H, W)).
    """
    h, w = iq_consec.shape[1], iq_consec.shape[2]

    iq_scale = float(np.percentile(np.abs(iq_consec), 99))
    iq_scale = max(iq_scale, 1e-6)
    iq_norm = (iq_consec / iq_scale).astype(np.complex64)

    vel_scale_gt = float(np.percentile(np.abs(gt_vel), 99))
    vel_scale_gt = max(vel_scale_gt, 1e-6)

    # Compute Loupas proxy scale from full IQ window when available
    if iq_full_window is not None:
        # iq_full_window: (H, W, T_full) complex64 — pre-clutter-filter
        vel_scale_loupas = estimate_velocity_scale_from_iq(iq_full_window, dt_seconds)
    else:
        # Fallback: transpose iq_consec (5, H, W) → (H, W, 5)
        iq_for_scale = np.transpose(iq_consec, (1, 2, 0))
        vel_scale_loupas = estimate_velocity_scale_from_iq(iq_for_scale, dt_seconds)

    vel_scale = vel_scale_loupas if scale_mode == "loupas" else vel_scale_gt

    inp = np.stack([iq_norm.real, iq_norm.imag], axis=0).astype(np.float32)  # (2, 5, H, W)
    inp_pad, _ = _pad_to_multiple(inp, multiple=16)

    inp_t = torch.from_numpy(inp_pad).unsqueeze(0).to(device)  # (1, 2, 5, H_pad, W_pad)
    dt_t = torch.tensor([dt_seconds], dtype=torch.float32, device=device)

    pred = model(inp_t, dt_seconds=dt_t)  # (1, 1, T_dense, H_pad, W_pad)
    pred_np = pred[0, 0].detach().cpu().numpy()[:, :h, :w]
    pred_mm = pred_np * vel_scale

    return pred_mm.astype(np.float32), iq_scale, vel_scale_gt, vel_scale_loupas


def _all_metadata_entries(meta: Dict) -> List[Dict]:
    entries: List[Dict] = []
    for split in ("train", "val", "inference"):
        entries.extend(meta.get(split, []))
    return entries


def pick_samples(meta: Dict, sample_name: Optional[str], only_w0: bool = True) -> List[Dict]:
    all_entries = _all_metadata_entries(meta)
    by_name = {s.get("name"): s for s in all_entries if s.get("name")}

    if sample_name:
        if sample_name not in by_name:
            raise ValueError(f"Sample not found in metadata: {sample_name}")
        return [by_name[sample_name]]

    # Primary lookup: 500Hz entries in metadata (case-insensitive).
    samples = []
    for s in all_entries:
        freq = str(s.get("frequency", "")).lower()
        name = str(s.get("name", ""))
        if freq != "500hz" and "500hz" not in name.lower():
            continue
        if only_w0 and not name.endswith("_w0"):
            continue
        if name:
            samples.append(s)

    if samples:
        dedup = {s["name"]: s for s in samples}
        return list(dedup.values())

    # Fallback: discover names directly from inference files and map back to metadata.
    inf_dir = PROJECT_ROOT / "data" / "fullscale_velocity" / "inference"
    discovered = []
    for p in sorted(inf_dir.glob("sparse_*500Hz*.npy")):
        name = p.stem.replace("sparse_", "")
        if only_w0 and not name.endswith("_w0"):
            continue
        if name in by_name:
            discovered.append(by_name[name])

    dedup = {s["name"]: s for s in discovered}
    return list(dedup.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="WavePINN v10b: 500Hz to virtual 6kHz export")
    parser.add_argument("--model_path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--data_meta", default=DATA_META_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--sample_name", default=None)
    parser.add_argument(
        "--include_all_windows",
        action="store_true",
        help="Include all 500Hz samples (w0/w1/... and CIRS samples without window suffix).",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["loupas", "gt"],
        default="loupas",
        dest="scale_mode",
        help=(
            "Velocity denormalisation mode: 'loupas' (default, GT-free, deployment-safe) "
            "uses a Loupas proxy scale from the IQ window; 'gt' uses the 99th-pct of GT "
            "velocity (evaluation mode only — GT required)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_meta) as f:
        meta = json.load(f)

    samples = pick_samples(meta, args.sample_name, only_w0=not args.include_all_windows)
    if not samples:
        mode = "_w0" if not args.include_all_windows else "all"
        raise RuntimeError(f"No 500Hz samples found for mode={mode} in metadata or inference files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_v10b_model(args.model_path, device)

    dt_out_ms = (1.0 / 500.0 * 1e3) / UPSAMPLING_FACTOR
    fs_out_hz = 1.0 / (dt_out_ms * 1e-3)

    all_metrics: List[Dict] = []

    for idx, sm in enumerate(samples, start=1):
        name = sm["name"]
        print(f"[{idx:03d}/{len(samples):03d}] {name}")

        try:
            iq_consec, iq_full_window = load_consecutive_iq(sm, start_frame=args.start_frame)
        except Exception as exc:
            print(f"  SKIP IQ load failed: {exc}")
            continue

        gt_path = PROJECT_ROOT / "data" / "fullscale_velocity" / "inference" / f"velocity_{name}.npy"
        if not gt_path.exists():
            print(f"  SKIP GT missing: {gt_path}")
            continue

        gt_vel = np.load(str(gt_path)).astype(np.float32)
        dt_in_s = float(1.0 / sm["prf_hz"])

        pred_mm, iq_scale, vel_scale_gt, vel_scale_loupas = infer_one(
            model=model,
            iq_consec=iq_consec,
            gt_vel=gt_vel,
            dt_seconds=dt_in_s,
            device=device,
            scale_mode=args.scale_mode,
            iq_full_window=iq_full_window,
        )
        vel_scale = vel_scale_loupas if args.scale_mode == "loupas" else vel_scale_gt
        print(f"  scale_mode={args.scale_mode}  vel_scale_gt={vel_scale_gt:.4f}  vel_scale_loupas={vel_scale_loupas:.4f}")

        # Compare in native 500Hz intervals by block-averaging the dense output.
        n_blocks = min(gt_vel.shape[0], pred_mm.shape[0] // UPSAMPLING_FACTOR)
        pred_blocks = np.stack(
            [pred_mm[k * UPSAMPLING_FACTOR : (k + 1) * UPSAMPLING_FACTOR].mean(axis=0) for k in range(n_blocks)],
            axis=0,
        )
        gt_blocks = gt_vel[:n_blocks]

        mae_active = float(np.abs(pred_blocks - gt_blocks).mean())

        pred_flat = pred_mm.flatten()
        gt_flat = gt_vel.flatten()
        r_full = float(np.corrcoef(pred_flat, gt_flat)[0, 1]) if pred_flat.std() > 1e-8 and gt_flat.std() > 1e-8 else float("nan")

        pred_block_flat = pred_blocks.flatten()
        gt_block_flat = gt_blocks.flatten()
        r_active = (
            float(np.corrcoef(pred_block_flat, gt_block_flat)[0, 1])
            if pred_block_flat.std() > 1e-8 and gt_block_flat.std() > 1e-8
            else float("nan")
        )

        print(f"  MAE(block)={mae_active:.4f} mm/s  r_full={r_full:.4f}  r_block={r_active:.4f}")

        np.save(str(out_dir / f"velocity_6khz_{name}.npy"), pred_mm)
        np.save(str(out_dir / f"gt_velocity_{name}.npy"), gt_vel)

        sample_meta = {
            "name": name,
            "model": "WavePINN v10b",
            "approach": "consecutive_iq_frames",
            "start_frame": args.start_frame,
            "input_iq_frames": list(range(args.start_frame, args.start_frame + 5)),
            "input_iq_dt_ms": dt_in_s * 1e3,
            "upsampling_factor": UPSAMPLING_FACTOR,
            "output_dt_ms": dt_out_ms,
            "output_Fs_hz": fs_out_hz,
            "output_n_frames": int(pred_mm.shape[0]),
            "output_total_duration_ms": dt_out_ms * (pred_mm.shape[0] - 1),
            "velocity_unit": "mm/s",
            "vel_scale_mms": vel_scale,
            "iq_scale": iq_scale,
            "metrics": {
                "mae_block_mm_s": mae_active,
                "r_full": r_full,
                "r_block": r_active,
            },
            "tissue_properties": sm.get("tissue_properties", {}),
            "note": (
                "5 consecutive 500Hz IQ frames are mapped to 49 dense velocity frames "
                "using WavePINN v10b. Per-sample IQ and velocity percentile scaling "
                "follows the training/evaluation protocol."
            ),
        }

        with open(out_dir / f"metadata_{name}.json", "w") as f:
            json.dump(sample_meta, f, indent=2)

        all_metrics.append(
            {
                "name": name,
                "mae_block_mm_s": mae_active,
                "r_full": r_full,
                "r_block": r_active,
                "vel_scale_gt_mms": vel_scale_gt,
                "vel_scale_loupas_mms": vel_scale_loupas,
                "scale_mode": args.scale_mode,
            }
        )

    if not all_metrics:
        raise RuntimeError("No samples were processed successfully")

    summary = {
        "description": "WavePINN v10b predictions for 500Hz to virtual 6kHz",
        "model": "WavePINN v10b",
        "output_Fs_hz": fs_out_hz,
        "output_dt_ms": dt_out_ms,
        "start_frame": args.start_frame,
        "scale_mode": args.scale_mode,
        "sample_selection": "all_500Hz" if args.include_all_windows else "w0_only",
        "n_samples": len(all_metrics),
        "mean_mae_block_mm_s": float(np.nanmean([m["mae_block_mm_s"] for m in all_metrics])),
        "std_mae_block_mm_s": float(np.nanstd([m["mae_block_mm_s"] for m in all_metrics])),
        "mean_r_full": float(np.nanmean([m["r_full"] for m in all_metrics])),
        "mean_r_block": float(np.nanmean([m["r_block"] for m in all_metrics])),
        "vel_scale_ratio_median": float(np.nanmedian([
            m["vel_scale_loupas_mms"] / m["vel_scale_gt_mms"]
            for m in all_metrics if m["vel_scale_gt_mms"] > 0
        ])),
        "samples": all_metrics,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone")
    print(f"  Outputs: {out_dir}")
    print(f"  Samples: {len(all_metrics)}")


if __name__ == "__main__":
    main()
