#!/usr/bin/env python3
"""
WavePINN v9 — Evaluation Script.

Runs two evaluations from a single checkpoint:

1. PRIMARY EVAL (5 kHz val, 18 samples)
   Inputs: 5 kHz sparse IQ (native sparse_step=12, dt=2.4 ms)
   Targets: 5 kHz dense velocity
   Metrics: pixel MAE/RMSE/Pearson r, scalar Cs MAE (vs phantom GT), G' MAE
   Success criterion (G5): Cs MAE ≤ 0.15 m/s, val_pixel L1 ≤ 0.06

2. DEPLOYMENT PROBE (500 Hz val, 24 samples)
   Inputs: 5 consecutive 500 Hz IQ frames (dt=2.0 ms, as deployed)
   Targets: not available — SWE metrics derived from model output
   The 6 kHz virtual output (T_dense=49 frames at dt≈0.167 ms) is run
   through ToA SWE analysis and compared to phantom GT cs.
   Success criterion (G6): Cs MAE ≤ 0.30 m/s

Usage:
    python scripts/evaluate_wavepinn_v9.py \\
        --checkpoint saved_models/wavepinn_v9_best.pt \\
        --data_dir data/fullscale_velocity \\
        --output_dir outputs/wavepinn_v9_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.architectures.wavepinn_v9 import WavePINNv9, create_wavepinn_v9
from src.utils.iq_velocity_dataset_v3 import IQVelocityDatasetV3, collate_fn_v3

# Re-use SWE analysis functions from the existing pipeline
from scripts.swe_analysis import estimate_cs_toa, estimate_viscosity_voigt

# ---------------------------------------------------------------------------
# Physical constants (matching swe_analysis.py)
# ---------------------------------------------------------------------------
DX_M = 1.013e-4   # lateral m/px
DY_M = 5.07e-5    # axial   m/px
RHO  = 1000.0     # kg/m³

# Phantom ground-truth shear speeds (from metadata.json)
PHANTOM_CS_GT: Dict[str, float] = {
    "2ft":  1.41,
    "3ft":  1.73,
    "cirs": 4.47,
}


# ===========================================================================
# SWE metric computation from a velocity volume
# ===========================================================================

def compute_swe_from_velocity(
    velocity_mm_s: np.ndarray,  # (T, H, W) in mm/s
    dt_s: float,
    dx_m: float = DX_M,
) -> Dict[str, float]:
    """
    Run ToA + Voigt analysis on a (T, H, W) velocity volume.

    Returns dict with: cs_m_s, cs_std, G_prime_kPa, E_kPa, eta_Pa_s
    (or nan if analysis fails).
    """
    cs, cs_std, _, _ = estimate_cs_toa(velocity_mm_s, dt_s, dx_m)
    G_prime = np.nan
    E = np.nan
    eta = np.nan

    if not np.isnan(cs):
        G_prime = RHO * cs**2 / 1e3   # kPa
        E = 3.0 * G_prime

    try:
        G_p, _, eta_v, _, _, _ = estimate_viscosity_voigt(velocity_mm_s, dt_s, dx_m)
        if not np.isnan(G_p):
            G_prime = G_p
            E = 3.0 * G_prime
        if not np.isnan(eta_v):
            eta = eta_v
    except Exception:
        pass

    return {
        "cs_m_s": float(cs) if not np.isnan(cs) else float("nan"),
        "cs_std": float(cs_std) if not np.isnan(cs_std) else float("nan"),
        "G_prime_kPa": float(G_prime) if not np.isnan(G_prime) else float("nan"),
        "E_kPa": float(E) if not np.isnan(E) else float("nan"),
        "eta_Pa_s": float(eta) if not np.isnan(eta) else float("nan"),
    }


# ===========================================================================
# Pixel metrics
# ===========================================================================

def compute_pixel_metrics(
    pred: np.ndarray,    # (T, H, W)
    target: np.ndarray,  # (T, H, W)
    mask: Optional[np.ndarray] = None,  # (H, W) bool
) -> Dict[str, float]:
    """MAE, RMSE, Pearson r — on active frames (|target|>0.1 mm/s after denorm)."""
    if mask is not None:
        p = pred[:, mask]
        t = target[:, mask]
    else:
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)

    # Full field
    mae_full = float(np.mean(np.abs(p - t)))
    rmse_full = float(np.sqrt(np.mean((p - t) ** 2)))

    r_full = float("nan")
    pf = p.flatten()
    tf = t.flatten()
    if np.std(pf) > 1e-8 and np.std(tf) > 1e-8:
        r_full = float(np.corrcoef(pf, tf)[0, 1])

    # Active frames: frames where mean |target| > 0.1 mm/s
    active = np.mean(np.abs(t), axis=1) > 0.1   # (T,)
    if active.sum() > 0:
        pa = p[active].flatten()
        ta = t[active].flatten()
        mae_act = float(np.mean(np.abs(pa - ta)))
        rmse_act = float(np.sqrt(np.mean((pa - ta) ** 2)))
        r_act = float("nan")
        if np.std(pa) > 1e-8 and np.std(ta) > 1e-8:
            r_act = float(np.corrcoef(pa, ta)[0, 1])
    else:
        mae_act = float("nan")
        rmse_act = float("nan")
        r_act = float("nan")

    return {
        "mae_full_mm_s": mae_full,
        "rmse_full_mm_s": rmse_full,
        "r_full": r_full,
        "mae_act_mm_s": mae_act,
        "rmse_act_mm_s": rmse_act,
        "r_act": r_act,
        "n_active_frames": int(active.sum()),
    }


# ===========================================================================
# Inference helpers
# ===========================================================================

def load_model(checkpoint_path: str, device: torch.device) -> WavePINNv9:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = dict(config.get("model", {}))
    model_cfg.pop("name", None)
    model = create_wavepinn_v9(config=model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded v9 checkpoint: epoch={epoch}, best_val_loss={val_loss:.4f}")
    return model, config


@torch.no_grad()
def predict_batch(
    model: WavePINNv9,
    x: torch.Tensor,
    dt_seconds: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.to(device)
    dt_seconds = dt_seconds.to(device)
    if use_amp and device.type == "cuda":
        from torch.amp import autocast
        with autocast("cuda"):
            velocity, cs_scalar = model(x, dt_seconds=dt_seconds, detach_head=False)
    else:
        velocity, cs_scalar = model(x, dt_seconds=dt_seconds, detach_head=False)
    return velocity.cpu(), cs_scalar.cpu()


# ===========================================================================
# Primary Evaluation — 5 kHz val
# ===========================================================================

def evaluate_5khz(
    model: WavePINNv9,
    data_dir: str,
    config: Dict,
    device: torch.device,
    batch_size: int = 2,
) -> Dict:
    """Evaluate on 5kHz val split. Returns dict of aggregated results."""
    upsampling = config.get("model", {}).get("upsampling_factor", 12)

    dataset = IQVelocityDatasetV3(
        data_dir=data_dir,
        split="val",
        enable_prf_augmentation=False,
        upsampling_factor=upsampling,
        per_sample_iq_norm=config.get("data", {}).get("per_sample_iq_norm", True),
        per_sample_vel_norm=config.get("data", {}).get("per_sample_vel_norm", True),
        training_frequencies=["5kHz"],
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn_v3,
    )

    # Per-sample results
    sample_results: List[Dict] = []

    for batch in loader:
        x = batch["input"]
        target = batch["target"]
        mask = batch["mask"]
        dt_seconds = batch["dt_seconds"]
        cs_target = batch["cs_target_scalar"]
        vel_scale = batch["vel_scale"]
        names = batch["name"]
        sources = batch["source"]
        B = x.shape[0]

        velocity, cs_scalar = predict_batch(model, x, dt_seconds, device)

        for b in range(B):
            T_pred = velocity.shape[2]
            tgt_crop = target[b, 0, :T_pred].numpy()  # (T, H, W) normalised
            pred_b = velocity[b, 0].numpy()            # (T, H, W) normalised
            scale = vel_scale[b].item()
            mask_b = mask[b, 0, 0].numpy().astype(bool)  # (H, W)
            dt_s_b = dt_seconds[b].item() / upsampling   # output frame interval

            # Denormalise to mm/s
            pred_mm = pred_b * scale * 1000.0
            tgt_mm  = tgt_crop * scale * 1000.0

            # Pixel metrics
            pix = compute_pixel_metrics(pred_mm, tgt_mm, mask=mask_b)

            # SWE analysis on prediction
            swe = compute_swe_from_velocity(pred_mm, dt_s=dt_s_b)

            cs_head = float(cs_scalar[b].item())
            cs_gt = float(cs_target[b].item())

            sample_results.append({
                "name": names[b],
                "source": sources[b],
                "cs_gt": cs_gt,
                "cs_toa": swe["cs_m_s"],
                "cs_head": cs_head,
                "G_prime_kPa": swe["G_prime_kPa"],
                "E_kPa": swe["E_kPa"],
                "eta_Pa_s": swe["eta_Pa_s"],
                **pix,
            })

    return {"samples": sample_results}


# ===========================================================================
# Deployment Probe — 500 Hz consecutive inputs → 6 kHz virtual output
# ===========================================================================

def evaluate_500hz_deployment(
    model: WavePINNv9,
    data_dir: str,
    config: Dict,
    device: torch.device,
    batch_size: int = 2,
) -> Dict:
    """
    Deployment probe: use 500 Hz val samples with 5 consecutive frames (dt=2ms).

    The model produces 49 virtual frames at dt ≈ 0.167 ms (6 kHz equivalent).
    SWE analysis runs on the virtual 6 kHz sequence.
    """
    upsampling = config.get("model", {}).get("upsampling_factor", 12)
    dt_sparse = 1.0 / 500   # 2.0 ms between consecutive 500 Hz frames

    dataset = IQVelocityDatasetV3(
        data_dir=data_dir,
        split="val",
        enable_prf_augmentation=False,
        upsampling_factor=upsampling,
        per_sample_iq_norm=config.get("data", {}).get("per_sample_iq_norm", True),
        per_sample_vel_norm=config.get("data", {}).get("per_sample_vel_norm", True),
        training_frequencies=["500Hz"],
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fn_v3,
    )

    sample_results: List[Dict] = []

    for batch in loader:
        x = batch["input"]
        vel_scale = batch["vel_scale"]
        cs_target = batch["cs_target_scalar"]
        names = batch["name"]
        sources = batch["source"]
        B = x.shape[0]

        # Override dt_seconds to 2ms (consecutive 500 Hz frames)
        dt_override = torch.full((B,), dt_sparse, dtype=torch.float32)

        velocity, cs_scalar = predict_batch(model, x, dt_override, device)

        for b in range(B):
            scale = vel_scale[b].item()
            pred_mm = velocity[b, 0].numpy() * scale * 1000.0   # (T, H, W) mm/s
            dt_output = dt_sparse / upsampling   # ≈ 0.167 ms at 6 kHz

            swe = compute_swe_from_velocity(pred_mm, dt_s=dt_output)

            cs_head = float(cs_scalar[b].item())
            cs_gt = float(cs_target[b].item())

            sample_results.append({
                "name": names[b],
                "source": sources[b],
                "cs_gt": cs_gt,
                "cs_toa": swe["cs_m_s"],
                "cs_head": cs_head,
                "G_prime_kPa": swe["G_prime_kPa"],
                "E_kPa": swe["E_kPa"],
                "eta_Pa_s": swe["eta_Pa_s"],
            })

    return {"samples": sample_results}


# ===========================================================================
# Reporting
# ===========================================================================

def _phantom_label(source: str) -> str:
    s = source.lower()
    if "2ft" in s or "2 ft" in s:
        return "2ft"
    if "3ft" in s or "3 ft" in s:
        return "3ft"
    if "cirs" in s:
        return "cirs"
    return source


def aggregate_by_phantom(samples: List[Dict]) -> Dict[str, Dict]:
    """Group and aggregate results by phantom type."""
    groups: Dict[str, List] = {}
    for s in samples:
        label = _phantom_label(s.get("source", ""))
        groups.setdefault(label, []).append(s)

    out: Dict[str, Dict] = {}
    for label, group in groups.items():
        cs_toa = [s["cs_toa"] for s in group if not np.isnan(s.get("cs_toa", float("nan")))]
        cs_head = [s["cs_head"] for s in group]
        cs_gt_vals = [s["cs_gt"] for s in group if s.get("cs_gt", 0) > 0]
        G_prime = [s["G_prime_kPa"] for s in group if not np.isnan(s.get("G_prime_kPa", float("nan")))]
        E = [s["E_kPa"] for s in group if not np.isnan(s.get("E_kPa", float("nan")))]
        eta = [s["eta_Pa_s"] for s in group if not np.isnan(s.get("eta_Pa_s", float("nan")))]

        cs_gt_mean = float(np.mean(cs_gt_vals)) if cs_gt_vals else float("nan")

        cs_toa_mean = float(np.mean(cs_toa)) if cs_toa else float("nan")
        cs_head_mean = float(np.mean(cs_head)) if cs_head else float("nan")
        cs_toa_mae = float(np.mean(np.abs(np.array(cs_toa) - cs_gt_mean))) if cs_toa and not np.isnan(cs_gt_mean) else float("nan")
        cs_head_mae = float(np.mean(np.abs(np.array(cs_head) - cs_gt_mean))) if cs_head and not np.isnan(cs_gt_mean) else float("nan")

        mae_act_vals = [s["mae_act_mm_s"] for s in group if "mae_act_mm_s" in s and not np.isnan(s.get("mae_act_mm_s", float("nan")))]
        r_act_vals = [s["r_act"] for s in group if "r_act" in s and not np.isnan(s.get("r_act", float("nan")))]

        out[label] = {
            "n": len(group),
            "cs_gt": cs_gt_mean,
            "cs_toa_mean": cs_toa_mean,
            "cs_head_mean": cs_head_mean,
            "cs_toa_mae": cs_toa_mae,
            "cs_head_mae": cs_head_mae,
            "G_prime_mean": float(np.mean(G_prime)) if G_prime else float("nan"),
            "E_mean": float(np.mean(E)) if E else float("nan"),
            "eta_mean": float(np.mean(eta)) if eta else float("nan"),
            "mae_act_mm_s": float(np.mean(mae_act_vals)) if mae_act_vals else float("nan"),
            "r_act": float(np.mean(r_act_vals)) if r_act_vals else float("nan"),
        }
    return out


def print_report(
    eval_5khz: Dict,
    eval_500hz: Dict,
    gates: Dict,
    output_dir: Path,
):
    lines = []

    def row(text=""):
        lines.append(text)
        print(text)

    sep = "=" * 80

    row(sep)
    row("  WavePINN v9 — EVALUATION REPORT")
    row(sep)

    # --- Primary eval ---
    row()
    row("PRIMARY EVALUATION: 5 kHz val split")
    row("-" * 60)

    samples_5k = eval_5khz["samples"]
    agg_5k = aggregate_by_phantom(samples_5k)

    # Overall pixel metrics
    all_mae_act = [s["mae_act_mm_s"] for s in samples_5k if not np.isnan(s.get("mae_act_mm_s", float("nan")))]
    all_r_act   = [s["r_act"]        for s in samples_5k if not np.isnan(s.get("r_act", float("nan")))]
    all_cs_head_mae = []
    for s in samples_5k:
        if s.get("cs_gt", 0) > 0 and not np.isnan(s.get("cs_head", float("nan"))):
            all_cs_head_mae.append(abs(s["cs_head"] - s["cs_gt"]))

    overall_mae = float(np.mean(all_mae_act)) if all_mae_act else float("nan")
    overall_r   = float(np.mean(all_r_act)) if all_r_act else float("nan")
    overall_cs_head_mae = float(np.mean(all_cs_head_mae)) if all_cs_head_mae else float("nan")

    row(f"  Overall pixel MAE  (active, mm/s): {overall_mae:.4f}")
    row(f"  Overall Pearson r  (active)      : {overall_r:.3f}")
    row(f"  Overall Cs MAE (ScalarCsHead, m/s): {overall_cs_head_mae:.3f}")
    row()

    row("  Per-phantom breakdown:")
    row(f"  {'Phantom':8s} {'n':>4}  {'cs_gt':>6}  {'cs_head':>7}  {'cs_head_MAE':>11}  {'cs_toa':>6}  {'cs_toa_MAE':>10}  {'MAE_act':>8}  {'r_act':>6}")
    row("  " + "-" * 80)
    for label in sorted(agg_5k.keys()):
        ag = agg_5k[label]
        row(
            f"  {label:8s} {ag['n']:>4}  {ag['cs_gt']:>6.3f}  {ag['cs_head_mean']:>7.3f}  "
            f"{ag['cs_head_mae']:>11.3f}  {ag['cs_toa_mean']:>6.3f}  {ag['cs_toa_mae']:>10.3f}  "
            f"{ag['mae_act_mm_s']:>8.4f}  {ag['r_act']:>6.3f}"
        )

    row()
    # G5 gate
    g5_cs_thresh  = gates.get("G5_phase3_end_cs_mae_max", 0.15)
    g5_pix_thresh = gates.get("G5_phase3_end_val_pixel_loss_max", 0.06)
    g5_cs_pass  = "PASS" if (not np.isnan(overall_cs_head_mae) and overall_cs_head_mae <= g5_cs_thresh) else "FAIL"
    g5_pix_pass = "PASS" if (not np.isnan(overall_mae) and overall_mae / 1000.0 <= g5_pix_thresh) else "WARN"

    row(f"  [G5 Cs_head_MAE] {overall_cs_head_mae:.3f} m/s (target ≤{g5_cs_thresh:.2f} m/s): {g5_cs_pass}")
    row(f"  [G5 pixel_L1]   approx {overall_mae:.4f} mm/s (G5 pixel target is on normalised scale)")

    # --- Deployment probe ---
    row()
    row(sep)
    row("DEPLOYMENT PROBE: 500 Hz → 6 kHz virtual sequences")
    row("-" * 60)

    samples_500 = eval_500hz["samples"]
    agg_500 = aggregate_by_phantom(samples_500)

    all_cs_head_mae_500 = []
    for s in samples_500:
        if s.get("cs_gt", 0) > 0 and not np.isnan(s.get("cs_head", float("nan"))):
            all_cs_head_mae_500.append(abs(s["cs_head"] - s["cs_gt"]))
    all_cs_toa_mae_500 = []
    for s in samples_500:
        if s.get("cs_gt", 0) > 0 and not np.isnan(s.get("cs_toa", float("nan"))):
            all_cs_toa_mae_500.append(abs(s["cs_toa"] - s["cs_gt"]))

    overall_cs_head_mae_500 = float(np.mean(all_cs_head_mae_500)) if all_cs_head_mae_500 else float("nan")
    overall_cs_toa_mae_500  = float(np.mean(all_cs_toa_mae_500))  if all_cs_toa_mae_500  else float("nan")

    row(f"  Overall Cs MAE (ScalarCsHead, m/s): {overall_cs_head_mae_500:.3f}")
    row(f"  Overall Cs MAE (ToA from 6kHz virt): {overall_cs_toa_mae_500:.3f}")
    row()
    row(f"  {'Phantom':8s} {'n':>4}  {'cs_gt':>6}  {'cs_head':>7}  {'cs_head_MAE':>11}  {'cs_toa':>6}  {'cs_toa_MAE':>10}")
    row("  " + "-" * 70)
    for label in sorted(agg_500.keys()):
        ag = agg_500[label]
        row(
            f"  {label:8s} {ag['n']:>4}  {ag['cs_gt']:>6.3f}  {ag['cs_head_mean']:>7.3f}  "
            f"{ag['cs_head_mae']:>11.3f}  {ag['cs_toa_mean']:>6.3f}  {ag['cs_toa_mae']:>10.3f}"
        )

    row()
    g6_thresh = gates.get("G6_deployment_cs_mae_max", 0.30)
    g6_pass = "PASS" if (not np.isnan(overall_cs_head_mae_500) and overall_cs_head_mae_500 <= g6_thresh) else "FAIL"
    row(f"  [G6 Cs_head_MAE] {overall_cs_head_mae_500:.3f} m/s (target ≤{g6_thresh:.2f} m/s): {g6_pass}")

    # --- Per-sample detail ---
    row()
    row(sep)
    row("PER-SAMPLE DETAIL — 5 kHz")
    row("-" * 60)
    row(f"  {'Sample':<55}  {'cs_gt':>5}  {'cs_head':>7}  {'cs_toa':>6}  {'MAE_act':>8}  {'r_act':>6}")
    row("  " + "-" * 100)
    for s in samples_5k:
        row(
            f"  {s['name'][:55]:<55}  {s['cs_gt']:>5.2f}  {s['cs_head']:>7.3f}  "
            f"{s['cs_toa']:>6.3f}  {s.get('mae_act_mm_s', float('nan')):>8.4f}  "
            f"{s.get('r_act', float('nan')):>6.3f}"
        )

    row()
    row(sep)
    row("PER-SAMPLE DETAIL — 500 Hz deployment probe")
    row("-" * 60)
    row(f"  {'Sample':<55}  {'cs_gt':>5}  {'cs_head':>7}  {'cs_toa(6kHz)':>12}")
    row("  " + "-" * 85)
    for s in samples_500:
        row(
            f"  {s['name'][:55]:<55}  {s['cs_gt']:>5.2f}  {s['cs_head']:>7.3f}  "
            f"{s['cs_toa']:>12.3f}"
        )

    row()
    row(sep)

    report_path = output_dir / "evaluation_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")

    # Save raw results as JSON
    results = {
        "eval_5khz": {
            "samples": eval_5khz["samples"],
            "aggregate": agg_5k,
            "overall_cs_head_mae": overall_cs_head_mae,
            "overall_pixel_mae_act_mm_s": overall_mae,
            "overall_r_act": overall_r,
        },
        "eval_500hz": {
            "samples": eval_500hz["samples"],
            "aggregate": agg_500,
            "overall_cs_head_mae": overall_cs_head_mae_500,
            "overall_cs_toa_mae": overall_cs_toa_mae_500,
        },
    }
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results saved to {results_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WavePINN v9")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to v9 .pt checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/fullscale_velocity",
                        help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/wavepinn_v9_eval",
                        help="Where to write report and results")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, device)
    gates = config.get("verification_gates", {})

    print("\n--- Primary Evaluation: 5 kHz val ---")
    eval_5khz = evaluate_5khz(
        model, args.data_dir, config, device, batch_size=args.batch_size,
    )
    print(f"  Processed {len(eval_5khz['samples'])} samples")

    print("\n--- Deployment Probe: 500 Hz val ---")
    eval_500hz = evaluate_500hz_deployment(
        model, args.data_dir, config, device, batch_size=args.batch_size,
    )
    print(f"  Processed {len(eval_500hz['samples'])} samples")

    print_report(eval_5khz, eval_500hz, gates, output_dir)


if __name__ == "__main__":
    main()
