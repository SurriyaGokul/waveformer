"""
WavePINN v6 plateau diagnostic.

Tests whether the val L1 ~0.0646 wall is:
  (a) model collapse (output ~0)             → val L1 of zero pred
  (b) temporal-mean fit                      → val L1 of per-pixel temporal mean
  (c) linear-temporal-interp limit           → val L1 of sparse-keyframe bilinear interp
  (d) actual model performance               → val L1 of model.forward(...)

All metrics computed inside the ROI mask, in fp32, single pass over val set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.wavepinn_v6 import create_wavepinn_v6  # noqa: E402
from src.utils.iq_velocity_dataset_v3 import create_iq_velocity_dataloaders_v3  # noqa: E402


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask_5d: torch.Tensor) -> float:
    """Mean(|pred - target| * mask) / mean(mask)."""
    err = (pred - target).abs() * mask_5d
    denom = mask_5d.sum().clamp(min=1.0)
    return (err.sum() / denom).item()


def masked_mean_abs(x: torch.Tensor, mask_5d: torch.Tensor) -> float:
    return ((x.abs() * mask_5d).sum() / mask_5d.sum().clamp(min=1.0)).item()


def masked_std(x: torch.Tensor, mask_5d: torch.Tensor) -> float:
    """Population std of x within the mask."""
    n = mask_5d.sum().clamp(min=1.0)
    mean = (x * mask_5d).sum() / n
    var = (((x - mean) ** 2) * mask_5d).sum() / n
    return var.sqrt().item()


def sparse_linear_interp_baseline(target: torch.Tensor, upsampling_factor: int) -> torch.Tensor:
    """
    Take target at sparse keyframes (every `upsampling_factor` frames),
    bilinear-interpolate temporally back to dense T.

    Upper bound on what any pure temporal-interpolation model can achieve.
    target: (B, 1, T, H, W) → returns same shape.
    """
    B, C, T, H, W = target.shape
    keyframe_idx = list(range(0, T, upsampling_factor))
    if keyframe_idx[-1] != T - 1:
        keyframe_idx.append(T - 1)
    sparse = target[:, :, keyframe_idx]  # (B, 1, T_sparse, H, W)
    flat = sparse.reshape(B * C, sparse.shape[2], H * W).permute(0, 2, 1)  # (B, H*W, T_sparse)
    flat = flat.reshape(B * C * H * W, 1, sparse.shape[2])
    interp = F.interpolate(flat, size=T, mode="linear", align_corners=True)
    interp = interp.reshape(B * C, H * W, T).permute(0, 2, 1).reshape(B, C, T, H, W)
    return interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pt")
    parser.add_argument("--data_dir", default=None, help="Override data dir if needed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"best_val_loss {ckpt.get('best_val_loss', '?')}")

    # ---- Build val loader (no augmentation, train normalisation) -----
    data_cfg = config["data"]
    if args.data_dir:
        data_cfg["data_dir"] = args.data_dir

    _, val_loader, info = create_iq_velocity_dataloaders_v3(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg.get("batch_size", 2),
        num_workers=0,  # deterministic, single-process
        enable_prf_augmentation=data_cfg.get("enable_prf_augmentation", True),
        augmentation_steps=data_cfg.get("augmentation_steps"),
        augmentation_step_weights=data_cfg.get("augmentation_step_weights"),
        upsampling_factor=config["model"].get("upsampling_factor", 12),
        per_sample_iq_norm=data_cfg.get("per_sample_iq_norm", True),
        per_sample_vel_norm=data_cfg.get("per_sample_vel_norm", True),
    )
    print(f"Val samples: {info.get('num_val', '?')}")

    # ---- Build model + load weights ---------------------------------
    model_cfg = dict(config["model"])
    model_cfg.pop("name", None)
    model = create_wavepinn_v6(config=model_cfg).to(args.device)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARNING missing keys: {len(missing)}  (first: {missing[:3]})")
    if unexpected:
        print(f"WARNING unexpected keys: {len(unexpected)}  (first: {unexpected[:3]})")
    model.eval()

    upsampling_factor = config["model"].get("upsampling_factor", 12)

    # ---- Aggregators ------------------------------------------------
    sums = {
        "l1_model": 0.0,
        "l1_zero": 0.0,
        "l1_temporal_mean": 0.0,
        "l1_sparse_linear": 0.0,
        "abs_pred": 0.0,
        "abs_target": 0.0,
        "std_pred": 0.0,
        "std_target": 0.0,
        "pred_min": 0.0,
        "pred_max": 0.0,
        "target_min": 0.0,
        "target_max": 0.0,
    }
    n_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch["input"].to(args.device).float()
            target = batch["target"].to(args.device).float()
            mask = batch["mask"].to(args.device).float()
            dt_seconds = batch["dt_seconds"].to(args.device)
            B = x.shape[0]

            pred = model(x, dt_seconds=dt_seconds).float()
            T_pred = pred.shape[2]
            target_c = target[:, :, :T_pred]

            # mask is (B, 1, 1, H, W) → expand to (B, 1, T, H, W)
            if mask.dim() == 5 and mask.shape[2] == 1:
                mask_5d = mask.expand_as(pred)
            else:
                mask_5d = mask

            # Baselines
            zero = torch.zeros_like(pred)
            tmean = (target_c * mask_5d).sum(dim=2, keepdim=True) / \
                    mask_5d.sum(dim=2, keepdim=True).clamp(min=1.0)
            tmean = tmean.expand_as(target_c)
            sparse_lin = sparse_linear_interp_baseline(target_c, upsampling_factor)

            # Metrics
            sums["l1_model"]         += B * masked_l1(pred,       target_c, mask_5d)
            sums["l1_zero"]          += B * masked_l1(zero,       target_c, mask_5d)
            sums["l1_temporal_mean"] += B * masked_l1(tmean,      target_c, mask_5d)
            sums["l1_sparse_linear"] += B * masked_l1(sparse_lin, target_c, mask_5d)

            sums["abs_pred"]   += B * masked_mean_abs(pred,     mask_5d)
            sums["abs_target"] += B * masked_mean_abs(target_c, mask_5d)
            sums["std_pred"]   += B * masked_std(pred,     mask_5d)
            sums["std_target"] += B * masked_std(target_c, mask_5d)

            sums["pred_min"]   += B * pred[mask_5d.bool()].min().item()
            sums["pred_max"]   += B * pred[mask_5d.bool()].max().item()
            sums["target_min"] += B * target_c[mask_5d.bool()].min().item()
            sums["target_max"] += B * target_c[mask_5d.bool()].max().item()

            n_samples += B
            if batch_idx == 0:
                print(f"  batch 0 shape: pred={tuple(pred.shape)}, target={tuple(target_c.shape)}, "
                      f"mask={tuple(mask_5d.shape)}")

    avg = {k: v / max(n_samples, 1) for k, v in sums.items()}

    # ---- Report -----------------------------------------------------
    print()
    print("=" * 70)
    print(f"DIAGNOSTIC REPORT  ({n_samples} val samples)")
    print("=" * 70)
    print()
    print("VAL L1 (masked, lower = better):")
    print(f"  Model prediction         : {avg['l1_model']:.6f}")
    print(f"  Zero baseline            : {avg['l1_zero']:.6f}   "
          f"(predicting 0 everywhere)")
    print(f"  Per-pixel temporal mean  : {avg['l1_temporal_mean']:.6f}   "
          f"(oracle: pred = target.mean(time))")
    print(f"  Sparse-keyframe linear   : {avg['l1_sparse_linear']:.6f}   "
          f"(oracle: 5 GT keyframes → linear interp)")
    print()
    print("OUTPUT MAGNITUDE (within ROI mask):")
    print(f"  mean|pred|   : {avg['abs_pred']:.4f}    "
          f"mean|target| : {avg['abs_target']:.4f}    "
          f"ratio: {avg['abs_pred'] / max(avg['abs_target'], 1e-9):.3f}")
    print(f"  std(pred)    : {avg['std_pred']:.4f}    "
          f"std(target)  : {avg['std_target']:.4f}    "
          f"ratio: {avg['std_pred'] / max(avg['std_target'], 1e-9):.3f}")
    print(f"  pred range   : [{avg['pred_min']:.3f}, {avg['pred_max']:.3f}]")
    print(f"  target range : [{avg['target_min']:.3f}, {avg['target_max']:.3f}]")
    print()
    print("INTERPRETATION:")
    if abs(avg["l1_model"] - avg["l1_zero"]) < 0.005:
        print("  ⚠ Model L1 ≈ Zero baseline → MODEL OUTPUT COLLAPSED to ~0")
    elif abs(avg["l1_model"] - avg["l1_temporal_mean"]) < 0.005:
        print("  ⚠ Model L1 ≈ Temporal-mean → model predicts time-invariant pattern")
    elif abs(avg["l1_model"] - avg["l1_sparse_linear"]) < 0.005:
        print("  ⚠ Model L1 ≈ Sparse-linear oracle → model = linear interp upper bound")
    elif avg["l1_model"] < avg["l1_sparse_linear"]:
        print("  ✓ Model L1 < sparse-linear oracle → model genuinely beats linear interp")
    else:
        print("  ⚠ Model L1 > sparse-linear oracle → suboptimal use of temporal info")

    if avg["std_pred"] / max(avg["std_target"], 1e-9) < 0.5:
        print(f"  ⚠ Output std collapsed: {avg['std_pred']:.3f} vs target {avg['std_target']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
