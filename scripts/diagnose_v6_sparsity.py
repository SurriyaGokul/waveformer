"""
Deep sparsity diagnostic for WavePINN v6 plateau.

Investigates the hypothesis: "wave content is so spatiotemporally sparse that
even amplitude-weighted L1 cannot find a non-trivial gradient signal."

Reports (no checkpoint needed for dataset-side stats):

1. Histogram of |target| over the entire val set (cumulative percentiles)
2. Wave-active volume fraction at multiple thresholds
3. Per-frame mean|target| — is the wave concentrated in early frames as expected?
4. Per-frame max|target| — when does wave activity peak in time?
5. Spatial concentration: per-sample, what fraction of (H,W) area has wave activity?
6. Active-volume / total-volume ratio (the "needle in haystack" number)

If a checkpoint is provided, also reports:
7. Per-frame L1 (model vs target)
8. Per-frame mean|pred| (does model predict zero in active frames vs background frames?)
9. Wave-active vs background L1 split (the key metric)
10. Per-PRF breakdown if dt_seconds varies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.iq_velocity_dataset_v3 import create_iq_velocity_dataloaders_v3  # noqa: E402


def percentiles(values: np.ndarray, qs=(50, 75, 90, 95, 99, 99.5, 99.9)) -> dict:
    out = {}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(values, q))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Optional checkpoint to also report pred stats")
    parser.add_argument("--data_dir", default="data/fullscale_velocity")
    parser.add_argument("--upsampling_factor", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=999)
    args = parser.parse_args()

    # Build val loader (no augmentation)
    _, val_loader, info = create_iq_velocity_dataloaders_v3(
        data_dir=args.data_dir,
        batch_size=4,
        num_workers=0,
        enable_prf_augmentation=False,
        upsampling_factor=args.upsampling_factor,
        per_sample_iq_norm=True,
        per_sample_vel_norm=True,
    )
    print(f"Val samples: {info.get('num_val', '?')}")

    # Optional model
    model = None
    if args.ckpt is not None:
        from src.architectures.wavepinn_v6 import create_wavepinn_v6
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model_cfg = dict(ckpt["config"]["model"])
        model_cfg.pop("name", None)
        model = create_wavepinn_v6(config=model_cfg).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"Loaded ckpt: epoch {ckpt.get('epoch', '?')}")

    # ---- Aggregators ----------------------------------------------------
    all_target_abs: list = []          # for histogram
    per_frame_mean = None              # (T,)
    per_frame_max = None
    per_frame_pred_mean = None
    per_frame_l1 = None
    sample_count = 0

    # Wave-active fraction at thresholds
    thresholds = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50]
    active_voxels = {t: 0 for t in thresholds}
    total_voxels = 0

    # Per-sample spatial coverage at threshold 0.10
    per_sample_active_spatial: list = []

    # Wave-active vs background L1 (model only)
    err_active_sum = 0.0
    err_bg_sum = 0.0
    n_active = 0
    n_bg = 0

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if sample_count >= args.max_samples:
                break

            target = batch["target"].float()  # (B, 1, T, H, W)
            mask = batch["mask"].float()      # (B, 1, 1, H, W)
            B, _, T, H, W = target.shape
            mask_5d = mask.expand_as(target)
            mask_bool = mask_5d.bool()

            t_in_mask = target[mask_bool]  # 1D
            abs_t = t_in_mask.abs()

            # Cumulative sample for histogram (subsample to keep memory bounded)
            sub = abs_t.numpy()
            if sub.size > 200_000:
                idx = np.random.default_rng(0).choice(sub.size, 200_000, replace=False)
                sub = sub[idx]
            all_target_abs.append(sub)

            # Per-frame mean|target| within mask
            mask_per_frame = mask_5d.sum(dim=(0, 1, 3, 4)).clamp(min=1.0)  # (T,)
            sum_per_frame = (target.abs() * mask_5d).sum(dim=(0, 1, 3, 4))
            mean_pf = (sum_per_frame / mask_per_frame).numpy()
            max_pf = (target.abs() * mask_5d).amax(dim=(0, 1, 3, 4)).numpy()
            if per_frame_mean is None:
                per_frame_mean = mean_pf * B
                per_frame_max = max_pf * B
            else:
                per_frame_mean += mean_pf * B
                per_frame_max = np.maximum(per_frame_max, max_pf * B / max(B, 1))

            # Voxel counts at thresholds
            for thr in thresholds:
                active_voxels[thr] += int(((target.abs() > thr) * mask_5d).sum().item())
            total_voxels += int(mask_5d.sum().item())

            # Per-sample spatial coverage at thr=0.10 (any-time max over T)
            for b in range(B):
                t_b = target[b, 0]                # (T, H, W)
                mask_b = mask[b, 0, 0].bool()     # (H, W)
                # Spatial pixels active at any time
                spatial_active = (t_b.abs() > 0.10).any(dim=0)
                area_active = float((spatial_active & mask_b).sum().item())
                area_total = float(mask_b.sum().item())
                per_sample_active_spatial.append(area_active / max(area_total, 1.0))

            sample_count += B

            # If we have a model: predict and gather extra stats
            if model is not None:
                x = batch["input"].to(args.device).float()
                dt_seconds = batch["dt_seconds"].to(args.device)
                pred = model(x, dt_seconds=dt_seconds).float().cpu()
                T_pred = pred.shape[2]
                target_c = target[:, :, :T_pred]
                mask_c = mask_5d[:, :, :T_pred]

                err = (pred - target_c).abs() * mask_c

                # Per-frame mean error and pred magnitude
                pf_err = (err.sum(dim=(0, 1, 3, 4)) /
                          mask_c.sum(dim=(0, 1, 3, 4)).clamp(min=1.0)).numpy()
                pf_pmag = ((pred.abs() * mask_c).sum(dim=(0, 1, 3, 4)) /
                           mask_c.sum(dim=(0, 1, 3, 4)).clamp(min=1.0)).numpy()
                if per_frame_l1 is None:
                    per_frame_l1 = pf_err * B
                    per_frame_pred_mean = pf_pmag * B
                else:
                    per_frame_l1 += pf_err * B
                    per_frame_pred_mean += pf_pmag * B

                # Wave-active vs background error split (threshold 0.10)
                active = (target_c.abs() > 0.10) & mask_c.bool()
                bg = (target_c.abs() <= 0.10) & mask_c.bool()
                err_active_sum += err[active].sum().item()
                err_bg_sum += err[bg].sum().item()
                n_active += int(active.sum().item())
                n_bg += int(bg.sum().item())

    # ---- Dataset-side report -------------------------------------------
    all_t = np.concatenate(all_target_abs)
    pcts = percentiles(all_t)
    per_frame_mean /= max(sample_count, 1)
    per_frame_max /= max(1.0, 1.0)  # already running max

    print()
    print("=" * 78)
    print("DATASET TARGET SPARSITY")
    print("=" * 78)
    print(f"Val pixels analysed   : {all_t.size:,}")
    print(f"mean|target| (in mask): {all_t.mean():.4f}")
    print(f"std (in mask)         : {all_t.std():.4f}")
    print(f"max (in mask)         : {all_t.max():.4f}")
    print()
    print("Percentiles of |target| inside ROI mask:")
    for k, v in pcts.items():
        print(f"  {k:>7s} : {v:.4f}")
    print()
    print("Wave-active VOXEL FRACTION (volume share at threshold):")
    for thr in thresholds:
        frac = active_voxels[thr] / max(total_voxels, 1)
        ratio = (1.0 - frac) / max(frac, 1e-12)
        print(f"  |target| > {thr:.3f}  →  {frac*100:7.4f}%  "
              f"(background : active = {ratio:>9.0f} : 1)")
    print()
    print("Per-sample spatial coverage at threshold 0.10 (any time):")
    cov = np.array(per_sample_active_spatial)
    print(f"  mean : {cov.mean()*100:.2f}%")
    print(f"  med  : {np.median(cov)*100:.2f}%")
    print(f"  min  : {cov.min()*100:.2f}%   max : {cov.max()*100:.2f}%")
    print()
    print(f"Per-frame mean|target| (T={len(per_frame_mean)} frames):")
    for t, v in enumerate(per_frame_mean):
        bar = "#" * int(v * 200)
        print(f"  t={t:02d} : {v:.4f}  {bar}")

    # ---- Model-side report ---------------------------------------------
    if model is not None and per_frame_l1 is not None:
        per_frame_l1 /= max(sample_count, 1)
        per_frame_pred_mean /= max(sample_count, 1)

        print()
        print("=" * 78)
        print("MODEL PREDICTIONS")
        print("=" * 78)
        print()
        print("Per-frame L1 error and pred magnitude:")
        print(f"  {'t':>3s} {'L1':>8s} {'mean|pred|':>12s} {'mean|tgt|':>11s}  ratio_p_t")
        for t in range(len(per_frame_l1)):
            tgt = per_frame_mean[t] if t < len(per_frame_mean) else 0
            ratio = per_frame_pred_mean[t] / max(tgt, 1e-9)
            print(f"  {t:3d} {per_frame_l1[t]:8.4f} {per_frame_pred_mean[t]:12.4f} "
                  f"{tgt:11.4f}  {ratio:7.3f}")
        print()
        print("Active vs background error split (threshold 0.10):")
        print(f"  Wave-active pixels   : {n_active:>12,}  "
              f"L1 = {err_active_sum / max(n_active, 1):.4f}")
        print(f"  Background pixels    : {n_bg:>12,}  "
              f"L1 = {err_bg_sum / max(n_bg, 1):.4f}")
        active_share = n_active / max(n_active + n_bg, 1)
        print(f"  Active share of mask : {active_share*100:.4f}%")
        print()
        print("INTERPRETATION:")
        if err_bg_sum / max(n_bg, 1) < 0.01 and err_active_sum / max(n_active, 1) > 0.3:
            print("  ⚠ Model predicts background well but FAILS on wave-active pixels")
            print("    → Loss landscape rewards 'predict zero' on bg, gives no signal on wave")
        if per_frame_pred_mean.max() < 0.05:
            print("  ⚠ Model output magnitude << target magnitude — still collapsed")
        if per_frame_pred_mean.argmax() != per_frame_mean.argmax():
            print(f"  ⚠ Pred peak frame {per_frame_pred_mean.argmax()} ≠ "
                  f"target peak frame {per_frame_mean.argmax()}")

    print("=" * 78)


if __name__ == "__main__":
    main()
