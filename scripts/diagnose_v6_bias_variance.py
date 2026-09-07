"""
Bias-variance decomposition: V2 vs V6.

Per-pixel over the val set:
  bias(x,y) = mean_over_samples(pred(x,y,t).mean(t) - target(x,y,t).mean(t))
  var_ratio(x,y) = median_over_samples(std_t(pred(x,y)) / std_t(target(x,y)))

Reports:
  1. Median bias inside ROI (near-zero expected for V6, positive for V2)
  2. Variance ratio in ACTIVE pixels (|target|>0.1) — expected 0.4-0.6 for V6 (smoothing)
  3. Variance ratio in BACKGROUND pixels (|target|<=0.1)
  4. Per-frame bias and variance ratio (temporal decomposition)
  5. Spatial bias map (saved as png)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.iq_velocity_dataset_v3 import create_iq_velocity_dataloaders_v3  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def run_model(model, batch, device, model_type="v6"):
    iq = batch["input"].to(device)
    with torch.no_grad():
        if model_type == "v2":
            # ComplexToRealUNet(x, high_res_inputs=None, log_prf=None)
            out = model(iq)
        else:
            dt = batch.get("dt_seconds", torch.tensor(0.002))
            if not isinstance(dt, torch.Tensor):
                dt = torch.tensor(dt)
            dt = dt.to(device)
            out = model(iq, dt)
    return out  # (B, 1, T, H, W)


def analyse_model(model, val_loader, device, model_type, n_samples, label):
    """Return per-sample arrays of bias and variance ratio."""
    bias_list = []
    varratio_active_list = []
    varratio_bg_list = []
    perframe_bias = []
    perframe_varratio = []

    for i, batch in enumerate(val_loader):
        if i >= n_samples:
            break

        target = batch["target"]  # (1, 1, T, H, W)
        mask = batch.get("mask", None)

        pred = run_model(model, batch, device, model_type)
        pred = pred.cpu().numpy()[0, 0]     # (T, H, W)
        tgt  = target.numpy()[0, 0]         # (T, H, W)
        if mask is not None:
            msk = mask.numpy()[0, 0, 0]     # (H, W)
        else:
            msk = np.ones(tgt.shape[1:], dtype=bool)

        # Temporal mean per pixel
        pred_mean_t = pred.mean(axis=0)     # (H, W)
        tgt_mean_t  = tgt.mean(axis=0)

        # Temporal std per pixel
        pred_std_t  = pred.std(axis=0)      # (H, W)
        tgt_std_t   = tgt.std(axis=0)

        # Bias map (inside mask)
        bias_map = (pred_mean_t - tgt_mean_t) * msk.astype(float)
        bias_list.append(bias_map)

        # Active / background split
        msk_bool    = msk.astype(bool)
        active_mask = (np.abs(tgt).max(axis=0) > 0.10) & msk_bool
        bg_mask     = (np.abs(tgt).max(axis=0) <= 0.10) & msk_bool

        # Variance ratio — avoid division by zero
        safe_tgt_std = np.where(tgt_std_t > 1e-6, tgt_std_t, np.nan)
        var_ratio = pred_std_t / safe_tgt_std

        if active_mask.any():
            varratio_active_list.append(np.nanmedian(var_ratio[active_mask]))
        if bg_mask.any():
            varratio_bg_list.append(np.nanmedian(var_ratio[bg_mask]))

        # Per-frame bias and var ratio
        for t in range(pred.shape[0]):
            p_t = pred[t][msk_bool]
            g_t = tgt[t][msk_bool]
            perframe_bias.append(float(np.mean(p_t - g_t)))
            g_std_t = float(np.std(g_t))
            p_std_t = float(np.std(p_t))
            perframe_varratio.append(p_std_t / g_std_t if g_std_t > 1e-6 else np.nan)

        name = batch.get("name", [f"s{i}"])[0]
        if i % 10 == 0:
            print(f"  [{label}] sample {i}: bias_med={np.nanmedian(bias_map[msk.astype(bool)]):.4f}  "
                  f"var_ratio_active={varratio_active_list[-1] if varratio_active_list else 'N/A':.3f}")

    return {
        "bias_maps": bias_list,
        "varratio_active": varratio_active_list,
        "varratio_bg": varratio_bg_list,
        "perframe_bias": np.array(perframe_bias),
        "perframe_varratio": np.array(perframe_varratio),
    }


def print_summary(results, label):
    bias_stacked = np.stack(results["bias_maps"], axis=0) if results["bias_maps"] else None
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if bias_stacked is not None:
        mean_bias = float(np.nanmedian(bias_stacked))
        std_bias  = float(np.nanstd(bias_stacked))
        print(f"  Spatial bias (median ± std): {mean_bias:+.4f} ± {std_bias:.4f}")
        print(f"    (near 0 expected for V6; positive/negative = systematic over/under-est)")

    if results["varratio_active"]:
        vra = float(np.median(results["varratio_active"]))
        print(f"  Variance ratio — ACTIVE pixels (|target|>0.1): {vra:.3f}")
        print(f"    (story predicts 0.4-0.6 for V6 due to smoothing)")
    if results["varratio_bg"]:
        vrb = float(np.median(results["varratio_bg"]))
        print(f"  Variance ratio — BACKGROUND pixels:            {vrb:.3f}")
        print(f"    (expected < 0.3 for collapsed background)")

    T = len(results["perframe_bias"]) // max(1, len(results["bias_maps"]))
    if T > 0:
        avg_bias = results["perframe_bias"].reshape(-1, T).mean(axis=0)
        avg_vr   = np.nanmean(results["perframe_varratio"].reshape(-1, T), axis=0)
        print(f"\n  Per-frame bias (mean over samples), first 10 frames:")
        for t in range(min(10, T)):
            bar = "#" * int(abs(avg_bias[t]) * 100)
            sign = "+" if avg_bias[t] >= 0 else "-"
            print(f"    t={t:02d}: bias={avg_bias[t]:+.4f}  var_ratio={avg_vr[t]:.3f}  {sign}{bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_v2", required=True, help="V2 checkpoint")
    parser.add_argument("--ckpt_v6", required=True, help="V6 checkpoint")
    parser.add_argument("--data_dir", default="data/fullscale_velocity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/diagnose_bias_variance")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load V2
    print(f"Loading V2 from {args.ckpt_v2}")
    import inspect
    from src.architectures.velocity_unet import ComplexToRealUNet
    ckpt_v2 = torch.load(args.ckpt_v2, map_location="cpu", weights_only=False)
    cfg_v2  = ckpt_v2["config"]  # flat dict for V2
    valid_v2 = set(inspect.signature(ComplexToRealUNet.__init__).parameters.keys()) - {"self"}
    mcfg_v2 = {k: v for k, v in cfg_v2.items() if k in valid_v2}
    model_v2 = ComplexToRealUNet(**mcfg_v2).eval().to(args.device)
    model_v2.load_state_dict(ckpt_v2["model_state_dict"], strict=False)

    # Load V6
    print(f"Loading V6 from {args.ckpt_v6}")
    from src.architectures.wavepinn_v6 import create_wavepinn_v6
    ckpt_v6 = torch.load(args.ckpt_v6, map_location="cpu", weights_only=False)
    cfg_v6  = ckpt_v6["config"]
    mcfg_v6 = dict(cfg_v6["model"]); mcfg_v6.pop("name", None)
    model_v6 = create_wavepinn_v6(config=mcfg_v6).eval().to(args.device)
    model_v6.load_state_dict(ckpt_v6["model_state_dict"], strict=False)

    # Build val loader
    _, val_loader, info = create_iq_velocity_dataloaders_v3(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=0,
        enable_prf_augmentation=False,
        upsampling_factor=cfg_v6["model"].get("upsampling_factor", 12),
        per_sample_iq_norm=True,
        per_sample_vel_norm=True,
    )
    print(f"Val samples: {info.get('num_val', '?')}\n")

    print("Analysing V2...")
    res_v2 = analyse_model(model_v2, val_loader, args.device, "v2", args.n_samples, "V2")

    print("\nAnalysing V6...")
    val_loader_v6 = create_iq_velocity_dataloaders_v3(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=0,
        enable_prf_augmentation=False,
        upsampling_factor=cfg_v6["model"].get("upsampling_factor", 12),
        per_sample_iq_norm=True,
        per_sample_vel_norm=True,
    )[1]
    res_v6 = analyse_model(model_v6, val_loader_v6, args.device, "v6", args.n_samples, "V6")

    print_summary(res_v2, "V2")
    print_summary(res_v6, "V6")

    # Comparison
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    if res_v2["varratio_active"] and res_v6["varratio_active"]:
        vr_v2 = float(np.median(res_v2["varratio_active"]))
        vr_v6 = float(np.median(res_v6["varratio_active"]))
        print(f"  Active var ratio — V2: {vr_v2:.3f}  V6: {vr_v6:.3f}")
        print(f"  Story: V6 smooths active region (var_ratio_v6 < var_ratio_v2)")
        print(f"  → Story {'CONFIRMED' if vr_v6 < vr_v2 else 'REFUTED'}")

    if res_v2["bias_maps"] and res_v6["bias_maps"]:
        b_v2 = float(np.nanmedian(np.stack(res_v2["bias_maps"])))
        b_v6 = float(np.nanmedian(np.stack(res_v6["bias_maps"])))
        print(f"\n  Spatial bias — V2: {b_v2:+.4f}  V6: {b_v6:+.4f}")
        print(f"  Story: V6 near-zero bias, V2 positive bias (overestimates)")
        print(f"  → V2 bias story {'CONFIRMED' if b_v2 > 0.005 else 'REFUTED'}")
        print(f"  → V6 bias story {'CONFIRMED' if abs(b_v6) < abs(b_v2) else 'REFUTED'}")

    # Save spatial bias map comparison
    if HAS_MPL and res_v2["bias_maps"] and res_v6["bias_maps"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        b_v2_mean = np.stack(res_v2["bias_maps"]).mean(axis=0)
        b_v6_mean = np.stack(res_v6["bias_maps"]).mean(axis=0)
        vmax = max(np.abs(b_v2_mean).max(), np.abs(b_v6_mean).max())
        im0 = axes[0].imshow(b_v2_mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[0].set_title(f'V2 bias map (med={b_v2:+.3f})')
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(b_v6_mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[1].set_title(f'V6 bias map (med={b_v6:+.3f})')
        plt.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        fig.savefig(out_dir / "bias_maps_v2_vs_v6.png", dpi=120)
        plt.close(fig)
        print(f"\n  Bias map saved to: {out_dir}/bias_maps_v2_vs_v6.png")


if __name__ == "__main__":
    main()
