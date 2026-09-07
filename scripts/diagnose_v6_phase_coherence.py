"""
Per-column phase coherence diagnostic: V2 vs V6.

Tests the mechanistic story: V6 has better lateral phase coherence
(driving better cs via ToA) despite worse per-pixel amplitude correlation.

Algorithm:
  1. Load V2 + V6 predictions on val samples
  2. Depth-average velocity → v_avg(t, x) for pred + target
  3. Per lateral column: phase of first Fourier harmonic
     phase(x) = angle(fft(v_avg[:, x])[1])
  4. Plot phase_pred(x) vs phase_target(x) for V2 and V6
  5. Also compute: |d_phase/d_x| (proportional to 1/cs) for each
  6. Report phase RMSE and phase-gradient RMSE for V2 vs V6
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_model_v2(ckpt_path: str, device: str):
    from src.architectures.velocity_unet import ComplexToRealUNet
    import inspect
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]  # flat dict for V2
    valid_keys = set(inspect.signature(ComplexToRealUNet.__init__).parameters.keys()) - {"self"}
    model_cfg = {k: v for k, v in config.items() if k in valid_keys}
    model = ComplexToRealUNet(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model, config


def load_model_v6(ckpt_path: str, device: str):
    from src.architectures.wavepinn_v6 import create_wavepinn_v6
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_cfg = dict(config["model"])
    model_cfg.pop("name", None)
    model = create_wavepinn_v6(config=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model, config


def depth_averaged_velocity(vel: np.ndarray, depth_frac=(0.2, 0.8)) -> np.ndarray:
    """vel: (T, H, W) → v_avg: (T, W). Depth-average over 20-80% of H."""
    H = vel.shape[1]
    lo = int(H * depth_frac[0])
    hi = int(H * depth_frac[1])
    return vel[:, lo:hi, :].mean(axis=1)  # (T, W)


def per_column_phase(v_avg: np.ndarray) -> np.ndarray:
    """v_avg: (T, W) → phase: (W,). Phase of first Fourier harmonic per column."""
    V = np.fft.fft(v_avg, axis=0)  # (T, W)
    phase = np.angle(V[1, :])      # (W,)
    return phase


def phase_rmse(phase_pred: np.ndarray, phase_tgt: np.ndarray) -> float:
    """Circular phase difference RMSE."""
    diff = np.angle(np.exp(1j * (phase_pred - phase_tgt)))  # wrapped to [-pi, pi]
    return float(np.sqrt(np.mean(diff ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_v2", required=True)
    parser.add_argument("--ckpt_v6", required=True)
    parser.add_argument("--data_dir", default="data/fullscale_velocity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of val samples to analyse")
    parser.add_argument("--output_dir", default="outputs/diagnose_phase_coherence")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading V2 model from {args.ckpt_v2}")
    model_v2, config_v2 = load_model_v2(args.ckpt_v2, args.device)

    print(f"Loading V6 model from {args.ckpt_v6}")
    model_v6, config_v6 = load_model_v6(args.ckpt_v6, args.device)

    _, val_loader, info = create_iq_velocity_dataloaders_v3(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=0,
        enable_prf_augmentation=False,
        upsampling_factor=config_v6["model"].get("upsampling_factor", 12),
        per_sample_iq_norm=True,
        per_sample_vel_norm=True,
    )
    print(f"Val samples: {info.get('num_val', '?')}")

    phase_rmse_v2_list = []
    phase_rmse_v6_list = []
    grad_rmse_v2_list = []
    grad_rmse_v6_list = []

    for i, batch in enumerate(val_loader):
        if i >= args.n_samples:
            break

        iq = batch["input"].to(args.device)          # (1, 2, T_sparse, H, W)
        target = batch["target"].to(args.device)  # (1, 1, T_dense, H, W)
        mask = batch.get("mask", None)

        dt = batch.get("dt_seconds", torch.tensor(0.002)).to(args.device)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt)

        with torch.no_grad():
            # V2: ComplexToRealUNet(x, high_res_inputs=None, log_prf=None)
            pred_v2 = model_v2(iq)

            # V6 forward
            pred_v6 = model_v6(iq, dt)

        # Convert to numpy (T, H, W)
        tgt_np  = target[0, 0].cpu().numpy()          # (T, H, W)
        pred_v2_np = pred_v2[0, 0].cpu().numpy()
        pred_v6_np = pred_v6[0, 0].cpu().numpy()

        # Depth-average
        v_tgt  = depth_averaged_velocity(tgt_np)       # (T, W)
        v_v2   = depth_averaged_velocity(pred_v2_np)
        v_v6   = depth_averaged_velocity(pred_v6_np)

        # Per-column phase
        ph_tgt = per_column_phase(v_tgt)
        ph_v2  = per_column_phase(v_v2)
        ph_v6  = per_column_phase(v_v6)

        # Phase RMSE
        rmse_v2 = phase_rmse(ph_v2, ph_tgt)
        rmse_v6 = phase_rmse(ph_v6, ph_tgt)
        phase_rmse_v2_list.append(rmse_v2)
        phase_rmse_v6_list.append(rmse_v6)

        # Phase gradient (d_phase/d_x) — proxy for 1/cs
        grad_tgt = np.diff(ph_tgt)
        grad_v2  = np.diff(ph_v2)
        grad_v6  = np.diff(ph_v6)

        # Circular difference for gradients
        grad_diff_v2 = np.angle(np.exp(1j * (grad_v2 - grad_tgt)))
        grad_diff_v6 = np.angle(np.exp(1j * (grad_v6 - grad_tgt)))
        grad_rmse_v2_list.append(float(np.sqrt(np.mean(grad_diff_v2 ** 2))))
        grad_rmse_v6_list.append(float(np.sqrt(np.mean(grad_diff_v6 ** 2))))

        name = batch.get("name", [f"sample_{i}"])[0]
        print(f"\nSample {i}: {name}")
        print(f"  Phase RMSE  — V2: {rmse_v2:.4f} rad   V6: {rmse_v6:.4f} rad  "
              f"({'V6 better' if rmse_v6 < rmse_v2 else 'V2 better'})")
        print(f"  GradΦ RMSE  — V2: {grad_rmse_v2_list[-1]:.4f} rad/px   "
              f"V6: {grad_rmse_v6_list[-1]:.4f} rad/px  "
              f"({'V6 better' if grad_rmse_v6_list[-1] < grad_rmse_v2_list[-1] else 'V2 better'})")

        # Amplitude comparison
        amp_tgt = np.abs(np.fft.fft(v_tgt, axis=0)[1, :])
        amp_v2  = np.abs(np.fft.fft(v_v2, axis=0)[1, :])
        amp_v6  = np.abs(np.fft.fft(v_v6, axis=0)[1, :])
        print(f"  Amp ratio   — V2/tgt: {(amp_v2.mean()/amp_tgt.mean()):.3f}   "
              f"V6/tgt: {(amp_v6.mean()/amp_tgt.mean()):.3f}  "
              f"(story predicts V6 has lower amp ratio)")

        # Plot
        if HAS_MPL and i < 5:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            xs = np.arange(len(ph_tgt))
            axes[0].plot(xs, ph_tgt, 'k-', lw=2, label='Target')
            axes[0].plot(xs, ph_v2, 'b--', lw=1.5, label=f'V2 (RMSE={rmse_v2:.3f}r)')
            axes[0].plot(xs, ph_v6, 'r-.', lw=1.5, label=f'V6 (RMSE={rmse_v6:.3f}r)')
            axes[0].set_xlabel('Lateral column (px)')
            axes[0].set_ylabel('Phase of F[1] (rad)')
            axes[0].set_title(f'{name} — Per-column phase (first harmonic)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(xs[:-1], np.diff(ph_tgt), 'k-', lw=2, label='Target ∂φ/∂x')
            axes[1].plot(xs[:-1], np.diff(ph_v2), 'b--', lw=1.5, label='V2')
            axes[1].plot(xs[:-1], np.diff(ph_v6), 'r-.', lw=1.5, label='V6')
            axes[1].set_xlabel('Lateral column (px)')
            axes[1].set_ylabel('Phase gradient (rad/px)')
            axes[1].set_title('Phase gradient ∝ 1/cs')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(out_dir / f"phase_coherence_{i:02d}_{Path(name).stem}.png", dpi=120)
            plt.close(fig)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"Phase RMSE (rad):      V2 = {np.mean(phase_rmse_v2_list):.4f} ± {np.std(phase_rmse_v2_list):.4f}"
          f"   V6 = {np.mean(phase_rmse_v6_list):.4f} ± {np.std(phase_rmse_v6_list):.4f}")
    print(f"Phase gradient RMSE:   V2 = {np.mean(grad_rmse_v2_list):.4f} ± {np.std(grad_rmse_v2_list):.4f}"
          f"   V6 = {np.mean(grad_rmse_v6_list):.4f} ± {np.std(grad_rmse_v6_list):.4f}")

    story_confirmed_phase = np.mean(phase_rmse_v6_list) < np.mean(phase_rmse_v2_list)
    story_confirmed_grad  = np.mean(grad_rmse_v6_list)  < np.mean(grad_rmse_v2_list)
    print(f"\nStory: 'V6 has better phase coherence than V2'")
    print(f"  Phase RMSE  → story {'CONFIRMED' if story_confirmed_phase else 'REFUTED'}")
    print(f"  Grad RMSE   → story {'CONFIRMED' if story_confirmed_grad else 'REFUTED'}")
    if HAS_MPL:
        print(f"\nPlots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
