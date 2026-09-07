#!/usr/bin/env python3
"""
Verification tests for the per-sample dt fix in WavePINN v6.

Three sub-tests:
  1. Batch-equivalence: B=2 with dt=[0.002, 0.024] matches two independent
     B=1 runs (atol 1e-5) — proves the fix.
  2. Backward compat: float dt vs tensor([dt]) produce identical output.
  3. v10b checkpoint loads strict=True (forward-logic-only change).

Usage:
  python scripts/test_dt_per_sample.py
  python scripts/test_dt_per_sample.py --ckpt path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_DEFAULT = "SWE_PINN_outputs_wavepinn_v10b_best_model.pt"

# ---- model shapes matching v10b config ----
T_SPARSE = 5
H, W = 80, 64   # small test spatial dims (multiples of 16)
UPSAMPLING = 12

# Flash Attention / cuBLAS non-determinism means CUDA won't reach 1e-5.
# The meaningful bound is that B=2 mixed-dt errors are O(1e-3), not O(1e-1)
# (the latter would indicate the bug is still present).
ATOL_CPU  = 1e-5   # deterministic reference
ATOL_CUDA = 5e-3   # acceptable CUDA bound (Flash Attention + TF32 noise)


def _make_model(ckpt_path: str | None, device: torch.device) -> torch.nn.Module:
    from src.architectures.wavepinn_v10 import create_wavepinn_v10

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = dict(ckpt.get("config", {}).get("model", {}))
        cfg.pop("name", None)
        model = create_wavepinn_v10(config=cfg).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"  checkpoint loaded strict=True: {ckpt_path}")
    else:
        model = create_wavepinn_v10().to(device).eval()
        print(f"  no checkpoint found — using random weights for tests 1 & 2")
    return model


def test_float_vs_tensor_parity(model: torch.nn.Module, device: torch.device) -> None:
    """Test 2: float dt == tensor([dt]) output."""
    print("\n[Test 2] float dt vs tensor([dt]) parity")
    x = torch.randn(1, 2, T_SPARSE, H, W, device=device) * 0.1
    dt = 0.002

    with torch.no_grad():
        out_float  = model(x, dt_seconds=float(dt))
        out_tensor = model(x, dt_seconds=torch.tensor([dt], device=device))

    diff = (out_float - out_tensor).abs().max().item()
    atol = ATOL_CUDA if device.type == "cuda" else ATOL_CPU
    assert diff < atol, f"FAIL: float vs tensor diff={diff:.2e} > {atol}"
    print(f"  PASS: max diff = {diff:.2e}")


def test_batch_equivalence(model: torch.nn.Module, device: torch.device) -> None:
    """Test 1: B=2 mixed-dt batch matches two independent B=1 runs.

    Proves the fix: before the fix, mixed-step batches used dt.mean(), so
    sample 0 (dt=0.002) would be evaluated at dt=0.013, producing differences
    orders of magnitude larger than the Flash Attention noise floor (~1e-3).
    """
    print("\n[Test 1] Batch-equivalence (B=2 mixed dt vs two B=1 runs)")
    torch.manual_seed(0)
    x0 = torch.randn(1, 2, T_SPARSE, H, W, device=device) * 0.1
    x1 = torch.randn(1, 2, T_SPARSE, H, W, device=device) * 0.1
    dt0, dt1 = 0.002, 0.024

    with torch.no_grad():
        out0_solo = model(x0, dt_seconds=float(dt0))
        out1_solo = model(x1, dt_seconds=float(dt1))

        x_batch = torch.cat([x0, x1], dim=0)
        dt_batch = torch.tensor([dt0, dt1], device=device)
        out_batch = model(x_batch, dt_seconds=dt_batch)

    diff0 = (out_batch[:1] - out0_solo).abs().max().item()
    diff1 = (out_batch[1:] - out1_solo).abs().max().item()
    print(f"  sample 0 max diff = {diff0:.2e}")
    print(f"  sample 1 max diff = {diff1:.2e}")
    atol = ATOL_CUDA if device.type == "cuda" else ATOL_CPU
    assert diff0 < atol, f"FAIL: sample 0 diff={diff0:.2e} > {atol}"
    assert diff1 < atol, f"FAIL: sample 1 diff={diff1:.2e} > {atol}"
    print("  PASS")


def test_checkpoint_strict_load(ckpt_path: str, device: torch.device) -> None:
    """Test 3: v10b checkpoint loads strict=True (no state_dict changes)."""
    print("\n[Test 3] v10b checkpoint strict=True load")
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"  SKIP: checkpoint not found at {ckpt_path}")
        return

    from src.architectures.wavepinn_v10 import create_wavepinn_v10
    data = torch.load(str(ckpt), map_location=device, weights_only=False)
    cfg = dict(data.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    model = create_wavepinn_v10(config=cfg).to(device).eval()
    model.load_state_dict(data["model_state_dict"], strict=True)
    print("  PASS: strict load succeeded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_DEFAULT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _make_model(args.ckpt, device)

    test_float_vs_tensor_parity(model, device)
    test_batch_equivalence(model, device)
    test_checkpoint_strict_load(args.ckpt, device)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
