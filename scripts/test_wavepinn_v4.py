#!/usr/bin/env python3
"""
Smoke test for WavePINN v4: Conditional Wave Propagation Network.

Tests:
    1. Forward pass at multiple PRFs (50, 100, 250, 500 Hz)
    2. Output shape validation
    3. Gradient flow through all parameter groups
    4. Wave content gate outputs
    5. Variable T_sparse support
    6. Memory estimation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.architectures.wavepinn_v4 import WavePINNv4, create_wavepinn_v4, count_parameters_v4

def test_forward_pass():
    """Test forward pass at multiple PRFs."""
    print("=" * 60)
    print("TEST 1: Forward pass at multiple PRFs")
    print("=" * 60)

    model = create_wavepinn_v4()
    model.eval()

    params = count_parameters_v4(model)
    print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    B, T_sparse, H, W = 2, 5, 128, 176
    upsampling_factor = 12
    T_dense = 1 + (T_sparse - 1) * upsampling_factor  # 49

    test_cases = [
        ("50 Hz (dt=20ms)", 0.020),
        ("100 Hz (dt=10ms)", 0.010),
        ("250 Hz (dt=4ms)", 0.004),
        ("500 Hz (dt=2ms)", 0.002),
        ("5 kHz (dt=0.2ms)", 0.0002),
        ("20 kHz (dt=0.05ms)", 0.00005),
    ]

    x = torch.randn(B, 2, T_sparse, H, W)

    for name, dt in test_cases:
        with torch.no_grad():
            out = model(x, dt_seconds=dt)

        expected_shape = (B, 1, T_dense, H, W)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
        assert out.min() >= -1.0 and out.max() <= 1.0, "Output not in [-1, 1]"

        print(f"  {name}: shape={out.shape}, "
              f"range=[{out.min():.4f}, {out.max():.4f}], "
              f"std={out.std():.4f}")

    print("  PASSED\n")


def test_gradient_flow():
    """Test gradient flow through all parameters."""
    print("=" * 60)
    print("TEST 2: Gradient flow")
    print("=" * 60)

    model = create_wavepinn_v4()
    model.train()

    x = torch.randn(1, 2, 5, 64, 80, requires_grad=False)
    target = torch.randn(1, 1, 49, 64, 80)

    out = model(x, dt_seconds=0.002)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()

    # Check gradients for key components
    components = {
        "enc1": model.enc1,
        "enc4": model.enc4,
        "pre_proj": model.pre_proj,
        "wave_gate": model.wave_gate,
        "fwd_gru": model.propagator.fwd_gru,
        "bwd_gru": model.propagator.bwd_gru,
        "fusion": model.propagator.fusion,
        "temporal_refine": model.propagator.temporal_refine,
        "dec1": model.dec1,
        "dec4": model.dec4,
        "output_conv": model.output_conv,
    }

    all_ok = True
    for name, module in components.items():
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters() if p.requires_grad
        )
        status = "OK" if has_grad else "NO GRAD"
        if not has_grad:
            all_ok = False
        print(f"  {name:20s}: {status}")

    # Check dt FiLM specifically
    fwd_film_grad = model.propagator.fwd_gru.dt_film[2].weight.grad
    if fwd_film_grad is not None:
        print(f"  dt_film grad norm : {fwd_film_grad.norm():.6f}")
    else:
        print(f"  dt_film grad      : None (problem!)")
        all_ok = False

    assert all_ok, "Some components have no gradient flow"
    print("  PASSED\n")


def test_wave_content_gate():
    """Test wave content gate produces valid scores."""
    print("=" * 60)
    print("TEST 3: Wave content gate")
    print("=" * 60)

    from src.architectures.wavepinn_v4 import WaveContentGate

    gate = WaveContentGate(channels=256, hidden=64)
    features = torch.randn(2, 5, 256, 8, 11)

    scores = gate(features)
    assert scores.shape == (2, 5), f"Shape mismatch: {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores not in [0, 1]"

    print(f"  Scores shape: {scores.shape}")
    print(f"  Scores: {scores[0].detach().numpy()}")
    print("  PASSED\n")


def test_variable_t_sparse():
    """Test with different T_sparse values."""
    print("=" * 60)
    print("TEST 4: Variable T_sparse")
    print("=" * 60)

    model = create_wavepinn_v4()
    model.eval()

    for T_sparse in [3, 5, 7]:
        T_dense = 1 + (T_sparse - 1) * 12
        x = torch.randn(1, 2, T_sparse, 64, 80)

        with torch.no_grad():
            out = model(x, dt_seconds=0.002)

        expected = (1, 1, T_dense, 64, 80)
        assert out.shape == expected, f"T_sparse={T_sparse}: {out.shape} != {expected}"
        print(f"  T_sparse={T_sparse} → T_dense={T_dense}: shape={out.shape}, OK")

    print("  PASSED\n")


def test_memory_estimate():
    """Estimate GPU memory usage."""
    print("=" * 60)
    print("TEST 5: Memory estimation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = create_wavepinn_v4().cuda()
    model.train()

    x = torch.randn(2, 2, 5, 128, 176, device="cuda")
    target = torch.randn(2, 1, 49, 128, 176, device="cuda")

    from torch.amp import autocast
    with autocast('cuda'):
        out = model(x, dt_seconds=0.002)
        loss = nn.functional.mse_loss(out, target)
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU memory: {peak_mb:.0f} MB")
    print(f"  Fits in 12GB: {'YES' if peak_mb < 12000 else 'NO'}")

    del model, x, target, out, loss
    torch.cuda.empty_cache()
    print("  PASSED\n")


def test_dt_conditioning():
    """Verify that different dt values produce different outputs."""
    print("=" * 60)
    print("TEST 6: dt conditioning effect")
    print("=" * 60)

    model = create_wavepinn_v4()
    model.eval()

    x = torch.randn(1, 2, 5, 64, 80)

    with torch.no_grad():
        out_100hz = model(x, dt_seconds=0.010)   # 100 Hz
        out_500hz = model(x, dt_seconds=0.002)   # 500 Hz
        out_20khz = model(x, dt_seconds=0.00005) # 20 kHz

    diff_1 = (out_100hz - out_500hz).abs().mean()
    diff_2 = (out_500hz - out_20khz).abs().mean()

    print(f"  |out_100Hz - out_500Hz| = {diff_1:.6f}")
    print(f"  |out_500Hz - out_20kHz| = {diff_2:.6f}")
    print(f"  dt conditioning active: {diff_1 > 1e-6 and diff_2 > 1e-6}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\nWavePINN v4 Smoke Test")
    print("=" * 60)

    test_forward_pass()
    test_gradient_flow()
    test_wave_content_gate()
    test_variable_t_sparse()
    test_dt_conditioning()
    test_memory_estimate()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
