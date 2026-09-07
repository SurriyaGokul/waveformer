"""
Smoke test for WavePINN v3.

Validates:
    1. Forward pass at multiple PRFs (50 Hz, 100 Hz, 250 Hz, 500 Hz)
    2. Correct output shapes
    3. Gradient flow (all parameters have gradients)
    4. Physics loss with dt_seconds
    5. Memory usage estimate
    6. Parameter count

Usage:
    python scripts/test_wavepinn_v3.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, ".")

from src.architectures.wavepinn_v3 import WavePINNv3, create_wavepinn_v3, count_parameters_v3
from src.losses.physics_loss_v3 import EnhancedVelocityLossV3, PhysicsAwareWaveRegularizer


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_forward_pass():
    """Test forward pass at multiple PRFs."""
    divider("1. Forward Pass at Multiple PRFs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_wavepinn_v3().to(device)

    B, T_sparse, H, W = 2, 5, 128, 160
    upsampling_factor = 12
    T_dense_expected = 1 + (T_sparse - 1) * upsampling_factor  # 49

    prfs = [50, 100, 250, 500]

    for prf in prfs:
        dt = 1.0 / prf
        x = torch.randn(B, 2, T_sparse, H, W, device=device)

        with torch.no_grad():
            out = model(x, dt_seconds=dt)

        assert out.shape == (B, 1, T_dense_expected, H, W), (
            f"PRF={prf}Hz: expected {(B, 1, T_dense_expected, H, W)}, got {out.shape}"
        )
        assert out.min() >= -1.0 and out.max() <= 1.0, (
            f"PRF={prf}Hz: output range [{out.min():.3f}, {out.max():.3f}] outside [-1, 1]"
        )
        print(f"  PRF={prf:4d} Hz  dt={dt*1000:.2f} ms  output={tuple(out.shape)}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]  OK")

    print("  All PRFs passed.")
    return model, device


def test_gradient_flow(model: nn.Module, device: torch.device):
    """Verify gradients flow to all parameters."""
    divider("2. Gradient Flow")

    model.train()
    x = torch.randn(1, 2, 5, 64, 80, device=device, requires_grad=False)
    target = torch.randn(1, 1, 49, 64, 80, device=device)

    out = model(x, dt_seconds=0.002)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()

    no_grad = []
    has_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or p.grad.abs().sum() == 0:
                no_grad.append(name)
            else:
                has_grad += 1

    print(f"  Parameters with gradients: {has_grad}")
    if no_grad:
        print(f"  WARNING: {len(no_grad)} parameters without gradients:")
        for n in no_grad[:10]:
            print(f"    - {n}")
    else:
        print("  All parameters have gradients. OK")

    model.zero_grad()


def test_physics_loss():
    """Test PhysicsAwareWaveRegularizer and EnhancedVelocityLossV3."""
    divider("3. Physics Loss (dt-aware)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test wave regularizer directly
    wave_reg = PhysicsAwareWaveRegularizer(init_cs=3.0).to(device)
    pred = torch.randn(2, 1, 20, 64, 80, device=device)

    for prf_out in [6000, 1200, 500]:
        dt = 1.0 / prf_out
        loss = wave_reg(pred, dt)
        print(f"  WaveReg PRF_out={prf_out}Hz  dt={dt*1000:.3f}ms  loss={loss.item():.4f}  "
              f"cs={wave_reg.cs_value:.3f} m/s")

    # Test composite loss
    criterion = EnhancedVelocityLossV3(
        l1_weight=1.0, mse_weight=0.5, gdl_weight=0.3,
        temporal_weight=0.2, ssim_weight=0.3, wave_weight=0.1,
    ).to(device)

    target = torch.randn(2, 1, 20, 64, 80, device=device)
    mask = torch.ones(2, 1, 1, 64, 80, device=device)

    total, components = criterion(pred, target, mask=mask, dt_seconds=1.0/6000)
    print(f"\n  Composite loss: {total.item():.4f}")
    for k, v in components.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Verify cs is learnable (use criterion's internal wave_reg)
    total.backward()
    cs_param = criterion.wave_reg.log_cs
    assert cs_param.grad is not None, "cs should have gradients"
    print(f"\n  cs gradient: {cs_param.grad.item():.6f}  OK")


def test_parameter_count():
    """Count parameters and compare with budget."""
    divider("4. Parameter Count")

    model = create_wavepinn_v3()
    counts = count_parameters_v3(model)
    print(f"  Total:         {counts['total']:>10,}")
    print(f"  Trainable:     {counts['trainable']:>10,}")
    print(f"  Non-trainable: {counts['non_trainable']:>10,}")

    # Per-module breakdown
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:30s}: {n:>10,}")


def test_memory():
    """Estimate peak memory at batch_size=2 with AMP."""
    divider("5. Memory Usage (batch_size=2, AMP)")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping memory test.")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = create_wavepinn_v3().to(device)
    criterion = EnhancedVelocityLossV3(wave_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=1e-4,
    )
    scaler = torch.amp.GradScaler('cuda')

    B, T_sparse, H, W = 2, 5, 128, 160

    x = torch.randn(B, 2, T_sparse, H, W, device=device)
    target = torch.randn(B, 1, 49, H, W, device=device)
    mask = torch.ones(B, 1, 1, H, W, device=device)

    # Warmup
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        out = model(x, dt_seconds=0.002)
        loss, _ = criterion(out, target, mask=mask, dt_seconds=0.002/12)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    print(f"  Peak allocated: {peak_mb:.0f} MB")
    print(f"  Peak reserved:  {reserved_mb:.0f} MB")
    if peak_mb < 12000:
        print(f"  Fits in 12 GB GPU. OK")
    else:
        print(f"  WARNING: Exceeds 12 GB budget!")

    del model, criterion, optimizer, x, target, mask
    torch.cuda.empty_cache()


def test_variable_t_sparse():
    """Test that the model handles different T_sparse values."""
    divider("6. Variable T_sparse")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_wavepinn_v3().to(device)
    model.eval()

    for T_sp in [2, 3, 5, 7]:
        x = torch.randn(1, 2, T_sp, 64, 80, device=device)
        T_dense_expected = 1 + (T_sp - 1) * 12
        with torch.no_grad():
            out = model(x, dt_seconds=0.002)
        assert out.shape[2] == T_dense_expected, (
            f"T_sparse={T_sp}: expected T_dense={T_dense_expected}, got {out.shape[2]}"
        )
        print(f"  T_sparse={T_sp} -> T_dense={out.shape[2]}  OK")


def main():
    print("WavePINN v3 Smoke Test")
    print("=" * 60)

    try:
        model, device = test_forward_pass()
        test_gradient_flow(model, device)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        test_physics_loss()
        test_parameter_count()
        test_memory()
        test_variable_t_sparse()

        divider("ALL TESTS PASSED")

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
