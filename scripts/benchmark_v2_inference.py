#!/usr/bin/env python3
"""
Benchmark inference time for ComplexToRealUNet v2 model.

Measures GPU and CPU inference latency with proper warmup, and reports
parameter counts and output shape.
"""

import sys
import time

import numpy as np
import torch

# Ensure the project root is on the path
sys.path.insert(0, "/home/surriya_gokul/SWE_PINN/SWE_PINN")

from src.architectures.velocity_unet import create_velocity_model, count_parameters


CHECKPOINT_PATH = "/home/surriya_gokul/SWE_PINN/SWE_PINN/saved_models/fullscale_velocity_v2_best.pt"
INPUT_SHAPE = (1, 2, 5, 128, 176)  # (B, real/imag, T_sparse, H, W)
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50


def build_model():
    """Create model with v2 config."""
    config = dict(
        in_channels=1,
        base_channels=32,
        encoder_depths=[32, 64, 128, 256],
        upsampling_factor=12,
        use_motion_interpolation=True,
        complex_to_real_mode="concat",
        use_naive_bn=True,
        flow_hidden_channels=32,
        flow_scale=0.1,
        use_prf_conditioning=False,
    )
    model = create_velocity_model(config)
    return model


def load_checkpoint(model, device):
    """Load checkpoint weights, handling DataParallel prefix if present."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Strip 'module.' prefix if saved from DataParallel
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=False)
    return model


def benchmark_device(model, device, tag):
    """Run warmup + timed inference on the given device, return (mean_ms, std_ms, output_shape)."""
    model = model.to(device).eval()
    x = torch.randn(*INPUT_SHAPE, device=device)

    use_cuda = device.type == "cuda"

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS):
            if use_cuda:
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                out = model(x)
                end = time.perf_counter()
            times.append((end - start) * 1000.0)  # ms

    times = np.array(times)
    mean_ms = times.mean()
    std_ms = times.std()
    median_ms = np.median(times)
    min_ms = times.min()
    max_ms = times.max()
    p5 = np.percentile(times, 5)
    p95 = np.percentile(times, 95)
    # Trimmed stats: remove top/bottom 10%
    trimmed = times[(times >= np.percentile(times, 10)) & (times <= np.percentile(times, 90))]
    trimmed_mean = trimmed.mean() if len(trimmed) > 0 else mean_ms
    trimmed_std = trimmed.std() if len(trimmed) > 0 else std_ms
    output_shape = tuple(out.shape)

    print(f"\n{'='*60}")
    print(f"  {tag} Inference Benchmark")
    print(f"{'='*60}")
    print(f"  Device          : {device}")
    print(f"  Input shape     : {INPUT_SHAPE}")
    print(f"  Output shape    : {output_shape}")
    print(f"  Warmup runs     : {WARMUP_RUNS}")
    print(f"  Benchmark runs  : {BENCHMARK_RUNS}")
    print(f"  Latency (mean)  : {mean_ms:.2f} ms")
    print(f"  Latency (std)   : {std_ms:.2f} ms")
    print(f"  Latency (median): {median_ms:.2f} ms")
    print(f"  Latency (min)   : {min_ms:.2f} ms")
    print(f"  Latency (max)   : {max_ms:.2f} ms")
    print(f"  Latency (p5-p95): {p5:.2f} - {p95:.2f} ms")
    print(f"  Trimmed mean    : {trimmed_mean:.2f} +/- {trimmed_std:.2f} ms (10-90th pct)")
    print(f"  Latency         : {mean_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"{'='*60}")

    return mean_ms, std_ms, trimmed_mean, trimmed_std, output_shape


def main():
    print("Building ComplexToRealUNet v2 model...")
    model = build_model()

    # Parameter count
    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total       : {params['total']:,}")
    print(f"  Trainable   : {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")

    # Load checkpoint
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")
    model = load_checkpoint(model, device_gpu)
    print("Checkpoint loaded successfully.")

    # GPU benchmark
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU detected: {gpu_name}")
        gpu_mean, gpu_std, gpu_tmean, gpu_tstd, out_shape = benchmark_device(model, device_gpu, f"GPU ({gpu_name})")
    else:
        print("\nNo CUDA GPU available — skipping GPU benchmark.")
        gpu_mean, gpu_std, gpu_tmean, gpu_tstd, out_shape = None, None, None, None, None

    # CPU benchmark
    device_cpu = torch.device("cpu")
    cpu_mean, cpu_std, cpu_tmean, cpu_tstd, out_shape_cpu = benchmark_device(model, device_cpu, "CPU")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Model            : ComplexToRealUNet v2")
    print(f"  Parameters       : {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"  Input shape      : {INPUT_SHAPE}")
    print(f"  Output shape     : {out_shape or out_shape_cpu}")
    if gpu_mean is not None:
        print(f"  GPU inference    : {gpu_mean:.2f} +/- {gpu_std:.2f} ms (raw)")
        print(f"  GPU inference    : {gpu_tmean:.2f} +/- {gpu_tstd:.2f} ms (trimmed)")
    print(f"  CPU inference    : {cpu_mean:.2f} +/- {cpu_std:.2f} ms (raw)")
    print(f"  CPU inference    : {cpu_tmean:.2f} +/- {cpu_tstd:.2f} ms (trimmed)")
    if gpu_tmean is not None:
        print(f"  GPU/CPU speedup  : {cpu_tmean / gpu_tmean:.1f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
