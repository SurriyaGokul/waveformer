#!/usr/bin/env python3
"""
Benchmark inference time for all saved models.
Measures time for IQ data prediction only (not displacement calculation).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import glob
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Paths
SAVED_MODELS_DIR = '/home/surriya_gokul/SWE_PINN/SWE_PINN/saved_models'
DATA_DIR = '/home/surriya_gokul/SWE_PINN/SWE_PINN/data/CIRS_phantom_dataset'

# Number of warmup and benchmark runs
N_WARMUP = 10
N_BENCHMARK = 50


def prepare_input(iq_data, upsampling_factor):
    """Prepare input tensor from IQ data."""
    H, W, T = iq_data.shape
    data = iq_data.transpose(2, 0, 1)  # (T, H, W)
    
    sparse_indices = list(range(0, T, upsampling_factor))
    sparse_data = data[sparse_indices]
    
    # Normalize
    real = sparse_data.real
    imag = sparse_data.imag
    mean_real, std_real = real.mean(), real.std() + 1e-8
    mean_imag, std_imag = imag.mean(), imag.std() + 1e-8
    
    sparse_real_norm = (real - mean_real) / std_real
    sparse_imag_norm = (imag - mean_imag) / std_imag
    
    # Convert to tensor: (1, T_sparse, 2, 1, H, W)
    sparse_tensor = torch.stack([
        torch.from_numpy(sparse_real_norm).float().unsqueeze(1),
        torch.from_numpy(sparse_imag_norm).float().unsqueeze(1)
    ], dim=1).unsqueeze(0)
    
    return sparse_tensor


def benchmark_model(model, input_tensor, n_warmup=N_WARMUP, n_runs=N_BENCHMARK):
    """Benchmark model inference time."""
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'output_shape': output.shape
    }


def load_pinn_model(checkpoint_path):
    """Load PI-CV-RecUNet model."""
    from src.architectures.pinn_recunet import PI_CV_RecUNet
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PI_CV_RecUNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        num_levels=config['num_levels'],
        lstm_hidden_channels=config['lstm_hidden_channels'],
        lstm_num_layers=config['lstm_num_layers'],
        upsampling_factor=config['upsampling_factor']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config


def load_bilstm_model(checkpoint_path):
    """Load BiLSTM model."""
    from src.architectures.bilstm_model import PI_CV_RecUNet_BiLSTM
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Extract bottleneck channels from saved state dict
    state_dict = checkpoint['model_state_dict']
    # Find the bottleneck channel size from saved weights
    for key in state_dict:
        if 'temporal_bottleneck.output_proj.conv_real.weight' in key:
            bottleneck_channels = state_dict[key].shape[1]
            break
    else:
        bottleneck_channels = config['lstm_hidden_channels']
    
    model = PI_CV_RecUNet_BiLSTM(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        num_levels=config['num_levels'],
        lstm_hidden_channels=bottleneck_channels,
        lstm_num_layers=config['lstm_num_layers'],
        upsampling_factor=config['upsampling_factor']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config


def load_input_guided_model(checkpoint_path):
    """Load Input-Guided Motion-Aware model."""
    from src.training.train_input_guided import create_input_guided_model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_input_guided_model(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_channels=config['base_channels'],
        num_levels=config['num_levels'],
        lstm_hidden_channels=config['lstm_hidden_channels'],
        lstm_num_layers=config['lstm_num_layers'],
        upsampling_factor=config['upsampling_factor'],
        flow_hidden_channels=config['flow_hidden_channels'],
        flow_scale=config['flow_scale'],
        use_residual_fusion=config['use_residual_fusion'],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config


def main():
    print("=" * 70)
    print("INFERENCE TIME BENCHMARK")
    print("=" * 70)
    
    # Load a test sample
    test_file = glob.glob(os.path.join(DATA_DIR, 'IQData1_10.npy'))[0]
    iq_data = np.load(test_file)
    print(f"\nTest data: {os.path.basename(test_file)}")
    print(f"Shape: {iq_data.shape} (H, W, T)")
    print(f"Warmup runs: {N_WARMUP}, Benchmark runs: {N_BENCHMARK}")
    
    results = {}
    
    # Benchmark PINN models (Trials 2, 3, 4)
    pinn_models = [
        ('pinn_trial_2_best.pt', 'PI-CV-RecUNet (5×)', 5),
        ('pinn_trial_3_best.pt', 'PI-CV-RecUNet (10×)', 10),
        ('pinn_trial_4_best.pt', 'PI-CV-RecUNet (20×)', 20),
    ]
    
    for filename, name, upsampling in pinn_models:
        path = os.path.join(SAVED_MODELS_DIR, filename)
        if os.path.exists(path):
            print(f"\nLoading {name}...")
            try:
                model, config = load_pinn_model(path)
                input_tensor = prepare_input(iq_data, upsampling)
                
                print(f"  Input shape: {input_tensor.shape}")
                result = benchmark_model(model, input_tensor)
                result['upsampling'] = upsampling
                result['params'] = sum(p.numel() for p in model.parameters())
                results[name] = result
                
                print(f"  Output shape: {result['output_shape']}")
                print(f"  Inference time: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
                
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error: {e}")
    
    # Benchmark BiLSTM model
    bilstm_path = os.path.join(SAVED_MODELS_DIR, 'bilstm_best.pt')
    if os.path.exists(bilstm_path):
        print(f"\nLoading BiLSTM (20×)...")
        try:
            model, config = load_bilstm_model(bilstm_path)
            input_tensor = prepare_input(iq_data, 20)
            
            print(f"  Input shape: {input_tensor.shape}")
            result = benchmark_model(model, input_tensor)
            result['upsampling'] = 20
            result['params'] = sum(p.numel() for p in model.parameters())
            results['BiLSTM (20×)'] = result
            
            print(f"  Output shape: {result['output_shape']}")
            print(f"  Inference time: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error: {e}")
    
    # Benchmark Input-Guided Motion-Aware model
    ig_path = os.path.join(SAVED_MODELS_DIR, 'SWE_PINN_outputs_input_guided_physics_trial_2_best_model.pt')
    if os.path.exists(ig_path):
        print(f"\nLoading Input-Guided Motion-Aware (20×)...")
        try:
            model, config = load_input_guided_model(ig_path)
            input_tensor = prepare_input(iq_data, 20)
            
            print(f"  Input shape: {input_tensor.shape}")
            result = benchmark_model(model, input_tensor)
            result['upsampling'] = 20
            result['params'] = sum(p.numel() for p in model.parameters())
            results['Input-Guided Motion (20×)'] = result
            
            print(f"  Output shape: {result['output_shape']}")
            print(f"  Inference time: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Params':>12} {'Time (ms)':>15} {'FPS':>10} {'Output Frames':>15}")
    print("-" * 87)
    
    for name, r in results.items():
        params_str = f"{r['params']/1e6:.2f}M"
        time_str = f"{r['mean_ms']:.2f} ± {r['std_ms']:.2f}"
        fps = 1000 / r['mean_ms']
        output_frames = r['output_shape'][1]
        print(f"{name:<35} {params_str:>12} {time_str:>15} {fps:>10.1f} {output_frames:>15}")
    
    print("-" * 87)
    
    # Real-time analysis
    print("\n" + "=" * 70)
    print("REAL-TIME FEASIBILITY ANALYSIS")
    print("=" * 70)
    
    # Typical ultrasound PRF is 10-20 kHz, frame rate for SWE is ~100-10000 fps
    target_fps = [30, 60, 100, 500, 1000]
    
    print(f"\n{'Model':<35}", end="")
    for fps in target_fps:
        print(f"{fps} FPS".center(12), end="")
    print()
    print("-" * 95)
    
    for name, r in results.items():
        achievable_fps = 1000 / r['mean_ms']
        print(f"{name:<35}", end="")
        for fps in target_fps:
            if achievable_fps >= fps:
                print("✓".center(12), end="")
            else:
                print("✗".center(12), end="")
        print()
    
    print("\n✓ = Achievable, ✗ = Not achievable with current hardware")
    
    # Output shape analysis
    print("\n" + "=" * 70)
    print("THROUGHPUT ANALYSIS (frames/second)")
    print("=" * 70)
    
    for name, r in results.items():
        output_frames = r['output_shape'][1]
        throughput = output_frames * (1000 / r['mean_ms'])
        print(f"{name}: {throughput:.1f} output frames/second")
    
    # Additional analysis for the report
    print("\n" + "=" * 70)
    print("LATEX TABLE OUTPUT (for report)")
    print("=" * 70)
    
    print(r"""
\begin{table}[H]
\centering
\caption{Inference time benchmark on NVIDIA GeForce RTX 4070 Laptop GPU}
\label{tab:inference_time}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Upsampling} & \textbf{Parameters} & \textbf{Time (ms)} & \textbf{FPS} & \textbf{Output Frames} & \textbf{Throughput} \\
\midrule""")
    
    for name, r in results.items():
        upsampling = r['upsampling']
        params = r['params'] / 1e6
        time_ms = r['mean_ms']
        std_ms = r['std_ms']
        fps = 1000 / time_ms
        output_frames = r['output_shape'][1]
        throughput = output_frames * fps
        
        # Shorten name for table
        short_name = name.replace(' (', '\\\\(').replace('×)', '$\\times$)')
        print(f"{short_name} & {upsampling}$\\times$ & {params:.2f}M & {time_ms:.1f} $\\pm$ {std_ms:.1f} & {fps:.1f} & {output_frames} & {throughput:.0f} fps \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    return results


if __name__ == '__main__':
    results = main()
