"""
Visualization script for Magnitude-Weighted Velocity Loss.

This script demonstrates how the magnitude weighting helps ignore "salt and pepper"
noise in the background of Kasai velocity maps.

The problem:
- In regions where the signal is dark (no echo), the phase is random
- Calculating phase shift of random noise → random high-velocity values
- This creates "salt and pepper" noise pattern in velocity maps

The solution:
- Weight velocity error by signal magnitude
- Bright pixels (high echo) → high weight → model must match velocity
- Dark pixels (noise) → low weight → model ignores the noise

Usage:
    python scripts/visualize_magnitude_weighted_loss.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.input_guided_data_loader import (
    compute_magnitude_mask,
    complex_to_tensor,
    clean_data,
    ComplexNormalizer
)


def compute_kasai_velocity(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Kasai velocity from complex IQ tensor.
    
    Velocity = angle(z_{t+1} * conj(z_t))
    
    Args:
        tensor: Complex tensor (T, 2, 1, H, W) where dim 1 is [real, imag]
    
    Returns:
        velocity: Phase velocity (T-1, H, W)
    """
    eps = 1e-8
    
    # Extract real and imaginary: (T, 2, 1, H, W)
    real = tensor[:, 0, 0, :, :]  # (T, H, W)
    imag = tensor[:, 1, 0, :, :]  # (T, H, W)
    
    # Consecutive frames
    real_t = real[:-1]
    imag_t = imag[:-1]
    real_t1 = real[1:]
    imag_t1 = imag[1:]
    
    # z_{t+1} * conj(z_t) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i
    interference_real = real_t1 * real_t + imag_t1 * imag_t
    interference_imag = imag_t1 * real_t - real_t1 * imag_t
    
    # Velocity is the angle
    velocity = torch.atan2(interference_imag + eps, interference_real + eps)
    
    return velocity


def visualize_magnitude_weighted_loss(data_path: str = None, output_dir: str = "results/magnitude_weighted_demo"):
    """
    Visualize how magnitude weighting affects velocity loss calculation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find a sample data file
    if data_path is None:
        # Try to find data in various locations
        search_dirs = [
            "data/CIRS_Phantom_Latest",
            "data/2FT(na100)/10mmfocus",
            "data/3FT(na100)/10mmfocus",
        ]
        
        for search_dir in search_dirs:
            search_path = PROJECT_ROOT / search_dir
            if search_path.exists():
                npy_files = list(search_path.rglob("*.npy"))
                if npy_files:
                    data_path = str(npy_files[0])
                    print(f"Found data file: {data_path}")
                    break
    
    if data_path is None or not Path(data_path).exists():
        # Create synthetic example if no real data found
        print("Creating synthetic example (no real data found)...")
        create_synthetic_visualization(output_dir)
        return
    
    # Load and process data
    print(f"Loading: {data_path}")
    data = np.load(data_path)
    data = clean_data(data)
    
    normalizer = ComplexNormalizer(method='per_sample')
    data = normalizer.normalize(data)
    
    tensor = complex_to_tensor(data)  # (T, 2, 1, H, W)
    
    # Compute magnitude mask
    magnitude_mask = compute_magnitude_mask(tensor, normalize=True)  # (T, 1, H, W)
    
    # Compute Kasai velocity
    velocity = compute_kasai_velocity(tensor)  # (T-1, H, W)
    
    # Create figure showing the effect of magnitude weighting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select a middle frame
    t_mid = velocity.shape[0] // 2
    
    # Row 1: Raw data
    # IQ Magnitude
    iq_magnitude = magnitude_mask[t_mid, 0].numpy()
    im1 = axes[0, 0].imshow(iq_magnitude, cmap='gray', aspect='auto')
    axes[0, 0].set_title('IQ Signal Magnitude\n(Where there is echo/signal)', fontsize=12)
    axes[0, 0].set_xlabel('Width')
    axes[0, 0].set_ylabel('Depth')
    plt.colorbar(im1, ax=axes[0, 0], label='Normalized magnitude')
    
    # Raw velocity (with noise)
    vel_raw = velocity[t_mid].numpy()
    vmax = np.percentile(np.abs(vel_raw), 99)
    im2 = axes[0, 1].imshow(vel_raw, cmap='RdBu_r', aspect='auto', 
                            vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Raw Kasai Velocity\n(Note: salt & pepper noise in background)', fontsize=12)
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Depth')
    plt.colorbar(im2, ax=axes[0, 1], label='Phase velocity (rad)')
    
    # Magnitude-weighted velocity visualization
    weighted_vel = vel_raw * iq_magnitude
    im3 = axes[0, 2].imshow(weighted_vel, cmap='RdBu_r', aspect='auto',
                            vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('Magnitude-Weighted Velocity\n(Background noise suppressed)', fontsize=12)
    axes[0, 2].set_xlabel('Width')
    axes[0, 2].set_ylabel('Depth')
    plt.colorbar(im3, ax=axes[0, 2], label='Weighted velocity')
    
    # Row 2: Loss demonstration
    # Simulated prediction error
    noise_amplitude = vmax * 0.3
    prediction_error = np.random.randn(*vel_raw.shape).astype(np.float32) * noise_amplitude
    
    # Unweighted error (treats all pixels equally)
    im4 = axes[1, 0].imshow(np.abs(prediction_error), cmap='hot', aspect='auto')
    axes[1, 0].set_title('Unweighted Error |pred - target|\n(Background errors counted equally)', fontsize=12)
    axes[1, 0].set_xlabel('Width')
    axes[1, 0].set_ylabel('Depth')
    plt.colorbar(im4, ax=axes[1, 0], label='Error')
    
    # Magnitude-weighted error
    weighted_error = np.abs(prediction_error) * iq_magnitude
    im5 = axes[1, 1].imshow(weighted_error, cmap='hot', aspect='auto')
    axes[1, 1].set_title('Magnitude-Weighted Error\n(Focus on signal regions)', fontsize=12)
    axes[1, 1].set_xlabel('Width')
    axes[1, 1].set_ylabel('Depth')
    plt.colorbar(im5, ax=axes[1, 1], label='Weighted error')
    
    # Weight visualization
    im6 = axes[1, 2].imshow(iq_magnitude, cmap='viridis', aspect='auto')
    axes[1, 2].set_title('Magnitude Mask (Weights)\nBright = High Weight, Dark = Low Weight', fontsize=12)
    axes[1, 2].set_xlabel('Width')
    axes[1, 2].set_ylabel('Depth')
    plt.colorbar(im6, ax=axes[1, 2], label='Weight (0-1)')
    
    plt.tight_layout()
    
    save_path = output_dir / "magnitude_weighted_loss_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Create additional figure explaining the loss function
    create_loss_explanation_figure(output_dir)
    
    print(f"\n✓ Visualization complete! Check {output_dir}/")


def create_synthetic_visualization(output_dir: Path):
    """Create synthetic example to demonstrate the concept."""
    print("Generating synthetic demonstration...")
    
    # Create synthetic ultrasound-like data
    H, W = 100, 128
    
    # Magnitude: simulate tissue (bright) and background (dark)
    y_coords = np.linspace(0, 1, H)
    x_coords = np.linspace(0, 1, W)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Simulate tissue region (elliptical)
    tissue_mask = ((X - 0.5)**2 / 0.15 + (Y - 0.5)**2 / 0.2) < 1
    magnitude = np.zeros((H, W), dtype=np.float32)
    magnitude[tissue_mask] = 0.7 + 0.3 * np.random.rand(tissue_mask.sum())
    magnitude += 0.05 * np.random.rand(H, W)  # Small background
    magnitude = np.clip(magnitude, 0, 1)
    
    # Simulate velocity: real wave inside tissue, noise everywhere
    velocity_true = np.zeros((H, W), dtype=np.float32)
    # Real wave pattern in tissue
    wave_pattern = np.sin(2 * np.pi * X * 5) * np.cos(2 * np.pi * Y * 3)
    velocity_true[tissue_mask] = wave_pattern[tissue_mask]
    
    # Salt and pepper noise in background (Kasai noise)
    noise = np.random.randn(H, W).astype(np.float32) * 2
    velocity_noisy = velocity_true.copy()
    velocity_noisy[~tissue_mask] = noise[~tissue_mask]
    velocity_noisy[tissue_mask] += noise[tissue_mask] * 0.1  # Small noise in tissue too
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    im1 = axes[0, 0].imshow(magnitude, cmap='gray', aspect='auto')
    axes[0, 0].set_title('IQ Signal Magnitude\n(Tissue appears bright)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], label='Magnitude')
    
    vmax = 2.5
    im2 = axes[0, 1].imshow(velocity_noisy, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Kasai Velocity with Noise\n(Background: salt & pepper)', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1], label='Velocity')
    
    weighted = velocity_noisy * magnitude
    im3 = axes[0, 2].imshow(weighted, cmap='RdBu_r', aspect='auto', vmin=-vmax * 0.5, vmax=vmax * 0.5)
    axes[0, 2].set_title('Magnitude-Weighted Velocity\n(Noise suppressed)', fontsize=12)
    plt.colorbar(im3, ax=axes[0, 2], label='Weighted velocity')
    
    # Row 2: Loss demonstration
    pred_error = np.random.randn(H, W).astype(np.float32) * 0.5
    
    im4 = axes[1, 0].imshow(np.abs(pred_error), cmap='hot', aspect='auto')
    axes[1, 0].set_title('Unweighted Loss\n(All pixels equal)', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 0], label='Loss')
    
    weighted_loss = np.abs(pred_error) * magnitude
    im5 = axes[1, 1].imshow(weighted_loss, cmap='hot', aspect='auto')
    axes[1, 1].set_title('Magnitude-Weighted Loss\n(Focus on tissue)', fontsize=12)
    plt.colorbar(im5, ax=axes[1, 1], label='Weighted loss')
    
    # Summary comparison
    unweighted_total = np.abs(pred_error).mean()
    weighted_total = (np.abs(pred_error) * magnitude).sum() / (magnitude.sum() + 1e-8)
    
    axes[1, 2].text(0.5, 0.7, 'Loss Comparison', fontsize=16, fontweight='bold',
                    ha='center', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.5, f'Unweighted: {unweighted_total:.4f}', fontsize=14,
                    ha='center', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.3, f'Weighted: {weighted_total:.4f}', fontsize=14,
                    ha='center', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.1, '(Lower = better noise handling)', fontsize=12,
                    ha='center', transform=axes[1, 2].transAxes, style='italic')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / "magnitude_weighted_loss_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    create_loss_explanation_figure(output_dir)


def create_loss_explanation_figure(output_dir: Path):
    """Create figure explaining the loss function formula."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    title = "Magnitude-Weighted Velocity Loss for Shear Wave Elastography"
    ax.text(0.5, 0.95, title, fontsize=18, fontweight='bold', ha='center', 
            transform=ax.transAxes)
    
    # Problem section
    ax.text(0.05, 0.85, "THE PROBLEM:", fontsize=14, fontweight='bold', 
            transform=ax.transAxes, color='darkred')
    problem_text = """
• In ultrasound, dark regions (background) have no echo signal
• The phase in these regions is random noise
• Kasai algorithm: velocity ∝ arg(z_{t+1} · conj(z_t))
• Random phase → random high velocity values ("salt and pepper" noise)
• Standard loss tries to learn this impossible random pattern
"""
    ax.text(0.05, 0.82, problem_text, fontsize=11, transform=ax.transAxes,
            verticalalignment='top', family='monospace')
    
    # Solution section
    ax.text(0.05, 0.55, "THE SOLUTION: Magnitude-Weighted Loss", fontsize=14, 
            fontweight='bold', transform=ax.transAxes, color='darkgreen')
    solution_text = """
Instead of treating every pixel equally, weight by signal strength:

    L = Σ |v_pred - v_target| × magnitude_mask
        ───────────────────────────────────────
               Σ magnitude_mask

Where:
    • v_pred, v_target = predicted/target Kasai velocity
    • magnitude_mask = |IQ| normalized to [0, 1]

Effect:
    • Bright pixels (tissue) → high weight → must match velocity
    • Dark pixels (noise) → low weight → ignored
"""
    ax.text(0.05, 0.52, solution_text, fontsize=11, transform=ax.transAxes,
            verticalalignment='top', family='monospace')
    
    # Implementation section
    ax.text(0.05, 0.18, "IMPLEMENTATION:", fontsize=14, fontweight='bold',
            transform=ax.transAxes, color='darkblue')
    impl_text = """
Config: "use_magnitude_weighted_loss": true, "lambda_kasai": 10.0
DataLoader: Returns magnitude_mask alongside sparse/dense pairs
Loss: MagnitudeWeightedKasaiLoss in src/utils/loss_functions.py
"""
    ax.text(0.05, 0.15, impl_text, fontsize=11, transform=ax.transAxes,
            verticalalignment='top', family='monospace')
    
    save_path = output_dir / "loss_explanation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize magnitude-weighted loss")
    parser.add_argument("--data", type=str, default=None, help="Path to .npy data file")
    parser.add_argument("--output", type=str, default="results/magnitude_weighted_demo",
                        help="Output directory")
    args = parser.parse_args()
    
    visualize_magnitude_weighted_loss(args.data, args.output)
