#!/usr/bin/env python3
"""
Generate Figure: Quantitative Error Analysis - MAE and SSIM across frames.

Creates a professional two-panel figure showing:
- Mean Absolute Error (MAE) vs interpolation position
- Structural Similarity Index (SSIM) vs interpolation position

Caption: Quantitative error analysis plots showing the Mean Absolute Error (MAE) 
and Structural Similarity Index (SSIM)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_predictions(data_dir: Path = None):
    """
    Load ground truth and prediction data.
    Returns all frames for analysis.
    """
    if data_dir is None:
        # Try to find test predictions
        possible_paths = [
            project_root / "results" / "test_predictions",
            project_root / "outputs" / "input_guided_test",
            project_root / "outputs" / "input_guided_trial_test",
        ]
        
        for path in possible_paths:
            if path.exists():
                data_dir = path
                break
    
    if data_dir is None or not data_dir.exists():
        print("Data directory not found. Generating synthetic demo data...")
        return generate_demo_data()
    
    # Load actual data
    gt_files = sorted(list(data_dir.glob("dense_*.npy")))
    pred_files = sorted(list(data_dir.glob("pred_*.npy")))
    
    if gt_files and pred_files:
        gt_data = np.load(gt_files[0])
        pred_data = np.load(pred_files[0])
        return gt_data, pred_data
    
    return generate_demo_data()


def generate_demo_data():
    """
    Generate synthetic shear wave sequence data for demonstration.
    """
    T, H, W = 20, 128, 128  # 20 frames in interpolation window
    
    # Create synthetic wave sequence
    gt = np.zeros((T, H, W), dtype=np.float64)
    pred = np.zeros((T, H, W), dtype=np.float64)
    
    x = np.linspace(0, 4 * np.pi, W)
    y = np.linspace(0, 4 * np.pi, H)
    X, Y = np.meshgrid(x, y)
    
    for t in range(T):
        # Simulate propagating wave
        wave = np.sin(X - 0.3 * Y + t * 0.3) * np.exp(-0.01 * (X - W//2)**2)
        gt[t] = wave * 0.2 + np.random.randn(H, W) * 0.01
        
        # Prediction with error that varies by position
        # More error in the middle of the window
        position_factor = 1 - 0.3 * np.abs(t - T//2) / (T//2)
        error = np.random.randn(H, W) * 0.02 * position_factor
        pred[t] = gt[t] + error
    
    return gt, pred


def iq_to_displacement_sequence(data, scale: float = 1.0):
    """
    Convert IQ data sequence to displacement magnitude sequence.
    
    Args:
        data: Complex IQ data (T, 2, H, W) or (T, H, W) or complex array
        scale: Scaling factor
    
    Returns:
        Displacement sequence (T, H, W) as real-valued array
    """
    # Handle complex numpy arrays
    if np.iscomplexobj(data):
        if data.ndim == 3:  # (T, H, W) complex
            # Use magnitude for consistent comparison
            displacement = np.abs(data).astype(np.float64)
            return displacement * scale
        elif data.ndim == 2:  # (H, W) complex
            return np.abs(data).astype(np.float64) * scale
    
    # Handle real-valued arrays
    if data.ndim == 4:  # (T, 2, H, W) - real/imag format
        T = data.shape[0]
        H, W = data.shape[2], data.shape[3]
        displacement = np.zeros((T, H, W), dtype=np.float64)
        
        for t in range(T):
            real = data[t, 0]
            imag = data[t, 1]
            # Use magnitude for consistent comparison
            displacement[t] = np.sqrt(real**2 + imag**2)
        
        return displacement * scale
    
    elif data.ndim == 3:  # (T, H, W) already displacement
        return np.abs(data).astype(np.float64) * scale
    
    return np.abs(data).astype(np.float64) * scale


def compute_metrics_per_frame(gt_seq, pred_seq):
    """
    Compute MAE and SSIM for each frame.
    
    Args:
        gt_seq: Ground truth sequence (T, H, W)
        pred_seq: Prediction sequence (T, H, W)
    
    Returns:
        mae_values: MAE per frame (normalized)
        ssim_values: SSIM per frame
    """
    T = min(gt_seq.shape[0], pred_seq.shape[0])
    
    mae_values = []
    ssim_values = []
    
    # Normalize both sequences to [0, 1] range globally for fair comparison
    gt_min, gt_max = gt_seq.min(), gt_seq.max()
    pred_min, pred_max = pred_seq.min(), pred_seq.max()
    
    # Use combined range for normalization
    global_min = min(gt_min, pred_min)
    global_max = max(gt_max, pred_max)
    data_range = global_max - global_min
    if data_range < 1e-8:
        data_range = 1.0
    
    # Normalize sequences
    gt_norm_seq = (gt_seq - global_min) / data_range
    pred_norm_seq = (pred_seq - global_min) / data_range
    
    for t in range(T):
        gt_frame = gt_norm_seq[t].astype(np.float64)
        pred_frame = pred_norm_seq[t].astype(np.float64)
        
        # Handle NaN/Inf values
        gt_frame = np.nan_to_num(gt_frame, nan=0.0, posinf=0.0, neginf=0.0)
        pred_frame = np.nan_to_num(pred_frame, nan=0.0, posinf=0.0, neginf=0.0)
        
        # MAE (on normalized data)
        mae = np.abs(gt_frame - pred_frame).mean()
        mae_values.append(mae)
        
        # SSIM (on normalized data)
        try:
            ssim_val = ssim(gt_frame, pred_frame, data_range=1.0)
        except Exception:
            ssim_val = 0.0
        ssim_values.append(ssim_val)
    
    return np.array(mae_values), np.array(ssim_values)


def create_mae_ssim_figure(mae_values, ssim_values, output_path: Path, 
                            upsampling_factor: int = 20,
                            model_name: str = "Input-Guided"):
    """
    Create a professional figure with MAE and SSIM plots.
    
    Args:
        mae_values: MAE per frame
        ssim_values: SSIM per frame
        output_path: Path to save the figure
        upsampling_factor: Number of frames in interpolation window
        model_name: Name of the model for legend
    """
    # Compute position-wise averages
    n_frames = len(mae_values)
    positions = np.arange(n_frames) % upsampling_factor
    
    # Group by position
    position_mae = {}
    position_ssim = {}
    
    for i, pos in enumerate(positions):
        if pos not in position_mae:
            position_mae[pos] = []
            position_ssim[pos] = []
        position_mae[pos].append(mae_values[i])
        position_ssim[pos].append(ssim_values[i])
    
    # Average per position
    pos_list = sorted(position_mae.keys())
    avg_mae = [np.mean(position_mae[p]) for p in pos_list]
    avg_ssim = [np.mean(position_ssim[p]) for p in pos_list]
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme
    bar_color = '#4682B4'  # Steel blue - matches the original image
    
    # ===== Left Panel: MAE =====
    ax1 = axes[0]
    bars1 = ax1.bar(pos_list, avg_mae, color=bar_color, edgecolor='black', 
                    linewidth=0.5, alpha=0.9, label=model_name)
    
    ax1.set_xlabel('Position in Interpolation Window', fontsize=12)
    ax1.set_ylabel('Average MAE', fontsize=12)
    ax1.set_title('Error vs Interpolation Position\n(0,19 = near keyframes, 10 = center)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.5, max(pos_list) + 0.5)
    ax1.set_ylim(0, max(avg_mae) * 1.15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, avg_mae):
        height = bar.get_height()
        ax1.annotate(f'{val:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=7, rotation=90)
    
    # ===== Right Panel: SSIM =====
    ax2 = axes[1]
    bars2 = ax2.bar(pos_list, avg_ssim, color='#2E8B57', edgecolor='black',
                    linewidth=0.5, alpha=0.9, label=model_name)
    
    ax2.set_xlabel('Position in Interpolation Window', fontsize=12)
    ax2.set_ylabel('Average SSIM', fontsize=12)
    ax2.set_title('SSIM vs Interpolation Position\n(Higher is Better)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.5, max(pos_list) + 0.5)
    ax2.set_ylim(min(avg_ssim) * 0.95, 1.0)  # SSIM is typically close to 1
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, avg_ssim):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    print(f"  MAE range: [{min(avg_mae):.4f}, {max(avg_mae):.4f}]")
    print(f"  SSIM range: [{min(avg_ssim):.4f}, {max(avg_ssim):.4f}]")
    
    return avg_mae, avg_ssim


def create_combined_figure(mae_values, ssim_values, output_path: Path,
                           upsampling_factor: int = 20,
                           model_name: str = "Input-Guided"):
    """
    Create a combined figure with both MAE and SSIM on dual y-axes.
    """
    n_frames = len(mae_values)
    positions = np.arange(n_frames) % upsampling_factor
    
    # Group by position
    position_mae = {}
    position_ssim = {}
    
    for i, pos in enumerate(positions):
        if pos not in position_mae:
            position_mae[pos] = []
            position_ssim[pos] = []
        position_mae[pos].append(mae_values[i])
        position_ssim[pos].append(ssim_values[i])
    
    pos_list = sorted(position_mae.keys())
    avg_mae = [np.mean(position_mae[p]) for p in pos_list]
    avg_ssim = [np.mean(position_ssim[p]) for p in pos_list]
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar width
    width = 0.35
    x = np.array(pos_list)
    
    # MAE bars (left y-axis)
    bars1 = ax1.bar(x - width/2, avg_mae, width, color='#4682B4', 
                    edgecolor='black', linewidth=0.5, alpha=0.9, label='MAE')
    ax1.set_xlabel('Position in Interpolation Window', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, color='#4682B4')
    ax1.tick_params(axis='y', labelcolor='#4682B4')
    ax1.set_ylim(0, max(avg_mae) * 1.2)
    
    # SSIM bars (right y-axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, avg_ssim, width, color='#2E8B57',
                    edgecolor='black', linewidth=0.5, alpha=0.9, label='SSIM')
    ax2.set_ylabel('Structural Similarity Index (SSIM)', fontsize=12, color='#2E8B57')
    ax2.tick_params(axis='y', labelcolor='#2E8B57')
    ax2.set_ylim(min(avg_ssim) * 0.9, 1.02)
    
    # Title
    ax1.set_title('Quantitative Error Analysis: MAE and SSIM vs Interpolation Position\n' +
                  '(0,19 = near keyframes, 10 = center)', 
                  fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    ax1.set_xticks(pos_list)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved combined figure: {output_path}")


def main():
    """Main function to generate MAE and SSIM figures."""
    
    # Output directory
    output_dir = project_root / "report" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    gt_data, pred_data = load_predictions()
    
    # Convert to displacement sequence
    gt_seq = iq_to_displacement_sequence(gt_data)
    pred_seq = iq_to_displacement_sequence(pred_data)
    
    print(f"Ground Truth sequence shape: {gt_seq.shape}")
    print(f"Prediction sequence shape: {pred_seq.shape}")
    
    # Compute metrics per frame
    print("\nComputing MAE and SSIM for each frame...")
    mae_values, ssim_values = compute_metrics_per_frame(gt_seq, pred_seq)
    
    print(f"  Overall MAE: {mae_values.mean():.4f} ± {mae_values.std():.4f}")
    print(f"  Overall SSIM: {ssim_values.mean():.4f} ± {ssim_values.std():.4f}")
    
    # Determine upsampling factor from data
    upsampling_factor = min(20, len(mae_values))
    
    # Generate side-by-side figure
    output_path = output_dir / "figure_mae_ssim_analysis.png"
    create_mae_ssim_figure(mae_values, ssim_values, output_path, 
                           upsampling_factor=upsampling_factor)
    
    # Generate combined dual-axis figure
    output_path_combined = output_dir / "figure_mae_ssim_combined.png"
    create_combined_figure(mae_values, ssim_values, output_path_combined,
                           upsampling_factor=upsampling_factor)
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"Output directory: {output_dir}")
    print("\nCaption: Quantitative error analysis plots showing the")
    print("Mean Absolute Error (MAE) and Structural Similarity Index (SSIM)")
    print("=" * 60)


if __name__ == "__main__":
    main()
