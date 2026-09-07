#!/usr/bin/env python3
"""
Generate displacement comparison visualizations for all trained models.
This script generates similar plots and animations as in visualize_shear_waves.ipynb
for Trial 2, Trial 3, Trial 4, and BiLSTM models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import os
import glob
from pathlib import Path
import sys

# Import models
from src.architectures.pinn_recunet import PI_CV_RecUNet
from src.architectures.bilstm_model import PI_CV_RecUNet_BiLSTM
from src.utils.data_loader import SWEDataset, DataConfig, complex_to_tensor

print("Libraries loaded successfully!")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def prepare_data_for_model(iq_data, upsampling_factor):
    """
    Prepare IQ data for the model.
    
    Args:
        iq_data: Complex array (H, W, T)
        upsampling_factor: Temporal upsampling factor
    
    Returns:
        sparse_tensor: Model input (1, T_sparse, 2, 1, H, W)
        dense_tensor: Ground truth (1, T_dense, 2, 1, H, W)
        normalization_params: For denormalization
    """
    H, W, T = iq_data.shape
    
    # Transpose to (T, H, W)
    data = iq_data.transpose(2, 0, 1)  # (T, H, W)
    
    # Get sparse frame indices (every upsampling_factor frames)
    sparse_indices = list(range(0, T, upsampling_factor))
    
    # Ensure we have complete segments
    n_sparse = len(sparse_indices)
    n_dense = (n_sparse - 1) * upsampling_factor + 1
    
    # Get dense frames (ground truth)
    dense_data = data[:n_dense]  # (T_dense, H, W)
    sparse_data = data[sparse_indices[:n_sparse]]  # (T_sparse, H, W)
    
    # Normalize (per-sample normalization)
    real = dense_data.real
    imag = dense_data.imag
    
    mean_real, std_real = real.mean(), real.std() + 1e-8
    mean_imag, std_imag = imag.mean(), imag.std() + 1e-8
    
    # Normalize dense
    dense_real_norm = (real - mean_real) / std_real
    dense_imag_norm = (imag - mean_imag) / std_imag
    
    # Normalize sparse
    sparse_real_norm = (sparse_data.real - mean_real) / std_real
    sparse_imag_norm = (sparse_data.imag - mean_imag) / std_imag
    
    # Convert to tensors: (T, 2, 1, H, W)
    dense_tensor = torch.stack([
        torch.from_numpy(dense_real_norm).float().unsqueeze(1),
        torch.from_numpy(dense_imag_norm).float().unsqueeze(1)
    ], dim=1)  # (T_dense, 2, 1, H, W)
    
    sparse_tensor = torch.stack([
        torch.from_numpy(sparse_real_norm).float().unsqueeze(1),
        torch.from_numpy(sparse_imag_norm).float().unsqueeze(1)
    ], dim=1)  # (T_sparse, 2, 1, H, W)
    
    # Add batch dimension
    sparse_tensor = sparse_tensor.unsqueeze(0)  # (1, T_sparse, 2, 1, H, W)
    dense_tensor = dense_tensor.unsqueeze(0)    # (1, T_dense, 2, 1, H, W)
    
    norm_params = {
        'mean_real': mean_real,
        'std_real': std_real,
        'mean_imag': mean_imag,
        'std_imag': std_imag
    }
    
    return sparse_tensor, dense_tensor, norm_params, sparse_indices[:n_sparse]


def iq_to_displacement(iq_tensor, method='phase_diff'):
    """
    Convert IQ data tensor to displacement.
    
    Args:
        iq_tensor: Tensor of shape (1, T, 2, 1, H, W)
                   where dim=2 is [real, imag]
        method: 'phase_diff' for phase difference,
                'magnitude' for magnitude only
    
    Returns:
        displacement: Array of shape (T-1, H, W) for phase_diff
                      or (T, H, W) for magnitude
    """
    # Remove batch and channel dimensions
    real = iq_tensor[0, :, 0, 0, :, :].numpy()  # (T, H, W)
    imag = iq_tensor[0, :, 1, 0, :, :].numpy()  # (T, H, W)
    
    # Create complex array
    complex_iq = real + 1j * imag
    
    if method == 'phase_diff':
        # Compute phase
        phase = np.angle(complex_iq)  # (T, H, W)
        
        # Compute phase difference (displacement)
        # Use unwrapped phase for continuous displacement
        phase_diff = np.diff(phase, axis=0)  # (T-1, H, W)
        
        # Wrap phase difference to [-pi, pi]
        displacement = np.angle(np.exp(1j * phase_diff))
        
        return displacement
    
    elif method == 'magnitude':
        # Just return magnitude (intensity)
        return np.abs(complex_iq)
    
    elif method == 'phase':
        # Return raw phase
        return np.angle(complex_iq)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_frame_comparison(gt, pred, frame_idx, title_prefix=""):
    """
    Plot ground truth vs prediction comparison for a single frame with shared colorbar.
    """
    # Create figure with extra space for colorbar on the right
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
    
    # Compute common color limits (scale down by 0.1 to see waves better)
    vmax = 0.1 * max(np.abs(gt[frame_idx]).max(), np.abs(pred[frame_idx]).max())
    vmin = -vmax
    
    # Use jet colormap like reference image
    cmap = 'jet'
    
    # Ground truth
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(gt[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f'{title_prefix}Ground Truth\nFrame {frame_idx}')
    ax1.axis('off')
    
    # Prediction
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(pred[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f'{title_prefix}Predicted\nFrame {frame_idx}')
    ax2.axis('off')
    
    # Difference (error)
    diff = gt[frame_idx] - pred[frame_idx]
    err_max = np.abs(diff).max()
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
    ax3.set_title(f'Difference (Error)\nMAE: {np.abs(diff).mean():.4f}')
    ax3.axis('off')
    
    # Add shared colorbar for GT and Pred on the right
    cbar_ax = fig.add_subplot(gs[3])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Displacement (a.u.)', fontsize=11, rotation=270, labelpad=15)
    
    plt.suptitle(f"Colorbar range: [{vmin:.3f}, {vmax:.3f}]", fontsize=10)
    plt.tight_layout()
    return fig


def create_wave_animation(gt, pred, fps=10, interval=100):
    """
    Create side-by-side animation of ground truth and predicted waves.
    """
    n_frames = min(gt.shape[0], pred.shape[0])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compute global color limits for consistency
    vmax = max(np.abs(gt).max(), np.abs(pred).max()) * 0.8
    
    # Initialize images
    im1 = axes[0].imshow(gt[0], cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label='Displacement')
    
    im2 = axes[1].imshow(pred[0], cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Predicted')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Displacement')
    
    diff = gt[0] - pred[0]
    err_max = np.abs(gt - pred).max()
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
    axes[2].set_title('Error')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, label='Error')
    
    title = fig.suptitle('Frame 0', fontsize=14)
    
    plt.tight_layout()
    
    def update(frame):
        im1.set_array(gt[frame])
        im2.set_array(pred[frame])
        im3.set_array(gt[frame] - pred[frame])
        title.set_text(f'Frame {frame}/{n_frames-1}')
        return [im1, im2, im3, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, 
        interval=interval, blit=True
    )
    
    plt.close(fig)
    return anim


def generate_visualizations_for_model(model_name, checkpoint_path, model_class, iq_data, is_bilstm=False):
    """
    Generate all visualizations for a given model checkpoint.
    
    Args:
        model_name: Name for output files (e.g., 'trial_2', 'trial_3', 'bilstm')
        checkpoint_path: Path to the model checkpoint
        model_class: The model class to use
        iq_data: The IQ data to use for inference
        is_bilstm: Whether this is a BiLSTM model
    """
    print("=" * 70)
    print(f"GENERATING VISUALIZATIONS FOR: {model_name.upper()}")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"\nModel Config:")
    print(f"  Upsampling factor: {config['upsampling_factor']}x")
    print(f"  Base channels: {config['base_channels']}")
    print(f"  LSTM hidden: {config['lstm_hidden_channels']}")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create model
    if is_bilstm:
        model = model_class(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            base_channels=config['base_channels'],
            num_levels=config['num_levels'],
            lstm_hidden_channels=config['lstm_hidden_channels'],
            lstm_num_layers=config['lstm_num_layers'],
            upsampling_factor=config['upsampling_factor'],
            bilstm_combine_mode=config.get('bilstm_combine_mode', 'concat')
        )
    else:
        model = model_class(
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
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Prepare test data
    upsampling_factor = config['upsampling_factor']
    sparse_tensor_model, dense_tensor_model, norm_params_model, sparse_indices_model = prepare_data_for_model(
        iq_data, upsampling_factor
    )
    
    # Run inference
    with torch.no_grad():
        sparse_input = sparse_tensor_model.to(device)
        predicted_model = model(sparse_input)
        predicted_model = predicted_model.cpu()
    
    # Compute PSNR
    mse_val = ((predicted_model - dense_tensor_model) ** 2).mean()
    psnr_val = 10 * torch.log10(1.0 / mse_val)
    print(f"PSNR on test sample: {psnr_val.item():.2f} dB")
    
    # Convert to displacement
    gt_disp = iq_to_displacement(dense_tensor_model, method='phase_diff')
    pred_disp = iq_to_displacement(predicted_model, method='phase_diff')
    
    # Create output directory
    output_dir_model = f'wave_visualizations_{model_name}'
    os.makedirs(output_dir_model, exist_ok=True)
    
    # ========== SINGLE FRAME COMPARISON ==========
    frame_to_show = 10
    fig = plot_frame_comparison(gt_disp, pred_disp, frame_to_show, f"{model_name.upper()} Displacement: ")
    fig.savefig(f'{output_dir_model}/single_frame_comparison_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir_model}/single_frame_comparison_{model_name}.png")
    
    # ========== MULTI-FRAME COMPARISON ==========
    frames_to_compare = [0, 5, 10, 15, 19]
    
    fig = plt.figure(figsize=(22, 8))
    gs = GridSpec(2, len(frames_to_compare) + 1, width_ratios=[1]*len(frames_to_compare) + [0.05], 
                  wspace=0.05, hspace=0.15)
    
    vmax = 0.1 * max(np.abs(gt_disp).max(), np.abs(pred_disp).max())
    vmin = -vmax
    cmap = 'jet'
    
    for i, frame_idx in enumerate(frames_to_compare):
        ax1 = fig.add_subplot(gs[0, i])
        im = ax1.imshow(gt_disp[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title(f'GT Frame {frame_idx}', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(pred_disp[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f'Pred Frame {frame_idx}', fontsize=10)
        ax2.axis('off')
    
    fig.text(0.01, 0.72, 'Ground\nTruth', fontsize=12, fontweight='bold', va='center')
    fig.text(0.01, 0.28, 'Predicted', fontsize=12, fontweight='bold', va='center')
    
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Displacement (a.u.)', fontsize=12, rotation=270, labelpad=20)
    
    plt.suptitle(f'{model_name.upper()} - Shear Wave Displacement Comparison', fontsize=14)
    fig.savefig(f'{output_dir_model}/displacement_comparison_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir_model}/displacement_comparison_{model_name}.png")
    
    # ========== SAVE KEY FRAMES ==========
    frames_to_save = [0, 5, 10, 15, 19, 20, 25, 30]
    for frame_idx in frames_to_save:
        if frame_idx < len(gt_disp):
            fig = plot_frame_comparison(gt_disp, pred_disp, frame_idx, f"{model_name.upper()}: ")
            fig.savefig(f"{output_dir_model}/displacement_frame_{frame_idx:03d}_{model_name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
    print(f"Saved {len([f for f in frames_to_save if f < len(gt_disp)])} individual frame comparisons")
    
    # ========== ANIMATION ==========
    n_frames_to_animate = min(40, gt_disp.shape[0])
    anim_model = create_wave_animation(
        gt_disp[:n_frames_to_animate], 
        pred_disp[:n_frames_to_animate],
        interval=150
    )
    
    try:
        anim_model.save(f"{output_dir_model}/wave_animation_{model_name}.gif", writer='pillow', fps=8)
        print(f"Saved: {output_dir_model}/wave_animation_{model_name}.gif")
    except Exception as e:
        print(f"Could not save GIF: {e}")
    
    # ========== ERROR ANALYSIS ==========
    mae_per_frame_model = np.abs(gt_disp - pred_disp).mean(axis=(1, 2))
    mse_per_frame_model = ((gt_disp - pred_disp) ** 2).mean(axis=(1, 2))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    frames = np.arange(len(mae_per_frame_model))
    
    axes[0].plot(frames, mae_per_frame_model, 'b-', linewidth=2, label='MAE')
    axes[0].set_xlabel('Frame Index')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title(f'{model_name.upper()} - Displacement Prediction Error Over Time')
    axes[0].grid(True, alpha=0.3)
    
    for i in range(0, len(mae_per_frame_model), upsampling_factor):
        axes[0].axvline(i, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].legend(['MAE', 'Sparse Frame Boundary'])
    
    window_positions = frames % upsampling_factor
    position_errors = {}
    for pos in range(upsampling_factor):
        mask = window_positions == pos
        if mask.sum() > 0:
            position_errors[pos] = mae_per_frame_model[mask].mean()
    
    positions = list(position_errors.keys())
    errors = list(position_errors.values())
    
    axes[1].bar(positions, errors, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Position Within Interpolation Window')
    axes[1].set_ylabel('Average MAE')
    axes[1].set_title(f'{model_name.upper()} - Error vs Position in Interpolation Window')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(f'{output_dir_model}/error_analysis_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir_model}/error_analysis_{model_name}.png")
    
    # ========== SUMMARY ==========
    print(f"\n{'='*50}")
    print(f"SUMMARY FOR {model_name.upper()}")
    print(f"{'='*50}")
    print(f"PSNR: {psnr_val.item():.2f} dB")
    print(f"MAE (displacement): {mae_per_frame_model.mean():.6f}")
    print(f"MSE (displacement): {mse_per_frame_model.mean():.6f}")
    print(f"Output directory: {output_dir_model}/")
    print(f"{'='*50}\n")
    
    return {
        'model_name': model_name,
        'psnr': psnr_val.item(),
        'mae': mae_per_frame_model.mean(),
        'mse': mse_per_frame_model.mean(),
        'output_dir': output_dir_model
    }


def main():
    # Load a test sample (use same test file for fair comparison)
    data_dir = "data/CIRS_phantom_dataset"
    all_files = sorted(glob.glob(os.path.join(data_dir, 'IQData*.npy')))
    print(f"Total files found: {len(all_files)}")
    
    # Use a consistent test file (same seed as in training)
    np.random.seed(42)  # Using seed 42 as a common seed
    indices = np.random.permutation(len(all_files))
    
    # Get test indices (last 15% of data)
    train_end = int(0.7 * len(all_files))
    val_end = train_end + int(0.15 * len(all_files))
    test_indices = indices[val_end:]
    test_files = [all_files[i] for i in test_indices]
    
    # Use first test file
    test_file = test_files[0]
    print(f"\nUsing test file: {os.path.basename(test_file)}")
    
    # Load IQ data
    iq_data = np.load(test_file)
    print(f"IQ data shape: {iq_data.shape}")
    
    # Define all models to process
    models_to_process = [
        {
            'name': 'trial_2',
            'checkpoint': 'best_model_trial_2.pt',
            'model_class': PI_CV_RecUNet,
            'is_bilstm': False
        },
        {
            'name': 'trial_3',
            'checkpoint': 'best_model_trial_3.pt',
            'model_class': PI_CV_RecUNet,
            'is_bilstm': False
        },
        {
            'name': 'trial_4',
            'checkpoint': 'best_model_trial_4.pt',
            'model_class': PI_CV_RecUNet,
            'is_bilstm': False
        },
        {
            'name': 'bilstm',
            'checkpoint': 'best_model_bilstm.pt',
            'model_class': PI_CV_RecUNet_BiLSTM,
            'is_bilstm': True
        }
    ]
    
    # Generate visualizations for each model
    all_results = []
    for model_info in models_to_process:
        if os.path.exists(model_info['checkpoint']):
            result = generate_visualizations_for_model(
                model_name=model_info['name'],
                checkpoint_path=model_info['checkpoint'],
                model_class=model_info['model_class'],
                iq_data=iq_data,
                is_bilstm=model_info['is_bilstm']
            )
            all_results.append(result)
        else:
            print(f"WARNING: Checkpoint not found: {model_info['checkpoint']}")
    
    # ========== COMPARISON SUMMARY ==========
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL MODELS")
    print("=" * 80)
    print(f"\n{'Model':<15} {'PSNR (dB)':<12} {'MAE':<15} {'MSE':<15} {'Output Dir':<25}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r['model_name']:<15} {r['psnr']:<12.2f} {r['mae']:<15.6f} {r['mse']:<15.6f} {r['output_dir']:<25}")
    
    print("=" * 80)
    
    # ========== CREATE COMPARISON BAR CHART ==========
    if len(all_results) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        model_names = [r['model_name'] for r in all_results]
        psnr_values = [r['psnr'] for r in all_results]
        mae_values = [r['mae'] for r in all_results]
        mse_values = [r['mse'] for r in all_results]
        
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'][:len(all_results)]
        
        # PSNR comparison
        axes[0].bar(model_names, psnr_values, color=colors, edgecolor='black')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Comparison (Higher is Better)')
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(psnr_values):
            axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
        
        # MAE comparison
        axes[1].bar(model_names, mae_values, color=colors, edgecolor='black')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('MAE Comparison (Lower is Better)')
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mae_values):
            axes[1].text(i, v + v*0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)
        
        # MSE comparison
        axes[2].bar(model_names, mse_values, color=colors, edgecolor='black')
        axes[2].set_ylabel('Mean Squared Error')
        axes[2].set_title('MSE Comparison (Lower is Better)')
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mse_values):
            axes[2].text(i, v + v*0.02, f'{v:.6f}', ha='center', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('model_comparison_metrics.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("\nSaved: model_comparison_metrics.png")
    
    # ========== LIST ALL GENERATED FILES ==========
    print("\n" + "=" * 60)
    print("ALL GENERATED VISUALIZATIONS")
    print("=" * 60)
    
    for r in all_results:
        print(f"\n{r['model_name'].upper()}:")
        output_dir = r['output_dir']
        if os.path.exists(output_dir):
            files = sorted(os.listdir(output_dir))
            for f in files:
                file_path = os.path.join(output_dir, f)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  - {f} ({size_kb:.1f} KB)")
    
    print("=" * 60)
    print("\nVisualization generation complete!")


if __name__ == '__main__':
    main()
