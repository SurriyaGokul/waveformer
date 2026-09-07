#!/usr/bin/env python3
"""
Velocity Map Visualization Script for Input-Guided Motion Architecture.

Generates Kasai velocity maps from the trained model with:
- Jet colormap
- Gaussian smoothing for cleaner visualization
- Per-sample normalization (matching training data loader)

Usage:
    python scripts/visualize_velocity_maps.py --model_path saved_models/Input_guided_physics_latest_newdata_best_model.pt
    python scripts/visualize_velocity_maps.py --model_path saved_models/Input_guided_physics_trial_2_best_model.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Set matplotlib style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_model(model_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    from src.training.train_input_guided import create_input_guided_model
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            model_config = config
        else:
            model_config = config.__dict__ if hasattr(config, '__dict__') else vars(config)
    else:
        # Default config
        model_config = {
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 32,
            'num_levels': 4,
            'lstm_hidden_channels': 128,
            'lstm_num_layers': 1,
            'upsampling_factor': 20,
            'flow_hidden_channels': 32,
            'flow_scale': 0.1,
            'use_residual_fusion': True,
        }
    
    print("Model Configuration:")
    for key in ['upsampling_factor', 'base_channels', 'lstm_hidden_channels', 'flow_hidden_channels']:
        if key in model_config:
            print(f"  {key}: {model_config[key]}")
    
    if 'epoch' in checkpoint:
        print(f"  Best epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create model
    model = create_input_guided_model(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('out_channels', 1),
        base_channels=model_config.get('base_channels', 32),
        num_levels=model_config.get('num_levels', 4),
        lstm_hidden_channels=model_config.get('lstm_hidden_channels', 128),
        lstm_num_layers=model_config.get('lstm_num_layers', 1),
        upsampling_factor=model_config.get('upsampling_factor', 20),
        flow_hidden_channels=model_config.get('flow_hidden_channels', 32),
        flow_scale=model_config.get('flow_scale', 0.1),
        use_residual_fusion=model_config.get('use_residual_fusion', True),
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded! Total parameters: {total_params:,}")
    
    return model, model_config


def collect_validation_files(data_dir: Path):
    """Collect OOD validation files (50kPa CIRS or Plane 5 from 2FT/3FT)."""
    val_files = []
    
    # CIRS 50kPa (OOD stiffness)
    cirs_dir = data_dir / 'CIRS_Phantom_Latest' / 'Data set 1(50)'
    if cirs_dir.exists():
        cirs_files = sorted(cirs_dir.glob('*.npy'))
        val_files.extend(cirs_files)
        print(f"  CIRS 50kPa files: {len(cirs_files)}")
    
    # 2FT/3FT Plane 5 files (10kHz and 20kHz only)
    for ft_type in ['2FT(na100)', '3FT(na100)']:
        ft_base = data_dir / ft_type
        if ft_base.exists():
            for focus in ['10mmfocus', '15mmfocus']:
                for freq in ['10kHz', '20kHz']:
                    plane5_dir = ft_base / focus / freq / 'Plane 5'
                    if plane5_dir.exists():
                        files = sorted(plane5_dir.glob('*.npy'))
                        val_files.extend(files)
                        if files:
                            print(f"  {ft_type}/{focus}/{freq}/Plane 5: {len(files)} files")
    
    if len(val_files) == 0:
        # Fallback: use any available .npy files
        print("  No OOD files found. Looking for any .npy files...")
        all_npy = list(data_dir.rglob('*.npy'))
        val_files = sorted(all_npy)[:10]
    
    return val_files


def prepare_data_for_model(iq_data, sparse_step=20, fixed_sparse_frames=3):
    """
    Prepare IQ data for the model using per-sample normalization.
    
    Matches the normalization from InputGuidedDataLoader.
    """
    # Handle different input shapes
    if iq_data.ndim == 3:
        if iq_data.shape[2] < iq_data.shape[0] and iq_data.shape[2] < iq_data.shape[1]:
            # (H, W, T) format - transpose to (T, H, W)
            data = np.transpose(iq_data, (2, 0, 1))
        else:
            data = iq_data
    else:
        data = iq_data
    
    T, H, W = data.shape
    
    # Clean data (handle NaN/Inf)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Per-sample normalization (matching training data loader)
    real = data.real
    imag = data.imag
    
    eps = 1e-8
    mean_real = np.mean(real)
    mean_imag = np.mean(imag)
    std_real = np.std(real) + eps
    std_imag = np.std(imag) + eps
    
    real_norm = (real - mean_real) / std_real
    imag_norm = (imag - mean_imag) / std_imag
    
    # Create sparse indices
    sparse_indices = list(range(0, T, sparse_step))
    
    # Ensure fixed number of sparse frames
    if len(sparse_indices) > fixed_sparse_frames:
        sparse_indices = sparse_indices[:fixed_sparse_frames]
    elif len(sparse_indices) < fixed_sparse_frames:
        while len(sparse_indices) < fixed_sparse_frames:
            sparse_indices.append(min(sparse_indices[-1] + sparse_step, T - 1))
    
    # Dense frames are between first and last sparse frame
    dense_start = sparse_indices[0]
    dense_end = sparse_indices[-1] + 1
    dense_indices = list(range(dense_start, dense_end))
    
    # Extract frames
    sparse_real = real_norm[sparse_indices]
    sparse_imag = imag_norm[sparse_indices]
    dense_real = real_norm[dense_indices]
    dense_imag = imag_norm[dense_indices]
    
    # Convert to tensors: (T, 2, 1, H, W)
    sparse_tensor = torch.stack([
        torch.from_numpy(sparse_real).float().unsqueeze(1),
        torch.from_numpy(sparse_imag).float().unsqueeze(1)
    ], dim=1)
    
    dense_tensor = torch.stack([
        torch.from_numpy(dense_real).float().unsqueeze(1),
        torch.from_numpy(dense_imag).float().unsqueeze(1)
    ], dim=1)
    
    # Add batch dimension: (1, T, 2, 1, H, W)
    sparse_tensor = sparse_tensor.unsqueeze(0)
    dense_tensor = dense_tensor.unsqueeze(0)
    
    norm_params = {
        'mean_real': mean_real,
        'std_real': std_real,
        'mean_imag': mean_imag,
        'std_imag': std_imag
    }
    
    return sparse_tensor, dense_tensor, norm_params, sparse_indices


def compute_kasai_velocity(iq_tensor, sigma=1.5):
    """
    Compute Kasai velocity map from IQ tensor.
    
    Kasai method: R = P(t) * conj(P(t-1))
    Velocity ∝ angle(R)
    
    Args:
        iq_tensor: Tensor of shape (B, T, 2, C, H, W) where dim 2 is [real, imag]
        sigma: Gaussian smoothing sigma for cleaner visualization
    
    Returns:
        velocity: Velocity maps (T-1, H, W) in radians
    """
    # Extract real and imaginary parts
    real = iq_tensor[0, :, 0, 0, :, :].numpy()
    imag = iq_tensor[0, :, 1, 0, :, :].numpy()
    
    # Reconstruct complex IQ
    complex_iq = real + 1j * imag
    
    # Compute lag-1 autocorrelation: R = P(t) * conj(P(t-1))
    P_t = complex_iq[1:]
    P_tm1 = complex_iq[:-1]
    R = P_t * np.conj(P_tm1)
    
    # Velocity is the angle of R (phase shift per frame)
    velocity = np.angle(R)
    
    # Apply Gaussian smoothing for cleaner visualization
    if sigma > 0:
        velocity_smooth = np.zeros_like(velocity)
        for t in range(velocity.shape[0]):
            velocity_smooth[t] = gaussian_filter(velocity[t], sigma=sigma)
        return velocity_smooth
    
    return velocity


def compute_psnr(pred, target):
    """Compute PSNR on complex magnitude."""
    pred_mag = torch.sqrt(pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2)
    target_mag = torch.sqrt(target[:, :, 0] ** 2 + target[:, :, 1] ** 2)
    
    mse = torch.mean((pred_mag - target_mag) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = torch.max(target_mag)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def plot_velocity_comparison(gt_vel, pred_vel, frame_idx, output_path, title_suffix=""):
    """
    Visualize velocity maps: Ground Truth, Prediction, and Error.
    Uses jet colormap with Gaussian smoothing.
    """
    # Physical dimensions
    total_width_mm = 40.0
    total_depth_mm = 40.0
    half_width = total_width_mm / 2.0
    extent_mm = [-half_width, half_width, total_depth_mm, 0]
    
    gt_frame = gt_vel[frame_idx]
    pred_frame = pred_vel[frame_idx]
    error_frame = np.abs(gt_frame - pred_frame)
    
    # Use SEPARATE scales for GT and Prediction
    # This ensures GT is visible even if prediction has artifacts
    gt_vmax = np.percentile(np.abs(gt_frame), 99)
    pred_vmax = np.percentile(np.abs(pred_frame), 99)
    
    if gt_vmax < 1e-6:
        gt_vmax = 0.1  # Fallback
    if pred_vmax < 1e-6:
        pred_vmax = 0.1
    
    cmap_vel = 'jet'
    cmap_err = 'hot'
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    # Panel (a): Ground Truth Velocity (with GT-specific scale)
    im0 = axes[0].imshow(gt_frame, cmap=cmap_vel, vmin=-gt_vmax, vmax=gt_vmax,
                         extent=extent_mm, origin='upper', aspect='equal')
    axes[0].set_title('Ground Truth Velocity', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lateral (mm)')
    axes[0].set_ylabel('Axial (mm)')
    axes[0].text(-0.08, 1.02, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold')
    
    # Add colorbar for GT
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('Vel (rad)', rotation=270, labelpad=15)
    
    # Panel (b): Predicted Velocity (with Pred-specific scale)
    im1 = axes[1].imshow(pred_frame, cmap=cmap_vel, vmin=-pred_vmax, vmax=pred_vmax,
                         extent=extent_mm, origin='upper', aspect='equal')
    axes[1].set_title('Predicted Velocity', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lateral (mm)')
    axes[1].tick_params(axis='y', labelleft=False)
    axes[1].text(-0.08, 1.02, '(b)', transform=axes[1].transAxes, fontsize=16, fontweight='bold')
    
    # Colorbar for prediction
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Vel (rad)', rotation=270, labelpad=15)
    
    # Panel (c): Absolute Error
    vmax_err = np.percentile(error_frame, 98)
    if vmax_err == 0:
        vmax_err = 0.01
    
    im2 = axes[2].imshow(error_frame, cmap=cmap_err, vmin=0, vmax=vmax_err,
                         extent=extent_mm, origin='upper', aspect='equal')
    
    mae = error_frame.mean()
    axes[2].set_title(f'Velocity Error\nMAE: {mae:.4f} rad', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Lateral (mm)')
    axes[2].tick_params(axis='y', labelleft=False)
    axes[2].text(-0.08, 1.02, '(c)', transform=axes[2].transAxes, fontsize=16, fontweight='bold')
    
    # Colorbar for error
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Error (rad)', rotation=270, labelpad=15)
    
    # Global title
    fig.suptitle(f'Kasai Velocity Map - Frame {frame_idx}{title_suffix}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return mae


def plot_velocity_evolution(gt_vel, pred_vel, frames, output_path):
    """Create a grid showing velocity evolution over multiple frames.
    
    Uses SEPARATE normalization for GT and Prediction rows so both are visible.
    """
    n_frames = len(frames)
    
    # Compute SEPARATE velocity limits for GT and Prediction
    # This ensures both are visible even if they have different scales
    gt_vmax = np.percentile(np.abs(gt_vel[frames]), 99)
    pred_vmax = np.percentile(np.abs(pred_vel[frames]), 99)
    
    # Use same scale if they're similar, otherwise separate
    if pred_vmax > 3 * gt_vmax:
        # Prediction has artifacts - use separate scales
        gt_vmin, gt_vmax = -gt_vmax, gt_vmax
        pred_vmin, pred_vmax = -pred_vmax, pred_vmax
        use_separate_scales = True
    else:
        # Similar scales - use global
        vmax = max(gt_vmax, pred_vmax)
        gt_vmin, gt_vmax = -vmax, vmax
        pred_vmin, pred_vmax = -vmax, vmax
        use_separate_scales = False
    
    # Physical extent
    total_width_mm = 40.0
    total_depth_mm = 40.0
    half_width = total_width_mm / 2.0
    extent_mm = [-half_width, half_width, total_depth_mm, 0]
    
    fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))
    
    for col, frame_idx in enumerate(frames):
        # Row 0: Ground Truth (with GT-specific colorscale)
        im_gt = axes[0, col].imshow(gt_vel[frame_idx], cmap='jet', 
                                     vmin=gt_vmin, vmax=gt_vmax,
                                     extent=extent_mm, origin='upper', aspect='equal')
        axes[0, col].set_title(f't = {frame_idx}', fontsize=12)
        axes[0, col].axis('off')
        
        # Row 1: Prediction (with Pred-specific colorscale)
        im_pred = axes[1, col].imshow(pred_vel[frame_idx], cmap='jet', 
                                       vmin=pred_vmin, vmax=pred_vmax,
                                       extent=extent_mm, origin='upper', aspect='equal')
        axes[1, col].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.72, 'Ground Truth', fontsize=14, fontweight='bold', 
             rotation=90, va='center')
    fig.text(0.02, 0.28, 'Predicted', fontsize=14, fontweight='bold', 
             rotation=90, va='center')
    
    # Add separate colorbars for each row if using separate scales
    if use_separate_scales:
        # GT colorbar
        cbar_ax_gt = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        cbar_gt = fig.colorbar(im_gt, cax=cbar_ax_gt)
        cbar_gt.set_label('GT Vel (rad)', fontsize=10, rotation=270, labelpad=15)
        
        # Pred colorbar
        cbar_ax_pred = fig.add_axes([0.92, 0.1, 0.015, 0.35])
        cbar_pred = fig.colorbar(im_pred, cax=cbar_ax_pred)
        cbar_pred.set_label('Pred Vel (rad)', fontsize=10, rotation=270, labelpad=15)
    else:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im_gt, cax=cbar_ax)
        cbar.set_label('Velocity (rad/frame)', fontsize=12, rotation=270, labelpad=18)
    
    plt.suptitle('Shear Wave Velocity Evolution', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 0.9, 0.95])
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_velocity_error_analysis(gt_vel, pred_vel, upsampling_factor, output_path):
    """Analyze velocity error over time and across interpolation positions."""
    # Compute MAE per frame
    mae_per_frame = np.abs(gt_vel - pred_vel).mean(axis=(1, 2))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Velocity MAE over time
    frames = np.arange(len(mae_per_frame))
    axes[0].plot(frames, mae_per_frame, 'b-', linewidth=2, marker='o', markersize=4)
    
    # Mark sparse frame boundaries
    for i in range(0, len(mae_per_frame), upsampling_factor):
        axes[0].axvline(i, color='red', linestyle='--', alpha=0.6, 
                        label='Keyframe' if i == 0 else '')
    
    axes[0].set_xlabel('Frame Index', fontsize=12)
    axes[0].set_ylabel('Velocity MAE (rad)', fontsize=12)
    axes[0].set_title('Velocity Error Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Right: Error vs position in interpolation window
    window_positions = frames % upsampling_factor
    
    pos_errors = {}
    for pos in range(upsampling_factor):
        mask = window_positions == pos
        if mask.sum() > 0:
            pos_errors[pos] = mae_per_frame[mask].mean()
    
    positions = list(pos_errors.keys())
    errors = list(pos_errors.values())
    
    colors = ['steelblue' if p not in [0, upsampling_factor-1] else 'coral' for p in positions]
    
    axes[1].bar(positions, errors, color=colors, edgecolor='black')
    axes[1].set_xlabel('Position in Interpolation Window', fontsize=12)
    axes[1].set_ylabel('Average Velocity MAE (rad)', fontsize=12)
    axes[1].set_title(f'Error vs Interpolation Position\n(0,{upsampling_factor-1} = near keyframes)', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return mae_per_frame.mean(), mae_per_frame.min(), mae_per_frame.max()


def plot_velocity_profiles(gt_vel, pred_vel, frame_idx, output_path):
    """Plot lateral and axial velocity profiles for a given frame."""
    H, W = gt_vel.shape[1:]
    
    # Physical dimensions
    total_width_mm = 40.0
    total_depth_mm = 40.0
    
    # Create spatial axes
    lateral_mm = np.linspace(-total_width_mm/2, total_width_mm/2, W)
    axial_mm = np.linspace(0, total_depth_mm, H)
    
    # Extract profiles at center
    center_row = H // 2
    center_col = W // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Velocity map with profile lines
    vmax = max(np.abs(gt_vel[frame_idx]).max(), np.abs(pred_vel[frame_idx]).max()) * 0.8
    extent_mm = [-total_width_mm/2, total_width_mm/2, total_depth_mm, 0]
    
    axes[0, 0].imshow(gt_vel[frame_idx], cmap='jet', vmin=-vmax, vmax=vmax,
                      extent=extent_mm, origin='upper', aspect='equal')
    axes[0, 0].axhline(axial_mm[center_row], color='white', linestyle='--', linewidth=2)
    axes[0, 0].axvline(lateral_mm[center_col], color='yellow', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Ground Truth Velocity\n(with profile lines)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Lateral (mm)')
    axes[0, 0].set_ylabel('Axial (mm)')
    
    # Top-right: Predicted velocity map
    im = axes[0, 1].imshow(pred_vel[frame_idx], cmap='jet', vmin=-vmax, vmax=vmax,
                           extent=extent_mm, origin='upper', aspect='equal')
    axes[0, 1].axhline(axial_mm[center_row], color='white', linestyle='--', linewidth=2)
    axes[0, 1].axvline(lateral_mm[center_col], color='yellow', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Predicted Velocity\n(with profile lines)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Lateral (mm)')
    axes[0, 1].set_ylabel('Axial (mm)')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[0, :].ravel().tolist(), shrink=0.6)
    cbar.set_label('Velocity (rad/frame)', fontsize=11)
    
    # Bottom-left: Lateral profile
    gt_lateral = gt_vel[frame_idx, center_row, :]
    pred_lateral = pred_vel[frame_idx, center_row, :]
    
    axes[1, 0].plot(lateral_mm, gt_lateral, 'b-', linewidth=2, label='Ground Truth')
    axes[1, 0].plot(lateral_mm, pred_lateral, 'r--', linewidth=2, label='Predicted')
    axes[1, 0].set_xlabel('Lateral Position (mm)', fontsize=11)
    axes[1, 0].set_ylabel('Velocity (rad/frame)', fontsize=11)
    axes[1, 0].set_title(f'Lateral Profile (Axial = {axial_mm[center_row]:.1f} mm)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: Axial profile
    gt_axial = gt_vel[frame_idx, :, center_col]
    pred_axial = pred_vel[frame_idx, :, center_col]
    
    axes[1, 1].plot(axial_mm, gt_axial, 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(axial_mm, pred_axial, 'r--', linewidth=2, label='Predicted')
    axes[1, 1].set_xlabel('Axial Position (mm)', fontsize=11)
    axes[1, 1].set_ylabel('Velocity (rad/frame)', fontsize=11)
    axes[1, 1].set_title(f'Axial Profile (Lateral = {lateral_mm[center_col]:.1f} mm)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity Spatial Profiles - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_velocity_animation(gt_vel, pred_vel, output_path, n_frames=None, fps=8):
    """Create animation comparing GT and predicted velocity maps."""
    if n_frames is None:
        n_frames = gt_vel.shape[0]
    n_frames = min(n_frames, gt_vel.shape[0])
    
    # Compute global velocity limits
    vmax = max(np.abs(gt_vel[:n_frames]).max(), np.abs(pred_vel[:n_frames]).max()) * 0.8
    vmin = -vmax
    
    # Physical extent
    total_width_mm = 40.0
    total_depth_mm = 40.0
    extent_mm = [-total_width_mm/2, total_width_mm/2, total_depth_mm, 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialize plots
    im0 = axes[0].imshow(gt_vel[0], cmap='jet', vmin=vmin, vmax=vmax,
                          extent=extent_mm, origin='upper', aspect='equal')
    axes[0].set_title('Ground Truth Velocity', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lateral (mm)')
    axes[0].set_ylabel('Axial (mm)')
    
    im1 = axes[1].imshow(pred_vel[0], cmap='jet', vmin=vmin, vmax=vmax,
                          extent=extent_mm, origin='upper', aspect='equal')
    axes[1].set_title('Predicted Velocity', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lateral (mm)')
    
    # Error map
    error = np.abs(gt_vel[0] - pred_vel[0])
    vmax_err = np.percentile(np.abs(gt_vel - pred_vel), 98)
    
    im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=vmax_err,
                          extent=extent_mm, origin='upper', aspect='equal')
    axes[2].set_title('Velocity Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Lateral (mm)')
    
    # Colorbar
    fig.colorbar(im0, ax=axes[:2], shrink=0.8, pad=0.02, label='Velocity (rad/frame)')
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label='Error (rad)')
    
    title = fig.suptitle('Frame 0', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    def update(frame):
        im0.set_array(gt_vel[frame])
        im1.set_array(pred_vel[frame])
        im2.set_array(np.abs(gt_vel[frame] - pred_vel[frame]))
        title.set_text(f'Frame {frame}')
        return [im0, im1, im2, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000//fps, blit=True
    )
    
    # Save as GIF
    try:
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Warning: Could not save animation: {e}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Velocity Map Visualization')
    parser.add_argument('--model_path', type=str, 
                        default='saved_models/Input_guided_physics_latest_newdata_best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results/velocity_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Gaussian smoothing sigma for velocity maps')
    parser.add_argument('--sparse_step', type=int, default=20,
                        help='Sparse frame step')
    parser.add_argument('--fixed_sparse_frames', type=int, default=3,
                        help='Number of sparse frames')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Resolve paths
    model_path = PROJECT_ROOT / args.model_path
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check model exists
    if not model_path.exists():
        # Try fallback
        fallback_path = PROJECT_ROOT / 'saved_models' / 'Input_guided_physics_trial_2_best_model.pt'
        if fallback_path.exists():
            print(f"Model not found at {model_path}, using fallback: {fallback_path}")
            model_path = fallback_path
        else:
            print(f"ERROR: Model not found at {model_path}")
            print("Available models in saved_models/:")
            for f in (PROJECT_ROOT / 'saved_models').glob('*.pt'):
                print(f"  {f.name}")
            return
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    model, model_config = load_model(str(model_path), device)
    
    upsampling_factor = model_config.get('upsampling_factor', args.sparse_step)
    
    # Collect validation files
    print("\n" + "="*60)
    print("Collecting Validation Files (OOD)")
    print("="*60)
    val_files = collect_validation_files(data_dir)
    print(f"Total validation files: {len(val_files)}")
    
    if len(val_files) == 0:
        print("ERROR: No validation files found!")
        return
    
    # Process samples
    print("\n" + "="*60)
    print(f"Processing {min(args.n_samples, len(val_files))} Samples")
    print("="*60)
    
    all_results = []
    
    for i, file_path in enumerate(val_files[:args.n_samples]):
        print(f"\n[{i+1}/{args.n_samples}] Processing: {file_path.name}")
        
        # Create sample output directory
        sample_name = file_path.stem
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        iq_data = np.load(file_path)
        print(f"  IQ data shape: {iq_data.shape}, dtype: {iq_data.dtype}")
        
        sparse_tensor, dense_tensor, norm_params, sparse_indices = prepare_data_for_model(
            iq_data, 
            sparse_step=args.sparse_step,
            fixed_sparse_frames=args.fixed_sparse_frames
        )
        print(f"  Sparse tensor: {sparse_tensor.shape}, Dense tensor: {dense_tensor.shape}")
        
        # Model inference
        with torch.no_grad():
            pred_tensor = model(sparse_tensor.to(device)).cpu()
        
        # Compute metrics
        psnr = compute_psnr(pred_tensor, dense_tensor)
        print(f"  PSNR: {psnr:.2f} dB")
        
        # Compute Kasai velocity maps
        gt_velocity = compute_kasai_velocity(dense_tensor, sigma=args.sigma)
        pred_velocity = compute_kasai_velocity(pred_tensor, sigma=args.sigma)
        
        velocity_mae = np.abs(gt_velocity - pred_velocity).mean()
        print(f"  Velocity MAE: {velocity_mae:.4f} rad")
        
        # Generate visualizations
        print("  Generating visualizations...")
        
        # 1. Single frame comparison (middle of interpolation)
        mid_frame = gt_velocity.shape[0] // 2
        plot_velocity_comparison(
            gt_velocity, pred_velocity, mid_frame,
            sample_dir / f'velocity_frame_{mid_frame}.png',
            title_suffix=f" ({sample_name})"
        )
        
        # 2. Multi-frame evolution
        T_vel = gt_velocity.shape[0]
        frames_to_show = [0, T_vel//4, T_vel//2, 3*T_vel//4, T_vel-1]
        frames_to_show = [f for f in frames_to_show if f < T_vel]
        plot_velocity_evolution(
            gt_velocity, pred_velocity, frames_to_show,
            sample_dir / 'velocity_evolution.png'
        )
        
        # 3. Error analysis
        mae_mean, mae_min, mae_max = plot_velocity_error_analysis(
            gt_velocity, pred_velocity, upsampling_factor,
            sample_dir / 'velocity_error_analysis.png'
        )
        
        # 4. Spatial profiles
        plot_velocity_profiles(
            gt_velocity, pred_velocity, mid_frame,
            sample_dir / 'velocity_profiles.png'
        )
        
        # 5. Animation (only for first sample to save time)
        if i == 0:
            create_velocity_animation(
                gt_velocity, pred_velocity,
                sample_dir / 'velocity_animation.gif',
                n_frames=min(40, gt_velocity.shape[0]),
                fps=8
            )
        
        all_results.append({
            'file': file_path.name,
            'psnr': psnr,
            'velocity_mae': velocity_mae,
            'mae_mean': mae_mean,
            'mae_min': mae_min,
            'mae_max': mae_max
        })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nModel: {model_path.name}")
    print(f"Upsampling factor: {upsampling_factor}x")
    print(f"Gaussian smoothing σ: {args.sigma}")
    print(f"\nResults ({len(all_results)} samples):")
    print("-"*60)
    
    psnr_values = [r['psnr'] for r in all_results]
    vel_mae_values = [r['velocity_mae'] for r in all_results]
    
    for r in all_results:
        print(f"  {r['file'][:35]:35s} | PSNR: {r['psnr']:6.2f} dB | Vel MAE: {r['velocity_mae']:.4f}")
    
    print("-"*60)
    print(f"  {'Average':35s} | PSNR: {np.mean(psnr_values):6.2f} dB | Vel MAE: {np.mean(vel_mae_values):.4f}")
    print(f"  {'Std Dev':35s} | PSNR: {np.std(psnr_values):6.2f} dB | Vel MAE: {np.std(vel_mae_values):.4f}")
    
    print(f"\nOutput saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
