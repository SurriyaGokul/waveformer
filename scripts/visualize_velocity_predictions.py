#!/usr/bin/env python3
"""
Velocity Prediction Visualization Script for ComplexToRealUNet.

Generates visualizations comparing model predictions against ground truth:
- Side-by-side comparison at selected time frames
- Temporal profile along spatial line
- Error maps and metrics (MAE, RMSE, SSIM)
- Animation of full temporal sequence

Usage:
    python scripts/visualize_velocity_predictions.py \
        --model_path saved_models/velocity_clean_trail1_best.pt \
        --output_dir results/velocity_viz

    # Visualize specific sample
    python scripts/visualize_velocity_predictions.py \
        --model_path saved_models/velocity_clean_trail1_best.pt \
        --sample_idx 0 \
        --save_animation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures.velocity_unet import ComplexToRealUNet
from src.utils.velocity_dataset import create_preprocessed_velocity_dataloaders

# Set matplotlib style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_model(
    model_path: str,
    device: torch.device,
) -> Tuple[ComplexToRealUNet, Dict]:
    """
    Load the trained ComplexToRealUNet model from checkpoint.
    
    Args:
        model_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, config_dict)
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            model_config = config
        else:
            model_config = vars(config) if hasattr(config, '__dict__') else {}
    else:
        # Default config matching train_velocity_preprocessed.py
        model_config = {
            'in_channels': 1,
            'base_channels': 32,
            'encoder_depths': (32, 64, 128, 256),
            'upsampling_factor': 20,
            'use_motion_interpolation': True,
        }
    
    print("Model Configuration:")
    for key in ['upsampling_factor', 'base_channels', 'encoder_depths']:
        if key in model_config:
            print(f"  {key}: {model_config[key]}")
    
    if 'epoch' in checkpoint:
        print(f"  Best epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    # Create model
    encoder_depths = model_config.get('encoder_depths', (32, 64, 128, 256))
    if isinstance(encoder_depths, list):
        encoder_depths = tuple(encoder_depths)
    
    model = ComplexToRealUNet(
        in_channels=model_config.get('in_channels', 1),
        base_channels=model_config.get('base_channels', 32),
        encoder_depths=list(encoder_depths),
        upsampling_factor=model_config.get('upsampling_factor', 20),
        use_motion_interpolation=model_config.get('use_motion_interpolation', True),
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded! Total parameters: {total_params:,}\n")
    
    return model, model_config


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics between prediction and target.
    
    Args:
        pred: Predicted velocity (T, H, W) or (H, W)
        target: Ground truth velocity (T, H, W) or (H, W)
        mask: Optional magnitude mask for weighted metrics
        
    Returns:
        Dictionary with MAE, RMSE, and masked variants
    """
    metrics = {}
    
    # Overall metrics
    diff = pred - target
    metrics['mae'] = float(np.abs(diff).mean())
    metrics['rmse'] = float(np.sqrt((diff ** 2).mean()))
    
    # Max absolute error
    metrics['max_error'] = float(np.abs(diff).max())
    
    # Masked metrics (if mask provided)
    if mask is not None:
        mask_binary = mask > 0.1
        if mask_binary.sum() > 0:
            metrics['masked_mae'] = float(np.abs(diff[mask_binary]).mean())
            metrics['masked_rmse'] = float(np.sqrt((diff[mask_binary] ** 2).mean()))
    
    # Correlation coefficient
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    if pred_flat.std() > 1e-8 and target_flat.std() > 1e-8:
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
        metrics['correlation'] = float(corr)
    
    return metrics


def run_inference(
    model: ComplexToRealUNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Run inference on validation data.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Device for inference
        num_samples: Max samples to process (None for all)
        
    Returns:
        List of dicts with 'input', 'prediction', 'target', 'mask'
    """
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            if num_samples and len(results) >= num_samples:
                break
            
            if len(batch) == 3:
                inputs, targets, masks = batch
            else:
                inputs, targets = batch
                masks = torch.ones_like(targets)
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Match temporal dimensions
            T_target = targets.shape[2]
            if outputs.shape[2] > T_target:
                outputs = outputs[:, :, :T_target, :, :]
            
            # Convert to numpy
            batch_size = inputs.size(0)
            for i in range(batch_size):
                if num_samples and len(results) >= num_samples:
                    break
                
                results.append({
                    'input': inputs[i].cpu().numpy(),  # (2, T_sparse, H, W)
                    'prediction': outputs[i, 0].cpu().numpy(),  # (T_dense, H, W)
                    'target': targets[i, 0].cpu().numpy(),  # (T_dense, H, W)
                    'mask': masks[i, 0].cpu().numpy(),  # (T_dense, H, W)
                })
    
    print(f"Processed {len(results)} samples")
    return results


def visualize_single_frame(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    frame_idx: int,
    save_path: Optional[str] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    smooth_sigma: float = 1.0,
) -> plt.Figure:
    """
    Visualize prediction vs target for a single time frame.
    
    Args:
        pred: Predicted velocity (T, H, W)
        target: Ground truth velocity (T, H, W)
        mask: Magnitude mask (T, H, W)
        frame_idx: Time frame index
        save_path: Path to save figure
        vmin, vmax: Color scale limits
        smooth_sigma: Gaussian smoothing sigma
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Get single frame
    pred_frame = pred[frame_idx]
    target_frame = target[frame_idx]
    mask_frame = mask[frame_idx]
    
    # Apply smoothing
    if smooth_sigma > 0:
        pred_smooth = gaussian_filter(pred_frame, sigma=smooth_sigma)
        target_smooth = gaussian_filter(target_frame, sigma=smooth_sigma)
    else:
        pred_smooth = pred_frame
        target_smooth = target_frame
    
    # Compute error
    error = pred_smooth - target_smooth
    
    # Compute metrics
    metrics = compute_metrics(pred_frame, target_frame, mask_frame)
    
    # Plot ground truth
    im0 = axes[0, 0].imshow(target_smooth, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'Ground Truth (Frame {frame_idx})')
    axes[0, 0].set_xlabel('Width')
    axes[0, 0].set_ylabel('Depth')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax, label='Velocity')
    
    # Plot prediction
    im1 = axes[0, 1].imshow(pred_smooth, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title(f'Prediction (Frame {frame_idx})')
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Depth')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax, label='Velocity')
    
    # Plot error map
    error_abs = np.abs(error)
    im2 = axes[0, 2].imshow(error_abs, cmap='hot', vmin=0, vmax=0.5, aspect='auto')
    axes[0, 2].set_title(f'Absolute Error (MAE: {metrics["mae"]:.4f})')
    axes[0, 2].set_xlabel('Width')
    axes[0, 2].set_ylabel('Depth')
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax, label='|Error|')
    
    # Plot magnitude mask
    im3 = axes[1, 0].imshow(mask_frame, cmap='gray', vmin=0, vmax=1, aspect='auto')
    axes[1, 0].set_title('Magnitude Mask')
    axes[1, 0].set_xlabel('Width')
    axes[1, 0].set_ylabel('Depth')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax, label='Mask')
    
    # Plot masked ground truth
    target_masked = target_smooth * mask_frame
    im4 = axes[1, 1].imshow(target_masked, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1, 1].set_title('Ground Truth (Masked)')
    axes[1, 1].set_xlabel('Width')
    axes[1, 1].set_ylabel('Depth')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax, label='Velocity')
    
    # Plot masked prediction
    pred_masked = pred_smooth * mask_frame
    im5 = axes[1, 2].imshow(pred_masked, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1, 2].set_title('Prediction (Masked)')
    axes[1, 2].set_xlabel('Width')
    axes[1, 2].set_ylabel('Depth')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax, label='Velocity')
    
    # Add metrics text
    metrics_text = (
        f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}"
    )
    if 'masked_mae' in metrics:
        metrics_text += f" | Masked MAE: {metrics['masked_mae']:.4f}"
    if 'correlation' in metrics:
        metrics_text += f" | Corr: {metrics['correlation']:.4f}"
    
    fig.suptitle(metrics_text, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_temporal_profile(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    depth_idx: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize temporal evolution along a horizontal line at given depth.
    
    Args:
        pred: Predicted velocity (T, H, W)
        target: Ground truth velocity (T, H, W)
        mask: Magnitude mask (T, H, W)
        depth_idx: Depth (row) index for profile
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    T, H, W = pred.shape
    
    # Extract profiles at given depth (x-t diagram)
    pred_profile = pred[:, depth_idx, :]  # (T, W)
    target_profile = target[:, depth_idx, :]  # (T, W)
    mask_profile = mask[:, depth_idx, :]  # (T, W)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vmin, vmax = -1.0, 1.0
    
    # Ground truth x-t diagram
    im0 = axes[0, 0].imshow(
        target_profile.T, cmap='jet', vmin=vmin, vmax=vmax,
        aspect='auto', origin='lower', extent=[0, T, 0, W]
    )
    axes[0, 0].set_title(f'Ground Truth (Depth index: {depth_idx})')
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Lateral Position')
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax, label='Velocity')
    
    # Prediction x-t diagram
    im1 = axes[0, 1].imshow(
        pred_profile.T, cmap='jet', vmin=vmin, vmax=vmax,
        aspect='auto', origin='lower', extent=[0, T, 0, W]
    )
    axes[0, 1].set_title(f'Prediction (Depth index: {depth_idx})')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Lateral Position')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax, label='Velocity')
    
    # Error x-t diagram
    error_profile = np.abs(pred_profile - target_profile)
    im2 = axes[1, 0].imshow(
        error_profile.T, cmap='hot', vmin=0, vmax=0.5,
        aspect='auto', origin='lower', extent=[0, T, 0, W]
    )
    axes[1, 0].set_title('Absolute Error')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Lateral Position')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax, label='|Error|')
    
    # Line plot at center lateral position
    center_w = W // 2
    pred_line = pred_profile[:, center_w]
    target_line = target_profile[:, center_w]
    
    axes[1, 1].plot(target_line, 'b-', linewidth=2, label='Ground Truth')
    axes[1, 1].plot(pred_line, 'r--', linewidth=2, label='Prediction')
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].set_title(f'Temporal Profile at Lateral Position {center_w}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_multi_frame(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    frame_indices: List[int],
    save_path: Optional[str] = None,
    smooth_sigma: float = 1.0,
) -> plt.Figure:
    """
    Visualize multiple time frames in a grid.
    
    Args:
        pred: Predicted velocity (T, H, W)
        target: Ground truth velocity (T, H, W)
        mask: Magnitude mask (T, H, W)
        frame_indices: List of frame indices to show
        save_path: Path to save figure
        smooth_sigma: Gaussian smoothing sigma
        
    Returns:
        matplotlib Figure
    """
    n_frames = len(frame_indices)
    fig, axes = plt.subplots(3, n_frames, figsize=(4 * n_frames, 10))
    
    if n_frames == 1:
        axes = axes.reshape(-1, 1)
    
    vmin, vmax = -1.0, 1.0
    
    for i, frame_idx in enumerate(frame_indices):
        pred_frame = pred[frame_idx]
        target_frame = target[frame_idx]
        
        if smooth_sigma > 0:
            pred_frame = gaussian_filter(pred_frame, sigma=smooth_sigma)
            target_frame = gaussian_filter(target_frame, sigma=smooth_sigma)
        
        error = np.abs(pred_frame - target_frame)
        
        # Ground truth
        im0 = axes[0, i].imshow(target_frame, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        axes[0, i].set_title(f'GT (t={frame_idx})')
        axes[0, i].axis('off')
        
        # Prediction
        im1 = axes[1, i].imshow(pred_frame, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, i].set_title(f'Pred (t={frame_idx})')
        axes[1, i].axis('off')
        
        # Error
        im2 = axes[2, i].imshow(error, cmap='hot', vmin=0, vmax=0.5, aspect='auto')
        mae = error.mean()
        axes[2, i].set_title(f'Error (MAE={mae:.3f})')
        axes[2, i].axis('off')
    
    # Add colorbars
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_animation(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    fps: int = 10,
    smooth_sigma: float = 1.0,
) -> None:
    """
    Create animation comparing prediction vs ground truth over time.
    
    Args:
        pred: Predicted velocity (T, H, W)
        target: Ground truth velocity (T, H, W)
        save_path: Path to save animation (gif or mp4)
        fps: Frames per second
        smooth_sigma: Gaussian smoothing sigma
    """
    T, H, W = pred.shape
    vmin, vmax = -1.0, 1.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialize plots
    if smooth_sigma > 0:
        pred_smooth = np.array([gaussian_filter(pred[t], sigma=smooth_sigma) for t in range(T)])
        target_smooth = np.array([gaussian_filter(target[t], sigma=smooth_sigma) for t in range(T)])
    else:
        pred_smooth = pred
        target_smooth = target
    
    im0 = axes[0].imshow(target_smooth[0], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(pred_smooth[0], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    error0 = np.abs(pred_smooth[0] - target_smooth[0])
    im2 = axes[2].imshow(error0, cmap='hot', vmin=0, vmax=0.5, aspect='auto')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    title = fig.suptitle(f'Frame 0 / {T}')
    
    plt.tight_layout()
    
    def update(frame):
        im0.set_data(target_smooth[frame])
        im1.set_data(pred_smooth[frame])
        error = np.abs(pred_smooth[frame] - target_smooth[frame])
        im2.set_data(error)
        mae = error.mean()
        title.set_text(f'Frame {frame} / {T} (MAE: {mae:.4f})')
        return im0, im1, im2, title
    
    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=False
    )
    
    # Save animation
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=fps)
    else:
        anim.save(save_path, writer='ffmpeg', fps=fps)
    
    plt.close(fig)
    print(f"Saved animation: {save_path}")


def compute_aggregate_metrics(
    results: List[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute aggregate metrics across all samples.
    
    Args:
        results: List of result dicts from run_inference
        
    Returns:
        Dictionary with aggregate metrics
    """
    all_mae = []
    all_rmse = []
    all_masked_mae = []
    all_corr = []
    
    for r in results:
        metrics = compute_metrics(r['prediction'], r['target'], r['mask'])
        all_mae.append(metrics['mae'])
        all_rmse.append(metrics['rmse'])
        if 'masked_mae' in metrics:
            all_masked_mae.append(metrics['masked_mae'])
        if 'correlation' in metrics:
            all_corr.append(metrics['correlation'])
    
    agg_metrics = {
        'mean_mae': float(np.mean(all_mae)),
        'std_mae': float(np.std(all_mae)),
        'mean_rmse': float(np.mean(all_rmse)),
        'std_rmse': float(np.std(all_rmse)),
    }
    
    if all_masked_mae:
        agg_metrics['mean_masked_mae'] = float(np.mean(all_masked_mae))
        agg_metrics['std_masked_mae'] = float(np.std(all_masked_mae))
    
    if all_corr:
        agg_metrics['mean_correlation'] = float(np.mean(all_corr))
        agg_metrics['std_correlation'] = float(np.std(all_corr))
    
    return agg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Visualize velocity predictions from ComplexToRealUNet"
    )
    
    # Paths
    parser.add_argument(
        '--model_path', type=str,
        default='saved_models/velocity_clean_trail1_best.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/preprocessed',
        help='Path to preprocessed data directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/velocity_predictions_viz',
        help='Output directory for visualizations'
    )
    
    # Sample selection
    parser.add_argument(
        '--sample_idx', type=int, default=None,
        help='Specific sample index to visualize (default: first few)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=5,
        help='Number of samples to process'
    )
    
    # Visualization options
    parser.add_argument(
        '--frame_indices', type=int, nargs='+', default=None,
        help='Specific frame indices to visualize (default: evenly spaced)'
    )
    parser.add_argument(
        '--smooth_sigma', type=float, default=1.0,
        help='Gaussian smoothing sigma for visualization'
    )
    parser.add_argument(
        '--save_animation', action='store_true',
        help='Save animation of temporal evolution'
    )
    parser.add_argument(
        '--use_val', action='store_true', default=True,
        help='Use validation set (default: True)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model(args.model_path, device)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = create_preprocessed_velocity_dataloaders(
        preprocessed_dir=args.data_dir,
        batch_size=1,  # Process one at a time for visualization
        num_workers=0,
        return_mask=True,
    )
    
    dataloader = val_loader if args.use_val else train_loader
    print(f"Using {'validation' if args.use_val else 'training'} set: {len(dataloader.dataset)} samples\n")
    
    # Run inference
    num_to_process = args.num_samples if args.sample_idx is None else 1
    if args.sample_idx is not None:
        # Create a subset dataloader for specific sample
        from torch.utils.data import Subset
        subset = Subset(dataloader.dataset, [args.sample_idx])
        dataloader = torch.utils.data.DataLoader(subset, batch_size=1)
    
    results = run_inference(model, dataloader, device, num_samples=num_to_process)
    
    # Compute aggregate metrics
    print("\nAggregate Metrics:")
    print("-" * 50)
    agg_metrics = compute_aggregate_metrics(results)
    for k, v in agg_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(agg_metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Generate visualizations for each sample
    print("\nGenerating visualizations...")
    for i, result in enumerate(results):
        sample_dir = output_dir / f'sample_{i}'
        sample_dir.mkdir(exist_ok=True)
        
        pred = result['prediction']
        target = result['target']
        mask = result['mask']
        T = pred.shape[0]
        
        # Determine frame indices
        if args.frame_indices is not None:
            frame_indices = [f for f in args.frame_indices if f < T]
        else:
            # Evenly spaced frames
            frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        
        # Single frame visualizations for key frames
        for frame_idx in [T // 4, T // 2, 3 * T // 4]:
            if frame_idx < T:
                visualize_single_frame(
                    pred, target, mask, frame_idx,
                    save_path=str(sample_dir / f'frame_{frame_idx:03d}.png'),
                    smooth_sigma=args.smooth_sigma
                )
        
        # Multi-frame comparison
        visualize_multi_frame(
            pred, target, mask, frame_indices,
            save_path=str(sample_dir / 'multi_frame_comparison.png'),
            smooth_sigma=args.smooth_sigma
        )
        
        # Temporal profile
        H = pred.shape[1]
        for depth_idx in [H // 4, H // 2, 3 * H // 4]:
            visualize_temporal_profile(
                pred, target, mask, depth_idx,
                save_path=str(sample_dir / f'temporal_profile_depth_{depth_idx}.png')
            )
        
        # Animation
        if args.save_animation:
            create_animation(
                pred, target,
                save_path=str(sample_dir / 'animation.gif'),
                fps=10,
                smooth_sigma=args.smooth_sigma
            )
        
        # Per-sample metrics
        sample_metrics = compute_metrics(pred, target, mask)
        sample_metrics_path = sample_dir / 'sample_metrics.json'
        with open(sample_metrics_path, 'w') as f:
            json.dump(sample_metrics, f, indent=2)
        
        print(f"Sample {i}: MAE={sample_metrics['mae']:.4f}, RMSE={sample_metrics['rmse']:.4f}")
    
    print(f"\nVisualization complete! Results saved to: {output_dir}")
    plt.close('all')


if __name__ == '__main__':
    main()
