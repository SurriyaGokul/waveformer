"""
Flow and Motion Weight Visualization for Improved Motion-Aware Model.

This script loads the trained model and visualizes:
1. Estimated optical flow fields (magnitude and direction)
2. Motion weight maps (where the model uses motion vs linear interpolation)
3. Comparison of linear vs motion-based interpolated frames

Usage:
    python -m src.tools.visualize_flows --checkpoint best_model_motion_aware.pt --num_samples 3
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.train_improved_motion import create_improved_motion_model
from src.utils.data_loader import SWEDataModule, DataConfig


def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    Convert optical flow to RGB color image using HSV color wheel.
    
    Args:
        flow: Flow field of shape (2, H, W) - (flow_x, flow_y)
        max_flow: Maximum flow magnitude for normalization
    
    Returns:
        RGB image of shape (H, W, 3)
    """
    flow_x = flow[0]
    flow_y = flow[1]
    
    # Compute magnitude and angle
    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    angle = np.arctan2(flow_y, flow_x)
    
    # Normalize magnitude
    if max_flow is None:
        max_flow = np.max(magnitude) + 1e-8
    magnitude = np.clip(magnitude / max_flow, 0, 1)
    
    # Convert angle to hue (0-1)
    hue = (angle + np.pi) / (2 * np.pi)
    
    # Create HSV image
    hsv = np.zeros((*flow_x.shape, 3))
    hsv[..., 0] = hue  # Hue = direction
    hsv[..., 1] = 1.0  # Saturation = 1
    hsv[..., 2] = magnitude  # Value = magnitude
    
    # Convert to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    
    return rgb


def visualize_sample(
    model: torch.nn.Module,
    sparse: torch.Tensor,
    dense: torch.Tensor,
    sample_idx: int,
    output_dir: Path,
    device: torch.device
):
    """
    Visualize flows and motion weights for a single sample.
    
    Args:
        model: Trained model
        sparse: Sparse input tensor (1, T_sparse, 2, C, H, W)
        dense: Dense ground truth tensor (1, T_dense, 2, C, H, W)
        sample_idx: Sample index for naming
        output_dir: Directory to save visualizations
        device: Device to run on
    """
    model.eval()
    
    sparse = sparse.to(device)
    dense = dense.to(device)
    
    with torch.no_grad():
        # Get encoded features
        t_sparse = sparse.shape[1]
        encoded_frames = []
        for t in range(t_sparse):
            frame = sparse[:, t]
            encoded, _ = model.encoder(frame)
            encoded_frames.append(encoded)
        sparse_encoded = torch.stack(encoded_frames, dim=1)
        
        # Project to hidden space
        sparse_encoded = model.get_projected_features(sparse_encoded)
        
        # Get flow estimator and motion interpolation
        flow_estimator = model.get_flow_estimator()
        motion_interp = model.get_motion_interpolation()
        
        # Run model forward to populate caches
        pred = model(sparse)
        
        # Get cached flows and motion weights
        cached_flows = motion_interp.get_cached_flows()
        motion_stats = motion_interp.get_motion_weight_stats()
        
        # Get flow statistics
        flow_stats_list = []
        for t in range(t_sparse - 1):
            frame_t = sparse_encoded[:, t]
            frame_t1 = sparse_encoded[:, t + 1]
            forward_flow, backward_flow = flow_estimator(frame_t, frame_t1)
            stats = flow_estimator.get_flow_stats(forward_flow)
            flow_stats_list.append({
                'forward_flow': forward_flow.cpu().numpy(),
                'backward_flow': backward_flow.cpu().numpy(),
                'stats': stats
            })
    
    # Create output directory for this sample
    sample_dir = output_dir / f"sample_{sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # =====================
    # 1. Flow Visualization
    # =====================
    n_pairs = len(flow_stats_list)
    fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8))
    if n_pairs == 1:
        axes = axes.reshape(2, 1)
    
    max_flow_global = max(fs['stats']['flow_max'] for fs in flow_stats_list) + 1e-8
    
    for t, flow_data in enumerate(flow_stats_list):
        forward_flow = flow_data['forward_flow'][0]  # (2, H, W)
        backward_flow = flow_data['backward_flow'][0]
        
        # Forward flow
        forward_rgb = flow_to_color(forward_flow, max_flow_global)
        axes[0, t].imshow(forward_rgb)
        axes[0, t].set_title(f'Forward Flow t={t}→{t+1}\nmax={flow_data["stats"]["flow_max"]:.3f}')
        axes[0, t].axis('off')
        
        # Backward flow
        backward_rgb = flow_to_color(backward_flow, max_flow_global)
        axes[1, t].imshow(backward_rgb)
        axes[1, t].set_title(f'Backward Flow t={t+1}→{t}')
        axes[1, t].axis('off')
    
    plt.suptitle(f'Sample {sample_idx}: Optical Flow Fields (color=direction, brightness=magnitude)')
    plt.tight_layout()
    plt.savefig(sample_dir / 'flow_fields.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================
    # 2. Flow Magnitude Heatmaps
    # ============================
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4))
    if n_pairs == 1:
        axes = [axes]
    
    for t, flow_data in enumerate(flow_stats_list):
        forward_flow = flow_data['forward_flow'][0]
        magnitude = np.sqrt(forward_flow[0] ** 2 + forward_flow[1] ** 2)
        
        im = axes[t].imshow(magnitude, cmap='hot', vmin=0, vmax=max_flow_global)
        axes[t].set_title(f't={t}→{t+1}, mean={magnitude.mean():.3f}')
        axes[t].axis('off')
        plt.colorbar(im, ax=axes[t], fraction=0.046)
    
    plt.suptitle(f'Sample {sample_idx}: Flow Magnitude (pixels)')
    plt.tight_layout()
    plt.savefig(sample_dir / 'flow_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================
    # 3. Motion Weight Visualization
    # ============================
    # Get motion weights from cached data
    cached_motion_weights = motion_interp.cached_motion_weights
    
    if cached_motion_weights:
        n_weights = len(cached_motion_weights)
        fig, axes = plt.subplots(1, n_weights, figsize=(4 * n_weights, 4))
        if n_weights == 1:
            axes = [axes]
        
        for i, weight in enumerate(cached_motion_weights):
            weight_np = weight[0, 0].cpu().numpy()  # (H, W)
            im = axes[i].imshow(weight_np, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[i].set_title(f'Pair {i}: mean={weight_np.mean():.3f}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        plt.suptitle(f'Sample {sample_idx}: Motion Weight Maps\n(1=motion, 0=linear, init bias ~0.12)')
        plt.tight_layout()
        plt.savefig(sample_dir / 'motion_weights.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # ============================
    # 4. Input/Output Comparison
    # ============================
    # Get magnitude of complex data
    # Data format: (T, 2, C, H, W) where dim=1 is (Real, Imag)
    def get_magnitude(x):
        # x shape: (2, C, H, W) for single frame
        real = x[0]  # (C, H, W)
        imag = x[1]  # (C, H, W)
        return np.sqrt(real ** 2 + imag ** 2)
    
    sparse_np = sparse[0].cpu().numpy()  # (T_sparse, 2, C, H, W)
    dense_np = dense[0].cpu().numpy()    # (T_dense, 2, C, H, W)
    pred_np = pred[0].cpu().numpy()      # (T_dense, 2, C, H, W)
    
    # Select frames to visualize
    t_dense = pred_np.shape[0]
    frame_indices = [0, t_dense // 4, t_dense // 2, 3 * t_dense // 4, t_dense - 1]
    
    fig, axes = plt.subplots(3, len(frame_indices), figsize=(3 * len(frame_indices), 9))
    
    for i, t in enumerate(frame_indices):
        # Ground truth
        gt_mag = get_magnitude(dense_np[t])
        if gt_mag.ndim == 3:
            gt_mag = gt_mag[0]
        im = axes[0, i].imshow(gt_mag, cmap='seismic', vmin=-0.3, vmax=0.3)
        axes[0, i].set_title(f'GT t={t}')
        axes[0, i].axis('off')
        
        # Prediction
        pred_mag = get_magnitude(pred_np[t])
        if pred_mag.ndim == 3:
            pred_mag = pred_mag[0]
        axes[1, i].imshow(pred_mag, cmap='seismic', vmin=-0.3, vmax=0.3)
        axes[1, i].set_title(f'Pred t={t}')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(gt_mag - pred_mag)
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
        axes[2, i].set_title(f'|Error| t={t}')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Ground Truth', fontsize=12)
    axes[1, 0].set_ylabel('Prediction', fontsize=12)
    axes[2, 0].set_ylabel('Abs Error', fontsize=12)
    
    plt.suptitle(f'Sample {sample_idx}: Displacement Comparison')
    plt.tight_layout()
    plt.savefig(sample_dir / 'prediction_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================
    # 5. Save Numerical Statistics
    # ============================
    stats_file = sample_dir / 'statistics.txt'
    with open(stats_file, 'w') as f:
        f.write(f"Sample {sample_idx} Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Flow Statistics per Pair:\n")
        for t, flow_data in enumerate(flow_stats_list):
            stats = flow_data['stats']
            f.write(f"  Pair {t}→{t+1}:\n")
            f.write(f"    Mean magnitude: {stats['flow_mean']:.6f}\n")
            f.write(f"    Max magnitude: {stats['flow_max']:.6f}\n")
            f.write(f"    Std: {stats.get('flow_std', 0):.6f}\n")
            f.write(f"    Learnable max_flow: {stats.get('max_flow_param', 2.0):.4f}\n")
        
        f.write("\nMotion Weight Statistics:\n")
        f.write(f"  Mean: {motion_stats.get('motion_weight_mean', 0):.4f}\n")
        f.write(f"  Max: {motion_stats.get('motion_weight_max', 0):.4f}\n")
        f.write(f"  Min: {motion_stats.get('motion_weight_min', 0):.4f}\n")
        
        # Compute PSNR
        pred_mag_full = get_magnitude(pred_np)
        gt_mag_full = get_magnitude(dense_np)
        mse = np.mean((pred_mag_full - gt_mag_full) ** 2)
        max_val = np.max(gt_mag_full)
        psnr = 20 * np.log10(max_val / np.sqrt(mse + 1e-10))
        f.write(f"\nReconstruction Quality:\n")
        f.write(f"  PSNR: {psnr:.2f} dB\n")
        f.write(f"  MSE: {mse:.6f}\n")
    
    print(f"  Saved visualizations to {sample_dir}")
    return {
        'flow_stats': flow_stats_list,
        'motion_stats': motion_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize flows and motion weights')
    parser.add_argument('--checkpoint', type=str, default='best_model_motion_aware.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/CIRS_phantom_dataset',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results/flow_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to use')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = create_improved_motion_model(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        num_levels=4,
        lstm_hidden_channels=128,
        lstm_num_layers=1,
        upsampling_factor=20,
        flow_hidden_channels=64,
        flow_scale=0.5,
        use_residual_fusion=True
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    data_config = DataConfig(
        data_dir=args.data_dir,
        sparse_frame_step=20,
        normalization='per_sample',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    data_module = SWEDataModule(data_config)
    data_module.setup()
    
    # Get appropriate dataset
    if args.split == 'train':
        dataset = data_module.train_dataset
    elif args.split == 'val':
        dataset = data_module.val_dataset
    else:
        dataset = data_module.test_dataset
    
    print(f"Using {args.split} split with {len(dataset)} samples")
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nVisualizing {num_samples} samples...")
    
    all_stats = []
    for i in range(num_samples):
        print(f"\nProcessing sample {i + 1}/{num_samples}...")
        sample = dataset[i]
        sparse = sample['sparse'].unsqueeze(0)  # Add batch dim
        dense = sample['dense'].unsqueeze(0)
        
        stats = visualize_sample(model, sparse, dense, i, output_dir, device)
        all_stats.append(stats)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics Across All Samples")
    print("=" * 60)
    
    all_flow_means = []
    all_flow_maxs = []
    all_motion_means = []
    
    for i, stats in enumerate(all_stats):
        for pair_stats in stats['flow_stats']:
            all_flow_means.append(pair_stats['stats']['flow_mean'])
            all_flow_maxs.append(pair_stats['stats']['flow_max'])
        all_motion_means.append(stats['motion_stats'].get('motion_weight_mean', 0))
    
    print(f"\nFlow Magnitude:")
    print(f"  Mean across all pairs: {np.mean(all_flow_means):.4f}")
    print(f"  Max across all pairs: {np.max(all_flow_maxs):.4f}")
    
    print(f"\nMotion Weight:")
    print(f"  Mean across samples: {np.mean(all_motion_means):.4f}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Flow Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples visualized: {num_samples}\n\n")
        f.write(f"Flow Magnitude:\n")
        f.write(f"  Mean: {np.mean(all_flow_means):.6f}\n")
        f.write(f"  Max: {np.max(all_flow_maxs):.6f}\n\n")
        f.write(f"Motion Weight:\n")
        f.write(f"  Mean: {np.mean(all_motion_means):.4f}\n")


if __name__ == '__main__':
    main()
