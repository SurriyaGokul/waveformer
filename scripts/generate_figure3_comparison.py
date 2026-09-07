import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(data_dir: Path = None):
    """
    Load ground truth and prediction data.
    Modify this function to point to your actual data files.
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
    gt_files = list(data_dir.glob("dense_*.npy"))
    pred_files = list(data_dir.glob("pred_*.npy"))
    
    if gt_files and pred_files:
        gt_data = np.load(gt_files[0])
        pred_data = np.load(pred_files[0])
        return gt_data, pred_data
    
    return generate_demo_data()


def generate_demo_data():
    """
    Generate synthetic shear wave data for demonstration.
    """
    H, W = 128, 128
    
    # Create synthetic shear wave pattern (ground truth)
    x = np.linspace(0, 4 * np.pi, W)
    y = np.linspace(0, 4 * np.pi, H)
    X, Y = np.meshgrid(x, y)
    
    # Simulate shear wave propagation (V-shaped pattern)
    wave_center = W // 2
    t = 10  # Frame index
    
    # Create V-shaped wavefront
    left_wave = np.sin(X - 0.3 * Y + t * 0.5) * np.exp(-0.02 * (X - wave_center + Y * 0.3)**2)
    right_wave = np.sin(X + 0.3 * Y + t * 0.5) * np.exp(-0.02 * (X - wave_center - Y * 0.3)**2)
    
    gt = (left_wave + right_wave) * 0.15  # Scale to realistic displacement
    
    # Add some texture/noise
    noise = np.random.randn(H, W) * 0.02
    gt = gt + noise
    
    # Prediction (slightly different but close)
    pred = gt + np.random.randn(H, W) * 0.015  # Small reconstruction error
    
    return gt, pred


def iq_to_displacement(data, scale_to_mm: float = 1.0, frame_idx: int = 10):
    """
    Convert IQ data to displacement magnitude.
    
    Args:
        data: Complex IQ data or magnitude data
        scale_to_mm: Scaling factor to convert to mm
        frame_idx: Frame index to extract
    
    Returns:
        Displacement in mm (real-valued array)
    """
    # Handle complex numpy arrays
    if np.iscomplexobj(data):
        if data.ndim == 3:  # (T, H, W) complex
            frame_idx = min(frame_idx, data.shape[0] - 1)
            frame_data = data[frame_idx]
            # Use phase difference for displacement
            if frame_idx > 0:
                phase_curr = np.angle(frame_data)
                phase_prev = np.angle(data[frame_idx - 1])
                displacement = np.angle(np.exp(1j * (phase_curr - phase_prev)))
            else:
                displacement = np.abs(frame_data)
            return displacement.real * scale_to_mm
        elif data.ndim == 2:  # (H, W) complex
            # Use phase or magnitude
            displacement = np.angle(data)  # or np.abs(data)
            return displacement.real * scale_to_mm
    
    # Handle real-valued arrays
    if data.ndim == 4:  # (T, 2, H, W) - real/imag format
        frame_idx = min(frame_idx, data.shape[0] - 1)
        real = data[frame_idx, 0]
        imag = data[frame_idx, 1]
        magnitude = np.sqrt(real**2 + imag**2)
        
        # Convert phase difference to displacement
        if frame_idx > 0:
            real_prev = data[frame_idx - 1, 0]
            imag_prev = data[frame_idx - 1, 1]
            
            # Phase difference method
            phase_curr = np.arctan2(imag, real)
            phase_prev = np.arctan2(imag_prev, real_prev)
            phase_diff = np.angle(np.exp(1j * (phase_curr - phase_prev)))
            
            displacement = phase_diff * scale_to_mm
        else:
            displacement = magnitude * scale_to_mm
            
        return displacement.astype(np.float64)
    
    elif data.ndim == 3:  # (T, H, W) real
        frame_idx = min(frame_idx, data.shape[0] - 1)
        return data[frame_idx].astype(np.float64) * scale_to_mm
    
    elif data.ndim == 2:  # (H, W) - already single frame
        return data.astype(np.float64) * scale_to_mm
    
    return np.abs(data).astype(np.float64) * scale_to_mm


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def plot_single_frame_comparison(gt, pred_ig, pred_base, frame_idx, save_path=None):
    """
    Create a publication-quality single frame comparison with physical units (mm).
    Style: Green Background (Jet), Symmetric Scale.
    """
    has_baseline = pred_base is not None
    
    # ==========================================
    # 1. PHYSICAL CALIBRATION (Adjust these!)
    # ==========================================
    # Assuming standard phantom dimensions ~40mm x 40mm
    # If you know the exact pitch, calculate: width_mm = pitch * num_pixels_width
    height_px, width_px = gt[frame_idx].shape
    
    total_depth_mm = 40.0
    total_width_mm = 40.0
    
    # Extent = [Left, Right, Bottom, Top] (Top is 0 for ultrasound)
    extent_mm = [0, total_width_mm, total_depth_mm, 0]

    # ==========================================
    # 2. SETUP FIGURE
    # ==========================================
    if has_baseline:
        fig = plt.figure(figsize=(20, 5), constrained_layout=True)
        # 4 columns: GT, Baseline, Proposed, Error
        gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], figure=fig)
    else:
        fig = plt.figure(figsize=(16, 5), constrained_layout=True)
        # 3 columns: GT, Proposed, Error
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], figure=fig)
    
    # Color limits - Symmetric for Green Zero effect
    # Adjust 0.3 if your waves are weaker/stronger
    vmax = 0.3 
    vmin = -vmax
    cmap = 'jet'
    
    # Font sizes
    fs_title = 14
    fs_label = 12
    fs_tick = 10

    # ==========================================
    # 3. PLOTTING PANELS
    # ==========================================
    
    # --- Panel 1: Ground Truth ---
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(gt[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax, 
                    extent=extent_mm, origin='upper')
    ax0.set_title('Ground Truth', fontsize=fs_title, fontweight='bold')
    ax0.set_xlabel('Lateral (mm)', fontsize=fs_label)
    ax0.set_ylabel('Axial (mm)', fontsize=fs_label)
    ax0.tick_params(labelsize=fs_tick)
    
    current_col = 1
    
    # --- Panel 2: Baseline (Optional) ---
    if has_baseline:
        ax1 = fig.add_subplot(gs[current_col])
        ax1.imshow(pred_base[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent_mm, origin='upper')
        ax1.set_title('Baseline\n(Bottleneck Flow)', fontsize=fs_title, fontweight='bold')
        ax1.set_xlabel('Lateral (mm)', fontsize=fs_label)
        # Hide Y ticks as they match Panel 1
        ax1.tick_params(axis='y', labelleft=False)
        ax1.tick_params(labelsize=fs_tick)
        current_col += 1
    
    # --- Panel 3: Proposed Method ---
    ax2 = fig.add_subplot(gs[current_col])
    ax2.imshow(pred_ig[frame_idx], cmap=cmap, vmin=vmin, vmax=vmax, 
               extent=extent_mm, origin='upper')
    ax2.set_title('Input-Guided\n(High-Res Flow)', fontsize=fs_title, fontweight='bold')
    ax2.set_xlabel('Lateral (mm)', fontsize=fs_label)
    ax2.tick_params(axis='y', labelleft=False)
    ax2.tick_params(labelsize=fs_tick)
    current_col += 1
    
    # --- Panel 4: Absolute Error ---
    ax3 = fig.add_subplot(gs[current_col])
    error = np.abs(gt[frame_idx] - pred_ig[frame_idx])
    
    # Auto-scale error max to 99th percentile to avoid outlier speckles
    vmax_err = np.percentile(error, 99)
    
    im_err = ax3.imshow(error, cmap='hot', vmin=0, vmax=vmax_err, 
                        extent=extent_mm, origin='upper')
    ax3.set_title(f'Absolute Error\nMAE: {error.mean():.4f}', fontsize=fs_title, fontweight='bold')
    ax3.set_xlabel('Lateral (mm)', fontsize=fs_label)
    ax3.tick_params(axis='y', labelleft=False)
    ax3.tick_params(labelsize=fs_tick)
    current_col += 1
    
    # ==========================================
    # 4. COLORBARS
    # ==========================================
    
    # Colorbar for Displacement (Shared)
    # Attaching to the last plot often looks best, or use the GridSpec slot
    cbar_ax = fig.add_subplot(gs[current_col])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Displacement (mm)', fontsize=fs_label, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=fs_tick)
    # Set explicit ticks to show symmetry
    cbar.set_ticks([-vmax, -vmax/2, 0, vmax/2, vmax])
    
    # Note: If you want a separate colorbar for Error, you can add another column 
    # to GridSpec or inset it. For now, this follows your original logic 
    # of one main bar, but be aware the Error map scale (0 to ~0.05) 
    # is different from the Displacement scale (-0.3 to 0.3).
    # Ideally, add a second colorbar:
    
    # Optional: Add small colorbar for error map inside or next to the error plot
    divider = make_axes_locatable(ax3)
    cax_err = divider.append_axes("right", size="5%", pad=0.05)
    cbar_err = fig.colorbar(im_err, cax=cax_err)
    cbar_err.set_label('Error (mm)', fontsize=10)
    
    # Global Title
    plt.suptitle(f'Frame {frame_idx} (Interpolated)', fontsize=16, fontweight='bold', y=1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def main():
    """Main function to generate Figure 3."""
    
    # Output directory
    output_dir = project_root / "report" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    gt_data, pred_data = load_data()
    
    # Convert to displacement in mm
    # Adjust scale_to_mm based on your actual data calibration
    scale_to_mm = 0.2  # Modify this if needed
    
    gt_disp = iq_to_displacement(gt_data, scale_to_mm=scale_to_mm)
    pred_disp = iq_to_displacement(pred_data, scale_to_mm=scale_to_mm)
    
    print(f"Ground Truth shape: {gt_disp.shape}")
    print(f"Prediction shape: {pred_disp.shape}")
    
    # Generate main figure
    output_path = output_dir / "figure3_displacement_comparison.png"
    plot_single_frame_comparison(gt_disp, pred_disp, output_path)
    
    # Generate alternative version
    output_path_alt = output_dir / "figure3_displacement_comparison_alt.png"
    
    print("\n" + "=" * 60)
    print("Figure 3 generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
