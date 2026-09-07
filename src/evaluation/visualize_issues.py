"""
Visualization and Analysis of Training Issues
Generates plots to understand the training dynamics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re

def parse_training_log(log_path):
    """Parse training log and extract metrics."""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    train_loss = []
    train_psnr = []
    val_loss = []
    val_psnr = []
    rho = []
    mu = []
    eta = []
    batch_loss = []
    batch_data_loss = []
    batch_psnr = []
    
    for line in lines:
        # Parse: Train - Loss: 292.6894, PSNR: 26.06 dB
        if 'Train - Loss:' in line:
            parts = line.split('Train - Loss:')[1]
            loss_val = float(parts.split(',')[0].strip())
            psnr_val = float(parts.split('PSNR:')[1].replace('dB', '').strip())
            train_loss.append(loss_val)
            train_psnr.append(psnr_val)
        
        # Parse: Val   - Loss: 312.3490, PSNR: 25.68 dB
        elif 'Val   - Loss:' in line:
            parts = line.split('Val   - Loss:')[1]
            loss_val = float(parts.split(',')[0].strip())
            psnr_val = float(parts.split('PSNR:')[1].replace('dB', '').strip())
            val_loss.append(loss_val)
            val_psnr.append(psnr_val)
        
        # Parse: Learned physics: rho=1000.0000, mu=1.0000, eta=0.1000
        elif 'Learned physics:' in line:
            parts = line.split('Learned physics:')[1]
            rho_val = float(parts.split('rho=')[1].split(',')[0])
            mu_val = float(parts.split('mu=')[1].split(',')[0])
            eta_val = float(parts.split('eta=')[1].strip())
            rho.append(rho_val)
            mu.append(mu_val)
            eta.append(eta_val)
        
        # Parse: Batch 0/50 - Loss: 296.5099, Data: 3.6248, PSNR: 26.00 dB
        elif 'Batch' in line and 'Loss:' in line and 'Data:' in line:
            try:
                loss_val = float(line.split('Loss:')[1].split(',')[0].strip())
                data_val = float(line.split('Data:')[1].split(',')[0].strip())
                psnr_val = float(line.split('PSNR:')[1].replace('dB', '').strip())
                batch_loss.append(loss_val)
                batch_data_loss.append(data_val)
                batch_psnr.append(psnr_val)
            except:
                pass
    
    return {
        'train_loss': train_loss,
        'train_psnr': train_psnr,
        'val_loss': val_loss,
        'val_psnr': val_psnr,
        'rho': rho,
        'mu': mu,
        'eta': eta,
        'batch_loss': batch_loss,
        'batch_data_loss': batch_data_loss,
        'batch_psnr': batch_psnr,
    }


def create_analysis_figure(data):
    """Create comprehensive analysis figure."""
    fig = plt.figure(figsize=(16, 12))
    
    epochs = range(1, len(data['train_loss']) + 1)
    
    # 1. Training and Validation Loss
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs, data['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, data['val_loss'], 'r-s', label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Loss Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotation about stagnation
    loss_change = (data['train_loss'][-1] - data['train_loss'][0]) / data['train_loss'][0] * 100
    ax1.annotate(f'Change: {loss_change:.1f}%', 
                 xy=(0.7, 0.9), xycoords='axes fraction',
                 fontsize=10, color='red' if abs(loss_change) < 5 else 'green')
    
    # 2. PSNR
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, data['train_psnr'], 'b-o', label='Train PSNR', markersize=4)
    ax2.plot(epochs, data['val_psnr'], 'r-s', label='Val PSNR', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR Progression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Physics Parameters
    ax3 = fig.add_subplot(2, 3, 3)
    physics_epochs = range(1, len(data['rho']) + 1)
    ax3.plot(physics_epochs, data['rho'], 'g-o', label='ρ (density)', markersize=4)
    ax3.set_xlabel('Checkpoint')
    ax3.set_ylabel('ρ value')
    ax3.set_title('Physics Parameter: ρ (Density)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Check if parameters changed
    if len(data['rho']) > 0:
        rho_change = data['rho'][-1] - data['rho'][0]
        status = "⚠️ NOT CHANGING" if abs(rho_change) < 1e-4 else "✓ Updating"
        ax3.annotate(status, xy=(0.5, 0.1), xycoords='axes fraction',
                     fontsize=12, color='red' if abs(rho_change) < 1e-4 else 'green',
                     ha='center')
    
    # 4. mu and eta
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(physics_epochs, data['mu'], 'b-o', label='μ (shear modulus)', markersize=4)
    ax4.plot(physics_epochs, data['eta'], 'r-s', label='η (viscosity)', markersize=4)
    ax4.set_xlabel('Checkpoint')
    ax4.set_ylabel('Value')
    ax4.set_title('Physics Parameters: μ and η')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Batch-level Loss Variance
    ax5 = fig.add_subplot(2, 3, 5)
    batch_nums = range(len(data['batch_loss']))
    ax5.scatter(batch_nums, data['batch_loss'], alpha=0.5, s=10, c='blue')
    ax5.axhline(y=np.mean(data['batch_loss']), color='red', linestyle='--', label=f'Mean: {np.mean(data["batch_loss"]):.1f}')
    ax5.set_xlabel('Batch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Batch-level Loss Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Issue Summary Box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    issues_text = """
    IDENTIFIED ISSUES:
    
    1. ❌ Physics Loss Numerically Unstable
       Raw value: ~19 million (way too large!)
       Cause: dx=dy=dt=1.0 (wrong units)
    
    2. ❌ Physics Parameters Not Learning
       rho, mu, eta unchanged after 11 epochs
       Cause: Gradients clipped or physics loss
              dominates but doesn't back-propagate
              properly to parameters
    
    3. ⚠️ Loss Stagnant (~292-293)
       Only ~0.2% change over 11 epochs
       Cause: Physics term dominating optimization
    
    4. ⚠️ PSNR Not Improving (~26 dB)
       Oscillating between 25-27 dB
       
    RECOMMENDED FIXES:
    
    • Set lambda_physics = 0 initially
    • Add physics warmup after 10-20 epochs  
    • Scale dx, dy, dt to physical units
    • Increase learning_rate to 5e-4
    • Increase gradient_clip to 10.0
    • Focus on data loss first
    """
    
    ax6.text(0.05, 0.95, issues_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Training Diagnostics: PI-CV-RecUNet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved training_analysis.png")
    
    return fig


def create_loss_component_analysis():
    """Create figure explaining loss imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Current config
    ax = axes[0]
    components = {
        'Data (λ=1.0)': 3.6,
        'Magnitude (λ=0.1)': 0.85 * 0.1,
        'Smoothness (λ=0.001)': 4.0 * 0.001,
        'Physics (λ=0.0001)': 19594120 * 0.0001
    }
    
    names = list(components.keys())
    values = list(components.values())
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel('Weighted Loss Contribution')
    ax.set_title('Current Config: Loss Component Contributions')
    ax.set_xscale('log')
    
    for bar, val in zip(bars, values):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', ha='left', fontsize=10)
    
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    # Recommended config
    ax = axes[1]
    components_fixed = {
        'Data (λ=1.0)': 3.6,
        'Magnitude (λ=0.5)': 0.85 * 0.5,
        'Smoothness (λ=0.01)': 4.0 * 0.01,
        'Physics (λ=0)': 0
    }
    
    values_fixed = list(components_fixed.values())
    
    bars = ax.barh(names, values_fixed, color=colors)
    ax.set_xlabel('Weighted Loss Contribution')
    ax.set_title('Recommended Config: Loss Component Contributions')
    
    for bar, val in zip(bars, values_fixed):
        ax.text(max(val, 0.1), bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('loss_component_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved loss_component_analysis.png")


def create_physics_scaling_illustration():
    """Illustrate the effect of dx/dy/dt scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Laplacian scaling
    ax = axes[0]
    dx_values = [1.0, 0.1, 0.01, 0.001, 0.0003]  # 0.3mm = 0.0003m
    laplacian_scale = [1/(dx**2) for dx in dx_values]
    
    ax.bar(range(len(dx_values)), laplacian_scale, color='steelblue')
    ax.set_xticks(range(len(dx_values)))
    ax.set_xticklabels([f'dx={dx}' for dx in dx_values], rotation=45)
    ax.set_ylabel('Laplacian Scale Factor (1/dx²)')
    ax.set_title('Effect of Spatial Resolution on Laplacian')
    ax.set_yscale('log')
    
    for i, (dx, scale) in enumerate(zip(dx_values, laplacian_scale)):
        ax.text(i, scale, f'{scale:.0e}', ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=laplacian_scale[0], color='red', linestyle='--', 
               label='Current (dx=1.0)', alpha=0.7)
    ax.legend()
    
    # Second derivative scaling
    ax = axes[1]
    dt_values = [1.0, 0.1, 0.01, 0.001]
    d2_scale = [1/(dt**2) for dt in dt_values]
    
    ax.bar(range(len(dt_values)), d2_scale, color='coral')
    ax.set_xticks(range(len(dt_values)))
    ax.set_xticklabels([f'dt={dt}' for dt in dt_values], rotation=45)
    ax.set_ylabel('Temporal 2nd Derivative Scale (1/dt²)')
    ax.set_title('Effect of Temporal Resolution on ∂²u/∂t²')
    ax.set_yscale('log')
    
    for i, (dt, scale) in enumerate(zip(dt_values, d2_scale)):
        ax.text(i, scale, f'{scale:.0e}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Why Physics Loss is ~19 Million: Unit Scaling Issues', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('physics_scaling_issue.png', dpi=150, bbox_inches='tight')
    print("Saved physics_scaling_issue.png")


def main():
    # Parse the log
    data = parse_training_log('training_log_sample.txt')
    
    print("Parsed data summary:")
    print(f"  Epochs logged: {len(data['train_loss'])}")
    print(f"  Physics checkpoints: {len(data['rho'])}")
    print(f"  Batch data points: {len(data['batch_loss'])}")
    
    # Create visualizations
    create_analysis_figure(data)
    create_loss_component_analysis()
    create_physics_scaling_illustration()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("""
Generated figures:
  1. training_analysis.png - Overall training dynamics
  2. loss_component_analysis.png - Loss term contributions  
  3. physics_scaling_issue.png - Why physics loss is huge

KEY FINDINGS:

1. The physics loss raw value is ~19.6 MILLION
   - This is because dx=dy=dt=1.0 causes the Laplacian
     and temporal derivatives to be computed at the wrong scale
   - If actual pixel spacing is 0.3mm, dx should be 0.0003
   - The 1/dx² term makes Laplacian values huge

2. Despite the huge physics loss value, the weighted contribution
   (lambda_physics=0.0001) makes it ~1960 which dominates the ~3.6
   data loss. BUT the gradients flow in a way that updates
   the model weights but NOT the physics parameters.

3. The model is essentially ignoring the reconstruction task
   and trying to minimize physics residual, but the physics
   residual is based on wrong units so it's learning nothing.

RECOMMENDED TRAINING STRATEGY:

Phase 1 (Epochs 1-20): Pure Data Learning
  - lambda_physics = 0
  - lambda_data = 1.0
  - lambda_magnitude = 0.5
  - learning_rate = 5e-4
  - Goal: Get model to reconstruct data first

Phase 2 (Epochs 21+): Add Physics Gradually  
  - Slowly increase lambda_physics from 0 to 1e-9
  - Set proper dx, dy, dt values based on actual units
  - Or normalize physics loss to similar scale as data loss
""")


if __name__ == "__main__":
    main()
