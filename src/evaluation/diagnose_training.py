"""
Diagnostic script to identify training issues in PI-CV-RecUNet.

This script checks for:
1. Physics parameters not updating (gradient flow)
2. Loss component imbalances
3. Data normalization issues
4. Gradient magnitude problems
5. Model output sanity checks
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

from src.architectures.pinn_recunet import create_model
from src.utils.loss_functions import PhysicsInformedLoss
from src.utils.data_loader import SWEDataModule, DataConfig


def diagnose_physics_gradients(criterion, device):
    """Check if physics parameters receive gradients."""
    print("\n" + "="*60)
    print("DIAGNOSIS 1: Physics Parameter Gradient Flow")
    print("="*60)
    
    if not hasattr(criterion, 'wave_equation'):
        print("❌ No wave equation found in criterion!")
        return
    
    we = criterion.wave_equation
    
    # Check if parameters are nn.Parameter
    for name in ['rho', 'mu', 'eta']:
        param = getattr(we, name, None)
        if param is None:
            print(f"❌ {name}: NOT FOUND")
        elif isinstance(param, nn.Parameter):
            print(f"✓ {name}: Is nn.Parameter with requires_grad={param.requires_grad}")
            print(f"   Value: {param.item():.6f}")
        else:
            print(f"❌ {name}: Is a buffer (not learnable), value={param.item():.6f}")
    
    # Create dummy data and check gradient flow
    print("\n--- Simulating forward/backward pass ---")
    batch_size, t_len, c, h, w = 2, 10, 1, 64, 64
    
    pred = torch.randn(batch_size, t_len, 2, c, h, w, device=device, requires_grad=True)
    target = torch.randn(batch_size, t_len, 2, c, h, w, device=device)
    
    # Zero gradients
    criterion.zero_grad()
    
    loss, components = criterion(pred, target, return_components=True)
    loss.backward()
    
    print(f"\nLoss components:")
    for name, val in components.items():
        print(f"  {name}: {val:.6f}")
    
    print(f"\nGradients on physics parameters:")
    for name in ['rho', 'mu', 'eta']:
        param = getattr(we, name, None)
        if param is not None and isinstance(param, nn.Parameter):
            if param.grad is not None:
                print(f"  {name}: grad={param.grad.item():.10f}")
                if abs(param.grad.item()) < 1e-10:
                    print(f"     ⚠️  WARNING: Gradient is essentially zero!")
            else:
                print(f"  ❌ {name}: grad is None - no gradient computed!")


def diagnose_loss_balance(criterion, device):
    """Analyze loss component magnitudes and weights."""
    print("\n" + "="*60)
    print("DIAGNOSIS 2: Loss Component Balance")
    print("="*60)
    
    # Create realistic-ish test data
    batch_size, t_len, c, h, w = 2, 25, 1, 64, 64
    
    # Simulate predictions close to target
    target = torch.randn(batch_size, t_len, 2, c, h, w, device=device) * 0.1
    pred = target + torch.randn_like(target) * 0.01  # Small error
    
    loss, components = criterion(pred, target, return_components=True)
    
    print(f"\nRaw loss values (before weighting):")
    print(f"  Data loss:        {components['data']:.6f}")
    print(f"  Magnitude loss:   {components['magnitude']:.6f}")
    print(f"  Smoothness loss:  {components['smoothness']:.6f}")
    print(f"  Physics loss:     {components['physics']:.6f}")
    
    # Get weights
    lambda_data = criterion.lambda_data
    lambda_mag = criterion.lambda_magnitude
    lambda_smooth = criterion.lambda_smooth
    lambda_physics = criterion.lambda_physics
    
    print(f"\nLoss weights:")
    print(f"  lambda_data:      {lambda_data}")
    print(f"  lambda_magnitude: {lambda_mag}")
    print(f"  lambda_smooth:    {lambda_smooth}")
    print(f"  lambda_physics:   {lambda_physics}")
    
    # Weighted contributions
    contrib_data = lambda_data * components['data']
    contrib_mag = lambda_mag * components['magnitude']
    contrib_smooth = lambda_smooth * components['smoothness']
    contrib_physics = lambda_physics * components['physics']
    total = contrib_data + contrib_mag + contrib_smooth + contrib_physics
    
    print(f"\nWeighted contributions to total loss:")
    print(f"  Data:       {contrib_data:.6f} ({100*contrib_data/total:.1f}%)")
    print(f"  Magnitude:  {contrib_mag:.6f} ({100*contrib_mag/total:.1f}%)")
    print(f"  Smoothness: {contrib_smooth:.6f} ({100*contrib_smooth/total:.1f}%)")
    print(f"  Physics:    {contrib_physics:.6f} ({100*contrib_physics/total:.1f}%)")
    print(f"  Total:      {total:.6f}")
    
    if contrib_physics / total < 0.001:
        print(f"\n⚠️  WARNING: Physics loss contributes <0.1% to total!")
        print(f"   The physics gradients are likely negligible.")
        print(f"   Consider increasing lambda_physics by 100-1000x")
    
    if components['physics'] > 1e6:
        print(f"\n⚠️  WARNING: Raw physics loss is very large ({components['physics']:.2e})")
        print(f"   This may indicate numerical issues or need for dx/dy/dt scaling")


def diagnose_data_distribution(data_config, num_samples=5):
    """Check the data loading and normalization."""
    print("\n" + "="*60)
    print("DIAGNOSIS 3: Data Distribution")
    print("="*60)
    
    data_module = SWEDataModule(data_config)
    data_module.setup()
    
    # Get a few samples
    train_loader = data_module.get_train_dataloader(batch_size=1, shuffle=False)
    
    sparse_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    dense_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    
    for i, batch in enumerate(train_loader):
        if i >= num_samples:
            break
        
        sparse = batch['sparse']  # (B, T, 2, C, H, W)
        dense = batch['dense']
        
        # Compute stats
        for name, data, stats in [('sparse', sparse, sparse_stats), ('dense', dense, dense_stats)]:
            stats['min'].append(data.min().item())
            stats['max'].append(data.max().item())
            stats['mean'].append(data.mean().item())
            stats['std'].append(data.std().item())
    
    print(f"\nData statistics over {min(num_samples, len(train_loader))} samples:")
    for name, stats in [('Sparse input', sparse_stats), ('Dense target', dense_stats)]:
        print(f"\n  {name}:")
        print(f"    Min:  {np.mean(stats['min']):.4f} (range: {np.min(stats['min']):.4f} to {np.max(stats['min']):.4f})")
        print(f"    Max:  {np.mean(stats['max']):.4f} (range: {np.min(stats['max']):.4f} to {np.max(stats['max']):.4f})")
        print(f"    Mean: {np.mean(stats['mean']):.4f}")
        print(f"    Std:  {np.mean(stats['std']):.4f}")
    
    # Check for issues
    if np.mean(dense_stats['std']) < 0.01:
        print(f"\n⚠️  WARNING: Data has very low variance (std={np.mean(dense_stats['std']):.6f})")
        print(f"   This may cause vanishing gradients or trivial solutions")
    
    if np.mean(dense_stats['std']) > 100:
        print(f"\n⚠️  WARNING: Data has very high variance (std={np.mean(dense_stats['std']):.2f})")
        print(f"   Consider better normalization")


def diagnose_model_gradients(model, criterion, data_loader, device):
    """Check gradient magnitudes through the model."""
    print("\n" + "="*60)
    print("DIAGNOSIS 4: Model Gradient Magnitudes")
    print("="*60)
    
    model.train()
    
    # Get one batch
    batch = next(iter(data_loader))
    sparse = batch['sparse'].to(device)
    dense = batch['dense'].to(device)
    
    # Forward pass
    model.zero_grad()
    criterion.zero_grad()
    
    pred = model(sparse)
    loss, components = criterion(pred, dense, return_components=True)
    loss.backward()
    
    print(f"\nGradient norms by layer type:")
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            # Group by layer type
            layer_type = name.split('.')[0]
            if layer_type not in grad_norms:
                grad_norms[layer_type] = []
            grad_norms[layer_type].append(grad_norm)
    
    for layer_type, norms in grad_norms.items():
        avg_norm = np.mean(norms)
        max_norm = np.max(norms)
        print(f"  {layer_type}: avg={avg_norm:.2e}, max={max_norm:.2e}")
        
        if avg_norm < 1e-8:
            print(f"     ⚠️  WARNING: Very small gradients - may indicate vanishing gradients")
        if max_norm > 1e3:
            print(f"     ⚠️  WARNING: Very large gradients - may need gradient clipping")
    
    # Check criterion parameters
    if hasattr(criterion, 'wave_equation'):
        we = criterion.wave_equation
        print(f"\n  Physics parameters:")
        for name in ['rho', 'mu', 'eta']:
            param = getattr(we, name, None)
            if param is not None and isinstance(param, nn.Parameter) and param.grad is not None:
                print(f"    {name}: grad_norm={param.grad.norm().item():.2e}")


def diagnose_output_range(model, data_loader, device):
    """Check model output statistics."""
    print("\n" + "="*60)
    print("DIAGNOSIS 5: Model Output Analysis")
    print("="*60)
    
    model.eval()
    
    batch = next(iter(data_loader))
    sparse = batch['sparse'].to(device)
    dense = batch['dense'].to(device)
    
    with torch.no_grad():
        pred = model(sparse)
    
    print(f"\nInput sparse shape:  {sparse.shape}")
    print(f"Output pred shape:   {pred.shape}")
    print(f"Target dense shape:  {dense.shape}")
    
    print(f"\nPrediction statistics:")
    print(f"  Min:  {pred.min().item():.4f}")
    print(f"  Max:  {pred.max().item():.4f}")
    print(f"  Mean: {pred.mean().item():.4f}")
    print(f"  Std:  {pred.std().item():.4f}")
    
    print(f"\nTarget statistics:")
    print(f"  Min:  {dense.min().item():.4f}")
    print(f"  Max:  {dense.max().item():.4f}")
    print(f"  Mean: {dense.mean().item():.4f}")
    print(f"  Std:  {dense.std().item():.4f}")
    
    # Check if prediction is just copying input
    diff = (pred - dense).abs()
    print(f"\nPrediction error:")
    print(f"  Mean absolute error: {diff.mean().item():.4f}")
    print(f"  Max absolute error:  {diff.max().item():.4f}")
    
    # Check if outputs are all the same (collapsed)
    pred_var = pred.var().item()
    if pred_var < 1e-6:
        print(f"\n⚠️  WARNING: Predictions have near-zero variance ({pred_var:.2e})")
        print(f"   This suggests mode collapse or the model is outputting constants")


def visualize_training_dynamics(log_path):
    """Parse and visualize training log."""
    print("\n" + "="*60)
    print("DIAGNOSIS 6: Training Dynamics from Log")
    print("="*60)
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    train_losses = []
    val_losses = []
    train_psnr = []
    val_psnr = []
    physics_params = {'rho': [], 'mu': [], 'eta': []}
    
    for line in lines:
        if 'Train - Loss:' in line:
            # Parse: Train - Loss: 292.6894, PSNR: 26.06 dB
            parts = line.split('Loss:')[1].split(',')
            train_losses.append(float(parts[0].strip()))
            psnr = float(parts[1].split(':')[1].replace('dB', '').strip())
            train_psnr.append(psnr)
        
        elif 'Val   - Loss:' in line:
            parts = line.split('Loss:')[1].split(',')
            val_losses.append(float(parts[0].strip()))
            psnr = float(parts[1].split(':')[1].replace('dB', '').strip())
            val_psnr.append(psnr)
        
        elif 'Learned physics:' in line:
            # Parse: Learned physics: rho=1000.0000, mu=1.0000, eta=0.1000
            parts = line.split('physics:')[1]
            for param in parts.split(','):
                name, val = param.split('=')
                name = name.strip()
                val = float(val.strip())
                if name in physics_params:
                    physics_params[name].append(val)
    
    print(f"\nParsed {len(train_losses)} epochs from log")
    
    if len(train_losses) > 0:
        print(f"\nTraining Loss:")
        print(f"  Start: {train_losses[0]:.2f}")
        print(f"  End:   {train_losses[-1]:.2f}")
        print(f"  Change: {train_losses[-1] - train_losses[0]:.2f} ({100*(train_losses[-1] - train_losses[0])/train_losses[0]:.1f}%)")
        
        if abs(train_losses[-1] - train_losses[0]) / train_losses[0] < 0.05:
            print(f"  ⚠️  WARNING: Loss changed by <5% - training is stagnant!")
    
    if len(physics_params['rho']) > 0:
        print(f"\nPhysics parameters evolution:")
        for name, vals in physics_params.items():
            if len(vals) > 0:
                print(f"  {name}: {vals[0]:.4f} → {vals[-1]:.4f} (change: {vals[-1] - vals[0]:.6f})")
                if abs(vals[-1] - vals[0]) < 1e-6:
                    print(f"       ⚠️  WARNING: {name} is NOT changing - no gradient or very small gradient!")
    
    # Create visualization
    if len(train_losses) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, 'b-', label='Train')
        if val_losses:
            ax.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # PSNR plot
        ax = axes[0, 1]
        ax.plot(epochs, train_psnr, 'b-', label='Train')
        if val_psnr:
            ax.plot(epochs[:len(val_psnr)], val_psnr, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR')
        ax.legend()
        ax.grid(True)
        
        # Physics params
        ax = axes[1, 0]
        for name, vals in physics_params.items():
            if vals:
                ax.plot(range(1, len(vals)+1), vals, label=name)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Value')
        ax.set_title('Physics Parameters')
        ax.legend()
        ax.grid(True)
        
        # Loss components (if we had them)
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'Run detailed batch logging\nto see loss components', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Loss Components (need detailed logs)')
        
        plt.tight_layout()
        plt.savefig('training_diagnostics.png', dpi=150)
        print(f"\n✓ Saved visualization to training_diagnostics.png")
        plt.close()


def main():
    """Run all diagnostics."""
    print("="*60)
    print("PI-CV-RecUNet Training Diagnostics")
    print("="*60)
    
    # Load config
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Setup data
    data_config = DataConfig(
        data_dir=config.get('data_dir', 'data/CIRS_phantom_dataset'),
        sparse_frame_step=config.get('sparse_frame_step', 20),
        normalization=config.get('normalization', 'per_sample'),
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
    )
    
    # Setup criterion (now with lambda_physics=0 initially)
    criterion = PhysicsInformedLoss(
        lambda_data=config.get('lambda_data', 1.0),
        lambda_physics=config.get('lambda_physics', 0.0),  # Changed from 0.0001
        lambda_smooth=config.get('lambda_smooth', 0.01),   # Changed from 0.001
        lambda_magnitude=config.get('lambda_magnitude', 0.5),  # Changed from 0.1
        learnable_physics=True,
        normalize_physics=config.get('normalize_physics_loss', True)  # NEW
    ).to(device)
    
    # Setup model
    model = create_model(
        in_channels=config.get('in_channels', 1),
        out_channels=config.get('out_channels', 1),
        base_channels=config.get('base_channels', 32),
        num_levels=config.get('num_levels', 4),
        lstm_hidden_channels=config.get('lstm_hidden_channels', 128),
        lstm_num_layers=config.get('lstm_num_layers', 1),
        upsampling_factor=config.get('upsampling_factor', 20)
    ).to(device)
    
    # Run diagnostics
    diagnose_physics_gradients(criterion, device)
    diagnose_loss_balance(criterion, device)
    diagnose_data_distribution(data_config)
    
    # Setup data module for model diagnostics
    data_module = SWEDataModule(data_config)
    data_module.setup()
    train_loader = data_module.get_train_dataloader(batch_size=2, shuffle=True)
    
    diagnose_model_gradients(model, criterion, train_loader, device)
    diagnose_output_range(model, train_loader, device)
    
    # Analyze training log
    visualize_training_dynamics('first_try_log.txt')
    
    print("\n" + "="*60)
    print("SUMMARY OF LIKELY ISSUES")
    print("="*60)
    print("""
Based on your training log, the key issues are:

1. PHYSICS PARAMETERS NOT UPDATING
   - rho, mu, eta stay at exactly 1000.0, 1.0, 0.1
   - This means either:
     a) Gradients are zero (physics loss too small)
     b) Parameters not in optimizer
     c) Gradient clipping is too aggressive

2. LOSS NOT DECREASING (~292-293 constant)
   - Training loss barely changes over 11 epochs
   - This suggests the model is stuck or learning rate is too low

3. POSSIBLE CAUSES & FIXES:
   
   a) lambda_physics is TOO SMALL (0.0001)
      - Physics loss ~289 but weighted contribution is ~0.03
      - Data loss dominates, physics gradients are negligible
      - FIX: Increase lambda_physics to 0.01 or 0.1
   
   b) Learning rate may be too low for the loss landscape
      - FIX: Try 1e-3 instead of 1e-4
   
   c) Model may need warmup before physics loss
      - FIX: Start with lambda_physics=0, increase after N epochs
   
   d) The dx, dy, dt values in wave equation are wrong
      - Using default 1.0 for everything
      - Should match actual pixel/time spacing
      - FIX: Set actual physical units

4. RECOMMENDED CONFIG CHANGES:
   {
     "learning_rate": 1e-3,
     "lambda_physics": 0.01,
     "lambda_data": 1.0,
     "gradient_clip": 5.0
   }

Run this diagnostic script to get detailed analysis!
""")


if __name__ == "__main__":
    main()
