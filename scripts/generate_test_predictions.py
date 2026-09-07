import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import (
    ComplexNormalizer, clean_data, complex_to_tensor, 
    create_sparse_dense_pair, tensor_to_complex
)
from src.training.train_input_guided import create_input_guided_model, InputGuidedConfig


def load_model(checkpoint_path: str, config_path: str = None, device: str = 'cuda'):
    """Load the trained model from checkpoint."""
    
    # Load config
    if config_path and Path(config_path).exists():
        config = InputGuidedConfig.load(Path(config_path))
        print(f"Loaded config from {config_path}")
    else:
        # Default config
        config = InputGuidedConfig()
        print("Using default config")
    
    # Create model
    model = create_input_guided_model(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        num_levels=config.num_levels,
        lstm_hidden_channels=config.lstm_hidden_channels,
        lstm_num_layers=config.lstm_num_layers,
        upsampling_factor=config.upsampling_factor,
        flow_hidden_channels=config.flow_hidden_channels,
        flow_scale=config.flow_scale,
        use_residual_fusion=config.use_residual_fusion,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    return model, config


def normalize_sample(data: np.ndarray, eps: float = 1e-8):
    """Normalize complex data and return stats for denormalization."""
    real = data.real
    imag = data.imag
    
    mean_real = np.mean(real)
    mean_imag = np.mean(imag)
    std_real = np.std(real) + eps
    std_imag = np.std(imag) + eps
    
    real_norm = (real - mean_real) / std_real
    imag_norm = (imag - mean_imag) / std_imag
    
    normalized_data = real_norm + 1j * imag_norm
    
    stats = {
        'mean_real': float(mean_real),
        'mean_imag': float(mean_imag),
        'std_real': float(std_real),
        'std_imag': float(std_imag)
    }
    
    return normalized_data, stats


def denormalize_tensor(tensor: torch.Tensor, stats: dict) -> np.ndarray:
    """Denormalize tensor back to original scale and return as complex numpy array."""
    # tensor: (B, T, 2, C, H, W) or (T, 2, C, H, W)
    if tensor.dim() == 6:
        tensor = tensor[0]  # Remove batch dim
    
    # tensor: (T, 2, C, H, W)
    data = tensor.cpu().numpy()
    real_norm = data[:, 0, 0, :, :]  # (T, H, W)
    imag_norm = data[:, 1, 0, :, :]
    
    # Denormalize
    real = real_norm * stats['std_real'] + stats['mean_real']
    imag = imag_norm * stats['std_imag'] + stats['mean_imag']
    
    return real + 1j * imag


def get_test_files(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """Get test file list using the same split as training."""
    data_path = Path(data_dir)
    data_files = sorted(data_path.glob("*.npy"))
    
    if len(data_files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")
    
    # Use same seed as training for reproducible split
    np.random.seed(seed)
    indices = list(range(len(data_files)))
    np.random.shuffle(indices)
    
    n_train = int(len(data_files) * train_ratio)
    n_val = int(len(data_files) * val_ratio)
    
    test_indices = indices[n_train + n_val:]
    test_files = [data_files[i] for i in test_indices]
    
    return test_files


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Model checkpoint path
    CHECKPOINT_PATH = "saved_models/SWE_PINN_outputs_input_guided_physics_trial_2_best_model.pt"
    
    # Config path (optional)
    CONFIG_PATH = "configs/config_input_guided.json"
    
    # Data directory
    DATA_DIR = "data/CIRS_phantom_dataset"
    
    # Output directory for predictions
    OUTPUT_DIR = Path("results/test_predictions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("TEST SET PREDICTION GENERATOR")
    print("="*70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print("="*70)
    
    # ============================================================
    # LOAD MODEL
    # ============================================================
    
    model, config = load_model(CHECKPOINT_PATH, CONFIG_PATH, DEVICE)
    
    SPARSE_FRAME_STEP = config.sparse_frame_step
    print(f"\nSparse frame step: {SPARSE_FRAME_STEP}")
    print(f"Upsampling factor: {config.upsampling_factor}")
    
    # ============================================================
    # GET TEST FILES
    # ============================================================
    
    test_files = get_test_files(
        DATA_DIR, 
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=42
    )
    print(f"\nFound {len(test_files)} test files")
    
    # ============================================================
    # GENERATE PREDICTIONS
    # ============================================================
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'checkpoint': CHECKPOINT_PATH,
        'config': CONFIG_PATH,
        'data_dir': DATA_DIR,
        'sparse_frame_step': SPARSE_FRAME_STEP,
        'upsampling_factor': config.upsampling_factor,
        'samples': []
    }
    
    print(f"\nGenerating predictions...")
    
    skipped_files = []
    
    with torch.no_grad():
        for idx, test_file in enumerate(tqdm(test_files, desc="Processing")):
            filename = test_file.stem  # e.g., "IQData1_10"
            
            # Load raw data (with error handling for corrupted files)
            try:
                raw_data = np.load(test_file)
            except Exception as e:
                print(f"\n  WARNING: Skipping {filename} - Error loading: {e}")
                skipped_files.append({'file': test_file.name, 'error': str(e)})
                continue
            
            raw_data = clean_data(raw_data)
            
            # Normalize
            normalized_data, norm_stats = normalize_sample(raw_data)
            
            # Convert to tensor
            tensor_norm = complex_to_tensor(normalized_data)
            
            # Create sparse/dense pairs
            sparse_norm, dense_norm = create_sparse_dense_pair(
                tensor_norm,
                sparse_step=SPARSE_FRAME_STEP
            )
            
            # Add batch dimension
            sparse_input = sparse_norm.unsqueeze(0).to(DEVICE)
            
            # Forward pass
            output_norm = model(sparse_input)
            
            # Denormalize outputs
            pred_complex = denormalize_tensor(output_norm, norm_stats)  # (T, H, W)
            
            # Also denormalize the sparse input and dense target for comparison
            sparse_complex = denormalize_tensor(sparse_norm.unsqueeze(0), norm_stats)
            dense_complex = denormalize_tensor(dense_norm.unsqueeze(0), norm_stats)
            
            # Save prediction
            pred_filename = f"pred_{filename}.npy"
            np.save(OUTPUT_DIR / pred_filename, pred_complex)
            
            # Save sparse input
            sparse_filename = f"sparse_{filename}.npy"
            np.save(OUTPUT_DIR / sparse_filename, sparse_complex)
            
            # Save dense target (ground truth)
            dense_filename = f"dense_{filename}.npy"
            np.save(OUTPUT_DIR / dense_filename, dense_complex)
            
            # Record metadata
            sample_info = {
                'index': idx,
                'original_file': test_file.name,
                'prediction_file': pred_filename,
                'sparse_file': sparse_filename,
                'dense_file': dense_filename,
                'pred_shape': list(pred_complex.shape),
                'sparse_shape': list(sparse_complex.shape),
                'dense_shape': list(dense_complex.shape),
                'norm_stats': norm_stats
            }
            metadata['samples'].append(sample_info)
            
            # Print info for first few
            if idx < 3:
                print(f"\n  {filename}:")
                print(f"    Sparse input: {sparse_complex.shape}")
                print(f"    Prediction:   {pred_complex.shape}")
                print(f"    Dense target: {dense_complex.shape}")
    
    # Save metadata
    metadata['skipped_files'] = skipped_files
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("PREDICTIONS SAVED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total samples: {len(test_files)}")
    print(f"Successfully processed: {len(test_files) - len(skipped_files)}")
    if skipped_files:
        print(f"Skipped (corrupted): {len(skipped_files)}")
        for sf in skipped_files:
            print(f"  - {sf['file']}")
    print(f"\nFiles saved for each test sample:")
    print(f"  - pred_<original_name>.npy   : Model prediction (complex)")
    print(f"  - sparse_<original_name>.npy : Input sparse frames (complex)")
    print(f"  - dense_<original_name>.npy  : Ground truth dense (complex)")
    print(f"\nMetadata: {metadata_path}")
    print("="*70)


if __name__ == "__main__":
    main()
