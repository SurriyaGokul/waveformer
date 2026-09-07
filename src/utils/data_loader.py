import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict, Optional, Union, Callable
from pathlib import Path
import random
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_dir: str = "Data set 1(CIRS phantom data)"
    
    # Temporal super-resolution settings
    upsampling_factor: int = 20  # How many frames to interpolate between sparse frames
    sparse_frame_step: int = 20  # Step between sparse frames (e.g., take every 20th frame)
    fixed_sparse_frames: int = 3  # Fixed number of sparse frames (ensures consistent batch sizes)
    
    # Normalization
    normalization: str = "per_sample"  # "per_sample", "global", "instance", "none"
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    use_augmentation: bool = False
    flip_prob: float = 0.5
    
    # Spatial cropping (None for no cropping)
    crop_size: Optional[Tuple[int, int]] = None
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Random seed for reproducibility
    seed: int = 42

class ComplexNormalizer:
    """
    Normalizer for complex-valued data.
    
    Supports multiple normalization strategies:
    - per_sample: Normalize each sample independently
    - global: Use pre-computed global statistics
    - instance: Normalize each spatial location independently
    - none: No normalization
    """
    
    def __init__(
        self,
        method: str = "per_sample",
        eps: float = 1e-8
    ):
        self.method = method
        self.eps = eps
        
        # For global normalization
        self.global_mean_real = None
        self.global_mean_imag = None
        self.global_std_real = None
        self.global_std_imag = None
    
    def fit(self, data_list: List[np.ndarray]) -> None:
        """
        Compute global statistics from a list of complex arrays.
        
        Args:
            data_list: List of complex numpy arrays
        """
        if self.method != "global":
            return
        
        all_real = []
        all_imag = []
        
        for data in data_list:
            all_real.append(data.real.flatten())
            all_imag.append(data.imag.flatten())
        
        all_real = np.concatenate(all_real)
        all_imag = np.concatenate(all_imag)
        
        self.global_mean_real = float(np.mean(all_real))
        self.global_mean_imag = float(np.mean(all_imag))
        self.global_std_real = float(np.std(all_real))
        self.global_std_imag = float(np.std(all_imag))
        
        logger.info(f"Global stats computed:")
        logger.info(f"  Real: mean={self.global_mean_real:.4e}, std={self.global_std_real:.4e}")
        logger.info(f"  Imag: mean={self.global_mean_imag:.4e}, std={self.global_std_imag:.4e}")
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize complex data.
        
        Args:
            data: Complex numpy array (H, W, T) or (T, H, W)
        
        Returns:
            Normalized complex array
        """
        if self.method == "none":
            return data
        
        real = data.real
        imag = data.imag
        
        if self.method == "per_sample":
            # Normalize using sample statistics
            mean_real = np.mean(real)
            mean_imag = np.mean(imag)
            std_real = np.std(real) + self.eps
            std_imag = np.std(imag) + self.eps
            
            real_norm = (real - mean_real) / std_real
            imag_norm = (imag - mean_imag) / std_imag
        
        elif self.method == "global":
            # Use pre-computed global statistics
            if self.global_mean_real is None:
                raise ValueError("Global normalization requires calling fit() first")
            
            real_norm = (real - self.global_mean_real) / (self.global_std_real + self.eps)
            imag_norm = (imag - self.global_mean_imag) / (self.global_std_imag + self.eps)
        
        elif self.method == "instance":
            # Normalize each spatial location independently across time
            # Useful for preserving relative temporal patterns
            mean_real = np.mean(real, axis=-1, keepdims=True)
            mean_imag = np.mean(imag, axis=-1, keepdims=True)
            std_real = np.std(real, axis=-1, keepdims=True) + self.eps
            std_imag = np.std(imag, axis=-1, keepdims=True) + self.eps
            
            real_norm = (real - mean_real) / std_real
            imag_norm = (imag - mean_imag) / std_imag
        
        elif self.method == "magnitude":
            # Normalize by magnitude only (preserves phase)
            magnitude = np.abs(data) + self.eps
            mean_mag = np.mean(magnitude)
            std_mag = np.std(magnitude) + self.eps
            
            # Scale factor based on magnitude statistics
            scale = std_mag
            real_norm = real / scale
            imag_norm = imag / scale
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return real_norm + 1j * imag_norm
    
    def get_state(self) -> Dict:
        """Get normalizer state for saving."""
        return {
            'method': self.method,
            'eps': self.eps,
            'global_mean_real': self.global_mean_real,
            'global_mean_imag': self.global_mean_imag,
            'global_std_real': self.global_std_real,
            'global_std_imag': self.global_std_imag
        }
    
    def load_state(self, state: Dict) -> None:
        """Load normalizer state."""
        self.method = state['method']
        self.eps = state['eps']
        self.global_mean_real = state['global_mean_real']
        self.global_mean_imag = state['global_mean_imag']
        self.global_std_real = state['global_std_real']
        self.global_std_imag = state['global_std_imag']

def check_data_quality(data: np.ndarray, file_path: str) -> Dict:
    """
    Check data quality and return statistics.
    
    Args:
        data: Complex numpy array (H, W, T)
        file_path: Path to the file (for logging)
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'file': file_path,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'has_nan': np.any(np.isnan(data)),
        'has_inf': np.any(np.isinf(data)),
        'min_magnitude': float(np.min(np.abs(data))),
        'max_magnitude': float(np.max(np.abs(data))),
        'mean_magnitude': float(np.mean(np.abs(data))),
        'std_magnitude': float(np.std(np.abs(data))),
        'is_valid': True
    }
    
    # Check for issues
    if metrics['has_nan'] or metrics['has_inf']:
        metrics['is_valid'] = False
        logger.warning(f"Invalid data in {file_path}: NaN={metrics['has_nan']}, Inf={metrics['has_inf']}")
    
    if metrics['max_magnitude'] == 0:
        metrics['is_valid'] = False
        logger.warning(f"Zero magnitude data in {file_path}")
    
    return metrics


def clean_data(data: np.ndarray, replace_nan: bool = True, replace_inf: bool = True) -> np.ndarray:
    """
    Clean data by handling NaN and Inf values.
    
    Args:
        data: Complex numpy array
        replace_nan: Replace NaN with 0
        replace_inf: Replace Inf with max finite value
    
    Returns:
        Cleaned complex array
    """
    real = data.real.copy()
    imag = data.imag.copy()
    
    if replace_nan:
        real = np.nan_to_num(real, nan=0.0)
        imag = np.nan_to_num(imag, nan=0.0)
    
    if replace_inf:
        # Replace inf with large but finite values
        max_real = np.max(np.abs(real[np.isfinite(real)])) if np.any(np.isfinite(real)) else 1.0
        max_imag = np.max(np.abs(imag[np.isfinite(imag)])) if np.any(np.isfinite(imag)) else 1.0
        
        real = np.nan_to_num(real, posinf=max_real, neginf=-max_real)
        imag = np.nan_to_num(imag, posinf=max_imag, neginf=-max_imag)
    
    return real + 1j * imag

def complex_to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert complex numpy array to PyTorch tensor format.
    
    Args:
        data: Complex numpy array (H, W, T) or (T, H, W)
    
    Returns:
        Tensor of shape (T, 2, 1, H, W) where dim 1 is [real, imag]
    """
    # Ensure shape is (H, W, T) -> transpose to (T, H, W)
    if data.ndim == 3:
        if data.shape[2] < data.shape[0]:  # Likely (H, W, T)
            data = np.transpose(data, (2, 0, 1))  # -> (T, H, W)
    
    t_frames, height, width = data.shape
    
    # Separate real and imaginary parts
    real = data.real.astype(np.float32)
    imag = data.imag.astype(np.float32)
    
    # Stack as (T, 2, H, W)
    stacked = np.stack([real, imag], axis=1)
    
    # Add channel dimension: (T, 2, 1, H, W)
    stacked = stacked[:, :, np.newaxis, :, :]
    
    return torch.from_numpy(stacked)


def tensor_to_complex(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor back to complex numpy array.
    
    Args:
        tensor: Tensor of shape (T, 2, 1, H, W) or (B, T, 2, 1, H, W)
    
    Returns:
        Complex numpy array
    """
    if tensor.ndim == 6:  # (B, T, 2, 1, H, W)
        tensor = tensor[0]  # Take first batch
    
    # tensor is (T, 2, 1, H, W)
    data = tensor.numpy()
    real = data[:, 0, 0, :, :]  # (T, H, W)
    imag = data[:, 1, 0, :, :]
    
    return real + 1j * imag

def create_sparse_dense_pair(
    data: torch.Tensor,
    sparse_step: int = 20,
    upsampling_factor: Optional[int] = None,
    fixed_sparse_frames: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sparse/dense frame pair for temporal super-resolution training.
    
    The sparse frames are the input (low frame rate).
    The dense frames are the target (high frame rate, ground truth).
    
    Args:
        data: Full sequence tensor (T, 2, 1, H, W)
        sparse_step: Step between sparse frames (e.g., 20 means take every 20th frame)
        upsampling_factor: If provided, select frames to match this factor exactly
        fixed_sparse_frames: If provided, ensure exactly this many sparse frames are returned.
                            This ensures consistent batch sizes when datasets have different
                            total frame counts. Defaults to 3 if not specified.
    
    Returns:
        sparse: Sparse frames (T_sparse, 2, 1, H, W)
        dense: Dense frames (T_dense, 2, 1, H, W) - contains sparse frames + intermediate
    """
    t_total = data.shape[0]
    
    # Create sparse indices
    sparse_indices = list(range(0, t_total, sparse_step))
    
    # If we have upsampling_factor, ensure dense frames align properly
    if upsampling_factor is not None:
        # We need exactly (sparse_step - 1) frames between each sparse pair
        # to achieve the upsampling factor
        n_sparse = len(sparse_indices)
        
        # Ensure we have at least 2 sparse frames
        if n_sparse < 2:
            sparse_indices = [0, t_total - 1]
            n_sparse = 2
    
    # Ensure consistent number of sparse frames across different datasets
    # This is crucial for batching data with different total frame counts
    if fixed_sparse_frames is None:
        # Default to 3 sparse frames (minimum to ensure at least 2 intervals)
        # This works for both 50-frame (50/20=2.5 -> 3 sparse) and 100-frame (100/20=5 -> truncate to 3) datasets
        fixed_sparse_frames = 3
    
    if len(sparse_indices) > fixed_sparse_frames:
        # Truncate to fixed number of sparse frames
        sparse_indices = sparse_indices[:fixed_sparse_frames]
    elif len(sparse_indices) < fixed_sparse_frames:
        # Not enough frames - this shouldn't happen with reasonable sparse_step
        # But handle it by repeating the last frame
        while len(sparse_indices) < fixed_sparse_frames:
            sparse_indices.append(sparse_indices[-1])
    
    # Dense frames are all frames between first and last sparse frame
    dense_start = sparse_indices[0]
    dense_end = sparse_indices[-1] + 1
    dense_indices = list(range(dense_start, dense_end))
    
    sparse = data[sparse_indices]  # (T_sparse, 2, 1, H, W)
    dense = data[dense_indices]    # (T_dense, 2, 1, H, W)
    
    return sparse, dense


def get_training_pairs_from_sequence(
    data: torch.Tensor,
    sparse_step: int = 20,
    window_size: Optional[int] = None,
    stride: int = 1
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate multiple training pairs from a single sequence using sliding windows.
    
    This increases training data by creating overlapping windows.
    
    Args:
        data: Full sequence tensor (T, 2, 1, H, W)
        sparse_step: Step between sparse frames
        window_size: Size of each window (in dense frames). None = use full sequence
        stride: Step between window starts
    
    Returns:
        List of (sparse, dense) tensor pairs
    """
    t_total = data.shape[0]
    pairs = []
    
    if window_size is None:
        # Use full sequence
        sparse, dense = create_sparse_dense_pair(data, sparse_step)
        pairs.append((sparse, dense))
    else:
        # Sliding window
        for start in range(0, t_total - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]
            sparse, dense = create_sparse_dense_pair(window, sparse_step)
            
            # Only add if we have at least 2 sparse frames
            if sparse.shape[0] >= 2:
                pairs.append((sparse, dense))
    
    return pairs

class ComplexDataAugmentation:
    """
    Data augmentation for complex-valued IQ data.
    
    Supports:
    - Horizontal/Vertical flipping
    - Random cropping
    - Phase rotation (complex multiplication by e^(i*theta))
    """
    
    def __init__(
        self,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        flip_prob: float = 0.5,
        random_crop: Optional[Tuple[int, int]] = None,
        phase_rotation: bool = False,
        phase_range: Tuple[float, float] = (-np.pi/4, np.pi/4)
    ):
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.flip_prob = flip_prob
        self.random_crop = random_crop
        self.phase_rotation = phase_rotation
        self.phase_range = phase_range
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to data.
        
        Args:
            data: Tensor of shape (T, 2, 1, H, W)
        
        Returns:
            Augmented tensor
        """
        # Horizontal flip
        if self.flip_horizontal and random.random() < self.flip_prob:
            data = torch.flip(data, dims=[-1])  # Flip W dimension
        
        # Vertical flip
        if self.flip_vertical and random.random() < self.flip_prob:
            data = torch.flip(data, dims=[-2])  # Flip H dimension
        
        # Random crop
        if self.random_crop is not None:
            crop_h, crop_w = self.random_crop
            _, _, _, h, w = data.shape
            
            if h > crop_h and w > crop_w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                data = data[:, :, :, top:top+crop_h, left:left+crop_w]
        
        # Phase rotation (complex rotation)
        if self.phase_rotation:
            theta = random.uniform(*self.phase_range)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            real = data[:, 0:1, :, :, :]
            imag = data[:, 1:2, :, :, :]
            
            # Complex multiplication by e^(i*theta)
            new_real = real * cos_theta - imag * sin_theta
            new_imag = real * sin_theta + imag * cos_theta
            
            data = torch.cat([new_real, new_imag], dim=1)
        
        return data

class SWEDataset(Dataset):
    """
    PyTorch Dataset for Shear Wave Elastography IQ data.
    
    Loads .npy files and prepares sparse/dense pairs for training.
    """
    
    def __init__(
        self,
        data_dir: str,
        file_list: Optional[List[str]] = None,
        config: Optional[DataConfig] = None,
        normalizer: Optional[ComplexNormalizer] = None,
        augmentation: Optional[ComplexDataAugmentation] = None,
        mode: str = "train"  # "train", "val", "test"
    ):
        """
        Args:
            data_dir: Directory containing .npy files
            file_list: Optional list of specific files to use
            config: Data configuration
            normalizer: Normalizer instance
            augmentation: Augmentation instance (only applied in train mode)
            mode: Dataset mode
        """
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig()
        self.normalizer = normalizer or ComplexNormalizer(self.config.normalization)
        self.augmentation = augmentation if mode == "train" else None
        self.mode = mode
        
        # Get file list
        if file_list is not None:
            # file_list may contain full paths or just filenames
            # Use paths directly if they exist, otherwise prepend data_dir
            self.file_list = []
            for f in file_list:
                p = Path(f)
                if p.exists():
                    self.file_list.append(p)
                elif (self.data_dir / p.name).exists():
                    self.file_list.append(self.data_dir / p.name)
                else:
                    # Try as-is (may fail later during validation)
                    self.file_list.append(p if p.is_absolute() else self.data_dir / p.name)
        else:
            self.file_list = sorted(self.data_dir.glob("*.npy"))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        logger.info(f"[{mode.upper()}] Found {len(self.file_list)} files")
        
        # Pre-load and validate data
        self.data_cache = {}
        self.valid_files = []
        self._validate_and_cache()
    
    def _validate_and_cache(self):
        """Validate all files and optionally cache them."""
        for file_path in self.file_list:
            try:
                data = np.load(file_path)
                metrics = check_data_quality(data, str(file_path))
                
                if metrics['is_valid']:
                    # Clean and normalize
                    data = clean_data(data)
                    data = self.normalizer.normalize(data)
                    
                    # Convert to tensor
                    tensor = complex_to_tensor(data)
                    
                    self.data_cache[str(file_path)] = tensor
                    self.valid_files.append(file_path)
                else:
                    logger.warning(f"Skipping invalid file: {file_path}")
            
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"[{self.mode.upper()}] Validated {len(self.valid_files)}/{len(self.file_list)} files")
    
    def __len__(self) -> int:
        return len(self.valid_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - sparse: Sparse input frames (T_sparse, 2, 1, H, W)
                - dense: Dense target frames (T_dense, 2, 1, H, W)
                - file_idx: Index of source file
        """
        file_path = str(self.valid_files[idx])
        data = self.data_cache[file_path].clone()
        
        # Apply augmentation
        if self.augmentation is not None:
            data = self.augmentation(data)
        
        # Create sparse/dense pair with fixed number of sparse frames for consistent batching
        fixed_sparse = getattr(self.config, 'fixed_sparse_frames', 3)
        sparse, dense = create_sparse_dense_pair(
            data,
            sparse_step=self.config.sparse_frame_step,
            fixed_sparse_frames=fixed_sparse
        )
        
        return {
            'sparse': sparse,
            'dense': dense,
            'file_idx': idx
        }


class SWEDatasetLazy(Dataset):
    """
    Lazy-loading version of SWEDataset.
    
    Loads data on-the-fly instead of caching, useful for large datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        file_list: Optional[List[str]] = None,
        config: Optional[DataConfig] = None,
        normalizer: Optional[ComplexNormalizer] = None,
        augmentation: Optional[ComplexDataAugmentation] = None,
        mode: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig()
        self.normalizer = normalizer or ComplexNormalizer(self.config.normalization)
        self.augmentation = augmentation if mode == "train" else None
        self.mode = mode
        
        if file_list is not None:
            # Handle both absolute paths and relative paths
            self.file_list = []
            for f in file_list:
                p = Path(f)
                if p.is_absolute():
                    self.file_list.append(p)
                elif p.exists():
                    self.file_list.append(p)
                elif (self.data_dir / p).exists():
                    self.file_list.append(self.data_dir / p)
                else:
                    # File path is already relative to data_dir or is a full path string
                    self.file_list.append(p if p.exists() else Path(f))
        else:
            self.file_list = sorted(self.data_dir.glob("*.npy"))
        
        logger.info(f"[{mode.upper()}] Found {len(self.file_list)} files (lazy loading)")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_list[idx]
        
        # Load data
        data = np.load(file_path)
        
        # Clean and normalize
        data = clean_data(data)
        data = self.normalizer.normalize(data)
        
        # Convert to tensor
        tensor = complex_to_tensor(data)
        
        # Apply augmentation
        if self.augmentation is not None:
            tensor = self.augmentation(tensor)
        
        # Create sparse/dense pair with fixed number of sparse frames for consistent batching
        fixed_sparse = getattr(self.config, 'fixed_sparse_frames', 3)
        sparse, dense = create_sparse_dense_pair(
            tensor,
            sparse_step=self.config.sparse_frame_step,
            fixed_sparse_frames=fixed_sparse
        )
        
        return {
            'sparse': sparse,
            'dense': dense,
            'file_idx': idx
        }

class SWEDataModule:
    """
    Complete data handling module for training.
    
    Handles:
    - Data loading and validation
    - Train/val/test splitting
    - DataLoader creation
    - Normalization fitting
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.normalizer = ComplexNormalizer(config.normalization)
        
        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    
    def setup(self, data_dir: Optional[str] = None) -> None:
        """
        Set up datasets with train/val/test splits.
        
        Args:
            data_dir: Override data directory from config
        """
        data_dir = Path(data_dir or self.config.data_dir)
        
        # Get all files - search recursively for 10kHz and 20kHz directories
        all_files = []
        
        # Check if we should search for 10kHz and 20kHz subdirectories
        khz_10_dirs = list(data_dir.rglob("10kHz"))
        khz_20_dirs = list(data_dir.rglob("20kHz"))
        khz_dirs = khz_10_dirs + khz_20_dirs
        
        if khz_dirs:
            # Found kHz directories - collect all .npy files from them (recursively)
            logger.info(f"Found {len(khz_10_dirs)} 10kHz directories and {len(khz_20_dirs)} 20kHz directories")
            for khz_dir in khz_dirs:
                # Use rglob to find .npy files in subdirectories (e.g., Plane 1, Plane 2, etc.)
                npy_files = sorted(khz_dir.rglob("*.npy"))
                all_files.extend(npy_files)
                logger.info(f"  {khz_dir.parent.name}/{khz_dir.name}: {len(npy_files)} files")
        else:
            # Fallback to original behavior - direct .npy files
            all_files = sorted(data_dir.glob("*.npy"))
        
        n_files = len(all_files)
        
        if n_files == 0:
            raise ValueError(f"No .npy files found in {data_dir} (searched for 10kHz and 20kHz subdirs)")
        
        logger.info(f"Found {n_files} files in {data_dir}")
        
        # Shuffle files
        file_indices = list(range(n_files))
        random.shuffle(file_indices)
        
        # Calculate split sizes
        n_train = int(n_files * self.config.train_ratio)
        n_val = int(n_files * self.config.val_ratio)
        n_test = n_files - n_train - n_val
        
        # Split indices
        train_indices = file_indices[:n_train]
        val_indices = file_indices[n_train:n_train + n_val]
        test_indices = file_indices[n_train + n_val:]
        
        # Get file lists
        train_files = [all_files[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]
        test_files = [all_files[i] for i in test_indices]
        
        logger.info(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
        
        # Fit normalizer on training data if using global normalization
        if self.config.normalization == "global":
            logger.info("Fitting global normalizer on training data...")
            train_data = [np.load(f) for f in train_files]
            self.normalizer.fit(train_data)
            del train_data  # Free memory
        
        # Create augmentation
        augmentation = None
        if self.config.use_augmentation:
            augmentation = ComplexDataAugmentation(
                flip_horizontal=True,
                flip_vertical=True,
                flip_prob=self.config.flip_prob,
                random_crop=self.config.crop_size
            )
        
        # Create datasets using lazy loading to avoid caching all data in RAM
        self.train_dataset = SWEDatasetLazy(
            data_dir=str(data_dir),
            file_list=[str(f) for f in train_files],
            config=self.config,
            normalizer=self.normalizer,
            augmentation=augmentation,
            mode="train"
        )
        
        self.val_dataset = SWEDatasetLazy(
            data_dir=str(data_dir),
            file_list=[str(f) for f in val_files],
            config=self.config,
            normalizer=self.normalizer,
            augmentation=None,
            mode="val"
        )
        
        self.test_dataset = SWEDatasetLazy(
            data_dir=str(data_dir),
            file_list=[str(f) for f in test_files],
            config=self.config,
            normalizer=self.normalizer,
            augmentation=None,
            mode="test"
        )
    
    def get_train_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
    
    def get_test_dataloader(self, batch_size: int) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function to handle variable-length sequences.
        
        Since all sequences have the same length (35 frames), 
        this is straightforward stacking.
        """
        sparse = torch.stack([item['sparse'] for item in batch], dim=0)
        dense = torch.stack([item['dense'] for item in batch], dim=0)
        file_indices = torch.tensor([item['file_idx'] for item in batch])
        
        return {
            'sparse': sparse,    # (B, T_sparse, 2, 1, H, W)
            'dense': dense,      # (B, T_dense, 2, 1, H, W)
            'file_idx': file_indices
        }
    
    def get_sample_batch(self, batch_size: int = 2) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing/debugging."""
        loader = self.get_train_dataloader(batch_size, shuffle=False)
        return next(iter(loader))

def analyze_dataset(data_dir: str) -> Dict:
    """
    Analyze the entire dataset and return statistics.
    
    Useful for understanding data characteristics before training.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npy"))
    
    stats = {
        'n_files': len(files),
        'shapes': [],
        'dtypes': [],
        'magnitude_stats': [],
        'phase_stats': [],
        'valid_count': 0,
        'invalid_count': 0
    }
    
    for file_path in files:
        try:
            data = np.load(file_path)
            stats['shapes'].append(data.shape)
            stats['dtypes'].append(str(data.dtype))
            
            magnitude = np.abs(data)
            phase = np.angle(data)
            
            stats['magnitude_stats'].append({
                'min': float(np.min(magnitude)),
                'max': float(np.max(magnitude)),
                'mean': float(np.mean(magnitude)),
                'std': float(np.std(magnitude))
            })
            
            stats['phase_stats'].append({
                'min': float(np.min(phase)),
                'max': float(np.max(phase)),
                'mean': float(np.mean(phase)),
                'std': float(np.std(phase))
            })
            
            if not (np.any(np.isnan(data)) or np.any(np.isinf(data))):
                stats['valid_count'] += 1
            else:
                stats['invalid_count'] += 1
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            stats['invalid_count'] += 1
    
    # Aggregate statistics
    if stats['magnitude_stats']:
        all_mags = stats['magnitude_stats']
        stats['aggregate'] = {
            'magnitude': {
                'global_min': min(m['min'] for m in all_mags),
                'global_max': max(m['max'] for m in all_mags),
                'mean_of_means': np.mean([m['mean'] for m in all_mags]),
                'mean_of_stds': np.mean([m['std'] for m in all_mags])
            }
        }
    
    return stats


def save_preprocessed_data(
    data_dir: str,
    output_dir: str,
    config: DataConfig
) -> None:
    """
    Preprocess and save all data in a more efficient format.
    
    Saves:
    - Preprocessed tensors as .pt files
    - Normalizer state
    - Dataset statistics
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_module = SWEDataModule(config)
    data_module.setup(data_dir)
    
    # Save normalizer state
    normalizer_state = data_module.normalizer.get_state()
    with open(output_dir / "normalizer_state.json", "w") as f:
        json.dump(normalizer_state, f, indent=2)
    
    # Save preprocessed tensors
    for split, dataset in [
        ("train", data_module.train_dataset),
        ("val", data_module.val_dataset),
        ("test", data_module.test_dataset)
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            torch.save({
                'sparse': sample['sparse'],
                'dense': sample['dense']
            }, split_dir / f"sample_{idx:04d}.pt")
        
        logger.info(f"Saved {len(dataset)} {split} samples")
    
    logger.info(f"Preprocessing complete. Data saved to {output_dir}")


