"""
Velocity Estimation Dataset for Shear Wave Elastography.

This module provides the VelocityEstimationDataset class for loading paired
sparse IQ input sequences and dense velocity target sequences.

Key preprocessing steps:
    1. Clutter Filter: Subtract time-averaged mean from IQ frames
    2. IQ Normalization: Divide by fixed scalar (5000.0)
    3. Velocity Normalization: Divide by pi to map [-pi, pi] -> [-1, 1]

Data Formats:
    Input (Sparse IQ): Complex64 or Float32 with shape (T_sparse, H, W) or (T_sparse, 2, H, W)
    Target (Velocity): Float32 with shape (T_dense, 1, H, W)

Output Tensors:
    input_tensor: (2, T_sparse, H, W) - Real/Imag channels
    target_tensor: (1, T_dense, H, W) - Velocity channel
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging
logger = logging.getLogger(__name__)


class VelocityEstimationDataset(Dataset):
    """
    PyTorch Dataset for Velocity Estimation from Sparse IQ Data.
    
    WARNING: This class does NOT return magnitude masks. Use PreprocessedVelocityDataset
    instead if you need magnitude-weighted loss (MagnitudeWeightedVelocityLoss).
    
    Loads paired sparse IQ sequences and dense velocity targets with
    appropriate preprocessing for direct velocity estimation.
    
    Preprocessing Pipeline:
        1. Load sparse IQ data (complex) and convert to real/imag channels
        2. Apply clutter filter: iq = iq - iq.mean(axis=0)
        3. Normalize IQ by fixed scalar: iq = iq / 5000.0
        4. Load dense velocity data
        5. Normalize velocity: velocity = velocity / pi
    
    Attributes:
        input_paths: List of paths to sparse IQ files
        target_paths: List of paths to dense velocity files
        iq_normalization_scalar: Fixed scalar for IQ normalization (default: 5000.0)
        apply_clutter_filter: Whether to apply temporal mean subtraction
        transform: Optional transform to apply to data
    """
    
    # Constants
    IQ_NORMALIZATION_SCALAR: float = 5000.0
    VELOCITY_NORMALIZATION_SCALAR: float = np.pi
    
    def __init__(
        self,
        input_paths: List[Union[str, Path]],
        target_paths: List[Union[str, Path]],
        iq_normalization_scalar: float = 5000.0,
        apply_clutter_filter: bool = True,
        transform: Optional[Callable] = None,
        validate_pairs: bool = True,
    ):
        """
        Initialize the VelocityEstimationDataset.
        
        Args:
            input_paths: List of paths to sparse IQ .npy files
            target_paths: List of paths to dense velocity .npy files
            iq_normalization_scalar: Scalar to divide IQ data by (default: 5000.0)
            apply_clutter_filter: Whether to subtract temporal mean from IQ
            transform: Optional callable transform to apply
            validate_pairs: Whether to validate that input/target pairs exist
            
        Raises:
            ValueError: If input_paths and target_paths have different lengths
            FileNotFoundError: If any path doesn't exist (when validate_pairs=True)
        """
        super().__init__()
        
        if len(input_paths) != len(target_paths):
            raise ValueError(
                f"Number of input paths ({len(input_paths)}) must match "
                f"number of target paths ({len(target_paths)})"
            )
        
        self.input_paths = [Path(p) for p in input_paths]
        self.target_paths = [Path(p) for p in target_paths]
        self.iq_normalization_scalar = iq_normalization_scalar
        self.apply_clutter_filter = apply_clutter_filter
        self.transform = transform
        
        if validate_pairs:
            self._validate_paths()
        
        logger.info(
            f"VelocityEstimationDataset initialized with {len(self)} samples. "
            f"IQ norm scalar: {self.iq_normalization_scalar}, "
            f"Clutter filter: {self.apply_clutter_filter}"
        )
    
    def _validate_paths(self) -> None:
        """Validate that all paths exist."""
        for i, (inp, tgt) in enumerate(zip(self.input_paths, self.target_paths)):
            if not inp.exists():
                raise FileNotFoundError(f"Input file not found: {inp}")
            if not tgt.exists():
                raise FileNotFoundError(f"Target file not found: {tgt}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.input_paths)
    
    def _load_iq_data(self, path: Path) -> np.ndarray:
        """
        Load IQ data and convert to complex format.
        
        Handles:
            - Complex64: shape (T, H, W)
            - Float32: shape (T, 2, H, W) where dim=1 is [Real, Imag]
        
        Args:
            path: Path to .npy file
            
        Returns:
            Complex64 array of shape (T, H, W)
        """
        data = np.load(path)
        
        # Already complex
        if np.iscomplexobj(data):
            if data.ndim == 3:
                return data.astype(np.complex64)
            elif data.ndim == 4 and data.shape[1] == 1:
                return data.squeeze(1).astype(np.complex64)
            else:
                raise ValueError(f"Unexpected complex shape: {data.shape}")
        
        # Real with separate channels
        if data.ndim == 4 and data.shape[1] == 2:
            real = data[:, 0, :, :]
            imag = data[:, 1, :, :]
            return (real + 1j * imag).astype(np.complex64)
        
        raise ValueError(f"Cannot parse IQ data: shape={data.shape}, dtype={data.dtype}")
    
    def _apply_clutter_filter(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Apply clutter filter by subtracting temporal mean.
        
        This removes static/slow-moving tissue signal, preserving
        the shear wave motion.
        
        Args:
            iq_data: Complex IQ data of shape (T, H, W)
            
        Returns:
            Filtered IQ data of shape (T, H, W)
        """
        # Compute mean across time (axis=0)
        temporal_mean = iq_data.mean(axis=0, keepdims=True)
        
        # Subtract mean from each frame
        filtered = iq_data - temporal_mean
        
        return filtered
    
    def _normalize_iq(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Normalize IQ data by fixed scalar.
        
        Uses simple scalar division (NOT min-max scaling) to preserve
        the physical meaning of the IQ signal amplitude.
        
        Args:
            iq_data: Complex IQ data of shape (T, H, W)
            
        Returns:
            Normalized IQ data of shape (T, H, W)
        """
        return iq_data / self.iq_normalization_scalar
    
    def _complex_to_channels(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Convert complex data to real/imag channel format.
        
        Args:
            iq_data: Complex array of shape (T, H, W)
            
        Returns:
            Real array of shape (2, T, H, W) where dim=0 is [Real, Imag]
        """
        real = iq_data.real.astype(np.float32)
        imag = iq_data.imag.astype(np.float32)
        
        # Stack to (2, T, H, W)
        return np.stack([real, imag], axis=0)
    
    def _load_velocity_data(self, path: Path) -> np.ndarray:
        """
        Load velocity ground truth data.
        
        Expected shape: (T, 1, H, W) or (T, H, W)
        
        Args:
            path: Path to velocity .npy file
            
        Returns:
            Velocity array of shape (T, 1, H, W)
        """
        data = np.load(path).astype(np.float32)
        
        # Handle different input shapes
        if data.ndim == 3:
            # (T, H, W) -> (T, 1, H, W)
            data = data[:, np.newaxis, :, :]
        elif data.ndim == 4:
            if data.shape[1] != 1:
                raise ValueError(
                    f"Expected velocity shape (T, 1, H, W), got {data.shape}"
                )
        else:
            raise ValueError(f"Unexpected velocity shape: {data.shape}")
        
        return data
    
    def _normalize_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Normalize velocity to [-1, 1] range.
        
        Divides by pi to map the range [-pi, pi] to [-1, 1].
        This matches the Tanh output activation of the model.
        
        Args:
            velocity: Velocity array in radians, range [-pi, pi]
            
        Returns:
            Normalized velocity array, range [-1, 1]
        """
        return velocity / self.VELOCITY_NORMALIZATION_SCALAR
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_tensor, target_tensor):
                - input_tensor: (2, T_sparse, H, W) float32 [Real/Imag channels]
                - target_tensor: (1, T_dense, H, W) float32 [Velocity channel]
        """
        # Load sparse IQ input
        iq_data = self._load_iq_data(self.input_paths[idx])
        
        # Apply preprocessing to IQ
        if self.apply_clutter_filter:
            iq_data = self._apply_clutter_filter(iq_data)
        
        iq_data = self._normalize_iq(iq_data)
        
        # Convert complex to channels: (T, H, W) -> (2, T, H, W)
        input_array = self._complex_to_channels(iq_data)
        
        # Load velocity target
        velocity_data = self._load_velocity_data(self.target_paths[idx])
        
        # Normalize velocity: (T, 1, H, W)
        velocity_data = self._normalize_velocity(velocity_data)
        
        # Reshape velocity: (T, 1, H, W) -> (1, T, H, W)
        target_array = velocity_data.transpose(1, 0, 2, 3)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_array)
        target_tensor = torch.from_numpy(target_array)
        
        # Apply optional transform
        if self.transform is not None:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        
        return input_tensor, target_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a sample without loading full data.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        return {
            "index": idx,
            "input_path": str(self.input_paths[idx]),
            "target_path": str(self.target_paths[idx]),
            "input_filename": self.input_paths[idx].name,
            "target_filename": self.target_paths[idx].name,
        }


def create_velocity_dataset_from_dirs(
    sparse_iq_dir: Union[str, Path],
    dense_velocity_dir: Union[str, Path],
    sparse_pattern: str = "sparse_*.npy",
    velocity_prefix: str = "velocity_",
    **dataset_kwargs,
) -> VelocityEstimationDataset:
    """
    Create a VelocityEstimationDataset from directories.
    
    Automatically pairs sparse IQ files with their corresponding velocity files
    based on filename patterns.
    
    Args:
        sparse_iq_dir: Directory containing sparse IQ files
        dense_velocity_dir: Directory containing velocity ground truth files
        sparse_pattern: Glob pattern for sparse IQ files
        velocity_prefix: Prefix for velocity files
        **dataset_kwargs: Additional arguments for VelocityEstimationDataset
        
    Returns:
        Configured VelocityEstimationDataset
    """
    sparse_iq_dir = Path(sparse_iq_dir)
    dense_velocity_dir = Path(dense_velocity_dir)
    
    # Find sparse IQ files
    sparse_files = sorted(sparse_iq_dir.glob(sparse_pattern))
    
    if not sparse_files:
        raise FileNotFoundError(f"No files matching '{sparse_pattern}' in {sparse_iq_dir}")
    
    input_paths = []
    target_paths = []
    
    for sparse_path in sparse_files:
        # Generate expected velocity filename
        base_name = sparse_path.stem
        if base_name.startswith("sparse_"):
            base_name = base_name[7:]  # Remove "sparse_" prefix
        
        velocity_name = f"{velocity_prefix}{base_name}.npy"
        velocity_path = dense_velocity_dir / velocity_name
        
        if velocity_path.exists():
            input_paths.append(sparse_path)
            target_paths.append(velocity_path)
        else:
            logger.warning(f"No velocity file found for {sparse_path.name}, skipping")
    
    logger.info(f"Found {len(input_paths)} valid input-target pairs")
    
    return VelocityEstimationDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        **dataset_kwargs,
    )


def create_velocity_dataset_from_lists(
    input_list: List[Union[str, Path]],
    target_list: List[Union[str, Path]],
    **dataset_kwargs,
) -> VelocityEstimationDataset:
    """
    Create a VelocityEstimationDataset from explicit file lists.
    
    Args:
        input_list: List of paths to sparse IQ files
        target_list: List of paths to velocity files
        **dataset_kwargs: Additional arguments for VelocityEstimationDataset
        
    Returns:
        Configured VelocityEstimationDataset
    """
    return VelocityEstimationDataset(
        input_paths=input_list,
        target_paths=target_list,
        **dataset_kwargs,
    )


def get_velocity_dataloader(
    dataset: VelocityEstimationDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the VelocityEstimationDataset.
    
    Args:
        dataset: VelocityEstimationDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for CUDA
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class VelocityDataTransform:
    """
    Transform class for data augmentation on velocity estimation data.
    
    Supports:
        - Random horizontal flip
        - Random vertical flip
        - Random temporal flip
    """
    
    def __init__(
        self,
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        temporal_flip_prob: float = 0.0,
    ):
        """
        Initialize the transform.
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip  
            temporal_flip_prob: Probability of temporal flip
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.temporal_flip_prob = temporal_flip_prob
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to input and target tensors.
        
        Args:
            input_tensor: (2, T_sparse, H, W)
            target_tensor: (1, T_dense, H, W)
            
        Returns:
            Transformed (input_tensor, target_tensor)
        """
        # Horizontal flip
        if torch.rand(1).item() < self.horizontal_flip_prob:
            input_tensor = torch.flip(input_tensor, dims=[-1])
            target_tensor = torch.flip(target_tensor, dims=[-1])
        
        # Vertical flip
        if torch.rand(1).item() < self.vertical_flip_prob:
            input_tensor = torch.flip(input_tensor, dims=[-2])
            target_tensor = torch.flip(target_tensor, dims=[-2])
        
        # Temporal flip (reverse time)
        if torch.rand(1).item() < self.temporal_flip_prob:
            input_tensor = torch.flip(input_tensor, dims=[1])
            target_tensor = torch.flip(target_tensor, dims=[1])
            # Note: For velocity, temporal flip should also negate values
            # since velocity direction reverses with time reversal
            target_tensor = -target_tensor
        
        return input_tensor, target_tensor


class PreprocessedVelocityDataset(Dataset):
    """
    Dataset for loading preprocessed velocity data with magnitude masks.
    
    This dataset loads data preprocessed by preprocess_velocity_data.py:
    - Sparse IQ: Complex (T_sparse, H, W) with clutter filtering applied
    - Velocity: Float (T_dense, H, W) from Kasai, range [-pi, pi]
    - Magnitude: Float (T_dense, H, W) mask, range [0, 1]
    
    Output Tensors:
        input_tensor: (2, T_sparse, H, W) - Real/Imag channels, normalized by global_max
        target_tensor: (1, T_dense, H, W) - Velocity in [-1, 1] (already normalized by pi during preprocessing)
        mask_tensor: (1, T_dense, H, W) - Magnitude mask [0, 1] for weighted loss
    """
    
    def __init__(
        self,
        preprocessed_dir: Union[str, Path],
        global_max: Optional[float] = None,
        transform: Optional[Callable] = None,
        return_mask: bool = True,
        validate_files: bool = True,
        t_sparse_filter: Optional[int] = None,
        t_dense_filter: Optional[int] = None,
        use_smooth_targets: bool = False,
        use_clean_targets: bool = False,
        use_loupas_targets: bool = False,
    ):
        """
        Initialize the PreprocessedVelocityDataset.
        
        Args:
            preprocessed_dir: Directory containing preprocessed files
            global_max: Global maximum for IQ normalization (from metadata.json)
            transform: Optional callable transform to apply
            return_mask: Whether to return magnitude mask (for weighted loss)
            validate_files: Whether to validate file existence
            t_sparse_filter: If set, only keep samples with this T_sparse value
            t_dense_filter: If set, only keep samples with this T_dense value
            use_smooth_targets: Load velocity_smooth_*.npy (naive Gaussian).
            use_clean_targets: Load velocity_clean_*.npy (normalised conv).
            use_loupas_targets: RECOMMENDED. Load velocity_loupas_*.npy —
                               Loupas estimator with clean wave structure.
            
        Raises:
            FileNotFoundError: If preprocessed_dir doesn't exist or no files found
        """
        super().__init__()
        
        self.t_sparse_filter = t_sparse_filter
        self.t_dense_filter = t_dense_filter
        self.use_smooth_targets = use_smooth_targets
        self.use_clean_targets = use_clean_targets
        self.use_loupas_targets = use_loupas_targets
        if sum([use_smooth_targets, use_clean_targets, use_loupas_targets]) > 1:
            logger.warning("Multiple target types set; priority: loupas > clean > smooth")
        
        self.preprocessed_dir = Path(preprocessed_dir)
        self.transform = transform
        self.return_mask = return_mask
        
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.preprocessed_dir}")
        
        # Try to load global_max from metadata if not provided
        self.global_max = global_max
        if self.global_max is None:
            metadata_path = self.preprocessed_dir.parent / 'metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.global_max = metadata.get('global_stats', {}).get('global_max')
                logger.info(f"Loaded global_max={self.global_max:.4e} from metadata")
        
        if self.global_max is None:
            logger.warning("No global_max specified, using 1.0 (IQ won't be normalized)")
            self.global_max = 1.0
        
        # Find all sparse input files
        sparse_files = sorted(self.preprocessed_dir.glob("sparse_*.npy"))
        
        if not sparse_files:
            raise FileNotFoundError(
                f"No sparse_*.npy files found in {self.preprocessed_dir}"
            )
        
        # Build file lists
        self.samples = []
        skipped_t_sparse = 0
        skipped_t_dense = 0
        for sparse_path in sparse_files:
            # Extract base name (e.g., "sparse_20260129T132003.npy" -> "20260129T132003")
            base_name = sparse_path.stem
            if base_name.startswith("sparse_"):
                base_name = base_name[7:]
            
            if self.use_loupas_targets:
                vel_prefix = "velocity_loupas_"
            elif self.use_clean_targets:
                vel_prefix = "velocity_clean_"
            elif self.use_smooth_targets:
                vel_prefix = "velocity_smooth_"
            else:
                vel_prefix = "velocity_"
            velocity_path = self.preprocessed_dir / f"{vel_prefix}{base_name}.npy"
            magnitude_path = self.preprocessed_dir / f"magnitude_{base_name}.npy"
            
            if validate_files:
                if not velocity_path.exists():
                    logger.warning(f"No velocity file for {sparse_path.name}, skipping")
                    continue
                if return_mask and not magnitude_path.exists():
                    logger.warning(f"No magnitude file for {sparse_path.name}, skipping")
                    continue
            
            # Filter by T_sparse if specified (check shape without loading full array)
            if self.t_sparse_filter is not None:
                # Use mmap to quickly check shape without loading data
                try:
                    arr = np.load(sparse_path, mmap_mode='r')
                    t_sparse = arr.shape[0]
                    del arr  # Release mmap
                except Exception as e:
                    logger.warning(f"Failed to read {sparse_path.name}: {e}, skipping")
                    continue
                if t_sparse != self.t_sparse_filter:
                    skipped_t_sparse += 1
                    continue
            
            # Filter by T_dense if specified (check velocity shape)
            if self.t_dense_filter is not None:
                try:
                    arr = np.load(velocity_path, mmap_mode='r')
                    t_dense = arr.shape[0]
                    del arr  # Release mmap
                except Exception as e:
                    logger.warning(f"Failed to read {velocity_path.name}: {e}, skipping")
                    continue
                if t_dense != self.t_dense_filter:
                    skipped_t_dense += 1
                    continue
            
            self.samples.append({
                "sparse": sparse_path,
                "velocity": velocity_path,
                "magnitude": magnitude_path,
                "base_name": base_name,
            })
        
        if skipped_t_sparse > 0:
            logger.info(f"Filtered out {skipped_t_sparse} samples with T_sparse != {self.t_sparse_filter}")
        
        if skipped_t_dense > 0:
            logger.info(f"Filtered out {skipped_t_dense} samples with T_dense != {self.t_dense_filter}")
        
        if not self.samples:
            raise FileNotFoundError(
                f"No valid sample triplets found in {self.preprocessed_dir}"
            )
        
        if self.use_loupas_targets:
            target_label = "loupas"
        elif self.use_clean_targets:
            target_label = "clean"
        elif self.use_smooth_targets:
            target_label = "smooth"
        else:
            target_label = "raw"
        logger.info(
            f"PreprocessedVelocityDataset: {len(self)} samples from {self.preprocessed_dir} "
            f"(targets: {target_label})"
        )
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Load a preprocessed sample.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_mask=True: (input_tensor, target_tensor, mask_tensor)
            If return_mask=False: (input_tensor, target_tensor)
            
        Shapes:
            input_tensor: (2, T_sparse, H, W) - Real/Imag channels
            target_tensor: (1, T_dense, H, W) - Velocity in [-1, 1]
            mask_tensor: (1, T_dense, H, W) - Magnitude mask in [0, 1]
        """
        sample = self.samples[idx]
        
        try:
            # Load sparse input (T_sparse, H, W) complex -> (2, T_sparse, H, W) float
            sparse_complex = np.load(sample["sparse"])
            
            # Normalize by global_max
            sparse_normalized = sparse_complex / self.global_max
            
            # Convert complex to real/imag channels: (T, H, W) -> (2, T, H, W)
            real = sparse_normalized.real.astype(np.float32)
            imag = sparse_normalized.imag.astype(np.float32)
            input_array = np.stack([real, imag], axis=0)  # (2, T_sparse, H, W)
            
            # Load velocity target (T_dense, H, W) -> (1, T_dense, H, W)
            # Data is ALREADY normalized to [-1, 1] during preprocessing (divided by pi)
            velocity = np.load(sample["velocity"]).astype(np.float32)
            # Add channel dimension: (T, H, W) -> (1, T, H, W)
            target_array = velocity[np.newaxis, :, :, :]
        except Exception as e:
            raise ValueError(
                f"Failed to load sample {idx} ({sample['base_name']}): {e}\n"
                f"  Sparse: {sample['sparse']}\n"
                f"  Velocity: {sample['velocity']}"
            ) from e
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_array)
        target_tensor = torch.from_numpy(target_array)
        
        if self.return_mask:
            # Load magnitude mask (T_dense, H, W) -> (1, T_dense, H, W)
            magnitude = np.load(sample["magnitude"]).astype(np.float32)
            # Add channel dimension
            mask_array = magnitude[np.newaxis, :, :, :]
            mask_tensor = torch.from_numpy(mask_array)
            
            # Apply optional transform
            if self.transform is not None:
                input_tensor, target_tensor, mask_tensor = self.transform(
                    input_tensor, target_tensor, mask_tensor
                )
            
            return input_tensor, target_tensor, mask_tensor
        else:
            # Apply optional transform
            if self.transform is not None:
                input_tensor, target_tensor = self.transform(
                    input_tensor, target_tensor
                )
            
            return input_tensor, target_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample."""
        sample = self.samples[idx]
        return {
            "index": idx,
            "base_name": sample["base_name"],
            "sparse_path": str(sample["sparse"]),
            "velocity_path": str(sample["velocity"]),
            "magnitude_path": str(sample["magnitude"]),
        }


def create_preprocessed_velocity_dataloaders(
    preprocessed_dir: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_mask: bool = True,
    global_max: Optional[float] = None,
    t_sparse_filter: Optional[int] = 5,
    t_dense_filter: Optional[int] = 80,
    use_smooth_targets: bool = False,
    use_clean_targets: bool = False,
    use_loupas_targets: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders from preprocessed train/ and val/ directories.
    
    Args:
        preprocessed_dir: Root directory with train/ and val/ subdirectories
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA
        return_mask: Whether to return magnitude masks
        global_max: Global max for IQ normalization (loaded from metadata if None)
        t_sparse_filter: Filter samples by T_sparse
        t_dense_filter: Filter samples by T_dense
        use_smooth_targets: Load naive Gaussian-smoothed targets
        use_clean_targets: Load normalised-convolution cleaned targets
        use_loupas_targets: RECOMMENDED. Load Loupas estimator targets.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    import json
    
    preprocessed_dir = Path(preprocessed_dir)
    
    # Load global_max from metadata if not provided
    if global_max is None:
        metadata_path = preprocessed_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            global_max = metadata.get('global_stats', {}).get('global_max')
            logger.info(f"Loaded global_max={global_max:.4e} from metadata")
    
    # Create datasets from train/ and val/ directories
    train_dir = preprocessed_dir / 'train'
    val_dir = preprocessed_dir / 'val'
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    train_dataset = PreprocessedVelocityDataset(
        preprocessed_dir=train_dir,
        global_max=global_max,
        return_mask=return_mask,
        t_sparse_filter=t_sparse_filter,
        t_dense_filter=t_dense_filter,
        use_smooth_targets=use_smooth_targets,
        use_clean_targets=use_clean_targets,
        use_loupas_targets=use_loupas_targets,
    )
    
    val_dataset = PreprocessedVelocityDataset(
        preprocessed_dir=val_dir,
        global_max=global_max,
        return_mask=return_mask,
        t_sparse_filter=t_sparse_filter,
        t_dense_filter=t_dense_filter,
        use_smooth_targets=use_smooth_targets,
        use_clean_targets=use_clean_targets,
        use_loupas_targets=use_loupas_targets,
    )
    
    logger.info(f"Datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


def verify_dataset_sample(
    dataset: VelocityEstimationDataset,
    idx: int = 0,
) -> Dict[str, Any]:
    """
    Verify a dataset sample and return diagnostic information.
    
    Useful for debugging data loading issues.
    
    Args:
        dataset: VelocityEstimationDataset instance
        idx: Sample index to verify
        
    Returns:
        Dictionary with sample diagnostics
    """
    input_tensor, target_tensor = dataset[idx]
    
    diagnostics = {
        "index": idx,
        "input_shape": tuple(input_tensor.shape),
        "target_shape": tuple(target_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "target_dtype": str(target_tensor.dtype),
        "input_stats": {
            "real_min": input_tensor[0].min().item(),
            "real_max": input_tensor[0].max().item(),
            "real_mean": input_tensor[0].mean().item(),
            "imag_min": input_tensor[1].min().item(),
            "imag_max": input_tensor[1].max().item(),
            "imag_mean": input_tensor[1].mean().item(),
        },
        "target_stats": {
            "min": target_tensor.min().item(),
            "max": target_tensor.max().item(),
            "mean": target_tensor.mean().item(),
            "std": target_tensor.std().item(),
        },
        "has_nan": {
            "input": torch.isnan(input_tensor).any().item(),
            "target": torch.isnan(target_tensor).any().item(),
        },
        "has_inf": {
            "input": torch.isinf(input_tensor).any().item(),
            "target": torch.isinf(target_tensor).any().item(),
        },
    }
    
    # Log diagnostics
    logger.info(f"Sample {idx} diagnostics:")
    logger.info(f"  Input shape: {diagnostics['input_shape']}")
    logger.info(f"  Target shape: {diagnostics['target_shape']}")
    logger.info(f"  Input real range: [{diagnostics['input_stats']['real_min']:.4f}, {diagnostics['input_stats']['real_max']:.4f}]")
    logger.info(f"  Target range: [{diagnostics['target_stats']['min']:.4f}, {diagnostics['target_stats']['max']:.4f}]")
    
    return diagnostics
