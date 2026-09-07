"""
Dataset for IQ-to-Velocity Training (Loupas Pipeline Targets).

Loads paired sparse IQ input sequences and dense velocity targets generated
by the 2D Loupas autocorrelator pipeline (scripts/iq_to_velocity.py).

Data Format:
    Input:  sparse_<name>.npy   — (T_sparse, H, W) complex64 (clutter-filtered IQ)
    Target: velocity_<name>.npy — (T_vel, H, W) float32 (velocity in mm/s)
    Mask:   magnitude_<name>.npy — (H, W) float32 (ROI binary mask)

Output Tensors:
    input_tensor:  (2, T_sparse, H_pad, W_pad) float32 — Real/Imag channels
    target_tensor: (1, T_vel, H_pad, W_pad)    float32 — Normalised velocity [-1, 1]
    mask_tensor:   (1, 1, H_pad, W_pad)        float32 — ROI mask
    prf_hz:        scalar float32 — Detection PRF in Hz (for PRF conditioning)
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRF Parsing Utilities
# ---------------------------------------------------------------------------

# Mapping from frequency folder names to PRF in Hz
FREQ_TO_PRF_HZ: Dict[str, float] = {
    '500Hz': 500.0,
    '1kHz': 1000.0,
    '5kHz': 5000.0,
    '10kHz': 10000.0,
    '20kHz': 20000.0,
}

# Regex to extract frequency string from a sample name
_FREQ_PATTERN = re.compile(r'(?:500Hz|[0-9]+kHz)')


def parse_prf_from_name(name: str) -> float:
    """
    Extract PRF (Hz) from a sample name like
    '10mmfocus_100kPa_inclusion_10kHz_20260128T165121'.

    Falls back to 10000.0 Hz if no frequency token is found (backward compat).
    """
    match = _FREQ_PATTERN.search(name)
    if match:
        freq_str = match.group(0)
        prf = FREQ_TO_PRF_HZ.get(freq_str)
        if prf is not None:
            return prf
    # Fallback for legacy datasets that were only 10/20kHz
    return 10000.0


def normalize_prf(prf_hz: float) -> float:
    """
    Normalise PRF to a model-friendly range using log10 scale.

    Returns log10(prf_hz / 10000): 500Hz→-1.30, 1kHz→-1.0, 5kHz→-0.30,
    10kHz→0.0, 20kHz→+0.30.
    """
    return math.log10(prf_hz / 10000.0)


def _pad_to_multiple(arr: np.ndarray, multiple: int, axes: Tuple[int, ...]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Pad array so that specified axes have sizes that are multiples of `multiple`."""
    pad_widths = [(0, 0)] * arr.ndim
    for ax in axes:
        size = arr.shape[ax]
        remainder = size % multiple
        if remainder != 0:
            pad_widths[ax] = (0, multiple - remainder)
    padded = np.pad(arr, pad_widths, mode='constant', constant_values=0)
    return padded, pad_widths


class IQVelocityDataset(Dataset):
    """
    PyTorch Dataset for IQ → Velocity training using Loupas-pipeline targets.

    Preprocessing (applied on load):
        1. Load sparse IQ (complex64) → separate real/imag → (2, T, H, W)
        2. Apply IQ normalisation: divide by iq_norm_scalar
        3. Load velocity (float32) → normalise to [-1, 1] via vel_norm_scalar
        4. Pad spatial dims to nearest multiple of 16 (required by UNet)
        5. Load ROI mask and apply same padding

    Attributes:
        samples: List of (sparse_path, velocity_path, mask_path) tuples
        iq_norm_scalar: Scalar for IQ normalisation (default: computed from metadata)
        vel_norm_scalar: Scalar for velocity normalisation (default: from metadata)
        spatial_multiple: Pad spatial dims to this multiple (default: 16)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        iq_norm_scalar: Optional[float] = None,
        vel_norm_scalar: Optional[float] = None,
        spatial_multiple: int = 16,
        apply_clutter_filter: bool = False,  # Already done in preprocessing
        transform: Optional[Callable] = None,
        per_sample_iq_norm: bool = False,
        per_sample_vel_norm: bool = False,
    ):
        """
        Args:
            data_dir: Root preprocessed directory (contains train/, val/, metadata.json)
            split: 'train' or 'val'
            iq_norm_scalar: Override IQ normalisation scalar
            vel_norm_scalar: Override velocity normalisation scalar (mm/s)
            spatial_multiple: Pad spatial dimensions to this multiple
            apply_clutter_filter: Whether to apply additional clutter filtering
            transform: Optional callable transform
            per_sample_iq_norm: If True, normalise each IQ sample by its own
                99th-percentile magnitude (recommended for datasets with large
                dynamic range across acquisitions)
            per_sample_vel_norm: If True, normalise each velocity target by its
                own 99th-percentile absolute value so targets fill the [-1, 1]
                Tanh output range (recommended when peak velocities vary widely)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.spatial_multiple = spatial_multiple
        self.apply_clutter_filter = apply_clutter_filter
        self.transform = transform
        self.per_sample_iq_norm = per_sample_iq_norm
        self.per_sample_vel_norm = per_sample_vel_norm

        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Set normalisation scalars
        if vel_norm_scalar is not None:
            self.vel_norm_scalar = vel_norm_scalar
        else:
            norm_info = self.metadata.get('normalization', {})
            self.vel_norm_scalar = norm_info.get('velocity_abs_max_99pct', 70.0)

        # Discover samples
        self.samples = self._discover_samples(split_dir)

        # Compute IQ normalisation from first sample if not provided
        if iq_norm_scalar is not None:
            self.iq_norm_scalar = iq_norm_scalar
        else:
            self.iq_norm_scalar = self._compute_iq_norm_scalar()

        logger.info(
            f"IQVelocityDataset[{split}]: {len(self.samples)} samples, "
            f"IQ norm={self.iq_norm_scalar:.2e}, "
            f"vel norm={self.vel_norm_scalar:.2f} mm/s"
        )

    def _discover_samples(self, split_dir: Path) -> List[Dict[str, Any]]:
        """Find all sparse/velocity/magnitude triplets in the split directory."""
        samples = []
        sparse_files = sorted(split_dir.glob('sparse_*.npy'))

        for sparse_path in sparse_files:
            name = sparse_path.stem[len('sparse_'):]  # Remove 'sparse_' prefix
            vel_path = split_dir / f'velocity_{name}.npy'
            mag_path = split_dir / f'magnitude_{name}.npy'

            if vel_path.exists() and mag_path.exists():
                prf_hz = parse_prf_from_name(name)
                samples.append({
                    'name': name,
                    'sparse': sparse_path,
                    'velocity': vel_path,
                    'magnitude': mag_path,
                    'prf_hz': prf_hz,
                })
            else:
                logger.warning(f"Missing target/mask for {name}, skipping")

        # Log PRF distribution
        prf_counts: Dict[float, int] = {}
        for s in samples:
            prf_counts[s['prf_hz']] = prf_counts.get(s['prf_hz'], 0) + 1
        if prf_counts:
            dist_str = ', '.join(f'{p:.0f}Hz: {c}' for p, c in sorted(prf_counts.items()))
            logger.info(f"  PRF distribution [{split_dir.name}]: {dist_str}")

        return samples

    def _compute_iq_norm_scalar(self) -> float:
        """Compute IQ normalisation scalar from first few samples."""
        if not self.samples:
            return 5000.0  # Default fallback

        max_vals = []
        n_check = min(5, len(self.samples))
        for i in range(n_check):
            iq = np.load(self.samples[i]['sparse'])
            max_vals.append(np.abs(iq).max())

        # Use 99th percentile of max magnitudes
        return float(np.percentile(max_vals, 99)) if max_vals else 5000.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.

        Returns:
            Dictionary with keys:
                'input':   (2, T_sparse, H_pad, W_pad) float32
                'target':  (1, T_vel, H_pad, W_pad)    float32 in [-1, 1]
                'mask':    (1, 1, H_pad, W_pad)         float32
                'prf_hz':  scalar float32 — detection PRF in Hz
                'log_prf': scalar float32 — log10(prf/10 kHz), for conditioning
                'name':    str
        """
        sample = self.samples[idx]

        # --- Load sparse IQ ---
        iq_data = np.load(sample['sparse'])  # (T_sparse, H, W) complex64

        # Optional additional clutter filter
        if self.apply_clutter_filter:
            iq_data = iq_data - iq_data.mean(axis=0, keepdims=True)

        # Normalise IQ
        if self.per_sample_iq_norm:
            # Per-sample normalisation: robust to 893× dynamic range across
            # acquisitions.  Each sample is scaled so its 99th-pct magnitude ≈ 1.
            sample_norm = float(np.percentile(np.abs(iq_data), 99))
            sample_norm = max(sample_norm, 1.0)   # guard against all-zero
            iq_data = iq_data / sample_norm
        else:
            iq_data = iq_data / self.iq_norm_scalar

        # Separate real/imag: (T, H, W) complex → (2, T, H, W) float32
        real = iq_data.real.astype(np.float32)
        imag = iq_data.imag.astype(np.float32)
        input_array = np.stack([real, imag], axis=0)  # (2, T, H, W)

        # --- Load velocity target ---
        vel_data = np.load(sample['velocity'])  # (T_vel, H, W) float32

        # Normalise velocity to approximately [-1, 1]
        if self.per_sample_vel_norm:
            vel_scale = float(np.percentile(np.abs(vel_data), 99))
            vel_scale = max(vel_scale, 1.0)  # guard against all-zero
            vel_data = np.clip(vel_data / vel_scale, -1.0, 1.0)
        else:
            vel_scale = self.vel_norm_scalar
            vel_data = np.clip(vel_data / vel_scale, -1.0, 1.0)

        # Add channel dim: (T, H, W) → (1, T, H, W)
        target_array = vel_data[np.newaxis, :, :, :]  # (1, T_vel, H, W)

        # --- Load magnitude mask ---
        mask = np.load(sample['magnitude'])  # (H, W)
        mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

        # --- Pad spatial dims ---
        input_array, _ = _pad_to_multiple(input_array, self.spatial_multiple, axes=(2, 3))
        target_array, _ = _pad_to_multiple(target_array, self.spatial_multiple, axes=(2, 3))
        mask, _ = _pad_to_multiple(mask, self.spatial_multiple, axes=(2, 3))

        # --- Convert to tensors ---
        input_tensor = torch.from_numpy(input_array)
        target_tensor = torch.from_numpy(target_array)
        mask_tensor = torch.from_numpy(mask)

        # --- PRF information ---
        prf_hz = sample.get('prf_hz', 10000.0)
        prf_tensor = torch.tensor(prf_hz, dtype=torch.float32)
        log_prf_tensor = torch.tensor(normalize_prf(prf_hz), dtype=torch.float32)

        # --- Optional transform ---
        if self.transform is not None:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return {
            'input': input_tensor,
            'target': target_tensor,
            'mask': mask_tensor,
            'prf_hz': prf_tensor,
            'log_prf': log_prf_tensor,
            'name': sample['name'],
            'vel_scale': torch.tensor(vel_scale, dtype=torch.float32),
        }

    def get_data_info(self) -> Dict[str, Any]:
        """Return dataset metadata for model configuration."""
        if not self.samples:
            return {}

        sample = self[0]
        prf_set = sorted(set(s['prf_hz'] for s in self.samples))
        return {
            'T_sparse': sample['input'].shape[1],
            'T_dense': sample['target'].shape[1],
            'H': sample['input'].shape[2],
            'W': sample['input'].shape[3],
            'num_train': len(self.samples) if self.split == 'train' else 0,
            'num_val': len(self.samples) if self.split == 'val' else 0,
            'iq_norm_scalar': self.iq_norm_scalar,
            'vel_norm_scalar': self.vel_norm_scalar,
            'prf_hz_set': prf_set,
        }


def create_iq_velocity_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    iq_norm_scalar: Optional[float] = None,
    vel_norm_scalar: Optional[float] = None,
    per_sample_iq_norm: bool = False,
    per_sample_vel_norm: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation DataLoaders.

    Args:
        data_dir: Root preprocessed data directory
        batch_size: Batch size
        num_workers: Data loading workers
        pin_memory: Pin memory for GPU transfer
        iq_norm_scalar: Override IQ norm scalar
        vel_norm_scalar: Override velocity norm scalar
        per_sample_iq_norm: Normalise each IQ sample individually
        per_sample_vel_norm: Normalise each velocity target individually

    Returns:
        (train_loader, val_loader, data_info)
    """
    train_dataset = IQVelocityDataset(
        data_dir=data_dir,
        split='train',
        iq_norm_scalar=iq_norm_scalar,
        vel_norm_scalar=vel_norm_scalar,
        per_sample_iq_norm=per_sample_iq_norm,
        per_sample_vel_norm=per_sample_vel_norm,
    )

    # Use same normalisation for validation
    val_dataset = IQVelocityDataset(
        data_dir=data_dir,
        split='val',
        iq_norm_scalar=train_dataset.iq_norm_scalar,
        vel_norm_scalar=train_dataset.vel_norm_scalar,
        per_sample_iq_norm=per_sample_iq_norm,
        per_sample_vel_norm=per_sample_vel_norm,
    )

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

    data_info = train_dataset.get_data_info()
    data_info['num_val'] = len(val_dataset)
    data_info['iq_norm_scalar'] = train_dataset.iq_norm_scalar
    data_info['vel_norm_scalar'] = train_dataset.vel_norm_scalar

    logger.info(
        f"DataLoaders created: "
        f"train={len(train_dataset)} samples ({len(train_loader)} batches), "
        f"val={len(val_dataset)} samples ({len(val_loader)} batches)"
    )

    return train_loader, val_loader, data_info
