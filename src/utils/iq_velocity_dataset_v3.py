"""
IQ-to-Velocity Dataset v3 with on-the-fly PRF Augmentation.

Extends :class:`IQVelocityDataset` with:
- **Real** low-PRF simulation via temporal subsampling of *original* IQ data
- Correct ``dt_seconds`` that accounts for ``sparse_step`` (dt = step / prf)
- ``dt_seconds`` and ``dt_output`` in the output dictionary

Key fix over the broken v3-alpha: PRF augmentation now *actually subsamples*
the original IQ frames with a variable sparse_step, producing genuinely
different temporal patterns for each simulated PRF.  Previously only the
dt_seconds label changed while the input data stayed identical -- the model
learned nothing about low-PRF degradation.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.iq_velocity_dataset import IQVelocityDataset, normalize_prf, _pad_to_multiple

logger = logging.getLogger(__name__)


class IQVelocityDatasetV3(IQVelocityDataset):
    """
    Extended dataset with on-the-fly PRF augmentation via temporal subsampling.

    PRF Augmentation (training only)
    --------------------------------
    For samples that have ``original_path`` in metadata, a random sparse_step
    is chosen from ``augmentation_steps``.  The original IQ is loaded, ROI/window
    extracted, and subsampled at the chosen step to produce 5 sparse frames.

    This creates *genuine* temporal diversity:
    - sparse_step=2  from 500 Hz data → dt = 4 ms  (like ~250 Hz)
    - sparse_step=5  from 500 Hz data → dt = 10 ms (like 100 Hz)
    - sparse_step=10 from 500 Hz data → dt = 20 ms (like ~50 Hz)
    - sparse_step=12 from 500 Hz data → dt = 24 ms (native 500 Hz)

    dt_seconds = sparse_step × frame_time_s (time between sparse input frames).

    For samples without original_path or when augmentation is disabled,
    the preprocessed sparse data is used with dt = native_sparse_step / prf.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        enable_prf_augmentation: bool = True,
        augmentation_steps: Optional[Sequence[int]] = None,
        augmentation_step_weights: Optional[Dict[int, float]] = None,
        upsampling_factor: int = 12,
        min_sparse_frames: int = 5,
        training_frequencies: Optional[List[str]] = None,
        frequency_step_overrides: Optional[Dict[str, List[int]]] = None,
        strict_iq: bool = False,
        interp_frame_weight: float = 0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        frequency_step_overrides : dict, optional
            Maps lowercase frequency tags (e.g. "500hz", "5khz") to a list of
            allowed sparse steps.  When a sample's PRF matches a key, ONLY the
            listed steps are used — both during training augmentation AND during
            validation (overriding enable_prf_augmentation=False).  This lets
            you hard-wire the deployment step for a specific acquisition type.

            Example: ``{"500hz": [1]}`` forces step=1 (consecutive frames,
            dt=2ms) for all 500 Hz samples, which is the V10 deployment scenario.
        """
        super().__init__(data_dir=data_dir, split=split, **kwargs)

        # --- Frequency filter (e.g. ["5kHz","10kHz","20kHz"] excludes 500Hz) ---
        if training_frequencies is not None:
            freq_tags = [f.lower() for f in training_frequencies]
            original_count = len(self.samples)
            self.samples = [
                s for s in self.samples
                if any(tag in s["name"].lower() for tag in freq_tags)
            ]
            logger.info(
                f"  Frequency filter {training_frequencies}: "
                f"{len(self.samples)}/{original_count} samples retained"
            )

        self.enable_prf_augmentation = enable_prf_augmentation and (split == "train")
        self.upsampling_factor = upsampling_factor
        self.min_sparse_frames = min_sparse_frames
        # When True, a configured PRF step that cannot be honoured (missing raw
        # IQ) raises instead of silently reverting to the native step.
        self.strict_iq = strict_iq
        # Loss weight given to target frames the dataset had to interpolate.
        self.interp_frame_weight = float(interp_frame_weight)

        # Frequency-specific step overrides (applied even during val)
        # Keys are lowercase, e.g. {"500hz": [1], "5khz": [10, 12]}
        self.frequency_step_overrides: Dict[str, List[int]] = {
            k.lower(): v for k, v in (frequency_step_overrides or {}).items()
        }

        # Steps to randomly choose from during augmentation (non-override samples)
        self.augmentation_steps = list(augmentation_steps or [2, 4, 5, 6, 8, 10, 12])

        # Weights for sampling steps (higher weight → more likely)
        _step_weights = augmentation_step_weights or {
            2: 0.05, 4: 0.10, 5: 0.20, 6: 0.10, 8: 0.10, 10: 0.15, 12: 0.30,
        }
        total_w = sum(_step_weights.get(s, 0.1) for s in self.augmentation_steps)
        self.step_probs = np.array(
            [_step_weights.get(s, 0.1) / total_w for s in self.augmentation_steps],
            dtype=np.float64,
        )

        # Load per-sample metadata (original_path, roi, window, etc.)
        self._load_sample_metadata()

        if self.strict_iq:
            self._drop_unhonourable_samples()

        n_augmentable = sum(1 for s in self.samples if s.get("original_path"))
        logger.info(
            f"IQVelocityDatasetV3 [{split}]: {len(self)} samples, "
            f"PRF aug={'ON' if self.enable_prf_augmentation else 'OFF'}, "
            f"augmentable={n_augmentable}, steps={self.augmentation_steps}, "
            f"upsampling={upsampling_factor}, "
            f"freq_step_overrides={self.frequency_step_overrides}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _drop_unhonourable_samples(self):
        """Remove samples whose configured PRF step cannot actually be applied.

        A 500 Hz sample with ``frequency_step_overrides={"500hz": [1]}`` but no
        raw IQ on disk would otherwise train at the native step 12, i.e.
        dt_in = 24 ms instead of 2 ms -- far outside the regime the model is
        deployed in, and invisible in the logs.  38 of 154 500 Hz training
        samples in this environment are in exactly that state.
        """
        kept, dropped = [], []
        for s in self.samples:
            prf = s.get("prf_hz") or (
                1.0 / s["frame_time_s"] if s.get("frame_time_s") else None
            )
            steps = self.frequency_step_overrides.get(
                self._prf_to_freq_tag(prf), []) if prf else []
            native = int(s.get("sparse_step", self.upsampling_factor))
            needs_iq = bool(steps) and any(st != native for st in steps)
            path = s.get("original_path")
            if needs_iq and not (path and Path(path).exists()):
                dropped.append(s["name"])
            else:
                kept.append(s)
        if dropped:
            logger.warning(
                f"strict_iq: dropped {len(dropped)}/{len(self.samples)} samples "
                f"whose configured PRF step needs raw IQ that is not on disk "
                f"(e.g. {dropped[:3]})"
            )
        self.samples = kept

    @staticmethod
    def _prf_to_freq_tag(prf_hz: float) -> str:
        """Map a PRF value (Hz) to the lowercase frequency tag used in overrides."""
        if prf_hz <= 750.0:
            return "500hz"
        elif prf_hz <= 1500.0:
            return "1khz"
        elif prf_hz <= 7500.0:
            return "5khz"
        elif prf_hz <= 15000.0:
            return "10khz"
        else:
            return "20khz"

    def _resample_target_temporally(
        self,
        native_target: torch.Tensor,
        n_gt_in_window: int,
        T_dense: int,
    ) -> torch.Tensor:
        """
        Linearly resample the native GT from n_gt_in_window frames to T_dense
        frames.  Used when the native GT temporal resolution is coarser than
        the model's dense output (e.g. 500 Hz GT at 2 ms vs. output at 0.167 ms).

        Parameters
        ----------
        native_target : (1, T_native, H_pad, W_pad) float tensor
        n_gt_in_window : int — number of native frames inside the input window
        T_dense : int — desired number of output frames

        Returns
        -------
        (1, T_dense, H_pad, W_pad) float tensor, clipped to [-1, 1]
        """
        import torch.nn.functional as F_interp

        n_gt = min(n_gt_in_window, native_target.shape[1])
        gt_select = native_target[:, :n_gt]          # (1, n_gt, H, W)
        _, _, H_pad, W_pad = gt_select.shape

        # Reshape to (H*W, 1, n_gt) for 1-D temporal interpolation
        gt_hw = (
            gt_select[0]                             # (n_gt, H, W)
            .permute(1, 2, 0)                        # (H, W, n_gt)
            .reshape(H_pad * W_pad, n_gt)            # (H*W, n_gt)
            .unsqueeze(1)                            # (H*W, 1, n_gt)
            .float()
        )
        gt_dense_hw = F_interp.interpolate(
            gt_hw, size=T_dense, mode="linear", align_corners=True
        )                                            # (H*W, 1, T_dense)
        gt_dense = (
            gt_dense_hw
            .squeeze(1)                              # (H*W, T_dense)
            .reshape(H_pad, W_pad, T_dense)          # (H, W, T_dense)
            .permute(2, 0, 1)                        # (T_dense, H, W)
            .unsqueeze(0)                            # (1, T_dense, H, W)
        )
        return gt_dense.clamp(-1.0, 1.0)

    def _load_sample_metadata(self):
        """Attach per-sample metadata from metadata.json for augmentation."""
        meta_path = Path(self.data_dir) / "metadata.json"
        if not meta_path.exists():
            logger.warning("metadata.json not found — augmentation from original IQ disabled")
            return

        with open(meta_path) as f:
            meta = json.load(f)

        # Build lookup: name → metadata entry
        lookup: Dict[str, Dict] = {}
        for split_key in ("train", "val", "test", "inference"):
            for entry in meta.get(split_key, []):
                lookup[entry["name"]] = entry

        matched = 0
        for sample in self.samples:
            name = sample["name"]
            if name in lookup:
                entry = lookup[name]
                orig_path = entry.get("original_path")
                sample["original_path"] = orig_path
                sample["roi"] = entry.get("roi")
                sample["window"] = entry.get("window")
                sample["frame_time_s"] = entry.get("frame_time_s")
                sample["sparse_step_native"] = entry.get("sparse_step", 12)

                # Pre-compute actual T_window by checking file shape via memmap
                win = entry.get("window", [0, 50])
                if orig_path and Path(orig_path).exists():
                    try:
                        raw = np.load(orig_path, mmap_mode="r")
                        actual_T = raw.shape[2]  # (H, W, T)
                        sample["actual_T_window"] = min(win[1], actual_T) - win[0]
                    except Exception:
                        sample["actual_T_window"] = win[1] - win[0]
                else:
                    iq_shape = entry.get("original_iq_shape")
                    orig_T = iq_shape[2] if iq_shape else None
                    if orig_T is not None:
                        sample["actual_T_window"] = min(win[1], orig_T) - win[0]
                    else:
                        sample["actual_T_window"] = win[1] - win[0]

                # SWE physics metadata for V7 multi-task supervision
                tissue = entry.get("tissue_properties", {})
                sample["cs_target_m_s"] = float(tissue.get("estimated_shear_speed_m_s", 0.0))
                sample["source_phantom"] = entry.get("source", "unknown")

                matched += 1

        logger.info(f"  Metadata matched: {matched}/{len(self.samples)} samples")

    def _load_augmented_iq(
        self, sample_meta: Dict, sparse_step: int
    ) -> torch.Tensor:
        """
        Load original IQ, extract ROI/window, subsample with given step.

        Returns
        -------
        Tensor (2, T_sparse, H_pad, W_pad) float32
        """
        raw = np.load(sample_meta["original_path"])  # (H_full, W_full, T_full)
        roi = sample_meta["roi"]
        win = sample_meta["window"]

        iq_win = raw[
            roi["min_row"] : roi["max_row"] + 1,
            roi["min_col"] : roi["max_col"] + 1,
            win[0] : win[1],
        ]  # (H_roi, W_roi, T_window)

        # Transpose to (T_window, H_roi, W_roi)
        iq_win = iq_win.transpose(2, 0, 1)

        # Subsample with the chosen step — take first min_sparse_frames frames
        indices = list(range(0, iq_win.shape[0], sparse_step))[: self.min_sparse_frames]
        if len(indices) < self.min_sparse_frames:
            raise ValueError(
                f"Only {len(indices)} frames available (need {self.min_sparse_frames}) "
                f"with step={sparse_step}, T_window={iq_win.shape[0]}"
            )
        sparse_iq = iq_win[indices]  # (T_sparse, H_roi, W_roi) complex64

        # Normalise (same logic as parent class)
        if self.per_sample_iq_norm:
            norm = float(np.percentile(np.abs(sparse_iq), 99))
            norm = max(norm, 1.0)
            sparse_iq = sparse_iq / norm
        else:
            sparse_iq = sparse_iq / self.iq_norm_scalar

        # Separate real/imag → (2, T, H, W)
        real = sparse_iq.real.astype(np.float32)
        imag = sparse_iq.imag.astype(np.float32)
        input_array = np.stack([real, imag], axis=0)

        # Pad spatial dims (same as parent)
        input_array, _ = _pad_to_multiple(input_array, self.spatial_multiple, axes=(2, 3))

        return torch.from_numpy(input_array)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the default sample from parent (preprocessed sparse data)
        sample = super().__getitem__(idx)
        sample_meta = self.samples[idx]

        native_step = sample_meta.get("sparse_step_native", 12)
        frame_time_s = sample_meta.get("frame_time_s")
        prf = float(sample["prf_hz"].item())

        # Determine this sample's frequency tag for override lookup
        freq_tag = self._prf_to_freq_tag(prf)
        override_steps = self.frequency_step_overrides.get(freq_tag)

        # -----------------------------------------------------------------
        # Step selection
        # Three paths:
        #   A) frequency_step_overrides defines steps for this freq → use them
        #      (applies to both train and val — this is the deployment-forcing
        #      mechanism, not controlled by enable_prf_augmentation)
        #   B) enable_prf_augmentation=True and original IQ available → random
        #      from augmentation_steps
        #   C) Otherwise → native step from metadata
        # -----------------------------------------------------------------
        if (
            override_steps
            and sample_meta.get("original_path")
            and frame_time_s is not None
        ):
            # Path A: forced step for this frequency (train or val)
            T_window = max(sample_meta.get("actual_T_window", 50), 1)
            max_step = (T_window - 1) // max(self.min_sparse_frames - 1, 1)
            feasible = [s for s in override_steps if s <= max_step]

            if feasible:
                # Training: random among feasible overrides; Val: always first
                if self.enable_prf_augmentation and len(feasible) > 1:
                    chosen_step = int(np.random.choice(feasible))
                else:
                    chosen_step = feasible[0]
            else:
                chosen_step = native_step

            if chosen_step != native_step:
                try:
                    sample["input"] = self._load_augmented_iq(sample_meta, chosen_step)
                except Exception as e:
                    # Falling back to the native step silently is how a whole
                    # training run was invalidated once before (see
                    # outputs/wavepinn_v12_INVALID_prefix_missing_raw_iq): for a
                    # 500 Hz sample it swaps dt_in = 2 ms for dt_in = 24 ms and
                    # nothing in the logs says so, because DataLoader workers
                    # swallow the warning.  strict_iq makes it fatal.
                    if self.strict_iq:
                        raise RuntimeError(
                            f"strict_iq: cannot load override IQ for "
                            f"{sample_meta['name']} at step={chosen_step}: {e}"
                        ) from e
                    logger.warning(
                        f"Failed loading override IQ {sample_meta['name']} "
                        f"step={chosen_step}: {e}"
                    )
                    chosen_step = native_step

            dt_seconds = chosen_step * frame_time_s

        elif (
            self.enable_prf_augmentation
            and sample_meta.get("original_path")
            and frame_time_s is not None
        ):
            # Path B: random augmentation
            T_window = max(sample_meta.get("actual_T_window", 50), 1)
            max_step = (T_window - 1) // max(self.min_sparse_frames - 1, 1)
            feasible_steps = [s for s in self.augmentation_steps if s <= max_step]
            feasible_probs = np.array(
                [self.step_probs[self.augmentation_steps.index(s)] for s in feasible_steps],
                dtype=np.float64,
            )

            if feasible_steps and feasible_probs.sum() > 0:
                feasible_probs /= feasible_probs.sum()
                chosen_step = int(np.random.choice(feasible_steps, p=feasible_probs))
            else:
                chosen_step = native_step

            if chosen_step != native_step:
                try:
                    sample["input"] = self._load_augmented_iq(sample_meta, chosen_step)
                except Exception as e:
                    logger.warning(
                        f"Failed augmenting {sample_meta['name']} step={chosen_step}: {e}"
                    )
                    chosen_step = native_step

            dt_seconds = chosen_step * frame_time_s

        else:
            # Path C: native step
            chosen_step = native_step
            if frame_time_s is not None:
                dt_seconds = native_step * frame_time_s
            else:
                dt_seconds = native_step / max(prf, 1.0)

        # -----------------------------------------------------------------
        # Temporal GT resampling — VERIFIED CORRECT (do not change logic).
        # When the native GT rate is much coarser than the model's dense
        # output rate, we must resample rather than taking the first T_dense
        # frames at the native rate (which would span a completely different
        # time window).
        #
        # Condition: dt_native > dt_output × RESAMPLE_RATIO_THRESHOLD
        #   5kHz step=12: dt_output=0.2ms, dt_native=0.2ms → ratio=1.0 → skip
        #   5kHz step=10: dt_output=0.167ms, dt_native=0.2ms → ratio=1.2 → skip
        #   500Hz step=1: dt_output=0.167ms, dt_native=2ms   → ratio=12  → resample
        #
        # Known half-frame timestamp ambiguity: for 500 Hz with step=1 and
        # upsampling_factor=12, the Loupas velocity target frame at index k
        # represents the interval (k*dt_native, (k+1)*dt_native], not the
        # midpoint.  The model's dense output is timestamped at
        # t = i * dt_output = i * dt_native / 12.  This off-by-half-frame
        # is negligible relative to the 2ms GT frame interval but is noted
        # here for completeness.  Do not compensate — the training/val
        # split is consistent and the effect on cs is immeasurable.
        # -----------------------------------------------------------------
        RESAMPLE_RATIO_THRESHOLD = 2.0

        T_dense = 1 + (self.min_sparse_frames - 1) * self.upsampling_factor
        # Which dense frames carry GENUINE ground truth.  All of them unless the
        # resampling branch below invents the in-between frames.
        frame_mask = torch.ones(T_dense, dtype=torch.float32)

        if frame_time_s is not None:
            dt_native = frame_time_s               # native PRF frame interval
            dt_output = dt_seconds / self.upsampling_factor

            if dt_native > dt_output * RESAMPLE_RATIO_THRESHOLD:
                # Number of native GT frames within the input time window [0, 4*dt_seconds]
                t_input_max     = (self.min_sparse_frames - 1) * dt_seconds
                n_gt_in_window  = int(round(t_input_max / dt_native)) + 1
                sample["target"] = self._resample_target_temporally(
                    sample["target"], n_gt_in_window, T_dense
                )
                # The resampled target is a LINEAR INTERPOLATION of
                # n_gt_in_window real frames: only those knot positions are
                # real supervision, everything between them is an artefact of
                # the interpolation.  Training against the invented frames
                # rewards piecewise-linear output, which is precisely the
                # failure mode V14 exists to remove.  Mark the knots so the
                # loss can down-weight the rest (see velocity_loss_v14.py).
                #
                # interp_frame_weight is a knob rather than a hard 0 because the
                # interpolated frames are not pure noise: below the knot Nyquist
                # (250 Hz for 500 Hz step-1) linear interpolation is a fair
                # estimate of the envelope, and only above that band does it
                # invent content.  Zeroing them outright leaves the model
                # entirely unconstrained between knots in the 500 Hz deployment
                # regime, which measurably distorts the output spectrum (a
                # trough at 250-500 Hz then a spurious bump above it) and
                # inflates the fitted viscosity.  A small positive weight pins
                # the envelope without re-teaching piecewise-linear motion.
                frame_mask = torch.full(
                    (T_dense,), float(self.interp_frame_weight), dtype=torch.float32
                )
                if n_gt_in_window > 1:
                    knots = torch.linspace(0, T_dense - 1, n_gt_in_window)
                    frame_mask[knots.round().long().clamp(0, T_dense - 1)] = 1.0
                else:
                    frame_mask[0] = 1.0

        sample["target_frame_mask"] = frame_mask

        dt_output = dt_seconds / self.upsampling_factor

        sample["dt_seconds"] = torch.tensor(dt_seconds, dtype=torch.float32)
        sample["dt_output"] = torch.tensor(dt_output, dtype=torch.float32)

        # --- V7 SWE physics targets ---
        # cs_target_scalar: phantom ground-truth shear wave speed (m/s)
        cs_gt = sample_meta.get("cs_target_m_s", 0.0)
        sample["cs_target_scalar"] = torch.tensor(cs_gt, dtype=torch.float32)

        # push_col: column index of push excitation (precomputed from target).
        # Used by differentiable ToA loss to exclude push zone.
        target_vel = sample["target"]  # (1, T, H, W)
        T_dense = target_vel.shape[1]
        H = target_vel.shape[2]
        row_lo = int(0.2 * H)
        row_hi = int(0.8 * H)
        # Depth-average |target| over 20-80% depth, first 10% of frames
        n_early = max(3, T_dense // 10)
        early_abs = target_vel[0, :n_early, row_lo:row_hi, :].abs()  # (n_early, H_roi, W)
        # Mean over depth → (n_early, W), then max over time → (W,)
        col_energy = early_abs.mean(dim=1).max(dim=0).values  # (W,)
        push_col = int(col_energy.argmax().item())
        sample["push_col"] = torch.tensor(push_col, dtype=torch.int64)

        # source phantom type for stratified sampling
        sample["source"] = sample_meta.get("source_phantom", "unknown")

        return sample


# =========================================================================
# Custom collate
# =========================================================================

def collate_fn_v3(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for v3 dataset.

    T_sparse is always min_sparse_frames (5), so this is a simple stack.
    ``name`` is kept as a list of strings.
    """
    result: Dict[str, Any] = {}
    str_keys = {"name", "source"}
    for key in batch[0]:
        if key in str_keys:
            result[key] = [s[key] for s in batch]
        else:
            result[key] = torch.stack([s[key] for s in batch])
    return result


# =========================================================================
# Factory
# =========================================================================

def create_iq_velocity_dataloaders_v3(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    enable_prf_augmentation: bool = True,
    augmentation_steps: Optional[Sequence[int]] = None,
    augmentation_step_weights: Optional[Dict[int, float]] = None,
    upsampling_factor: int = 12,
    training_frequencies: Optional[List[str]] = None,
    frequency_step_overrides: Optional[Dict[str, List[int]]] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train / val DataLoaders with on-the-fly PRF augmentation.

    Parameters
    ----------
    training_frequencies : list of str or None
        If set, only samples whose name contains one of these frequency tags
        (case-insensitive) are included.  E.g. ["5kHz","10kHz","20kHz"]
        excludes the 500 Hz native-rate samples.  None = keep all.

    Returns
    -------
    train_loader, val_loader, data_info
    """
    # Remove old v3-alpha parameters that are no longer used
    kwargs.pop("simulated_prfs", None)
    kwargs.pop("prf_weights", None)

    train_dataset = IQVelocityDatasetV3(
        data_dir=data_dir,
        split="train",
        enable_prf_augmentation=enable_prf_augmentation,
        augmentation_steps=augmentation_steps,
        augmentation_step_weights=augmentation_step_weights,
        upsampling_factor=upsampling_factor,
        training_frequencies=training_frequencies,
        frequency_step_overrides=frequency_step_overrides,
        **kwargs,
    )

    # Avoid passing duplicate normalisation kwargs when explicitly setting
    # val scalars from the train dataset below.
    val_kwargs = dict(kwargs)
    val_kwargs.pop("iq_norm_scalar", None)
    val_kwargs.pop("vel_norm_scalar", None)

    val_dataset = IQVelocityDatasetV3(
        data_dir=data_dir,
        split="val",
        enable_prf_augmentation=False,  # no augmentation during validation
        upsampling_factor=upsampling_factor,
        iq_norm_scalar=train_dataset.iq_norm_scalar,
        vel_norm_scalar=train_dataset.vel_norm_scalar,
        frequency_step_overrides=frequency_step_overrides,  # still applied on val
        **val_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn_v3,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn_v3,
    )

    data_info = train_dataset.get_data_info()
    data_info["num_val"] = len(val_dataset)

    return train_loader, val_loader, data_info
