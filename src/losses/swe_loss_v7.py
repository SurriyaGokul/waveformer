"""
SWE-Aware Loss Functions v7 — Multi-Task Physics-Informed.

Three new loss components for WavePINN v7:

1. DiffToaCsLoss
   Differentiable surrogate of the ToA cross-correlation pipeline from
   swe_analysis.py:estimate_cs_toa().  Backpropagates through the predicted
   velocity field to directly supervise shear wave speed.

2. SpatialSparsityLoss
   Penalises spurious model output in background regions where the target
   is near-zero.  Addresses V6's 3x background variance ratio.

3. TemporalEnvelopeLoss
   Penalises mismatch in per-frame energy envelope between pred and target.
   Addresses V6's temporal compression (over-predict early, under-predict late).

Plus the composite EnhancedVelocityLossV7 that wraps everything.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_loss_v5 import WaveCoherenceLoss

logger = logging.getLogger(__name__)

# Physical constants (same as swe_analysis.py)
LATERAL_M_PER_PX = 1.013e-4  # m per lateral pixel


# =============================================================================
# 1. Differentiable ToA Cross-Correlation Loss
# =============================================================================

class DiffToaCsLoss(nn.Module):
    """
    Differentiable surrogate of swe_analysis.py::estimate_cs_toa().

    Mirrors the real ToA pipeline:
      1. Depth-average velocity over 20-80% of H
      2. Unit-energy normalise each lateral column
      3. Cross-correlate column pairs with stride STEP
      4. Parabolic sub-pixel peak-lag extraction
      5. Convert lag → local cs
      6. Loss = weighted L1(cs_local, cs_target)

    Differentiable through the predicted velocity field via soft-argmax
    (temperature-scaled softmax over the full cross-correlation vector).
    Unlike the original parabolic approach, this provides gradients through
    ALL lag positions, not just sub-pixel corrections around a detached peak.

    Parameters
    ----------
    depth_frac : tuple
        Depth fraction for ROI [lo, hi].
    push_exclusion_px : int
        Columns excluded each side of push centre.
    step : int
        Lateral column stride for cross-correlation pairs.
    cs_lo, cs_hi : float
        Physical plausibility bounds (m/s) for soft gating.
    softargmax_tau : float
        Temperature for soft-argmax lag extraction.  Higher = more peaked
        (closer to hard argmax).  Default 10.0 works well for unit-energy
        normalised xcorr values in [-1, 1].
    """

    def __init__(
        self,
        depth_frac: Tuple[float, float] = (0.2, 0.8),
        push_exclusion_px: int = 15,
        step: int = 5,
        cs_lo: float = 0.2,
        cs_hi: float = 15.0,
        softargmax_tau: float = 10.0,
    ):
        super().__init__()
        self.depth_frac = depth_frac
        self.push_exclusion_px = push_exclusion_px
        self.step = step
        self.cs_lo = cs_lo
        self.cs_hi = cs_hi
        self.softargmax_tau = softargmax_tau

    def forward(
        self,
        pred: torch.Tensor,
        vel_scale: torch.Tensor,
        dt_output: torch.Tensor,
        cs_target: torch.Tensor,
        push_col: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred        : (B, 1, T, H, W)  normalised velocity in [-1, 1]
        vel_scale   : (B,)  per-sample denormalisation scale (mm/s)
        dt_output   : (B,)  time between dense output frames (seconds)
        cs_target   : (B,)  ground-truth shear wave speed (m/s)
        push_col    : (B,)  push excitation column index
        mask        : (B, 1, 1, H, W)  optional ROI mask

        Returns
        -------
        Scalar loss (mean over batch).
        """
        B, _, T, H, W = pred.shape
        device = pred.device

        # 1. Denormalise to m/s
        v = pred[:, 0] * vel_scale[:, None, None, None] * 1e-3  # (B, T, H, W) m/s

        # 2. Depth slice
        row_lo = int(self.depth_frac[0] * H)
        row_hi = int(self.depth_frac[1] * H)
        v_roi = v[:, :, row_lo:row_hi, :]  # (B, T, H_roi, W)

        # 3. Depth-averaged signed velocity
        v_avg = v_roi.mean(dim=2)  # (B, T, W)

        # 4. Soft amplitude gate (differentiable)
        peak_amp = v_avg.abs().max(dim=1).values  # (B, W)
        amp_mean = peak_amp.mean(dim=1, keepdim=True)  # (B, 1)
        amp_thresh = amp_mean * 0.3
        w_col = torch.sigmoid((peak_amp - amp_thresh) * 50.0)  # (B, W)

        # 5. Push-zone exclusion (fixed from target, not differentiable)
        col_idx = torch.arange(W, device=device, dtype=torch.float32)  # (W,)
        push_col_f = push_col.float()  # (B,)
        dist = (col_idx[None, :] - push_col_f[:, None]).abs()  # (B, W)
        w_push = torch.sigmoid((dist - float(self.push_exclusion_px)) * 0.5)  # (B, W)

        w_valid = w_col * w_push  # (B, W)

        # 6. Unit-energy normalisation per column
        energy = (v_avg ** 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()  # (B, 1, W)
        v_norm = v_avg / energy  # (B, T, W)

        # 7. Cross-correlation between column pairs at stride STEP
        STEP = self.step
        n_pairs = max(0, W - STEP)
        if n_pairs == 0:
            return torch.zeros([], device=device)

        # Vectorised cross-correlation via conv1d
        # For each pair (x, x+STEP), compute xcorr = conv1d(sig_a, flip(sig_b))
        # Batch all pairs across all samples
        sig_a = v_norm[:, :, :n_pairs]           # (B, T, n_pairs)
        sig_b = v_norm[:, :, STEP:STEP+n_pairs]  # (B, T, n_pairs)

        # Reshape for grouped conv1d: (B*n_pairs, 1, T)
        sig_a_flat = sig_a.permute(0, 2, 1).reshape(B * n_pairs, 1, T)
        sig_b_flat = sig_b.permute(0, 2, 1).reshape(B * n_pairs, 1, T)

        # Cross-correlation via conv1d with flipped kernel
        # Pad sig_a so output has length 2T-1
        padded = F.pad(sig_a_flat, (T - 1, T - 1))  # (B*n_pairs, 1, 3T-2)
        sig_b_flip = sig_b_flat.flip(dims=[2])        # (B*n_pairs, 1, T)

        # Groups=B*n_pairs: each pair independently
        xcorr = F.conv1d(
            padded.reshape(1, B * n_pairs, -1),
            sig_b_flip.reshape(B * n_pairs, 1, T),
            groups=B * n_pairs,
        )  # (1, B*n_pairs, 2T-1)
        xcorr = xcorr.reshape(B, n_pairs, 2 * T - 1)  # (B, n_pairs, 2T-1)

        # 8. Soft-argmax peak lag extraction (FULLY differentiable)
        # Temperature-scaled softmax over the full xcorr vector gives gradients
        # through ALL lag positions.  This replaces the V7.0 approach of
        # detached-integer + parabolic-frac which could only adjust ±0.5 frames
        # and provided no gradient when the integer lag was wrong.
        lag_indices = torch.arange(
            2 * T - 1, device=device, dtype=xcorr.dtype,
        ) - (T - 1)  # centred: [-(T-1), ..., 0, ..., T-1]
        weights = F.softmax(xcorr * self.softargmax_tau, dim=2)  # (B, n_pairs, 2T-1)
        peak_lag = (weights * lag_indices[None, None, :]).sum(dim=2)  # (B, n_pairs)

        # 9. Lag → cs
        dt_per_pair = dt_output[:, None].expand(-1, n_pairs)  # (B, n_pairs)
        delay_s = peak_lag * dt_per_pair  # (B, n_pairs) seconds
        cs_local = (STEP * LATERAL_M_PER_PX) / delay_s.abs().clamp(min=1e-9)  # (B, n_pairs)

        # 10. Physical plausibility soft gate
        w_phys = (
            torch.sigmoid((cs_local - self.cs_lo) * 20.0)
            * torch.sigmoid((self.cs_hi - cs_local) * 20.0)
        )  # (B, n_pairs)

        # Column-pair weight
        w_a = w_valid[:, :n_pairs]        # (B, n_pairs)
        w_b = w_valid[:, STEP:STEP+n_pairs]  # (B, n_pairs)
        w_pair = w_a * w_b * w_phys       # (B, n_pairs)

        # 11. Loss: weighted L1 between local cs and target
        cs_tgt = cs_target[:, None].expand(-1, n_pairs)  # (B, n_pairs)
        loss_per_pair = (cs_local - cs_tgt).abs() * w_pair
        loss_per_sample = loss_per_pair.sum(dim=1) / w_pair.sum(dim=1).clamp(min=1e-6)

        return loss_per_sample.mean()


# =============================================================================
# 2. Spatial Sparsity Loss
# =============================================================================

class SpatialSparsityLoss(nn.Module):
    """
    Penalises model output in background regions where target is near-zero.

    Addresses V6's 3x background variance ratio.  Encourages the model to
    output near-zero where no wave is expected.

    Loss = mean(|pred|) in background region (|target| < threshold).
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred, target : (B, 1, T, H, W)
        mask : (B, 1, 1, H, W) optional
        """
        # Background: where max-over-time |target| < threshold
        max_abs_target = target.abs().max(dim=2, keepdim=True).values  # (B, 1, 1, H, W)
        bg_mask = (max_abs_target < self.threshold).float()  # (B, 1, 1, H, W)

        if mask is not None:
            bg_mask = bg_mask * mask  # Only within ROI

        bg_mask_expanded = bg_mask.expand_as(pred)  # (B, 1, T, H, W)
        n_bg = bg_mask_expanded.sum().clamp(min=1.0)

        loss = (pred.abs() * bg_mask_expanded).sum() / n_bg
        return loss


# =============================================================================
# 3. Temporal Envelope Loss
# =============================================================================

class TemporalEnvelopeLoss(nn.Module):
    """
    Penalises mismatch in per-frame energy envelope.

    Addresses V6's temporal compression (over-predict early, under-predict late).
    Computes mean(|v|) per frame inside the ROI mask and penalises the L1
    difference between pred and target envelopes.

    This is a lightweight constraint that forces the model to match the
    temporal *decay profile* of the wave without constraining per-pixel values.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred, target : (B, 1, T, H, W)
        mask : (B, 1, 1, H, W) optional
        """
        if mask is not None:
            mask_expanded = mask.expand_as(pred)
            # Per-frame mean absolute value inside mask
            n_per_frame = mask_expanded.sum(dim=(1, 3, 4)).clamp(min=1.0)  # (B, T)
            pred_env = (pred.abs() * mask_expanded).sum(dim=(1, 3, 4)) / n_per_frame  # (B, T)
            tgt_env = (target.abs() * mask_expanded).sum(dim=(1, 3, 4)) / n_per_frame
        else:
            pred_env = pred.abs().mean(dim=(1, 3, 4))   # (B, T)
            tgt_env = target.abs().mean(dim=(1, 3, 4))

        return (pred_env - tgt_env).abs().mean()


# =============================================================================
# 4. Composite Loss: EnhancedVelocityLossV7
# =============================================================================

class EnhancedVelocityLossV7(nn.Module):
    """
    Multi-task composite loss for WavePINN v7.

    Components (inherited from V6):
      L1, MSE          : amplitude-weighted pixel losses
      GDL              : spatial gradient difference
      Temporal          : temporal gradient fidelity
      Wave coherence   : scale-free wave equation enforcement

    New in V7:
      DiffToA cs       : differentiable ToA cross-correlation cs loss
      Cs map           : direct physics head supervision
      Cs mean          : scalar cs consistency
      Spatial sparsity : suppress spurious background
      Temporal envelope: fix temporal compression

    All weights are dynamically settable via set_weights() for curriculum.
    """

    def __init__(
        self,
        # V6 heritage
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.15,
        wave_weight: float = 0.0,
        use_mask: bool = True,
        min_energy_fraction: float = 0.1,
        amplitude_weighting: bool = False,
        amplitude_weight_gamma: float = 1.0,
        amplitude_weight_bias: float = 0.01,  # reduced from V6's 0.05
        # V7 new
        toa_weight: float = 0.0,
        cs_map_weight: float = 0.0,
        cs_mean_weight: float = 0.0,
        sparse_weight: float = 0.0,
        envelope_weight: float = 0.0,
        # DiffToA config
        toa_step: int = 5,
        toa_push_exclusion: int = 15,
        toa_loss_clamp: float = 3.0,
        softargmax_tau: float = 10.0,
        # Sparsity config
        sparse_threshold: float = 0.05,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.wave_weight = wave_weight
        self.use_mask = use_mask

        # Amplitude weighting
        self.amplitude_weighting = amplitude_weighting
        self.amplitude_weight_gamma = amplitude_weight_gamma
        self.amplitude_weight_bias = amplitude_weight_bias

        # V7 new weights
        self.toa_weight = toa_weight
        self.cs_map_weight = cs_map_weight
        self.cs_mean_weight = cs_mean_weight
        self.sparse_weight = sparse_weight
        self.envelope_weight = envelope_weight
        self.toa_loss_clamp = toa_loss_clamp

        # Sub-losses
        self.wave_coherence = WaveCoherenceLoss(min_energy_fraction=min_energy_fraction)
        self.diff_toa = DiffToaCsLoss(
            step=toa_step,
            push_exclusion_px=toa_push_exclusion,
            softargmax_tau=softargmax_tau,
        )
        self.spatial_sparsity = SpatialSparsityLoss(threshold=sparse_threshold)
        self.temporal_envelope = TemporalEnvelopeLoss()

    def _masked_loss(self, loss_map, mask):
        if mask is not None and self.use_mask:
            return (loss_map * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_map.mean()

    def _amplitude_weight(self, target, mask_5d):
        if not self.amplitude_weighting:
            return None
        w = (target.abs() + self.amplitude_weight_bias).pow(self.amplitude_weight_gamma)
        if mask_5d is not None and self.use_mask:
            denom = (w * mask_5d).sum() / mask_5d.sum().clamp(min=1.0)
        else:
            denom = w.mean()
        return w / denom.clamp(min=1e-6)

    def _weighted_masked_loss(self, loss_map, mask, weight):
        if weight is not None:
            loss_map = loss_map * weight
        return self._masked_loss(loss_map, mask)

    @staticmethod
    def _spatial_gradient(x):
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        return dx, dy

    @staticmethod
    def _temporal_gradient(x):
        return x[:, :, 1:] - x[:, :, :-1]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        # V7 multi-task inputs
        cs_map_pred: Optional[torch.Tensor] = None,
        cs_target_scalar: Optional[torch.Tensor] = None,
        vel_scale: Optional[torch.Tensor] = None,
        dt_output: Optional[torch.Tensor] = None,
        push_col: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        pred           : (B, 1, T, H, W) velocity prediction
        target         : (B, 1, T, H, W) velocity target
        mask           : (B, 1, 1, H, W) ROI mask
        cs_map_pred    : (B, 1, H, W) physics head Cs prediction (m/s)
        cs_target_scalar : (B,) phantom ground-truth Cs (m/s)
        vel_scale      : (B,) denormalisation scale (mm/s)
        dt_output      : (B,) time between dense output frames (s)
        push_col       : (B,) push excitation column index

        Returns
        -------
        total_loss, components dict
        """
        components: Dict[str, float] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Expand mask
        if mask is not None and mask.dim() == 5 and mask.shape[2] == 1:
            mask_5d = mask.expand_as(pred)
        else:
            mask_5d = mask

        amp_w = self._amplitude_weight(target, mask_5d)

        # ===== V6 heritage losses =====

        # L1
        if self.l1_weight > 0:
            l1_map = (pred - target).abs()
            l1 = self._weighted_masked_loss(l1_map, mask_5d, amp_w)
            total = total + self.l1_weight * l1
            components["l1"] = l1.item()

        # MSE
        if self.mse_weight > 0:
            mse_map = (pred - target) ** 2
            mse = self._weighted_masked_loss(mse_map, mask_5d, amp_w)
            total = total + self.mse_weight * mse
            components["mse"] = mse.item()

        # GDL
        if self.gdl_weight > 0:
            pred_dx, pred_dy = self._spatial_gradient(pred)
            tgt_dx, tgt_dy = self._spatial_gradient(target)
            mask_dx = mask_5d[:, :, :, :, 1:] if mask_5d is not None else None
            mask_dy = mask_5d[:, :, :, 1:, :] if mask_5d is not None else None
            gdl = (
                self._masked_loss((pred_dx - tgt_dx).abs(), mask_dx)
                + self._masked_loss((pred_dy - tgt_dy).abs(), mask_dy)
            )
            total = total + self.gdl_weight * gdl
            components["gdl"] = gdl.item()

        # Temporal gradient
        if self.temporal_weight > 0 and pred.shape[2] > 1:
            pred_dt = self._temporal_gradient(pred)
            tgt_dt = self._temporal_gradient(target)
            mask_dt = mask_5d[:, :, :pred_dt.shape[2]] if mask_5d is not None else None
            temporal = self._masked_loss((pred_dt - tgt_dt).abs(), mask_dt)
            total = total + self.temporal_weight * temporal
            components["temporal"] = temporal.item()

        # Wave coherence
        if self.wave_weight > 0 and pred.shape[2] > 2:
            wave_val = self.wave_coherence(pred, mask)
            total = total + self.wave_weight * wave_val
            components["wave_coherence"] = wave_val.item()

        # ===== V7 new losses =====

        # Differentiable ToA cs loss
        if (
            self.toa_weight > 0
            and vel_scale is not None
            and dt_output is not None
            and cs_target_scalar is not None
            and push_col is not None
        ):
            # Guard: only compute if cs targets are valid (> 0)
            valid_cs = cs_target_scalar > 0
            if valid_cs.any():
                toa_loss = self.diff_toa(
                    pred[valid_cs],
                    vel_scale[valid_cs],
                    dt_output[valid_cs],
                    cs_target_scalar[valid_cs],
                    push_col[valid_cs],
                    mask[valid_cs] if mask is not None else None,
                )
                # Guard NaN/Inf before adding to total.  DiffToaCsLoss uses
                # conv1d cross-correlation which can blow up on degenerate
                # (near-zero or uniform) velocity fields early in training.
                if torch.isfinite(toa_loss):
                    toa_loss = toa_loss.clamp(max=self.toa_loss_clamp)
                    total = total + self.toa_weight * toa_loss
                    components["toa_cs"] = toa_loss.item()
                else:
                    components["toa_cs"] = 0.0

        # Cs map supervision (physics head)
        if self.cs_map_weight > 0 and cs_map_pred is not None and cs_target_scalar is not None:
            valid_cs = cs_target_scalar > 0
            if valid_cs.any():
                # Broadcast scalar cs_target to spatial map
                B_valid = valid_cs.sum().item()
                cs_tgt_map = cs_target_scalar[valid_cs].reshape(B_valid, 1, 1, 1).expand_as(
                    cs_map_pred[valid_cs]
                )
                # Masked L1
                cs_map_loss_map = (cs_map_pred[valid_cs] - cs_tgt_map).abs()
                if mask is not None:
                    mask_2d = mask[valid_cs, :, 0, :, :]  # (B, 1, H, W)
                    cs_map_loss = (cs_map_loss_map * mask_2d).sum() / mask_2d.sum().clamp(min=1.0)
                else:
                    cs_map_loss = cs_map_loss_map.mean()
                total = total + self.cs_map_weight * cs_map_loss
                components["cs_map"] = cs_map_loss.item()

        # Cs mean consistency
        if self.cs_mean_weight > 0 and cs_map_pred is not None and cs_target_scalar is not None:
            valid_cs = cs_target_scalar > 0
            if valid_cs.any():
                if mask is not None:
                    mask_2d = mask[valid_cs, :, 0, :, :]  # (B, 1, H, W)
                    cs_pred_mean = (
                        (cs_map_pred[valid_cs] * mask_2d).sum(dim=(1, 2, 3))
                        / mask_2d.sum(dim=(1, 2, 3)).clamp(min=1.0)
                    )
                else:
                    cs_pred_mean = cs_map_pred[valid_cs].mean(dim=(1, 2, 3))
                cs_mean_loss = (cs_pred_mean - cs_target_scalar[valid_cs]).abs().mean()
                total = total + self.cs_mean_weight * cs_mean_loss
                components["cs_mean"] = cs_mean_loss.item()

        # Spatial sparsity
        if self.sparse_weight > 0:
            sparse_loss = self.spatial_sparsity(pred, target, mask)
            total = total + self.sparse_weight * sparse_loss
            components["spatial_sparsity"] = sparse_loss.item()

        # Temporal envelope
        if self.envelope_weight > 0:
            env_loss = self.temporal_envelope(pred, target, mask)
            total = total + self.envelope_weight * env_loss
            components["temporal_envelope"] = env_loss.item()

        components["total"] = total.item()
        return total, components

    def set_weights(self, **kwargs) -> None:
        """Update loss component weights dynamically (for curriculum ramp)."""
        for key, val in kwargs.items():
            attr = f"{key}_weight" if not key.endswith("_weight") else key
            if hasattr(self, attr):
                setattr(self, attr, val)
                logger.debug(f"Loss weight {attr} = {val}")
