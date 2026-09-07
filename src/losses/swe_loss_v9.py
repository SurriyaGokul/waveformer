"""
SWE-Aware Loss Functions v9 — Clean Physics-Informed Composite.

Design principles vs V7/V8:
  - NO DiffToaCsLoss: ToA regression was the single largest cause of V8
    phase-2 divergence.  ToA loss value stuck at ~2.0 m/s and never
    decreased; weight ramping to 0.5 let it overwhelm the pixel loss.
  - NO amplitude weighting in L1/MSE: V2's unweighted recipe is the proven
    approach for maintaining phase coherence at the pixel level.
  - WaveCoherenceLoss (scale-free Pearson, from V5/V6): this was the loss
    that achieved 0.141 m/s Cs MAE in V6.  It regularises wave propagation
    shape without competing with pixel supervision on absolute values.
  - Scalar Cs supervision (L_cs): direct L1 between ScalarCsHead output and
    the ground-truth shear speed from metadata.
  - Consistency (L_cons): ties the scalar head to the wave-equation learnable
    cs parameter during Phase 3.

All weights are updatable via set_weights() for the 3-phase curriculum.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_loss_v5 import WaveCoherenceLoss

logger = logging.getLogger(__name__)

# Physical pixel spacing (same as swe_analysis.py)
DX_M = 1.013e-4   # lateral m / pixel
DY_M = 5.07e-5    # axial   m / pixel


# =============================================================================
# 1. Gradient Difference Loss  (spatial sharpness)
# =============================================================================

class GradientDifferenceLoss(nn.Module):
    """L1 on spatial gradient differences between pred and target."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # pred, target: (B, 1, T, H, W)
        B, _, T, H, W = pred.shape
        # Collapse B*T into batch for 2D gradient ops
        p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
        t = target.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)

        # Sobel-style finite differences
        dp_dx = p[:, :, :, 1:] - p[:, :, :, :-1]
        dp_dy = p[:, :, 1:, :] - p[:, :, :-1, :]
        dt_dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        dt_dy = t[:, :, 1:, :] - t[:, :, :-1, :]

        loss_x = (dp_dx - dt_dx).abs().mean()
        loss_y = (dp_dy - dt_dy).abs().mean()
        return (loss_x + loss_y) * 0.5


# =============================================================================
# 2. Temporal Gradient Loss  (frame-to-frame fidelity)
# =============================================================================

class TemporalGradientLoss(nn.Module):
    """L1 on first temporal differences between pred and target."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred, target: (B, 1, T, H, W)
        dp = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dt = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        return (dp - dt).abs().mean()


# =============================================================================
# 3. Composite Loss: EnhancedVelocityLossV9
# =============================================================================

class EnhancedVelocityLossV9(nn.Module):
    """
    Multi-task composite loss for WavePINN v9.

    Pixel supervision (always on):
      L1           : unweighted mean absolute error on velocity
      MSE          : unweighted mean squared error on velocity
      GDL          : spatial gradient difference (sharpness)
      Temporal     : frame-to-frame temporal gradient fidelity

    Physics terms (ramped in Phase 2 & 3):
      WaveCoherence: scale-free Pearson(d²v/dt², ∇²v).  1−r loss.
                     No learnable cs parameter — fully shape-based.
      L_cs         : L1(cs_scalar_pred, cs_target).  Direct scalar supervision.
      L_cons       : L1(cs_scalar_pred, cs_learn_param).  Head ↔ wave-param
                     consistency.  Activated in Phase 3 only.

    Parameters
    ----------
    l1_weight, mse_weight : float
        Pixel loss weights.
    gdl_weight : float
        Spatial gradient difference weight.
    temporal_weight : float
        Temporal gradient weight.
    wave_weight : float
        WaveCoherence weight (ramped from 0 in Phase 2).
    cs_weight : float
        Scalar Cs supervision weight (ramped from 0 in Phase 2).
    cons_weight : float
        Head ↔ wave-cs consistency weight (Phase 3 only).
    use_mask : bool
        Apply ROI mask to pixel losses.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.15,
        wave_weight: float = 0.0,
        cs_weight: float = 0.0,
        cons_weight: float = 0.0,
        use_mask: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.wave_weight = wave_weight
        self.cs_weight = cs_weight
        self.cons_weight = cons_weight
        self.use_mask = use_mask

        self.gdl_loss = GradientDifferenceLoss()
        self.temporal_loss = TemporalGradientLoss()
        self.wave_loss = WaveCoherenceLoss(min_energy_fraction=0.1)

    def set_weights(self, **kwargs: float) -> None:
        """Update loss weights in-place (called each epoch by curriculum).

        Config keys use short names (``wave``, ``cs``) while class attributes
        use the ``_weight`` suffix (``wave_weight``, ``cs_weight``).  Try the
        suffixed form first, fall back to the bare name so both conventions work.
        """
        for k, v in kwargs.items():
            attr = f"{k}_weight"
            if hasattr(self, attr):
                setattr(self, attr, float(v))
            elif hasattr(self, k):
                setattr(self, k, float(v))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cs_scalar_pred: torch.Tensor,
        cs_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pred           : (B, 1, T, H, W)  normalised predicted velocity [-1, 1]
        target         : (B, 1, T, H, W)  normalised ground-truth velocity
        cs_scalar_pred : (B,)  predicted shear speed from ScalarCsHead (m/s)
        cs_target      : (B,)  ground-truth shear speed per sample (m/s)
        mask           : (B, 1, 1, H, W)  optional ROI mask

        Returns
        -------
        Dict with keys: total, l1, mse, gdl, temporal, wave, cs, cons
        (all scalar tensors; zeros when weight=0)
        """
        device = pred.device
        zero = torch.zeros([], device=device)

        # --- Mask application ---
        if self.use_mask and mask is not None:
            # Expand mask to (B, 1, T, H, W)
            mask_expanded = mask.expand_as(pred)
            n = mask_expanded.sum().clamp(min=1.0)
            pred_m = pred * mask_expanded
            tgt_m  = target * mask_expanded
        else:
            mask_expanded = None
            n = pred.numel()
            pred_m = pred
            tgt_m  = target

        # --- Pixel losses ---
        l1_loss  = (pred_m - tgt_m).abs().sum() / n
        mse_loss = ((pred_m - tgt_m) ** 2).sum() / n

        # --- Spatial gradient difference (no masking — boundary effects negligible) ---
        gdl_val  = self.gdl_loss(pred, target) if self.gdl_weight > 0 else zero
        temp_val = self.temporal_loss(pred, target) if self.temporal_weight > 0 else zero

        # --- Wave coherence (scale-free physics) ---
        if self.wave_weight > 0:
            wave_val = self.wave_loss(pred, mask)
        else:
            wave_val = zero

        # --- Scalar Cs supervision ---
        if self.cs_weight > 0:
            # Only include samples where ground-truth Cs is available (> 0)
            valid = cs_target > 0
            if valid.any():
                cs_val = (cs_scalar_pred[valid] - cs_target[valid]).abs().mean()
            else:
                cs_val = zero
        else:
            cs_val = zero

        # --- Consistency (head ↔ wave): no extra learnable param in V9 ---
        # L_cons is reserved for future use when a global cs_learn parameter
        # is added.  For now it is always zero (Phase 3 coupling via cs_weight
        # + wave_weight is sufficient).
        cons_val = zero

        # --- Total ---
        total = (
            self.l1_weight       * l1_loss
            + self.mse_weight    * mse_loss
            + self.gdl_weight    * gdl_val
            + self.temporal_weight * temp_val
            + self.wave_weight   * wave_val
            + self.cs_weight     * cs_val
            + self.cons_weight   * cons_val
        )

        return {
            "total":    total,
            "l1":       l1_loss,
            "mse":      mse_loss,
            "gdl":      gdl_val,
            "temporal": temp_val,
            "wave":     wave_val,
            "cs":       cs_val,
            "cons":     cons_val,
        }
