"""
Pure pixel-only loss for WavePINN V10.

Design rationale:
  - NO wave coherence: On near-zero outputs (70% of frames are inactive),
    wave coherence gradient is ~256x larger than L1 gradient. This overwhelmed
    V9 between epochs 65-72, collapsing r_active from 0.436 to 0.078.
  - NO Cs loss: No ScalarCsHead in V10. Cs is estimated by ToA analysis on
    the output velocity (standard clinical pipeline, proven stable).
  - NO amplitude weighting: V2's unweighted recipe is the proven approach.

Four loss terms (all proven stable across V2/V6/V9 Phase 1):
  L_total = l1_weight     * L1(v_pred, v_gt)       [pixel reconstruction]
          + mse_weight    * MSE(v_pred, v_gt)       [large-error penalty]
          + gdl_weight    * GDL(v_pred, v_gt)       [spatial sharpness]
          + temporal_weight * Temporal(v_pred, v_gt) [frame-to-frame fidelity]

Temporal weight increased from V9's 0.15 to 0.30. Deployment SWE depends
entirely on temporal structure for ToA cross-correlation estimation.
Stronger temporal loss directly optimises the signal that matters at deployment.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Gradient Difference Loss  (spatial sharpness)
# =============================================================================

class GradientDifferenceLoss(nn.Module):
    """L1 on spatial gradient differences between pred and target."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred, target: (B, 1, T, H, W)
        B, _, T, H, W = pred.shape
        # Collapse B*T into batch for 2D gradient ops
        p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
        t = target.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)

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
        dp = pred[:, :, 1:] - pred[:, :, :-1]
        dt = target[:, :, 1:] - target[:, :, :-1]
        return (dp - dt).abs().mean()


# =============================================================================
# 3. Composite Loss: VelocityLossV10
# =============================================================================

class VelocityLossV10(nn.Module):
    """
    Pure pixel loss for WavePINN V10.

    Parameters
    ----------
    l1_weight : float
        Weight for mean absolute error.  Default 1.0.
    mse_weight : float
        Weight for mean squared error.  Default 1.0.
    gdl_weight : float
        Weight for spatial gradient difference loss.  Default 0.3.
    temporal_weight : float
        Weight for frame-to-frame temporal gradient loss.  Default 0.3.
        (Increased from V9's 0.15 — deployment SWE depends on temporal structure.)
    use_mask : bool
        Apply ROI mask to L1 and MSE.  GDL and Temporal are computed globally.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.3,
        use_mask: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.use_mask = use_mask

        self.gdl_loss = GradientDifferenceLoss()
        self.temporal_loss = TemporalGradientLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pred   : (B, 1, T, H, W)  — model velocity output in [-1, 1]
        target : (B, 1, T, H, W)  — ground truth velocity in [-1, 1]
        mask   : (B, 1, 1, H, W)  — ROI binary mask (optional)

        Returns
        -------
        Dict with keys: 'total', 'l1', 'mse', 'gdl', 'temporal'
        """
        if self.use_mask and mask is not None:
            # Expand mask across time: (B, 1, 1, H, W) -> (B, 1, T, H, W)
            mask_t = mask.expand_as(pred)
            pred_m = pred * mask_t
            target_m = target * mask_t
        else:
            pred_m = pred
            target_m = target

        l1 = F.l1_loss(pred_m, target_m)
        mse = F.mse_loss(pred_m, target_m)
        # GDL and Temporal also use masked volumes so that zero-padded ROI
        # boundaries do not create spurious high-gradient training signal.
        gdl = self.gdl_loss(pred_m, target_m)
        temporal = self.temporal_loss(pred_m, target_m)

        total = (
            self.l1_weight * l1
            + self.mse_weight * mse
            + self.gdl_weight * gdl
            + self.temporal_weight * temporal
        )

        return {
            "total": total,
            "l1": l1,
            "mse": mse,
            "gdl": gdl,
            "temporal": temporal,
        }
