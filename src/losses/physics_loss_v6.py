"""
Physics-Informed Loss Functions v6 — Parameter-Free Wave Coherence.

Reuses WaveCoherenceLoss from v5 (scale-free Pearson correlation between
d²v/dt² and ∇²v).  No learnable tissue parameters (cs, viscosity, etc.).

Drops SSIM from the composite loss — SSIM weights > 0.3 caused gradient
competition in v4 experiments without meaningful quality improvement.

Components: L1, MSE, GDL, Temporal gradient, Wave coherence.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.losses.physics_loss_v5 import WaveCoherenceLoss

logger = logging.getLogger(__name__)


class EnhancedVelocityLossV6(nn.Module):
    """
    Multi-component loss for velocity prediction — no SWE parameter assumptions.

    Components
    ----------
    L1              : pixel-level absolute error
    MSE             : pixel-level squared error
    GDL             : gradient difference (spatial sharpness)
    Temporal        : temporal gradient fidelity
    Wave coherence  : scale-free wave equation enforcement (no cs/dt/dx)

    Parameters
    ----------
    l1_weight ... wave_weight : float
        Component weights.
    use_mask : bool
        If True, supervised losses are ROI-masked.
    min_energy_fraction : float
        Minimum energy fraction for wave coherence active region.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.15,
        wave_weight: float = 0.0,
        use_mask: bool = True,
        min_energy_fraction: float = 0.1,
        amplitude_weighting: bool = False,
        amplitude_weight_gamma: float = 1.0,
        amplitude_weight_bias: float = 0.05,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.wave_weight = wave_weight
        self.use_mask = use_mask

        # Amplitude weighting (fix for plateau where model collapsed to zero
        # because background pixels dominated the L1/MSE average)
        self.amplitude_weighting = amplitude_weighting
        self.amplitude_weight_gamma = amplitude_weight_gamma
        self.amplitude_weight_bias = amplitude_weight_bias

        self.wave_coherence = WaveCoherenceLoss(
            min_energy_fraction=min_energy_fraction,
        )

    def _masked_loss(
        self,
        loss_map: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is not None and self.use_mask:
            return (loss_map * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_map.mean()

    def _amplitude_weight(
        self,
        target: torch.Tensor,
        mask_5d: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Build per-pixel weight = (|target| + bias)^gamma, normalised so the
        masked-mean weight equals 1.0. The normalisation keeps the optimised
        loss in the same numerical range as the unweighted version.

        Returns None if amplitude_weighting is disabled.
        """
        if not self.amplitude_weighting:
            return None
        w = (target.abs() + self.amplitude_weight_bias).pow(self.amplitude_weight_gamma)
        if mask_5d is not None and self.use_mask:
            denom = (w * mask_5d).sum() / mask_5d.sum().clamp(min=1.0)
        else:
            denom = w.mean()
        return w / denom.clamp(min=1e-6)

    def _weighted_masked_loss(
        self,
        loss_map: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean of loss_map * weight inside mask. weight already normalised."""
        if weight is not None:
            loss_map = loss_map * weight
        return self._masked_loss(loss_map, mask)

    @staticmethod
    def _spatial_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        return dx, dy

    @staticmethod
    def _temporal_gradient(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:] - x[:, :, :-1]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.

        Parameters
        ----------
        pred : Tensor (B, 1, T, H, W)
        target : Tensor (B, 1, T, H, W)
        mask : Tensor, optional (B, 1, 1, H, W)

        Returns
        -------
        total_loss : Tensor (scalar)
        components : dict[str, float]
        """
        components: Dict[str, float] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Expand mask for broadcasting
        if mask is not None and mask.dim() == 5 and mask.shape[2] == 1:
            mask_5d = mask.expand_as(pred)
        else:
            mask_5d = mask

        # Build amplitude weight once (used for L1 + MSE only).
        amp_w = self._amplitude_weight(target, mask_5d)

        # --- L1 ---
        if self.l1_weight > 0:
            l1_map = torch.abs(pred - target)
            l1 = self._weighted_masked_loss(l1_map, mask_5d, amp_w)
            total = total + self.l1_weight * l1
            components["l1"] = l1.item()
            if amp_w is not None:
                # Unweighted L1 for comparability with old runs / val curves
                components["l1_unweighted"] = self._masked_loss(l1_map, mask_5d).item()

        # --- MSE ---
        if self.mse_weight > 0:
            mse_map = (pred - target) ** 2
            mse = self._weighted_masked_loss(mse_map, mask_5d, amp_w)
            total = total + self.mse_weight * mse
            components["mse"] = mse.item()
            if amp_w is not None:
                components["mse_unweighted"] = self._masked_loss(mse_map, mask_5d).item()

        # --- GDL ---
        if self.gdl_weight > 0:
            pred_dx, pred_dy = self._spatial_gradient(pred)
            tgt_dx, tgt_dy = self._spatial_gradient(target)
            if mask_5d is not None:
                mask_dx = mask_5d[:, :, :, :, 1:]
                mask_dy = mask_5d[:, :, :, 1:, :]
            else:
                mask_dx = mask_dy = None
            gdl = (
                self._masked_loss(torch.abs(pred_dx - tgt_dx), mask_dx)
                + self._masked_loss(torch.abs(pred_dy - tgt_dy), mask_dy)
            )
            total = total + self.gdl_weight * gdl
            components["gdl"] = gdl.item()

        # --- Temporal gradient ---
        if self.temporal_weight > 0 and pred.shape[2] > 1:
            pred_dt = self._temporal_gradient(pred)
            tgt_dt = self._temporal_gradient(target)
            mask_dt = mask_5d[:, :, :pred_dt.shape[2]] if mask_5d is not None else None
            temporal = self._masked_loss(torch.abs(pred_dt - tgt_dt), mask_dt)
            total = total + self.temporal_weight * temporal
            components["temporal"] = temporal.item()

        # --- Wave coherence (cs-free) ---
        if self.wave_weight > 0 and pred.shape[2] > 2:
            wave_val = self.wave_coherence(pred, mask)
            total = total + self.wave_weight * wave_val
            components["wave_coherence"] = wave_val.item()

        components["total"] = total.item()
        return total, components

    def set_weights(self, **kwargs) -> None:
        """Update loss component weights dynamically (for curriculum ramp)."""
        for key, val in kwargs.items():
            attr = f"{key}_weight" if not key.endswith("_weight") else key
            if hasattr(self, attr):
                setattr(self, attr, val)
                logger.debug(f"Loss weight {attr} = {val}")
