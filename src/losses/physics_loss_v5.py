"""
Physics-Informed Loss Functions v5 — SWE-Parameter-Free.

Key difference from v3: the wave coherence regularizer does NOT learn or
assume any tissue-specific parameters (cs, viscosity, elasticity).  It
enforces that the predicted velocity field *behaves like a propagating wave*
without specifying what wave speed that propagation should have.

This avoids turning the forward model into an inverse problem.  Tissue
characterization (cs, G', E, eta) remains a strictly post-hoc measurement
via time-of-arrival or k-space analysis on the predicted velocity maps.

Approach
--------
The scalar wave equation  d²v/dt² = cs² · ∇²v  states that temporal
curvature is proportional to spatial curvature, with the proportionality
constant being cs².  Instead of learning cs, we enforce the proportionality
via a **scale-free Pearson correlation** between d²v/dt² and ∇²v within the
ROI.  This:

1. Penalises velocity fields where temporal and spatial curvatures are
   misaligned (non-wave-like behaviour)
2. Is completely agnostic to the actual wave speed (correlation is invariant
   to the proportionality constant)
3. Requires no dt or dx/dy — the correlation is dimensionless
4. Works at any PRF without coefficient scaling
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_loss import SSIMLoss  # reuse v2 SSIM

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Wave Coherence Loss (cs-free, no SWE parameters)
# =============================================================================

class WaveCoherenceLoss(nn.Module):
    """
    Scale-free wave equation regularizer — no learnable or assumed SWE params.

    Computes Pearson correlation between d²v/dt² and the spatial Laplacian
    ∇²v = d²v/dx² + d²v/dy² over the ROI.  A perfect propagating wave
    yields correlation = +1 regardless of wave speed.

    The loss is ``1 − r`` where ``r`` is the per-sample Pearson correlation,
    so a perfectly wave-like field has loss 0.

    No learnable parameters.  No dt, dx, dy, or cs.
    """

    def __init__(self, min_energy_fraction: float = 0.1):
        """
        Parameters
        ----------
        min_energy_fraction : float
            Minimum fraction of max spatial energy a voxel must have to be
            included in the correlation.  Filters out background noise where
            both curvatures are near zero and the correlation is meaningless.
        """
        super().__init__()
        self.min_energy_fraction = min_energy_fraction

    def forward(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor  (B, 1, T, H, W)
            Predicted (normalised) velocity.
        mask : Tensor, optional  (B, 1, 1, H, W)
            ROI mask.

        Returns
        -------
        Scalar loss in [0, 2].  0 = perfect wave, 1 = uncorrelated, 2 = anti-wave.
        """
        # Need at least 3 frames for temporal second derivative
        if pred.shape[2] < 3:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # --- Temporal second derivative ---
        # d²v/dt² ≈ v[t+1] - 2v[t] + v[t-1]
        d2v_dt2 = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]

        # --- Spatial Laplacian at temporal interior ---
        v_mid = pred[:, :, 1:-1]
        d2v_dx2 = v_mid[:, :, :, :, 2:] - 2 * v_mid[:, :, :, :, 1:-1] + v_mid[:, :, :, :, :-2]
        d2v_dy2 = v_mid[:, :, :, 2:, :] - 2 * v_mid[:, :, :, 1:-1, :] + v_mid[:, :, :, :-2, :]

        # Crop to common interior  (T-2, H-2, W-2)
        T_int = d2v_dt2.shape[2]
        H_int = min(d2v_dx2.shape[3], d2v_dy2.shape[3])
        W_int = min(d2v_dx2.shape[4], d2v_dy2.shape[4])

        d2v = d2v_dt2[:, :, :T_int, 1:-1, 1:-1][:, :, :, :H_int, :W_int]
        laplacian = (
            d2v_dx2[:, :, :T_int, 1:-1, :][:, :, :, :H_int, :W_int]
            + d2v_dy2[:, :, :T_int, :, 1:-1][:, :, :, :H_int, :W_int]
        )

        # --- Build active-region mask ---
        # Combine ROI mask with energy threshold
        energy = d2v.abs() + laplacian.abs()

        if mask is not None:
            m = mask[:, :, :, 1:-1, 1:-1][:, :, :, :H_int, :W_int]
            if m.shape[2] == 1:
                m = m.expand_as(energy)
            energy = energy * m

        # Per-sample energy threshold: only correlate where there's signal
        max_energy = energy.flatten(1).amax(dim=1, keepdim=True).reshape(-1, 1, 1, 1, 1)
        active = (energy > self.min_energy_fraction * max_energy.clamp(min=1e-10)).float()

        # --- Per-sample Pearson correlation ---
        # Flatten spatial+temporal dims
        d2v_flat = (d2v * active).flatten(1)          # (B, N)
        lap_flat = (laplacian * active).flatten(1)     # (B, N)
        n_active = active.flatten(1).sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

        # Masked means
        d2v_mean = d2v_flat.sum(dim=1, keepdim=True) / n_active
        lap_mean = lap_flat.sum(dim=1, keepdim=True) / n_active

        # Centre
        d2v_c = (d2v_flat - d2v_mean) * active.flatten(1)
        lap_c = (lap_flat - lap_mean) * active.flatten(1)

        # Correlation
        numerator = (d2v_c * lap_c).sum(dim=1)
        denom_d2v = (d2v_c ** 2).sum(dim=1).clamp(min=1e-12)
        denom_lap = (lap_c ** 2).sum(dim=1).clamp(min=1e-12)
        denominator = (denom_d2v * denom_lap).sqrt()

        r = numerator / denominator  # per-sample correlation in [-1, 1]

        # Loss = 1 - r  (0 when perfectly wave-like)
        loss = (1.0 - r).mean()

        return loss


# =============================================================================
# 2. Enhanced Velocity Loss v5 (composite, cs-free)
# =============================================================================

class EnhancedVelocityLossV5(nn.Module):
    """
    Multi-component loss for velocity prediction — no SWE parameter assumptions.

    Components
    ----------
    L1, MSE          : pixel-level reconstruction
    GDL              : gradient difference (spatial sharpness)
    Temporal          : temporal gradient fidelity
    SSIM              : perceptual structure
    Wave coherence    : scale-free wave equation enforcement (no cs/dt/dx)

    The wave component uses :class:`WaveCoherenceLoss` which enforces that
    temporal and spatial curvatures are correlated (wave-like behaviour)
    without assuming any specific wave speed.

    Parameters
    ----------
    l1_weight ... wave_weight : float
        Component weights.
    use_mask : bool
        If True, supervised losses are ROI-masked.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.15,
        ssim_weight: float = 0.0,
        wave_weight: float = 0.05,
        use_mask: bool = True,
        min_energy_fraction: float = 0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.ssim_weight = ssim_weight
        self.wave_weight = wave_weight
        self.use_mask = use_mask

        self.wave_coherence = WaveCoherenceLoss(
            min_energy_fraction=min_energy_fraction,
        )
        self.ssim_loss = SSIMLoss(window_size=7, data_range=2.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _masked_loss(
        self,
        loss_map: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is not None and self.use_mask:
            return (loss_map * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_map.mean()

    @staticmethod
    def _spatial_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        return dx, dy

    @staticmethod
    def _temporal_gradient(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:] - x[:, :, :-1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        pred : Tensor  (B, 1, T, H, W)
        target : Tensor  (B, 1, T, H, W)
        mask : Tensor, optional  (B, 1, 1, H, W)

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

        # --- L1 ---
        if self.l1_weight > 0:
            l1 = self._masked_loss(torch.abs(pred - target), mask_5d)
            total = total + self.l1_weight * l1
            components["l1"] = l1.item()

        # --- MSE ---
        if self.mse_weight > 0:
            mse = self._masked_loss((pred - target) ** 2, mask_5d)
            total = total + self.mse_weight * mse
            components["mse"] = mse.item()

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

        # --- SSIM ---
        if self.ssim_weight > 0:
            ssim_val = self.ssim_loss(pred, target)
            total = total + self.ssim_weight * ssim_val
            components["ssim"] = ssim_val.item()

        # --- Wave coherence (cs-free) ---
        if self.wave_weight > 0 and pred.shape[2] > 2:
            wave_val = self.wave_coherence(pred, mask)
            total = total + self.wave_weight * wave_val
            components["wave_coherence"] = wave_val.item()

        components["total"] = total.item()
        return total, components

    def set_weights(self, **kwargs) -> None:
        """Update loss component weights dynamically (for wave ramp)."""
        for key, val in kwargs.items():
            attr = f"{key}_weight" if not key.endswith("_weight") else key
            if hasattr(self, attr):
                setattr(self, attr, val)
                logger.debug(f"Loss weight {attr} = {val}")
