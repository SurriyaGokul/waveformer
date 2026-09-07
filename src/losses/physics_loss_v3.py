"""
Physics-Informed Loss Functions v3 for PRF-Agnostic SWE.

Key improvement over v2: dt-aware wave equation regularizer that uses physical
units (seconds, meters) for PRF-invariant cs learning.

CRITICAL: The learnable cs is a REGULARIZER only, NOT a per-sample tissue
measurement. Tissue-specific shear wave speed must be computed post-hoc via
estimate_cs_toa() on predicted velocity maps.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.physics_loss import SSIMLoss  # reuse v2 SSIM

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Physics-Aware Wave Equation Regularizer
# =============================================================================

class PhysicsAwareWaveRegularizer(nn.Module):
    """
    Wave equation regularizer using dt-dependent effective coefficient.

    The scalar wave equation is:

        d²v/dt² = cs² · (d²v/dx² + d²v/dy²)

    In pixel/frame finite-difference form this becomes:

        Δt²v = β_x · Δx²v + β_y · Δy²v

    where:
        β_x = cs² · dt² / dx²   (auto-scales with PRF)
        β_y = cs² · dt² / dy²

    All finite differences are in pixel/frame units (O(1) values) — the
    dt-dependent coefficients keep the residual numerically stable.

    Parameters
    ----------
    init_cs : float
        Initial shear wave speed in m/s.
    dx_m : float
        Lateral pixel spacing in meters.
    dy_m : float
        Axial pixel spacing in meters.

    Notes
    -----
    cs is a **global regularizer**, NOT a per-sample tissue measurement.
    The learned cs converges to a compromise across phantom types.
    Tissue-specific cs must be computed post-hoc with ``estimate_cs_toa()``.
    """

    def __init__(
        self,
        init_cs: float = 3.0,
        dx_m: float = 1.013e-4,
        dy_m: float = 5.07e-5,
    ):
        super().__init__()
        # Learnable cs in m/s (log-parameterised for positivity)
        self.log_cs = nn.Parameter(torch.tensor(math.log(init_cs)))
        self.dx_m = dx_m
        self.dy_m = dy_m

    @property
    def cs_value(self) -> float:
        """Current shear wave speed in m/s (for logging only)."""
        return math.exp(self.log_cs.item())

    def forward(
        self,
        pred: torch.Tensor,
        dt_seconds: Union[float, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked MSE of the wave equation residual.

        Parameters
        ----------
        pred : Tensor  (B, 1, T, H, W)
            Predicted (normalised) velocity.
        dt_seconds : float or Tensor
            Time between **output** frames in seconds.
        mask : Tensor, optional  (B, 1, 1, H, W)
            ROI mask.

        Returns
        -------
        Scalar loss.
        """
        cs = torch.exp(self.log_cs)  # m/s
        cs_sq = cs ** 2

        # Resolve dt
        if isinstance(dt_seconds, torch.Tensor):
            dt = dt_seconds.mean().item()
        else:
            dt = float(dt_seconds)

        dt = max(dt, 1e-12)  # guard

        # Effective coefficients (keeps residual O(1))
        beta_x = cs_sq * (dt ** 2) / (self.dx_m ** 2)
        beta_y = cs_sq * (dt ** 2) / (self.dy_m ** 2)

        # --- Temporal second derivative (pixel/frame units) ---
        # d²v/dt² ≈ v[t+1] - 2v[t] + v[t-1]  (O(1))
        d2v_dt2 = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]

        # --- Spatial Laplacian (pixel/frame units) ---
        v_mid = pred[:, :, 1:-1]  # temporal interior
        d2v_dx2 = v_mid[:, :, :, :, 2:] - 2 * v_mid[:, :, :, :, 1:-1] + v_mid[:, :, :, :, :-2]
        d2v_dy2 = v_mid[:, :, :, 2:, :] - 2 * v_mid[:, :, :, 1:-1, :] + v_mid[:, :, :, :-2, :]

        # Crop to common interior
        T_int = d2v_dt2.shape[2]
        H_int = min(d2v_dx2.shape[3], d2v_dy2.shape[3])
        W_int = min(d2v_dx2.shape[4], d2v_dy2.shape[4])

        d2v = d2v_dt2[:, :, :T_int, 1:-1, 1:-1][:, :, :, :H_int, :W_int]
        laplacian = (
            beta_x * d2v_dx2[:, :, :T_int, 1:-1, :][:, :, :, :H_int, :W_int]
            + beta_y * d2v_dy2[:, :, :T_int, :, 1:-1][:, :, :, :H_int, :W_int]
        )

        residual = d2v - laplacian

        # Clamp residual to prevent gradient explosion in early training
        residual = residual.clamp(-10.0, 10.0)

        if mask is not None:
            # Crop mask to interior
            m = mask[:, :, :, 1:-1, 1:-1][:, :, :, :H_int, :W_int]
            if m.shape[2] == 1:
                m = m.expand_as(residual)
            residual = residual * m
            return (residual ** 2).sum() / m.sum().clamp(min=1.0)

        return (residual ** 2).mean()


# =============================================================================
# 2. Enhanced Velocity Loss v3 (composite, dt-aware)
# =============================================================================

class EnhancedVelocityLossV3(nn.Module):
    """
    Multi-component loss for velocity prediction.

    Components
    ----------
    L1, MSE          : pixel-level reconstruction
    GDL              : gradient difference (spatial sharpness)
    Temporal          : temporal gradient fidelity
    SSIM              : perceptual structure
    Wave (v3)         : dt-aware wave equation regularizer

    The wave component passes ``dt_seconds`` to
    :class:`PhysicsAwareWaveRegularizer` so that the effective PDE
    coefficient automatically scales with PRF.

    Parameters
    ----------
    l1_weight ... wave_weight : float
        Component weights.
    use_mask : bool
        If True, all losses are ROI-masked.
    init_cs : float
        Initial cs for the wave regularizer (m/s).
    dx_m, dy_m : float
        Pixel spacing in meters.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.5,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.2,
        ssim_weight: float = 0.3,
        wave_weight: float = 0.05,
        fg_l1_weight: float = 0.0,
        use_mask: bool = True,
        init_cs: float = 3.0,
        dx_m: float = 1.013e-4,
        dy_m: float = 5.07e-5,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.ssim_weight = ssim_weight
        self.wave_weight = wave_weight
        self.fg_l1_weight = fg_l1_weight
        self.use_mask = use_mask

        self.wave_reg = PhysicsAwareWaveRegularizer(init_cs, dx_m, dy_m)
        self.ssim_loss = SSIMLoss(window_size=7, data_range=2.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _masked_loss(
        self,
        loss_map: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Masked mean or plain mean."""
        if mask is not None and self.use_mask:
            return (loss_map * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_map.mean()

    @staticmethod
    def _spatial_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spatial gradients along H and W."""
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        return dx, dy

    @staticmethod
    def _temporal_gradient(x: torch.Tensor) -> torch.Tensor:
        """First-order temporal gradient."""
        return x[:, :, 1:] - x[:, :, :-1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dt_seconds: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.

        Parameters
        ----------
        pred : Tensor  (B, 1, T, H, W)
        target : Tensor  (B, 1, T, H, W)
        mask : Tensor, optional  (B, 1, 1, H, W)
        dt_seconds : float or Tensor, optional
            Time between output frames (seconds).  Defaults to 1/6000.

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

        # --- Foreground-weighted L1 (upweights wave-active pixels) ---
        if self.fg_l1_weight > 0:
            abs_target = target.abs()
            fg_weight = 1.0 + 10.0 * (abs_target / (abs_target.amax(dim=(2, 3, 4), keepdim=True) + 1e-8))
            if mask_5d is not None:
                fg_weight = fg_weight * mask_5d
            fg_l1_map = torch.abs(pred - target) * fg_weight
            fg_l1 = fg_l1_map.sum() / fg_weight.sum().clamp(min=1.0)
            total = total + self.fg_l1_weight * fg_l1
            components["fg_l1"] = fg_l1.item()

        # --- GDL (Gradient Difference Loss) ---
        if self.gdl_weight > 0:
            pred_dx, pred_dy = self._spatial_gradient(pred)
            tgt_dx, tgt_dy = self._spatial_gradient(target)
            # Crop mask to match spatial gradient dimensions
            if mask_5d is not None:
                mask_dx = mask_5d[:, :, :, :, 1:]   # W-1
                mask_dy = mask_5d[:, :, :, 1:, :]   # H-1
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
            # Mask temporal: T-1 frames, mask broadcasts from T=1
            mask_dt = mask_5d[:, :, :pred_dt.shape[2]] if mask_5d is not None else None
            temporal = self._masked_loss(torch.abs(pred_dt - tgt_dt), mask_dt)
            total = total + self.temporal_weight * temporal
            components["temporal"] = temporal.item()

        # --- SSIM ---
        if self.ssim_weight > 0:
            # SSIMLoss expects 5D (B, 1, T, H, W) and flattens internally
            ssim_val = self.ssim_loss(pred, target)
            total = total + self.ssim_weight * ssim_val
            components["ssim"] = ssim_val.item()

        # --- Wave equation (dt-aware) ---
        if self.wave_weight > 0 and pred.shape[2] > 2:
            if dt_seconds is None:
                dt_seconds = 1.0 / 6000.0  # default 6 kHz output
            wave_val = self.wave_reg(pred, dt_seconds, mask)
            total = total + self.wave_weight * wave_val
            components["wave"] = wave_val.item()
            components["wave_cs"] = self.wave_reg.cs_value

        components["total"] = total.item()
        return total, components

    def set_weights(self, **kwargs) -> None:
        """Update loss component weights dynamically (for curriculum)."""
        for key, val in kwargs.items():
            attr = f"{key}_weight" if not key.endswith("_weight") else key
            if hasattr(self, attr):
                setattr(self, attr, val)
                logger.debug(f"Loss weight {attr} = {val}")
