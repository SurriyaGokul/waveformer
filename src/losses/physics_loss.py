"""
Physics-Informed Loss Functions for Shear Wave Velocity Estimation.

This module provides loss components that encode prior knowledge about
shear-wave propagation physics:

1. **WaveEquationRegularizer** — soft penalty enforcing the 2-D scalar
   wave equation  ∂²v/∂t² = β (∂²v/∂x² + ∂²v/∂y²)  with a learnable
   coefficient β ∝ c_s².  This pushes predictions toward physically
   plausible propagating wavefronts.

2. **SSIMLoss** — Structural Similarity (SSIM) computed per-frame between
   prediction and target.  SSIM captures luminance / contrast / structure
   together and is far more perceptually aligned than pixel-wise L1/MSE.

3. **EnhancedVelocityLoss** — Drop-in replacement for CombinedVelocityLoss
   that adds the physics and SSIM components on top of the existing
   L1 + MSE + GDL + temporal loss.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Wave Equation Regularizer
# ============================================================================

class WaveEquationRegularizer(nn.Module):
    """
    Soft wave-equation constraint on the predicted velocity field.

    For shear waves in tissue the particle velocity v satisfies
        ∂²v/∂t²  =  c_s²  (∂²v/∂x² + ∂²v/∂y²)

    We parameterise the squared speed c_s² = exp(log_beta) so it stays
    positive and is learnable.  The loss is the squared residual of
    this PDE:
        L_wave = mean[ (∂²v/∂t² - β ∇²v)² ]

    where β ≡ exp(log_beta) is trainable and initialised to 1.0.

    Input:
        pred: (B, 1, T, H, W) — predicted normalised velocity
    """

    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        # Learnable coefficient ≈ c_s² in pixel²/frame² units
        self.log_beta = nn.Parameter(torch.tensor(math.log(init_beta)))

    def forward(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute wave-equation residual loss.

        Args:
            pred: (B, 1, T, H, W) predicted velocity field
            mask: (B, 1, 1, H, W) optional ROI mask

        Returns:
            Scalar loss value.
        """
        beta = torch.exp(self.log_beta)

        # --- temporal second derivative: d²v/dt² ---
        # pred[:,:, t-1] - 2*pred[:,:, t] + pred[:,:, t+1]
        d2v_dt2 = pred[:, :, 2:, :, :] - 2 * pred[:, :, 1:-1, :, :] + pred[:, :, :-2, :, :]

        # --- spatial Laplacian at the same temporal slice ---
        v_mid = pred[:, :, 1:-1, :, :]  # (B, 1, T-2, H, W)
        lap_x = v_mid[:, :, :, :, 2:] - 2 * v_mid[:, :, :, :, 1:-1] + v_mid[:, :, :, :, :-2]
        lap_y = v_mid[:, :, :, 2:, :] - 2 * v_mid[:, :, :, 1:-1, :] + v_mid[:, :, :, :-2, :]

        # Crop to common interior
        T2 = d2v_dt2.shape[2]
        min_H = min(lap_x.shape[3], lap_y.shape[3])
        min_W = min(lap_x.shape[4], lap_y.shape[4])
        d2v = d2v_dt2[:, :, :, 1:-1, 1:-1][:, :, :, :min_H, :min_W]
        laplacian = (
            lap_x[:, :, :, 1:-1, :][:, :, :, :min_H, :min_W]
            + lap_y[:, :, :, :, 1:-1][:, :, :, :min_H, :min_W]
        )

        residual = d2v - beta * laplacian  # (B, 1, T-2, H', W')

        if mask is not None:
            # Crop mask to match
            m = mask[:, :, :, 1:-1, 1:-1][:, :, :, :min_H, :min_W] if mask.dim() == 5 else mask
            # Broadcast over time
            if m.dim() == 5 and m.shape[2] == 1:
                m = m.expand_as(residual)
            residual = residual * m
            denom = m.sum().clamp(min=1.0)
            return (residual ** 2).sum() / denom

        return (residual ** 2).mean()


# ============================================================================
# SSIM Loss
# ============================================================================

def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1-D Gaussian kernel (normalised to sum=1)."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _create_ssim_window(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
    """Create a 2-D Gaussian window for SSIM, shape (channel, 1, ws, ws)."""
    g1d = _gaussian_window(window_size, 1.5, device)
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # (ws, ws)
    window = g2d.unsqueeze(0).unsqueeze(0)       # (1, 1, ws, ws)
    return window.expand(channel, 1, -1, -1).contiguous()


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss computed per temporal frame.

    SSIM ∈ [0, 1] where 1 = identical.
    Loss = 1 - mean(SSIM) so that 0 = perfect.

    Applied to each (H, W) slice of the temporal sequence independently.

    Input shapes:
        pred, target: (B, 1, T, H, W) — normalised velocity
    """

    def __init__(self, window_size: int = 7, data_range: float = 2.0):
        """
        Args:
            window_size: Size of the Gaussian averaging window.
            data_range: Dynamic range of input data (default 2.0 for [-1, 1]).
        """
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self._window: Optional[torch.Tensor] = None

    def _get_window(self, channel: int, device: torch.device) -> torch.Tensor:
        if self._window is None or self._window.device != device:
            self._window = _create_ssim_window(self.window_size, channel, device)
        return self._window

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute 1 - SSIM averaged over batch and time.

        Args:
            pred:   (B, 1, T, H, W)
            target: (B, 1, T, H, W)
            mask:   ignored (SSIM is inherently local)

        Returns:
            Scalar ∈ [0, 2] (typically [0, 1]).
        """
        B, C, T, H, W = pred.shape
        # Reshape to (B*T, C, H, W) for 2-D SSIM
        pred_flat = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        tgt_flat = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        window = self._get_window(C, pred.device)
        pad = self.window_size // 2

        mu_x = F.conv2d(pred_flat, window, padding=pad, groups=C)
        mu_y = F.conv2d(tgt_flat, window, padding=pad, groups=C)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(pred_flat ** 2, window, padding=pad, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(tgt_flat ** 2, window, padding=pad, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(pred_flat * tgt_flat, window, padding=pad, groups=C) - mu_xy

        L = self.data_range
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )

        return 1.0 - ssim_map.mean()


# ============================================================================
# Enhanced Velocity Loss
# ============================================================================

class EnhancedVelocityLoss(nn.Module):
    """
    Multi-component loss for velocity prediction with physics regularisation.

    Combines the original supervised components with physics-informed
    and perceptual terms:

    Supervised:
        L1   |pred - tgt|
        MSE  (pred - tgt)²
        GDL  |∇pred - ∇tgt|       (spatial gradient difference)
        Temporal |∂pred/∂t - ∂tgt/∂t|

    Physics-informed:
        Wave  (∂²v/∂t² - β ∇²v)²  (soft PDE constraint, learnable β)

    Perceptual:
        SSIM  1 - SSIM(pred, tgt)

    All supervised components optionally weighted by a tissue ROI mask.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.5,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.2,
        ssim_weight: float = 0.3,
        wave_weight: float = 0.05,
        use_mask: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.ssim_weight = ssim_weight
        self.wave_weight = wave_weight
        self.use_mask = use_mask

        # Physics regulariser (learnable wave-speed²)
        self.wave_reg = WaveEquationRegularizer(init_beta=1.0)

        # SSIM loss
        self.ssim_loss = SSIMLoss(window_size=7, data_range=2.0)

    # ------------------------------------------------------------------
    # helpers (identical semantics to CombinedVelocityLoss)
    # ------------------------------------------------------------------

    def _masked_loss(
        self, loss_map: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is not None and self.use_mask:
            while mask.dim() < loss_map.dim():
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(loss_map)
            loss_map = loss_map * mask
            denom = mask.sum().clamp(min=1.0)
            return loss_map.sum() / denom
        return loss_map.mean()

    def _spatial_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    def _temporal_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, 1:, :, :] - x[:, :, :-1, :, :]

    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred:   (B, 1, T, H, W) predicted velocity
            target: (B, 1, T, H, W) target velocity
            mask:   (B, 1, 1, H, W) ROI mask

        Returns:
            (total_loss, component_dict)
        """
        components: Dict[str, float] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # --- L1 ---
        if self.l1_weight > 0:
            l1 = self._masked_loss(torch.abs(pred - target), mask)
            total = total + self.l1_weight * l1
            components['l1'] = l1.item()

        # --- MSE ---
        if self.mse_weight > 0:
            mse = self._masked_loss((pred - target) ** 2, mask)
            total = total + self.mse_weight * mse
            components['mse'] = mse.item()

        # --- GDL (spatial gradient difference) ---
        if self.gdl_weight > 0:
            pred_dx, pred_dy = self._spatial_gradient(pred)
            tgt_dx, tgt_dy = self._spatial_gradient(target)
            mask_dx = mask[..., :, 1:] if mask is not None else None
            mask_dy = mask[..., 1:, :] if mask is not None else None
            gdl = (
                self._masked_loss(torch.abs(pred_dx - tgt_dx), mask_dx)
                + self._masked_loss(torch.abs(pred_dy - tgt_dy), mask_dy)
            ) / 2.0
            total = total + self.gdl_weight * gdl
            components['gdl'] = gdl.item()

        # --- Temporal consistency ---
        if self.temporal_weight > 0 and pred.shape[2] > 1:
            pred_dt = self._temporal_gradient(pred)
            tgt_dt = self._temporal_gradient(target)
            temp = self._masked_loss(torch.abs(pred_dt - tgt_dt), mask)
            total = total + self.temporal_weight * temp
            components['temporal'] = temp.item()

        # --- SSIM ---
        if self.ssim_weight > 0:
            ssim_val = self.ssim_loss(pred, target)
            total = total + self.ssim_weight * ssim_val
            components['ssim'] = ssim_val.item()

        # --- Wave-equation regulariser ---
        if self.wave_weight > 0 and pred.shape[2] > 2:
            wave_val = self.wave_reg(pred, mask)
            total = total + self.wave_weight * wave_val
            components['wave'] = wave_val.item()
            components['wave_beta'] = torch.exp(self.wave_reg.log_beta).item()

        components['total'] = total.item()
        return total, components
