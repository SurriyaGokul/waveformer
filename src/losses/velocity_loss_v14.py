"""
Loss for WavePINN V14 = V10 pixel loss + per-frame supervision masking
+ a temporal spectral term.

Two problems in the V10/V12/V13 recipe motivate this file.

1. FAKE DENSE TARGETS.  For any sample whose native ground truth is coarser
   than the dense output grid, ``iq_velocity_dataset_v3.py`` linearly
   interpolates the target up to 49 frames.  For 500 Hz samples that means the
   target is a linear interpolation of the same 5 frames the model receives as
   input: it carries no information above 250 Hz, and training against it
   actively rewards piecewise-linear output.  V14 supervises those samples only
   at the frames where ground truth genuinely exists (``target_frame_mask``),
   so the invented frames stop pulling the model toward straight lines.

2. NO FREQUENCY-DOMAIN TERM.  Matched-ground-truth evaluation of v12
   (scripts/evaluate_matched_gt.py) showed the model reconstructs
   high-frequency PHASE well (band-passed coherence 0.73 over 250-1500 Hz,
   against ~0.0 for linear interpolation) but carries only ~46% of the true
   high-frequency ENERGY -- the conditional-mean shrinkage an L1/L2 objective
   produces.  ``TemporalSpectralLoss`` penalises that magnitude deficit
   directly, and is applied only to samples with genuine dense ground truth.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.velocity_loss_v10 import GradientDifferenceLoss


def _masked_mean(x: torch.Tensor, frame_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of ``x`` (B,1,T,H,W) over frames selected by ``frame_mask`` (B,T)."""
    if frame_mask is None:
        return x.mean()
    w = frame_mask.to(x.dtype)[:, None, :, None, None]     # (B,1,T,1,1)
    denom = (w.expand_as(x)).sum().clamp_min(1.0)
    return (x * w).sum() / denom


class MaskedTemporalGradientLoss(nn.Module):
    """L1 on first temporal differences, restricted to adjacent frame PAIRS that
    are both genuinely supervised.

    A difference spanning an invented frame is meaningless, so for knot-only
    samples this term simply contributes nothing rather than teaching the model
    the slope of an interpolation artefact.
    """

    def forward(self, pred, target, frame_mask: Optional[torch.Tensor] = None):
        dp = pred[:, :, 1:] - pred[:, :, :-1]
        dt = target[:, :, 1:] - target[:, :, :-1]
        diff = (dp - dt).abs()
        if frame_mask is None:
            return diff.mean()
        pair = (frame_mask[:, 1:] * frame_mask[:, :-1])        # (B, T-1)
        if pair.sum() == 0:
            return diff.sum() * 0.0
        w = pair.to(diff.dtype)[:, None, :, None, None]
        return (diff * w).sum() / (w.expand_as(diff)).sum().clamp_min(1.0)


class TemporalSpectralLoss(nn.Module):
    """L1 between log temporal-magnitude spectra of prediction and target.

    Operates on the rFFT along the time axis of the (masked) velocity volume,
    with a frequency weight that ramps from 1 to ``hf_weight`` across
    [``f_lo``, ``f_hi``] so the term concentrates where the deficit actually is.

    Only meaningful for samples whose dense ground truth is real; callers must
    gate on a fully-supervised frame mask.
    """

    def __init__(self, f_lo: float = 250.0, f_hi: float = 1500.0,
                 hf_weight: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.f_lo, self.f_hi, self.hf_weight, self.eps = f_lo, f_hi, hf_weight, eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                dt_seconds: torch.Tensor) -> torch.Tensor:
        # (B,1,T,H,W) -> FFT along T
        T = pred.shape[2]
        P = torch.fft.rfft(pred.float(), dim=2)
        G = torch.fft.rfft(target.float(), dim=2)
        mag_p = torch.log(P.abs() + self.eps)
        mag_g = torch.log(G.abs() + self.eps)
        diff = (mag_p - mag_g).abs()                       # (B,1,F,H,W)

        # Per-sample frequency axis: dt differs across the batch.
        dt = dt_seconds.reshape(-1).to(pred.device).float()          # (B,)
        n_f = P.shape[2]
        k = torch.arange(n_f, device=pred.device, dtype=torch.float32)
        freqs = k[None, :] / (T * dt[:, None]).clamp_min(1e-12)      # (B,F) Hz

        w = 1.0 + (self.hf_weight - 1.0) * (
            (freqs - self.f_lo) / max(self.f_hi - self.f_lo, 1e-6)
        ).clamp(0.0, 1.0)
        w = w[:, None, :, None, None]
        return (diff * w).sum() / w.expand_as(diff).sum().clamp_min(1.0)


class VelocityLossV14(nn.Module):
    """V10 loss with per-frame supervision masking and an optional spectral term.

    With ``frame_mask=None`` and ``spectral_weight=0`` this reduces to
    ``VelocityLossV10`` exactly, so v12 recipes remain reproducible.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gdl_weight: float = 0.3,
        temporal_weight: float = 0.5,
        spectral_weight: float = 0.0,
        spectral_f_lo: float = 250.0,
        spectral_f_hi: float = 1500.0,
        spectral_hf_weight: float = 2.0,
        use_mask: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.temporal_weight = temporal_weight
        self.spectral_weight = spectral_weight
        self.use_mask = use_mask

        self.gdl_loss = GradientDifferenceLoss()
        self.temporal_loss = MaskedTemporalGradientLoss()
        self.spectral_loss = TemporalSpectralLoss(
            spectral_f_lo, spectral_f_hi, spectral_hf_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        dt_seconds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        pred, target : (B, 1, T, H, W)
        mask         : (B, 1, 1, H, W) spatial ROI mask
        frame_mask   : (B, T) 1 where the target frame is genuine ground truth,
                       0 where the dataset interpolated it.  None = all genuine.
        dt_seconds   : (B,) output frame interval, needed by the spectral term.
        """
        if self.use_mask and mask is not None:
            mask_t = mask.expand_as(pred)
            pred_m, target_m = pred * mask_t, target * mask_t
        else:
            pred_m, target_m = pred, target

        l1 = _masked_mean((pred_m - target_m).abs(), frame_mask)
        mse = _masked_mean((pred_m - target_m) ** 2, frame_mask)
        gdl = self.gdl_loss(pred_m, target_m) if frame_mask is None else \
            self._masked_gdl(pred_m, target_m, frame_mask)
        temporal = self.temporal_loss(pred_m, target_m, frame_mask)

        total = (
            self.l1_weight * l1
            + self.mse_weight * mse
            + self.gdl_weight * gdl
            + self.temporal_weight * temporal
        )

        out = {"total": total, "l1": l1, "mse": mse, "gdl": gdl,
               "temporal": temporal}

        if self.spectral_weight > 0 and dt_seconds is not None:
            # Only samples with fully genuine dense ground truth: a spectrum
            # computed over invented frames measures the interpolator, not the
            # tissue.
            if frame_mask is None:
                sel = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
            else:
                sel = frame_mask.min(dim=1).values > 0.5
            if sel.any():
                spec = self.spectral_loss(
                    pred_m[sel], target_m[sel], dt_seconds.reshape(-1)[sel])
            else:
                spec = pred.sum() * 0.0
            out["spectral"] = spec
            out["total"] = total + self.spectral_weight * spec

        return out

    @staticmethod
    def _masked_gdl(pred, target, frame_mask):
        """Spatial GDL restricted to genuinely supervised frames."""
        dp_dx = pred[..., 1:] - pred[..., :-1]
        dt_dx = target[..., 1:] - target[..., :-1]
        dp_dy = pred[..., 1:, :] - pred[..., :-1, :]
        dt_dy = target[..., 1:, :] - target[..., :-1, :]
        lx = _masked_mean((dp_dx - dt_dx).abs(), frame_mask)
        ly = _masked_mean((dp_dy - dt_dy).abs(), frame_mask)
        return 0.5 * (lx + ly)
