"""
WavePINN v5: Targeted improvements to v2 ComplexToRealUNet.

Changes from v2
----------------
1. **TemporalSkipRefiner** at decoder skip levels 3 & 4 — learns to correct
   bilinear temporal interpolation artifacts that cause systematic wave-speed
   overestimation on stiff tissue (3FT phantom).

Everything else is identical to v2: same encoder, ConvLSTM + optical-flow
bottleneck, ComplexToRealBridge, and real-valued decoder.

Design Constraints
------------------
- No cross-attention (v3 failure mode — destroyed wave front coherence)
- No ConvGRU propagation (v4 instability — 50 % NaN runs)
- No learnable SWE parameters (cs, viscosity) — avoids turning the model
  into an inverse problem.  Tissue properties are measured post-hoc.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.velocity_unet import (
    ComplexToRealUNet,
    create_velocity_model,
    count_parameters,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Temporal Skip Refiner
# =============================================================================

class TemporalSkipRefiner(nn.Module):
    """
    Learns to correct bilinear temporal interpolation of skip connections.

    After the standard ``_interpolate_skip_temporal`` produces a bilinearly
    interpolated complex skip tensor of shape ``(B, T_dense, 2, C, H, W)``,
    this module applies a lightweight depthwise-separable 1-D convolution
    along the temporal axis to fix interpolation artifacts.

    A gated residual (``alpha`` initialised to 0) ensures the model starts
    identical to v2 and gradually learns corrections.

    Parameters
    ----------
    channels : int
        Number of *complex* channels at this skip level (e.g. 128 or 256).
        The actual Conv1d operates on ``2 * channels`` (real + imaginary
        concatenated).
    kernel_size : int
        Temporal kernel size (default 5 — covers ±2 frames).
    """

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        c2 = 2 * channels  # real + imag concatenated along channel dim
        pad = kernel_size // 2

        # Depthwise temporal conv (each channel independently)
        self.dw_conv = nn.Conv1d(c2, c2, kernel_size, padding=pad, groups=c2, bias=False)
        # Pointwise mixing across channels
        self.pw_conv = nn.Conv1d(c2, c2, 1, bias=True)

        # Gated residual — starts at 0 so output == input initially
        self.alpha = nn.Parameter(torch.zeros(1))

        # Initialise near-identity
        nn.init.dirac_(self.dw_conv.weight)
        nn.init.zeros_(self.pw_conv.weight)
        nn.init.zeros_(self.pw_conv.bias)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        skip : Tensor  (B, T_dense, 2, C, H, W)
            Bilinearly interpolated complex skip connection.

        Returns
        -------
        Tensor  (B, T_dense, 2, C, H, W)
            Refined skip connection.
        """
        B, T, two, C, H, W = skip.shape

        # Merge spatial dims → (B*H*W, 2*C, T)
        x = skip.permute(0, 4, 5, 2, 3, 1).reshape(B * H * W, two * C, T)

        # Depthwise-separable temporal conv
        refined = self.dw_conv(x)
        refined = F.gelu(refined)
        refined = self.pw_conv(refined)

        # Reshape back → (B, T, 2, C, H, W)
        refined = refined.reshape(B, H, W, two, C, T).permute(0, 5, 3, 4, 1, 2)

        # Gated residual: starts as pure identity
        return skip + self.alpha * refined


# =============================================================================
# ComplexToRealUNet V5
# =============================================================================

class ComplexToRealUNetV5(ComplexToRealUNet):
    """
    V5 extension of ComplexToRealUNet with temporal skip refinement.

    Identical to v2 except that skip connections at levels 3 and 4 pass
    through a :class:`TemporalSkipRefiner` after bilinear temporal
    interpolation and before the complex-to-real bridge.

    Parameters
    ----------
    skip_refine_levels : list[int]
        Which skip levels to refine (default [3, 4]).  Level numbering
        matches the encoder: 1 = shallowest (32 ch), 4 = deepest (256 ch).
    skip_refine_kernel : int
        Temporal kernel size for the refiners (default 5).
    **kwargs
        All other arguments are forwarded to :class:`ComplexToRealUNet`.
    """

    def __init__(
        self,
        skip_refine_levels: List[int] = [3, 4],
        skip_refine_kernel: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.skip_refine_levels = set(skip_refine_levels)

        # Create refiners for requested levels
        self._skip_refiners = nn.ModuleDict()
        for lvl in skip_refine_levels:
            ch = self.encoder_depths[lvl - 1]  # level 1→depths[0], etc.
            self._skip_refiners[str(lvl)] = TemporalSkipRefiner(
                ch, kernel_size=skip_refine_kernel,
            )

        n_refine_params = sum(
            p.numel() for p in self._skip_refiners.parameters()
        )
        logger.info(
            f"ComplexToRealUNetV5: skip refinement at levels "
            f"{sorted(skip_refine_levels)}, +{n_refine_params:,} params"
        )

    # ------------------------------------------------------------------
    def _refine_skip(self, skip: torch.Tensor, level: int) -> torch.Tensor:
        """Apply temporal refinement if this level has a refiner."""
        key = str(level)
        if key in self._skip_refiners:
            return self._skip_refiners[key](skip)
        return skip

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        high_res_inputs: Optional[torch.Tensor] = None,
        log_prf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass — identical to v2 except skip connections at the
        configured levels pass through TemporalSkipRefiner.
        """
        B, _, T_sparse, H, W = x.shape

        # Derive high-res inputs for flow estimator if not provided
        if high_res_inputs is None and self.use_motion_interpolation:
            high_res_inputs = x.permute(0, 2, 1, 3, 4).unsqueeze(3)

        # ====================== ENCODER ===================================
        x_enc, B, T = self._reshape_for_encoder(x)  # (B*T, 2, 1, H, W)

        complex_skips: List[torch.Tensor] = []

        x_enc, skip1 = self.enc1(x_enc)
        complex_skips.append(skip1)

        x_enc, skip2 = self.enc2(x_enc)
        complex_skips.append(skip2)

        x_enc, skip3 = self.enc3(x_enc)
        complex_skips.append(skip3)

        x_enc, skip4 = self.enc4(x_enc)
        complex_skips.append(skip4)

        # Reshape to (B, T_sparse, 2, C, H_b, W_b)
        _, _, C_b, H_b, W_b = x_enc.shape
        x_temporal = x_enc.reshape(B, T_sparse, 2, C_b, H_b, W_b)

        # ====================== BOTTLENECK ================================
        if self.use_motion_interpolation:
            x_dense_complex = self.bottleneck(x_temporal, high_res_inputs)
        else:
            x_dense_complex = self.bottleneck(x_temporal)

        T_dense = x_dense_complex.shape[1]

        # ====================== BRIDGE: bottleneck complex → real =========
        x_real = self.bottleneck_bridge(x_dense_complex)

        # Expand log_prf for frame-level FiLM application
        log_prf_exp = None
        if self.prf_conditioner is not None and log_prf is not None:
            log_prf_exp = log_prf.unsqueeze(1).expand(B, T_dense).reshape(B * T_dense)

        # ====================== DECODER ===================================

        # --- Level 4 ---
        skip4_c = complex_skips.pop()
        skip4_c = self._reshape_skip_for_temporal(skip4_c, B, T_sparse)
        skip4_c = self._interpolate_skip_temporal(skip4_c, T_dense)
        skip4_c = self._refine_skip(skip4_c, level=4)  # v5: temporal refinement
        skip4_r = self.skip_bridge_l4(skip4_c)

        x_dec = self._flatten_for_decoder(x_real, B, T_dense)
        skip4_flat = self._flatten_for_decoder(skip4_r, B, T_dense)
        x_dec = self.dec4(x_dec, skip4_flat)

        if self.prf_conditioner is not None and log_prf_exp is not None:
            g, b = self.prf_conditioner(log_prf_exp, level_idx=0)
            x_dec = x_dec * (1.0 + g) + b

        # --- Level 3 ---
        skip3_c = complex_skips.pop()
        skip3_c = self._reshape_skip_for_temporal(skip3_c, B, T_sparse)
        skip3_c = self._interpolate_skip_temporal(skip3_c, T_dense)
        skip3_c = self._refine_skip(skip3_c, level=3)  # v5: temporal refinement
        skip3_r = self.skip_bridge_l3(skip3_c)
        skip3_flat = self._flatten_for_decoder(skip3_r, B, T_dense)
        x_dec = self.dec3(x_dec, skip3_flat)

        if self.prf_conditioner is not None and log_prf_exp is not None:
            g, b = self.prf_conditioner(log_prf_exp, level_idx=1)
            x_dec = x_dec * (1.0 + g) + b

        # --- Level 2 (no refinement — too large spatially) ---
        skip2_c = complex_skips.pop()
        skip2_c = self._reshape_skip_for_temporal(skip2_c, B, T_sparse)
        skip2_c = self._interpolate_skip_temporal(skip2_c, T_dense)
        skip2_r = self.skip_bridge_l2(skip2_c)
        skip2_flat = self._flatten_for_decoder(skip2_r, B, T_dense)
        x_dec = self.dec2(x_dec, skip2_flat)

        if self.prf_conditioner is not None and log_prf_exp is not None:
            g, b = self.prf_conditioner(log_prf_exp, level_idx=2)
            x_dec = x_dec * (1.0 + g) + b

        # --- Level 1 (no refinement) ---
        skip1_c = complex_skips.pop()
        skip1_c = self._reshape_skip_for_temporal(skip1_c, B, T_sparse)
        skip1_c = self._interpolate_skip_temporal(skip1_c, T_dense)
        skip1_r = self.skip_bridge_l1(skip1_c)
        skip1_flat = self._flatten_for_decoder(skip1_r, B, T_dense)
        x_dec = self.dec1(x_dec, skip1_flat)

        if self.prf_conditioner is not None and log_prf_exp is not None:
            g, b = self.prf_conditioner(log_prf_exp, level_idx=3)
            x_dec = x_dec * (1.0 + g) + b

        # ====================== OUTPUT HEAD ===============================
        velocity = self.output_activation(self.output_conv(x_dec))
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        return velocity


# =============================================================================
# Factory
# =============================================================================

def create_wavepinn_v5(
    config: Optional[Dict] = None,
    **kwargs,
) -> ComplexToRealUNetV5:
    """
    Create a configured :class:`ComplexToRealUNetV5`.

    Accepts the same parameters as ``create_velocity_model`` plus:

    - ``skip_refine_levels`` (default [3, 4])
    - ``skip_refine_kernel`` (default 5)
    """
    defaults = dict(
        in_channels=1,
        base_channels=32,
        encoder_depths=[32, 64, 128, 256],
        upsampling_factor=12,
        use_motion_interpolation=True,
        complex_to_real_mode='concat',
        use_naive_bn=True,
        flow_hidden_channels=32,
        flow_scale=0.1,
        use_prf_conditioning=False,
        prf_hidden_dim=64,
        skip_refine_levels=[3, 4],
        skip_refine_kernel=5,
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)

    # Separate v5-specific args from base-class args
    v5_keys = {"skip_refine_levels", "skip_refine_kernel"}
    v5_args = {k: defaults.pop(k) for k in v5_keys if k in defaults}

    return ComplexToRealUNetV5(**v5_args, **defaults)
