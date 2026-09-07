"""
WavePINN v7: Multi-Task SWE-Optimised Temporal Super-Resolution.

Extends WavePINNv6 with a physics head that predicts a spatial shear-wave
speed (Cs) map directly from bottleneck features.  Two outputs:

  velocity : (B, 1, T_dense, H, W)  — dense velocity field in [-1, 1]
  cs_map   : (B, 1, H, W)           — shear wave speed map in [cs_min, cs_max] m/s

The physics head taps the dense bottleneck features *before* the
complex→real bridge.  This is deliberate: Cs is a spatially-smooth, time-
invariant tissue property that benefits from the global receptive field of
the spatiotemporal attention, not the per-frame decoder features.

Architecture changes vs V6:
  - PhysicsHead: learnable time-collapse → complex→real concat → small CNN
    → bilinear upsample → sigmoid → [cs_min, cs_max]
  - forward() returns (velocity, cs_map) tuple instead of velocity alone
  - Everything else (encoder, bottleneck, skips, decoder) is UNCHANGED.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.wavepinn_v6 import WavePINNv6, create_wavepinn_v6

logger = logging.getLogger(__name__)


# =============================================================================
# Physics Head: Cs spatial map from bottleneck features
# =============================================================================

class PhysicsHead(nn.Module):
    """
    Predicts a spatial shear-wave speed map from dense bottleneck features.

    Pipeline:
        1. Learnable temporal collapse: softmax-weighted sum over T_dense
        2. Complex→real by concatenation (2*C channels)
        3. Conv(2*C → mid_ch, 3x3) + GELU
        4. Bilinear upsample to full (H, W)
        5. Conv(mid_ch → mid_ch//2, 3x3) + GELU + Conv(mid_ch//2 → 1, 1x1)
        6. sigmoid scaled to [cs_min, cs_max]

    Parameters
    ----------
    bottleneck_channels : int
        Number of complex channels at bottleneck (256 for V6 default).
    mid_channels : int
        Intermediate feature channels after temporal collapse.
    cs_min : float
        Minimum output Cs (m/s).  Default 0.3 covers slowest tissue.
    cs_max : float
        Maximum output Cs (m/s).  Default 8.0 covers stiffest phantom.
    max_T_dense : int
        Maximum T_dense the model will see (for time-collapse weight init).
    """

    def __init__(
        self,
        bottleneck_channels: int = 256,
        mid_channels: int = 128,
        cs_min: float = 0.3,
        cs_max: float = 8.0,
        max_T_dense: int = 64,
    ):
        super().__init__()
        self.cs_min = cs_min
        self.cs_max = cs_max

        # Learnable time-collapse weights (max_T_dense entries, slice at runtime)
        self.time_logits = nn.Parameter(torch.zeros(max_T_dense))

        in_ch = 2 * bottleneck_channels  # after complex concat

        self.conv1 = nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

        self.conv_out = nn.Conv2d(mid_channels // 2, 1, kernel_size=1)

        # Initialise output conv so initial sigmoid output ≈ 0.5
        # → initial Cs ≈ (cs_min + cs_max) / 2 ≈ 4.15 m/s.
        # NOTE: Avoid exact-zero weights (which make the head output perfectly
        # constant and block gradients to earlier head layers).
        nn.init.zeros_(self.conv_out.bias)
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        x_dense_complex: torch.Tensor,
        target_H: int,
        target_W: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_dense_complex : (B, T_dense, 2, C, H_b, W_b)
            Dense bottleneck features (complex-valued).
        target_H, target_W : int
            Full spatial resolution for output.

        Returns
        -------
        cs_map : (B, 1, H, W) in [cs_min, cs_max] m/s
        """
        B, T_dense, _, C, H_b, W_b = x_dense_complex.shape

        # 1. Learnable temporal collapse
        w = torch.softmax(self.time_logits[:T_dense], dim=0)  # (T_dense,)
        # Weighted sum: (B, T_dense, 2, C, H_b, W_b) → (B, 2, C, H_b, W_b)
        x = torch.einsum("btcdhw,t->bcdhw", x_dense_complex, w)

        # 2. Complex→real concat: (B, 2, C, H_b, W_b) → (B, 2*C, H_b, W_b)
        x = x.reshape(B, 2 * C, H_b, W_b)

        # 3. First conv
        x = self.act1(self.conv1(x))  # (B, mid_ch, H_b, W_b)

        # 4. Bilinear upsample to full resolution
        x = F.interpolate(
            x, size=(target_H, target_W), mode="bilinear", align_corners=True
        )

        # 5. Second conv + output
        x = self.act2(self.conv2(x))         # (B, mid_ch//2, H, W)
        x = self.conv_out(x)                 # (B, 1, H, W)

        # 6. Bounded output: sigmoid → [cs_min, cs_max]
        cs_map = self.cs_min + (self.cs_max - self.cs_min) * torch.sigmoid(x)

        return cs_map


# =============================================================================
# WavePINN v7 Main Model
# =============================================================================

class WavePINNv7(WavePINNv6):
    """
    Multi-task WavePINN: velocity reconstruction + spatial Cs estimation.

    Inherits all encoder/bottleneck/decoder/skip logic from WavePINNv6.
    Adds PhysicsHead that taps bottleneck features.

    Parameters (new vs V6)
    ----------------------
    cs_head_mid_channels : int
        PhysicsHead intermediate channels (default 128).
    cs_min, cs_max : float
        Output range for Cs map (m/s).
    """

    def __init__(
        self,
        # V6 params (all forwarded)
        in_channels: int = 1,
        encoder_depths: Optional[List[int]] = None,
        upsampling_factor: int = 12,
        complex_to_real_mode: str = "concat",
        use_naive_bn: bool = True,
        d_model: int = 256,
        n_heads: int = 8,
        n_self_layers: int = 2,
        n_freq: int = 32,
        n_refine_iters: int = 2,
        dropout: float = 0.1,
        use_attention_skips: bool = True,
        skip_attn_d_model: int = 128,
        skip_attn_heads: int = 4,
        skip_attn_n_freq: int = 32,
        skip_attn_dropout: float = 0.1,
        # V7 new params
        cs_head_mid_channels: int = 128,
        cs_min: float = 0.3,
        cs_max: float = 8.0,
    ):
        if encoder_depths is None:
            encoder_depths = [32, 64, 128, 256]

        super().__init__(
            in_channels=in_channels,
            encoder_depths=encoder_depths,
            upsampling_factor=upsampling_factor,
            complex_to_real_mode=complex_to_real_mode,
            use_naive_bn=use_naive_bn,
            d_model=d_model,
            n_heads=n_heads,
            n_self_layers=n_self_layers,
            n_freq=n_freq,
            n_refine_iters=n_refine_iters,
            dropout=dropout,
            use_attention_skips=use_attention_skips,
            skip_attn_d_model=skip_attn_d_model,
            skip_attn_heads=skip_attn_heads,
            skip_attn_n_freq=skip_attn_n_freq,
            skip_attn_dropout=skip_attn_dropout,
        )

        # Physics head
        bottleneck_ch = encoder_depths[-1]
        self.physics_head = PhysicsHead(
            bottleneck_channels=bottleneck_ch,
            mid_channels=cs_head_mid_channels,
            cs_min=cs_min,
            cs_max=cs_max,
        )

        logger.info(
            f"WavePINNv7 (multi-task) initialised:\n"
            f"  PhysicsHead mid_ch : {cs_head_mid_channels}\n"
            f"  Cs output range    : [{cs_min}, {cs_max}] m/s\n"
            f"  PhysicsHead params : {sum(p.numel() for p in self.physics_head.parameters()):,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        dt_seconds: Union[float, torch.Tensor] = 0.002,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 2, T_sparse, H, W)
        dt_seconds : float or Tensor

        Returns
        -------
        velocity : (B, 1, T_dense, H, W) in [-1, 1]
        cs_map   : (B, 1, H, W) in [cs_min, cs_max] m/s
        """
        B, _, T_sparse, H, W = x.shape

        # ===================== ENCODER ====================================
        x_enc, B, T = self._reshape_for_encoder(x)

        complex_skips: List[torch.Tensor] = []

        x_enc, skip1 = self.enc1(x_enc)
        complex_skips.append(skip1)

        x_enc, skip2 = self.enc2(x_enc)
        complex_skips.append(skip2)

        x_enc, skip3 = self.enc3(x_enc)
        complex_skips.append(skip3)

        x_enc, skip4 = self.enc4(x_enc)
        complex_skips.append(skip4)

        _, _, C_b, H_b, W_b = x_enc.shape
        x_temporal = x_enc.reshape(B, T_sparse, 2, C_b, H_b, W_b)

        # ===================== BOTTLENECK =================================
        x_dense_complex = self.bottleneck(x_temporal, dt_seconds)
        # (B, T_dense, 2, C, H_b, W_b)

        T_dense = x_dense_complex.shape[1]

        # ===================== PHYSICS HEAD (tap bottleneck) ==============
        cs_map = self.physics_head(x_dense_complex, H, W)
        # (B, 1, H, W)

        # ===================== BRIDGE + DECODER (same as V6) ==============
        # NOTE: The bottleneck already received the full per-sample dt_seconds tensor
        # and generates per-sample temporal encodings correctly.  The skip temporal
        # attention modules take a single time array (t_sparse_arr / t_dense_arr), so
        # we use the batch-mean dt as a practical approximation.  When PRF augmentation
        # produces mixed-dt batches, this is slightly physically inconsistent for the
        # skip connections (bottleneck is still exact per-sample).  Fixing this properly
        # requires changing the skip attention interface to accept (B, T) time tensors.
        if isinstance(dt_seconds, torch.Tensor):
            dt_scalar = dt_seconds.mean().item()
        else:
            dt_scalar = float(dt_seconds)
        dt_output = dt_scalar / self.upsampling_factor
        device = x.device
        t_sparse_arr = torch.arange(T_sparse, device=device, dtype=torch.float32) * dt_scalar
        t_dense_arr = torch.arange(T_dense, device=device, dtype=torch.float32) * dt_output

        x_real = self.bottleneck_bridge(x_dense_complex)

        # --- Level 4 ---
        skip4_c = complex_skips.pop()
        skip4_c = self._reshape_skip_for_temporal(skip4_c, B, T_sparse)
        if self.skip4_temporal_attn is not None:
            skip4_c = self.skip4_temporal_attn(skip4_c, t_sparse_arr, t_dense_arr)
        else:
            skip4_c = self._interpolate_skip_temporal(skip4_c, T_dense)
        skip4_r = self.skip_bridge_l4(skip4_c)

        x_dec = self._flatten_for_decoder(x_real, B, T_dense)
        skip4_flat = self._flatten_for_decoder(skip4_r, B, T_dense)
        x_dec = self.dec4(x_dec, skip4_flat)

        # --- Level 3 ---
        skip3_c = complex_skips.pop()
        skip3_c = self._reshape_skip_for_temporal(skip3_c, B, T_sparse)
        if self.skip3_temporal_attn is not None:
            skip3_c = self.skip3_temporal_attn(skip3_c, t_sparse_arr, t_dense_arr)
        else:
            skip3_c = self._interpolate_skip_temporal(skip3_c, T_dense)
        skip3_r = self.skip_bridge_l3(skip3_c)
        skip3_flat = self._flatten_for_decoder(skip3_r, B, T_dense)
        x_dec = self.dec3(x_dec, skip3_flat)

        # --- Level 2 ---
        skip2_c = complex_skips.pop()
        skip2_c = self._reshape_skip_for_temporal(skip2_c, B, T_sparse)
        skip2_c = self._interpolate_skip_temporal(skip2_c, T_dense)
        skip2_r = self.skip_bridge_l2(skip2_c)
        skip2_flat = self._flatten_for_decoder(skip2_r, B, T_dense)
        x_dec = self.dec2(x_dec, skip2_flat)

        # --- Level 1 ---
        skip1_c = complex_skips.pop()
        skip1_c = self._reshape_skip_for_temporal(skip1_c, B, T_sparse)
        skip1_c = self._interpolate_skip_temporal(skip1_c, T_dense)
        skip1_r = self.skip_bridge_l1(skip1_c)
        skip1_flat = self._flatten_for_decoder(skip1_r, B, T_dense)
        x_dec = self.dec1(x_dec, skip1_flat)

        # ===================== OUTPUT HEAD ================================
        velocity = self.output_activation(self.output_conv(x_dec))
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        return velocity, cs_map


# =============================================================================
# Factory
# =============================================================================

def create_wavepinn_v7(
    config: Optional[Dict] = None,
    **kwargs,
) -> WavePINNv7:
    """Create and return a configured WavePINNv7."""
    defaults = dict(
        in_channels=1,
        encoder_depths=[32, 64, 128, 256],
        upsampling_factor=12,
        complex_to_real_mode="concat",
        use_naive_bn=True,
        d_model=256,
        n_heads=8,
        n_self_layers=2,
        n_freq=32,
        n_refine_iters=2,
        dropout=0.1,
        use_attention_skips=True,
        skip_attn_d_model=128,
        skip_attn_heads=4,
        skip_attn_n_freq=32,
        skip_attn_dropout=0.1,
        # V7 defaults
        cs_head_mid_channels=128,
        cs_min=0.3,
        cs_max=8.0,
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return WavePINNv7(**defaults)


def count_parameters_v7(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }
