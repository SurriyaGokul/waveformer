"""
WavePINN v9: SWE-Accurate Temporal Super-Resolution.

Architecture is identical to WavePINNv6 (proven backbone achieving
0.141 m/s Cs MAE on 5 kHz evaluation) plus a lightweight ScalarCsHead
that outputs one shear-wave speed estimate per sample.

Key design decisions vs V7/V8:
  - ScalarCsHead replaces the spatial PhysicsHead (B,1,H,W) map.
    The only available supervision is a scalar cs per phantom, so a
    scalar head is the correct matching output.
  - Bias initialised to logit((cs_init - cs_min)/(cs_max - cs_min))
    where cs_init ≈ training-set mean (2.54 m/s), so sigmoid(bias)=0.286
    → output ≈ 2.54 m/s at init.  This avoids the sigmoid-midpoint freeze
    that locked V7/V8's PhysicsHead output at 4.15 m/s from step 0.
  - No ToA loss — the ToA regression approach caused phase-2 divergence in
    V8.  Physics regularisation comes from WaveCoherenceLoss + direct
    scalar Cs supervision.

Returns:
    (velocity, cs_scalar)
    velocity  : (B, 1, T_dense, H, W) in [-1, 1]  (V6-identical output)
    cs_scalar : (B,)                   in [cs_min, cs_max] m/s
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse V6 backbone unchanged
from src.architectures.wavepinn_v6 import WavePINNv6, create_wavepinn_v6

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Scalar Cs Head
# =============================================================================

class ScalarCsHead(nn.Module):
    """
    Lightweight head that produces one shear-wave speed estimate per sample.

    Taps the dense bottleneck features (same tap-point used by V7/V8
    PhysicsHead) and collapses them to a single scalar via:

        bottleneck (B, T, 2, C, H_b, W_b)
            ↓  softmax-weighted temporal collapse  →  (B, 2, C, H_b, W_b)
            ↓  complex→real concat                 →  (B, 2C, H_b, W_b)
            ↓  global average pool (spatial)       →  (B, 2C)
            ↓  Linear(2C→64) + GELU                →  (B, 64)
            ↓  Linear(64→1)                        →  (B, 1) logit
            ↓  cs_min + (cs_max−cs_min)·sigmoid     →  (B,)  [m/s]

    Parameters
    ----------
    bottleneck_channels : int
        Channel count at bottleneck (C in V6 = 256).
    cs_min, cs_max : float
        Output range in m/s.
    cs_init : float
        Target mean output at initialisation.  Bias of the final linear layer
        is set so sigmoid(bias) = (cs_init − cs_min)/(cs_max − cs_min).
    max_T_dense : int
        Maximum T_dense for the learnable time-collapse logits buffer.
    """

    def __init__(
        self,
        bottleneck_channels: int = 256,
        cs_min: float = 0.3,
        cs_max: float = 8.0,
        cs_init: float = 2.54,
        max_T_dense: int = 64,
    ):
        super().__init__()
        self.cs_min = cs_min
        self.cs_max = cs_max

        in_ch = 2 * bottleneck_channels  # complex concat → real

        # Learnable temporal weights (softmax over T_dense)
        self.time_logits = nn.Parameter(torch.zeros(max_T_dense))

        # MLP: global-pooled features → scalar logit
        self.fc1 = nn.Linear(in_ch, 64)
        self.fc2 = nn.Linear(64, 1)

        # --- Initialisation ---
        # fc1: standard He init (default)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)

        # fc2: zero weights so logit = bias only at init.
        # Set bias = logit inverse so sigmoid(bias) = target fraction.
        target_frac = (cs_init - cs_min) / (cs_max - cs_min)
        target_frac = float(min(max(target_frac, 1e-3), 1.0 - 1e-3))
        init_bias = math.log(target_frac / (1.0 - target_frac))
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)

        logger.info(
            f"ScalarCsHead init: cs_init={cs_init:.2f} m/s "
            f"→ target_frac={target_frac:.3f} → logit_bias={init_bias:.3f} "
            f"→ sigmoid={torch.sigmoid(torch.tensor(init_bias)).item():.3f}"
        )

    def forward(self, x_dense_complex: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_dense_complex : (B, T_dense, 2, C, H_b, W_b)

        Returns
        -------
        cs_scalar : (B,) in [cs_min, cs_max] m/s
        """
        B, T, two, C, H_b, W_b = x_dense_complex.shape

        # Temporal collapse: softmax over T_dense
        w = torch.softmax(self.time_logits[:T], dim=0)  # (T,)
        # einsum: "b t c h w, t -> b c h w"
        x_collapsed = torch.einsum("btchw,t->bchw",
                                   x_dense_complex.reshape(B, T, two * C, H_b, W_b),
                                   w)  # (B, 2C, H_b, W_b)

        # Global average pool
        x_pool = x_collapsed.mean(dim=(2, 3))  # (B, 2C)

        # MLP
        h = F.gelu(self.fc1(x_pool))   # (B, 64)
        logit = self.fc2(h).squeeze(1)  # (B,)

        cs = self.cs_min + (self.cs_max - self.cs_min) * torch.sigmoid(logit)
        return cs  # (B,)


# =============================================================================
# 2. WavePINNv9 — V6 Backbone + ScalarCsHead
# =============================================================================

class WavePINNv9(WavePINNv6):
    """
    WavePINN v9: V6 backbone + ScalarCsHead.

    Inherits V6's encoder, attention bottleneck, bridge, decoder, and output
    head verbatim.  Adds a ScalarCsHead that taps the dense bottleneck output.

    Parameters
    ----------
    cs_min, cs_max : float
        Shear-wave speed range for sigmoid scaling.
    cs_init : float
        Target mean Cs at initialisation (≈ training-set mean across phantoms).
    All other parameters forwarded to WavePINNv6.
    """

    def __init__(
        self,
        cs_min: float = 0.3,
        cs_max: float = 8.0,
        cs_init: float = 2.54,
        **kwargs,
    ):
        super().__init__(**kwargs)

        bottleneck_channels = self.encoder_depths[-1]  # 256 by default

        self.cs_head = ScalarCsHead(
            bottleneck_channels=bottleneck_channels,
            cs_min=cs_min,
            cs_max=cs_max,
            cs_init=cs_init,
        )

        self.cs_min = cs_min
        self.cs_max = cs_max

    def forward(
        self,
        x: torch.Tensor,
        dt_seconds: Union[float, torch.Tensor] = 0.002,
        detach_head: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 2, T_sparse, H, W)  sparse IQ input
        dt_seconds : float or (B,)  input frame gap in seconds
        detach_head : bool
            If True, the cs_head receives a detached copy of the bottleneck
            features — its loss does NOT backpropagate into the backbone.
            Used during Phase 1 to prevent Cs supervision corrupting the
            velocity reconstruction.

        Returns
        -------
        velocity  : (B, 1, T_dense, H, W) in [-1, 1]
        cs_scalar : (B,)                   in [cs_min, cs_max] m/s
        """
        B, _, T_sparse, H, W = x.shape

        # ===================== ENCODER ====================================
        x_enc, B, T = self._reshape_for_encoder(x)

        complex_skips: List[torch.Tensor] = []
        x_enc, skip1 = self.enc1(x_enc); complex_skips.append(skip1)
        x_enc, skip2 = self.enc2(x_enc); complex_skips.append(skip2)
        x_enc, skip3 = self.enc3(x_enc); complex_skips.append(skip3)
        x_enc, skip4 = self.enc4(x_enc); complex_skips.append(skip4)

        _, _, C_b, H_b, W_b = x_enc.shape
        x_temporal = x_enc.reshape(B, T_sparse, 2, C_b, H_b, W_b)

        # ===================== BOTTLENECK =================================
        x_dense_complex = self.bottleneck(x_temporal, dt_seconds)
        # (B, T_dense, 2, C_b, H_b, W_b)

        # ===================== SCALAR CS HEAD =============================
        head_input = x_dense_complex.detach() if detach_head else x_dense_complex
        cs_scalar = self.cs_head(head_input)  # (B,)

        # ===================== VELOCITY DECODER ===========================
        T_dense = x_dense_complex.shape[1]

        if isinstance(dt_seconds, torch.Tensor):
            dt_scalar = dt_seconds.mean().item()
        else:
            dt_scalar = float(dt_seconds)
        dt_output = dt_scalar / self.upsampling_factor
        device = x.device
        t_sparse_arr = torch.arange(T_sparse, device=device, dtype=torch.float32) * dt_scalar
        t_dense_arr  = torch.arange(T_dense,  device=device, dtype=torch.float32) * dt_output

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

        # Output
        velocity = self.output_activation(self.output_conv(x_dec))  # (B*T, 1, H, W)
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        return velocity, cs_scalar


# =============================================================================
# 3. Factory + Utilities
# =============================================================================

def create_wavepinn_v9(config: Optional[Dict] = None, **kwargs) -> WavePINNv9:
    """Create a WavePINNv9 from a config dict or keyword overrides."""
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
        cs_min=0.3,
        cs_max=8.0,
        cs_init=2.54,
    )
    if config:
        defaults.update(config)
    defaults.update(kwargs)
    return WavePINNv9(**defaults)


def count_parameters_v9(model: WavePINNv9) -> Dict[str, int]:
    """Return total / trainable / head parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.cs_head.parameters())
    backbone = total - head
    return {
        "total": total,
        "trainable": trainable,
        "backbone": backbone,
        "cs_head": head,
    }
