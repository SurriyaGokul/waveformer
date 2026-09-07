"""
WavePINN v3: Cross-Attention + Physics Refinement Temporal Super-Resolution.

Replaces v2's optical-flow-based bottleneck with a cross-attention temporal
interpolator and iterative physics refinement block.  Works at any PRF
(50 Hz -- 20 kHz) without explicit motion estimation.

Architecture:
    Input:  (B, 2, T_sparse, H, W) + dt_seconds

    Encoder (complex-valued, reused from v2):
        ComplexDownBlock x4: [32, 64, 128, 256] with AvgPool

    Bottleneck -- PhysicsAwareTemporalSuperResolution:
        1. ComplexBiLSTM  (bidirectional, 256 -> 128 per direction, concat -> 256)
        2. TemporalCrossAttentionInterpolator  (T_sparse -> T_dense)
        3. PhysicsRefinementBlock  (2 iterations, shared weights)

    Bridge: ComplexToRealBridge (concat mode)

    Decoder (real-valued, NO FiLM):
        RealUpBlock x4 with skip connections

    Output: Conv(32 -> 1) + Tanh -> (B, 1, T_dense, H, W) in [-1, 1]
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.architectures.complex_modules import ComplexConv2d
from src.architectures.complex_lstm import ComplexConvLSTM
from src.architectures.velocity_unet import (
    ComplexConvBlock,
    ComplexDownBlock,
    ComplexToRealBridge,
    RealConvBlock,
    RealUpBlock,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Continuous Time Encoding
# =============================================================================

class ContinuousTimeEncoding(nn.Module):
    """
    Sinusoidal encoding of absolute time values in seconds.

    Uses log-spaced frequencies from 10^{-4} to 10^{2} Hz so the encoding
    is informative across the full PRF range (50 Hz -- 20 kHz).

    Parameters
    ----------
    n_freq : int
        Number of frequency bands.  Output dimension = 2 * n_freq
        (sin and cos for each frequency).
    """

    def __init__(self, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        # Log-spaced frequencies: 10^linspace(-4, 2, n_freq)
        freqs = torch.pow(10.0, torch.linspace(-4.0, 2.0, n_freq))  # (n_freq,)
        self.register_buffer("freqs", freqs)  # not a parameter

    def forward(self, t_seconds: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t_seconds : Tensor
            1-D tensor of time values in seconds, shape (T,).

        Returns
        -------
        Tensor of shape (T, 2 * n_freq).
        """
        # t_seconds: (T,)  freqs: (n_freq,)
        # angles: (T, n_freq) = 2*pi*f*t
        angles = 2.0 * math.pi * t_seconds.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (T, 2*n_freq)


# =============================================================================
# 2. Complex Bidirectional LSTM
# =============================================================================

class ComplexBiLSTM(nn.Module):
    """
    Bidirectional wrapper around two ComplexConvLSTM instances.

    Forward direction processes t = 0 .. T-1; backward direction flips
    the time axis, processes, and flips back.  Outputs are concatenated
    on the channel dimension (dim=3) to give 2 * hidden_channels.

    Input:  (B, T, 2, C_in, H, W)
    Output: (B, T, 2, 2*C_hidden, H, W)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.forward_lstm = ComplexConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        self.backward_lstm = ComplexConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, T, 2, C_in, H, W)

        Returns
        -------
        Tensor  (B, T, 2, 2*hidden_channels, H, W)
        """
        # Forward direction
        fwd_out, _ = self.forward_lstm(x)  # (B, T, 2, C_hid, H, W)

        # Backward direction: flip time, process, flip back
        x_flip = x.flip(dims=[1])
        bwd_out, _ = self.backward_lstm(x_flip)
        bwd_out = bwd_out.flip(dims=[1])  # (B, T, 2, C_hid, H, W)

        # Concatenate on channel dim (dim=3)
        return torch.cat([fwd_out, bwd_out], dim=3)  # (B, T, 2, 2*C_hid, H, W)


# =============================================================================
# 3. Temporal Cross-Attention Interpolator
# =============================================================================

class TemporalCrossAttentionInterpolator(nn.Module):
    """
    Cross-attention from sparse to dense temporal positions.

    Queries are sinusoidal encodings of the dense target times;
    keys/values come from LSTM features plus sparse time encodings.
    Attention is computed independently per spatial position (B*H*W tokens).

    A learnable residual blending parameter ``alpha`` (initialised to 0.01)
    mixes the attention output with a safe linear-interpolation baseline.
    """

    def __init__(
        self,
        channels: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_freq: int = 32,
        upsampling_factor: int = 12,
    ):
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.upsampling_factor = upsampling_factor

        # Time encoding
        self.time_enc = ContinuousTimeEncoding(n_freq=n_freq)
        time_dim = 2 * n_freq  # 64 by default

        # Key/Value projection: ComplexConv2d 256 -> 64 (1x1)
        # After flattening complex dim: 2*64 = 128 = d_model
        self.kv_proj = ComplexConv2d(channels, d_model // 2, kernel_size=1, padding=0)

        # Time encoding projection to d_model
        self.key_time_proj = nn.Linear(time_dim, d_model)
        self.query_time_proj = nn.Linear(time_dim, d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # Post-attention: LayerNorm + FFN + LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

        # Output projection: ComplexConv2d 64 -> 256 (1x1)
        self.out_proj = ComplexConv2d(d_model // 2, channels, kernel_size=1, padding=0)

        # Learnable residual blending (starts small)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def _linear_interpolate(self, x: torch.Tensor, T_dense: int) -> torch.Tensor:
        """
        Temporal linear interpolation of complex features.

        Same approach as SimpleTemporalBottleneck in velocity_unet.py:
        reshape to (B, 2*C, T, H*W), bilinear interpolate along T,
        then reshape back.

        Parameters
        ----------
        x : Tensor  (B, T_sparse, 2, C, H, W)
        T_dense : int

        Returns
        -------
        Tensor  (B, T_dense, 2, C, H, W)
        """
        B, T_sp, _, C, H, W = x.shape
        # -> (B, 2, C, T_sp, H, W) -> (B, 2*C, T_sp, H*W)
        flat = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 2 * C, T_sp, H * W)
        interp = F.interpolate(
            flat, size=(T_dense, H * W), mode="bilinear", align_corners=True,
        )
        # -> (B, 2, C, T_dense, H, W) -> (B, T_dense, 2, C, H, W)
        interp = interp.reshape(B, 2, C, T_dense, H, W).permute(0, 3, 1, 2, 4, 5)
        return interp

    def _cross_attention_forward(
        self,
        sparse_features: torch.Tensor,
        t_sparse: torch.Tensor,
        t_dense: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core cross-attention computation (wrapped for gradient checkpointing).
        """
        B, T_sp, _, C, H, W = sparse_features.shape
        T_dn = t_dense.shape[0]

        # --- Key / Value projection ---
        # Reshape to (B*T_sp, 2, C, H, W) for ComplexConv2d
        kv_in = sparse_features.reshape(B * T_sp, 2, C, H, W)
        kv = self.kv_proj(kv_in)  # (B*T_sp, 2, d_model//2, H, W)

        # Flatten complex dim: (B*T_sp, 2, d_model//2, H, W) -> (B*T_sp, d_model, H, W)
        kv = kv.reshape(B * T_sp, self.d_model, H, W)

        # -> (B, T_sp, d_model, H, W) -> (B, H, W, T_sp, d_model)
        kv = kv.reshape(B, T_sp, self.d_model, H, W).permute(0, 3, 4, 1, 2)
        # -> (B*H*W, T_sp, d_model)
        kv = kv.reshape(B * H * W, T_sp, self.d_model)

        # --- Add time encoding to keys ---
        sparse_time_enc = self.time_enc(t_sparse)  # (T_sp, time_dim)
        sparse_time_feat = self.key_time_proj(sparse_time_enc)  # (T_sp, d_model)
        # Broadcast and add: (B*H*W, T_sp, d_model)
        kv = kv + sparse_time_feat.unsqueeze(0)

        # --- Query from dense time encoding ---
        dense_time_enc = self.time_enc(t_dense)  # (T_dn, time_dim)
        query = self.query_time_proj(dense_time_enc)  # (T_dn, d_model)
        # Expand to (B*H*W, T_dn, d_model)
        query = query.unsqueeze(0).expand(B * H * W, -1, -1)

        # --- Cross-attention ---
        attn_out, _ = self.cross_attn(query, kv, kv)  # (B*H*W, T_dn, d_model)

        # --- Post-attention: LN + FFN + LN ---
        attn_out = self.ln1(attn_out)
        attn_out = attn_out + self.ffn(attn_out)
        attn_out = self.ln2(attn_out)

        # --- Reshape back to complex spatial ---
        # (B*H*W, T_dn, d_model) -> (B, H, W, T_dn, d_model) -> (B, T_dn, d_model, H, W)
        attn_out = attn_out.reshape(B, H, W, T_dn, self.d_model)
        attn_out = attn_out.permute(0, 3, 4, 1, 2)  # (B, T_dn, d_model, H, W)

        # -> (B*T_dn, 2, d_model//2, H, W)
        attn_out = attn_out.reshape(B * T_dn, 2, self.d_model // 2, H, W)

        # Output projection: ComplexConv2d d_model//2 -> channels
        attn_out = self.out_proj(attn_out)  # (B*T_dn, 2, channels, H, W)

        # -> (B, T_dn, 2, channels, H, W)
        attn_out = attn_out.reshape(B, T_dn, 2, self.channels, H, W)

        return attn_out

    def forward(
        self,
        sparse_features: torch.Tensor,
        t_sparse: torch.Tensor,
        t_dense: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sparse_features : Tensor  (B, T_sparse, 2, C=256, H, W)
        t_sparse : Tensor  (T_sparse,)  time values in seconds
        t_dense  : Tensor  (T_dense,)   time values in seconds

        Returns
        -------
        Tensor  (B, T_dense, 2, C, H, W)
        """
        T_dn = t_dense.shape[0]

        # Linear interpolation baseline
        linear_interp = self._linear_interpolate(sparse_features, T_dn)

        # Cross-attention with gradient checkpointing
        attn_out = checkpoint.checkpoint(
            self._cross_attention_forward,
            sparse_features,
            t_sparse,
            t_dense,
            use_reentrant=False,
        )

        # Residual blending
        return linear_interp + self.alpha * attn_out


# =============================================================================
# 4. Physics Refinement Block
# =============================================================================

class PhysicsRefinementBlock(nn.Module):
    """
    Iterative physics-aware refinement with shared weights.

    Each iteration applies:
        1. Temporal convolution (along T axis)
        2. Spatial convolution  (along H, W axes)
        3. dt FiLM modulation  of temporal features
        4. Fusion of [input, temporal, spatial] via 1x1 convolution

    Residual connection: z = z + residual_scale * fused.

    Parameters
    ----------
    channels : int
        Complex channel count C (real tensor width = 2*C).
    n_iterations : int
        Number of refinement iterations (shared weights).
    residual_scale : float
        Scale factor for the residual update.
    """

    def __init__(self, channels: int = 256, n_iterations: int = 2, residual_scale: float = 0.1):
        super().__init__()
        self.channels = channels
        self.n_iterations = n_iterations
        self.residual_scale = residual_scale

        C2 = 2 * channels  # flattened complex channel count

        # Temporal conv: Conv1d on (B*H*W, 2*C, T)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(C2, C2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(C2, C2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Spatial conv: Conv2d on (B*T, 2*C, H, W)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(C2, C2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(C2, C2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # dt FiLM: log10(dt_output) -> gamma, beta for temporal features
        self.dt_film = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 4 * channels),  # -> gamma (2*C) + beta (2*C)
        )
        # Initialise dt FiLM to near-identity (small weights, zero bias)
        # Using small random weights instead of zeros so gradients flow
        # to the first layer immediately (zero weights block backprop).
        nn.init.normal_(self.dt_film[2].weight, std=0.001)
        nn.init.zeros_(self.dt_film[2].bias)

        # Fusion: concat [input, temporal, spatial] -> 1x1 conv
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * C2, C2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(C2, C2, kernel_size=1),
        )

    def forward(self, z: torch.Tensor, dt_seconds: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor  (B, T, 2, C, H, W)
        dt_seconds : float or Tensor (B,)
            Time step between dense output frames (seconds). Tensor path enables
            per-sample dt conditioning in mixed-PRF batches; float path is
            bit-identical to the original and used by v3/v4/v5 callers.

        Returns
        -------
        Tensor  (B, T, 2, C, H, W)
        """
        B, T, _, C, H, W = z.shape
        C2 = 2 * C

        # Compute FiLM parameters from log10(dt)
        if isinstance(dt_seconds, torch.Tensor):
            dt_t = dt_seconds.to(device=z.device, dtype=torch.float32).reshape(-1)  # (B,) or (1,)
            log_dt = torch.log10(dt_t.clamp_min(1e-12)).reshape(-1, 1)  # (B_or_1, 1)
            film_params = self.dt_film(log_dt)  # (B_or_1, 4*C)
            gamma = film_params[:, :C2].reshape(-1, 1, C2, 1, 1)  # (B_or_1, 1, C2, 1, 1)
            beta  = film_params[:, C2:].reshape(-1, 1, C2, 1, 1)
        else:
            log_dt = math.log10(max(float(dt_seconds), 1e-12))
            log_dt_tensor = z.new_tensor([[log_dt]])  # (1, 1)
            film_params = self.dt_film(log_dt_tensor)  # (1, 4*C)
            gamma = film_params[:, :C2].reshape(1, 1, C2, 1, 1)
            beta  = film_params[:, C2:].reshape(1, 1, C2, 1, 1)

        for _ in range(self.n_iterations):
            # Flatten complex: (B, T, 2, C, H, W) -> (B, T, 2*C, H, W)
            z_flat = z.reshape(B, T, C2, H, W)

            # --- Temporal convolution ---
            # (B, T, 2*C, H, W) -> (B*H*W, 2*C, T)
            z_temporal = z_flat.permute(0, 3, 4, 2, 1).reshape(B * H * W, C2, T)
            z_temporal = self.temporal_conv(z_temporal)
            # -> (B, H, W, 2*C, T) -> (B, T, 2*C, H, W)
            z_temporal = z_temporal.reshape(B, H, W, C2, T).permute(0, 4, 3, 1, 2)

            # Apply dt FiLM to temporal features
            z_temporal = z_temporal * (1.0 + gamma) + beta

            # --- Spatial convolution ---
            # (B, T, 2*C, H, W) -> (B*T, 2*C, H, W)
            z_spatial = z_flat.reshape(B * T, C2, H, W)
            z_spatial = self.spatial_conv(z_spatial)
            # -> (B, T, 2*C, H, W)
            z_spatial = z_spatial.reshape(B, T, C2, H, W)

            # --- Fusion ---
            # Concat [input, temporal, spatial]: (B, T, 3*2*C, H, W)
            fused_input = torch.cat([z_flat, z_temporal, z_spatial], dim=2)
            # -> (B*T, 3*2*C, H, W)
            fused_input = fused_input.reshape(B * T, 3 * C2, H, W)
            fused = self.fusion(fused_input)  # (B*T, 2*C, H, W)
            # -> (B, T, 2, C, H, W)
            fused = fused.reshape(B, T, 2, C, H, W)

            # Residual update
            z = z + self.residual_scale * fused

        return z


# =============================================================================
# 5. Physics-Aware Temporal Super-Resolution
# =============================================================================

class PhysicsAwareTemporalSuperResolution(nn.Module):
    """
    Full bottleneck module assembling BiLSTM + CrossAttention + PhysicsRefinement.

    Parameters
    ----------
    channels : int
        Encoder bottleneck channel count (256).
    hidden_channels : int
        Per-direction LSTM hidden channels (128 default, concat -> 256).
    upsampling_factor : int
        Temporal upsampling factor.
    d_model : int
        Cross-attention model dimension.
    n_heads : int
        Number of attention heads.
    n_freq : int
        Number of sinusoidal frequency bands for time encoding.
    n_refine_iters : int
        Number of physics refinement iterations.
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 128,
        upsampling_factor: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        n_freq: int = 32,
        n_refine_iters: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.upsampling_factor = upsampling_factor

        # 1. Bidirectional LSTM: input C -> hidden per dir, concat -> 2*hidden = C
        assert 2 * hidden_channels == channels, (
            f"2 * hidden_channels ({2 * hidden_channels}) must equal channels ({channels}) "
            "so that BiLSTM output matches encoder channel width."
        )
        self.bilstm = ComplexBiLSTM(
            input_channels=channels,
            hidden_channels=hidden_channels,
            num_layers=1,
            kernel_size=3,
        )

        # 2. Cross-attention interpolator
        self.cross_attn = TemporalCrossAttentionInterpolator(
            channels=channels,
            d_model=d_model,
            n_heads=n_heads,
            n_freq=n_freq,
            upsampling_factor=upsampling_factor,
        )

        # 3. Physics refinement
        self.physics_refine = PhysicsRefinementBlock(
            channels=channels,
            n_iterations=n_refine_iters,
        )

    def forward(
        self,
        sparse_features: torch.Tensor,
        dt_seconds: Union[float, torch.Tensor] = 0.002,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sparse_features : Tensor  (B, T_sparse, 2, C, H, W)
        dt_seconds : float or Tensor
            Time between sparse input frames (seconds).  If Tensor, .mean().item()
            is used for time array construction.

        Returns
        -------
        Tensor  (B, T_dense, 2, C, H, W)
        """
        B, T_sparse = sparse_features.shape[:2]
        device = sparse_features.device

        # Resolve dt to scalar for time arrays
        if isinstance(dt_seconds, torch.Tensor):
            dt_scalar = dt_seconds.mean().item()
        else:
            dt_scalar = float(dt_seconds)

        T_dense = 1 + (T_sparse - 1) * self.upsampling_factor
        dt_output = dt_scalar / self.upsampling_factor

        # Build time arrays (in seconds)
        t_sparse = torch.arange(T_sparse, device=device, dtype=torch.float32) * dt_scalar
        t_dense = torch.arange(T_dense, device=device, dtype=torch.float32) * dt_output

        # 1. BiLSTM
        lstm_out = self.bilstm(sparse_features)  # (B, T_sparse, 2, C, H, W)

        # 2. Cross-attention interpolation
        dense_features = self.cross_attn(lstm_out, t_sparse, t_dense)  # (B, T_dense, 2, C, H, W)

        # 3. Physics refinement
        dense_features = self.physics_refine(dense_features, dt_output)

        return dense_features


# =============================================================================
# 6. WavePINN v3 Main Model
# =============================================================================

class WavePINNv3(nn.Module):
    """
    WavePINN v3: Cross-Attention + Physics Refinement UNet.

    Parameters
    ----------
    in_channels : int
        Input channels per complex component (1 for IQ).
    encoder_depths : list[int]
        Channel widths at each encoder level.
    upsampling_factor : int
        Temporal super-resolution factor.
    complex_to_real_mode : str
        Bridge mode ('concat' | 'magnitude_phase' | 'magnitude').
    use_naive_bn : bool
        Use naive (per-component) batch normalisation.
    d_model : int
        Cross-attention model dimension.
    n_heads : int
        Cross-attention head count.
    n_freq : int
        Number of sinusoidal frequency bands.
    hidden_channels : int
        Per-direction LSTM hidden channels.
    n_refine_iters : int
        Physics refinement iterations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_depths: Optional[List[int]] = None,
        upsampling_factor: int = 12,
        complex_to_real_mode: str = "concat",
        use_naive_bn: bool = True,
        d_model: int = 128,
        n_heads: int = 4,
        n_freq: int = 32,
        hidden_channels: int = 128,
        n_refine_iters: int = 2,
    ):
        super().__init__()
        if encoder_depths is None:
            encoder_depths = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.encoder_depths = encoder_depths
        self.upsampling_factor = upsampling_factor
        self.complex_to_real_mode = complex_to_real_mode

        # ----- Complex encoder -----
        self.enc1 = ComplexDownBlock(in_channels, encoder_depths[0], use_naive_bn=use_naive_bn)
        self.enc2 = ComplexDownBlock(encoder_depths[0], encoder_depths[1], use_naive_bn=use_naive_bn)
        self.enc3 = ComplexDownBlock(encoder_depths[1], encoder_depths[2], use_naive_bn=use_naive_bn)
        self.enc4 = ComplexDownBlock(encoder_depths[2], encoder_depths[3], use_naive_bn=use_naive_bn)

        bottleneck_ch = encoder_depths[3]

        # ----- Temporal bottleneck -----
        self.bottleneck = PhysicsAwareTemporalSuperResolution(
            channels=bottleneck_ch,
            hidden_channels=hidden_channels,
            upsampling_factor=upsampling_factor,
            d_model=d_model,
            n_heads=n_heads,
            n_freq=n_freq,
            n_refine_iters=n_refine_iters,
        )

        # ----- Complex -> Real bridges -----
        self.bottleneck_bridge = ComplexToRealBridge(bottleneck_ch, mode=complex_to_real_mode)
        self.skip_bridge_l1 = ComplexToRealBridge(encoder_depths[0], mode=complex_to_real_mode)
        self.skip_bridge_l2 = ComplexToRealBridge(encoder_depths[1], mode=complex_to_real_mode)
        self.skip_bridge_l3 = ComplexToRealBridge(encoder_depths[2], mode=complex_to_real_mode)
        self.skip_bridge_l4 = ComplexToRealBridge(encoder_depths[3], mode=complex_to_real_mode)

        bottleneck_real_ch = self.bottleneck_bridge.out_channels
        skip_real_ch = [
            self.skip_bridge_l1.out_channels,
            self.skip_bridge_l2.out_channels,
            self.skip_bridge_l3.out_channels,
            self.skip_bridge_l4.out_channels,
        ]

        # ----- Real decoder (NO FiLM) -----
        self.dec4 = RealUpBlock(bottleneck_real_ch, skip_real_ch[3], encoder_depths[3])
        self.dec3 = RealUpBlock(encoder_depths[3], skip_real_ch[2], encoder_depths[2])
        self.dec2 = RealUpBlock(encoder_depths[2], skip_real_ch[1], encoder_depths[1])
        self.dec1 = RealUpBlock(encoder_depths[1], skip_real_ch[0], encoder_depths[0])

        # ----- Output head -----
        self.output_conv = nn.Conv2d(encoder_depths[0], 1, kernel_size=1)
        self.output_activation = nn.Tanh()
        nn.init.zeros_(self.output_conv.bias)

        logger.info(
            f"WavePINNv3 initialised:\n"
            f"  Encoder depths      : {encoder_depths}\n"
            f"  Bottleneck real ch  : {bottleneck_real_ch}\n"
            f"  Skip real channels  : {skip_real_ch}\n"
            f"  Upsampling factor   : {upsampling_factor}\n"
            f"  d_model / n_heads   : {d_model} / {n_heads}\n"
            f"  Hidden channels     : {hidden_channels}\n"
            f"  Refine iterations   : {n_refine_iters}\n"
            f"  Complex->Real mode  : {complex_to_real_mode}"
        )

    # ------------------------------------------------------------------
    # Static helpers (copied from velocity_unet.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_for_encoder(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B, 2, T, H, W) -> (B*T, 2, 1, H, W)."""
        B, _, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, 2, 1, H, W)
        return x, B, T

    @staticmethod
    def _reshape_skip_for_temporal(skip: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """(B*T, 2, C, H, W) -> (B, T, 2, C, H, W)."""
        _, _, C, H, W = skip.shape
        return skip.reshape(B, T, 2, C, H, W)

    @staticmethod
    def _interpolate_skip_temporal(skip: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        Temporal-only linear interpolation of complex skip connections.

        Reshapes to (B, 2*C, T, H*W) and interpolates along T with
        align_corners=True.
        """
        B, T, _, C, H, W = skip.shape
        if T == target_T:
            return skip
        flat = skip.permute(0, 2, 3, 1, 4, 5).reshape(B, 2 * C, T, H * W)
        interp = F.interpolate(
            flat, size=(target_T, H * W), mode="bilinear", align_corners=True,
        )
        interp = interp.reshape(B, 2, C, target_T, H, W).permute(0, 3, 1, 2, 4, 5)
        return interp

    @staticmethod
    def _flatten_for_decoder(x: torch.Tensor, B: int, T_dense: int) -> torch.Tensor:
        """(B, T, C, H, W) -> (B*T, C, H, W)."""
        return x.reshape(B * T_dense, -1, x.shape[-2], x.shape[-1])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        dt_seconds: Union[float, torch.Tensor] = 0.002,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Sparse IQ input, shape ``(B, 2, T_sparse, H, W)``.
        dt_seconds : float or Tensor
            Time between sparse input frames (seconds).
            If Tensor of shape (B,), .mean().item() is used for time arrays.

        Returns
        -------
        velocity : Tensor
            Dense velocity maps ``(B, 1, T_dense, H, W)`` in [-1, 1].
        """
        B, _, T_sparse, H, W = x.shape

        # ===================== ENCODER ====================================
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

        # ===================== BOTTLENECK =================================
        x_dense_complex = self.bottleneck(x_temporal, dt_seconds)
        # (B, T_dense, 2, C, H_b, W_b)

        T_dense = x_dense_complex.shape[1]

        # ===================== BRIDGE: bottleneck complex -> real ==========
        x_real = self.bottleneck_bridge(x_dense_complex)
        # (B, T_dense, C_real, H_b, W_b)

        # ===================== DECODER ====================================

        # --- Level 4 ---
        skip4_c = complex_skips.pop()
        skip4_c = self._reshape_skip_for_temporal(skip4_c, B, T_sparse)
        skip4_c = self._interpolate_skip_temporal(skip4_c, T_dense)
        skip4_r = self.skip_bridge_l4(skip4_c)

        x_dec = self._flatten_for_decoder(x_real, B, T_dense)
        skip4_flat = self._flatten_for_decoder(skip4_r, B, T_dense)
        x_dec = self.dec4(x_dec, skip4_flat)

        # --- Level 3 ---
        skip3_c = complex_skips.pop()
        skip3_c = self._reshape_skip_for_temporal(skip3_c, B, T_sparse)
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
        velocity = self.output_activation(self.output_conv(x_dec))  # (B*T_dense, 1, H, W)

        # Reshape: (B*T_dense, 1, H, W) -> (B, 1, T_dense, H, W)
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        return velocity


# =============================================================================
# 7. Factory + Utilities
# =============================================================================

def create_wavepinn_v3(
    config: Optional[Dict] = None,
    **kwargs,
) -> WavePINNv3:
    """
    Create and return a configured ``WavePINNv3``.

    Parameters
    ----------
    config : dict, optional
        Model configuration dictionary.
    **kwargs
        Override individual hyper-parameters.

    Returns
    -------
    WavePINNv3
    """
    defaults = dict(
        in_channels=1,
        encoder_depths=[32, 64, 128, 256],
        upsampling_factor=12,
        complex_to_real_mode="concat",
        use_naive_bn=True,
        d_model=128,
        n_heads=4,
        n_freq=32,
        hidden_channels=128,
        n_refine_iters=2,
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return WavePINNv3(**defaults)


def count_parameters_v3(model: nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and non-trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict with keys 'total', 'trainable', 'non_trainable'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }
