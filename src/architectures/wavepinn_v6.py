"""
WavePINN v6: Spatiotemporal Attention Temporal Super-Resolution.

Replaces v2's ConvLSTM + optical flow bottleneck and v3's per-spatial-position
cross-attention with full spatiotemporal self-attention and cross-attention.

Key insight: at the bottleneck resolution (8×10), the full spatiotemporal
volume is only 5×80 = 400 tokens — full quadratic attention is trivially
cheap.  This fixes v3's factorisation problem where attention was per-spatial-
position with no spatial interaction.

Architecture:
    Input:  (B, 2, T_sparse, H, W) + dt_seconds

    Encoder (complex-valued, reused from v2):
        ComplexDownBlock x4: [32, 64, 128, 256] with AvgPool

    Bottleneck — SpatioTemporalAttentionBottleneck:
        1. SpatioTemporalSelfAttention  (400 tokens, 2 encoder layers)
        2. SparseToDenseCrossAttention  (3920 queries → 400 keys)
        3. PhysicsRefinementBlock  (2 iterations, shared weights)

    Bridge: ComplexToRealBridge (concat mode)

    Decoder (real-valued, NO FiLM):
        RealUpBlock x4 with skip connections

    Output: Conv(32 → 1) + Tanh → (B, 1, T_dense, H, W) in [-1, 1]
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.architectures.velocity_unet import (
    ComplexConvBlock,
    ComplexDownBlock,
    ComplexToRealBridge,
    RealConvBlock,
    RealUpBlock,
)
from src.architectures.wavepinn_v3 import (
    ContinuousTimeEncoding,
    PhysicsRefinementBlock,
)

logger = logging.getLogger(__name__)


def _dt_to_vector(
    dt_seconds: Union[float, torch.Tensor],
    B: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalise dt to a per-sample vector of shape (B,).

    - float → constant (B,) tensor
    - 0-dim / 1-elem tensor → expand to (B,)
    - (B,) tensor → passthrough (moved to device)
    """
    if isinstance(dt_seconds, torch.Tensor):
        dt = dt_seconds.to(device=device, dtype=torch.float32).reshape(-1)
        if dt.numel() == 1:
            dt = dt.expand(B)
    else:
        dt = torch.full((B,), float(dt_seconds), dtype=torch.float32, device=device)
    return dt  # (B,)


# =============================================================================
# 1. Spatiotemporal Positional Encoding
# =============================================================================

class SpatioTemporalPositionalEncoding(nn.Module):
    """
    Combined temporal (sinusoidal) + spatial (learned) positional encoding.

    Temporal encoding uses ContinuousTimeEncoding (log-spaced frequencies
    from 10^{-4} to 10^{2} Hz) to handle arbitrary dt spacing from PRF
    augmentation.

    Spatial encoding is a learned embedding table for H*W positions.
    The bottleneck grid (8×10) is too small and irregular for sinusoidal
    spatial encoding.

    Parameters
    ----------
    d_model : int
        Model dimension for the attention layers.
    n_freq : int
        Number of sinusoidal frequency bands for temporal encoding.
    max_spatial : int
        Maximum number of spatial positions (H_b * W_b).
    """

    def __init__(self, d_model: int = 256, n_freq: int = 32, max_spatial: int = 80):
        super().__init__()
        self.d_model = d_model

        # Temporal: sinusoidal → project to d_model
        self.time_enc = ContinuousTimeEncoding(n_freq=n_freq)
        self.time_proj = nn.Linear(2 * n_freq, d_model)

        # Spatial: learned embedding
        self.spatial_emb = nn.Parameter(torch.randn(max_spatial, d_model) * 0.02)

    def forward(
        self,
        t_seconds: torch.Tensor,
        n_spatial: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t_seconds : Tensor (T,) or (B, T)
            Absolute time values in seconds. 1-D returns (T*n_spatial, d_model);
            2-D returns (B, T*n_spatial, d_model) for per-sample dt paths.
        n_spatial : int
            Number of spatial positions (H_b * W_b).

        Returns
        -------
        Tensor (T*n_spatial, d_model) or (B, T*n_spatial, d_model)
        """
        spatial_feat = self.spatial_emb[:n_spatial]  # (n_spatial, d_model)

        if t_seconds.dim() == 1:
            # (T,) → (T*n_spatial, d_model)
            T = t_seconds.shape[0]
            time_feat = self.time_proj(self.time_enc(t_seconds))  # (T, d_model)
            combined = time_feat.unsqueeze(1) + spatial_feat.unsqueeze(0)  # (T, n_spatial, d_model)
            return combined.reshape(T * n_spatial, self.d_model)
        else:
            # (B, T) → (B, T*n_spatial, d_model)
            B, T = t_seconds.shape
            time_feat = self.time_proj(self.time_enc(t_seconds.reshape(-1)))  # (B*T, d_model)
            time_feat = time_feat.reshape(B, T, self.d_model)  # (B, T, d_model)
            # (B, T, 1, d) + (1, 1, n_spatial, d) → (B, T, n_spatial, d)
            combined = time_feat.unsqueeze(2) + spatial_feat.unsqueeze(0).unsqueeze(0)
            return combined.reshape(B, T * n_spatial, self.d_model)


# =============================================================================
# 2. Spatiotemporal Self-Attention (Stage 1)
# =============================================================================

class SpatioTemporalSelfAttention(nn.Module):
    """
    Full self-attention over the sparse spatiotemporal volume.

    With T_sparse=5 and H_b×W_b=80, we have 400 tokens — trivially
    cheap for quadratic attention.  Each token sees every other token
    across both space and time.

    Parameters
    ----------
    in_channels : int
        Flattened complex channel count (2 * C_encoder = 512).
    d_model : int
        Attention model dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate in attention and FFN.
    """

    def __init__(
        self,
        in_channels: int = 512,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project from flattened complex channels to d_model
        self.input_proj = nn.Linear(in_channels, d_model)

        # Standard transformer encoder layers (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project back to flattened complex channels.
        # Standard PyTorch init (Kaiming uniform). The previous v6 release
        # multiplied the residual by a learnable alpha initialised to 0.01,
        # which scaled gradients through this branch by 0.01 and prevented
        # the attention path from ever learning (epoch-5 plateau). The gate
        # is removed; gradients now flow through the residual at full
        # magnitude. NOTE: we deliberately do NOT zero-init the weights
        # here, because zero weights set ∂L/∂x = W^T · ∂L/∂y = 0 and would
        # kill the gradient signal upstream of this layer.
        self.output_proj = nn.Linear(d_model, in_channels)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, N_tokens, in_channels)
            Flattened spatiotemporal tokens.
        pos_enc : Tensor (N_tokens, d_model)
            Positional encoding for each token.

        Returns
        -------
        Tensor (B, N_tokens, in_channels) — full residual with input.
        """
        # Project to d_model and add positional encoding
        h = self.input_proj(x)  # (B, N, d_model)
        # pos_enc: (N, d_model) shared or (B, N, d_model) per-sample
        if pos_enc.dim() == 2:
            h = h + pos_enc.unsqueeze(0)  # broadcast (1, N, d_model)
        else:
            h = h + pos_enc  # (B, N, d_model)

        # Self-attention (pre-norm transformer with internal residuals)
        h = self.encoder(h)  # (B, N, d_model)

        # Project back
        h = self.output_proj(h)  # (B, N, in_channels)

        # Standard full residual (no learnable gate)
        return x + h


# =============================================================================
# 3. Sparse-to-Dense Cross-Attention (Stage 2)
# =============================================================================

class SparseToDenseCrossAttention(nn.Module):
    """
    Cross-attention from dense spatiotemporal queries to sparse keys/values.

    Queries are pure positional encodings for all T_dense × H_b × W_b
    positions.  Keys/values are the contextualized sparse tokens from
    Stage 1.  This generates 49 dense frames from 5 sparse frames.

    The attention matrix is 3920 × 400 = 1.57M entries — ~3MB at fp16.

    Parameters
    ----------
    in_channels : int
        Flattened complex channel count (512).
    d_model : int
        Attention model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 512,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels

        # Project sparse tokens to K, V space
        self.kv_proj = nn.Linear(in_channels, d_model)

        # Query projection from positional encoding (already d_model)
        # Add a learnable content bias so queries aren't purely positional
        self.query_bias = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention: LN + FFN + LN
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Project back to full channel width. Standard PyTorch init
        # (see SpatioTemporalSelfAttention.output_proj for the rationale —
        # zero-init would kill upstream gradient flow).
        self.output_proj = nn.Linear(d_model, in_channels)

    def _cross_attention_forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        sparse_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapped for gradient checkpointing."""
        # Add positional encoding to keys
        # sparse_pos: (N_sparse, d_model) shared or (B, N_sparse, d_model) per-sample
        if sparse_pos.dim() == 2:
            kv = kv + sparse_pos.unsqueeze(0)
        else:
            kv = kv + sparse_pos  # (B, N_sparse, d_model)

        # Cross-attention: Q attends to K/V
        attn_out, _ = self.cross_attn(query, kv, kv)  # (B, N_dense, d_model)

        # Post-attention with residual
        attn_out = self.norm1(query + attn_out)
        attn_out = self.norm2(attn_out + self.ffn(attn_out))

        return attn_out

    def forward(
        self,
        sparse_tokens: torch.Tensor,
        dense_pos_enc: torch.Tensor,
        sparse_pos_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sparse_tokens : Tensor (B, N_sparse, in_channels)
            Contextualized sparse spatiotemporal tokens from Stage 1.
        dense_pos_enc : Tensor (N_dense, d_model)
            Positional encoding for all dense output positions.
        sparse_pos_enc : Tensor (N_sparse, d_model)
            Positional encoding for sparse positions (added to keys).

        Returns
        -------
        Tensor (B, N_dense, in_channels)
        """
        B = sparse_tokens.shape[0]

        # Project sparse tokens to d_model for K, V
        kv = self.kv_proj(sparse_tokens)  # (B, N_sparse, d_model)

        # Queries: positional encoding + learned content bias
        # dense_pos_enc: (N_dense, d_model) shared or (B, N_dense, d_model) per-sample
        if dense_pos_enc.dim() == 2:
            query = dense_pos_enc.unsqueeze(0).expand(B, -1, -1) + self.query_bias  # (B, N_dense, d_model)
        else:
            query = dense_pos_enc + self.query_bias  # (B, N_dense, d_model)

        # Cross-attention with gradient checkpointing
        attn_out = checkpoint.checkpoint(
            self._cross_attention_forward,
            query,
            kv,
            sparse_pos_enc,
            use_reentrant=False,
        )  # (B, N_dense, d_model)

        # Project back to full channel width
        return self.output_proj(attn_out)  # (B, N_dense, in_channels)


# =============================================================================
# 3b. Per-Position Temporal Cross-Attention for Skip Connections (Option B)
# =============================================================================

class TemporalCrossAttentionSkip(nn.Module):
    """
    Per-position temporal cross-attention for skip-connection upsampling.

    Replaces linear temporal interpolation of skip features (which v3-v6
    inherited from v2) with learned attention. Each spatial position is an
    independent sequence: T_sparse keys, T_dense queries. No spatial mixing
    across positions — spatial structure is preserved by the encoder's
    convolutional receptive field.

    Linear interpolation is kept as a residual baseline so the layer starts
    no worse than the previous behaviour and the optimizer can grow the
    learned refinement freely.

    Cost (per skip):
        kv tokens : B * H * W * T_sparse
        query     : B * H * W * T_dense
        attn mat  : B * H * W * T_dense * T_sparse  (per spatial position)
    For skip3 at H/4 × W/4 = 1280 positions, T_sparse=5, T_dense=49:
        kv = 6.4K, query = 62.7K, attn = 313K  → trivial.
    """

    def __init__(
        self,
        encoder_channels: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_freq: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.in_channels = 2 * encoder_channels  # flattened complex
        self.d_model = d_model

        # Sinusoidal temporal positional encoding (handles arbitrary dt)
        self.time_enc = ContinuousTimeEncoding(n_freq=n_freq)
        self.time_proj = nn.Linear(2 * n_freq, d_model)

        # KV projection: feature space → d_model
        self.kv_proj = nn.Linear(self.in_channels, d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention LN + FFN + LN
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Project back to feature space
        self.output_proj = nn.Linear(d_model, self.in_channels)

    @staticmethod
    def _linear_interp_temporal(skip: torch.Tensor, target_T: int) -> torch.Tensor:
        """Linear temporal interpolation, kept as residual baseline."""
        B, T, _, C, H, W = skip.shape
        if T == target_T:
            return skip
        flat = skip.permute(0, 2, 3, 1, 4, 5).reshape(B, 2 * C, T, H * W)
        interp = F.interpolate(
            flat, size=(target_T, H * W), mode="bilinear", align_corners=True,
        )
        interp = interp.reshape(B, 2, C, target_T, H, W).permute(0, 3, 1, 2, 4, 5)
        return interp

    def _attention_forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention forward, wrapped for gradient checkpointing."""
        attn_out, _ = self.cross_attn(query, kv, kv)
        attn_out = self.norm1(query + attn_out)
        attn_out = self.norm2(attn_out + self.ffn(attn_out))
        return attn_out

    def forward(
        self,
        sparse_skip: torch.Tensor,
        t_sparse: torch.Tensor,
        t_dense: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sparse_skip : Tensor (B, T_sparse, 2, C, H, W)
        t_sparse    : Tensor (T_sparse,) absolute times in seconds
        t_dense     : Tensor (T_dense,)  absolute times in seconds

        Returns
        -------
        Tensor (B, T_dense, 2, C, H, W)
        """
        B, T_sparse, _, C, H, W = sparse_skip.shape
        # t_sparse/t_dense may be (T,) shared or (B, T) per-sample
        T_dense = t_dense.shape[-1]
        C2 = 2 * C
        N_pos = B * H * W  # independent attention sequences
        d_model = self.d_model

        # Linear interpolation residual baseline
        linear_dense = self._linear_interp_temporal(sparse_skip, T_dense)

        # Reshape to per-position sequences (batch-major):
        # (B, T_sparse, 2, C, H, W) → (B, H, W, T_sparse, 2C) → (N_pos, T_sparse, 2C)
        sparse_flat = sparse_skip.permute(0, 4, 5, 1, 2, 3).reshape(
            N_pos, T_sparse, C2,
        )

        # Project to d_model
        kv = self.kv_proj(sparse_flat)  # (N_pos, T_sparse, d_model)

        if t_sparse.dim() == 1:
            # Shared time arrays: (T,) → broadcast across all positions
            sparse_pos = self.time_proj(self.time_enc(t_sparse))  # (T_sparse, d_model)
            dense_pos  = self.time_proj(self.time_enc(t_dense))   # (T_dense, d_model)
            kv    = kv    + sparse_pos.unsqueeze(0)  # (N_pos, T_sparse, d_model)
            query = dense_pos.unsqueeze(0).expand(N_pos, -1, -1).contiguous()  # (N_pos, T_dense, d_model)
        else:
            # Per-sample time arrays: (B, T) — expand to (B*H*W, T, d_model)
            sparse_pos = self.time_proj(
                self.time_enc(t_sparse.reshape(-1))
            ).reshape(B, T_sparse, d_model)  # (B, T_sparse, d_model)
            dense_pos = self.time_proj(
                self.time_enc(t_dense.reshape(-1))
            ).reshape(B, T_dense, d_model)   # (B, T_dense, d_model)
            # Expand: (B, 1, T, d) → (B, H*W, T, d) → (N_pos, T, d)
            kv    = kv    + sparse_pos.unsqueeze(1).expand(B, H * W, T_sparse, d_model).reshape(N_pos, T_sparse, d_model)
            query = dense_pos.unsqueeze(1).expand(B, H * W, T_dense, d_model).reshape(N_pos, T_dense, d_model).contiguous()

        # Cross-attention with gradient checkpointing
        attn_out = checkpoint.checkpoint(
            self._attention_forward,
            query,
            kv,
            use_reentrant=False,
        )  # (N_pos, T_dense, d_model)

        # Project back to feature space
        attn_out = self.output_proj(attn_out)  # (N_pos, T_dense, 2C)

        # Reshape: (N_pos, T_dense, 2C) → (B, H, W, T_dense, 2, C) → (B, T_dense, 2, C, H, W)
        attn_dense = attn_out.reshape(B, H, W, T_dense, 2, C).permute(
            0, 3, 4, 5, 1, 2,
        )

        # Residual sum: linear baseline + learned refinement
        return linear_dense + attn_dense


# =============================================================================
# 4. Spatiotemporal Attention Bottleneck
# =============================================================================

class SpatioTemporalAttentionBottleneck(nn.Module):
    """
    Full bottleneck: Self-Attention + Cross-Attention + PhysicsRefinement.

    Drop-in replacement for InputGuidedTemporalBottleneck.

    Parameters
    ----------
    encoder_channels : int
        Complex channel count at bottleneck (256).
    upsampling_factor : int
        Temporal upsampling factor (12).
    d_model : int
        Attention model dimension.
    n_heads : int
        Number of attention heads.
    n_self_layers : int
        Number of self-attention encoder layers.
    n_freq : int
        Number of sinusoidal frequency bands.
    n_refine_iters : int
        Number of physics refinement iterations.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        encoder_channels: int = 256,
        upsampling_factor: int = 12,
        d_model: int = 256,
        n_heads: int = 8,
        n_self_layers: int = 2,
        n_freq: int = 32,
        n_refine_iters: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.upsampling_factor = upsampling_factor
        self.in_channels = 2 * encoder_channels  # flattened complex

        # Positional encoding
        self.pos_enc = SpatioTemporalPositionalEncoding(
            d_model=d_model,
            n_freq=n_freq,
            max_spatial=128,  # generous upper bound for H_b * W_b
        )

        # Stage 1: Sparse self-attention
        self.self_attn = SpatioTemporalSelfAttention(
            in_channels=self.in_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_self_layers,
            dropout=dropout,
        )

        # Stage 2: Sparse-to-dense cross-attention
        self.cross_attn = SparseToDenseCrossAttention(
            in_channels=self.in_channels,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Stage 3: Physics refinement
        # residual_scale bumped from default 0.1 -> 0.5 so the refinement
        # branch isn't crippled by another tiny gate (lesson from v6 plateau).
        self.physics_refine = PhysicsRefinementBlock(
            channels=encoder_channels,
            n_iterations=n_refine_iters,
            residual_scale=0.5,
        )

    def _linear_interpolate(self, x: torch.Tensor, T_dense: int) -> torch.Tensor:
        """
        Temporal linear interpolation of complex features.

        Parameters
        ----------
        x : Tensor (B, T_sparse, 2, C, H, W)
        T_dense : int

        Returns
        -------
        Tensor (B, T_dense, 2, C, H, W)
        """
        B, T_sp, _, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 2 * C, T_sp, H * W)
        interp = F.interpolate(
            flat, size=(T_dense, H * W), mode="bilinear", align_corners=True,
        )
        interp = interp.reshape(B, 2, C, T_dense, H, W).permute(0, 3, 1, 2, 4, 5)
        return interp

    def forward(
        self,
        sparse_features: torch.Tensor,
        dt_seconds: Union[float, torch.Tensor] = 0.002,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sparse_features : Tensor (B, T_sparse, 2, C, H, W)
        dt_seconds : float or Tensor
            Time between sparse input frames (seconds).

        Returns
        -------
        Tensor (B, T_dense, 2, C, H, W)
        """
        B, T_sparse, _, C, H, W = sparse_features.shape
        device = sparse_features.device
        n_spatial = H * W

        # Resolve dt to a per-sample vector (B,) — fixes the batch-averaging bug
        # where mixed-step batches produced a wrong averaged dt scalar.
        dt_vec = _dt_to_vector(dt_seconds, B, device)  # (B,)
        dt_out_vec = dt_vec / self.upsampling_factor   # (B,)

        T_dense = 1 + (T_sparse - 1) * self.upsampling_factor

        # Build per-sample time arrays: (B, T) — t[b, i] = i * dt_vec[b]
        t_arange_sp = torch.arange(T_sparse, device=device, dtype=torch.float32)
        t_arange_de = torch.arange(T_dense, device=device, dtype=torch.float32)
        t_sparse = t_arange_sp.unsqueeze(0) * dt_vec.unsqueeze(1)    # (B, T_sparse)
        t_dense  = t_arange_de.unsqueeze(0) * dt_out_vec.unsqueeze(1)  # (B, T_dense)

        # Flatten complex: (B, T, 2, C, H, W) → (B, T, 2*C, H, W)
        x_flat = sparse_features.reshape(B, T_sparse, 2 * C, H, W)
        # → (B, T*H*W, 2*C) tokens
        x_tokens = x_flat.permute(0, 1, 3, 4, 2).reshape(B, T_sparse * n_spatial, 2 * C)

        # Positional encodings (batched — (B, T*n_spatial, d_model))
        sparse_pos = self.pos_enc(t_sparse, n_spatial)  # (B, T_sparse*n_spatial, d_model)
        dense_pos  = self.pos_enc(t_dense,  n_spatial)  # (B, T_dense*n_spatial,  d_model)

        # ---- Stage 1: Self-attention ----
        x_ctx = self.self_attn(x_tokens, sparse_pos)  # (B, T_sparse*n_spatial, 2*C)

        # ---- Stage 2: Cross-attention ----
        dense_tokens = self.cross_attn(
            x_ctx, dense_pos, sparse_pos,
        )  # (B, T_dense*n_spatial, 2*C)

        # Reshape to spatial: (B, T_dense, 2*C, H, W)
        dense_spatial = dense_tokens.reshape(B, T_dense, H, W, 2 * C).permute(0, 1, 4, 2, 3)
        # → (B, T_dense, 2, C, H, W)
        dense_complex = dense_spatial.reshape(B, T_dense, 2, C, H, W)

        # Linear interpolation residual — dt-invariant, unchanged by the per-sample fix.
        linear_interp = self._linear_interpolate(sparse_features, T_dense)

        # Full residual sum — no scalar gate. Gradients through the
        # attention path are unimpeded so the optimizer can grow it freely.
        dense_features = linear_interp + dense_complex

        # ---- Stage 3: Physics refinement with per-sample dt ----
        dense_features = self.physics_refine(dense_features, dt_out_vec)

        return dense_features


# =============================================================================
# 5. WavePINN v6 Main Model
# =============================================================================

class WavePINNv6(nn.Module):
    """
    WavePINN v6: Spatiotemporal Attention UNet.

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
        Attention model dimension.
    n_heads : int
        Attention head count.
    n_self_layers : int
        Number of self-attention encoder layers.
    n_freq : int
        Number of sinusoidal frequency bands.
    n_refine_iters : int
        Physics refinement iterations.
    dropout : float
        Dropout rate for attention.
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        if encoder_depths is None:
            encoder_depths = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.encoder_depths = encoder_depths
        self.upsampling_factor = upsampling_factor
        self.complex_to_real_mode = complex_to_real_mode
        self.use_attention_skips = use_attention_skips

        # ----- Complex encoder -----
        self.enc1 = ComplexDownBlock(in_channels, encoder_depths[0], use_naive_bn=use_naive_bn)
        self.enc2 = ComplexDownBlock(encoder_depths[0], encoder_depths[1], use_naive_bn=use_naive_bn)
        self.enc3 = ComplexDownBlock(encoder_depths[1], encoder_depths[2], use_naive_bn=use_naive_bn)
        self.enc4 = ComplexDownBlock(encoder_depths[2], encoder_depths[3], use_naive_bn=use_naive_bn)

        bottleneck_ch = encoder_depths[3]

        # ----- Temporal bottleneck -----
        self.bottleneck = SpatioTemporalAttentionBottleneck(
            encoder_channels=bottleneck_ch,
            upsampling_factor=upsampling_factor,
            d_model=d_model,
            n_heads=n_heads,
            n_self_layers=n_self_layers,
            n_freq=n_freq,
            n_refine_iters=n_refine_iters,
            dropout=dropout,
        )

        # ----- Per-position temporal attention for deep skips (v6 Option B) -----
        # Replaces linear interpolation on skip3 + skip4 with learned attention.
        # skip1/skip2 stay linear (too many positions to be cheap).
        if use_attention_skips:
            self.skip4_temporal_attn = TemporalCrossAttentionSkip(
                encoder_channels=encoder_depths[3],
                d_model=skip_attn_d_model,
                n_heads=skip_attn_heads,
                n_freq=skip_attn_n_freq,
                dropout=skip_attn_dropout,
            )
            self.skip3_temporal_attn = TemporalCrossAttentionSkip(
                encoder_channels=encoder_depths[2],
                d_model=skip_attn_d_model,
                n_heads=skip_attn_heads,
                n_freq=skip_attn_n_freq,
                dropout=skip_attn_dropout,
            )
        else:
            self.skip4_temporal_attn = None
            self.skip3_temporal_attn = None

        # ----- Complex → Real bridges -----
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
            f"WavePINNv6 initialised:\n"
            f"  Encoder depths      : {encoder_depths}\n"
            f"  Bottleneck real ch  : {bottleneck_real_ch}\n"
            f"  Skip real channels  : {skip_real_ch}\n"
            f"  Upsampling factor   : {upsampling_factor}\n"
            f"  d_model / n_heads   : {d_model} / {n_heads}\n"
            f"  Self-attn layers    : {n_self_layers}\n"
            f"  Refine iterations   : {n_refine_iters}\n"
            f"  Dropout             : {dropout}\n"
            f"  Complex->Real mode  : {complex_to_real_mode}\n"
            f"  Attention skips     : {use_attention_skips}"
            + (
                f" (d={skip_attn_d_model}, h={skip_attn_heads})"
                if use_attention_skips else ""
            )
        )

    # ------------------------------------------------------------------
    # Static helpers (same as v3)
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_for_encoder(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B, 2, T, H, W) → (B*T, 2, 1, H, W)."""
        B, _, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, 2, 1, H, W)
        return x, B, T

    @staticmethod
    def _reshape_skip_for_temporal(skip: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """(B*T, 2, C, H, W) → (B, T, 2, C, H, W)."""
        _, _, C, H, W = skip.shape
        return skip.reshape(B, T, 2, C, H, W)

    @staticmethod
    def _interpolate_skip_temporal(skip: torch.Tensor, target_T: int) -> torch.Tensor:
        """Temporal-only linear interpolation of complex skip connections."""
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
        """(B, T, C, H, W) → (B*T, C, H, W)."""
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
        Parameters
        ----------
        x : Tensor
            Sparse IQ input, shape (B, 2, T_sparse, H, W).
        dt_seconds : float or Tensor
            Time between sparse input frames (seconds).

        Returns
        -------
        velocity : Tensor
            Dense velocity maps (B, 1, T_dense, H, W) in [-1, 1].
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

        # Build per-sample time arrays for skip-attention temporal queries.
        # Uses _dt_to_vector so mixed-step batches get correct per-sample times.
        device = x.device
        dt_vec     = _dt_to_vector(dt_seconds, B, device)   # (B,)
        dt_out_vec = dt_vec / self.upsampling_factor          # (B,)
        t_sparse_arr = torch.arange(T_sparse, device=device, dtype=torch.float32).unsqueeze(0) * dt_vec.unsqueeze(1)    # (B, T_sparse)
        t_dense_arr  = torch.arange(T_dense,  device=device, dtype=torch.float32).unsqueeze(0) * dt_out_vec.unsqueeze(1)  # (B, T_dense)

        # ===================== BRIDGE: bottleneck complex → real ==========
        x_real = self.bottleneck_bridge(x_dense_complex)
        # (B, T_dense, C_real, H_b, W_b)

        # ===================== DECODER ====================================

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
        velocity = self.output_activation(self.output_conv(x_dec))  # (B*T_dense, 1, H, W)

        # Reshape: (B*T_dense, 1, H, W) → (B, 1, T_dense, H, W)
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        return velocity


# =============================================================================
# 6. Factory + Utilities
# =============================================================================

def create_wavepinn_v6(
    config: Optional[Dict] = None,
    **kwargs,
) -> WavePINNv6:
    """Create and return a configured WavePINNv6."""
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
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return WavePINNv6(**defaults)


def count_parameters_v6(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }
