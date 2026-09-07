"""
Physics-Informed Complex-to-Real UNet for Velocity Estimation.

Adapts the Input-Guided Motion-Aware PI-CV-RecUNet architecture to predict
dense particle-velocity maps from sparse complex IQ data.

Key Design Principles:
    1. **Complex-valued encoder** preserves IQ phase/magnitude throughout,
       honouring the intrinsic complex nature of the radio-frequency signal.
    2. **ConvLSTM temporal bottleneck** captures wave propagation dynamics
       across the sparse time frames before temporal super-resolution.
    3. **Input-guided flow estimation** computes optical flow at the original
       high-resolution IQ frames (not abstracted bottleneck features), then
       down-samples and scales the flow to the bottleneck grid.  This ensures
       sub-pixel motion is captured faithfully.
    4. **Adaptive fusion** of linear-interpolated and motion-warped bottleneck
       features lets the network fall back to a safe baseline in regions with
       little motion while leveraging physics-based warping where shear waves
       propagate.
    5. **Real-valued decoder with skip bridges** receives temporally-dense,
       bridged features and produces a single-channel velocity map per frame,
       activated with Tanh (output in [-1, 1] after normalisation).
    6. **Optional PRF (FiLM) conditioning** modulates each decoder level so
       a single model generalises across pulse-repetition frequencies
       (500 Hz – 20 kHz).

Architecture Overview:
    Input:  (B, 2, T_sparse, H, W) — Real + Imaginary IQ channels

    Encoder (Complex-valued):
        Level 1: C_in → 32   (skip₁)  → AvgPool
        Level 2: 32   → 64   (skip₂)  → AvgPool
        Level 3: 64   → 128  (skip₃)  → AvgPool
        Level 4: 128  → 256  (skip₄)  → AvgPool

    Bottleneck (Complex-valued):
        ConvLSTM temporal modelling  (256 → 128 hidden → 256)
        Input-guided motion interpolation  T_sparse → T_dense

    Complex-to-Real Bridge:
        (B, T_dense, 2, C, H_b, W_b) → (B, T_dense, 2*C, H_b, W_b)

    Decoder (Real-valued):
        Level 4: 512 + skip₄_real(512)  → 256   + FiLM
        Level 3: 256 + skip₃_real(256)  → 128   + FiLM
        Level 2: 128 + skip₂_real(128)  →  64   + FiLM
        Level 1:  64 + skip₁_real( 64)  →  32   + FiLM

    Output Head:
        Conv2d(32 → 1) + Tanh  → (B, 1, T_dense, H, W)

Data contract (IQVelocityDataset):
    input :  (2, T_sparse, H_pad, W_pad)  float32  [real, imag]
    target:  (1, T_vel,    H_pad, W_pad)  float32  normalised velocity in [-1,1]
    mask  :  (1, 1,        H_pad, W_pad)  float32  ROI mask
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.complex_modules import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    NaiveComplexBatchNorm2d,
    CReLU,
)
from src.architectures.input_guided_motion import (
    InputGuidedTemporalBottleneck,
    InputGuidedMotionInterpolation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Complex Encoder Blocks
# =============================================================================

class ComplexConvBlock(nn.Module):
    """
    Double complex-convolution block with batch-norm and CReLU.

    Structure:
        ComplexConv → ComplexBN → CReLU → ComplexConv → ComplexBN → CReLU

    Input / Output shape: (B, 2, C_in/C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_naive_bn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        BN = NaiveComplexBatchNorm2d if use_naive_bn else ComplexBatchNorm2d

        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = BN(out_channels)
        self.act1 = CReLU()

        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = BN(out_channels)
        self.act2 = CReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class ComplexDownBlock(nn.Module):
    """
    Complex encoder level: ComplexConvBlock + complex average-pooling (2×2).

    Returns both the pre-pooling skip features and the down-sampled output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_naive_bn: bool = True,
    ):
        super().__init__()
        self.conv_block = ComplexConvBlock(
            in_channels, out_channels, kernel_size, use_naive_bn
        )

    @staticmethod
    def _complex_avg_pool(x: torch.Tensor) -> torch.Tensor:
        """Average-pool real and imaginary parts independently (2×2, stride 2)."""
        real = F.avg_pool2d(x[:, 0], kernel_size=2, stride=2)
        imag = F.avg_pool2d(x[:, 1], kernel_size=2, stride=2)
        return torch.stack([real, imag], dim=1)

    def forward(
        self, x: torch.Tensor, return_skip: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.conv_block(x)
        pooled = self._complex_avg_pool(features)
        if return_skip:
            return pooled, features
        return pooled


# =============================================================================
# Complex-to-Real Bridge
# =============================================================================

class ComplexToRealBridge(nn.Module):
    """
    Convert complex features to real-valued features for the decoder.

    Modes:
        'concat'          — cat(real, imag)          → 2·C channels
        'magnitude_phase' — cat(|z|, ∠z)             → 2·C channels
        'magnitude'       — |z|                      → C channels

    Input:  (..., 2, C, H, W)      (last 4 dims)
    Output: (..., C_out, H, W)
    """

    def __init__(self, in_channels: int, mode: str = 'magnitude_phase', eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.mode = mode
        self.eps = eps

        if mode == 'concat':
            self.out_channels = 2 * in_channels
        elif mode == 'magnitude_phase':
            self.out_channels = 2 * in_channels
        elif mode == 'magnitude':
            self.out_channels = in_channels
        else:
            raise ValueError(f"Unknown bridge mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_time = x.dim() == 6  # (B, T, 2, C, H, W)
        if has_time:
            real = x[:, :, 0]
            imag = x[:, :, 1]
        else:
            real = x[:, 0]
            imag = x[:, 1]

        if self.mode == 'concat':
            return torch.cat([real, imag], dim=-3)
        elif self.mode == 'magnitude_phase':
            mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
            phase = torch.atan2(imag, real)
            return torch.cat([mag, phase], dim=-3)
        else:  # magnitude
            return torch.sqrt(real ** 2 + imag ** 2 + self.eps)


# =============================================================================
# Real Decoder Blocks
# =============================================================================

class RealConvBlock(nn.Module):
    """Conv → BN → ReLU × 2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RealUpBlock(nn.Module):
    """
    Decoder level: bilinear upsample → concat(skip) → RealConvBlock.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv_block = RealConvBlock(in_channels + skip_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# =============================================================================
# PRF FiLM Conditioning
# =============================================================================

class PRFConditioner(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) driven by the detection PRF.

    Encodes log₁₀(PRF / 10 kHz) through a small MLP and produces per-channel
    scale (γ) and shift (β) at each decoder level, allowing a single model to
    handle PRFs from 500 Hz to 20 kHz.

    Initialised to near-identity: γ ≈ 0 → (1 + γ) ≈ 1, β = 0.
    """

    def __init__(self, channels_per_level: List[int], hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.film_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * c) for c in channels_per_level
        ])
        for layer in self.film_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, log_prf: torch.Tensor, level_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if log_prf.dim() == 1:
            log_prf = log_prf.unsqueeze(-1)
        h = self.embed(log_prf)
        params = self.film_layers[level_idx](h)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


# =============================================================================
# Simple (non-motion) Temporal Bottleneck
# =============================================================================

class SimpleTemporalBottleneck(nn.Module):
    """
    Linear temporal interpolation fallback (no flow / LSTM).

    Upsamples along the time axis using bilinear interpolation on a
    flattened (B, 2·C, T, H·W) tensor so the spatial axes stay intact.
    """

    def __init__(self, channels: int, upsampling_factor: int = 12):
        super().__init__()
        self.channels = channels
        self.upsampling_factor = upsampling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_sp, _, C, H, W = x.shape
        T_dense = 1 + (T_sp - 1) * self.upsampling_factor
        flat = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 2 * C, T_sp, H * W)
        interp = F.interpolate(flat, size=(T_dense, H * W), mode='bilinear', align_corners=True)
        interp = interp.reshape(B, 2, C, T_dense, H, W).permute(0, 3, 1, 2, 4, 5)
        return interp


# =============================================================================
# Main Model
# =============================================================================

class ComplexToRealUNet(nn.Module):
    """
    Physics-informed Complex-to-Real UNet for IQ → Velocity prediction.

    See module docstring for full architecture diagram.

    Parameters
    ----------
    in_channels : int
        Number of input channels per complex component (typically 1 for IQ).
    base_channels : int
        Base filter count (unused — encoder_depths is authoritative).
    encoder_depths : list[int]
        Channel widths at each encoder level, e.g. [32, 64, 128, 256].
    upsampling_factor : int
        Temporal super-resolution factor. T_dense = 1 + (T_sparse - 1) * factor.
    use_motion_interpolation : bool
        If True, use ConvLSTM + input-guided flow; otherwise linear interpolation.
    complex_to_real_mode : str
        Bridge conversion mode ('concat' | 'magnitude_phase' | 'magnitude').
    use_naive_bn : bool
        Use per-component BN (True) or full complex BN (False).
    flow_hidden_channels : int
        Hidden width of the high-res flow estimator.
    flow_scale : float
        Conservative scaling of initial flow predictions.
    use_prf_conditioning : bool
        Enable FiLM conditioning on detection PRF.
    prf_hidden_dim : int
        Hidden dimension of the PRF MLP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        encoder_depths: List[int] = [32, 64, 128, 256],
        upsampling_factor: int = 12,
        use_motion_interpolation: bool = True,
        complex_to_real_mode: str = 'concat',
        use_naive_bn: bool = True,
        flow_hidden_channels: int = 32,
        flow_scale: float = 0.1,
        use_prf_conditioning: bool = False,
        prf_hidden_dim: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.upsampling_factor = upsampling_factor
        self.use_motion_interpolation = use_motion_interpolation
        self.encoder_depths = encoder_depths
        self.complex_to_real_mode = complex_to_real_mode
        self.use_prf_conditioning = use_prf_conditioning

        # ----- Complex encoder -----
        self.enc1 = ComplexDownBlock(in_channels, encoder_depths[0], use_naive_bn=use_naive_bn)
        self.enc2 = ComplexDownBlock(encoder_depths[0], encoder_depths[1], use_naive_bn=use_naive_bn)
        self.enc3 = ComplexDownBlock(encoder_depths[1], encoder_depths[2], use_naive_bn=use_naive_bn)
        self.enc4 = ComplexDownBlock(encoder_depths[2], encoder_depths[3], use_naive_bn=use_naive_bn)

        bottleneck_ch = encoder_depths[3]

        # ----- Temporal bottleneck -----
        if use_motion_interpolation:
            self.bottleneck = InputGuidedTemporalBottleneck(
                encoder_channels=bottleneck_ch,
                hidden_channels=bottleneck_ch // 2,
                input_channels=in_channels,
                num_layers=1,
                kernel_size=3,
                upsampling_factor=upsampling_factor,
                flow_hidden_channels=flow_hidden_channels,
                flow_scale=flow_scale,
            )
        else:
            self.bottleneck = SimpleTemporalBottleneck(
                channels=bottleneck_ch,
                upsampling_factor=upsampling_factor,
            )

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

        # ----- Real decoder -----
        self.dec4 = RealUpBlock(bottleneck_real_ch, skip_real_ch[3], encoder_depths[3])
        self.dec3 = RealUpBlock(encoder_depths[3], skip_real_ch[2], encoder_depths[2])
        self.dec2 = RealUpBlock(encoder_depths[2], skip_real_ch[1], encoder_depths[1])
        self.dec1 = RealUpBlock(encoder_depths[1], skip_real_ch[0], encoder_depths[0])

        # ----- Output head -----
        self.output_conv = nn.Conv2d(encoder_depths[0], 1, kernel_size=1)
        self.output_activation = nn.Tanh()
        # Keep default Kaiming weight init; zero bias only.
        nn.init.zeros_(self.output_conv.bias)

        # ----- PRF FiLM conditioning (optional) -----
        if use_prf_conditioning:
            decoder_channels = list(reversed(encoder_depths))  # [256, 128, 64, 32]
            self.prf_conditioner = PRFConditioner(decoder_channels, prf_hidden_dim)
        else:
            self.prf_conditioner = None

        logger.info(
            f"ComplexToRealUNet initialised:\n"
            f"  Encoder depths     : {encoder_depths}\n"
            f"  Bottleneck real ch : {bottleneck_real_ch}\n"
            f"  Skip real channels : {skip_real_ch}\n"
            f"  Upsampling factor  : {upsampling_factor}\n"
            f"  Motion interpolation: {use_motion_interpolation}\n"
            f"  Complex→Real mode  : {complex_to_real_mode}\n"
            f"  PRF conditioning   : {use_prf_conditioning}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
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
        """
        Temporal-only linear interpolation of complex skip connections.

        Reshapes to (B, 2·C, T, H·W) and interpolates along the T axis
        with align_corners=True.  Because H·W stays constant the bilinear
        kernel degenerates to 1-D temporal interpolation with zero spatial
        cross-talk.
        """
        B, T, _, C, H, W = skip.shape
        if T == target_T:
            return skip
        flat = skip.permute(0, 2, 3, 1, 4, 5).reshape(B, 2 * C, T, H * W)
        interp = F.interpolate(flat, size=(target_T, H * W), mode='bilinear', align_corners=True)
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
        high_res_inputs: Optional[torch.Tensor] = None,
        log_prf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Sparse IQ input, shape ``(B, 2, T_sparse, H, W)``.
            Dim-1 holds [Real, Imaginary].
        high_res_inputs : Tensor, optional
            High-resolution frames for flow estimation,
            ``(B, T_sparse, 2, C_in, H, W)``.  Derived from *x* when None.
        log_prf : Tensor, optional
            Normalised PRF ``(B,)`` — log₁₀(prf_hz / 10 000).
            Required when ``use_prf_conditioning=True``.

        Returns
        -------
        velocity : Tensor
            Dense velocity maps ``(B, 1, T_dense, H, W)`` in [-1, 1].
        """
        B, _, T_sparse, H, W = x.shape

        # Derive high-res inputs for flow estimator if not provided
        if high_res_inputs is None and self.use_motion_interpolation:
            # x is (B, 2, T, H, W) → (B, T, 2, 1, H, W)
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
        # (B, T_dense, C_real, H_b, W_b)

        # Expand log_prf for frame-level FiLM application
        log_prf_exp = None
        if self.prf_conditioner is not None and log_prf is not None:
            log_prf_exp = log_prf.unsqueeze(1).expand(B, T_dense).reshape(B * T_dense)

        # ====================== DECODER ===================================

        # --- Level 4 ---
        skip4_c = complex_skips.pop()
        skip4_c = self._reshape_skip_for_temporal(skip4_c, B, T_sparse)
        skip4_c = self._interpolate_skip_temporal(skip4_c, T_dense)
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
        skip3_r = self.skip_bridge_l3(skip3_c)
        skip3_flat = self._flatten_for_decoder(skip3_r, B, T_dense)
        x_dec = self.dec3(x_dec, skip3_flat)

        if self.prf_conditioner is not None and log_prf_exp is not None:
            g, b = self.prf_conditioner(log_prf_exp, level_idx=1)
            x_dec = x_dec * (1.0 + g) + b

        # --- Level 2 ---
        skip2_c = complex_skips.pop()
        skip2_c = self._reshape_skip_for_temporal(skip2_c, B, T_sparse)
        skip2_c = self._interpolate_skip_temporal(skip2_c, T_dense)
        skip2_r = self.skip_bridge_l2(skip2_c)
        skip2_flat = self._flatten_for_decoder(skip2_r, B, T_dense)
        x_dec = self.dec2(x_dec, skip2_flat)

        if self.prf_conditioner is not None and log_prf_exp is not None:
            g, b = self.prf_conditioner(log_prf_exp, level_idx=2)
            x_dec = x_dec * (1.0 + g) + b

        # --- Level 1 ---
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
        velocity = self.output_activation(self.output_conv(x_dec))  # (B*T_dense, 1, H, W)

        # Reshape: (B*T_dense, 1, H, W) → (B, 1, T_dense, H, W)
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        return velocity

    # ------------------------------------------------------------------
    # Utility accessors
    # ------------------------------------------------------------------

    def get_flow_stats(self) -> Dict:
        """Return flow statistics from the last forward pass (for logging)."""
        if hasattr(self.bottleneck, 'get_flow_stats'):
            return self.bottleneck.get_flow_stats()
        return {}

    def get_motion_interpolation(self):
        """Return the motion interpolation sub-module (for flow losses)."""
        if hasattr(self.bottleneck, 'get_motion_interpolation'):
            return self.bottleneck.get_motion_interpolation()
        return None


# =============================================================================
# Factory
# =============================================================================

def create_velocity_model(
    config: Optional[Dict] = None,
    **kwargs,
) -> ComplexToRealUNet:
    """
    Create and return a configured ``ComplexToRealUNet``.

    Parameters
    ----------
    config : dict, optional
        Model configuration dictionary.
    **kwargs
        Override individual hyper-parameters.
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
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return ComplexToRealUNet(**defaults)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}
