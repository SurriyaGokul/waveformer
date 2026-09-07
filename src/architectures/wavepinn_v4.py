"""
WavePINN v4: Conditional Wave Propagation Network for SWE Temporal Super-Resolution.

Replaces v3's BiLSTM + Cross-Attention bottleneck with a bidirectional ConvGRU
wave propagator that GENERATES dense temporal sequences from sparse observations
using learned wave dynamics.

Key innovation: At low PRF (100 Hz), only 1-2 of 5 sparse frames contain wave
information. v3 tried to interpolate between frames — impossible when most frames
are empty background. v4 instead PROPAGATES from informative frames using a
learned physics prior acquired from high-PRF training data.

How it works:
    1. A WaveContentGate identifies which sparse frames contain wave information
       (learned per-frame scoring: 0=background, 1=wave present).
    2. A BidirectionalWavePropagator uses dt-conditioned ConvGRU cells to
       propagate features forward and backward through time. At sparse frame
       positions, observed features are injected (gated by wave content score).
       Between frames, the GRU generates new features from its hidden state
       using learned wave dynamics.
    3. Forward/backward propagations are fused via learned 1×1 convolution.

The learned prior:
    - Trained on high-PRF data (5-20 kHz) where ALL frames contain wave content,
      the ConvGRU learns shear wave propagation dynamics (speed, direction,
      attenuation, reflection).
    - At 100 Hz inference, the wave content gate identifies frame 0 as informative,
      and the forward GRU propagates from it using learned dynamics. The backward
      GRU (starting from background frame 4) contributes tissue structure only.
    - dt-conditioning via FiLM allows the propagator to adapt its step size to
      any PRF — the number of GRU steps is always (T_dense), only dt changes.

Architecture:
    Input:  (B, 2, T_sparse, H, W) + dt_seconds

    Encoder (complex-valued, reused from v2):
        ComplexDownBlock ×4: [32, 64, 128, 256] with AvgPool

    Bridge: ComplexToRealBridge (concat mode) → 512 channels

    Feature Projection: Conv2d(512, 256, 1) → 256 channels

    Bottleneck — BidirectionalWavePropagator:
        1. WaveContentGate: per-frame wave scoring
        2. Forward DtConvGRU: propagate 0 → T_dense-1
        3. Backward DtConvGRU: propagate T_dense-1 → 0
        4. Fusion: Conv2d(512, 256, 1)

    Decoder (real-valued):
        RealUpBlock ×4 with skip connections

    Output: Conv2d(32 → 1) + Tanh → (B, 1, T_dense, H, W) in [-1, 1]
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.complex_modules import ComplexConv2d
from src.architectures.velocity_unet import (
    ComplexConvBlock,
    ComplexDownBlock,
    ComplexToRealBridge,
    RealConvBlock,
    RealUpBlock,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. dt-Adaptive ConvGRU Cell
# =============================================================================

class DtAdaptiveConvGRUCell(nn.Module):
    """
    Convolutional GRU cell with dt-conditioned FiLM modulation.

    Standard ConvGRU gates are modulated by log10(dt) via FiLM (Feature-wise
    Linear Modulation), allowing the cell to adapt its propagation dynamics
    to different PRFs:
        - At small dt (high PRF): small updates (wave moves little per step)
        - At large dt (low PRF): larger updates (wave moves more per step)

    Parameters
    ----------
    channels : int
        Both input and hidden channel count (must match for self-feeding
        between sparse observations).
    kernel_size : int
        Spatial kernel size for gate convolutions.
    dt_film_hidden : int
        Hidden dimension for the dt FiLM MLP.
    """

    def __init__(
        self,
        channels: int = 256,
        kernel_size: int = 3,
        dt_film_hidden: int = 64,
    ):
        super().__init__()
        self.channels = channels
        pad = kernel_size // 2

        # Gate convolutions: input [x, h] → gates
        # x and h have same channel count
        self.reset_gate = nn.Conv2d(2 * channels, channels, kernel_size, padding=pad)
        self.update_gate = nn.Conv2d(2 * channels, channels, kernel_size, padding=pad)
        self.candidate = nn.Conv2d(2 * channels, channels, kernel_size, padding=pad)

        # dt FiLM: log10(dt) → per-gate modulation (gamma for each gate)
        self.dt_film = nn.Sequential(
            nn.Linear(1, dt_film_hidden),
            nn.GELU(),
            nn.Linear(dt_film_hidden, 3 * channels),
        )

        # Initialise FiLM to near-identity
        nn.init.normal_(self.dt_film[2].weight, std=0.01)
        nn.init.zeros_(self.dt_film[2].bias)

        # Initialise update gate bias positive (default: propagate, i.e. keep h)
        nn.init.constant_(self.update_gate.bias, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        log_dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single propagation step.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Input features. Zero tensor between sparse frames;
            observed features at sparse frame positions.
        h : Tensor (B, C, H, W)
            Previous hidden state (propagated wave field).
        log_dt : Tensor (1, 3*C, 1, 1)
            Pre-computed FiLM parameters from log10(dt_output).

        Returns
        -------
        h_new : Tensor (B, C, H, W)
        """
        C = self.channels
        gamma_r = log_dt[:, :C]
        gamma_u = log_dt[:, C:2*C]
        gamma_c = log_dt[:, 2*C:]

        combined = torch.cat([x, h], dim=1)  # (B, 2C, H, W)

        r = torch.sigmoid(self.reset_gate(combined) * (1.0 + gamma_r))
        z = torch.sigmoid(self.update_gate(combined) * (1.0 + gamma_u))

        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.candidate(combined_r) * (1.0 + gamma_c))

        h_new = (1.0 - z) * h + z * h_tilde
        return h_new


# =============================================================================
# 2. Wave Content Gate
# =============================================================================

class WaveContentGate(nn.Module):
    """
    Per-frame wave content scoring.

    Estimates whether each sparse frame contains meaningful wave information
    (score → 1) or is pure background (score → 0). Uses global average pooling
    + MLP + sigmoid.

    At 100 Hz: frame 0 ≈ 1.0, frames 1-4 ≈ 0.0
    At 20 kHz: all frames ≈ 1.0

    Parameters
    ----------
    channels : int
        Input feature channels.
    hidden : int
        MLP hidden dimension.
    """

    def __init__(self, channels: int = 256, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B*T, C, 1, 1)
            nn.Flatten(),             # (B*T, C)
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : Tensor (B, T, C, H, W)

        Returns
        -------
        scores : Tensor (B, T) in [0, 1]
        """
        B, T, C, H, W = features.shape
        flat = features.reshape(B * T, C, H, W)
        scores = self.gate(flat)  # (B*T, 1)
        return scores.reshape(B, T)


# =============================================================================
# 3. Bidirectional Wave Propagator
# =============================================================================

class BidirectionalWavePropagator(nn.Module):
    """
    Core temporal generation via bidirectional ConvGRU with observation anchoring.

    Forward GRU:  Propagates from first sparse frame toward last.
    Backward GRU: Propagates from last sparse frame toward first.

    At sparse frame positions, observed features are injected into the hidden
    state, weighted by the wave content gate. Between frames, the GRU propagates
    its own hidden state (generating new features from learned wave dynamics).

    The bidirectional outputs are fused via a learned 1×1 convolution, allowing
    the model to adaptively weight forward vs backward propagation based on
    which direction carries more useful information.

    Parameters
    ----------
    channels : int
        Feature channel count.
    kernel_size : int
        ConvGRU spatial kernel size.
    dt_film_hidden : int
        dt FiLM MLP hidden size.
    """

    def __init__(
        self,
        channels: int = 256,
        kernel_size: int = 3,
        dt_film_hidden: int = 64,
    ):
        super().__init__()
        self.channels = channels

        self.fwd_gru = DtAdaptiveConvGRUCell(channels, kernel_size, dt_film_hidden)
        self.bwd_gru = DtAdaptiveConvGRUCell(channels, kernel_size, dt_film_hidden)

        # Fusion: concat [fwd, bwd] → channels
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        # Temporal refinement: smooth along T after fusion
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
        )

    def _propagate(
        self,
        sparse_features: torch.Tensor,
        wave_scores: torch.Tensor,
        log_dt_film: torch.Tensor,
        gru_cell: DtAdaptiveConvGRUCell,
        upsampling_factor: int,
        forward: bool = True,
    ) -> torch.Tensor:
        """
        Single-direction propagation with observation anchoring.

        Parameters
        ----------
        sparse_features : (B, T_sparse, C, H, W)
        wave_scores : (B, T_sparse) in [0, 1]
        log_dt_film : (1, 3*C, 1, 1) pre-computed FiLM params
        gru_cell : DtAdaptiveConvGRUCell
        upsampling_factor : int
        forward : bool

        Returns
        -------
        outputs : (B, T_dense, C, H, W)
        """
        B, T_sp, C, H, W = sparse_features.shape
        T_dense = 1 + (T_sp - 1) * upsampling_factor
        device = sparse_features.device

        # Zero input for between-frame propagation
        zero_input = torch.zeros(B, C, H, W, device=device, dtype=sparse_features.dtype)

        outputs = []

        if forward:
            # Initialise with first frame (gated by wave score)
            score_0 = wave_scores[:, 0].reshape(B, 1, 1, 1)  # (B, 1, 1, 1)
            h = sparse_features[:, 0] * score_0
            outputs.append(h)

            for t in range(1, T_dense):
                sparse_idx = t // upsampling_factor
                local_pos = t % upsampling_factor

                if local_pos == 0 and sparse_idx < T_sp:
                    # At a sparse frame position: gate observed features by wave score.
                    # High score → inject observation; low score → treat as background.
                    obs = sparse_features[:, sparse_idx]
                    score = wave_scores[:, sparse_idx].reshape(B, 1, 1, 1)
                    gated_obs = score * obs + (1.0 - score) * zero_input
                    h = gru_cell(gated_obs, h, log_dt_film)
                else:
                    # Between frames: propagate with zero input (self-driven)
                    h = gru_cell(zero_input, h, log_dt_film)

                outputs.append(h)
        else:
            # Backward: reverse order
            score_last = wave_scores[:, -1].reshape(B, 1, 1, 1)
            h = sparse_features[:, -1] * score_last
            outputs.append(h)

            for t in range(T_dense - 2, -1, -1):
                sparse_idx = t // upsampling_factor
                local_pos = t % upsampling_factor

                if local_pos == 0 and sparse_idx < T_sp:
                    obs = sparse_features[:, sparse_idx]
                    score = wave_scores[:, sparse_idx].reshape(B, 1, 1, 1)
                    gated_obs = score * obs + (1.0 - score) * zero_input
                    h = gru_cell(gated_obs, h, log_dt_film)
                else:
                    h = gru_cell(zero_input, h, log_dt_film)

                outputs.append(h)

            outputs.reverse()

        return torch.stack(outputs, dim=1)  # (B, T_dense, C, H, W)

    def forward(
        self,
        sparse_features: torch.Tensor,
        wave_scores: torch.Tensor,
        dt_seconds: float,
        upsampling_factor: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sparse_features : (B, T_sparse, C, H, W)
        wave_scores : (B, T_sparse)
        dt_seconds : float
            Time between sparse input frames.
        upsampling_factor : int

        Returns
        -------
        dense_features : (B, T_dense, C, H, W)
        """
        B, T_sp, C, H, W = sparse_features.shape

        # Pre-compute dt FiLM parameters (shared across all steps)
        dt_output = dt_seconds / upsampling_factor
        log_dt = math.log10(max(dt_output, 1e-12))
        log_dt_tensor = sparse_features.new_tensor([[log_dt]])  # (1, 1)
        film_params = self.fwd_gru.dt_film(log_dt_tensor)  # (1, 3*C)
        log_dt_film = film_params.reshape(1, 3 * C, 1, 1)  # broadcast-ready

        # Also for backward (separate FiLM network)
        film_params_bwd = self.bwd_gru.dt_film(log_dt_tensor)
        log_dt_film_bwd = film_params_bwd.reshape(1, 3 * C, 1, 1)

        # Forward propagation
        fwd_out = self._propagate(
            sparse_features, wave_scores, log_dt_film,
            self.fwd_gru, upsampling_factor, forward=True,
        )

        # Backward propagation
        bwd_out = self._propagate(
            sparse_features, wave_scores, log_dt_film_bwd,
            self.bwd_gru, upsampling_factor, forward=False,
        )

        # Fuse forward + backward: (B, T_dense, 2C, H, W) → (B, T_dense, C, H, W)
        T_dense = fwd_out.shape[1]
        fused = torch.cat([fwd_out, bwd_out], dim=2)  # (B, T_dense, 2C, H, W)
        fused = fused.reshape(B * T_dense, 2 * C, H, W)
        fused = self.fusion(fused)  # (B*T_dense, C, H, W)
        fused = fused.reshape(B, T_dense, C, H, W)

        # Temporal refinement: Conv1d along T
        # (B, T_dense, C, H, W) → (B*H*W, C, T_dense)
        fused_t = fused.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, T_dense)
        fused_t = fused_t + self.temporal_refine(fused_t)  # residual
        fused = fused_t.reshape(B, H, W, C, T_dense).permute(0, 4, 3, 1, 2)

        return fused  # (B, T_dense, C, H, W)


# =============================================================================
# 4. WavePINN v4 Main Model
# =============================================================================

class WavePINNv4(nn.Module):
    """
    WavePINN v4: Conditional Wave Propagation Network.

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
    propagator_channels : int
        Channel count for the wave propagator ConvGRU (default: 256).
    propagator_kernel : int
        Spatial kernel size for ConvGRU gates (default: 3).
    dt_film_hidden : int
        Hidden dim for dt FiLM MLP (default: 64).
    wave_gate_hidden : int
        Hidden dim for wave content gate MLP (default: 64).
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_depths: Optional[List[int]] = None,
        upsampling_factor: int = 12,
        complex_to_real_mode: str = "concat",
        use_naive_bn: bool = True,
        propagator_channels: int = 256,
        propagator_kernel: int = 3,
        dt_film_hidden: int = 64,
        wave_gate_hidden: int = 64,
    ):
        super().__init__()
        if encoder_depths is None:
            encoder_depths = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.encoder_depths = encoder_depths
        self.upsampling_factor = upsampling_factor
        self.complex_to_real_mode = complex_to_real_mode
        self.propagator_channels = propagator_channels

        # ----- Complex encoder -----
        self.enc1 = ComplexDownBlock(in_channels, encoder_depths[0], use_naive_bn=use_naive_bn)
        self.enc2 = ComplexDownBlock(encoder_depths[0], encoder_depths[1], use_naive_bn=use_naive_bn)
        self.enc3 = ComplexDownBlock(encoder_depths[1], encoder_depths[2], use_naive_bn=use_naive_bn)
        self.enc4 = ComplexDownBlock(encoder_depths[2], encoder_depths[3], use_naive_bn=use_naive_bn)

        bottleneck_ch = encoder_depths[3]

        # ----- Complex → Real bridge (at bottleneck) -----
        self.bottleneck_bridge = ComplexToRealBridge(bottleneck_ch, mode=complex_to_real_mode)
        bottleneck_real_ch = self.bottleneck_bridge.out_channels  # 512 for concat

        # ----- Feature projection: 512 → propagator_channels -----
        self.pre_proj = nn.Sequential(
            nn.Conv2d(bottleneck_real_ch, propagator_channels, kernel_size=1),
            nn.GELU(),
        )

        # ----- Wave content gate -----
        self.wave_gate = WaveContentGate(
            channels=propagator_channels,
            hidden=wave_gate_hidden,
        )

        # ----- Bidirectional wave propagator -----
        self.propagator = BidirectionalWavePropagator(
            channels=propagator_channels,
            kernel_size=propagator_kernel,
            dt_film_hidden=dt_film_hidden,
        )

        # ----- Skip bridges (complex → real) -----
        self.skip_bridge_l1 = ComplexToRealBridge(encoder_depths[0], mode=complex_to_real_mode)
        self.skip_bridge_l2 = ComplexToRealBridge(encoder_depths[1], mode=complex_to_real_mode)
        self.skip_bridge_l3 = ComplexToRealBridge(encoder_depths[2], mode=complex_to_real_mode)
        self.skip_bridge_l4 = ComplexToRealBridge(encoder_depths[3], mode=complex_to_real_mode)

        skip_real_ch = [
            self.skip_bridge_l1.out_channels,
            self.skip_bridge_l2.out_channels,
            self.skip_bridge_l3.out_channels,
            self.skip_bridge_l4.out_channels,
        ]

        # ----- Real decoder -----
        # Input to dec4 is propagator_channels (256), not bottleneck_real_ch (512)
        self.dec4 = RealUpBlock(propagator_channels, skip_real_ch[3], encoder_depths[3])
        self.dec3 = RealUpBlock(encoder_depths[3], skip_real_ch[2], encoder_depths[2])
        self.dec2 = RealUpBlock(encoder_depths[2], skip_real_ch[1], encoder_depths[1])
        self.dec1 = RealUpBlock(encoder_depths[1], skip_real_ch[0], encoder_depths[0])

        # ----- Output head -----
        self.output_conv = nn.Conv2d(encoder_depths[0], 1, kernel_size=1)
        self.output_activation = nn.Tanh()
        nn.init.zeros_(self.output_conv.bias)

        logger.info(
            f"WavePINNv4 initialised:\n"
            f"  Encoder depths       : {encoder_depths}\n"
            f"  Propagator channels  : {propagator_channels}\n"
            f"  Propagator kernel    : {propagator_kernel}\n"
            f"  Skip real channels   : {skip_real_ch}\n"
            f"  Upsampling factor    : {upsampling_factor}\n"
            f"  dt FiLM hidden       : {dt_film_hidden}\n"
            f"  Wave gate hidden     : {wave_gate_hidden}\n"
            f"  Complex→Real mode    : {complex_to_real_mode}"
        )

    # ------------------------------------------------------------------
    # Gate regularization
    # ------------------------------------------------------------------

    def gate_entropy_loss(self, wave_scores: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization to prevent gate collapse to all-0 or all-1.

        Encourages the gate to produce diverse scores across frames.
        Returns a loss that should be MINIMIZED (negative entropy).
        """
        # Per-sample mean score should be moderate (not 0, not 1)
        mean_score = wave_scores.mean(dim=1)  # (B,)
        # Encourage mean ∈ [0.2, 0.8] via soft penalty
        target_mean = 0.4  # expect ~40% of frames to have wave content
        mean_penalty = (mean_score - target_mean).pow(2).mean()

        # Binary entropy per element: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-6
        p = wave_scores.clamp(eps, 1 - eps)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

        # We want HIGH entropy (diverse scores), so return negative entropy + mean penalty
        return -0.1 * entropy + mean_penalty

    # ------------------------------------------------------------------
    # Static helpers
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
        """Temporal linear interpolation of complex skip connections."""
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
        return_gate_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Sparse IQ input, shape ``(B, 2, T_sparse, H, W)``.
        dt_seconds : float or Tensor
            Time between sparse input frames (seconds).
        return_gate_scores : bool
            If True, return (velocity, wave_scores) tuple.

        Returns
        -------
        velocity : Tensor
            Dense velocity maps ``(B, 1, T_dense, H, W)`` in [-1, 1].
        wave_scores : Tensor (optional)
            Gate scores ``(B, T_sparse)`` if return_gate_scores=True.
        """
        B, _, T_sparse, H, W = x.shape

        # Resolve dt to scalar
        if isinstance(dt_seconds, torch.Tensor):
            dt_scalar = dt_seconds.mean().item()
        else:
            dt_scalar = float(dt_seconds)

        T_dense = 1 + (T_sparse - 1) * self.upsampling_factor

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

        # ===================== BRIDGE + PROJECT ============================
        # (B*T, 2, C_b, H_b, W_b) → (B, T, 2, C_b, H_b, W_b)
        _, _, C_b, H_b, W_b = x_enc.shape
        x_temporal = x_enc.reshape(B, T_sparse, 2, C_b, H_b, W_b)

        # Complex → Real bridge: (B, T, 2, 256, H_b, W_b) → (B, T, 512, H_b, W_b)
        x_real = self.bottleneck_bridge(x_temporal)  # (B, T_sparse, 512, H_b, W_b)

        # Project to propagator channels: (B, T, 512, H_b, W_b) → (B, T, 256, H_b, W_b)
        BT = B * T_sparse
        x_proj = self.pre_proj(x_real.reshape(BT, -1, H_b, W_b))
        x_proj = x_proj.reshape(B, T_sparse, self.propagator_channels, H_b, W_b)

        # ===================== WAVE PROPAGATION ============================
        # Wave content gate: which frames have wave info?
        wave_scores = self.wave_gate(x_proj)  # (B, T_sparse)

        # Bidirectional propagation: (B, T_sparse, C, H_b, W_b) → (B, T_dense, C, H_b, W_b)
        x_dense = self.propagator(
            x_proj, wave_scores, dt_scalar, self.upsampling_factor,
        )

        # ===================== DECODER ====================================
        # --- Level 4 ---
        skip4_c = complex_skips.pop()
        skip4_c = self._reshape_skip_for_temporal(skip4_c, B, T_sparse)
        skip4_c = self._interpolate_skip_temporal(skip4_c, T_dense)
        skip4_r = self.skip_bridge_l4(skip4_c)  # (B, T_dense, 512, H4, W4)

        x_dec = self._flatten_for_decoder(x_dense, B, T_dense)
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
        velocity = velocity.reshape(B, T_dense, 1, H, W).permute(0, 2, 1, 3, 4)

        if return_gate_scores:
            return velocity, wave_scores
        return velocity


# =============================================================================
# 5. Factory + Utilities
# =============================================================================

def create_wavepinn_v4(
    config: Optional[Dict] = None,
    **kwargs,
) -> WavePINNv4:
    """Create and return a configured ``WavePINNv4``."""
    defaults = dict(
        in_channels=1,
        encoder_depths=[32, 64, 128, 256],
        upsampling_factor=12,
        complex_to_real_mode="concat",
        use_naive_bn=True,
        propagator_channels=256,
        propagator_kernel=3,
        dt_film_hidden=64,
        wave_gate_hidden=64,
    )
    if config is not None:
        defaults.update(config)
    defaults.update(kwargs)
    return WavePINNv4(**defaults)


def count_parameters_v4(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }
