"""
WavePINN v11: Deployment-focused hybrid of v9 + gated spectral bottleneck adapter.

Why this design
---------------
v9 is the current strongest backbone for 500 Hz -> virtual 6 kHz deployment.
Instead of replacing the temporal core with a pure FNO block (which previously
caused instability/mismatch), v11 adds a *bounded* spectral residual adapter at
v9's dense bottleneck output:

    x_dense = bottleneck(x_sparse, dt)
    x_spec  = FNO(x_dense)
    x_out   = x_dense + alpha * x_spec

where alpha is learnable but constrained to [0, spectral_max_gate].
With small alpha init, training starts near v9 behavior and only uses spectral
corrections if they improve the objective.

Outputs are identical to v9:
    (velocity, cs_scalar)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.architectures.complex_fno import ComplexSpectralRefinement
from src.architectures.wavepinn_v9 import WavePINNv9

logger = logging.getLogger(__name__)


class BottleneckSpectralAdapter(nn.Module):
    """
    Wrap a temporal bottleneck with a bounded spectral residual adapter.

    Parameters
    ----------
    base_bottleneck:
        Existing bottleneck module returning (B, T_dense, 2, C, H_b, W_b).
    channels:
        Bottleneck channel count C.
    num_modes_height, num_modes_width:
        Kept Fourier modes for spectral refinement.
    num_layers:
        Number of stacked FNO layers.
    use_batchnorm:
        Whether to use complex BN inside FNO layers.
    fft_padding:
        Optional FFT padding in FNO layers.
    dropout:
        Dropout on the spectral residual branch.
    gate_init:
        Initial effective gate value (small -> starts close to base model).
    max_gate:
        Upper bound for the effective gate.
    enabled:
        If False, adapter is a pass-through.
    """

    def __init__(
        self,
        base_bottleneck: nn.Module,
        channels: int,
        hidden_channels: int = 64,
        num_modes_height: int = 6,
        num_modes_width: int = 6,
        num_layers: int = 2,
        use_batchnorm: bool = True,
        fft_padding: int = 0,
        dropout: float = 0.05,
        gate_init: float = 0.02,
        max_gate: float = 0.35,
        enabled: bool = True,
    ):
        super().__init__()
        self.base_bottleneck = base_bottleneck
        self.enabled = bool(enabled)
        self.max_gate = float(max(max_gate, 1e-3))

        if self.enabled:
            self.spectral_block = ComplexSpectralRefinement(
                in_channels=channels,
                hidden_channels=hidden_channels,
                out_channels=channels,
                num_modes_height=num_modes_height,
                num_modes_width=num_modes_width,
                num_layers=num_layers,
                use_batchnorm=use_batchnorm,
                use_residual=True,
                fft_padding=fft_padding,
            )
            self.dropout = nn.Dropout(p=float(max(dropout, 0.0)))

            # Parameterize gate through a logit for smooth optimization.
            gate_init_clamped = min(max(float(gate_init), 1e-4), self.max_gate - 1e-4)
            ratio = gate_init_clamped / self.max_gate
            init_logit = math.log(ratio / (1.0 - ratio))
            self.gate_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

            logger.info(
                "BottleneckSpectralAdapter enabled: "
                f"hidden={hidden_channels}, modes=({num_modes_height},{num_modes_width}), "
                f"layers={num_layers}, "
                f"max_gate={self.max_gate:.3f}, gate_init={gate_init_clamped:.4f}"
            )
        else:
            self.spectral_block = None
            self.dropout = nn.Identity()
            self.gate_logit = None
            logger.info("BottleneckSpectralAdapter disabled (pass-through)")

    def current_gate(self) -> float:
        """Return current effective gate value as Python float."""
        if not self.enabled or self.gate_logit is None:
            return 0.0
        with torch.no_grad():
            g = self.max_gate * torch.sigmoid(self.gate_logit)
            return float(g.item())

    def forward(self, x_temporal: torch.Tensor, dt_seconds) -> torch.Tensor:
        x_dense = self.base_bottleneck(x_temporal, dt_seconds)
        if not self.enabled or self.spectral_block is None:
            return x_dense

        x_spec = self.spectral_block(x_dense)
        x_spec = self.dropout(x_spec)

        gate = self.max_gate * torch.sigmoid(self.gate_logit)
        # Keep output dtype aligned with the base branch for AMP stability.
        x_spec = x_spec.to(dtype=x_dense.dtype)
        out = x_dense + gate.to(dtype=x_dense.dtype) * x_spec
        return out


class WavePINNv11(WavePINNv9):
    """
    v11 model: v9 backbone + bounded spectral bottleneck adapter.

    v11 keeps v9's proven training/outputs and only augments bottleneck features
    with a controlled spectral residual branch for deployment robustness.
    """

    def __init__(
        self,
        use_spectral_calibrator: bool = True,
        spectral_hidden_channels: int = 64,
        spectral_num_modes_height: int = 6,
        spectral_num_modes_width: int = 6,
        spectral_num_layers: int = 2,
        spectral_use_batchnorm: bool = True,
        spectral_fft_padding: int = 0,
        spectral_dropout: float = 0.05,
        spectral_gate_init: float = 0.02,
        spectral_max_gate: float = 0.35,
        **kwargs,
    ):
        super().__init__(**kwargs)

        bottleneck_channels = self.encoder_depths[-1]
        base_bottleneck = self.bottleneck

        self.spectral_adapter = BottleneckSpectralAdapter(
            base_bottleneck=base_bottleneck,
            channels=bottleneck_channels,
            hidden_channels=spectral_hidden_channels,
            num_modes_height=spectral_num_modes_height,
            num_modes_width=spectral_num_modes_width,
            num_layers=spectral_num_layers,
            use_batchnorm=spectral_use_batchnorm,
            fft_padding=spectral_fft_padding,
            dropout=spectral_dropout,
            gate_init=spectral_gate_init,
            max_gate=spectral_max_gate,
            enabled=use_spectral_calibrator,
        )
        # Replace bottleneck path used in WavePINNv9.forward.
        self.bottleneck = self.spectral_adapter

    def spectral_gate(self) -> float:
        return self.spectral_adapter.current_gate()


def create_wavepinn_v11(config: Optional[Dict] = None, **kwargs) -> WavePINNv11:
    """Create WavePINNv11 from config dict + keyword overrides."""
    defaults = dict(
        # v9 defaults
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
        # v11 spectral adapter defaults
        use_spectral_calibrator=True,
        spectral_hidden_channels=64,
        spectral_num_modes_height=6,
        spectral_num_modes_width=6,
        spectral_num_layers=2,
        spectral_use_batchnorm=True,
        spectral_fft_padding=0,
        spectral_dropout=0.05,
        spectral_gate_init=0.02,
        spectral_max_gate=0.35,
    )
    if config:
        defaults.update(config)
    defaults.update(kwargs)
    # Allow human-readable metadata keys in config JSON without breaking init.
    defaults = {k: v for k, v in defaults.items() if not str(k).startswith("_")}
    return WavePINNv11(**defaults)


def count_parameters_v11(model: WavePINNv11) -> Dict[str, int]:
    """Return total/trainable and branch-wise parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cs_head = sum(p.numel() for p in model.cs_head.parameters())
    spectral = 0
    if hasattr(model, "spectral_adapter") and model.spectral_adapter.enabled:
        spectral = sum(p.numel() for p in model.spectral_adapter.parameters())
    backbone = total - cs_head - spectral
    return {
        "total": total,
        "trainable": trainable,
        "backbone": backbone,
        "cs_head": cs_head,
        "spectral_adapter": spectral,
    }
