"""
Temporal FNO Architecture for Shear Wave Elastography.

This module implements a memory-efficient temporal modeling approach that combines:
1. ConvGRU for temporal state propagation (captures wave physics/velocity)
2. FNO for spatial refinement (ensures spatial consistency)

Key Innovation: The "Temporal Physics Block" processes frames sequentially through
a recurrent unit (ConvGRU), allowing the model to learn temporal dynamics
(velocity, motion) rather than treating each frame independently.

Architecture Flow:
    Input -> Encoder -> Interpolate -> [ConvGRU + FNO] -> Decoder
    
The ConvGRU maintains a hidden state that accumulates "physics" information
(wave propagation patterns, velocity fields) across time steps. The FNO
then refines each frame spatially while respecting the learned temporal state.

Memory Efficiency:
- ConvGRU uses ~25% less memory than LSTM
- Sequential processing avoids batch-time memory explosion
- Gradient checkpointing compatible

Author: Auto-generated for SWE-PINN project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from torch.amp import custom_fwd, custom_bwd

from .complex_modules import ComplexConv2d, ComplexBatchNorm2d
from .complex_fno import ComplexFNOLayer, ComplexSpectralConv2d, ComplexReLU


class ComplexConvGRUCell(nn.Module):
    """
    Complex-valued Convolutional GRU Cell.
    
    Processes complex-valued tensors (Real, Imag format) through a GRU cell.
    Uses 2D convolutions for spatial structure preservation.
    
    GRU equations for complex domain:
        r_t = σ(W_xr * x_t + W_hr * h_{t-1} + b_r)  # Reset gate
        z_t = σ(W_xz * x_t + W_hz * h_{t-1} + b_z)  # Update gate
        ñ_t = tanh(W_xn * x_t + W_hn * (r_t ⊙ h_{t-1}) + b_n)  # Candidate
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t  # New hidden state
    
    Applied separately to real and imaginary components with shared gates.
    
    Input shape: (Batch, 2, Channels, Height, Width)
    Hidden shape: (Batch, 2, Hidden, Height, Width)
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        """
        Args:
            input_channels: Number of input feature channels
            hidden_channels: Number of hidden state channels
            kernel_size: Convolution kernel size
            bias: Whether to use bias in convolutions
        """
        super(ComplexConvGRUCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Gates convolutions (input-to-hidden)
        # Combined reset and update gates for efficiency
        self.conv_gates_ih = nn.Conv2d(
            input_channels * 2,  # Real + Imag flattened
            hidden_channels * 4,  # (reset_r, reset_i, update_r, update_i)
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
        # Gates convolutions (hidden-to-hidden)
        self.conv_gates_hh = nn.Conv2d(
            hidden_channels * 2,  # Real + Imag flattened
            hidden_channels * 4,  # (reset_r, reset_i, update_r, update_i)
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
        # Candidate convolution (input-to-hidden)
        self.conv_candidate_ih = nn.Conv2d(
            input_channels * 2,
            hidden_channels * 2,  # (candidate_r, candidate_i)
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
        # Candidate convolution (hidden-to-hidden)
        self.conv_candidate_hh = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _flatten_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten complex tensor (B, 2, C, H, W) -> (B, 2*C, H, W)."""
        B, _, C, H, W = x.shape
        return x.reshape(B, 2 * C, H, W)
    
    def _unflatten_complex(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        """Unflatten tensor (B, 2*C, H, W) -> (B, 2, C, H, W)."""
        B, _, H, W = x.shape
        return x.reshape(B, 2, channels, H, W)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for ConvGRU cell.
        
        Args:
            x: Input tensor (B, 2, C_in, H, W)
            h: Previous hidden state (B, 2, C_hidden, H, W) or None
        
        Returns:
            h_new: Updated hidden state (B, 2, C_hidden, H, W)
        """
        B, _, _, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize hidden state if None
        if h is None:
            h = torch.zeros(B, 2, self.hidden_channels, H, W, 
                          device=device, dtype=dtype)
        
        # Flatten complex dimension for convolutions
        x_flat = self._flatten_complex(x)  # (B, 2*C_in, H, W)
        h_flat = self._flatten_complex(h)  # (B, 2*C_hidden, H, W)
        
        # Compute gates
        gates_ih = self.conv_gates_ih(x_flat)  # (B, 4*C_hidden, H, W)
        gates_hh = self.conv_gates_hh(h_flat)  # (B, 4*C_hidden, H, W)
        gates = gates_ih + gates_hh
        
        # Split into reset and update gates (each has real + imag)
        reset_gate, update_gate = gates.chunk(2, dim=1)  # Each (B, 2*C_hidden, H, W)
        
        # Apply sigmoid activation to gates
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        # Compute candidate hidden state
        candidate_ih = self.conv_candidate_ih(x_flat)  # (B, 2*C_hidden, H, W)
        candidate_hh = self.conv_candidate_hh(reset_gate * h_flat)  # (B, 2*C_hidden, H, W)
        candidate = torch.tanh(candidate_ih + candidate_hh)
        
        # Compute new hidden state
        h_new_flat = (1 - update_gate) * h_flat + update_gate * candidate
        
        # Unflatten back to complex format
        h_new = self._unflatten_complex(h_new_flat, self.hidden_channels)
        
        return h_new


class ComplexConvGRU(nn.Module):
    """
    Multi-layer Complex Convolutional GRU.
    
    Stacks multiple ConvGRU cells for deeper temporal modeling.
    Each layer processes the output of the previous layer.
    
    Input shape: (Batch, 2, Channels, Height, Width) per time step
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels per layer
            kernel_size: Convolution kernel size
            num_layers: Number of stacked GRU layers
            dropout: Dropout between layers (not applied to last layer)
        """
        super(ComplexConvGRU, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(
                ComplexConvGRUCell(
                    input_channels=in_ch,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size
                )
            )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for multi-layer ConvGRU.
        
        Args:
            x: Input tensor (B, 2, C_in, H, W)
            h: List of hidden states per layer, or None
        
        Returns:
            output: Output from last layer (B, 2, C_hidden, H, W)
            h_new: List of updated hidden states per layer
        """
        if h is None:
            h = [None] * self.num_layers
        
        h_new = []
        current_input = x
        
        for i, cell in enumerate(self.cells):
            h_i = cell(current_input, h[i])
            h_new.append(h_i)
            current_input = h_i
            
            # Apply dropout between layers (not on last)
            if self.dropout is not None and i < self.num_layers - 1:
                # Flatten for dropout, then unflatten
                B, _, C, H, W = current_input.shape
                flat = current_input.reshape(B, 2 * C, H, W)
                flat = self.dropout(flat)
                current_input = flat.reshape(B, 2, C, H, W)
        
        return current_input, h_new


class TemporalPhysicsBlock(nn.Module):
    """
    Temporal Physics Block: The core innovation for temporal wave modeling.
    
    This block combines:
    1. ConvGRU: Captures temporal dynamics (velocity, wave propagation)
    2. FNO: Refines spatial structure (ensures wave-like patterns)
    
    The key insight: The ConvGRU maintains a "physics state" (hidden state)
    that encodes the wave's velocity and propagation direction. This state
    is updated at each time step, allowing the model to predict consistent
    motion across frames.
    
    Memory Efficiency:
    - Processes frames sequentially (O(1) in batch-time dimension)
    - Hidden state maintains temporal context without storing all frames
    - Compatible with gradient checkpointing
    
    Input shape: (Batch, Time, 2, Channels, Height, Width)
    Output shape: (Batch, Time, 2, Channels, Height, Width)
    """
    
    def __init__(
        self,
        channels: int,
        num_modes_height: int = 12,
        num_modes_width: int = 12,
        gru_hidden_channels: Optional[int] = None,
        gru_kernel_size: int = 3,
        gru_num_layers: int = 1,
        fno_num_layers: int = 2,
        use_batchnorm: bool = True,
        fft_padding: int = 8,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            channels: Number of input/output channels
            num_modes_height: Fourier modes in height dimension
            num_modes_width: Fourier modes in width dimension
            gru_hidden_channels: GRU hidden channels (defaults to channels)
            gru_kernel_size: GRU convolution kernel size
            gru_num_layers: Number of GRU layers
            fno_num_layers: Number of FNO layers for spatial refinement
            use_batchnorm: Whether to use batch normalization in FNO
            fft_padding: Anti-spectral-ringing padding for FNO
            use_residual: Whether to add residual connection
            dropout: Dropout rate for GRU
        """
        super(TemporalPhysicsBlock, self).__init__()
        
        self.channels = channels
        self.use_residual = use_residual
        gru_hidden = gru_hidden_channels or channels
        
        # Temporal engine: ConvGRU for physics state propagation
        self.temporal_engine = ComplexConvGRU(
            input_channels=channels,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            num_layers=gru_num_layers,
            dropout=dropout
        )
        
        # Project GRU output back to channels if needed
        if gru_hidden != channels:
            self.gru_proj = ComplexConv2d(
                gru_hidden, channels, kernel_size=1, padding=0
            )
        else:
            self.gru_proj = None
        
        # Spatial refiner: FNO layers for wave-like spatial structure
        self.spatial_refiner = nn.ModuleList([
            ComplexFNOLayer(
                channels=channels,
                num_modes_height=num_modes_height,
                num_modes_width=num_modes_width,
                use_batchnorm=use_batchnorm,
                fft_padding=fft_padding
            )
            for _ in range(fno_num_layers)
        ])
    
    def forward(
        self,
        x_sequence: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for Temporal Physics Block.
        
        Processes the sequence frame-by-frame through ConvGRU, then refines
        each frame spatially with FNO. The hidden state propagates "physics"
        (velocity, wave direction) across time steps.
        
        Args:
            x_sequence: Input tensor (Batch, Time, 2, Channels, Height, Width)
            return_hidden: If True, also return final hidden states
        
        Returns:
            output: Processed tensor (Batch, Time, 2, Channels, Height, Width)
            hidden (optional): Final hidden states if return_hidden=True
        """
        B, T, _, C, H, W = x_sequence.shape
        
        # Initialize hidden state (None lets GRU create zeros)
        h = None
        outputs = []
        
        # Process each time step sequentially
        for t in range(T):
            # Current input frame
            current_input = x_sequence[:, t]  # (B, 2, C, H, W)
            
            # Store input for residual
            identity = current_input
            
            # A. Update temporal state via ConvGRU
            gru_output, h = self.temporal_engine(current_input, h)
            
            # Project GRU output if dimensions differ
            if self.gru_proj is not None:
                gru_output = self.gru_proj(gru_output)
            
            # B. Refine spatial structure via FNO
            # The FNO operates on the hidden state which contains physics info
            refined_frame = gru_output
            for fno_layer in self.spatial_refiner:
                refined_frame = fno_layer(refined_frame)
            
            # C. Add residual connection
            if self.use_residual:
                refined_frame = refined_frame + identity
            
            outputs.append(refined_frame)
        
        # Stack outputs along time dimension
        output = torch.stack(outputs, dim=1)  # (B, T, 2, C, H, W)
        
        if return_hidden:
            return output, h
        return output


class TemporalComplexSpectralRefinement(nn.Module):
    """
    Temporal-aware Complex Spectral Refinement Module.
    
    This is a drop-in replacement for ComplexSpectralRefinement that adds
    temporal modeling via the TemporalPhysicsBlock.
    
    Architecture:
        Input Projection -> TemporalPhysicsBlock -> Output Projection
    
    The TemporalPhysicsBlock provides:
        - ConvGRU for temporal state propagation
        - FNO for spatial refinement
        - Residual connections for stable training
    
    Input shape: (Batch, Time, 2, Channels, Height, Width)
    Output shape: (Batch, Time, 2, Channels, Height, Width)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_modes_height: int = 12,
        num_modes_width: int = 12,
        gru_hidden_channels: Optional[int] = None,
        gru_kernel_size: int = 3,
        gru_num_layers: int = 1,
        fno_num_layers: int = 2,
        use_batchnorm: bool = True,
        use_residual: bool = True,
        fft_padding: int = 8,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels for processing
            out_channels: Number of output channels
            num_modes_height: Fourier modes in height
            num_modes_width: Fourier modes in width
            gru_hidden_channels: GRU hidden dim (defaults to hidden_channels)
            gru_kernel_size: GRU convolution kernel size
            gru_num_layers: Number of GRU layers
            fno_num_layers: Number of FNO layers in physics block
            use_batchnorm: Whether to use batch norm in FNO
            use_residual: Whether to use residual connections
            fft_padding: Anti-ringing padding for FNO
            dropout: Dropout rate
        """
        super(TemporalComplexSpectralRefinement, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        
        # Input projection
        if in_channels != hidden_channels:
            self.input_proj = ComplexConv2d(
                in_channels, hidden_channels, kernel_size=1, padding=0
            )
        else:
            self.input_proj = None
        
        # Core temporal physics block
        self.temporal_physics = TemporalPhysicsBlock(
            channels=hidden_channels,
            num_modes_height=num_modes_height,
            num_modes_width=num_modes_width,
            gru_hidden_channels=gru_hidden_channels,
            gru_kernel_size=gru_kernel_size,
            gru_num_layers=gru_num_layers,
            fno_num_layers=fno_num_layers,
            use_batchnorm=use_batchnorm,
            fft_padding=fft_padding,
            use_residual=True,  # Internal residual in physics block
            dropout=dropout
        )
        
        # Output projection
        if hidden_channels != out_channels:
            self.output_proj = ComplexConv2d(
                hidden_channels, out_channels, kernel_size=1, padding=0
            )
        else:
            self.output_proj = None
        
        # Residual projection for skip connection
        if use_residual and in_channels != out_channels:
            self.residual_proj = ComplexConv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal spectral refinement.
        
        Args:
            x: Input tensor (Batch, Time, 2, Channels, Height, Width)
        
        Returns:
            Output tensor (Batch, Time, 2, OutChannels, Height, Width)
        """
        B, T, _, C, H, W = x.shape
        identity = x
        
        # Input projection if needed
        if self.input_proj is not None:
            # Reshape for conv: (B*T, 2, C, H, W)
            x = x.reshape(B * T, 2, C, H, W)
            x = self.input_proj(x)
            x = x.reshape(B, T, 2, self.hidden_channels, H, W)
        
        # Apply temporal physics block
        x = self.temporal_physics(x)
        
        # Output projection if needed
        if self.output_proj is not None:
            x = x.reshape(B * T, 2, self.hidden_channels, H, W)
            x = self.output_proj(x)
            x = x.reshape(B, T, 2, self.out_channels, H, W)
        
        # Global residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                identity = identity.reshape(B * T, 2, self.in_channels, H, W)
                identity = self.residual_proj(identity)
                identity = identity.reshape(B, T, 2, self.out_channels, H, W)
            x = x + identity
        
        return x


def create_temporal_spectral_refinement(
    channels: int,
    num_modes: int = 12,
    gru_num_layers: int = 1,
    fno_num_layers: int = 2,
    use_batchnorm: bool = True,
    use_residual: bool = True,
    fft_padding: int = 8
) -> TemporalComplexSpectralRefinement:
    """
    Factory function to create a TemporalComplexSpectralRefinement module.
    
    Args:
        channels: Number of channels (same for input, hidden, output)
        num_modes: Number of Fourier modes
        gru_num_layers: Number of GRU layers
        fno_num_layers: Number of FNO layers
        use_batchnorm: Whether to use batch normalization
        use_residual: Whether to use residual connections
        fft_padding: Anti-ringing padding
    
    Returns:
        TemporalComplexSpectralRefinement module
    """
    return TemporalComplexSpectralRefinement(
        in_channels=channels,
        hidden_channels=channels,
        out_channels=channels,
        num_modes_height=num_modes,
        num_modes_width=num_modes,
        gru_num_layers=gru_num_layers,
        fno_num_layers=fno_num_layers,
        use_batchnorm=use_batchnorm,
        use_residual=use_residual,
        fft_padding=fft_padding
    )
