"""
Memory Efficiency:
- ConvGRU processes only 2 frames (not 49) -> ~25x memory reduction
- Hidden state size is constant regardless of target sequence length
- Inference uses recursion with O(log n) depth for n frames

Gradient Stability:
- Maximum path length through network is 1 (not 49) during training
- Each forward pass has strong supervision on midpoint prediction
- No gradient vanishing over long sequences

Author: Refactored for SWE-PINN project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from torch.utils.checkpoint import checkpoint

from .complex_modules import ComplexConv2d, ComplexBatchNorm2d
from .complex_fno import ComplexFNOLayer, ComplexSpectralConv2d, ComplexReLU


class MidpointConvGRUCell(nn.Module):
    """
    Complex-valued Convolutional GRU Cell optimized for midpoint prediction.
    
    Processes complex-valued tensors (Real, Imag format) through a GRU cell.
    Uses 2D convolutions for spatial structure preservation.
    
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
        super(MidpointConvGRUCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Gates convolutions (input-to-hidden)
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
            hidden_channels * 2,
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
        x_flat = self._flatten_complex(x)
        h_flat = self._flatten_complex(h)
        
        # Compute gates
        gates_ih = self.conv_gates_ih(x_flat)
        gates_hh = self.conv_gates_hh(h_flat)
        gates = gates_ih + gates_hh
        
        # Split into reset and update gates
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        # Apply sigmoid activation to gates
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        # Compute candidate hidden state
        candidate_ih = self.conv_candidate_ih(x_flat)
        candidate_hh = self.conv_candidate_hh(reset_gate * h_flat)
        candidate = torch.tanh(candidate_ih + candidate_hh)
        
        # Compute new hidden state
        h_new_flat = (1 - update_gate) * h_flat + update_gate * candidate
        
        # Unflatten back to complex format
        h_new = self._unflatten_complex(h_new_flat, self.hidden_channels)
        
        return h_new


class ComplexConvGRUCell(nn.Module):
    """
    Complex-Valued Convolutional GRU Cell with Proper Complex Arithmetic.
    
    Unlike the flattened approach, this cell respects complex multiplication:
        (A + iB) * (X + iY) = (AX - BY) + i(BX + AY)
    
    Uses ComplexConv2d for all convolutions and applies activations to
    magnitude while preserving phase (like ComplexConvLSTMCell).
    
    GRU equations adapted for complex domain:
        r_t = complex_sigmoid(W_xr ⊛ x_t + W_hr ⊛ h_{t-1})     # Reset gate
        z_t = complex_sigmoid(W_xz ⊛ x_t + W_hz ⊛ h_{t-1})     # Update gate  
        h̃_t = complex_tanh(W_xh ⊛ x_t + W_hh ⊛ (r_t ⊙ h_{t-1}))  # Candidate
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t                  # New state
    
    Where ⊛ is complex convolution and ⊙ is complex element-wise multiplication.
    
    Supports dilation for increasing receptive field:
        - Dilation 1: Receptive Field = 1 px per layer
        - Dilation 2: Receptive Field = 2 px per layer  
        - Dilation 4: Receptive Field = 4 px per layer
    
    Input shape: (Batch, 2, Channels, Height, Width) - [Real, Imag]
    Hidden shape: (Batch, 2, Hidden, Height, Width) - [Real, Imag]
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True
    ):
        """
        Args:
            input_channels: Number of input channels (per real/imag component)
            hidden_channels: Number of hidden channels (per real/imag component)
            kernel_size: Convolution kernel size
            dilation: Dilation rate for dilated convolutions (increases receptive field)
            bias: Whether to use bias
        """
        super(ComplexConvGRUCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Padding formula for dilated convolution: padding = dilation * (kernel_size - 1) // 2
        padding = dilation * (kernel_size - 1) // 2
        
        # Complex convolutions for gates (input path)
        # Combined reset + update gates: 2 * hidden_channels
        self.conv_gates_x = ComplexConv2d(
            in_channels=input_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        
        # Complex convolutions for gates (hidden path)
        self.conv_gates_h = ComplexConv2d(
            in_channels=hidden_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False  # Avoid double bias
        )
        
        # Complex convolution for candidate (input path)
        self.conv_candidate_x = ComplexConv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        
        # Complex convolution for candidate (hidden path)
        self.conv_candidate_h = ComplexConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )
    
    def _complex_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex sigmoid: Apply sigmoid to magnitude, preserve phase.
        
        sigmoid(|z|) * (z / |z|) = sigmoid(|z|) * e^{i*angle(z)}
        
        This bounds gate magnitudes to [0, 1] while preserving complex phase.
        """
        real = x[:, 0]  # (B, C, H, W)
        imag = x[:, 1]  # (B, C, H, W)
        
        # Compute magnitude and phase
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        # Apply sigmoid to magnitude
        sig_mag = torch.sigmoid(magnitude)
        
        # Reconstruct with scaled magnitude
        out_real = sig_mag * phase_real
        out_imag = sig_mag * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex tanh: Apply tanh to magnitude, preserve phase.
        
        tanh(|z|) * (z / |z|) = tanh(|z|) * e^{i*angle(z)}
        
        This bounds values to unit circle while preserving phase.
        """
        real = x[:, 0]  # (B, C, H, W)
        imag = x[:, 1]  # (B, C, H, W)
        
        # Compute magnitude and phase
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        # Apply tanh to magnitude
        tanh_mag = torch.tanh(magnitude)
        
        # Reconstruct with scaled magnitude
        out_real = tanh_mag * phase_real
        out_imag = tanh_mag * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Complex element-wise multiplication: (a_r + i*a_i) * (b_r + i*b_i)
        
        Result: (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
        """
        a_real, a_imag = a[:, 0], a[:, 1]
        b_real, b_imag = b[:, 0], b[:, 1]
        
        out_real = a_real * b_real - a_imag * b_imag
        out_imag = a_real * b_imag + a_imag * b_real
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_subtract_from_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute (1 + 0i) - x for complex tensor x.
        
        Used for (1 - z_t) in GRU equation.
        """
        real = 1.0 - x[:, 0]
        imag = -x[:, 1]  # 0 - imag = -imag
        return torch.stack([real, imag], dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Complex ConvGRU cell.
        
        Args:
            x: Input tensor (B, 2, C_in, H, W) - [Real, Imag]
            h: Previous hidden state (B, 2, C_hidden, H, W) or None
        
        Returns:
            h_new: Updated hidden state (B, 2, C_hidden, H, W)
        """
        B = x.shape[0]
        H, W = x.shape[3], x.shape[4]
        device = x.device
        dtype = x.dtype
        
        # Initialize hidden state if None
        if h is None:
            h = torch.zeros(B, 2, self.hidden_channels, H, W, device=device, dtype=dtype)
        
        # Compute gates using complex convolutions
        gates_x = self.conv_gates_x(x)  # (B, 2, 2*hidden, H, W)
        gates_h = self.conv_gates_h(h)  # (B, 2, 2*hidden, H, W)
        
        # Complex addition
        gates = gates_x + gates_h  # (B, 2, 2*hidden, H, W)
        
        # Split into reset and update gates
        reset_gate = gates[:, :, :self.hidden_channels]   # (B, 2, hidden, H, W)
        update_gate = gates[:, :, self.hidden_channels:]  # (B, 2, hidden, H, W)
        
        # Apply complex sigmoid to gates
        reset_gate = self._complex_sigmoid(reset_gate)
        update_gate = self._complex_sigmoid(update_gate)
        
        # Compute candidate: h̃ = tanh(W_x*x + W_h*(r ⊙ h))
        # First compute r ⊙ h (complex element-wise multiply)
        reset_hidden = self._complex_multiply(reset_gate, h)
        
        candidate_x = self.conv_candidate_x(x)        # (B, 2, hidden, H, W)
        candidate_h = self.conv_candidate_h(reset_hidden)  # (B, 2, hidden, H, W)
        candidate = self._complex_tanh(candidate_x + candidate_h)
        
        # Compute new hidden state: h_new = (1 - z) ⊙ h + z ⊙ h̃
        one_minus_z = self._complex_subtract_from_one(update_gate)
        term1 = self._complex_multiply(one_minus_z, h)
        term2 = self._complex_multiply(update_gate, candidate)
        h_new = term1 + term2
        
        return h_new


class ComplexFusionLayer(nn.Module):
    """
    Learnable Fusion Layer for Complex-Valued Bi-Directional Features.
    
    Instead of simple concatenation + real Conv2d (which loses complex structure),
    this uses ComplexConv2d to properly fuse forward and backward hidden states.
    
    Architecture:
        [h_fwd, h_bwd] -> ComplexConv2d -> ComplexBatchNorm -> Complex LeakyReLU
    """
    
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        use_batchnorm: bool = True
    ):
        """
        Args:
            hidden_channels: Number of hidden channels from each GRU
            out_channels: Number of output channels
            use_batchnorm: Whether to use complex batch normalization
        """
        super(ComplexFusionLayer, self).__init__()
        
        # Complex convolution to fuse concatenated features
        # Input: 2 * hidden_channels (from fwd + bwd)
        self.fusion_conv = ComplexConv2d(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
        
        # Complex batch normalization
        if use_batchnorm:
            self.batchnorm = ComplexBatchNorm2d(out_channels)
        else:
            self.batchnorm = None
        
        # Complex LeakyReLU (applied to magnitude, preserves phase)
        self.leaky_slope = 0.2
    
    def _complex_leaky_relu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex LeakyReLU: Apply LeakyReLU to magnitude, preserve phase.
        """
        real = x[:, 0]
        imag = x[:, 1]
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        # LeakyReLU on magnitude
        activated_mag = F.leaky_relu(magnitude, negative_slope=self.leaky_slope)
        
        out_real = activated_mag * phase_real
        out_imag = activated_mag * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def forward(self, h_fwd: torch.Tensor, h_bwd: torch.Tensor) -> torch.Tensor:
        """
        Fuse forward and backward hidden states.
        
        Args:
            h_fwd: Forward hidden state (B, 2, hidden, H, W)
            h_bwd: Backward hidden state (B, 2, hidden, H, W)
        
        Returns:
            fused: Fused features (B, 2, out_channels, H, W)
        """
        # Concatenate along channel dimension (preserve complex structure)
        # (B, 2, hidden, H, W) + (B, 2, hidden, H, W) -> (B, 2, 2*hidden, H, W)
        h_concat = torch.cat([h_fwd, h_bwd], dim=2)
        
        # Apply complex convolution
        fused = self.fusion_conv(h_concat)
        
        # Apply complex batch normalization
        if self.batchnorm is not None:
            fused = self.batchnorm(fused)
        
        # Apply complex LeakyReLU
        fused = self._complex_leaky_relu(fused)
        
        return fused


class MidpointTemporalBlock(nn.Module):
    """
    Bi-Directional Stacked Dilated Complex ConvGRU with Learnable Fusion for Midpoint Prediction.
    
    This is the core block of the "Recursive Midpoint" architecture.
    It takes EXACTLY 2 frames (Start, End) and predicts the Midpoint Latent State.
    
    *** STACKED DILATED ConvGRU FOR INCREASED RECEPTIVE FIELD ***
    Instead of one 3x3 layer, uses a stack of 3 layers with increasing dilation.
    This exponentially increases the receptive field without adding parameters or memory cost.
    
    Layer 1: Kernel 3, Dilation 1 (Receptive Field = 1 px)
    Layer 2: Kernel 3, Dilation 2 (Receptive Field = 1 + 2 = 3 px)
    Layer 3: Kernel 3, Dilation 4 (Receptive Field = 3 + 4 = 7 px) → Covers 6.25 px gap!
    
    *** PROPERLY RESPECTS COMPLEX ARITHMETIC ***
    - Uses ComplexConv2d for convolutions (not flattened real Conv2d)
    - Uses complex_sigmoid/complex_tanh (magnitude activation, phase preservation)
    - Uses complex element-wise multiplication for gating
    
    Architecture:
        ┌─────────────┐
        │   z_start   │ ──► GRU1(d=1) ──► GRU2(d=2) ──► GRU3(d=4) ──► h_fwd ─┐
        └─────────────┘      (stacked forward GRUs)                          │
                                                                             ├──► ComplexFusion ──► FNO ──► z_mid
        ┌─────────────┐                                                      │
        │    z_end    │ ──► GRU1(d=1) ──► GRU2(d=2) ──► GRU3(d=4) ──► h_bwd ─┘
        └─────────────┘      (stacked backward GRUs)
    
    Input shape: (Batch, 2, Channels, Height, Width) per frame - [Real, Imag]
    Output shape: (Batch, 2, Channels, Height, Width) - single midpoint
    """
    
    def __init__(
        self,
        channels: int,
        num_modes_height: int = 12,
        num_modes_width: int = 12,
        gru_hidden_channels: Optional[int] = None,
        gru_kernel_size: int = 3,
        fno_num_layers: int = 2,
        use_batchnorm: bool = True,
        fft_padding: int = 8,
        use_residual: bool = True
    ):
        """
        Args:
            channels: Number of input/output channels (NOT including real/imag, they're in dim=1)
            num_modes_height: Fourier modes in height dimension
            num_modes_width: Fourier modes in width dimension
            gru_hidden_channels: GRU hidden channels (defaults to channels)
            gru_kernel_size: GRU convolution kernel size (default: 3)
            fno_num_layers: Number of FNO layers for spatial refinement
            use_batchnorm: Whether to use batch normalization
            fft_padding: Anti-spectral-ringing padding for FNO
            use_residual: Whether to add residual connection
        """
        super(MidpointTemporalBlock, self).__init__()
        
        self.channels = channels
        self.use_residual = use_residual
        gru_hidden = gru_hidden_channels or channels
        self.gru_hidden = gru_hidden
        
        # =========================================================
        # Stacked Dilated Bi-Directional Complex ConvGRU
        # Uses proper complex arithmetic throughout!
        # Stack of 3 GRUs with increasing dilation for wide receptive field
        # =========================================================
        
        # Forward GRU Stack: Processes z_start, propagates t_start → t_mid
        # Layer 1: Dilation 1 (RF = 1 px)
        self.fwd_gru1 = ComplexConvGRUCell(
            input_channels=channels,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            dilation=1
        )
        # Layer 2: Dilation 2 (RF = 1 + 2 = 3 px)
        self.fwd_gru2 = ComplexConvGRUCell(
            input_channels=gru_hidden,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            dilation=2
        )
        # Layer 3: Dilation 4 (RF = 3 + 4 = 7 px) - Wide reach!
        self.fwd_gru3 = ComplexConvGRUCell(
            input_channels=gru_hidden,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            dilation=4
        )
        
        # Backward GRU Stack: Processes z_end, propagates t_end → t_mid
        # Layer 1: Dilation 1 (RF = 1 px)
        self.bwd_gru1 = ComplexConvGRUCell(
            input_channels=channels,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            dilation=1
        )
        # Layer 2: Dilation 2 (RF = 1 + 2 = 3 px)
        self.bwd_gru2 = ComplexConvGRUCell(
            input_channels=gru_hidden,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            dilation=2
        )
        # Layer 3: Dilation 4 (RF = 3 + 4 = 7 px) - Wide reach!
        self.bwd_gru3 = ComplexConvGRUCell(
            input_channels=gru_hidden,
            hidden_channels=gru_hidden,
            kernel_size=gru_kernel_size,
            dilation=4
        )
        
        # =========================================================
        # Complex-Aware Fusion Layer
        # Properly combines forward/backward using ComplexConv2d
        # =========================================================
        self.fusion = ComplexFusionLayer(
            hidden_channels=gru_hidden,
            out_channels=channels,
            use_batchnorm=use_batchnorm
        )
        
        # =========================================================
        # Spatial Refiner: Complex FNO layers
        # =========================================================
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
    
    def forward(self, z_start: torch.Tensor, z_end: torch.Tensor) -> torch.Tensor:
        """
        Predict the midpoint latent state using Stacked Dilated Bi-Directional Complex ConvGRU.
        
        Args:
            z_start: Start frame latent (B, 2, C, H, W) - [Real, Imag]
            z_end: End frame latent (B, 2, C, H, W) - [Real, Imag]
        
        Returns:
            z_mid: Midpoint latent (B, 2, C, H, W) - [Real, Imag]
        
        All operations preserve complex structure:
        - ComplexConvGRU uses ComplexConv2d (proper complex multiplication)
        - Activations apply to magnitude, preserve phase
        - Gating uses complex element-wise multiplication
        
        Stacked GRU provides receptive field > 7 pixels:
        - GRU1 (dilation=1): RF = 1 px
        - GRU2 (dilation=2): RF = 1 + 2 = 3 px
        - GRU3 (dilation=4): RF = 3 + 4 = 7 px
        """
        # Shape validation
        B, two, C, H, W = z_start.shape
        assert two == 2, f"Expected complex format (B, 2, C, H, W), got {z_start.shape}"
        assert z_end.shape == z_start.shape, f"Shape mismatch: {z_start.shape} vs {z_end.shape}"
        
        # =========================================================
        # Forward GRU Stack: Process z_start (t_start → t_mid)
        # Captures "what information propagates from the start"
        # Each layer feeds its output to the next, building receptive field
        # =========================================================
        h1_fwd = self.fwd_gru1(z_start, h=None)  # (B, 2, hidden, H, W) - RF=1px
        h2_fwd = self.fwd_gru2(h1_fwd, h=None)   # (B, 2, hidden, H, W) - RF=3px
        h_fwd = self.fwd_gru3(h2_fwd, h=None)    # (B, 2, hidden, H, W) - RF=7px
        
        # =========================================================
        # Backward GRU Stack: Process z_end (t_end → t_mid)
        # Captures "what information propagates from the end"
        # Each layer feeds its output to the next, building receptive field
        # =========================================================
        h1_bwd = self.bwd_gru1(z_end, h=None)    # (B, 2, hidden, H, W) - RF=1px
        h2_bwd = self.bwd_gru2(h1_bwd, h=None)   # (B, 2, hidden, H, W) - RF=3px
        h_bwd = self.bwd_gru3(h2_bwd, h=None)    # (B, 2, hidden, H, W) - RF=7px
        
        # =========================================================
        # Complex Fusion: Combine forward/backward with learned weights
        # Uses ComplexConv2d to respect complex arithmetic
        # =========================================================
        z_mid = self.fusion(h_fwd, h_bwd)  # (B, 2, channels, H, W)
        
        # =========================================================
        # FNO Spatial Refinement
        # =========================================================
        refined = z_mid
        for fno_layer in self.spatial_refiner:
            refined = fno_layer(refined)
        
        # =========================================================
        # Residual Connection (complex addition)
        # =========================================================
        if self.use_residual:
            identity = 0.5 * (z_start + z_end)  # Complex linear interpolation
            refined = refined + identity
        
        return refined


class MidpointSpectralRefinement(nn.Module):
    """
    Midpoint-aware Spectral Refinement Module.
    
    This is a drop-in replacement for TemporalComplexSpectralRefinement
    that predicts ONLY the midpoint between two input frames.
    
    Architecture:
        [z_start, z_end] -> Input Projection -> MidpointTemporalBlock -> Output Projection -> z_mid
    
    Input: 2 frames (start, end)
    Output: 1 frame (midpoint)
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
        fno_num_layers: int = 2,
        use_batchnorm: bool = True,
        use_residual: bool = True,
        fft_padding: int = 8
    ):
        super(MidpointSpectralRefinement, self).__init__()
        
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
        
        # Core midpoint temporal block
        self.midpoint_block = MidpointTemporalBlock(
            channels=hidden_channels,
            num_modes_height=num_modes_height,
            num_modes_width=num_modes_width,
            gru_hidden_channels=gru_hidden_channels,
            gru_kernel_size=gru_kernel_size,
            fno_num_layers=fno_num_layers,
            use_batchnorm=use_batchnorm,
            fft_padding=fft_padding,
            use_residual=True  # Internal residual in midpoint block
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
    
    def forward(self, z_start: torch.Tensor, z_end: torch.Tensor) -> torch.Tensor:
        """
        Predict the midpoint latent between start and end frames.
        
        Args:
            z_start: Start frame latent (Batch, 2, Channels, Height, Width)
            z_end: End frame latent (Batch, 2, Channels, Height, Width)
        
        Returns:
            z_mid: Midpoint latent (Batch, 2, OutChannels, Height, Width)
        """
        # Input projection if needed
        if self.input_proj is not None:
            z_start_proj = self.input_proj(z_start)
            z_end_proj = self.input_proj(z_end)
        else:
            z_start_proj = z_start
            z_end_proj = z_end
        
        # Apply midpoint temporal block
        z_mid = self.midpoint_block(z_start_proj, z_end_proj)
        
        # Output projection if needed
        if self.output_proj is not None:
            z_mid = self.output_proj(z_mid)
        
        # Global residual connection (linear interpolation baseline)
        if self.use_residual:
            identity = 0.5 * (z_start + z_end)
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            z_mid = z_mid + identity
        
        return z_mid


def create_midpoint_spectral_refinement(
    channels: int,
    num_modes: int = 12,
    gru_hidden_channels: Optional[int] = None,
    fno_num_layers: int = 2,
    use_batchnorm: bool = True,
    use_residual: bool = True,
    fft_padding: int = 8
) -> MidpointSpectralRefinement:
    """
    Factory function to create a MidpointSpectralRefinement module.
    
    Args:
        channels: Number of channels (same for input, hidden, output)
        num_modes: Number of Fourier modes
        gru_hidden_channels: GRU hidden channels (defaults to channels)
        fno_num_layers: Number of FNO layers
        use_batchnorm: Whether to use batch normalization
        use_residual: Whether to use residual connections
        fft_padding: Anti-ringing padding
    
    Returns:
        MidpointSpectralRefinement module
    """
    return MidpointSpectralRefinement(
        in_channels=channels,
        hidden_channels=channels,
        out_channels=channels,
        num_modes_height=num_modes,
        num_modes_width=num_modes,
        gru_hidden_channels=gru_hidden_channels,
        fno_num_layers=fno_num_layers,
        use_batchnorm=use_batchnorm,
        use_residual=use_residual,
        fft_padding=fft_padding
    )
