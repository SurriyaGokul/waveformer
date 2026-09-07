import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.amp import custom_fwd, custom_bwd

from .complex_modules import ComplexConv2d, ComplexBatchNorm2d

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU to real and imaginary parts separately.
        
        Args:
            x: Complex tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Activated tensor of same shape
        """
        real = self.relu(x[:, 0, :, :, :])
        imag = self.relu(x[:, 1, :, :, :])
        return torch.stack([real, imag], dim=1)

class ComplexSpectralConv2d(nn.Module):
    """
    Complex-Valued 2D Spectral Convolution Layer.
    
    This layer performs convolution in the Fourier domain:
    1. Convert (Real, Imag) tensor to native PyTorch complex64 tensor
    2. Apply 2D FFT to transform to frequency domain
    3. Truncate to lowest num_modes frequency modes (low-pass filter)
    4. Multiply by learnable complex weights in frequency domain
    5. Apply inverse 2D FFT to transform back to spatial domain
    6. Convert back to (Real, Imag) format
    
    The learnable weight has shape (in_channels, out_channels, num_modes, num_modes)
    and is complex-valued.
    
    Input shape: (Batch, 2, Channels, Height, Width)
        where dim=1 contains [Real, Imaginary] components
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_modes_height: int,
        num_modes_width: int
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_modes_height: Number of Fourier modes to keep in height dimension
            num_modes_width: Number of Fourier modes to keep in width dimension
        """
        super(ComplexSpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes_height = num_modes_height
        self.num_modes_width = num_modes_width
        
        # Scale factor for initialization (similar to Xavier/He init)
        self.scale = 1 / (in_channels * out_channels)
        
        # Learnable complex weights for the spectral domain
        # We need weights for both positive and negative frequency quadrants
        # Weight shape: (in_channels, out_channels, num_modes_height, num_modes_width)
        # Stored as real and imaginary parts separately
        
        # Weights for the first quadrant (top-left: low frequencies)
        self.weights1_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, num_modes_height, num_modes_width)
        )
        self.weights1_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, num_modes_height, num_modes_width)
        )
        
        # Weights for the second quadrant (top-right: mixed frequencies)
        self.weights2_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, num_modes_height, num_modes_width)
        )
        self.weights2_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, num_modes_height, num_modes_width)
        )
    
    def _complex_matmul_2d(
        self,
        input_fft: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform complex matrix multiplication in the Fourier domain.
        
        Args:
            input_fft: Complex tensor of shape (Batch, in_channels, modes_h, modes_w)
            weights_real: Real part of weights (in_channels, out_channels, modes_h, modes_w)
            weights_imag: Imaginary part of weights (in_channels, out_channels, modes_h, modes_w)
        
        Returns:
            Complex tensor of shape (Batch, out_channels, modes_h, modes_w)
        """
        # Ensure weights are float32 for numerical stability (important for AMP)
        weights_real = weights_real.float()
        weights_imag = weights_imag.float()
        
        # Construct complex weights
        weights = torch.complex(weights_real, weights_imag)
        
        # Einstein summation for batched complex matrix multiplication
        # input: (batch, in_channels, modes_h, modes_w)
        # weights: (in_channels, out_channels, modes_h, modes_w)
        # output: (batch, out_channels, modes_h, modes_w)
        return torch.einsum("bihw,iohw->bohw", input_fft, weights)
    
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectral convolution.
        
        Note: Uses @custom_fwd to ensure FFT operations always run in float32
        for numerical stability, even when AMP is enabled.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
               where x[:, 0, ...] is Real and x[:, 1, ...] is Imaginary
        
        Returns:
            Output tensor of shape (Batch, 2, OutChannels, Height, Width)
        """
        batch_size = x.shape[0]
        height = x.shape[-2]
        width = x.shape[-1]
        
        # Input is already float32 due to @custom_fwd(cast_inputs=torch.float32)
        
        # Step 1: Convert (Real, Imag) format to native PyTorch complex tensor
        # x[:, 0, ...] = Real, x[:, 1, ...] = Imaginary
        x_complex = torch.complex(x[:, 0, :, :, :], x[:, 1, :, :, :])
        # x_complex shape: (Batch, Channels, Height, Width) as complex64
        
        # Step 2: Perform 2D FFT
        x_fft = torch.fft.fft2(x_complex, norm='ortho')
        # x_fft shape: (Batch, Channels, Height, Width) as complex64
        
        # CRITICAL FIX: Dynamically clamp num_modes to actual spatial dimensions
        # This handles cases where feature map is smaller than num_modes
        effective_modes_h = min(self.num_modes_height, height)
        effective_modes_w = min(self.num_modes_width, width)
        
        # Step 3 & 4: Truncate to low-frequency modes and multiply by learnable weights
        # Initialize output tensor in Fourier domain
        out_fft = torch.zeros(
            batch_size, self.out_channels, height, width,
            dtype=torch.complex64, device=x.device
        )
        
        # Process first quadrant (top-left corner: positive frequencies in both dims)
        # Use only the subset of weights that fits the actual spatial dimensions
        out_fft[:, :, :effective_modes_h, :effective_modes_w] = \
            self._complex_matmul_2d(
                x_fft[:, :, :effective_modes_h, :effective_modes_w],
                self.weights1_real[:, :, :effective_modes_h, :effective_modes_w],
                self.weights1_imag[:, :, :effective_modes_h, :effective_modes_w]
            )
        
        # Process second quadrant (top-right corner: positive h, negative w frequencies)
        # In FFT output, negative frequencies are at the end of the array
        # Only process if there's room for negative frequencies
        if width > effective_modes_w:
            out_fft[:, :, :effective_modes_h, -effective_modes_w:] = \
                self._complex_matmul_2d(
                    x_fft[:, :, :effective_modes_h, -effective_modes_w:],
                    self.weights2_real[:, :, :effective_modes_h, :effective_modes_w],
                    self.weights2_imag[:, :, :effective_modes_h, :effective_modes_w]
                )
        
        # Step 5: Perform inverse 2D FFT
        x_ifft = torch.fft.ifft2(out_fft, norm='ortho')
        # x_ifft shape: (Batch, OutChannels, Height, Width) as complex64
        
        # Step 6: Convert back to (Real, Imag) format
        output = torch.stack([x_ifft.real, x_ifft.imag], dim=1)
        # output shape: (Batch, 2, OutChannels, Height, Width)
        
        # Output stays in float32 (no conversion back to float16)
        # This ensures gradient computation is numerically stable
        
        return output

class ComplexFNOLayer(nn.Module):
    """
    Single Complex FNO Layer.
    
    Combines:
    1. Spectral Convolution Path (K): Global convolution in Fourier space
    2. Skip Connection Path (W): Local convolution (standard ComplexConv2d)
    3. Complex ReLU activation
    4. Complex Batch Normalization
    
    Architecture:
        output = ComplexBatchNorm(ComplexReLU(K(x) + W(x)))
    
    where K is the spectral convolution and W is a 1x1 complex convolution.
    
    Anti-Spectral Ringing:
        To prevent Gibbs phenomenon (spectral ringing) at boundaries, this layer
        pads the input before FFT and crops after IFFT. This pushes boundary
        ripples into the padded region which is then discarded.
    
    Input shape: (Batch, 2, Channels, Height, Width)
    """
    
    def __init__(
        self,
        channels: int,
        num_modes_height: int,
        num_modes_width: int,
        use_batchnorm: bool = True,
        fft_padding: int = 16  # Padding to prevent spectral ringing
    ):
        """
        Args:
            channels: Number of channels (same for input and output)
            num_modes_height: Number of Fourier modes to keep in height dimension
            num_modes_width: Number of Fourier modes to keep in width dimension
            use_batchnorm: Whether to apply batch normalization
            fft_padding: Padding size for anti-spectral-ringing (default: 16)
        """
        super(ComplexFNOLayer, self).__init__()
        
        self.channels = channels
        self.use_batchnorm = use_batchnorm
        self.fft_padding = fft_padding
        
        # Spectral convolution path (K path)
        self.spectral_conv = ComplexSpectralConv2d(
            in_channels=channels,
            out_channels=channels,
            num_modes_height=num_modes_height,
            num_modes_width=num_modes_width
        )
        
        # Skip connection path (W path) - 1x1 convolution
        self.skip_conv = ComplexConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # Activation
        self.activation = ComplexReLU()
        
        # Batch normalization
        if self.use_batchnorm:
            self.batchnorm = ComplexBatchNorm2d(num_features=channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FNO layer.
        
        Includes anti-spectral-ringing: pads input before FFT, crops after IFFT.
        This pushes boundary ripples (Gibbs phenomenon) into the padded region
        which is then discarded, leaving the center clean.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Output tensor of shape (Batch, 2, Channels, Height, Width)
        """
        # Store original spatial dimensions for cropping
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        
        # 1. PAD the input before FFT to prevent spectral ringing
        # F.pad with reflect mode for 5D tensors requires 6-element tuple (last 3 dims)
        # To pad only H,W, we reshape to 4D, pad, then reshape back
        if self.fft_padding > 0:
            batch_size, two, channels, h, w = x.shape
            # Reshape (B, 2, C, H, W) -> (B*2, C, H, W) for 4D padding
            x_4d = x.reshape(batch_size * 2, channels, h, w)
            x_padded_4d = F.pad(x_4d, (0, self.fft_padding, 0, self.fft_padding), mode='reflect')
            # Reshape back (B*2, C, H+pad, W+pad) -> (B, 2, C, H+pad, W+pad)
            x_padded = x_padded_4d.reshape(batch_size, 2, channels, h + self.fft_padding, w + self.fft_padding)
        else:
            x_padded = x
        
        # 2. Spectral convolution path (includes FFT -> spectral conv -> IFFT)
        x_spectral_padded = self.spectral_conv(x_padded)
        
        # 3. CROP back to original size (removes boundary ripples)
        if self.fft_padding > 0:
            x_spectral = x_spectral_padded[..., :orig_h, :orig_w]
        else:
            x_spectral = x_spectral_padded
        
        # Skip connection path (1x1 conv) - operates on original (unpadded) input
        x_skip = self.skip_conv(x)
        
        # Combine paths
        x_combined = x_spectral + x_skip
        
        # Apply activation
        x_activated = self.activation(x_combined)
        
        # Apply batch normalization if enabled
        if self.use_batchnorm:
            x_out = self.batchnorm(x_activated)
        else:
            x_out = x_activated
        
        return x_out


class ComplexFNOBlock(nn.Module):
    """
    Complex FNO Block for processing temporal sequences.
    
    This block processes input tensors with a time dimension by:
    1. Reshaping (B, T, 2, C, H, W) to (B*T, 2, C, H, W)
    2. Applying multiple ComplexFNOLayers
    3. Reshaping back to (B, T, 2, C, H, W)
    
    This approach processes all frames in parallel, which is more efficient
    than iterating over the time dimension.
    
    Input shape: (Batch, Time, 2, Channels, Height, Width)
    Output shape: (Batch, Time, 2, Channels, Height, Width)
    """
    # TODO: Implement sequential processing over time dimension as parallel causing memory bottleneck and no performance loss by doing so.
    
    def __init__(
        self,
        channels: int,
        num_modes_height: int,
        num_modes_width: int,
        num_layers: int = 2,
        use_batchnorm: bool = True,
        fft_padding: int = 16  # Anti-spectral-ringing padding
    ):
        """
        Args:
            channels: Number of channels
            num_modes_height: Number of Fourier modes to keep in height dimension
            num_modes_width: Number of Fourier modes to keep in width dimension
            num_layers: Number of FNO layers to stack
            use_batchnorm: Whether to apply batch normalization in each layer
            fft_padding: Padding size for anti-spectral-ringing (default: 16)
        """
        super(ComplexFNOBlock, self).__init__()
        
        self.channels = channels
        self.num_layers = num_layers
        
        # Stack of FNO layers
        self.fno_layers = nn.ModuleList([
            ComplexFNOLayer(
                channels=channels,
                num_modes_height=num_modes_height,
                num_modes_width=num_modes_width,
                use_batchnorm=use_batchnorm,
                fft_padding=fft_padding
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FNO block.
        
        Args:
            x: Input tensor of shape (Batch, Time, 2, Channels, Height, Width)
        
        Returns:
            Output tensor of shape (Batch, Time, 2, Channels, Height, Width)
        """
        # Get input dimensions
        batch_size, time_steps, _, channels, height, width = x.shape
        
        # Reshape: (B, T, 2, C, H, W) -> (B*T, 2, C, H, W)
        # Use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(batch_size * time_steps, 2, channels, height, width)
        
        # Apply FNO layers sequentially
        for fno_layer in self.fno_layers:
            x = fno_layer(x)
        
        # Reshape back: (B*T, 2, C, H, W) -> (B, T, 2, C, H, W)
        x = x.reshape(batch_size, time_steps, 2, channels, height, width)
        
        return x


class ComplexSpectralRefinement(nn.Module):
    """
    Complex Spectral Refinement Module using Fourier Neural Operator.
    
    This is the main module designed to replace ConvLSTM refinement blocks.
    It applies global spectral convolution to refine latent features after
    they have been warped/interpolated.
    
    Features:
    - Processes complex-valued tensors (Real, Imag format)
    - Uses Fourier Neural Operator for global receptive field
    - Low-pass filtering via mode truncation
    - Residual connection for stable training
    
    Input shape: (Batch, Time, 2, Channels, Height, Width)
        where dim=2 contains [Real, Imaginary] components
    
    Output shape: (Batch, Time, 2, Channels, Height, Width)
    
    Usage:
        refinement = ComplexSpectralRefinement(
            in_channels=64,
            hidden_channels=64,
            out_channels=64,
            num_modes_height=12,
            num_modes_width=12,
            num_layers=3
        )
        
        # Input: (B, T, 2, C, H, W)
        output = refinement(input_tensor)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_modes_height: int,
        num_modes_width: int,
        num_layers: int = 3,
        use_batchnorm: bool = True,
        use_residual: bool = True,
        fft_padding: int = 16
    ):
        """
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels for FNO processing
            out_channels: Number of output channels
            num_modes_height: Number of Fourier modes to keep in height dimension
            num_modes_width: Number of Fourier modes to keep in width dimension
            num_layers: Number of FNO layers to stack
            use_batchnorm: Whether to apply batch normalization
            use_residual: Whether to add residual connection from input to output
            fft_padding: Padding to add before FFT to reduce spectral ringing (Gibbs phenomenon)
        """
        super(ComplexSpectralRefinement, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        
        # Input projection (if channels don't match)
        if in_channels != hidden_channels:
            self.input_proj = ComplexConv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        else:
            self.input_proj = None
        
        # Main FNO block with anti-spectral-ringing padding
        self.fno_block = ComplexFNOBlock(
            channels=hidden_channels,
            num_modes_height=num_modes_height,
            num_modes_width=num_modes_width,
            num_layers=num_layers,
            use_batchnorm=use_batchnorm,
            fft_padding=fft_padding
        )
        
        # Output projection (if channels don't match)
        if hidden_channels != out_channels:
            self.output_proj = ComplexConv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        else:
            self.output_proj = None
        
        # Residual projection (if input and output channels don't match)
        if use_residual and in_channels != out_channels:
            self.residual_proj = ComplexConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectral refinement.
        
        Args:
            x: Input tensor of shape (Batch, Time, 2, Channels, Height, Width)
               where x[:, :, 0, ...] is Real and x[:, :, 1, ...] is Imaginary
        
        Returns:
            Output tensor of shape (Batch, Time, 2, OutChannels, Height, Width)
        """
        # Store input for residual connection
        identity = x
        
        # Get input dimensions
        batch_size, time_steps, _, channels, height, width = x.shape
        
        # Input projection if needed
        if self.input_proj is not None:
            # Reshape for projection: (B, T, 2, C, H, W) -> (B*T, 2, C, H, W)
            # Use reshape instead of view to handle non-contiguous tensors
            x = x.reshape(batch_size * time_steps, 2, channels, height, width)
            x = self.input_proj(x)
            # Reshape back: (B*T, 2, C', H, W) -> (B, T, 2, C', H, W)
            x = x.reshape(batch_size, time_steps, 2, self.hidden_channels, height, width)
        
        # Apply FNO block
        x = self.fno_block(x)
        
        # Output projection if needed
        if self.output_proj is not None:
            # Reshape for projection: (B, T, 2, C, H, W) -> (B*T, 2, C, H, W)
            # Use reshape instead of view to handle non-contiguous tensors
            x = x.reshape(batch_size * time_steps, 2, self.hidden_channels, height, width)
            x = self.output_proj(x)
            # Reshape back: (B*T, 2, C', H, W) -> (B, T, 2, C', H, W)
            x = x.reshape(batch_size, time_steps, 2, self.out_channels, height, width)
        
        # Add residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                # Project identity to match output channels
                # Reshape: (B, T, 2, C, H, W) -> (B*T, 2, C, H, W)
                # Use reshape instead of view to handle non-contiguous tensors
                identity = identity.reshape(batch_size * time_steps, 2, self.in_channels, height, width)
                identity = self.residual_proj(identity)
                # Reshape back: (B*T, 2, C', H, W) -> (B, T, 2, C', H, W)
                identity = identity.reshape(batch_size, time_steps, 2, self.out_channels, height, width)
            x = x + identity
        
        return x

def create_complex_spectral_refinement(
    channels: int,
    num_modes: int = 12,
    num_layers: int = 3,
    use_batchnorm: bool = True,
    use_residual: bool = True,
    fft_padding: int = 16
) -> ComplexSpectralRefinement:
    """
    Factory function to create a ComplexSpectralRefinement module with
    common default settings.
    
    Args:
        channels: Number of channels (same for input, hidden, and output)
        num_modes: Number of Fourier modes to keep (same for height and width)
        num_layers: Number of FNO layers
        use_batchnorm: Whether to use batch normalization
        use_residual: Whether to use residual connection
        fft_padding: Padding to add before FFT to reduce spectral ringing (Gibbs phenomenon)
    
    Returns:
        ComplexSpectralRefinement module
    """
    return ComplexSpectralRefinement(
        in_channels=channels,
        hidden_channels=channels,
        out_channels=channels,
        num_modes_height=num_modes,
        num_modes_width=num_modes,
        num_layers=num_layers,
        use_batchnorm=use_batchnorm,
        use_residual=use_residual,
        fft_padding=fft_padding
    )

