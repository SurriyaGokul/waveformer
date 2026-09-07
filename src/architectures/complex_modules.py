import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D Convolution Layer.
    
    Implements complex convolution using the formula:
    If Input Z = A + iB and Weight W = X + iY:
        Real Output = (A * X) - (B * Y)
        Imaginary Output = (B * X) + (A * Y)
    
    Input tensor shape: (Batch, 2, Channels, Height, Width)
        where dim=1 contains [Real, Imaginary] components
    
    Uses Complex He Initialization for weights.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (per real/imag component)
            out_channels: Number of output channels (per real/imag component)
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output
            bias: If True, adds a learnable bias
        """
        super(ComplexConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Real part of weights (X)
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # Imaginary part of weights (Y)
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

        self._complex_he_init()
    
    def _complex_he_init(self):
        """
        Complex He Initialization.
        
        Initialize magnitude from Rayleigh distribution and phase uniformly.
        This ensures proper variance scaling for complex-valued networks.
        """
        # Calculate fan_in for He initialization
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        
        # Standard deviation for He init (factor of 2 for complex)
        std = np.sqrt(2.0 / fan_in)
        
        # Initialize magnitude (Rayleigh distributed) and phase (uniform)
        with torch.no_grad():
            # Generate magnitude from Rayleigh distribution
            # Rayleigh with scale=std is equivalent to sqrt(X^2 + Y^2) where X,Y ~ N(0, std)
            magnitude = torch.abs(torch.randn_like(self.conv_real.weight) * std)
            
            # Generate uniform phase in [-pi, pi]
            phase = torch.empty_like(self.conv_real.weight).uniform_(-np.pi, np.pi)
            
            # Convert to Cartesian form
            self.conv_real.weight.copy_(magnitude * torch.cos(phase))
            self.conv_imag.weight.copy_(magnitude * torch.sin(phase))
            
            # Initialize biases to zero if they exist
            if self.conv_real.bias is not None:
                nn.init.zeros_(self.conv_real.bias)
            if self.conv_imag.bias is not None:
                nn.init.zeros_(self.conv_imag.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for complex convolution.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
               where x[:, 0, ...] is Real and x[:, 1, ...] is Imaginary
        
        Returns:
            Output tensor of shape (Batch, 2, OutChannels, H_out, W_out)
        """
        # Split input into real (A) and imaginary (B) components
        real_input = x[:, 0, :, :, :]  # A: (Batch, Channels, H, W)
        imag_input = x[:, 1, :, :, :]  # B: (Batch, Channels, H, W)
        
        # Complex multiplication: (A + iB) * (X + iY) = (AX - BY) + i(BX + AY)
        # Real output = A*X - B*Y
        real_output = self.conv_real(real_input) - self.conv_imag(imag_input)
        
        # Imaginary output = B*X + A*Y
        imag_output = self.conv_real(imag_input) + self.conv_imag(real_input)
        
        # Stack back to (Batch, 2, OutChannels, H, W)
        output = torch.stack([real_output, imag_output], dim=1)
        
        return output


# ============================================================================
# Normalization Layers
# ============================================================================

class ComplexBatchNorm2d(nn.Module):
    """
    Complex Batch Normalization for 2D inputs.
    
    This implements proper complex batch normalization by treating the 
    Real and Imaginary parts as a 2D vector and whitening the covariance matrix,
    not just normalizing individual variances.
    
    The covariance matrix is:
        C = [[Vrr, Vri],
             [Vri, Vii]]
    
    where:
        Vrr = Var(real), Vii = Var(imag), Vri = Cov(real, imag)
    
    We compute C^(-1/2) to whiten the data, preserving the correlation structure.
    
    Reference: "Deep Complex Networks" by Trabelsi et al.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        """
        Args:
            num_features: Number of channels (C in input shape)
            eps: Small constant for numerical stability
            momentum: Momentum for running stats update
            affine: If True, learnable affine parameters (gamma, beta)
            track_running_stats: If True, track running mean and covariance
        """
        super(ComplexBatchNorm2d, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            # Learnable scale parameters for the 2x2 whitening matrix
            # gamma_rr, gamma_ri, gamma_ii form the upper triangular of scaling matrix
            self.gamma_rr = nn.Parameter(torch.ones(num_features))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_ii = nn.Parameter(torch.ones(num_features))
            # Learnable bias (complex)
            self.beta_real = nn.Parameter(torch.zeros(num_features))
            self.beta_imag = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma_rr', None)
            self.register_parameter('gamma_ri', None)
            self.register_parameter('gamma_ii', None)
            self.register_parameter('beta_real', None)
            self.register_parameter('beta_imag', None)
        
        if self.track_running_stats:
            # Running mean for real and imaginary parts
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))
            # Running covariance matrix elements
            self.register_buffer('running_Vrr', torch.ones(num_features))
            self.register_buffer('running_Vii', torch.ones(num_features))
            self.register_buffer('running_Vri', torch.zeros(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean_real', None)
            self.register_buffer('running_mean_imag', None)
            self.register_buffer('running_Vrr', None)
            self.register_buffer('running_Vii', None)
            self.register_buffer('running_Vri', None)
            self.register_buffer('num_batches_tracked', None)
    
    def _compute_inverse_sqrt_covariance(
        self, Vrr: torch.Tensor, Vii: torch.Tensor, Vri: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the inverse square root of the 2x2 covariance matrix.
        
        For a 2x2 positive definite matrix [[a, b], [b, c]], we compute
        the matrix W such that W @ W.T = C^(-1).
        
        Returns the elements of the inverse sqrt matrix (w_rr, w_ri, w_ii).
        """
        # Compute determinant and trace
        det = Vrr * Vii - Vri * Vri
        # Add eps for numerical stability
        det = det.clamp(min=self.eps)
        
        # Trace
        trace = Vrr + Vii
        
        # s = sqrt(det)
        s = torch.sqrt(det)
        
        # t = sqrt(trace + 2*s)
        t = torch.sqrt(trace + 2 * s + self.eps)
        
        # Inverse sqrt matrix elements (derived from Cayley-Hamilton theorem)
        # W = (C + s*I) / t  where I is identity
        inverse_st = 1.0 / (s * t + self.eps)
        
        w_rr = (Vii + s) * inverse_st
        w_ii = (Vrr + s) * inverse_st
        w_ri = -Vri * inverse_st
        
        return w_rr, w_ri, w_ii
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complex batch normalization.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Normalized output tensor of same shape
        """
        # Split into real and imaginary
        real = x[:, 0, :, :, :]  # (B, C, H, W)
        imag = x[:, 1, :, :, :]  # (B, C, H, W)
        
        if self.training or not self.track_running_stats:
            # Compute batch statistics
            # Mean over batch, height, width (keep channels)
            mean_real = real.mean(dim=(0, 2, 3))  # (C,)
            mean_imag = imag.mean(dim=(0, 2, 3))  # (C,)
            
            # Center the data
            real_centered = real - mean_real.view(1, -1, 1, 1)
            imag_centered = imag - mean_imag.view(1, -1, 1, 1)
            
            # Compute covariance elements
            n = real.numel() / real.size(1)  # Total elements per channel
            
            Vrr = (real_centered * real_centered).mean(dim=(0, 2, 3)) + self.eps
            Vii = (imag_centered * imag_centered).mean(dim=(0, 2, 3)) + self.eps
            Vri = (real_centered * imag_centered).mean(dim=(0, 2, 3))
            
            # Update running stats
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        # Cumulative moving average
                        exp_avg_factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        exp_avg_factor = self.momentum
                    
                    self.running_mean_real.mul_(1 - exp_avg_factor).add_(mean_real * exp_avg_factor)
                    self.running_mean_imag.mul_(1 - exp_avg_factor).add_(mean_imag * exp_avg_factor)
                    self.running_Vrr.mul_(1 - exp_avg_factor).add_(Vrr * exp_avg_factor)
                    self.running_Vii.mul_(1 - exp_avg_factor).add_(Vii * exp_avg_factor)
                    self.running_Vri.mul_(1 - exp_avg_factor).add_(Vri * exp_avg_factor)
        else:
            # Use running stats
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            Vrr = self.running_Vrr
            Vii = self.running_Vii
            Vri = self.running_Vri
            
            real_centered = real - mean_real.view(1, -1, 1, 1)
            imag_centered = imag - mean_imag.view(1, -1, 1, 1)
        
        # Compute inverse sqrt of covariance matrix
        w_rr, w_ri, w_ii = self._compute_inverse_sqrt_covariance(Vrr, Vii, Vri)
        
        # Apply whitening: [real_out, imag_out] = W @ [real_centered, imag_centered]
        w_rr = w_rr.view(1, -1, 1, 1)
        w_ri = w_ri.view(1, -1, 1, 1)
        w_ii = w_ii.view(1, -1, 1, 1)
        
        real_whitened = w_rr * real_centered + w_ri * imag_centered
        imag_whitened = w_ri * real_centered + w_ii * imag_centered
        
        # Apply affine transformation if enabled
        if self.affine:
            gamma_rr = self.gamma_rr.view(1, -1, 1, 1)
            gamma_ri = self.gamma_ri.view(1, -1, 1, 1)
            gamma_ii = self.gamma_ii.view(1, -1, 1, 1)
            beta_real = self.beta_real.view(1, -1, 1, 1)
            beta_imag = self.beta_imag.view(1, -1, 1, 1)
            
            real_out = gamma_rr * real_whitened + gamma_ri * imag_whitened + beta_real
            imag_out = gamma_ri * real_whitened + gamma_ii * imag_whitened + beta_imag
        else:
            real_out = real_whitened
            imag_out = imag_whitened
        
        # Stack back together
        return torch.stack([real_out, imag_out], dim=1)


class NaiveComplexBatchNorm2d(nn.Module):
    """
    Naive Complex Batch Normalization.
    
    A simpler version that applies standard BatchNorm independently 
    to real and imaginary parts. Does not properly handle the 
    covariance structure, but can be more stable during training.
    
    Use this as a fallback if ComplexBatchNorm2d causes training issues.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super(NaiveComplexBatchNorm2d, self).__init__()
        
        self.bn_real = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats
        )
        self.bn_imag = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply naive complex batch normalization.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Normalized output tensor of same shape
        """
        real = x[:, 0, :, :, :]
        imag = x[:, 1, :, :, :]
        
        real_out = self.bn_real(real)
        imag_out = self.bn_imag(imag)
        
        return torch.stack([real_out, imag_out], dim=1)


class ComplexLayerNorm(nn.Module):
    """
    Complex Layer Normalization.
    
    Normalizes across the channel, height, and width dimensions
    while treating real and imaginary as a 2D vector.
    
    Useful when batch sizes are small or for recurrent networks.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """
        Args:
            normalized_shape: Number of channels to normalize over
            eps: Small constant for numerical stability
        """
        super(ComplexLayerNorm, self).__init__()
        
        self.eps = eps
        self.normalized_shape = normalized_shape
        
        # Learnable parameters (complex scale and shift)
        self.gamma_real = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_imag = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_real = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_imag = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complex layer normalization.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Normalized output tensor of same shape
        """
        real = x[:, 0, :, :, :]  # (B, C, H, W)
        imag = x[:, 1, :, :, :]
        
        # Compute mean over C, H, W
        mean_real = real.mean(dim=(1, 2, 3), keepdim=True)
        mean_imag = imag.mean(dim=(1, 2, 3), keepdim=True)
        
        # Center
        real_centered = real - mean_real
        imag_centered = imag - mean_imag
        
        # Compute variance (combined for complex)
        var = (real_centered ** 2 + imag_centered ** 2).mean(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(var + self.eps)
        
        # Normalize
        real_norm = real_centered / std
        imag_norm = imag_centered / std
        
        # Apply learnable parameters
        gamma_r = self.gamma_real.view(1, -1, 1, 1)
        gamma_i = self.gamma_imag.view(1, -1, 1, 1)
        beta_r = self.beta_real.view(1, -1, 1, 1)
        beta_i = self.beta_imag.view(1, -1, 1, 1)
        
        # Complex scaling: (gamma_r + i*gamma_i) * (real_norm + i*imag_norm)
        real_out = gamma_r * real_norm - gamma_i * imag_norm + beta_r
        imag_out = gamma_r * imag_norm + gamma_i * real_norm + beta_i
        
        return torch.stack([real_out, imag_out], dim=1)


class ComplexInstanceNorm2d(nn.Module):
    """
    Complex Instance Normalization for 2D inputs.
    
    Normalizes each sample independently across spatial dimensions.
    Useful for style transfer and when batch statistics are unreliable.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            num_features: Number of channels
            eps: Small constant for numerical stability
            affine: If True, learnable affine parameters
        """
        super(ComplexInstanceNorm2d, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.gamma_real = nn.Parameter(torch.ones(num_features))
            self.gamma_imag = nn.Parameter(torch.zeros(num_features))
            self.beta_real = nn.Parameter(torch.zeros(num_features))
            self.beta_imag = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma_real', None)
            self.register_parameter('gamma_imag', None)
            self.register_parameter('beta_real', None)
            self.register_parameter('beta_imag', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complex instance normalization.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Normalized output tensor of same shape
        """
        real = x[:, 0, :, :, :]  # (B, C, H, W)
        imag = x[:, 1, :, :, :]
        
        # Compute mean over H, W (per instance, per channel)
        mean_real = real.mean(dim=(2, 3), keepdim=True)
        mean_imag = imag.mean(dim=(2, 3), keepdim=True)
        
        # Center
        real_centered = real - mean_real
        imag_centered = imag - mean_imag
        
        # Compute variance (combined magnitude)
        var = (real_centered ** 2 + imag_centered ** 2).mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(var + self.eps)
        
        # Normalize
        real_norm = real_centered / std
        imag_norm = imag_centered / std
        
        if self.affine:
            gamma_r = self.gamma_real.view(1, -1, 1, 1)
            gamma_i = self.gamma_imag.view(1, -1, 1, 1)
            beta_r = self.beta_real.view(1, -1, 1, 1)
            beta_i = self.beta_imag.view(1, -1, 1, 1)
            
            real_out = gamma_r * real_norm - gamma_i * imag_norm + beta_r
            imag_out = gamma_r * imag_norm + gamma_i * real_norm + beta_i
        else:
            real_out = real_norm
            imag_out = imag_norm
        
        return torch.stack([real_out, imag_out], dim=1)


class CReLU(nn.Module):
    """
    Complex ReLU Activation.
    
    Applies ReLU separately to the Real and Imaginary parts:
        out = ReLU(real) + i * ReLU(imag)
    
    This is the simplest complex activation and provides good stability.
    """
    
    def __init__(self):
        super(CReLU, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CReLU activation.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
               where x[:, 0, ...] is Real and x[:, 1, ...] is Imaginary
        
        Returns:
            Output tensor with ReLU applied to both components
        """
        # Apply ReLU to both real and imaginary parts independently
        real_output = F.relu(x[:, 0, :, :, :])
        imag_output = F.relu(x[:, 1, :, :, :])
        
        # Stack back together
        output = torch.stack([real_output, imag_output], dim=1)
        
        return output


class ModReLU(nn.Module):
    """
    Modulus ReLU Activation.
    
    Applies ReLU to the magnitude while preserving phase:
        out = ReLU(|z| + b) * (z / |z|)
    
    where b is a learnable bias parameter.
    
    This activation preserves phase information better than CReLU
    but may be less stable during training.
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-8):
        """
        Args:
            num_channels: Number of channels for the learnable bias
            eps: Small constant for numerical stability
        """
        super(ModReLU, self).__init__()
        
        self.eps = eps
        # Learnable bias for the ReLU applied to magnitude
        self.bias = nn.Parameter(torch.zeros(1, 1, num_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ModReLU activation.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
               where x[:, 0, ...] is Real and x[:, 1, ...] is Imaginary
        
        Returns:
            Output tensor with ModReLU applied
        """
        real = x[:, 0, :, :, :]  # (Batch, Channels, H, W)
        imag = x[:, 1, :, :, :]  # (Batch, Channels, H, W)
        
        # Compute magnitude: |z| = sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        
        # Apply ReLU to (magnitude + bias)
        # Squeeze bias to match: (1, Channels, 1, 1)
        bias_squeezed = self.bias.squeeze(1)  # (1, Channels, 1, 1)
        activated_magnitude = F.relu(magnitude + bias_squeezed)
        
        # Compute phase (unit vector): z / |z|
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        # Scale by activated magnitude
        real_output = activated_magnitude * phase_real
        imag_output = activated_magnitude * phase_imag
        
        # Stack back together
        output = torch.stack([real_output, imag_output], dim=1)
        
        return output


class ComplexActivation(nn.Module):
    """
    Wrapper class for complex activations.
    
    Provides a unified interface to select between different
    complex activation functions.
    """
    
    def __init__(self, activation_type: str = 'crelu', num_channels: Optional[int] = None):
        """
        Args:
            activation_type: Type of activation ('crelu' or 'modrelu')
            num_channels: Number of channels (required for 'modrelu')
        """
        super(ComplexActivation, self).__init__()
        
        if activation_type.lower() == 'crelu':
            self.activation = CReLU()
        elif activation_type.lower() == 'modrelu':
            if num_channels is None:
                raise ValueError("num_channels is required for ModReLU")
            self.activation = ModReLU(num_channels)
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the selected complex activation.
        
        Args:
            x: Input tensor of shape (Batch, 2, Channels, Height, Width)
        
        Returns:
            Activated output tensor
        """
        return self.activation(x)
  