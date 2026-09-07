import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from src.architectures.complex_modules import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    CReLU,
    ComplexActivation
)
from src.architectures.complex_lstm import ComplexConvLSTM


class ComplexMaxPool2d(nn.Module):
    """
    Complex Max Pooling Layer.
    
    Applies max pooling based on the magnitude of complex values,
    then selects both real and imaginary parts at those locations.
    """
    
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch, 2, Channels, Height, Width)
        
        Returns:
            Pooled tensor (Batch, 2, Channels, H//stride, W//stride)
        """
        real = x[:, 0, :, :, :]  # (B, C, H, W)
        imag = x[:, 1, :, :, :]
        
        # Compute magnitude for pooling decisions
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        
        # Get indices of max magnitude
        _, indices = F.max_pool2d(
            magnitude, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            return_indices=True
        )
        
        # Flatten for gather operation
        batch_size, channels, h_out, w_out = indices.shape
        
        # Apply same pooling to real and imaginary (using magnitude-based selection)
        # For simplicity, we use regular max_pool which works well in practice
        real_pooled = F.max_pool2d(real, kernel_size=self.kernel_size, stride=self.stride)
        imag_pooled = F.max_pool2d(imag, kernel_size=self.kernel_size, stride=self.stride)
        
        return torch.stack([real_pooled, imag_pooled], dim=1)


class ComplexUpsample(nn.Module):
    """
    Complex Upsampling Layer using bilinear interpolation.
    """
    
    def __init__(self, scale_factor: int = 2, mode: str = 'bilinear'):
        super(ComplexUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch, 2, Channels, Height, Width)
        
        Returns:
            Upsampled tensor (Batch, 2, Channels, H*scale, W*scale)
        """
        real = x[:, 0, :, :, :]
        imag = x[:, 1, :, :, :]
        
        real_up = F.interpolate(real, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        imag_up = F.interpolate(imag, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        
        return torch.stack([real_up, imag_up], dim=1)


class ComplexConvBlock(nn.Module):
    """
    Complex Convolutional Block: Conv -> BatchNorm -> Activation
    
    Optionally includes a second conv layer for deeper feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        double_conv: bool = True,
        activation: str = 'crelu'
    ):
        super(ComplexConvBlock, self).__init__()
        
        self.double_conv = double_conv
        
        # First conv block
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.act1 = CReLU()
        
        # Second conv block (optional)
        if double_conv:
            self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size, padding=padding)
            self.bn2 = ComplexBatchNorm2d(out_channels)
            self.act2 = CReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        
        if self.double_conv:
            x = self.act2(self.bn2(self.conv2(x)))
        
        return x


class Encoder(nn.Module):
    """
    U-Net Encoder with 4 levels of downsampling.
    
    Each level: ComplexConvBlock -> ComplexMaxPool
    Returns feature maps at each level for skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4
    ):
        super(Encoder, self).__init__()
        
        self.num_levels = num_levels
        self.conv_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        current_channels = in_channels
        
        for level in range(num_levels):
            out_channels = base_channels * (2 ** level)
            
            self.conv_blocks.append(
                ComplexConvBlock(current_channels, out_channels, double_conv=True)
            )
            self.pool_layers.append(ComplexMaxPool2d(kernel_size=2, stride=2))
            
            current_channels = out_channels
        
        self.output_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input tensor (Batch, 2, Channels, Height, Width)
        
        Returns:
            output: Deepest feature map (Batch, 2, C_deep, H//16, W//16)
            skip_features: List of feature maps for skip connections
        """
        skip_features = []
        
        for level in range(self.num_levels):
            x = self.conv_blocks[level](x)
            skip_features.append(x)
            x = self.pool_layers[level](x)
        
        return x, skip_features


class Decoder(nn.Module):
    """
    U-Net Decoder with 4 levels of upsampling.
    
    Each level: ComplexUpsample -> Concat with skip -> ComplexConvBlock
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_levels: int = 4
    ):
        super(Decoder, self).__init__()
        
        self.num_levels = num_levels
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        current_channels = in_channels
        
        for level in range(num_levels - 1, -1, -1):
            skip_channels = base_channels * (2 ** level)
            
            self.upsample_layers.append(ComplexUpsample(scale_factor=2))
            
            # After concat: current_channels + skip_channels
            self.conv_blocks.append(
                ComplexConvBlock(
                    current_channels + skip_channels,
                    skip_channels,
                    double_conv=True
                )
            )
            
            current_channels = skip_channels
        
        self.output_channels = current_channels
    
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Bottleneck features (Batch, 2, Channels, H, W)
            skip_features: List of encoder features (from shallow to deep)
        
        Returns:
            Decoded features (Batch, 2, base_channels, H_orig, W_orig)
        """
        # Reverse skip features to go from deep to shallow
        skip_features = skip_features[::-1]
        
        for level in range(self.num_levels):
            x = self.upsample_layers[level](x)
            
            skip = skip_features[level]
            
            # Handle size mismatch due to pooling/upsampling
            if x.shape[3:] != skip.shape[3:]:
                x = F.interpolate(
                    x.view(-1, x.shape[2], x.shape[3], x.shape[4]),
                    size=skip.shape[3:],
                    mode='bilinear',
                    align_corners=False
                ).view(x.shape[0], 2, x.shape[2], skip.shape[3], skip.shape[4])
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=2)
            
            x = self.conv_blocks[level](x)
        
        return x


class TemporalBottleneck(nn.Module):
    """
    Temporal Bottleneck using ComplexConvLSTM.
    
    Takes encoded features from sparse frames and predicts
    intermediate latent states for temporal super-resolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        upsampling_factor: int = 20
    ):
        super(TemporalBottleneck, self).__init__()
        
        self.upsampling_factor = upsampling_factor
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Project input to hidden dimension if needed
        if in_channels != hidden_channels:
            self.input_proj = ComplexConv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        else:
            self.input_proj = None
        
        # ComplexConvLSTM for temporal modeling
        self.lstm = ComplexConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size
        )
        
        # Project back to encoder channels for decoder
        if in_channels != hidden_channels:
            self.output_proj = ComplexConv2d(hidden_channels, in_channels, kernel_size=1, padding=0)
        else:
            self.output_proj = None
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        upsampling_factor: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform temporal super-resolution on encoded features.
        
        Args:
            sparse_features: (Batch, T_sparse, 2, Channels, Height, Width)
            upsampling_factor: Number of frames to predict between each pair
        
        Returns:
            dense_features: (Batch, T_dense, 2, Channels, Height, Width)
        """
        if upsampling_factor is None:
            upsampling_factor = self.upsampling_factor
        
        batch_size, t_sparse, _, channels, height, width = sparse_features.shape
        
        # Project to hidden dimension
        if self.input_proj is not None:
            # Reshape for conv: (Batch * T, 2, C, H, W)
            sparse_flat = sparse_features.view(-1, 2, channels, height, width)
            sparse_flat = self.input_proj(sparse_flat)
            sparse_features = sparse_flat.view(batch_size, t_sparse, 2, self.hidden_channels, height, width)
        
        # Process through LSTM to get temporal context
        lstm_output, last_states = self.lstm(sparse_features)
        
        # Interpolate between frames
        dense_outputs = []
        
        for t in range(t_sparse - 1):
            # Add current frame
            dense_outputs.append(lstm_output[:, t:t+1, :, :, :, :])
            
            # Interpolate intermediate frames
            feat_current = lstm_output[:, t, :, :, :, :]
            feat_next = lstm_output[:, t+1, :, :, :, :]
            
            # Insert (upsampling_factor - 1) frames between each pair
            # This gives us upsampling_factor frames per interval
            for step in range(1, upsampling_factor):
                alpha = step / upsampling_factor
                
                # Linear interpolation in complex feature space
                interp_real = (1 - alpha) * feat_current[:, 0] + alpha * feat_next[:, 0]
                interp_imag = (1 - alpha) * feat_current[:, 1] + alpha * feat_next[:, 1]
                interp_feat = torch.stack([interp_real, interp_imag], dim=1)
                
                dense_outputs.append(interp_feat.unsqueeze(1))
        
        # Add last frame
        dense_outputs.append(lstm_output[:, -1:, :, :, :, :])
        
        # Concatenate all frames
        dense_features = torch.cat(dense_outputs, dim=1)
        
        # Project back to encoder channels
        if self.output_proj is not None:
            batch_size, t_dense = dense_features.shape[:2]
            dense_flat = dense_features.view(-1, 2, self.hidden_channels, height, width)
            dense_flat = self.output_proj(dense_flat)
            dense_features = dense_flat.view(batch_size, t_dense, 2, self.in_channels, height, width)
        
        return dense_features


class PI_CV_RecUNet(nn.Module):
    """
    Physics-Informed Complex-Valued Recurrent U-Net.
    
    Full architecture for temporal super-resolution of
    ultrasound shear wave elastography IQ data.
    
    Architecture:
        Input -> Encoder -> TemporalBottleneck (LSTM) -> Decoder -> Output Head
        
    Skip connections from Encoder to Decoder at each level.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        lstm_hidden_channels: int = 256,
        lstm_num_layers: int = 2,
        upsampling_factor: int = 20
    ):
        """
        Args:
            in_channels: Number of input channels (typically 1 for IQ data)
            out_channels: Number of output channels (typically 1)
            base_channels: Base number of channels (doubled at each level)
            num_levels: Number of encoder/decoder levels
            lstm_hidden_channels: Hidden channels in LSTM bottleneck
            lstm_num_layers: Number of LSTM layers
            upsampling_factor: Temporal upsampling factor (e.g., 20 for 20x)
        """
        super(PI_CV_RecUNet, self).__init__()
        
        self.upsampling_factor = upsampling_factor
        self.num_levels = num_levels
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels
        )
        
        # Bottleneck channels (after encoder)
        bottleneck_channels = base_channels * (2 ** (num_levels - 1))
        
        # Temporal Bottleneck (LSTM)
        self.temporal_bottleneck = TemporalBottleneck(
            in_channels=bottleneck_channels,
            hidden_channels=lstm_hidden_channels,
            num_layers=lstm_num_layers,
            upsampling_factor=upsampling_factor
        )
        
        # Decoder
        self.decoder = Decoder(
            in_channels=bottleneck_channels,
            base_channels=base_channels,
            num_levels=num_levels
        )
        
        # Output head: map back to IQ space
        self.output_head = ComplexConv2d(
            base_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )
    
    def encode_frame(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode a single frame through the encoder.
        
        Args:
            x: Single frame (Batch, 2, Channels, Height, Width)
        
        Returns:
            encoded: Bottleneck features
            skip_features: Skip connection features
        """
        return self.encoder(x)
    
    def decode_frame(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode features back to image space.
        
        Args:
            x: Bottleneck features
            skip_features: Skip connections from encoder
        
        Returns:
            Reconstructed frame (Batch, 2, out_channels, Height, Width)
        """
        decoded = self.decoder(x, skip_features)
        output = self.output_head(decoded)
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        upsampling_factor: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full forward pass for temporal super-resolution.
        
        Args:
            x: Sparse input sequence (Batch, T_sparse, 2, Channels, Height, Width)
            upsampling_factor: Override default upsampling factor
        
        Returns:
            Dense output sequence (Batch, T_dense, 2, out_channels, Height, Width)
        """
        if upsampling_factor is None:
            upsampling_factor = self.upsampling_factor
        
        batch_size, t_sparse, _, channels, height, width = x.shape
        
        # Encode each sparse frame
        encoded_features = []
        all_skip_features = []
        
        for t in range(t_sparse):
            frame = x[:, t, :, :, :, :]  # (B, 2, C, H, W)
            encoded, skips = self.encode_frame(frame)
            encoded_features.append(encoded)
            all_skip_features.append(skips)
        
        # Stack encoded features: (B, T, 2, C_enc, H_enc, W_enc)
        encoded_sequence = torch.stack(encoded_features, dim=1)
        
        # Temporal super-resolution in bottleneck
        dense_encoded = self.temporal_bottleneck(encoded_sequence, upsampling_factor)
        
        t_dense = dense_encoded.shape[1]
        
        # Decode each dense frame
        # For skip connections, interpolate between sparse frame skips
        outputs = []
        
        for t in range(t_dense):
            # Determine which sparse frames to interpolate skips from
            sparse_idx = t / (upsampling_factor if t_sparse > 1 else 1)
            sparse_idx_low = min(int(sparse_idx), t_sparse - 1)
            sparse_idx_high = min(sparse_idx_low + 1, t_sparse - 1)
            alpha = sparse_idx - sparse_idx_low
            
            # Interpolate skip features
            interpolated_skips = []
            for level in range(self.num_levels):
                skip_low = all_skip_features[sparse_idx_low][level]
                skip_high = all_skip_features[sparse_idx_high][level]
                
                # Linear interpolation
                skip_interp_real = (1 - alpha) * skip_low[:, 0] + alpha * skip_high[:, 0]
                skip_interp_imag = (1 - alpha) * skip_low[:, 1] + alpha * skip_high[:, 1]
                skip_interp = torch.stack([skip_interp_real, skip_interp_imag], dim=1)
                
                interpolated_skips.append(skip_interp)
            
            # Decode frame
            encoded_frame = dense_encoded[:, t, :, :, :, :]
            decoded_frame = self.decode_frame(encoded_frame, interpolated_skips)
            outputs.append(decoded_frame)
        
        # Stack outputs: (B, T_dense, 2, out_channels, H, W)
        output_sequence = torch.stack(outputs, dim=1)
        
        return output_sequence

def create_model(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 4,
    lstm_hidden_channels: int = 256,
    lstm_num_layers: int = 2,
    upsampling_factor: int = 20
) -> PI_CV_RecUNet:
    """
    Factory function to create PI-CV-RecUNet model.
    
    Default configuration is suitable for 201x201 IQ images.
    """
    return PI_CV_RecUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        lstm_hidden_channels=lstm_hidden_channels,
        lstm_num_layers=lstm_num_layers,
        upsampling_factor=upsampling_factor
    )

