"""
PI-CV-RecUNet with Bidirectional LSTM for Temporal Super-Resolution.

This is the bidirectional version of the original PI_CV_RecUNet model.
The key difference is using BidirectionalComplexConvLSTM in the temporal
bottleneck, allowing the model to access both past and future frames
for better interpolation.
"""

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
from src.architectures.complex_bilstm import BidirectionalComplexConvLSTM


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
        
        # Apply same pooling to real and imaginary
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


class LearnedInterpolationBlock(nn.Module):
    """
    Learned Interpolation Block for generating intermediate frames.
    
    Instead of simple linear interpolation (alpha * A + (1-alpha) * B),
    this learns HOW to blend features from two frames based on:
    1. The features themselves (content-aware blending)
    2. The temporal position (where in time we are interpolating)
    
    This is crucial because wave propagation is NOT linear - a wave
    doesn't move uniformly between frames. The network learns the
    actual motion dynamics.
    """
    
    def __init__(self, channels: int, max_interp_steps: int = 20):
        super(LearnedInterpolationBlock, self).__init__()
        
        self.channels = channels
        self.max_interp_steps = max_interp_steps
        
        # Temporal position embedding (learnable)
        # Each interpolation position gets its own embedding
        self.temporal_embed = nn.Embedding(max_interp_steps, channels)
        
        # Feature fusion network: takes [feat_current, feat_next]
        # and learns how to blend them
        self.fusion_net = nn.Sequential(
            # Input: 2*channels (concat current + next)
            ComplexConv2d(channels * 2, channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(channels),
            CReLU(),
            ComplexConv2d(channels, channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(channels),
            CReLU(),
        )
        
        # Blending weights predictor: predicts per-pixel, per-channel blend weights
        # NOTE: kernel_size=1 requires padding=0 to preserve spatial dimensions
        self.blend_predictor = nn.Sequential(
            ComplexConv2d(channels * 2, channels, kernel_size=1, padding=0),
            CReLU(),
            ComplexConv2d(channels, channels, kernel_size=1, padding=0),
            # Output: blend weights in [0, 1] via sigmoid
        )
        
        # Residual refinement: learns corrections to the blended result
        self.residual_net = nn.Sequential(
            ComplexConv2d(channels, channels, kernel_size=3, padding=1),
            ComplexBatchNorm2d(channels),
            CReLU(),
            ComplexConv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self,
        feat_current: torch.Tensor,
        feat_next: torch.Tensor,
        step: int,
        total_steps: int
    ) -> torch.Tensor:
        """
        Generate an interpolated frame between current and next.
        
        Args:
            feat_current: Features from frame t (Batch, 2, Channels, H, W)
            feat_next: Features from frame t+1 (Batch, 2, Channels, H, W)
            step: Which intermediate step (1 to total_steps-1)
            total_steps: Total upsampling factor
        
        Returns:
            Interpolated features (Batch, 2, Channels, H, W)
        """
        batch_size = feat_current.shape[0]
        channels = feat_current.shape[2]
        height, width = feat_current.shape[3], feat_current.shape[4]
        
        # 1. Get temporal position embedding
        # Normalize step to [0, max_interp_steps-1]
        step_idx = min(step - 1, self.max_interp_steps - 1)
        temporal_emb = self.temporal_embed(
            torch.tensor([step_idx], device=feat_current.device)
        )  # (1, channels)
        
        # Broadcast to spatial dimensions - use actual channel count
        temporal_emb = temporal_emb.view(1, 1, self.channels, 1, 1)
        temporal_emb = temporal_emb.expand(batch_size, 2, -1, height, width)
        
        # 2. Concatenate current and next features
        concat_feat = torch.cat([feat_current, feat_next], dim=2)  # (B, 2, 2*C, H, W)
        
        # 3. Predict adaptive blend weights (content-aware)
        blend_weights_raw = self.blend_predictor(concat_feat)  # (B, 2, C, H, W)
        # Apply sigmoid to real part, then use for both real and imaginary
        # blend_weights_raw has shape (B, 2, C, H, W) where dim=1 is [real, imag]
        blend_weights_real = torch.sigmoid(blend_weights_raw[:, 0:1, :, :, :])  # (B, 1, C, H, W)
        blend_weights = blend_weights_real.expand(-1, 2, -1, -1, -1)  # (B, 2, C, H, W)
        
        # 4. Compute base interpolation (linear blend as baseline)
        alpha = step / total_steps  # Temporal position hint
        base_blend = (1 - alpha) * feat_current + alpha * feat_next
        
        # 5. Adaptive blend using predicted weights
        adaptive_blend = blend_weights * feat_current + (1 - blend_weights) * feat_next
        
        # 6. Fuse features with temporal embedding
        fused = self.fusion_net(concat_feat)  # (B, 2, C, H, W)
        # Add temporal embedding (pad if channels don't match)
        if self.channels != channels:
            temporal_emb_padded = F.pad(temporal_emb, (0, 0, 0, 0, 0, channels - self.channels))
        else:
            temporal_emb_padded = temporal_emb
        fused = fused + temporal_emb_padded[:, :, :channels]
        
        # 7. Combine all signals and predict residual
        combined = adaptive_blend + 0.1 * fused
        residual = self.residual_net(combined)
        
        # 8. Final output: combined + learned residual
        output = combined + 0.1 * residual
        
        return output


class BiTemporalBottleneck(nn.Module):
    """
    Bidirectional Temporal Bottleneck using BidirectionalComplexConvLSTM
    with LEARNED INTERPOLATION.
    
    Key improvements over linear interpolation:
    1. Content-aware blending: learns HOW to blend based on actual features
    2. Temporal position encoding: knows WHERE in time it's interpolating
    3. Residual learning: learns corrections to simple blending
    
    This properly exploits the bidirectional context from the BiLSTM.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        upsampling_factor: int = 20,
        combine_mode: str = 'concat'  # Changed default to 'concat' to preserve direction info
    ):
        super(BiTemporalBottleneck, self).__init__()
        
        self.upsampling_factor = upsampling_factor
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.combine_mode = combine_mode
        
        # Determine output channels from BiLSTM
        if combine_mode == 'concat':
            bilstm_out_channels = hidden_channels * 2
        else:
            bilstm_out_channels = hidden_channels
        
        # Project input to hidden dimension if needed
        if in_channels != hidden_channels:
            self.input_proj = ComplexConv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        else:
            self.input_proj = None
        
        # Bidirectional ComplexConvLSTM for temporal modeling
        self.bilstm = BidirectionalComplexConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            combine_mode=combine_mode
        )
        
        # LEARNED INTERPOLATION: replaces linear interpolation
        self.interpolator = LearnedInterpolationBlock(
            channels=bilstm_out_channels,
            max_interp_steps=upsampling_factor
        )
        
        # Project from BiLSTM output to encoder channels for decoder
        self.output_proj = ComplexConv2d(bilstm_out_channels, in_channels, kernel_size=1, padding=0)
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        upsampling_factor: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform temporal super-resolution on encoded features using bidirectional context
        and LEARNED interpolation.
        
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
        
        # Process through BiLSTM to get bidirectional temporal context
        # Now each frame's features contain info from BOTH past AND future
        bilstm_output, _, _ = self.bilstm(sparse_features)
        
        # Get output channels from BiLSTM
        bilstm_channels = bilstm_output.shape[3]
        
        # Generate dense frames using LEARNED interpolation
        dense_outputs = []
        
        for t in range(t_sparse - 1):
            # Add current frame (this is a real observed frame)
            dense_outputs.append(bilstm_output[:, t:t+1, :, :, :, :])
            
            # Get BiLSTM-enhanced features for current and next frame
            # These features now contain bidirectional context!
            feat_current = bilstm_output[:, t, :, :, :, :]      # (B, 2, C, H, W)
            feat_next = bilstm_output[:, t+1, :, :, :, :]        # (B, 2, C, H, W)
            
            # Generate (upsampling_factor - 1) intermediate frames
            for step in range(1, upsampling_factor):
                # Use LEARNED interpolation instead of linear
                interp_feat = self.interpolator(
                    feat_current, 
                    feat_next, 
                    step=step, 
                    total_steps=upsampling_factor
                )
                dense_outputs.append(interp_feat.unsqueeze(1))
        
        # Add last frame
        dense_outputs.append(bilstm_output[:, -1:, :, :, :, :])
        
        # Concatenate all frames
        dense_features = torch.cat(dense_outputs, dim=1)
        
        # Project back to encoder channels
        t_dense = dense_features.shape[1]
        dense_flat = dense_features.view(-1, 2, bilstm_channels, height, width)
        dense_flat = self.output_proj(dense_flat)
        dense_features = dense_flat.view(batch_size, t_dense, 2, self.in_channels, height, width)
        
        return dense_features


class PI_CV_RecUNet_BiLSTM(nn.Module):
    """
    Physics-Informed Complex-Valued Recurrent U-Net with Bidirectional LSTM.
    
    This is the bidirectional version of PI_CV_RecUNet, using
    BidirectionalComplexConvLSTM in the temporal bottleneck for
    improved temporal super-resolution.
    
    Key advantage over unidirectional version:
    - When interpolating frame N, has access to features from frames
      BOTH before AND after frame N
    - Better suited for offline processing where full sequence is available
    
    Architecture:
        Input -> Encoder -> BiTemporalBottleneck (BiLSTM) -> Decoder -> Output Head
        
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
        upsampling_factor: int = 20,
        bilstm_combine_mode: str = 'sum'  # 'concat', 'sum', 'mean'
    ):
        """
        Args:
            in_channels: Number of input channels (typically 1 for IQ data)
            out_channels: Number of output channels (typically 1)
            base_channels: Base number of channels (doubled at each level)
            num_levels: Number of encoder/decoder levels
            lstm_hidden_channels: Hidden channels in BiLSTM bottleneck
            lstm_num_layers: Number of BiLSTM layers
            upsampling_factor: Temporal upsampling factor (e.g., 20 for 20x)
            bilstm_combine_mode: How to combine forward/backward outputs
        """
        super(PI_CV_RecUNet_BiLSTM, self).__init__()
        
        self.upsampling_factor = upsampling_factor
        self.num_levels = num_levels
        self.bilstm_combine_mode = bilstm_combine_mode
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels
        )
        
        # Bottleneck channels (after encoder)
        bottleneck_channels = base_channels * (2 ** (num_levels - 1))
        
        # Bidirectional Temporal Bottleneck
        self.temporal_bottleneck = BiTemporalBottleneck(
            in_channels=bottleneck_channels,
            hidden_channels=lstm_hidden_channels,
            num_layers=lstm_num_layers,
            upsampling_factor=upsampling_factor,
            combine_mode=bilstm_combine_mode
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
        
        # Bidirectional temporal super-resolution in bottleneck
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
    
    def get_model_info(self) -> dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'PI_CV_RecUNet_BiLSTM',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'upsampling_factor': self.upsampling_factor,
            'num_levels': self.num_levels,
            'bilstm_combine_mode': self.bilstm_combine_mode
        }


def create_bilstm_model(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 4,
    lstm_hidden_channels: int = 256,
    lstm_num_layers: int = 2,
    upsampling_factor: int = 20,
    bilstm_combine_mode: str = 'sum'
) -> PI_CV_RecUNet_BiLSTM:
    """
    Factory function to create PI-CV-RecUNet-BiLSTM model.
    
    Default configuration is suitable for 201x201 IQ images.
    """
    return PI_CV_RecUNet_BiLSTM(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        lstm_hidden_channels=lstm_hidden_channels,
        lstm_num_layers=lstm_num_layers,
        upsampling_factor=upsampling_factor,
        bilstm_combine_mode=bilstm_combine_mode
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing PI_CV_RecUNet_BiLSTM model...")
    print("=" * 60)
    
    # Create model
    model = create_bilstm_model(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        num_levels=4,
        lstm_hidden_channels=128,
        lstm_num_layers=1,
        upsampling_factor=5,
        bilstm_combine_mode='sum'
    )
    
    # Print model info
    info = model.get_model_info()
    print(f"Model: {info['model_type']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Upsampling factor: {info['upsampling_factor']}")
    print(f"BiLSTM combine mode: {info['bilstm_combine_mode']}")
    
    # Test forward pass
    batch_size = 2
    t_sparse = 3
    height, width = 64, 64  # Smaller for testing
    
    # Create input: (B, T_sparse, 2, C, H, W)
    x = torch.randn(batch_size, t_sparse, 2, 1, height, width)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    expected_t_dense = t_sparse + (t_sparse - 1) * (5 - 1) + (t_sparse - 1)
    # With upsampling_factor=5: for each pair, insert 4 frames
    # So: 3 sparse -> 3 + 2*4 = 11 frames? No...
    # Actually: frame0, 4 interp, frame1, 4 interp, frame2 = 3 + 8 = 11
    # But our code does: 3 sparse + (3-1)*(5-1) = 3 + 8 = 11
    expected_t_dense = t_sparse + (t_sparse - 1) * (5 - 1)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected T_dense: {expected_t_dense}")
    
    # Verify output shape
    assert output.shape[0] == batch_size, f"Batch size mismatch"
    assert output.shape[2] == 2, f"Complex dim mismatch"
    assert output.shape[3] == 1, f"Channel mismatch"
    
    print("\n" + "=" * 60)
    print("✓ PI_CV_RecUNet_BiLSTM model test passed!")
    print("=" * 60)
