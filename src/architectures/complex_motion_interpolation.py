import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FlowEstimator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        flow_scale: float = 1.0
    ):
        """
        Args:
            in_channels: Number of feature channels (C in complex tensor).
            hidden_channels: Number of hidden channels in the flow estimator.
            flow_scale: Scaling factor for the output flow (for numerical stability).
        """
        super(FlowEstimator, self).__init__()
        
        self.flow_scale = flow_scale
        
        # Input channels: 2 frames * 2 (real + imag) * C
        total_in_channels = 4 * in_channels
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(total_in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 2
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 3
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 4 - Output layer
            # 4 channels: (forward_flow_x, forward_flow_y, backward_flow_x, backward_flow_y)
            nn.Conv2d(hidden_channels // 2, 4, kernel_size=3, padding=1)
        )
        
        # Initialize last layer to output small flow values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable flow estimation."""
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize last layer with smaller weights for stability
        last_conv = self.conv_layers[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)
    
    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate bidirectional flow between two consecutive frames.
        
        Args:
            frame_t: Feature map at time t, shape (B, 2, C, H, W)
            frame_t1: Feature map at time t+1, shape (B, 2, C, H, W)
        
        Returns:
            forward_flow: Flow from t to t+1, shape (B, 2, H, W)
            backward_flow: Flow from t+1 to t, shape (B, 2, H, W)
        """
        B, _, C, H, W = frame_t.shape
        
        # Flatten complex dimension with channels: (B, 2*C, H, W)
        frame_t_flat = frame_t.view(B, 2 * C, H, W)
        frame_t1_flat = frame_t1.view(B, 2 * C, H, W)
        
        # Concatenate frames: (B, 4*C, H, W)
        concat_frames = torch.cat([frame_t_flat, frame_t1_flat], dim=1)
        
        # Estimate flow
        flow_output = self.conv_layers(concat_frames)  # (B, 4, H, W)
        
        # Split into forward and backward flow
        forward_flow = flow_output[:, :2, :, :]  # (B, 2, H, W)
        backward_flow = flow_output[:, 2:, :, :]  # (B, 2, H, W)
        
        # Apply flow scale
        forward_flow = forward_flow * self.flow_scale
        backward_flow = backward_flow * self.flow_scale
        
        return forward_flow, backward_flow


class FusionNet(nn.Module):
    """
    Learned Fusion Network for blending warped features.
    
    Takes the warped forward and backward features along with the original
    features and produces a soft blending mask (per-pixel, per-channel weights).
    
    This allows the network to adaptively blend based on occlusion,
    motion reliability, and feature content.
    """
    
    def __init__(
        self,
        in_channels: int,
        use_learned_fusion: bool = True
    ):
        """
        Args:
            in_channels: Number of feature channels (C in complex tensor).
            use_learned_fusion: If True, use learned fusion weights.
                               If False, use simple soft blending based on alpha.
        """
        super(FusionNet, self).__init__()
        
        self.use_learned_fusion = use_learned_fusion
        
        if use_learned_fusion:
            # Input: warped_forward (2*C) + warped_backward (2*C) + alpha (1)
            # Output: fusion weight map (1 channel, broadcast to all)
            fusion_in_channels = 4 * in_channels + 1
            
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_in_channels, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid()  # Output in [0, 1] for blending weight
            )
    
    def forward(
        self,
        warped_forward: torch.Tensor,
        warped_backward: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Compute fusion weights for blending warped features.
        
        Args:
            warped_forward: Forward-warped features (B, 2, C, H, W)
            warped_backward: Backward-warped features (B, 2, C, H, W)
            alpha: Interpolation position in [0, 1]
        
        Returns:
            fusion_weight: Blending weight for forward features (B, 1, 1, H, W)
                          (1 - fusion_weight) will be applied to backward features
        """
        B, _, C, H, W = warped_forward.shape
        
        if self.use_learned_fusion:
            # Flatten complex features
            wf_flat = warped_forward.view(B, 2 * C, H, W)
            wb_flat = warped_backward.view(B, 2 * C, H, W)
            
            # Create alpha channel
            alpha_map = torch.full((B, 1, H, W), alpha, 
                                   device=warped_forward.device,
                                   dtype=warped_forward.dtype)
            
            # Concatenate inputs
            fusion_input = torch.cat([wf_flat, wb_flat, alpha_map], dim=1)
            
            # Compute fusion weight
            fusion_weight = self.fusion_conv(fusion_input)  # (B, 1, H, W)
            fusion_weight = fusion_weight.unsqueeze(2)  # (B, 1, 1, H, W)
        else:
            # Simple alpha-based soft blending
            fusion_weight = torch.tensor(
                1.0 - alpha,
                device=warped_forward.device,
                dtype=warped_forward.dtype
            ).view(1, 1, 1, 1, 1)
        
        return fusion_weight


def warp_complex(
    feature_map: torch.Tensor,
    flow: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border'
) -> torch.Tensor:
    """
    Warp a complex feature map using optical flow.
    
    This function warps both Real and Imaginary components of a complex
    feature map using the same flow field via grid_sample.
    
    Args:
        feature_map: Complex feature map of shape (B, 2, C, H, W)
                    where dim=1 is (Real, Imaginary)
        flow: Optical flow field of shape (B, 2, H, W)
              where dim=1 is (flow_x, flow_y) in pixel units
        mode: Interpolation mode for grid_sample ('bilinear' or 'nearest')
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')
    
    Returns:
        warped_feature: Warped complex feature map of shape (B, 2, C, H, W)
    """
    B, _, C, H, W = feature_map.shape
    
    # Create base grid
    # grid_sample expects grid in [-1, 1] range
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feature_map.device, dtype=feature_map.dtype),
        torch.linspace(-1, 1, W, device=feature_map.device, dtype=feature_map.dtype),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Convert flow from pixel units to normalized coordinates [-1, 1]
    # flow[:, 0] is x-direction (horizontal), flow[:, 1] is y-direction (vertical)
    flow_normalized = torch.zeros_like(base_grid)
    flow_normalized[..., 0] = flow[:, 0, :, :] * 2.0 / (W - 1) if W > 1 else flow[:, 0, :, :]
    flow_normalized[..., 1] = flow[:, 1, :, :] * 2.0 / (H - 1) if H > 1 else flow[:, 1, :, :]
    
    # Add flow to base grid
    sample_grid = base_grid + flow_normalized  # (B, H, W, 2)
    
    # Warp Real component
    real_component = feature_map[:, 0, :, :, :]  # (B, C, H, W)
    warped_real = F.grid_sample(
        real_component,
        sample_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    
    # Warp Imaginary component
    imag_component = feature_map[:, 1, :, :, :]  # (B, C, H, W)
    warped_imag = F.grid_sample(
        imag_component,
        sample_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )
    
    # Stack back to complex format
    warped_feature = torch.stack([warped_real, warped_imag], dim=1)  # (B, 2, C, H, W)
    
    return warped_feature


class ComplexMotionInterpolation(nn.Module):
    """
    Complex Motion-Aware Interpolation Module.
    
    This module performs temporal upsampling of complex-valued feature maps
    using optical flow-based warping instead of simple linear interpolation.
    
    The process for each pair of consecutive frames:
        1. Estimate bidirectional flow between F_t and F_{t+1}
        2. For each intermediate step i in [1, k-1]:
            a. Compute intermediate time alpha = i/k
            b. Scale forward flow by alpha and backward flow by (1-alpha)
            c. Warp F_t forward and F_{t+1} backward
            d. Fuse warped features using learned or soft blending
    
    Input Shape: (B, T_sparse, 2, C, H, W)
    Output Shape: (B, T_dense, 2, C, H, W)
    
    where T_dense = (T_sparse - 1) * k + 1
    """
    
    def __init__(
        self,
        in_channels: int,
        upsampling_factor: int = 20,
        flow_hidden_channels: int = 64,
        flow_scale: float = 1.0,
        use_learned_fusion: bool = True,
        warp_mode: str = 'bilinear',
        warp_padding_mode: str = 'border'
    ):
        """
        Args:
            in_channels: Number of feature channels (C in complex tensor).
            upsampling_factor: Temporal upsampling factor k.
            flow_hidden_channels: Hidden channels in the flow estimator.
            flow_scale: Scaling factor for flow output.
            use_learned_fusion: If True, use learned fusion network.
                               If False, use simple alpha-based blending.
            warp_mode: Interpolation mode for grid_sample.
            warp_padding_mode: Padding mode for grid_sample.
        """
        super(ComplexMotionInterpolation, self).__init__()
        
        self.in_channels = in_channels
        self.upsampling_factor = upsampling_factor
        self.warp_mode = warp_mode
        self.warp_padding_mode = warp_padding_mode
        
        # Flow estimator network
        self.flow_estimator = FlowEstimator(
            in_channels=in_channels,
            hidden_channels=flow_hidden_channels,
            flow_scale=flow_scale
        )
        
        # Fusion network
        self.fusion_net = FusionNet(
            in_channels=in_channels,
            use_learned_fusion=use_learned_fusion
        )
    
    def interpolate_pair(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate a single intermediate frame between two consecutive frames.
        
        Args:
            frame_t: Feature map at time t, shape (B, 2, C, H, W)
            frame_t1: Feature map at time t+1, shape (B, 2, C, H, W)
            forward_flow: Flow from t to t+1, shape (B, 2, H, W)
            backward_flow: Flow from t+1 to t, shape (B, 2, H, W)
            alpha: Interpolation position in (0, 1)
        
        Returns:
            interpolated_frame: Interpolated feature map (B, 2, C, H, W)
        """
        # Scale flows by alpha and (1-alpha)
        # Forward warping: move frame_t towards t+1 by alpha
        scaled_forward_flow = forward_flow * alpha
        # Backward warping: move frame_t1 towards t by (1-alpha)
        scaled_backward_flow = backward_flow * (1.0 - alpha)
        
        # Warp frames
        warped_forward = warp_complex(
            frame_t,
            scaled_forward_flow,
            mode=self.warp_mode,
            padding_mode=self.warp_padding_mode
        )
        
        warped_backward = warp_complex(
            frame_t1,
            scaled_backward_flow,
            mode=self.warp_mode,
            padding_mode=self.warp_padding_mode
        )
        
        # Fuse warped features
        fusion_weight = self.fusion_net(warped_forward, warped_backward, alpha)
        
        # Blend: higher weight on warped_forward when closer to t (lower alpha)
        # fusion_weight is for forward, (1 - fusion_weight) for backward
        interpolated_frame = fusion_weight * warped_forward + (1.0 - fusion_weight) * warped_backward
        
        return interpolated_frame
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        upsampling_factor: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform motion-aware temporal interpolation.
        
        Args:
            sparse_features: Sparse feature maps of shape (B, T_sparse, 2, C, H, W)
            upsampling_factor: Optional override for the upsampling factor.
        
        Returns:
            dense_features: Dense feature maps of shape (B, T_dense, 2, C, H, W)
                           where T_dense = (T_sparse - 1) * k + 1
        """
        if upsampling_factor is None:
            upsampling_factor = self.upsampling_factor
        
        B, T_sparse, _, C, H, W = sparse_features.shape
        
        dense_outputs = []
        
        for t in range(T_sparse - 1):
            frame_t = sparse_features[:, t, :, :, :, :]  # (B, 2, C, H, W)
            frame_t1 = sparse_features[:, t + 1, :, :, :, :]  # (B, 2, C, H, W)
            
            # Add current frame
            dense_outputs.append(frame_t.unsqueeze(1))
            
            # Estimate bidirectional flow
            forward_flow, backward_flow = self.flow_estimator(frame_t, frame_t1)
            
            # Generate intermediate frames
            for step in range(1, upsampling_factor):
                alpha = step / upsampling_factor
                
                interpolated = self.interpolate_pair(
                    frame_t, frame_t1,
                    forward_flow, backward_flow,
                    alpha
                )
                
                dense_outputs.append(interpolated.unsqueeze(1))
        
        # Add last frame
        dense_outputs.append(sparse_features[:, -1:, :, :, :, :])
        
        # Concatenate all frames
        dense_features = torch.cat(dense_outputs, dim=1)
        
        return dense_features


class MotionAwareTemporalBottleneck(nn.Module):
    """
    Motion-Aware Temporal Bottleneck using ComplexConvLSTM and Motion Interpolation.
    
    This is a drop-in replacement for the original TemporalBottleneck that uses
    motion-aware interpolation instead of simple linear interpolation.
    
    The architecture:
        1. Optional input projection (if in_channels != hidden_channels)
        2. ComplexConvLSTM for temporal modeling
        3. ComplexMotionInterpolation for temporal upsampling
        4. Optional output projection (if in_channels != hidden_channels)
    
    Input: (B, T_sparse, 2, C, H, W)
    Output: (B, T_dense, 2, C, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        upsampling_factor: int = 20,
        flow_hidden_channels: int = 64,
        flow_scale: float = 1.0,
        use_learned_fusion: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels for LSTM.
            num_layers: Number of LSTM layers.
            kernel_size: Kernel size for LSTM convolutions.
            upsampling_factor: Temporal upsampling factor.
            flow_hidden_channels: Hidden channels in flow estimator.
            flow_scale: Scaling factor for flow output.
            use_learned_fusion: Whether to use learned fusion weights.
        """
        super(MotionAwareTemporalBottleneck, self).__init__()
        
        self.upsampling_factor = upsampling_factor
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Import here to avoid circular imports
        from src.architectures.complex_modules import ComplexConv2d
        from src.architectures.complex_lstm import ComplexConvLSTM
        
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
        
        # Motion-aware interpolation
        self.motion_interpolation = ComplexMotionInterpolation(
            in_channels=hidden_channels,
            upsampling_factor=upsampling_factor,
            flow_hidden_channels=flow_hidden_channels,
            flow_scale=flow_scale,
            use_learned_fusion=use_learned_fusion
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
        Perform motion-aware temporal super-resolution on encoded features.
        
        Args:
            sparse_features: (B, T_sparse, 2, C, H, W)
            upsampling_factor: Number of frames to predict between each pair.
        
        Returns:
            dense_features: (B, T_dense, 2, C, H, W)
        """
        if upsampling_factor is None:
            upsampling_factor = self.upsampling_factor
        
        batch_size, t_sparse, _, channels, height, width = sparse_features.shape
        
        # Project to hidden dimension
        if self.input_proj is not None:
            sparse_flat = sparse_features.view(-1, 2, channels, height, width)
            sparse_flat = self.input_proj(sparse_flat)
            sparse_features = sparse_flat.view(batch_size, t_sparse, 2, self.hidden_channels, height, width)
        
        # Process through LSTM to get temporal context
        lstm_output, _ = self.lstm(sparse_features)
        
        # Apply motion-aware interpolation
        dense_features = self.motion_interpolation(lstm_output, upsampling_factor)
        
        # Project back to encoder channels
        if self.output_proj is not None:
            batch_size, t_dense = dense_features.shape[:2]
            dense_flat = dense_features.view(-1, 2, self.hidden_channels, height, width)
            dense_flat = self.output_proj(dense_flat)
            dense_features = dense_flat.view(batch_size, t_dense, 2, self.in_channels, height, width)
        
        return dense_features

def compute_flow_consistency_loss(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor
) -> torch.Tensor:
    """
    Compute flow consistency loss for regularization.
    
    This loss encourages the forward and backward flows to be consistent,
    i.e., warping with forward flow and then backward flow should return
    to the original position.
    
    Args:
        forward_flow: Flow from t to t+1, shape (B, 2, H, W)
        backward_flow: Flow from t+1 to t, shape (B, 2, H, W)
        frame_t: Feature map at time t (B, 2, C, H, W)
        frame_t1: Feature map at time t+1 (B, 2, C, H, W)
    
    Returns:
        consistency_loss: Scalar loss value
    """
    # Warp forward flow by backward flow
    B, _, H, W = forward_flow.shape
    
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=forward_flow.device, dtype=forward_flow.dtype),
        torch.linspace(-1, 1, W, device=forward_flow.device, dtype=forward_flow.dtype),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Normalize backward flow
    flow_normalized = torch.zeros_like(base_grid)
    flow_normalized[..., 0] = backward_flow[:, 0, :, :] * 2.0 / (W - 1) if W > 1 else backward_flow[:, 0, :, :]
    flow_normalized[..., 1] = backward_flow[:, 1, :, :] * 2.0 / (H - 1) if H > 1 else backward_flow[:, 1, :, :]
    
    sample_grid = base_grid + flow_normalized
    
    # Warp forward flow
    warped_forward_flow = F.grid_sample(
        forward_flow,
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    # Sum of flows should be close to zero for consistency
    consistency_loss = torch.mean((forward_flow + warped_forward_flow) ** 2)
    
    return consistency_loss


def compute_photometric_loss(
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
    forward_flow: torch.Tensor
) -> torch.Tensor:
    """
    Compute photometric loss for flow supervision.
    
    Warping frame_t with forward_flow should reconstruct frame_t1.
    
    Args:
        frame_t: Feature map at time t (B, 2, C, H, W)
        frame_t1: Feature map at time t+1 (B, 2, C, H, W)
        forward_flow: Flow from t to t+1 (B, 2, H, W)
    
    Returns:
        photometric_loss: Scalar loss value
    """
    warped_t = warp_complex(frame_t, forward_flow)
    
    # L1 loss between warped frame_t and frame_t1
    loss = torch.mean(torch.abs(warped_t - frame_t1))
    
    return loss  