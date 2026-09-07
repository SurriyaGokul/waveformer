import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class MultiScaleFlowEstimator(nn.Module):
    """
    Multi-scale flow estimator that works at multiple resolutions.
    
    For shear wave data where motion is smooth and coherent, we:
    1. Upsample features to higher resolution for finer flow estimation
    2. Estimate flow at higher resolution (captures sub-pixel motion better)
    3. Downsample flow back to original resolution
    4. Use residual flow prediction for stability
    
    Key improvement: Estimating flow at 2x or 4x resolution allows capturing
    finer motion details that are lost at 8x8 bottleneck resolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        flow_scale: float = 0.5,  # Reduced from 1.0 for stability
        use_residual_flow: bool = True,
        upsample_factor: int = 4  # Estimate flow at 4x resolution (8x8 -> 32x32)
    ):
        """
        Args:
            in_channels: Number of feature channels.
            hidden_channels: Number of hidden channels.
            flow_scale: Scaling factor for flow output (smaller = more stable).
            use_residual_flow: If True, predict flow residual on top of zero init.
            upsample_factor: Factor to upsample features before flow estimation.
        """
        super(MultiScaleFlowEstimator, self).__init__()
        
        self.flow_scale = flow_scale
        self.use_residual_flow = use_residual_flow
        self.upsample_factor = upsample_factor
        
        # Input: 2 frames * 2 (real + imag) * C
        total_in_channels = 4 * in_channels
        
        # Feature compression before upsampling (reduce channels for efficiency)
        self.feature_compress = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Flow estimation at higher resolution
        self.flow_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Flow prediction head with smoothing (at high resolution)
        self.flow_head = nn.Sequential(
            # 5x5 kernel for spatial smoothness (important for coherent shear waves)
            nn.Conv2d(hidden_channels // 2, 4, kernel_size=5, padding=2),
        )
        
        # Flow magnitude constraint (learnable)
        self.max_flow = nn.Parameter(torch.tensor(4.0))  # Increased for higher resolution
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable flow estimation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize flow head output to zero for stable start
        for m in self.flow_head:
            if isinstance(m, nn.Conv2d) and m.out_channels == 4:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate bidirectional flow between two consecutive frames.
        
        Process:
        1. Compress and concatenate frame features
        2. Upsample to higher resolution (e.g., 8x8 -> 32x32)
        3. Estimate flow at high resolution
        4. Downsample flow back to original resolution
        
        Args:
            frame_t: Feature map at time t, shape (B, 2, C, H, W)
            frame_t1: Feature map at time t+1, shape (B, 2, C, H, W)
        
        Returns:
            forward_flow: Flow from t to t+1, shape (B, 2, H, W) at original resolution
            backward_flow: Flow from t+1 to t, shape (B, 2, H, W) at original resolution
        """
        B, _, C, H, W = frame_t.shape
        
        # Flatten complex dimension with channels: (B, 2*C, H, W)
        frame_t_flat = frame_t.view(B, 2 * C, H, W)
        frame_t1_flat = frame_t1.view(B, 2 * C, H, W)
        
        # Concatenate frames: (B, 4*C, H, W)
        concat_frames = torch.cat([frame_t_flat, frame_t1_flat], dim=1)
        
        # Compress features
        compressed = self.feature_compress(concat_frames)  # (B, hidden, H, W)
        
        # Upsample to higher resolution for finer flow estimation
        H_high = H * self.upsample_factor
        W_high = W * self.upsample_factor
        compressed_high = F.interpolate(
            compressed, 
            size=(H_high, W_high), 
            mode='bilinear', 
            align_corners=True
        )
        
        # Estimate flow at high resolution
        flow_features = self.flow_conv(compressed_high)
        flow_output_high = self.flow_head(flow_features)  # (B, 4, H_high, W_high)
        
        # Downsample flow back to original resolution
        # Scale flow values by upsample_factor since we're downsampling
        flow_output = F.interpolate(
            flow_output_high,
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ) / self.upsample_factor  # Scale flow magnitude appropriately
        
        # Split into forward and backward flow
        forward_flow = flow_output[:, :2, :, :]  # (B, 2, H, W)
        backward_flow = flow_output[:, 2:, :, :]  # (B, 2, H, W)
        
        # Apply soft magnitude constraint with learnable max
        max_flow = torch.abs(self.max_flow)
        forward_flow = max_flow * torch.tanh(forward_flow * self.flow_scale)
        backward_flow = max_flow * torch.tanh(backward_flow * self.flow_scale)
        
        return forward_flow, backward_flow
    
    def get_flow_at_high_res(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get flow at high resolution (for visualization/analysis).
        
        Returns flow at upsampled resolution without downsampling.
        """
        B, _, C, H, W = frame_t.shape
        
        frame_t_flat = frame_t.view(B, 2 * C, H, W)
        frame_t1_flat = frame_t1.view(B, 2 * C, H, W)
        concat_frames = torch.cat([frame_t_flat, frame_t1_flat], dim=1)
        
        compressed = self.feature_compress(concat_frames)
        
        H_high = H * self.upsample_factor
        W_high = W * self.upsample_factor
        compressed_high = F.interpolate(
            compressed, size=(H_high, W_high), mode='bilinear', align_corners=True
        )
        
        flow_features = self.flow_conv(compressed_high)
        flow_output_high = self.flow_head(flow_features)
        
        forward_flow_high = flow_output_high[:, :2, :, :]
        backward_flow_high = flow_output_high[:, 2:, :, :]
        
        max_flow = torch.abs(self.max_flow)
        forward_flow_high = max_flow * torch.tanh(forward_flow_high * self.flow_scale)
        backward_flow_high = max_flow * torch.tanh(backward_flow_high * self.flow_scale)
        
        return forward_flow_high, backward_flow_high
    
    def get_flow_stats(self, forward_flow: torch.Tensor) -> dict:
        """Get flow statistics for monitoring."""
        with torch.no_grad():
            flow_mag = torch.sqrt(forward_flow[:, 0]**2 + forward_flow[:, 1]**2)
            return {
                'flow_mean': flow_mag.mean().item(),
                'flow_max': flow_mag.max().item(),
                'flow_std': flow_mag.std().item(),
                'max_flow_param': torch.abs(self.max_flow).item()
            }


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion network that blends linear interpolation with motion-warped results.
    
    Key insight: For shear wave data, linear interpolation is already a good baseline.
    We learn to ADD motion-based corrections where needed.
    """
    
    def __init__(
        self,
        in_channels: int,
        use_residual_fusion: bool = True
    ):
        """
        Args:
            in_channels: Number of feature channels.
            use_residual_fusion: If True, learn residual on top of linear interp.
        """
        super(AdaptiveFusion, self).__init__()
        
        self.use_residual_fusion = use_residual_fusion
        
        # Input: linear_interp (2*C) + motion_result (2*C) + alpha (1)
        fusion_in_channels = 4 * in_channels + 1
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
        # Initialize to prefer linear interpolation initially
        nn.init.zeros_(self.fusion_conv[-1].weight)
        nn.init.constant_(self.fusion_conv[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12, mostly linear
    
    def forward(
        self,
        linear_result: torch.Tensor,
        motion_result: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Adaptively blend linear and motion-based interpolation.
        
        Args:
            linear_result: Linear interpolated features (B, 2, C, H, W)
            motion_result: Motion-warped features (B, 2, C, H, W)
            alpha: Interpolation position in [0, 1]
        
        Returns:
            fused: Blended result (B, 2, C, H, W)
        """
        B, _, C, H, W = linear_result.shape
        
        # Flatten complex features
        linear_flat = linear_result.view(B, 2 * C, H, W)
        motion_flat = motion_result.view(B, 2 * C, H, W)
        
        # Create alpha channel
        alpha_map = torch.full((B, 1, H, W), alpha,
                               device=linear_result.device,
                               dtype=linear_result.dtype)
        
        # Concatenate inputs
        fusion_input = torch.cat([linear_flat, motion_flat, alpha_map], dim=1)
        
        # Compute fusion weight (how much motion to use)
        motion_weight_logit = self.fusion_conv(fusion_input)  # (B, 1, H, W)
        motion_weight = torch.sigmoid(motion_weight_logit).unsqueeze(2)  # (B, 1, 1, H, W)
        
        if self.use_residual_fusion:
            # Residual approach: linear + weight * (motion - linear)
            fused = linear_result + motion_weight * (motion_result - linear_result)
        else:
            # Standard blending
            fused = (1 - motion_weight) * linear_result + motion_weight * motion_result
        
        return fused, motion_weight.squeeze(2)


def warp_complex_improved(
    feature_map: torch.Tensor,
    flow: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border'
) -> torch.Tensor:
    """
    Warp a complex feature map using optical flow.
    
    Improved version with better numerical stability.
    """
    B, _, C, H, W = feature_map.shape
    
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feature_map.device, dtype=feature_map.dtype),
        torch.linspace(-1, 1, W, device=feature_map.device, dtype=feature_map.dtype),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Convert flow from pixel units to normalized coordinates [-1, 1]
    flow_normalized = torch.zeros_like(base_grid)
    if W > 1:
        flow_normalized[..., 0] = flow[:, 0, :, :] * 2.0 / (W - 1)
    if H > 1:
        flow_normalized[..., 1] = flow[:, 1, :, :] * 2.0 / (H - 1)
    
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
    warped_feature = torch.stack([warped_real, warped_imag], dim=1)
    
    return warped_feature


class ResidualMotionInterpolation(nn.Module):
    """
    Residual Motion-Aware Interpolation for Shear Wave Data.
    
    Key design decisions based on shear wave characteristics:
    1. Linear interpolation as strong baseline (shear waves are smooth)
    2. Motion-based correction as learned residual
    3. Adaptive per-pixel blending of linear vs motion
    4. Flow magnitude constraint for stability
    5. Smooth flow fields (5x5 conv kernel)
    6. Multi-scale flow estimation at higher resolution
    
    Input Shape: (B, T_sparse, 2, C, H, W)
    Output Shape: (B, T_dense, 2, C, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        upsampling_factor: int = 20,
        flow_hidden_channels: int = 64,
        flow_scale: float = 0.5,
        use_residual_fusion: bool = True,
        warp_mode: str = 'bilinear',
        warp_padding_mode: str = 'border',
        flow_upsample_factor: int = 4  # Estimate flow at 4x resolution
    ):
        super(ResidualMotionInterpolation, self).__init__()
        
        self.in_channels = in_channels
        self.upsampling_factor = upsampling_factor
        self.warp_mode = warp_mode
        self.warp_padding_mode = warp_padding_mode
        
        # Multi-scale flow estimator with higher resolution
        self.flow_estimator = MultiScaleFlowEstimator(
            in_channels=in_channels,
            hidden_channels=flow_hidden_channels,
            flow_scale=flow_scale,
            use_residual_flow=True,
            upsample_factor=flow_upsample_factor
        )
        
        # Adaptive fusion network
        self.adaptive_fusion = AdaptiveFusion(
            in_channels=in_channels,
            use_residual_fusion=use_residual_fusion
        )
        
        # Store flows for loss computation
        self.cached_flows = []
        self.cached_frames = []
        self.cached_motion_weights = []
    
    def linear_interpolate(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Simple linear interpolation between frames."""
        return (1.0 - alpha) * frame_t + alpha * frame_t1
    
    def motion_interpolate(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Motion-based interpolation using warping."""
        # Scale flows by alpha
        scaled_forward = forward_flow * alpha
        scaled_backward = backward_flow * (1.0 - alpha)
        
        # Warp frames
        warped_forward = warp_complex_improved(
            frame_t, scaled_forward,
            mode=self.warp_mode,
            padding_mode=self.warp_padding_mode
        )
        warped_backward = warp_complex_improved(
            frame_t1, scaled_backward,
            mode=self.warp_mode,
            padding_mode=self.warp_padding_mode
        )
        
        # Simple soft blending for motion result
        return (1.0 - alpha) * warped_forward + alpha * warped_backward
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        upsampling_factor: Optional[int] = None,
        return_flows: bool = False
    ) -> torch.Tensor:
        """
        Perform residual motion-aware temporal interpolation.
        
        Args:
            sparse_features: Sparse feature maps of shape (B, T_sparse, 2, C, H, W)
            upsampling_factor: Optional override for the upsampling factor.
            return_flows: If True, also return estimated flows.
        
        Returns:
            dense_features: Dense feature maps of shape (B, T_dense, 2, C, H, W)
        """
        if upsampling_factor is None:
            upsampling_factor = self.upsampling_factor
        
        B, T_sparse, _, C, H, W = sparse_features.shape
        
        dense_outputs = []
        self.cached_flows = []
        self.cached_frames = []
        self.cached_motion_weights = []
        
        for t in range(T_sparse - 1):
            frame_t = sparse_features[:, t, :, :, :, :]  # (B, 2, C, H, W)
            frame_t1 = sparse_features[:, t + 1, :, :, :, :]  # (B, 2, C, H, W)
            
            # Add current frame
            dense_outputs.append(frame_t.unsqueeze(1))
            
            # Estimate bidirectional flow
            forward_flow, backward_flow = self.flow_estimator(frame_t, frame_t1)
            
            # Cache for loss computation
            self.cached_flows.append((forward_flow, backward_flow))
            self.cached_frames.append((frame_t, frame_t1))
            
            # Generate intermediate frames
            for step in range(1, upsampling_factor):
                alpha = step / upsampling_factor
                
                # Linear interpolation baseline
                linear_result = self.linear_interpolate(frame_t, frame_t1, alpha)
                
                # Motion-based interpolation
                motion_result = self.motion_interpolate(
                    frame_t, frame_t1,
                    forward_flow, backward_flow,
                    alpha
                )
                
                # Adaptive fusion
                fused, motion_weight = self.adaptive_fusion(
                    linear_result, motion_result, alpha
                )
                
                if step == upsampling_factor // 2:  # Cache midpoint weight for monitoring
                    self.cached_motion_weights.append(motion_weight)
                
                dense_outputs.append(fused.unsqueeze(1))
        
        # Add last frame
        dense_outputs.append(sparse_features[:, -1:, :, :, :, :])
        
        # Concatenate all frames
        dense_features = torch.cat(dense_outputs, dim=1)
        
        if return_flows:
            return dense_features, self.cached_flows
        
        return dense_features
    
    def get_cached_flows(self):
        """Get cached flows for loss computation."""
        return self.cached_flows
    
    def get_cached_frames(self):
        """Get cached frames for photometric loss."""
        return self.cached_frames
    
    def get_motion_weight_stats(self) -> dict:
        """Get statistics on motion weight usage."""
        if not self.cached_motion_weights:
            return {}
        
        weights = torch.stack(self.cached_motion_weights)
        return {
            'motion_weight_mean': weights.mean().item(),
            'motion_weight_max': weights.max().item(),
            'motion_weight_min': weights.min().item(),
        }


class ImprovedMotionAwareTemporalBottleneck(nn.Module):
    """
    Improved Motion-Aware Temporal Bottleneck.
    
    Key improvements:
    1. Residual motion interpolation (linear + learned motion correction)
    2. Multi-scale flow estimation at higher resolution
    3. Adaptive fusion with learnable blending
    4. Flow magnitude constraints for stability
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        upsampling_factor: int = 20,
        flow_hidden_channels: int = 64,
        flow_scale: float = 0.5,
        use_residual_fusion: bool = True,
        flow_upsample_factor: int = 4  # Flow estimation at 4x resolution
    ):
        super(ImprovedMotionAwareTemporalBottleneck, self).__init__()
        
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
        
        # Improved motion interpolation with multi-scale flow
        self.motion_interpolation = ResidualMotionInterpolation(
            in_channels=hidden_channels,
            upsampling_factor=upsampling_factor,
            flow_hidden_channels=flow_hidden_channels,
            flow_scale=flow_scale,
            use_residual_fusion=use_residual_fusion,
            flow_upsample_factor=flow_upsample_factor
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
        Perform improved motion-aware temporal super-resolution.
        
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
        
        # Apply improved motion interpolation
        dense_features = self.motion_interpolation(lstm_output, upsampling_factor)
        
        # Project back to encoder channels
        if self.output_proj is not None:
            batch_size, t_dense = dense_features.shape[:2]
            dense_flat = dense_features.view(-1, 2, self.hidden_channels, height, width)
            dense_flat = self.output_proj(dense_flat)
            dense_features = dense_flat.view(batch_size, t_dense, 2, self.in_channels, height, width)
        
        return dense_features
    
    def get_flow_estimator(self):
        """Get the flow estimator for external access."""
        return self.motion_interpolation.flow_estimator
    
    def get_motion_interpolation(self):
        """Get the motion interpolation module."""
        return self.motion_interpolation


def compute_flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute flow smoothness loss for coherent motion.
    
    For shear waves, flow should be spatially smooth.
    
    Args:
        flow: Flow field (B, 2, H, W)
    
    Returns:
        smoothness_loss: Scalar loss
    """
    # Gradient in x direction
    flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    # Gradient in y direction  
    flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    
    # L1 smoothness
    smoothness = torch.mean(torch.abs(flow_dx)) + torch.mean(torch.abs(flow_dy))
    
    return smoothness


def compute_flow_consistency_loss_improved(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor
) -> torch.Tensor:
    """
    Compute improved flow consistency loss.
    
    Forward and backward flows should be approximately opposite.
    
    Args:
        forward_flow: Flow from t to t+1, shape (B, 2, H, W)
        backward_flow: Flow from t+1 to t, shape (B, 2, H, W)
    
    Returns:
        consistency_loss: Scalar loss value
    """
    # Simple consistency: forward + backward should be approximately zero
    consistency_loss = torch.mean((forward_flow + backward_flow) ** 2)
    
    return consistency_loss


def compute_photometric_loss_improved(
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
    forward_flow: torch.Tensor
) -> torch.Tensor:
    """
    Compute photometric loss for flow supervision.
    
    Warping frame_t with forward_flow should approximate frame_t1.
    Uses Charbonnier loss for robustness.
    
    Args:
        frame_t: Feature map at time t (B, 2, C, H, W)
        frame_t1: Feature map at time t+1 (B, 2, C, H, W)
        forward_flow: Flow from t to t+1 (B, 2, H, W)
    
    Returns:
        photometric_loss: Scalar loss value
    """
    warped_t = warp_complex_improved(frame_t, forward_flow)
    
    # Charbonnier loss for robustness
    epsilon = 1e-6
    diff = warped_t - frame_t1
    loss = torch.mean(torch.sqrt(diff ** 2 + epsilon))
    
    return loss
