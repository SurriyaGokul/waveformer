import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class HighResFlowEstimator(nn.Module):
    """
    Flow estimator that operates on high-resolution input frames.
    
    Takes original input frames (e.g., 128 × 128) and estimates optical flow
    at full resolution, capturing true pixel-level motion.
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # Input image channels (1 for grayscale)
        hidden_channels: int = 32,
        flow_scale: float = 0.1,  # Conservative flow scaling
    ):
        """
        Args:
            in_channels: Number of input image channels (typically 1).
            hidden_channels: Number of hidden channels in flow network.
            flow_scale: Scaling factor for flow output (smaller = more stable).
        """
        super(HighResFlowEstimator, self).__init__()
        
        self.flow_scale = flow_scale
        
        # Input: 2 frames × 2 (real + imag) × C_in
        total_in_channels = 4 * in_channels
        
        # Multi-scale feature extraction with downsampling for efficiency
        # This is a lightweight encoder for flow estimation
        self.encoder = nn.Sequential(
            # Level 1: Full resolution
            nn.Conv2d(total_in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Downsample for context (128 → 64 → 32)
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Upsample back with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Final flow prediction at full resolution
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=5, padding=2),  # 5×5 for smooth flow
        )
        
        # Learnable max flow magnitude (in pixels at high-res)
        self.max_flow = nn.Parameter(torch.tensor(20.0))  # Max ~20 pixels at 128×128
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable flow estimation."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize flow head to zero for stable start
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
        Estimate bidirectional flow between two high-resolution frames.
        
        Args:
            frame_t: High-res frame at time t, shape (B, 2, C_in, H, W)
            frame_t1: High-res frame at time t+1, shape (B, 2, C_in, H, W)
        
        Returns:
            forward_flow: Flow from t to t+1, shape (B, 2, H, W)
            backward_flow: Flow from t+1 to t, shape (B, 2, H, W)
        """
        B, _, C, H, W = frame_t.shape
        
        # Flatten complex dimension: (B, 2*C, H, W)
        frame_t_flat = frame_t.reshape(B, 2 * C, H, W)
        frame_t1_flat = frame_t1.reshape(B, 2 * C, H, W)
        
        # Concatenate frames: (B, 4*C, H, W)
        concat_frames = torch.cat([frame_t_flat, frame_t1_flat], dim=1)
        
        # Encode at full resolution
        x0 = self.encoder(concat_frames)  # (B, hidden, H, W)
        
        # Downsample for context
        x1 = self.down1(x0)  # (B, hidden*2, H/2, W/2)
        x2 = self.down2(x1)  # (B, hidden*4, H/4, W/4)
        
        # Upsample with skip connections
        # Use interpolate to match exact sizes when needed
        x1_up = self.up1(x2)  # (B, hidden*2, ~H/2, ~W/2)
        if x1_up.shape[2:] != x1.shape[2:]:
            x1_up = F.interpolate(x1_up, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1_cat = torch.cat([x1_up, x1], dim=1)  # (B, hidden*4, H/2, W/2)
        
        x0_up = self.up2(x1_cat)  # (B, hidden, ~H, ~W)
        if x0_up.shape[2:] != x0.shape[2:]:
            x0_up = F.interpolate(x0_up, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x0_cat = torch.cat([x0_up, x0], dim=1)  # (B, hidden*2, H, W)
        
        # Predict flow at full resolution
        flow_output = self.flow_head(x0_cat)  # (B, 4, H, W)
        
        # Split into forward and backward flow
        forward_flow = flow_output[:, :2, :, :]  # (B, 2, H, W)
        backward_flow = flow_output[:, 2:, :, :]  # (B, 2, H, W)
        
        # Apply soft magnitude constraint
        max_flow = torch.abs(self.max_flow)
        forward_flow = max_flow * torch.tanh(forward_flow * self.flow_scale)
        backward_flow = max_flow * torch.tanh(backward_flow * self.flow_scale)
        
        return forward_flow, backward_flow
    
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
    
    Operates at bottleneck resolution. Learns per-pixel weights to blend
    linear interpolation (safe baseline) with motion-warped features.
    """
    
    def __init__(
        self,
        in_channels: int,
        use_residual_fusion: bool = True
    ):
        super(AdaptiveFusion, self).__init__()
        
        self.use_residual_fusion = use_residual_fusion
        
        # Input: linear (2*C) + motion (2*C) + alpha (1)
        fusion_in_channels = 4 * in_channels + 1
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
        # Initialize to prefer linear interpolation
        nn.init.zeros_(self.fusion_conv[-1].weight)
        nn.init.constant_(self.fusion_conv[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12
    
    def forward(
        self,
        linear_result: torch.Tensor,
        motion_result: torch.Tensor,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blend linear and motion-based interpolation.
        
        Args:
            linear_result: Linear interpolated features (B, 2, C, H, W)
            motion_result: Motion-warped features (B, 2, C, H, W)
            alpha: Interpolation position in [0, 1]
        
        Returns:
            fused: Blended result (B, 2, C, H, W)
            motion_weight: Per-pixel fusion weight (B, 1, H, W)
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
        
        # Compute fusion weight
        motion_weight_logit = self.fusion_conv(fusion_input)  # (B, 1, H, W)
        motion_weight = torch.sigmoid(motion_weight_logit)  # (B, 1, H, W)
        
        # Expand for complex dimension
        motion_weight_expanded = motion_weight.unsqueeze(2)  # (B, 1, 1, H, W)
        
        if self.use_residual_fusion:
            # Residual: linear + weight * (motion - linear)
            fused = linear_result + motion_weight_expanded * (motion_result - linear_result)
        else:
            fused = (1 - motion_weight_expanded) * linear_result + motion_weight_expanded * motion_result
        
        return fused, motion_weight


def warp_complex(
    feature_map: torch.Tensor,
    flow: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'border'
) -> torch.Tensor:
    """
    Warp a complex feature map using optical flow.
    
    Args:
        feature_map: Complex features (B, 2, C, H, W)
        flow: Optical flow in PIXEL units (B, 2, H, W)
        mode: Interpolation mode for grid_sample
        padding_mode: Padding mode for grid_sample
    
    Returns:
        Warped feature map (B, 2, C, H, W)
    """
    B, _, C, H, W = feature_map.shape
    
    # Create base grid in normalized coordinates [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feature_map.device, dtype=feature_map.dtype),
        torch.linspace(-1, 1, W, device=feature_map.device, dtype=feature_map.dtype),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Convert flow from pixel units to normalized coordinates
    flow_normalized = torch.zeros_like(base_grid)
    if W > 1:
        flow_normalized[..., 0] = flow[:, 0, :, :] * 2.0 / (W - 1)
    if H > 1:
        flow_normalized[..., 1] = flow[:, 1, :, :] * 2.0 / (H - 1)
    
    # Add flow to base grid
    sample_grid = base_grid + flow_normalized
    
    # Warp Real component
    real = feature_map[:, 0, :, :, :]  # (B, C, H, W)
    warped_real = F.grid_sample(real, sample_grid, mode=mode, 
                                 padding_mode=padding_mode, align_corners=True)
    
    # Warp Imaginary component
    imag = feature_map[:, 1, :, :, :]  # (B, C, H, W)
    warped_imag = F.grid_sample(imag, sample_grid, mode=mode,
                                 padding_mode=padding_mode, align_corners=True)
    
    return torch.stack([warped_real, warped_imag], dim=1)


class InputGuidedMotionInterpolation(nn.Module):
    """
    Input-Guided Motion-Aware Interpolation.
    
    Key Innovation: Estimates optical flow from ORIGINAL high-resolution input frames,
    not from abstracted bottleneck features. This captures true pixel-level motion.
    
    Pipeline:
        1. Extract sparse frames from high-res inputs
        2. Estimate flow at high resolution (e.g., 128×128)
        3. Downsample flow to bottleneck resolution (e.g., 8×8)
        4. CRITICAL: Scale flow magnitude by (H_small / H_large)
        5. Warp bottleneck features using scaled flow
        6. Blend with linear interpolation
    
    Input Shapes:
        feature_map: (B, T_sparse, 2, C, H_small, W_small) - Bottleneck features
        inputs_high_res: (B, T_sparse, 2, C_in, H_large, W_large) - Original frames
    
    Output Shape:
        (B, T_dense, 2, C, H_small, W_small)
    """
    
    def __init__(
        self,
        bottleneck_channels: int,  # C in bottleneck
        input_channels: int = 1,   # C_in of original frames
        upsampling_factor: int = 20,
        flow_hidden_channels: int = 32,
        flow_scale: float = 0.1,
        use_residual_fusion: bool = True,
    ):
        super(InputGuidedMotionInterpolation, self).__init__()
        
        self.bottleneck_channels = bottleneck_channels
        self.input_channels = input_channels
        self.upsampling_factor = upsampling_factor
        
        # Flow estimator operates on HIGH-RESOLUTION input frames
        self.flow_estimator = HighResFlowEstimator(
            in_channels=input_channels,
            hidden_channels=flow_hidden_channels,
            flow_scale=flow_scale,
        )
        
        # Fusion operates at BOTTLENECK resolution
        self.adaptive_fusion = AdaptiveFusion(
            in_channels=bottleneck_channels,
            use_residual_fusion=use_residual_fusion,
        )
        
        # Cache for loss computation
        self.cached_flows_high_res = []
        self.cached_flows_low_res = []
        self.cached_frames_high_res = []
        self.cached_motion_weights = []
        self.cached_scale_factor = None
    
    def downsample_and_scale_flow(
        self,
        flow_high_res: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Downsample flow and scale magnitude appropriately.
        
        CRITICAL: Flow magnitude must be scaled by (H_small / H_large).
        Example: 16px flow at 128×128 → 1px flow at 8×8
        
        Args:
            flow_high_res: Flow at high resolution (B, 2, H_large, W_large)
            target_size: Target size (H_small, W_small)
        
        Returns:
            Scaled flow at low resolution (B, 2, H_small, W_small)
        """
        H_large, W_large = flow_high_res.shape[2], flow_high_res.shape[3]
        H_small, W_small = target_size
        
        # Downsample flow field
        flow_low_res = F.interpolate(
            flow_high_res,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )
        
        # CRITICAL: Scale flow magnitude
        # Flow X component scales by W_small / W_large
        # Flow Y component scales by H_small / H_large
        scale_x = W_small / W_large
        scale_y = H_small / H_large
        
        flow_low_res[:, 0, :, :] *= scale_x  # Scale X flow
        flow_low_res[:, 1, :, :] *= scale_y  # Scale Y flow
        
        # Cache scale factor for logging
        self.cached_scale_factor = (scale_x, scale_y)
        
        return flow_low_res
    
    def linear_interpolate(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Linear interpolation baseline."""
        return (1 - alpha) * frame_t + alpha * frame_t1
    
    def forward(
        self,
        feature_map: torch.Tensor,
        inputs_high_res: torch.Tensor,
        return_flows: bool = False
    ) -> torch.Tensor:
        """
        Perform input-guided motion-aware temporal interpolation.
        
        Args:
            feature_map: Bottleneck features (B, T_sparse, 2, C, H_small, W_small)
            inputs_high_res: Original high-res frames (B, T_sparse, 2, C_in, H_large, W_large)
            return_flows: If True, also return cached flows
        
        Returns:
            Dense features (B, T_dense, 2, C, H_small, W_small)
        """
        B, T_sparse, _, C, H_small, W_small = feature_map.shape
        _, _, _, C_in, H_large, W_large = inputs_high_res.shape
        
        k = self.upsampling_factor
        
        # Clear caches
        self.cached_flows_high_res = []
        self.cached_flows_low_res = []
        self.cached_frames_high_res = []
        self.cached_motion_weights = []
        
        dense_outputs = []
        
        # Add first sparse frame
        dense_outputs.append(feature_map[:, 0:1, :, :, :, :])
        
        # Process each pair of consecutive sparse frames
        for t in range(T_sparse - 1):
            # Get bottleneck features
            feat_t = feature_map[:, t, :, :, :, :]      # (B, 2, C, H_small, W_small)
            feat_t1 = feature_map[:, t + 1, :, :, :, :] # (B, 2, C, H_small, W_small)
            
            # Get HIGH-RES input frames for flow estimation
            input_t = inputs_high_res[:, t, :, :, :, :]      # (B, 2, C_in, H_large, W_large)
            input_t1 = inputs_high_res[:, t + 1, :, :, :, :] # (B, 2, C_in, H_large, W_large)
            
            # Estimate flow at HIGH RESOLUTION
            forward_flow_high, backward_flow_high = self.flow_estimator(input_t, input_t1)
            # forward_flow_high: (B, 2, H_large, W_large)
            
            # Cache high-res flows and frames
            self.cached_flows_high_res.append((forward_flow_high, backward_flow_high))
            self.cached_frames_high_res.append((input_t, input_t1))
            
            # Downsample and scale flow for bottleneck resolution
            forward_flow_low = self.downsample_and_scale_flow(
                forward_flow_high, (H_small, W_small)
            )
            backward_flow_low = self.downsample_and_scale_flow(
                backward_flow_high, (H_small, W_small)
            )
            
            self.cached_flows_low_res.append((forward_flow_low, backward_flow_low))
            
            # Generate k-1 intermediate frames between sparse frames
            # For T_sparse=5, k=20: we want 1 + 4*20 = 81 total frames
            # Logic: sparse[0], then for each interval: k-1 intermediates + sparse[t+1]
            # This gives: 1 + 4*(19+1) = 1 + 80 = 81 ✓
            for i in range(1, k):
                alpha = i / k
                
                # 1. Linear interpolation (baseline)
                linear_result = self.linear_interpolate(feat_t, feat_t1, alpha)
                
                # 2. Motion-based interpolation
                # Scale flow by interpolation factor
                flow_t_to_mid = alpha * forward_flow_low
                flow_t1_to_mid = (1 - alpha) * backward_flow_low
                
                # Warp both frames toward intermediate position
                warped_t = warp_complex(feat_t, flow_t_to_mid)
                warped_t1 = warp_complex(feat_t1, -flow_t1_to_mid)  # Negative for backward
                
                # Blend warped results
                motion_result = (1 - alpha) * warped_t + alpha * warped_t1
                
                # 3. Adaptive fusion
                fused, motion_weight = self.adaptive_fusion(linear_result, motion_result, alpha)
                
                self.cached_motion_weights.append(motion_weight)
                
                dense_outputs.append(fused.unsqueeze(1))
            
            # Add the endpoint sparse frame (t+1) for this interval
            # This is NOT added for the last interval (handled separately below)
            if t < T_sparse - 2:
                dense_outputs.append(feature_map[:, t+1:t+2, :, :, :, :])
        
        # Add last sparse frame
        dense_outputs.append(feature_map[:, -1:, :, :, :, :])
        
        # Concatenate all frames
        dense_features = torch.cat(dense_outputs, dim=1)
        
        if return_flows:
            return dense_features, self.cached_flows_high_res
        
        return dense_features
    
    def get_cached_flows_high_res(self):
        """Get cached high-resolution flows for loss computation."""
        return self.cached_flows_high_res
    
    def get_cached_flows_low_res(self):
        """Get cached low-resolution (scaled) flows."""
        return self.cached_flows_low_res
    
    def get_cached_frames_high_res(self):
        """Get cached high-resolution frames for photometric loss."""
        return self.cached_frames_high_res
    
    def get_motion_weight_stats(self) -> dict:
        """Get statistics on motion weight usage."""
        if not self.cached_motion_weights:
            return {}
        
        weights = torch.cat([w.flatten() for w in self.cached_motion_weights])
        return {
            'motion_weight_mean': weights.mean().item(),
            'motion_weight_max': weights.max().item(),
            'motion_weight_min': weights.min().item(),
        }
    
    def get_scale_factor(self) -> Optional[Tuple[float, float]]:
        """Get the flow scale factor (for logging)."""
        return self.cached_scale_factor


class InputGuidedTemporalBottleneck(nn.Module):
    """
    Temporal Bottleneck with Input-Guided Motion Interpolation.
    
    Combines:
    1. ComplexConvLSTM for temporal feature modeling
    2. Input-guided motion interpolation for temporal upsampling
    
    The key difference from previous versions: flow is estimated from
    original high-resolution input frames, not bottleneck features.
    """
    
    def __init__(
        self,
        encoder_channels: int,    # Channels from encoder (e.g., 256)
        hidden_channels: int,     # LSTM hidden channels (e.g., 128)
        input_channels: int = 1,  # Original input channels
        num_layers: int = 1,
        kernel_size: int = 3,
        upsampling_factor: int = 20,
        flow_hidden_channels: int = 32,
        flow_scale: float = 0.1,
        use_residual_fusion: bool = True,
    ):
        super(InputGuidedTemporalBottleneck, self).__init__()
        
        self.upsampling_factor = upsampling_factor
        self.encoder_channels = encoder_channels
        self.hidden_channels = hidden_channels
        
        # Import here to avoid circular imports
        from src.architectures.complex_modules import ComplexConv2d
        from src.architectures.complex_lstm import ComplexConvLSTM
        
        # Project encoder output to hidden dimension
        if encoder_channels != hidden_channels:
            self.input_proj = ComplexConv2d(encoder_channels, hidden_channels, kernel_size=1, padding=0)
        else:
            self.input_proj = None
        
        # ComplexConvLSTM for temporal modeling
        self.lstm = ComplexConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size
        )
        
        # Input-guided motion interpolation
        self.motion_interpolation = InputGuidedMotionInterpolation(
            bottleneck_channels=hidden_channels,
            input_channels=input_channels,
            upsampling_factor=upsampling_factor,
            flow_hidden_channels=flow_hidden_channels,
            flow_scale=flow_scale,
            use_residual_fusion=use_residual_fusion,
        )
        
        # Project back to encoder channels
        if encoder_channels != hidden_channels:
            self.output_proj = ComplexConv2d(hidden_channels, encoder_channels, kernel_size=1, padding=0)
        else:
            self.output_proj = None
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        inputs_high_res: torch.Tensor,
        upsampling_factor: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform input-guided temporal super-resolution.
        
        Args:
            sparse_features: Encoded sparse frames (B, T_sparse, 2, C_enc, H, W)
            inputs_high_res: Original high-res frames (B, T_sparse, 2, C_in, H_large, W_large)
            upsampling_factor: Override default upsampling factor
        
        Returns:
            Dense features (B, T_dense, 2, C_enc, H, W)
        """
        B, T_sparse, _, C_enc, H, W = sparse_features.shape
        
        if upsampling_factor is not None:
            self.motion_interpolation.upsampling_factor = upsampling_factor
        
        # Project to hidden dimension
        if self.input_proj is not None:
            sparse_flat = sparse_features.view(B * T_sparse, 2, C_enc, H, W)
            projected = self.input_proj(sparse_flat)
            C_hidden = projected.shape[2]
            sparse_features_proj = projected.view(B, T_sparse, 2, C_hidden, H, W)
        else:
            sparse_features_proj = sparse_features
        
        # Apply LSTM for temporal modeling
        lstm_out, _ = self.lstm(sparse_features_proj)  # (B, T_sparse, 2, C_hidden, H, W), discard hidden states
        
        # Input-guided motion interpolation
        dense_features = self.motion_interpolation(
            lstm_out,
            inputs_high_res
        )  # (B, T_dense, 2, C_hidden, H, W)
        
        # Project back to encoder channels
        if self.output_proj is not None:
            T_dense = dense_features.shape[1]
            C_hidden = dense_features.shape[3]
            dense_flat = dense_features.view(B * T_dense, 2, C_hidden, H, W)
            output = self.output_proj(dense_flat)
            dense_features = output.view(B, T_dense, 2, C_enc, H, W)
        
        return dense_features
    
    def get_flow_estimator(self):
        """Get the flow estimator module."""
        return self.motion_interpolation.flow_estimator
    
    def get_motion_interpolation(self):
        """Get the motion interpolation module."""
        return self.motion_interpolation
    
    def get_flow_stats(self) -> dict:
        """Get flow statistics from the last forward pass."""
        if not self.motion_interpolation.cached_flows_high_res:
            return {}
        
        # Get stats from first flow pair
        fwd_flow, _ = self.motion_interpolation.cached_flows_high_res[0]
        stats = self.motion_interpolation.flow_estimator.get_flow_stats(fwd_flow)
        
        # Add scale factor info
        scale = self.motion_interpolation.get_scale_factor()
        if scale:
            stats['flow_scale_x'] = scale[0]
            stats['flow_scale_y'] = scale[1]
        
        return stats


# ============================================================
# Flow Loss Functions (operate on high-res flows)
# ============================================================

def flow_consistency_loss(
    forward_flows: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Forward-backward flow consistency loss.
    
    If we go forward then backward, we should return to the start.
    Operates on HIGH-RESOLUTION flows.
    """
    total_loss = 0.0
    count = 0
    
    for fwd_flow, bwd_flow in forward_flows:
        # Warp backward flow using forward flow
        B, _, H, W = fwd_flow.shape
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=fwd_flow.device, dtype=fwd_flow.dtype),
            torch.linspace(-1, 1, W, device=fwd_flow.device, dtype=fwd_flow.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Normalize forward flow from pixel units to [-1, 1]
        # fwd_flow is (B, 2, H, W), need to convert to (B, H, W, 2) for grid_sample
        fwd_norm_x = fwd_flow[:, 0] * 2.0 / (W - 1) if W > 1 else fwd_flow[:, 0]
        fwd_norm_y = fwd_flow[:, 1] * 2.0 / (H - 1) if H > 1 else fwd_flow[:, 1]
        # Stack along last dim to get (B, H, W, 2)
        fwd_norm = torch.stack([fwd_norm_x, fwd_norm_y], dim=-1)
        
        sample_grid = base_grid + fwd_norm
        
        # Warp backward flow
        warped_bwd = F.grid_sample(
            bwd_flow, sample_grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Forward + warped backward should be zero
        consistency = fwd_flow + warped_bwd
        total_loss += torch.mean(consistency ** 2)
        count += 1
    
    return total_loss / max(count, 1)


def flow_smoothness_loss(
    forward_flows: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Flow smoothness regularization.
    Encourages spatially smooth flow fields.
    """
    total_loss = 0.0
    count = 0
    
    for fwd_flow, bwd_flow in forward_flows:
        for flow in [fwd_flow, bwd_flow]:
            # Spatial gradients
            dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
            dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
            
            total_loss += torch.mean(dx ** 2) + torch.mean(dy ** 2)
            count += 1
    
    return total_loss / max(count, 1)


def photometric_loss(
    flows: List[Tuple[torch.Tensor, torch.Tensor]],
    frames: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Photometric consistency loss.
    Warped frame should match target frame.
    """
    total_loss = 0.0
    count = 0
    
    for (fwd_flow, bwd_flow), (frame_t, frame_t1) in zip(flows, frames):
        # Warp frame_t using forward flow → should match frame_t1
        warped_t = warp_complex(frame_t, fwd_flow)
        loss_fwd = F.l1_loss(warped_t, frame_t1)
        
        # Warp frame_t1 using backward flow → should match frame_t
        warped_t1 = warp_complex(frame_t1, bwd_flow)
        loss_bwd = F.l1_loss(warped_t1, frame_t)
        
        total_loss += loss_fwd + loss_bwd
        count += 2
    
    return total_loss / max(count, 1)
