import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class InterferometricPhaseLoss(nn.Module):
    """
    Interferometric Phase Loss for Complex IQ Data.
    
    Maximizes the cosine similarity between predicted and target phase.
    This loss is robust to:
        - Phase wrapping (handles -π to π discontinuity)
        - Amplitude differences (only cares about phase alignment)
        - Background noise (applies tissue mask)
    
    The key insight: When phases align, the interference pattern
    Pred * Conj(Target) lies on the positive real axis (angle = 0).
    
    Loss = 1 - cos(angle(Pred * Conj(Target)))
    
    When phases match perfectly: angle = 0 → cos(0) = 1 → loss = 0
    When phases are opposite: angle = π → cos(π) = -1 → loss = 2
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        threshold_ratio: float = 0.05,
        eps: float = 1e-8
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            threshold_ratio: Fraction of max magnitude to use as tissue mask threshold.
                            Pixels below threshold_ratio * max(|target|) are ignored.
            eps: Small epsilon for numerical stability
        """
        super(InterferometricPhaseLoss, self).__init__()
        self.reduction = reduction
        self.threshold_ratio = threshold_ratio
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interferometric phase loss.
        
        Args:
            pred: Predicted complex tensor (B, T, 2, C, H, W)
                  where dim 2 contains [real, imag]
            target: Ground truth complex tensor (same shape as pred)
        
        Returns:
            Interferometric phase loss (scalar if reduction='mean'/'sum')
        """
        # Extract real and imaginary parts
        # Shape: (B, T, 2, C, H, W) -> indexing dim 2
        pred_real = pred[:, :, 0, :, :, :]   # (B, T, C, H, W)
        pred_imag = pred[:, :, 1, :, :, :]
        target_real = target[:, :, 0, :, :, :]
        target_imag = target[:, :, 1, :, :, :]
        
        # Compute interference: Pred * Conj(Target)
        # (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
        interference_real = pred_real * target_real + pred_imag * target_imag
        interference_imag = pred_imag * target_real - pred_real * target_imag
        
        # Compute angle of interference
        # When phases align, interference lies on positive real axis (angle ≈ 0)
        angle_error = torch.atan2(interference_imag + self.eps, interference_real + self.eps)
        
        # Create tissue mask: only compute loss where signal is significant
        # This avoids noisy background pixels dominating the loss
        target_magnitude = torch.sqrt(target_real ** 2 + target_imag ** 2 + self.eps)
        max_magnitude = target_magnitude.max()
        threshold = self.threshold_ratio * max_magnitude
        valid_mask = target_magnitude > threshold
        
        # Compute loss: 1 - cos(angle_error)
        # This is 0 when phases align (angle=0) and 2 when opposite (angle=π)
        phase_loss = 1.0 - torch.cos(angle_error)
        
        # Apply mask - only consider tissue regions
        if valid_mask.sum() > 0:
            masked_loss = phase_loss[valid_mask]
        else:
            # Fallback if mask is empty (shouldn't happen in practice)
            masked_loss = phase_loss.view(-1)
        
        # Reduction
        if self.reduction == 'mean':
            return masked_loss.mean()
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return phase_loss  # Return full tensor for 'none'


class KasaiAutocorrelationLoss(nn.Module):
    """
    Kasai Autocorrelation Loss for Shear Wave Elastography.
    
    This is the most robust method for SWE velocity estimation. Instead of asking 
    "Does this pixel match?", we ask "Does the velocity at this pixel match?"
    
    The Kasai autocorrelation method computes Lag-1 autocorrelation:
        R = P(t) * conj(P(t-1))
    
    The argument (angle) of R is proportional to velocity (phase shift per frame).
    
    Loss: Minimize the complex distance between R_pred and R_true
        L_velocity = |R_pred - R_true|
    
    This forces the network to preserve the change in phase between frames (Δφ),
    which is the mathematical definition of particle velocity in ultrasound.
    If the network smooths the phase, this loss explodes, forcing sharp temporal waves.
    
    Benefits:
        - Directly measures tissue velocity (what we care about for elastography)
        - Differentiable through the autocorrelation formula
        - Robust to magnitude variations
        - Naturally handles phase wrapping through complex arithmetic
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        threshold_ratio: float = 0.05,
        eps: float = 1e-8
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            threshold_ratio: Fraction of max magnitude for tissue masking
            eps: Small epsilon for numerical stability
        """
        super(KasaiAutocorrelationLoss, self).__init__()
        self.reduction = reduction
        self.threshold_ratio = threshold_ratio
        self.eps = eps
    
    def _compute_lag1_autocorrelation(
        self,
        data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Lag-1 autocorrelation: R = P(t) * conj(P(t-1))
        
        Args:
            data: Complex tensor (B, T, 2, C, H, W) where dim 2 is [real, imag]
        
        Returns:
            R_real: Real part of autocorrelation (B, T-1, C, H, W)
            R_imag: Imaginary part of autocorrelation (B, T-1, C, H, W)
            magnitude: Magnitude for masking (B, T-1, C, H, W)
        """
        # Extract real and imaginary parts
        real = data[:, :, 0, :, :, :]  # (B, T, C, H, W)
        imag = data[:, :, 1, :, :, :]
        
        # Get consecutive frame pairs
        # P(t-1) = z_t = c + di
        real_t = real[:, :-1]      
        imag_t = imag[:, :-1]
        # P(t) = z_{t+1} = a + bi
        real_t1 = real[:, 1:]      
        imag_t1 = imag[:, 1:]
        
        # Compute R = P(t) * conj(P(t-1)) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i
        R_real = real_t1 * real_t + imag_t1 * imag_t
        R_imag = imag_t1 * real_t - real_t1 * imag_t
        
        # Compute magnitude for masking (use geometric mean of both frames)
        mag_t = torch.sqrt(real_t ** 2 + imag_t ** 2 + self.eps)
        mag_t1 = torch.sqrt(real_t1 ** 2 + imag_t1 ** 2 + self.eps)
        magnitude = torch.sqrt(mag_t * mag_t1)
        
        return R_real, R_imag, magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Kasai Autocorrelation loss.
        
        L = |R_pred - R_true| = sqrt((R_pred_real - R_true_real)² + (R_pred_imag - R_true_imag)²)
        
        Args:
            pred: Predicted complex tensor (B, T, 2, C, H, W)
            target: Ground truth complex tensor (same shape)
        
        Returns:
            Kasai autocorrelation loss (scalar if reduction='mean'/'sum')
        """
        # Compute lag-1 autocorrelations
        R_pred_real, R_pred_imag, pred_mag = self._compute_lag1_autocorrelation(pred)
        R_true_real, R_true_imag, target_mag = self._compute_lag1_autocorrelation(target)
        
        # Compute complex distance: |R_pred - R_true|
        diff_real = R_pred_real - R_true_real
        diff_imag = R_pred_imag - R_true_imag
        autocorr_distance = torch.sqrt(diff_real ** 2 + diff_imag ** 2 + self.eps)
        
        # Create tissue mask based on target magnitude
        max_magnitude = target_mag.max()
        threshold = self.threshold_ratio * max_magnitude
        valid_mask = target_mag > threshold
        
        # Apply mask
        if valid_mask.sum() > 0:
            masked_loss = autocorr_distance[valid_mask]
        else:
            masked_loss = autocorr_distance.view(-1)
        
        # Reduction
        if self.reduction == 'mean':
            return masked_loss.mean()
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return autocorr_distance


class VelocityLoss(nn.Module):
    """
    Velocity Field Loss for Complex IQ Data.
    
    Computes the phase velocity (phase difference between consecutive frames)
    and compares predicted vs target velocity fields. This is the ultimate
    metric for shear wave elastography as it directly measures tissue motion.
    
    Velocity is computed as: angle(z_{t+1} * conj(z_t))
    This approach naturally handles phase wrapping and gives the phase shift
    (proportional to tissue velocity) between consecutive frames.
    
    The loss can use either:
        - MSE: Simple mean squared error on velocity fields
        - Cosine: Interferometric cosine similarity (more robust to outliers)
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        threshold_ratio: float = 0.05,
        loss_type: str = 'mse',  # 'mse' or 'cosine'
        eps: float = 1e-8
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            threshold_ratio: Fraction of max magnitude for tissue masking
            loss_type: 'mse' for MSE loss, 'cosine' for cosine similarity loss
            eps: Small epsilon for numerical stability
        """
        super(VelocityLoss, self).__init__()
        self.reduction = reduction
        self.threshold_ratio = threshold_ratio
        self.loss_type = loss_type
        self.eps = eps
    
    def _compute_velocity(
        self,
        data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phase velocity from complex IQ sequence.
        
        Velocity = angle(z_{t+1} * conj(z_t))
        
        Args:
            data: Complex tensor (B, T, 2, C, H, W) where dim 2 is [real, imag]
        
        Returns:
            velocity: Phase velocity (B, T-1, C, H, W)
            magnitude: Magnitude for masking (B, T-1, C, H, W)
        """
        # Extract real and imaginary parts
        real = data[:, :, 0, :, :, :]  # (B, T, C, H, W)
        imag = data[:, :, 1, :, :, :]
        
        # Get consecutive frame pairs
        real_t = real[:, :-1]      # z_t
        imag_t = imag[:, :-1]
        real_t1 = real[:, 1:]      # z_{t+1}
        imag_t1 = imag[:, 1:]
        
        # Compute z_{t+1} * conj(z_t) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i
        # where z_{t+1} = a+bi and z_t = c+di
        interference_real = real_t1 * real_t + imag_t1 * imag_t
        interference_imag = imag_t1 * real_t - real_t1 * imag_t
        
        # Phase velocity is the angle of the interference
        velocity = torch.atan2(interference_imag + self.eps, interference_real + self.eps)
        
        # Compute magnitude at time t for masking (use geometric mean of both frames)
        mag_t = torch.sqrt(real_t ** 2 + imag_t ** 2 + self.eps)
        mag_t1 = torch.sqrt(real_t1 ** 2 + imag_t1 ** 2 + self.eps)
        magnitude = torch.sqrt(mag_t * mag_t1)  # Geometric mean
        
        return velocity, magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity field loss.
        
        Args:
            pred: Predicted complex tensor (B, T, 2, C, H, W)
            target: Ground truth complex tensor (same shape)
        
        Returns:
            Velocity loss (scalar if reduction='mean'/'sum')
        """
        # Compute velocity fields
        pred_velocity, pred_mag = self._compute_velocity(pred)
        target_velocity, target_mag = self._compute_velocity(target)
        
        # Create tissue mask based on target magnitude
        max_magnitude = target_mag.max()
        threshold = self.threshold_ratio * max_magnitude
        valid_mask = target_mag > threshold
        
        if self.loss_type == 'mse':
            # Simple MSE on velocity fields
            velocity_diff = pred_velocity - target_velocity
            velocity_loss = velocity_diff ** 2
            
            # Apply mask
            if valid_mask.sum() > 0:
                masked_loss = velocity_loss[valid_mask]
            else:
                masked_loss = velocity_loss.view(-1)
                
        elif self.loss_type == 'cosine':
            # Cosine similarity loss on velocity (robust to outliers)
            # When velocities match: angle difference = 0 → cos = 1 → loss = 0
            velocity_diff = pred_velocity - target_velocity
            velocity_loss = 1.0 - torch.cos(velocity_diff)
            
            # Apply mask
            if valid_mask.sum() > 0:
                masked_loss = velocity_loss[valid_mask]
            else:
                masked_loss = velocity_loss.view(-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'mse' or 'cosine'.")
        
        # Reduction
        if self.reduction == 'mean':
            return masked_loss.mean()
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return velocity_loss


class MagnitudeWeightedVelocityLoss(nn.Module):
    """
    Magnitude-Weighted Velocity Loss for Shear Wave Elastography.
    
    This loss addresses the "salt and pepper" noise problem in Kasai velocity maps.
    In regions where the signal is dark (no echo), the phase is random. Calculating
    the phase shift of random noise results in random high-velocity values.
    
    Solution: Weight the velocity error by the signal magnitude.
    - Bright pixels (high echo) → high weight → model must match velocity
    - Dark pixels (low echo/noise) → low weight → model ignores the noise
    
    This prevents the model from trying to learn random noise in the background,
    which is impossible and wastes model capacity.
    
    Loss = mean(|pred_vel - target_vel| * magnitude_mask)
    
    Where magnitude_mask is the normalized IQ magnitude (0-1).
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        loss_type: str = 'l1',  # 'l1', 'mse', or 'cosine'
        eps: float = 1e-8,
        min_weight: float = 0.01  # Minimum weight to avoid complete zero-masking
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            loss_type: Type of velocity loss ('l1', 'mse', or 'cosine')
            eps: Small epsilon for numerical stability
            min_weight: Minimum weight for any pixel (prevents division issues)
        """
        super(MagnitudeWeightedVelocityLoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.eps = eps
        self.min_weight = min_weight
    
    def _compute_velocity(
        self,
        data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phase velocity from complex IQ sequence.
        
        Velocity = angle(z_{t+1} * conj(z_t))
        
        Args:
            data: Complex tensor (B, T, 2, C, H, W) where dim 2 is [real, imag]
        
        Returns:
            velocity: Phase velocity (B, T-1, C, H, W)
            magnitude: Magnitude for masking (B, T-1, C, H, W)
        """
        # Extract real and imaginary parts
        real = data[:, :, 0, :, :, :]  # (B, T, C, H, W)
        imag = data[:, :, 1, :, :, :]
        
        # Get consecutive frame pairs
        real_t = real[:, :-1]
        imag_t = imag[:, :-1]
        real_t1 = real[:, 1:]
        imag_t1 = imag[:, 1:]
        
        # Compute z_{t+1} * conj(z_t)
        interference_real = real_t1 * real_t + imag_t1 * imag_t
        interference_imag = imag_t1 * real_t - real_t1 * imag_t
        
        # Phase velocity is the angle of the interference
        velocity = torch.atan2(interference_imag + self.eps, interference_real + self.eps)
        
        # Compute magnitude (geometric mean of consecutive frames)
        mag_t = torch.sqrt(real_t ** 2 + imag_t ** 2 + self.eps)
        mag_t1 = torch.sqrt(real_t1 ** 2 + imag_t1 ** 2 + self.eps)
        magnitude = torch.sqrt(mag_t * mag_t1)
        
        return velocity, magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        magnitude_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute magnitude-weighted velocity loss.
        
        Args:
            pred: Predicted complex tensor (B, T, 2, C, H, W)
            target: Ground truth complex tensor (same shape)
            magnitude_mask: Optional pre-computed magnitude mask (B, 1, T, H, W) or (B, T-1, C, H, W).
                           If None, computed from target IQ data.
        
        Returns:
            Weighted velocity loss (scalar if reduction='mean'/'sum')
        """
        # Compute velocity fields
        pred_velocity, pred_mag = self._compute_velocity(pred)
        target_velocity, target_mag = self._compute_velocity(target)
        
        # Get or compute magnitude mask
        if magnitude_mask is not None:
            # Reshape magnitude_mask to match velocity shape if needed
            # Expected: (B, 1, T, H, W) -> need (B, T-1, C, H, W)
            if magnitude_mask.ndim == 5 and magnitude_mask.shape[1] == 1:
                # (B, 1, T, H, W) -> (B, T-1, 1, H, W)
                mask = magnitude_mask[:, :, :-1, :, :]  # Remove last frame (velocity has T-1 frames)
                mask = mask.permute(0, 2, 1, 3, 4)  # (B, T-1, 1, H, W)
            elif magnitude_mask.shape == target_velocity.shape:
                mask = magnitude_mask
            else:
                # Use target magnitude as fallback
                mask = target_mag
        else:
            # Use target magnitude (normalized 0-1)
            mask = target_mag
        
        # Normalize mask to 0-1 range
        mask_max = mask.max()
        if mask_max > 0:
            mask = mask / mask_max
        
        # Apply minimum weight to prevent complete masking
        mask = torch.clamp(mask, min=self.min_weight)
        
        # Compute velocity error based on loss type
        if self.loss_type == 'l1':
            # L1 loss (more robust to outliers)
            velocity_error = torch.abs(pred_velocity - target_velocity)
        elif self.loss_type == 'mse':
            # MSE loss
            velocity_error = (pred_velocity - target_velocity) ** 2
        elif self.loss_type == 'cosine':
            # Cosine similarity loss (robust to phase wrapping)
            velocity_error = 1.0 - torch.cos(pred_velocity - target_velocity)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'l1', 'mse', or 'cosine'.")
        
        # Apply magnitude weighting
        weighted_error = velocity_error * mask
        
        # Reduction
        if self.reduction == 'mean':
            # Weighted mean: sum(error * weight) / sum(weight)
            # This gives proper averaging accounting for the weights
            return weighted_error.sum() / (mask.sum() + self.eps)
        elif self.reduction == 'sum':
            return weighted_error.sum()
        else:
            return weighted_error


class MagnitudeWeightedKasaiLoss(nn.Module):
    """
    Magnitude-Weighted Kasai Autocorrelation Loss.
    
    Combines the benefits of Kasai autocorrelation (robust velocity estimation)
    with magnitude weighting (ignores noise in dark background regions).
    
    In shear wave elastography, the background often contains "salt and pepper"
    noise due to random phase in regions with no echo. This loss weights the
    Kasai autocorrelation error by the signal magnitude, effectively telling
    the model: "Only care about velocity accuracy where there's actual signal."
    
    Loss = mean(|R_pred - R_true| * magnitude_mask)
    
    Where:
        - R = P(t) * conj(P(t-1)) is the Kasai autocorrelation
        - magnitude_mask is the normalized IQ magnitude (0-1)
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        eps: float = 1e-8,
        min_weight: float = 0.01
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            eps: Small epsilon for numerical stability
            min_weight: Minimum weight for any pixel
        """
        super(MagnitudeWeightedKasaiLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.min_weight = min_weight
    
    def _compute_lag1_autocorrelation(
        self,
        data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Lag-1 autocorrelation: R = P(t) * conj(P(t-1))
        
        Args:
            data: Complex tensor (B, T, 2, C, H, W) where dim 2 is [real, imag]
        
        Returns:
            R_real: Real part of autocorrelation (B, T-1, C, H, W)
            R_imag: Imaginary part of autocorrelation (B, T-1, C, H, W)
            magnitude: Magnitude for weighting (B, T-1, C, H, W)
        """
        # Extract real and imaginary parts
        real = data[:, :, 0, :, :, :]  # (B, T, C, H, W)
        imag = data[:, :, 1, :, :, :]
        
        # Get consecutive frame pairs
        real_t = real[:, :-1]
        imag_t = imag[:, :-1]
        real_t1 = real[:, 1:]
        imag_t1 = imag[:, 1:]
        
        # Compute R = P(t) * conj(P(t-1)) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i
        R_real = real_t1 * real_t + imag_t1 * imag_t
        R_imag = imag_t1 * real_t - real_t1 * imag_t
        
        # Compute magnitude (geometric mean of consecutive frames)
        mag_t = torch.sqrt(real_t ** 2 + imag_t ** 2 + self.eps)
        mag_t1 = torch.sqrt(real_t1 ** 2 + imag_t1 ** 2 + self.eps)
        magnitude = torch.sqrt(mag_t * mag_t1)
        
        return R_real, R_imag, magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        magnitude_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute magnitude-weighted Kasai autocorrelation loss.
        
        L = |R_pred - R_true| * magnitude_mask
        
        Args:
            pred: Predicted complex tensor (B, T, 2, C, H, W)
            target: Ground truth complex tensor (same shape)
            magnitude_mask: Optional pre-computed magnitude mask.
                           If None, computed from target IQ data.
        
        Returns:
            Weighted Kasai loss (scalar if reduction='mean'/'sum')
        """
        # Compute lag-1 autocorrelations
        R_pred_real, R_pred_imag, pred_mag = self._compute_lag1_autocorrelation(pred)
        R_true_real, R_true_imag, target_mag = self._compute_lag1_autocorrelation(target)
        
        # Compute complex distance: |R_pred - R_true|
        diff_real = R_pred_real - R_true_real
        diff_imag = R_pred_imag - R_true_imag
        autocorr_distance = torch.sqrt(diff_real ** 2 + diff_imag ** 2 + self.eps)
        
        # Get or compute magnitude mask
        if magnitude_mask is not None:
            # Reshape if needed
            if magnitude_mask.ndim == 5 and magnitude_mask.shape[1] == 1:
                mask = magnitude_mask[:, :, :-1, :, :]
                mask = mask.permute(0, 2, 1, 3, 4)
            elif magnitude_mask.shape == autocorr_distance.shape:
                mask = magnitude_mask
            else:
                mask = target_mag
        else:
            mask = target_mag
        
        # Normalize mask to 0-1
        mask_max = mask.max()
        if mask_max > 0:
            mask = mask / mask_max
        
        # Apply minimum weight
        mask = torch.clamp(mask, min=self.min_weight)
        
        # Apply magnitude weighting
        weighted_error = autocorr_distance * mask
        
        # Reduction
        if self.reduction == 'mean':
            return weighted_error.sum() / (mask.sum() + self.eps)
        elif self.reduction == 'sum':
            return weighted_error.sum()
        else:
            return weighted_error


class ComplexMSELoss(nn.Module):
    """
    Complex Mean Squared Error Loss.
    
    Computes MSE in the complex domain:
        L = mean(|z_pred - z_true|²)
          = mean((real_pred - real_true)² + (imag_pred - imag_true)²)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(ComplexMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted complex tensor (..., 2, ...)
                  where dim with size 2 contains [real, imag]
            target: Ground truth complex tensor (same shape as pred)
        
        Returns:
            Complex MSE loss (scalar)
        """
        # Compute squared difference for real and imaginary parts
        diff_real = pred.select(-4, 0) - target.select(-4, 0)  # Assuming (B, T, 2, C, H, W)
        diff_imag = pred.select(-4, 1) - target.select(-4, 1)
        
        # Complex magnitude squared: |z|² = real² + imag²
        squared_error = diff_real ** 2 + diff_imag ** 2
        
        if self.reduction == 'mean':
            return squared_error.mean()
        elif self.reduction == 'sum':
            return squared_error.sum()
        else:
            return squared_error


class ComplexL1Loss(nn.Module):
    """
    Complex L1 Loss (Mean Absolute Error in complex domain).
    
    Computes L1 based on complex magnitude:
        L = mean(|z_pred - z_true|)
          = mean(sqrt((real_pred - real_true)² + (imag_pred - imag_true)²))
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super(ComplexL1Loss, self).__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted complex tensor (B, T, 2, C, H, W)
            target: Ground truth complex tensor
        
        Returns:
            Complex L1 loss (scalar)
        """
        diff_real = pred.select(-4, 0) - target.select(-4, 0)
        diff_imag = pred.select(-4, 1) - target.select(-4, 1)
        
        # Complex magnitude: |z| = sqrt(real² + imag²)
        magnitude = torch.sqrt(diff_real ** 2 + diff_imag ** 2 + self.eps)
        
        if self.reduction == 'mean':
            return magnitude.mean()
        elif self.reduction == 'sum':
            return magnitude.sum()
        else:
            return magnitude


class SpatialGradients(nn.Module):
    """
    Compute spatial gradients using finite differences.
    
    Uses central differences for interior points:
        ∂u/∂x ≈ (u[x+1] - u[x-1]) / (2*dx)
        ∂²u/∂x² ≈ (u[x+1] - 2*u[x] + u[x-1]) / dx²
    """
    
    def __init__(self, dx: float = 1.0, dy: float = 1.0):
        """
        Args:
            dx: Spatial step in x direction (pixels or physical units)
            dy: Spatial step in y direction
        """
        super(SpatialGradients, self).__init__()
        self.dx = dx
        self.dy = dy
        
        # Sobel-like kernels for first derivatives
        # Using central differences
        kernel_dx = torch.tensor([[[[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]]], dtype=torch.float32) / (8 * dx)
        
        kernel_dy = torch.tensor([[[[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]]]], dtype=torch.float32) / (8 * dy)
        
        # Laplacian kernel for second derivatives
        kernel_laplacian = torch.tensor([[[[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]]]], dtype=torch.float32) / (dx * dy)
        
        self.register_buffer('kernel_dx', kernel_dx)
        self.register_buffer('kernel_dy', kernel_dy)
        self.register_buffer('kernel_laplacian', kernel_laplacian)
    
    def gradient_x(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ∂u/∂x"""
        # u shape: (B, C, H, W)
        return F.conv2d(u, self.kernel_dx.expand(u.shape[1], 1, 3, 3), 
                       padding=1, groups=u.shape[1])
    
    def gradient_y(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ∂u/∂y"""
        return F.conv2d(u, self.kernel_dy.expand(u.shape[1], 1, 3, 3),
                       padding=1, groups=u.shape[1])
    
    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ∇²u = ∂²u/∂x² + ∂²u/∂y²"""
        return F.conv2d(u, self.kernel_laplacian.expand(u.shape[1], 1, 3, 3),
                       padding=1, groups=u.shape[1])


class TemporalGradients(nn.Module):
    """
    Compute temporal gradients using finite differences.
    
    First derivative (central difference):
        ∂u/∂t ≈ (u[t+1] - u[t-1]) / (2*dt)
    
    Second derivative:
        ∂²u/∂t² ≈ (u[t+1] - 2*u[t] + u[t-1]) / dt²
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Args:
            dt: Temporal step (frame interval)
        """
        super(TemporalGradients, self).__init__()
        self.dt = dt
    
    def first_derivative(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂u/∂t using central differences.
        
        Args:
            u: Tensor of shape (B, T, ...) 
        
        Returns:
            du/dt of shape (B, T-2, ...) for interior points
        """
        # Central difference: (u[t+1] - u[t-1]) / (2*dt)
        du_dt = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)
        return du_dt
    
    def second_derivative(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂²u/∂t² using central differences.
        
        Args:
            u: Tensor of shape (B, T, ...)
        
        Returns:
            d²u/dt² of shape (B, T-2, ...) for interior points
        """
        # Second derivative: (u[t+1] - 2*u[t] + u[t-1]) / dt²
        d2u_dt2 = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (self.dt ** 2)
        return d2u_dt2


class KelvinVoigtWaveEquation(nn.Module):
    """
    Kelvin-Voigt Viscoelastic Wave Equation Residual.
    
    The governing equation:
        ρ * ∂²u/∂t² = μ * ∇²u + η * ∂/∂t(∇²u)
    
    Residual:
        R = ρ * ∂²u/∂t² - μ * ∇²u - η * ∂/∂t(∇²u)
    
    For shear wave elastography, we work with the displacement field
    derived from the IQ phase.
    """
    
    def __init__(
        self,
        rho: float = 1000.0,  # Tissue density ~1000 kg/m³ (like water)
        mu: float = 1.0,      # Shear modulus (learnable or fixed)
        eta: float = 0.1,     # Viscosity (learnable or fixed)
        dx: float = 1.0,      # Spatial step x
        dy: float = 1.0,      # Spatial step y
        dt: float = 1.0,      # Temporal step
        learnable_params: bool = True
    ):
        super(KelvinVoigtWaveEquation, self).__init__()
        
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        if learnable_params:
            # Learnable physics parameters - ensure float dtype
            self.rho = nn.Parameter(torch.tensor(float(rho), dtype=torch.float32))
            self.mu = nn.Parameter(torch.tensor(float(mu), dtype=torch.float32))
            self.eta = nn.Parameter(torch.tensor(float(eta), dtype=torch.float32))
        else:
            self.register_buffer('rho', torch.tensor(float(rho), dtype=torch.float32))
            self.register_buffer('mu', torch.tensor(float(mu), dtype=torch.float32))
            self.register_buffer('eta', torch.tensor(float(eta), dtype=torch.float32))
        
        self.spatial_grad = SpatialGradients(dx, dy)
        self.temporal_grad = TemporalGradients(dt)
    
    def extract_displacement(self, iq_data: torch.Tensor) -> torch.Tensor:
        """
        Extract displacement field from complex IQ data.
        
        The phase of IQ data is proportional to tissue displacement:
            u ∝ arg(IQ) = atan2(imag, real)
        
        Args:
            iq_data: Complex IQ data (B, T, 2, C, H, W)
        
        Returns:
            Displacement field (B, T, C, H, W)
        """
        real = iq_data[:, :, 0, :, :, :]  # (B, T, C, H, W)
        imag = iq_data[:, :, 1, :, :, :]
        
        # Phase extraction with numerical stability
        # Add small epsilon to avoid issues when both real and imag are near zero
        eps = 1e-8
        displacement = torch.atan2(imag + eps, real + eps)
        
        # Clamp to avoid extreme values
        displacement = displacement.clamp(-torch.pi, torch.pi)
        
        return displacement
    
    def compute_residual(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual.
        
        R = ρ * ∂²u/∂t² - μ * ∇²u - η * ∂/∂t(∇²u)
        
        Args:
            displacement: Displacement field (B, T, C, H, W)
        
        Returns:
            PDE residual (B, T-2, C, H, W) - reduced in time due to derivatives
        """
        batch_size, t_len, channels, height, width = displacement.shape
        
        # Compute spatial Laplacian for each time step
        laplacians = []
        for t in range(t_len):
            u_t = displacement[:, t, :, :, :]  # (B, C, H, W)
            lap_t = self.spatial_grad.laplacian(u_t)
            laplacians.append(lap_t)
        
        laplacian = torch.stack(laplacians, dim=1)  # (B, T, C, H, W)
        
        # Temporal derivatives
        # Second derivative of displacement: ∂²u/∂t²
        d2u_dt2 = self.temporal_grad.second_derivative(displacement)  # (B, T-2, C, H, W)
        
        # Time derivative of Laplacian: ∂/∂t(∇²u)
        d_laplacian_dt = self.temporal_grad.first_derivative(laplacian)  # (B, T-2, C, H, W)
        
        # Laplacian at interior time points
        laplacian_interior = laplacian[:, 1:-1, :, :, :]  # (B, T-2, C, H, W)
        
        # PDE Residual
        residual = (
            self.rho * d2u_dt2 
            - self.mu * laplacian_interior 
            - self.eta * d_laplacian_dt
        )
        
        return residual
    
    def forward(self, iq_data: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual from IQ data.
        
        Args:
            iq_data: Complex IQ data (B, T, 2, C, H, W)
        
        Returns:
            PDE residual (B, T-2, C, H, W)
        """
        displacement = self.extract_displacement(iq_data)
        residual = self.compute_residual(displacement)
        return residual


class PhysicsInformedLoss(nn.Module):
    """
    Combined Physics-Informed Loss for PI-CV-RecUNet.
    
    Total Loss = λ_data * L_data + λ_phase * L_phase + λ_velocity * L_velocity
                 + λ_physics * L_physics + λ_smooth * L_smooth + λ_magnitude * L_magnitude
    
    Where:
        - L_data: Complex MSE between prediction and ground truth
        - L_phase: Interferometric phase loss (robust to phase wrapping)
        - L_velocity: Velocity field loss (the ultimate goal for elastography)
        - L_physics: PDE residual (Kelvin-Voigt wave equation)
        - L_smooth: Temporal smoothness regularization
        - L_magnitude: Magnitude preservation loss
    
    Supports:
        - Normalized physics loss (scales physics to match data loss magnitude)
        - Dynamic lambda adjustment for warmup
        - Interferometric phase loss for phase-accurate reconstruction
        - Velocity loss for direct velocity field supervision
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.01,
        lambda_smooth: float = 0.001,
        lambda_magnitude: float = 0.1,
        lambda_phase: float = 0.5,
        lambda_velocity: float = 0.5,  # NEW: weight for velocity loss
        rho: float = 1000.0,
        mu: float = 1.0,
        eta: float = 0.1,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float = 1.0,
        learnable_physics: bool = True,
        use_l1: bool = False,
        normalize_physics: bool = True,
        phase_threshold_ratio: float = 0.05,
        velocity_loss_type: str = 'mse'  # NEW: 'mse' or 'cosine' for velocity loss
    ):
        """
        Args:
            lambda_data: Weight for data reconstruction loss
            lambda_physics: Weight for physics (PDE) loss
            lambda_smooth: Weight for temporal smoothness
            lambda_magnitude: Weight for magnitude preservation
            lambda_phase: Weight for interferometric phase loss
            lambda_velocity: Weight for velocity field loss
            rho: Tissue density
            mu: Shear modulus
            eta: Viscosity
            dx, dy: Spatial steps
            dt: Temporal step
            learnable_physics: Whether physics params are learnable
            use_l1: Use L1 loss instead of MSE for data term
            normalize_physics: Whether to normalize physics loss magnitude
            phase_threshold_ratio: Fraction of max magnitude for tissue masking
            velocity_loss_type: 'mse' or 'cosine' for velocity loss computation
        """
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth
        self.lambda_magnitude = lambda_magnitude
        self.lambda_phase = lambda_phase
        self.lambda_velocity = lambda_velocity
        self.normalize_physics = normalize_physics
        
        # Running statistics for physics loss normalization
        self.register_buffer('physics_loss_ema', torch.tensor(1.0))
        self.register_buffer('data_loss_ema', torch.tensor(1.0))
        self.ema_decay = 0.99
        
        # Data loss
        if use_l1:
            self.data_loss = ComplexL1Loss(reduction='mean')
        else:
            self.data_loss = ComplexMSELoss(reduction='mean')
        
        # Interferometric phase loss (robust to phase wrapping)
        self.phase_loss_fn = InterferometricPhaseLoss(
            reduction='mean',
            threshold_ratio=phase_threshold_ratio
        )
        
        # Velocity loss (the ultimate goal for elastography)
        self.velocity_loss_fn = VelocityLoss(
            reduction='mean',
            threshold_ratio=phase_threshold_ratio,
            loss_type=velocity_loss_type
        )
        
        # Physics loss (PDE residual)
        self.wave_equation = KelvinVoigtWaveEquation(
            rho=rho, mu=mu, eta=eta,
            dx=dx, dy=dy, dt=dt,
            learnable_params=learnable_physics
        )
    
    def magnitude_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss on magnitude preservation.
        
        Ensures predicted complex values have similar magnitude to ground truth.
        """
        pred_mag = torch.sqrt(pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2 + 1e-8)
        target_mag = torch.sqrt(target[:, :, 0] ** 2 + target[:, :, 1] ** 2 + 1e-8)
        
        return F.mse_loss(pred_mag, target_mag)
    
    def temporal_smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Temporal smoothness regularization.
        
        Penalizes large temporal gradients (jerky motion).
        L_smooth = mean(|u[t+1] - u[t]|²)
        """
        # Temporal difference
        temporal_diff = pred[:, 1:] - pred[:, :-1]
        
        # Squared magnitude of difference
        diff_real = temporal_diff[:, :, 0]
        diff_imag = temporal_diff[:, :, 1]
        
        smoothness = (diff_real ** 2 + diff_imag ** 2).mean()
        
        return smoothness
    
    def physics_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Physics (PDE) loss.
        
        Computes how well the predicted sequence satisfies
        the Kelvin-Voigt viscoelastic wave equation.
        """
        # Compute PDE residual
        residual = self.wave_equation(pred)
        
        # L2 norm of residual
        physics_loss = (residual ** 2).mean()
        
        return physics_loss
    
    def set_lambda_physics(self, value: float):
        """Dynamically set physics loss weight (for warmup)."""
        self.lambda_physics = value
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total physics-informed loss.
        
        Args:
            pred: Predicted dense sequence (B, T_dense, 2, C, H, W)
            target: Ground truth dense sequence (B, T_dense, 2, C, H, W)
            return_components: Whether to return individual loss components
        
        Returns:
            total_loss: Combined loss (scalar)
            components: Dict of individual losses (if return_components=True)
        """
        # Data reconstruction loss
        l_data = self.data_loss(pred, target)
        
        # Interferometric phase loss (robust to phase wrapping)
        l_phase = self.phase_loss_fn(pred, target)
        
        # Velocity field loss (the ultimate goal for elastography)
        # Requires at least 2 frames for velocity computation
        if pred.shape[1] >= 2:
            l_velocity = self.velocity_loss_fn(pred, target)
        else:
            l_velocity = torch.tensor(0.0, device=pred.device)
        
        # Magnitude preservation loss
        l_mag = self.magnitude_loss(pred, target)
        
        # Temporal smoothness
        l_smooth = self.temporal_smoothness_loss(pred)
        
        # Physics loss computation
        # IMPORTANT: Always compute physics loss for EMA updates, even when lambda_physics = 0
        # This ensures EMA is properly initialized for when physics loss becomes active
        if pred.shape[1] >= 3:
            l_physics_raw = self.physics_loss(pred)
            
            # Check for NaN/Inf in physics loss
            if torch.isnan(l_physics_raw) or torch.isinf(l_physics_raw):
                l_physics = torch.tensor(0.0, device=pred.device)
                l_physics_raw = torch.tensor(0.0, device=pred.device)
            elif self.normalize_physics:
                # During training: update EMA and normalize
                # During validation: use stored EMA values to normalize (don't update)
                if self.training:
                    # Update EMA of loss magnitudes (only with valid values)
                    with torch.no_grad():
                        if not torch.isnan(l_data) and not torch.isinf(l_data):
                            self.data_loss_ema = self.ema_decay * self.data_loss_ema + (1 - self.ema_decay) * l_data.detach()
                        if not torch.isnan(l_physics_raw) and not torch.isinf(l_physics_raw):
                            self.physics_loss_ema = self.ema_decay * self.physics_loss_ema + (1 - self.ema_decay) * l_physics_raw.detach()
                
                # Scale physics loss to match data loss magnitude using stored EMA
                # This is applied in BOTH training and validation for consistency
                if self.physics_loss_ema > 1e-12:
                    scale_factor = (self.data_loss_ema / self.physics_loss_ema).clamp(min=1e-6, max=1e6)
                else:
                    scale_factor = torch.tensor(1.0, device=pred.device)
                l_physics = l_physics_raw * scale_factor
            else:
                l_physics = l_physics_raw
        else:
            l_physics = torch.tensor(0.0, device=pred.device)
            l_physics_raw = l_physics
        
        # Total loss - compute contributions only if lambda > 0
        physics_contribution = self.lambda_physics * l_physics if self.lambda_physics > 0 else 0.0
        phase_contribution = self.lambda_phase * l_phase if self.lambda_phase > 0 else 0.0
        velocity_contribution = self.lambda_velocity * l_velocity if self.lambda_velocity > 0 else 0.0
        
        total_loss = (
            self.lambda_data * l_data
            + phase_contribution
            + velocity_contribution
            + self.lambda_magnitude * l_mag
            + self.lambda_smooth * l_smooth
            + physics_contribution
        )
        
        # Final NaN check - if total loss is NaN, fall back to data loss only
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = self.lambda_data * l_data + self.lambda_magnitude * l_mag
            # If still NaN, use a placeholder
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(1.0, device=pred.device, requires_grad=True)
        
        if return_components:
            components = {
                'data': l_data.item(),
                'phase': l_phase.item(),
                'velocity': l_velocity.item() if isinstance(l_velocity, torch.Tensor) else l_velocity,
                'magnitude': l_mag.item(),
                'smoothness': l_smooth.item(),
                'physics': l_physics.item() if isinstance(l_physics, torch.Tensor) else l_physics,
                'physics_raw': l_physics_raw.item() if isinstance(l_physics_raw, torch.Tensor) else l_physics_raw,
                'total': total_loss.item()
            }
            return total_loss, components
        
        return total_loss


class WaveEquationConsistencyLoss(nn.Module):
    """
    PDE Consistency Loss (Wave Equation) - No Explicit Parameter Estimation.
    
    For a wave equation: ∂²u/∂t² = c² * ∇²u
    
    We don't know the wave speed 'c', but we know that the second time
    derivative and the Laplacian MUST be correlated (they differ only by
    a scalar factor c²).
    
    IMPROVEMENT: Weight by velocity magnitude to prevent trivial solutions.
    - Old approach: "Physics must hold everywhere" → Model cheats by making everything zero
    - New approach: "Physics must hold where the wave is moving" → Only active regions matter
    
    Loss = 1 - pearson_correlation(d²u/dt²[active], ∇²u[active])
    
    Where 'active' = regions where velocity magnitude > threshold
    
    This is physics-informed without requiring explicit parameter estimation!
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        eps: float = 1e-8,
        use_magnitude: bool = True,
        velocity_threshold: float = 0.1,  # Threshold for active wave region
        min_active_fraction: float = 0.01  # Minimum fraction of active pixels required
    ):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            eps: Small epsilon for numerical stability
            use_magnitude: If True, compute wave equation on magnitude.
                          If False, compute separately on real and imag.
            velocity_threshold: Normalized velocity threshold (0-1) for active region
            min_active_fraction: Minimum fraction of pixels that must be active
        """
        super(WaveEquationConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.use_magnitude = use_magnitude
        self.velocity_threshold = velocity_threshold
        self.min_active_fraction = min_active_fraction
        
        # Laplacian kernel (5-point stencil)
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]]]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def _compute_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².
        
        Args:
            u: Tensor of shape (B, T, C, H, W)
        
        Returns:
            Laplacian of shape (B, T, C, H-2, W-2) (valid convolution)
        """
        B, T, C, H, W = u.shape
        # Reshape to (B*T*C, 1, H, W) for convolution
        u_flat = u.reshape(B * T * C, 1, H, W)
        
        # Ensure kernel is on same device as input
        kernel = self.laplacian_kernel.to(u_flat.device, u_flat.dtype)
        
        # Apply Laplacian kernel (no padding to avoid boundary effects)
        laplacian_flat = F.conv2d(u_flat, kernel, padding=0)
        
        # Reshape back: (B, T, C, H-2, W-2)
        _, _, H_out, W_out = laplacian_flat.shape
        return laplacian_flat.reshape(B, T, C, H_out, W_out)
    
    def _compute_d2u_dt2(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute second time derivative ∂²u/∂t².
        
        Using central differences: d²u/dt² ≈ u[t+1] - 2*u[t] + u[t-1]
        
        Args:
            u: Tensor of shape (B, T, C, H, W)
        
        Returns:
            d²u/dt² of shape (B, T-2, C, H, W)
        """
        return u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]
    
    def _compute_velocity_mask(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity-based mask for active wave regions.
        
        Velocity = |u[t+1] - u[t]| (temporal difference)
        Mask = normalized velocity > threshold
        
        Args:
            u: Tensor of shape (B, T, C, H, W)
        
        Returns:
            mask: Boolean tensor of shape (B, T-2, C, H-2, W-2) matching PDE terms
        """
        # Compute velocity magnitude (temporal difference)
        velocity = torch.abs(u[:, 1:] - u[:, :-1])  # (B, T-1, C, H, W)
        
        # Average velocity over the middle frames (to match d²u/dt² shape)
        # We need T-2 frames, so average consecutive velocity frames
        velocity_middle = (velocity[:, 1:] + velocity[:, :-1]) / 2  # (B, T-2, C, H, W)
        
        # Crop spatially to match Laplacian output
        velocity_cropped = velocity_middle[:, :, :, 1:-1, 1:-1]  # (B, T-2, C, H-2, W-2)
        
        # Normalize to 0-1 range (per-batch normalization)
        vel_max = velocity_cropped.max() + self.eps
        velocity_normalized = velocity_cropped / vel_max
        
        # Create soft mask
        mask = velocity_normalized > self.velocity_threshold
        
        return mask, velocity_normalized
    
    def _weighted_pearson_correlation(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Pearson correlation coefficient only in masked regions.
        
        Args:
            x, y: Tensors of the same shape
            mask: Boolean mask of same shape (True = include in correlation)
        
        Returns:
            Correlation coefficient (scalar)
        """
        # Extract values in active region
        x_active = x[mask]
        y_active = y[mask]
        
        if x_active.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        
        # Center the data
        x_centered = x_active - x_active.mean()
        y_centered = y_active - y_active.mean()
        
        # Compute covariance and standard deviations
        cov = (x_centered * y_centered).mean()
        std_x = torch.sqrt((x_centered ** 2).mean() + self.eps)
        std_y = torch.sqrt((y_centered ** 2).mean() + self.eps)
        
        # Pearson correlation
        corr = cov / (std_x * std_y + self.eps)
        
        # Clamp to [-1, 1] for numerical stability
        return torch.clamp(corr, -1.0, 1.0)
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity-weighted wave equation consistency loss.
        
        Only verifies physics where the wave is actually moving.
        
        Args:
            pred: Predicted sequence (B, T, 2, C, H, W) where dim 2 is [real, imag]
                  Must have T >= 3 for second time derivative.
        
        Returns:
            PDE consistency loss (scalar)
        """
        B, T, _, C, H, W = pred.shape
        
        # Need at least 3 frames for second time derivative
        if T < 3:
            return torch.tensor(0.0, device=pred.device)
        
        if self.use_magnitude:
            # Compute magnitude: |z| = sqrt(real² + imag²)
            # WARNING: This is physically incorrect for shear waves!
            # The magnitude is just the static tissue speckle pattern.
            real = pred[:, :, 0, :, :, :]  # (B, T, C, H, W)
            imag = pred[:, :, 1, :, :, :]
            u = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        else:
            # Compute PHASE: angle(z) = atan2(imag, real)
            # This is the displacement field that the shear wave equation governs.
            # The wave travels through phase shifts, not magnitude changes.
            real = pred[:, :, 0, :, :, :]  # (B, T, C, H, W)
            imag = pred[:, :, 1, :, :, :]
            u = torch.atan2(imag + self.eps, real + self.eps)  # Phase in [-π, π]
        
        # Compute second time derivative: d²u/dt²
        # Shape: (B, T-2, C, H, W)
        d2u_dt2 = self._compute_d2u_dt2(u)
        
        # Compute Laplacian on middle frames (to match time derivative shape)
        u_middle = u[:, 1:-1]  # (B, T-2, C, H, W)
        laplacian = self._compute_laplacian(u_middle)  # (B, T-2, C, H-2, W-2)
        
        # Crop d2u_dt2 to match Laplacian spatial dims
        d2u_dt2_cropped = d2u_dt2[:, :, :, 1:-1, 1:-1]  # (B, T-2, C, H-2, W-2)
        
        # Compute velocity-based mask for active wave regions
        mask, velocity_normalized = self._compute_velocity_mask(u)
        
        # Check if we have enough active pixels
        active_fraction = mask.float().mean()
        if active_fraction < self.min_active_fraction:
            # Not enough active region - fall back to global correlation
            # This prevents division issues when wave hasn't started yet
            x_flat = d2u_dt2_cropped.flatten()
            y_flat = laplacian.flatten()
            x_centered = x_flat - x_flat.mean()
            y_centered = y_flat - y_flat.mean()
            cov = (x_centered * y_centered).mean()
            std_x = torch.sqrt((x_centered ** 2).mean() + self.eps)
            std_y = torch.sqrt((y_centered ** 2).mean() + self.eps)
            correlation = cov / (std_x * std_y + self.eps)
            correlation = torch.clamp(correlation, -1.0, 1.0)
        else:
            # Compute correlation only in active (moving) regions
            correlation = self._weighted_pearson_correlation(
                d2u_dt2_cropped, laplacian, mask
            )
        
        # Loss: 1 - correlation
        # Perfect wave in active region: corr = 1 → loss = 0
        # No wave structure: corr = 0 → loss = 1
        loss = 1.0 - correlation
        
        return loss


class SupervisedOnlyLoss(nn.Module):
    """
    Simpler loss without explicit physics parameters - for ablation studies.
    
    Total Loss = λ_data * L_data + λ_phase * L_phase + λ_velocity * L_velocity 
                 + λ_smooth * L_smooth + λ_pde * L_pde
    
    Now includes:
    - Interferometric phase loss for phase-accurate reconstruction
    - Velocity loss for velocity-accurate reconstruction  
    - PDE consistency loss (wave equation) without explicit parameters
    - Physics-first training support (can disable data loss temporarily)
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_phase: float = 0.5,
        lambda_velocity: float = 0.5,
        lambda_pde: float = 0.1,  # NEW: weight for PDE consistency loss
        phase_threshold_ratio: float = 0.05,
        velocity_loss_type: str = 'mse',
        use_l1: bool = False
    ):
        super(SupervisedOnlyLoss, self).__init__()
        
        self.lambda_data = lambda_data
        self.lambda_data_original = lambda_data  # Store original for physics-first
        self.lambda_smooth = lambda_smooth
        self.lambda_phase = lambda_phase
        self.lambda_velocity = lambda_velocity
        self.lambda_pde = lambda_pde
        
        if use_l1:
            self.data_loss = ComplexL1Loss(reduction='mean')
        else:
            self.data_loss = ComplexMSELoss(reduction='mean')
        
        # Interferometric phase loss (robust to phase wrapping)
        self.phase_loss_fn = InterferometricPhaseLoss(
            reduction='mean',
            threshold_ratio=phase_threshold_ratio
        )
        
        # Velocity loss (the ultimate goal for elastography)
        self.velocity_loss_fn = VelocityLoss(
            reduction='mean',
            threshold_ratio=phase_threshold_ratio,
            loss_type=velocity_loss_type
        )
        
        # PDE consistency loss (wave equation without explicit parameters)
        # CRITICAL: use_magnitude=False because the shear wave equation governs
        # displacement (phase), NOT echogenicity (magnitude). The magnitude is 
        # just the static tissue speckle pattern - the wave travels through phase.
        self.pde_loss_fn = WaveEquationConsistencyLoss(
            reduction='mean',
            use_magnitude=False  # Must use phase for wave equation, not magnitude
        )
    
    def set_physics_first_mode(self, enabled: bool):
        """
        Enable/disable physics-first mode.
        
        When enabled: lambda_data = 0 (model learns motion dynamics only)
        When disabled: lambda_data = original value (full training)
        """
        if enabled:
            self.lambda_data = 0.0
        else:
            self.lambda_data = self.lambda_data_original
    
    def temporal_smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        temporal_diff = pred[:, 1:] - pred[:, :-1]
        diff_real = temporal_diff[:, :, 0]
        diff_imag = temporal_diff[:, :, 1]
        return (diff_real ** 2 + diff_imag ** 2).mean()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        l_data = self.data_loss(pred, target)
        l_phase = self.phase_loss_fn(pred, target)
        l_smooth = self.temporal_smoothness_loss(pred)
        
        # Velocity loss requires at least 2 frames
        if pred.shape[1] >= 2:
            l_velocity = self.velocity_loss_fn(pred, target)
        else:
            l_velocity = torch.tensor(0.0, device=pred.device)
        
        # PDE consistency loss requires at least 3 frames
        if pred.shape[1] >= 3 and self.lambda_pde > 0:
            l_pde = self.pde_loss_fn(pred)
        else:
            l_pde = torch.tensor(0.0, device=pred.device)
        
        phase_contribution = self.lambda_phase * l_phase if self.lambda_phase > 0 else 0.0
        velocity_contribution = self.lambda_velocity * l_velocity if self.lambda_velocity > 0 else 0.0
        pde_contribution = self.lambda_pde * l_pde if self.lambda_pde > 0 else 0.0
        
        total_loss = (
            self.lambda_data * l_data 
            + phase_contribution 
            + velocity_contribution
            + pde_contribution
            + self.lambda_smooth * l_smooth
        )
        
        if return_components:
            components = {
                'data': l_data.item(),
                'phase'  : l_phase.item(),
                'velocity': l_velocity.item() if isinstance(l_velocity, torch.Tensor) else l_velocity,
                'pde': l_pde.item() if isinstance(l_pde, torch.Tensor) else l_pde,
                'smoothness': l_smooth.item(),
                'total': total_loss.item()
            }
            return total_loss, components
        
        return total_loss

def create_physics_informed_loss(
    lambda_data: float = 1.0,
    lambda_physics: float = 0.01,
    lambda_smooth: float = 0.001,
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 1.0
) -> PhysicsInformedLoss:
    """Factory function to create physics-informed loss."""
    return PhysicsInformedLoss(
        lambda_data=lambda_data,
        lambda_physics=lambda_physics,
        lambda_smooth=lambda_smooth,
        dx=dx, dy=dy, dt=dt
    )


