import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from src.architectures.complex_modules import ComplexConv2d


class ComplexConvLSTMCell(nn.Module):
    """
    Complex-Valued Convolutional LSTM Cell.
    
    (Same implementation as in complex_lstm.py, included here for completeness
    and to avoid circular imports)
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        super(ComplexConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.conv_input = ComplexConv2d(
            in_channels=input_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        self.conv_hidden = ComplexConv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False
        )
    
    def _complex_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0, :, :, :]
        imag = x[:, 1, :, :, :]
        
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        sig_magnitude = torch.sigmoid(magnitude)
        
        out_real = sig_magnitude * phase_real
        out_imag = sig_magnitude * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_tanh(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0, :, :, :]
        imag = x[:, 1, :, :, :]
        
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        tanh_magnitude = torch.tanh(magnitude)
        
        out_real = tanh_magnitude * phase_real
        out_imag = tanh_magnitude * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_real, a_imag = a[:, 0], a[:, 1]
        b_real, b_imag = b[:, 0], b[:, 1]
        
        out_real = a_real * b_real - a_imag * b_imag
        out_imag = a_real * b_imag + a_imag * b_real
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        height = x.size(3)
        width = x.size(4)
        device = x.device
        dtype = x.dtype
        
        if hidden_state is None:
            h_prev = torch.zeros(
                batch_size, 2, self.hidden_channels, height, width,
                device=device, dtype=dtype
            )
            c_prev = torch.zeros(
                batch_size, 2, self.hidden_channels, height, width,
                device=device, dtype=dtype
            )
        else:
            h_prev, c_prev = hidden_state
        
        gates_input = self.conv_input(x)
        gates_hidden = self.conv_hidden(h_prev)
        gates = gates_input + gates_hidden
        
        i_gate, f_gate, o_gate, g_gate = torch.split(
            gates, self.hidden_channels, dim=2
        )
        
        i_t = self._complex_sigmoid(i_gate)
        f_t = self._complex_sigmoid(f_gate)
        o_t = self._complex_sigmoid(o_gate)
        g_t = self._complex_tanh(g_gate)
        
        c_new = self._complex_multiply(f_t, c_prev) + self._complex_multiply(i_t, g_t)
        h_new = self._complex_multiply(o_t, self._complex_tanh(c_new))
        
        return h_new, c_new


class BidirectionalComplexConvLSTM(nn.Module):
    """
    Bidirectional Complex Convolutional LSTM.
    
    Processes sequences in both forward and backward directions, then combines
    the outputs. This allows the model to use information from both past and
    future frames when making predictions for intermediate time steps.
    
    For temporal super-resolution:
    - Forward LSTM: Captures motion from past to future
    - Backward LSTM: Captures motion from future to past
    - Combined: Full temporal context for accurate interpolation
    
    Combination modes:
    - 'concat': Concatenate forward and backward outputs (doubles channels)
    - 'sum': Element-wise sum of forward and backward
    - 'mean': Element-wise mean of forward and backward
    - 'complex_multiply': Complex multiplication of forward and backward
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        combine_mode: str = 'concat',
        return_all_layers: bool = False
    ):
        """
        Args:
            input_channels: Number of input channels (per real/imag)
            hidden_channels: Hidden channels per direction (total output = 2x if concat)
            num_layers: Number of stacked LSTM layers per direction
            kernel_size: Convolution kernel size
            bias: If True, adds learnable bias
            combine_mode: How to combine forward/backward outputs
                         ('concat', 'sum', 'mean', 'complex_multiply')
            return_all_layers: If True, return hidden states from all layers
        """
        super(BidirectionalComplexConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.combine_mode = combine_mode
        self.return_all_layers = return_all_layers
        
        # Handle hidden_channels as int or list
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        
        assert len(hidden_channels) == num_layers, \
            "hidden_channels must match num_layers"
        
        self.hidden_channels = hidden_channels
        
        # Output channels depend on combine mode
        if combine_mode == 'concat':
            self.output_channels = hidden_channels[-1] * 2
        else:
            self.output_channels = hidden_channels[-1]
        
        # Build forward LSTM cells
        self.forward_cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input = input_channels if i == 0 else hidden_channels[i - 1]
            self.forward_cells.append(
                ComplexConvLSTMCell(
                    input_channels=cur_input,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size,
                    bias=bias
                )
            )
        
        # Build backward LSTM cells (separate parameters)
        self.backward_cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input = input_channels if i == 0 else hidden_channels[i - 1]
            self.backward_cells.append(
                ComplexConvLSTMCell(
                    input_channels=cur_input,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size,
                    bias=bias
                )
            )
        
        # Optional projection layer for sum/mean modes to allow learned combination
        if combine_mode in ['sum', 'mean']:
            self.projection = None  # Can add learnable weights if needed
        elif combine_mode == 'concat':
            self.projection = None  # Output is concatenated
    
    def _init_hidden(
        self, batch_size: int, height: int, width: int,
        device: torch.device, dtype: torch.dtype
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            h = torch.zeros(
                batch_size, 2, self.hidden_channels[i], height, width,
                device=device, dtype=dtype
            )
            c = torch.zeros(
                batch_size, 2, self.hidden_channels[i], height, width,
                device=device, dtype=dtype
            )
            init_states.append((h, c))
        return init_states
    
    def _complex_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise complex multiplication."""
        a_real, a_imag = a[:, :, 0], a[:, :, 1]
        b_real, b_imag = b[:, :, 0], b[:, :, 1]
        
        out_real = a_real * b_real - a_imag * b_imag
        out_imag = a_real * b_imag + a_imag * b_real
        
        return torch.stack([out_real, out_imag], dim=2)
    
    def _run_direction(
        self,
        x: torch.Tensor,
        cells: nn.ModuleList,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        reverse: bool = False
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run LSTM in one direction (forward or backward).
        
        Args:
            x: Input sequence (Batch, SeqLen, 2, Channels, H, W)
            cells: List of LSTM cells
            hidden_state: Initial hidden states
            reverse: If True, process sequence in reverse order
        
        Returns:
            output: Output sequence (Batch, SeqLen, 2, HiddenChannels, H, W)
            last_states: Final hidden states for each layer
        """
        batch_size, seq_len = x.size(0), x.size(1)
        height, width = x.size(4), x.size(5)
        device, dtype = x.device, x.dtype
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device, dtype)
        
        # Determine processing order
        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        
        cur_layer_input = x
        layer_outputs = []
        last_states = []
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in time_steps:
                h, c = cells[layer_idx](
                    cur_layer_input[:, t, :, :, :, :],
                    (h, c)
                )
                output_inner.append(h)
            
            # Reverse output order if processing backward to align with forward
            if reverse:
                output_inner = output_inner[::-1]
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_outputs.append(layer_output)
            last_states.append((h, c))
        
        return layer_outputs[-1], last_states
    
    def forward(
        self,
        x: torch.Tensor,
        forward_hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        backward_hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through bidirectional LSTM.
        
        Args:
            x: Input sequence (Batch, SeqLen, 2, Channels, Height, Width)
            forward_hidden: Initial hidden state for forward direction
            backward_hidden: Initial hidden state for backward direction
        
        Returns:
            output: Combined output (Batch, SeqLen, 2, OutChannels, H, W)
            forward_states: Final hidden states for forward direction
            backward_states: Final hidden states for backward direction
        """
        # Process forward direction
        forward_output, forward_states = self._run_direction(
            x, self.forward_cells, forward_hidden, reverse=False
        )
        
        # Process backward direction
        backward_output, backward_states = self._run_direction(
            x, self.backward_cells, backward_hidden, reverse=True
        )
        
        # Combine outputs based on mode
        if self.combine_mode == 'concat':
            # Concatenate along channel dimension
            # forward_output: (B, T, 2, C, H, W)
            output = torch.cat([forward_output, backward_output], dim=3)
        
        elif self.combine_mode == 'sum':
            output = forward_output + backward_output
        
        elif self.combine_mode == 'mean':
            output = (forward_output + backward_output) / 2.0
        
        elif self.combine_mode == 'complex_multiply':
            # Complex multiplication at each time step
            output = self._complex_multiply(forward_output, backward_output)
        
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")
        
        return output, forward_states, backward_states
    
    def interpolate_frames(
        self,
        sparse_frames: torch.Tensor,
        upsampling_factor: int
    ) -> torch.Tensor:
        """
        Interpolate intermediate frames between sparse input frames.
        
        Uses bidirectional context to generate intermediate frames.
        
        Args:
            sparse_frames: Sparse input sequence (Batch, T_sparse, 2, C, H, W)
            upsampling_factor: Number of frames to generate between each pair
        
        Returns:
            dense_frames: Dense sequence (Batch, T_dense, 2, OutChannels, H, W)
                         where T_dense = T_sparse + (T_sparse-1) * upsampling_factor
        """
        batch_size, num_sparse, _, channels, height, width = sparse_frames.shape
        device, dtype = sparse_frames.device, sparse_frames.dtype
        
        # First, get bidirectional features for all sparse frames
        output, _, _ = self.forward(sparse_frames)
        
        # Now interpolate between consecutive frames
        dense_outputs = []
        
        for t in range(num_sparse - 1):
            # Add current frame features
            dense_outputs.append(output[:, t:t+1, :, :, :, :])
            
            # Get features from current and next frame
            feat_current = output[:, t, :, :, :, :]  # (B, 2, C, H, W)
            feat_next = output[:, t+1, :, :, :, :]
            
            # Linear interpolation in feature space
            for step in range(1, upsampling_factor + 1):
                alpha = step / (upsampling_factor + 1)
                # Linear blend of complex features
                interp_real = (1 - alpha) * feat_current[:, 0] + alpha * feat_next[:, 0]
                interp_imag = (1 - alpha) * feat_current[:, 1] + alpha * feat_next[:, 1]
                interp_feat = torch.stack([interp_real, interp_imag], dim=1)
                dense_outputs.append(interp_feat.unsqueeze(1))
        
        # Add last frame
        dense_outputs.append(output[:, -1:, :, :, :, :])
        
        # Concatenate all frames
        dense_sequence = torch.cat(dense_outputs, dim=1)
        
        return dense_sequence


class AttentionBidirectionalComplexConvLSTM(nn.Module):
    """
    Bidirectional Complex ConvLSTM with Temporal Attention.
    
    Adds an attention mechanism to weight the importance of forward
    and backward features at each time step, allowing the model to
    adaptively combine directional information.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        return_all_layers: bool = False
    ):
        super(AttentionBidirectionalComplexConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        
        self.hidden_channels = hidden_channels
        self.output_channels = hidden_channels[-1]
        
        # Forward and backward cells
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        
        for i in range(num_layers):
            cur_input = input_channels if i == 0 else hidden_channels[i - 1]
            self.forward_cells.append(
                ComplexConvLSTMCell(cur_input, hidden_channels[i], kernel_size, bias)
            )
            self.backward_cells.append(
                ComplexConvLSTMCell(cur_input, hidden_channels[i], kernel_size, bias)
            )
        
        # Attention mechanism: learns to weight forward vs backward
        # Uses both forward and backward features to compute attention weights
        self.attention_conv = ComplexConv2d(
            in_channels=hidden_channels[-1] * 2,  # Concatenated forward + backward
            out_channels=2,  # 2 attention weights (forward, backward)
            kernel_size=1,
            padding=0
        )
    
    def _init_hidden(self, batch_size, height, width, device, dtype):
        init_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, 2, self.hidden_channels[i], height, width, device=device, dtype=dtype)
            c = torch.zeros(batch_size, 2, self.hidden_channels[i], height, width, device=device, dtype=dtype)
            init_states.append((h, c))
        return init_states
    
    def _run_direction(self, x, cells, hidden_state, reverse=False):
        batch_size, seq_len = x.size(0), x.size(1)
        height, width = x.size(4), x.size(5)
        device, dtype = x.device, x.dtype
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device, dtype)
        
        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        
        cur_layer_input = x
        last_states = []
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in time_steps:
                h, c = cells[layer_idx](cur_layer_input[:, t], (h, c))
                output_inner.append(h)
            
            if reverse:
                output_inner = output_inner[::-1]
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            last_states.append((h, c))
        
        return layer_output, last_states
    
    def forward(
        self,
        x: torch.Tensor,
        forward_hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        backward_hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with attention-weighted combination.
        
        Returns:
            output: Attention-weighted output (Batch, SeqLen, 2, HiddenChannels, H, W)
            forward_states: Final forward hidden states
            backward_states: Final backward hidden states
        """
        # Get forward and backward outputs
        forward_output, forward_states = self._run_direction(
            x, self.forward_cells, forward_hidden, reverse=False
        )
        backward_output, backward_states = self._run_direction(
            x, self.backward_cells, backward_hidden, reverse=True
        )
        
        # Compute attention weights for each time step
        batch_size, seq_len = x.size(0), x.size(1)
        outputs = []
        
        for t in range(seq_len):
            fwd_t = forward_output[:, t]   # (B, 2, C, H, W)
            bwd_t = backward_output[:, t]  # (B, 2, C, H, W)
            
            # Concatenate for attention computation
            concat_feat = torch.cat([fwd_t, bwd_t], dim=2)  # (B, 2, 2C, H, W)
            
            # Compute attention logits
            attn_logits = self.attention_conv(concat_feat)  # (B, 2, 2, H, W)
            
            # Get magnitude for softmax (use real part magnitude)
            attn_real = attn_logits[:, 0, :, :, :]  # (B, 2, H, W)
            attn_weights = torch.softmax(attn_real, dim=1)  # Softmax over 2 directions
            
            # Apply weights
            weight_fwd = attn_weights[:, 0:1, :, :].unsqueeze(1)  # (B, 1, 1, H, W)
            weight_bwd = attn_weights[:, 1:2, :, :].unsqueeze(1)
            
            # Weighted combination
            combined_real = weight_fwd * fwd_t[:, 0:1] + weight_bwd * bwd_t[:, 0:1]
            combined_imag = weight_fwd * fwd_t[:, 1:2] + weight_bwd * bwd_t[:, 1:2]
            combined = torch.cat([combined_real, combined_imag], dim=1)
            
            outputs.append(combined)
        
        output = torch.stack(outputs, dim=1)
        
        return output, forward_states, backward_states


# ============================================================================
# Testing / Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Bidirectional ComplexConvLSTM modules...")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 5
    channels = 32
    height, width = 32, 32
    
    # Create sequence input: (Batch, SeqLen, 2, Channels, H, W)
    x_seq = torch.randn(batch_size, seq_len, 2, channels, height, width)
    print(f"Input shape: {x_seq.shape}")
    
    # Test BidirectionalComplexConvLSTM with different combine modes
    print("\n--- Testing BidirectionalComplexConvLSTM ---")
    
    for mode in ['concat', 'sum', 'mean', 'complex_multiply']:
        bilstm = BidirectionalComplexConvLSTM(
            input_channels=channels,
            hidden_channels=64,
            num_layers=1,
            kernel_size=3,
            combine_mode=mode
        )
        output, fwd_states, bwd_states = bilstm(x_seq)
        print(f"  Mode '{mode}': output shape = {output.shape}")
    
    # Test frame interpolation
    print("\n--- Testing Frame Interpolation ---")
    bilstm = BidirectionalComplexConvLSTM(
        input_channels=channels,
        hidden_channels=64,
        num_layers=2,
        combine_mode='concat'
    )
    
    # Simulate sparse frames (5 frames, want to interpolate to get more)
    sparse_frames = x_seq
    upsampling_factor = 3  # Generate 3 intermediate frames between each pair
    
    dense_frames = bilstm.interpolate_frames(sparse_frames, upsampling_factor)
    expected_dense = seq_len + (seq_len - 1) * upsampling_factor
    print(f"Sparse frames: {sparse_frames.shape[1]}")
    print(f"Dense frames: {dense_frames.shape[1]} (expected: {expected_dense})")
    
    # Test AttentionBidirectionalComplexConvLSTM
    print("\n--- Testing AttentionBidirectionalComplexConvLSTM ---")
    attn_bilstm = AttentionBidirectionalComplexConvLSTM(
        input_channels=channels,
        hidden_channels=64,
        num_layers=1,
        kernel_size=3
    )
    attn_output, _, _ = attn_bilstm(x_seq)
    print(f"Attention BiLSTM output shape: {attn_output.shape}")
    
    # Compare parameter counts
    print("\n--- Parameter Counts ---")
    bilstm_params = sum(p.numel() for p in bilstm.parameters())
    attn_bilstm_params = sum(p.numel() for p in attn_bilstm.parameters())
    print(f"BidirectionalComplexConvLSTM: {bilstm_params:,} parameters")
    print(f"AttentionBidirectionalComplexConvLSTM: {attn_bilstm_params:,} parameters")
    
    print("\n" + "=" * 60)
    print("✓ All Bidirectional ComplexConvLSTM modules working correctly!")
