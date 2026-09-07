import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from src.architectures.complex_modules import ComplexConv2d


class ComplexConvLSTMCell(nn.Module):
    """
    Complex-Valued Convolutional LSTM Cell.
    
    Adapts the standard ConvLSTM cell to use ComplexConv2d operations.
    Both hidden state (H_t) and cell state (C_t) are complex-valued tensors.
    
    This cell captures the wave momentum (velocity/direction) between frames
    for temporal super-resolution of shear wave data.
    
    LSTM equations (adapted for complex):
        i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)   # Input gate
        f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)   # Forget gate
        o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)   # Output gate
        g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)      # Cell candidate
        c_t = f_t * c_{t-1} + i_t * g_t                    # Cell state
        h_t = o_t * tanh(c_t)                              # Hidden state
    
    For complex, we use modulus-based sigmoid/tanh applied to complex tensors.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        """
        Args:
            input_channels: Number of input channels (per real/imag component)
            hidden_channels: Number of hidden channels (per real/imag component)
            kernel_size: Size of the convolving kernel
            bias: If True, adds a learnable bias
        """
        super(ComplexConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Combined convolution for all gates (more efficient)
        # Input to gates: 4 gates * hidden_channels
        self.conv_input = ComplexConv2d(
            in_channels=input_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # Hidden to gates
        self.conv_hidden = ComplexConv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False
        )
    
    def _complex_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex sigmoid activation.
        
        Applies sigmoid to the magnitude while preserving phase:
            sigmoid(|z|) * (z / |z|)
        
        This ensures gate values are bounded [0, 1] in magnitude.
        """
        real = x[:, 0, :, :, :]
        imag = x[:, 1, :, :, :]
        
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        # Apply sigmoid to magnitude
        sig_magnitude = torch.sigmoid(magnitude)
        
        out_real = sig_magnitude * phase_real
        out_imag = sig_magnitude * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex tanh activation.
        
        Applies tanh to the magnitude while preserving phase:
            tanh(|z|) * (z / |z|)
        
        This bounds the magnitude to [-1, 1] while keeping phase.
        """
        real = x[:, 0, :, :, :]
        imag = x[:, 1, :, :, :]
        
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        # Apply tanh to magnitude
        tanh_magnitude = torch.tanh(magnitude)
        
        out_real = tanh_magnitude * phase_real
        out_imag = tanh_magnitude * phase_imag
        
        return torch.stack([out_real, out_imag], dim=1)
    
    def _complex_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Element-wise complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)
        """
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
        """
        Forward pass for one time step.
        
        Args:
            x: Input tensor of shape (Batch, 2, InputChannels, Height, Width)
            hidden_state: Tuple of (h_prev, c_prev) or None for initial step
                - h_prev: (Batch, 2, HiddenChannels, Height, Width)
                - c_prev: (Batch, 2, HiddenChannels, Height, Width)
        
        Returns:
            Tuple of (h_new, c_new) with same shapes as hidden state
        """
        batch_size = x.size(0)
        height = x.size(3)
        width = x.size(4)
        device = x.device
        dtype = x.dtype
        
        # Initialize hidden state if not provided
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
        
        # Compute gates: combined convolution then split
        # Shape: (Batch, 2, 4*HiddenChannels, H, W)
        gates_input = self.conv_input(x)
        gates_hidden = self.conv_hidden(h_prev)
        gates = gates_input + gates_hidden
        
        # Split into 4 gates along channel dimension
        # Each gate: (Batch, 2, HiddenChannels, H, W)
        i_gate, f_gate, o_gate, g_gate = torch.split(
            gates, self.hidden_channels, dim=2
        )
        
        # Apply activations
        i_t = self._complex_sigmoid(i_gate)  # Input gate
        f_t = self._complex_sigmoid(f_gate)  # Forget gate
        o_t = self._complex_sigmoid(o_gate)  # Output gate
        g_t = self._complex_tanh(g_gate)     # Cell candidate
        
        # Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
        c_new = self._complex_multiply(f_t, c_prev) + self._complex_multiply(i_t, g_t)
        
        # Update hidden state: h_t = o_t * tanh(c_t)
        h_new = self._complex_multiply(o_t, self._complex_tanh(c_new))
        
        return h_new, c_new


class ComplexConvLSTM(nn.Module):
    """
    Multi-layer Complex Convolutional LSTM.
    
    Processes sequences of complex-valued feature maps through multiple
    ComplexConvLSTMCell layers. Used as the temporal bottleneck in
    the PI-CV-RecUNet architecture.
    
    For temporal super-resolution, this module:
    1. Takes encoded features from sparse time frames
    2. Unrolls to predict intermediate latent states
    3. Outputs dense sequence of latent features for decoding
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
        """
        Args:
            input_channels: Number of input channels (per real/imag)
            hidden_channels: Number of hidden channels (can be int or list for each layer)
            num_layers: Number of stacked LSTM layers
            kernel_size: Convolution kernel size
            bias: If True, adds learnable bias
            return_all_layers: If True, return hidden states from all layers
        """
        super(ComplexConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        
        # Handle hidden_channels as int or list
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        
        assert len(hidden_channels) == num_layers, \
            "hidden_channels must match num_layers"
        
        self.hidden_channels = hidden_channels
        
        # Build layers
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ComplexConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size,
                    bias=bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
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
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the stacked LSTM.
        
        Args:
            x: Input sequence of shape (Batch, SeqLen, 2, Channels, Height, Width)
            hidden_state: List of (h, c) tuples for each layer, or None
        
        Returns:
            output: Output sequence (Batch, SeqLen, 2, HiddenChannels[-1], H, W)
            last_states: List of final (h, c) for each layer
        """
        batch_size, seq_len = x.size(0), x.size(1)
        height, width = x.size(4), x.size(5)
        device, dtype = x.device, x.dtype
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device, dtype)
        
        # Store outputs
        layer_output_list = []
        last_state_list = []
        
        # Current input to process
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each time step
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[:, t, :, :, :, :],
                    (h, c)
                )
                output_inner.append(h)
            
            # Stack outputs: (Batch, SeqLen, 2, HiddenChannels, H, W)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if self.return_all_layers:
            return layer_output_list, last_state_list
        else:
            return layer_output_list[-1], last_state_list
    
    def unroll_prediction(
        self,
        initial_input: torch.Tensor,
        num_steps: int,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Unroll the LSTM for prediction (autoregressive mode).
        
        Given an initial input, predict num_steps future states by feeding
        each output as the next input. Used for temporal super-resolution.
        
        Args:
            initial_input: Starting frame (Batch, 2, Channels, Height, Width)
            num_steps: Number of steps to predict
            hidden_state: Optional initial hidden state
        
        Returns:
            predictions: Predicted sequence (Batch, num_steps, 2, HiddenChannels, H, W)
        """
        batch_size = initial_input.size(0)
        height, width = initial_input.size(3), initial_input.size(4)
        device, dtype = initial_input.device, initial_input.dtype
        
        # Initialize hidden states
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device, dtype)
        
        predictions = []
        current_input = initial_input
        
        for step in range(num_steps):
            # Pass through all layers
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](current_input, (h, c))
                hidden_state[layer_idx] = (h, c)
                current_input = h  # Output becomes input to next layer
            
            predictions.append(h)
            current_input = h  # Use prediction as next input (autoregressive)
        
        # Stack: (Batch, num_steps, 2, HiddenChannels, H, W)
        return torch.stack(predictions, dim=1)
