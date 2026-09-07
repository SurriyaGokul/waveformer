# Model Architectures

from .complex_fno import (
    ComplexReLU,
    ComplexSpectralConv2d,
    ComplexFNOLayer,
    ComplexFNOBlock,
    ComplexSpectralRefinement,
    create_complex_spectral_refinement
)

# Velocity Estimation Architecture
from .velocity_unet import (
    ComplexToRealUNet,
    ComplexConvBlock,
    ComplexDownBlock,
    ComplexToRealBridge,
    RealConvBlock,
    RealUpBlock,
    SimpleTemporalBottleneck,
    create_velocity_model,
    count_parameters,
)

from .temporal_fno import (
    ComplexConvGRUCell,
    ComplexConvGRU,
    TemporalPhysicsBlock,
    TemporalComplexSpectralRefinement,
    create_temporal_spectral_refinement
)

# Recursive Midpoint Architecture (memory-efficient temporal modeling)
from .midpoint_fno import (
    ComplexConvGRUCell,       # Complex-valued ConvGRU cell with proper complex arithmetic
    ComplexFusionLayer,       # Learnable fusion for bi-directional GRU outputs
    MidpointTemporalBlock,    # Bi-Directional ComplexConvGRU + Fusion + FNO
    MidpointSpectralRefinement,
    create_midpoint_spectral_refinement
)