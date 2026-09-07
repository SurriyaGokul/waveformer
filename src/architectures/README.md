# `src/architectures/`

Model definitions. Everything here operates on complex-valued ultrasound IQ data
(represented as separate real/imaginary channels), spatially downsampled and temporally
sparse, and reconstructs dense, high-frame-rate tissue velocity fields.

## Complex-valued building blocks

| File | Contents |
|---|---|
| `complex_modules.py` | `ComplexConv2d`, `ComplexBatchNorm2d`, `NaiveComplexBatchNorm2d` — base complex-arithmetic layers reused across all models |
| `complex_lstm.py` | `ComplexConvLSTMCell`, `ComplexConvLSTM` |
| `complex_bilstm.py` | `ComplexConvLSTMCell`, `BidirectionalComplexConvLSTM`, `AttentionBidirectionalComplexConvLSTM` |
| `complex_fno.py` | `ComplexReLU`, `ComplexSpectralConv2d`, `ComplexFNOLayer` — complex Fourier Neural Operator layers |
| `bilstm_model.py` | `ComplexMaxPool2d`, `ComplexUpsample`, `ComplexConvBlock` — encoder/decoder blocks for the BiLSTM baseline |

## Baseline / alternative backbones

| File | Contents |
|---|---|
| `velocity_unet.py` | `ComplexConvBlock`, `ComplexDownBlock`, `ComplexToRealBridge` — complex U-Net baseline |
| `pinn_recunet.py` | Recurrent U-Net physics-informed baseline |
| `midpoint_fno.py` | `MidpointConvGRUCell`, `ComplexConvGRUCell`, `ComplexFusionLayer` — FNO + ConvGRU hybrid |
| `temporal_fno.py` | `ComplexConvGRU`, `TemporalPhysicsBlock` — temporal FNO variant |
| `complex_motion_interpolation.py` | `FlowEstimator`, `FusionNet`, `ComplexMotionInterpolation` — optical-flow-style interpolation baseline |
| `improved_motion_interpolation.py` | `MultiScaleFlowEstimator`, `AdaptiveFusion`, `ResidualMotionInterpolation` |
| `input_guided_motion.py` | `HighResFlowEstimator`, `AdaptiveFusion`, `InputGuidedMotionInterpolation` |

## WavePINN family (main line of development, v3 → v11)

Each version is a self-contained model file; later versions are not strict supersets
of earlier ones — see `docs/` for the design rationale behind each jump.

| File | Key classes | Notes |
|---|---|---|
| `wavepinn_v3.py` | `ContinuousTimeEncoding`, `ComplexBiLSTM`, `TemporalCrossAttentionInterpolator` | First cross-attention temporal interpolator |
| `wavepinn_v4.py` | `DtAdaptiveConvGRUCell`, `WaveContentGate`, `BidirectionalWavePropagator` | dt-conditioned ConvGRU propagation |
| `wavepinn_v5.py` | `TemporalSkipRefiner`, `ComplexToRealUNetV5` | Adds temporal skip refinement |
| `wavepinn_v6.py` | `SpatioTemporalPositionalEncoding`, `SpatioTemporalSelfAttention`, `SparseToDenseCrossAttention` | **Backbone used by the production v10b model.** Full spatio-temporal attention bottleneck |
| `wavepinn_v7.py` | `PhysicsHead`, `WavePINNv7` | Adds explicit physics (wave-equation) output head |
| `wavepinn_v9.py` | `ScalarCsHead`, `WavePINNv9` | Adds scalar shear-wave-speed (cs) regression head |
| `wavepinn_v10.py` | `create_model(...)` factory | `WavePINNv10 == WavePINNv6`; this file mainly wraps v6 with the `TrainerV10` training loop (see `training/`) |
| `wavepinn_v11.py` | `BottleneckSpectralAdapter`, `WavePINNv11` | Spectral adapter at the attention bottleneck |

`configs/config_wavepinn_v*.json` map 1:1 to these files by version suffix; see
[`configs/README.md`](../../configs/README.md).

## Usage

Models are constructed from a config dict (as loaded from `configs/*.json`), e.g.:

```python
import json
from src.architectures.wavepinn_v6 import WavePINNv6

with open("configs/config_wavepinn_v10b.json") as f:
    cfg = json.load(f)
model = WavePINNv6(**cfg["model"])
```
