# `src/` — Core library

Installable-style Python package for the WaveFormer / WavePINN shear-wave-elastography
(SWE) pipeline: complex-valued neural architectures that reconstruct high-frame-rate
tissue velocity fields from low-PRF ultrasound IQ data, the physics-informed losses
used to train them, and the analysis/evaluation utilities used to score results.

Import as a package from the repo root, e.g.:

```python
from src.architectures.wavepinn_v6 import WavePINNv6
from src.utils.iq_velocity_dataset_v3 import IQVelocityDatasetV3
```

## Subpackages

| Package | Contents |
|---|---|
| [`architectures/`](architectures/README.md) | Model definitions (complex-valued CNN/LSTM/FNO/attention backbones, WavePINN v3–v11) |
| [`training/`](training/README.md) | PyTorch Lightning training entry points, one per model/experiment |
| [`losses/`](losses/README.md) | Physics-informed and supervised loss functions |
| [`evaluation/`](evaluation/README.md) | Training diagnostics and result visualization helpers |
| [`analysis/`](analysis/README.md) | Post-hoc signal analysis (dispersion fitting, MUSE-lab post-processing) |
| [`utils/`](utils/README.md) | Datasets, dataloaders, Loupas velocity estimation, deployment helpers |
| [`tools/`](tools/README.md) | Small standalone visualization utilities |

## Note on the production model

The production model referenced throughout the codebase (and in `configs/`) is
**WavePINN v10b**, which uses the **v6 architecture** (`architectures/wavepinn_v6.py`)
trained with pure pixel loss via `training/train_wavepinn_v10b.py`. Later versions
(v11–v14) are experimental follow-ups tracked in `docs/`.
