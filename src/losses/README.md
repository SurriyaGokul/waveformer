# `src/losses/`

Loss functions used by the trainers in `src/training/`. Versioned in step with the
WavePINN architecture versions — later files generally supersede earlier ones for
new training runs, but old versions are kept so old configs still reproduce.

| File | Key classes | Notes |
|---|---|---|
| `physics_loss.py` | `WaveEquationRegularizer` | Base wave-equation PDE residual regularizer, plus SSIM helpers |
| `physics_loss_v3.py` | `PhysicsAwareWaveRegularizer`, `EnhancedVelocityLossV3` | |
| `physics_loss_v5.py` | `WaveCoherenceLoss`, `EnhancedVelocityLossV5` | |
| `physics_loss_v6.py` | `EnhancedVelocityLossV6` | |
| `swe_loss_v7.py` | `DiffToaCsLoss`, `SpatialSparsityLoss`, `TemporalEnvelopeLoss` | Adds a differentiable time-of-arrival shear-wave-speed loss |
| `swe_loss_v9.py` | `GradientDifferenceLoss`, `TemporalGradientLoss`, `EnhancedVelocityLossV9` | |
| `velocity_loss_v10.py` | `GradientDifferenceLoss`, `TemporalGradientLoss`, `VelocityLossV10` | **Loss used by the production v10b model** (pure pixel/gradient loss — no PDE physics term; see `docs/` for why physics losses were dropped) |
| `velocity_loss_v14.py` | `MaskedTemporalGradientLoss`, `TemporalSpectralLoss` | Adds a masked spectral-domain term for later experiments |

> **Note:** physics-informed (PDE) loss terms caused a training collapse in the v9
> line of experiments. The production model (v10b) trains with pure pixel/gradient
> loss only (`velocity_loss_v10.py`) — see `docs/` for details before re-enabling a
> physics term.
