# `src/training/`

Training entry points. Each `train_*.py` file wires one model (from
`src/architectures/`), one loss (from `src/losses/`), and a dataset (from
`src/utils/`) into a training loop, driven by a JSON config from `configs/`.

## WavePINN main line

| File | Trains | Config |
|---|---|---|
| `train_wavepinn_v3.py` … `train_wavepinn_v9.py` | Corresponding `wavepinn_v*.py` architecture | `configs/config_wavepinn_v*.json` |
| `train_wavepinn_v10.py` | Defines `TrainerV10`, the shared Lightning trainer used by v10/v10b/v12/v13 | `configs/config_wavepinn_v10.json` |
| `train_wavepinn_v10b.py` | **Production model.** Thin wrapper around `TrainerV10` for the v6 backbone, pure pixel loss | `configs/config_wavepinn_v10b.json` |
| `train_wavepinn_v11.py` | `WavePINNv11` (spectral adapter) | `configs/config_wavepinn_v11.json` |
| `train_wavepinn_v12.py` | v12 (per-sample dt fix over v10b, see `docs/`) | `configs/config_wavepinn_v12.json` |
| `train_wavepinn_v13.py` / `train_wavepinn_v14.py` | Later iterations | `configs/config_wavepinn_v13.json` / `config_wavepinn_v14*.json` |

## Baselines / ablations

| File | Trains |
|---|---|
| `train_pinn.py` | Generic physics-informed baseline |
| `train_motion_aware.py` / `train_improved_motion.py` | Motion-interpolation baselines |
| `train_input_guided.py` | `InputGuidedMotionInterpolation` |
| `train_iq_to_velocity.py` | Direct IQ → velocity regression baseline |
| `train_velocity_preprocessed.py` | Trains on pre-computed (Loupas) velocity fields rather than raw IQ |
| `velocity_training.py` | Shared training utilities imported by the velocity-baseline scripts above |

## Usage

Each script takes a `--config` path (see the file's `argparse` block, or run with
`--help`) and expects the corresponding `data/` directory (see
[`../../data` note in the root README](../../README.md#data--model-weights-not-in-this-repo)
— not included in this repository) to already be preprocessed by the matching
`scripts/preprocess_*.py` script. Example:

```bash
python -m src.training.train_wavepinn_v10b --config configs/config_wavepinn_v10b.json
```
