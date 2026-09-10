# WaveFormer — Physics-Informed Temporal Super-Resolution for Shear-Wave Elastography

Deep-learning pipeline that reconstructs high-frame-rate (up to ~10–20 kHz) tissue
velocity fields from low-pulse-repetition-frequency (as low as 500 Hz) ultrasound IQ
acquisitions, for shear-wave elastography (SWE). The reconstructed velocity fields are
used downstream to estimate shear-wave speed (cs) and viscoelastic tissue properties
(shear storage/loss modulus G′/η) via dispersion analysis.

The core model family is **WavePINN** — complex-valued spatio-temporal attention
networks trained on paired (sparse IQ, dense Loupas-estimated velocity) samples from
CIRS and PVA tissue-mimicking phantom and ex-vivo (goat heart) acquisitions. The
production model is **WavePINN v10b** (`src/architectures/wavepinn_v6.py` backbone).

> **This repository is code-only.** Raw/preprocessed data, trained checkpoints, and
> training run outputs (~60 GB total) are not committed — see [`DATA.md`](DATA.md) for
> what exists locally and how to regenerate it.

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/README.md) | Core library: model architectures, training loops, losses, evaluation & analysis utilities |
| [`scripts/`](scripts/README.md) | 98 standalone CLI scripts: preprocessing, prediction, evaluation, diagnosis, SWE/dispersion analysis, visualization |
| [`configs/`](configs/README.md) | JSON training configs, one per model/experiment |
| [`matlab/`](matlab/README.md) | Reference MATLAB script for the pre-Python (Verasonics) acquisition/velocity pipeline |
| [`reports/`](reports/README.md) | Detailed write-ups: dispersion post-processing pipeline, elastography audit |
| [`docs/`](docs/README.md) | Design/task planning docs (the "why" behind version jumps) |
| [`DATA.md`](DATA.md) | What data/weights exist locally, their sizes, and how to regenerate them |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
```

Developed against Python 3.12, PyTorch 2.8 (CUDA 12.8), PyTorch Lightning 2.6.

## Typical workflow

1. **Preprocess** a raw acquisition set into `data/` — see the matching
   `scripts/preprocess_*.py` (e.g. `preprocess_1ft.py`, `preprocess_goat_heart.py`).
2. **Train** a model — e.g. the production model:
   ```bash
   python -m src.training.train_wavepinn_v10b --config configs/config_wavepinn_v10b.json
   ```
3. **Predict** on held-out/deployment IQ — e.g.
   ```bash
   python scripts/predict_6khz_from_500hz_v10b.py --model_path <checkpoint> --output_dir outputs/my_run
   ```
4. **Evaluate / analyze** — `scripts/evaluate_wavepinn_v10.py` for deployment metrics
   (r_active, MAE, cs MAE vs GT), `scripts/swe_publication_v10b.py` for the full
   SWE (shear-wave-speed) publication figure/table set.

Every script accepts `--help` for its exact arguments and defaults.
