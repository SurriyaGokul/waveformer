# Data & model weights (not in this repository)

This repository is **code-only** by design — the raw and preprocessed datasets and
trained checkpoints together total roughly 60 GB, well beyond what's practical to
store in a Git/GitHub repository (GitHub's free Git LFS quota is 1 GB storage /
1 GB bandwidth per month). See the repo README for the decision.

Git LFS *is* configured in this repo (`.gitattributes` tracks `*.pt`, `*.npy`, `*.mat`,
etc.), so if you want to add any of the artifacts below later, `git lfs track` is
already set up — just be aware of GitHub's LFS storage/bandwidth pricing before pushing
large files.

## What lives where locally (not committed)

| Path | Size (approx.) | Contents |
|---|---|---|
| `data/fullscale_velocity/` | 2.5 GB | Preprocessed train/val/test IQ + Loupas-velocity pairs, main training set |
| `data/PVA_Phantom/` | 4.7 GB | Preprocessed PVA phantom velocity data |
| `data/pva_velocity/`, `data/goat_heart_velocity/` | 166 MB, 5.4 MB | Preprocessed PVA / goat-heart velocity sets |
| `1FT/` | 16 GB | Raw "1FT" phantom acquisitions (source for `scripts/preprocess_1ft.py`) |
| `500hz_raw_transfer/` | 14 GB | Raw 500 Hz acquisition transfers |
| `PVA_Phantom/` (repo root) | 4.7 GB | Raw PVA phantom acquisitions |
| `Goat Heart Data/` | 174 MB | Raw goat-heart acquisitions |
| `outputs/` | 20 GB | Training run outputs: checkpoints, tensorboard logs, evaluation figures/videos. Fully regenerable by re-running `src/training/train_*.py` / `scripts/evaluate_*.py` |
| `saved_models/` | 1.1 GB | Best checkpoints per WavePINN version (`wavepinn_v12_best.pt` … `wavepinn_v14d_best.pt`) |
| `Final_Publication_Model_Weights.pt` | 23 MB | Final publication checkpoint |
| `SWE_PINN_outputs_wavepinn_v10b_best_model.pt` | 187 MB | Production v10b best checkpoint |

## Regenerating from scratch

1. Preprocess raw acquisitions with the matching `scripts/preprocess_*.py` script (see
   [`scripts/README.md`](scripts/README.md)) to produce a `data/` layout matching what
   `src/utils/iq_velocity_dataset_v3.py` expects.
2. Train with `python -m src.training.train_wavepinn_v10b --config configs/config_wavepinn_v10b.json`
   (or the relevant `train_*.py`/config pair — see [`src/training/README.md`](src/training/README.md)).
3. Evaluate/analyze with the corresponding `scripts/evaluate_*.py` / `scripts/swe_*.py`.

If you'd rather keep weights (and/or a curated slice of data) versioned in this repo
after all, say so and they can be added via the Git LFS config that's already in place
— just confirm you're OK with a GitHub LFS data pack if the total exceeds 1 GB.
