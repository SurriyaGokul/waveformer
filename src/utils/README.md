# `src/utils/`

Datasets, dataloaders, and the classical (non-learned) Loupas velocity estimator used
both as a baseline and to generate training targets / deployment normalization.

| File | Key contents |
|---|---|
| `loupas.py` | `LoupasConfig`, `determine_roi`, `loupas_2d_displacement`, `smooth_displacement`, `compute_velocity` — the classical 2-D autocorrelation (Loupas) tissue velocity estimator, the source of "ground truth" velocity used for training targets |
| `deployment_scale.py` | `estimate_velocity_scale_from_iq(...)` — recovers the velocity normalization scale from raw IQ alone (no GT needed), for real deployment where GT isn't available |
| `deployment_input.py` | `resolve_raw`, `build_sparse_iq` — builds model input tensors from raw acquisition IQ at deployment time |
| `iq_velocity_dataset_v3.py` | `IQVelocityDatasetV3`, `collate_fn_v3`, `create_iq_velocity_dataloaders_v3` — **current dataset used by v10b/v12+ training**, loads sparse IQ + dense Loupas-velocity pairs with PRF augmentation |
| `iq_velocity_dataset.py` | Earlier version of the IQ→velocity dataset (PRF parsing/normalization helpers still reused: `parse_prf_from_name`, `normalize_prf`) |
| `velocity_dataset.py` | `VelocityEstimationDataset`, `create_velocity_dataset_from_dirs/_lists` — dataset for the velocity-only (non-IQ) baselines |
| `data_loader.py` | `DataConfig`, `ComplexNormalizer`, `check_data_quality` — shared normalization/QC helpers |
| `loss_functions.py` | `InterferometricPhaseLoss`, `KasaiAutocorrelationLoss`, `VelocityLoss` — earlier/alternative loss implementations (superseded by `src/losses/` for current models) |

## Data layout these expect

The datasets above expect a preprocessed directory (produced by
`scripts/preprocess_*.py`) with `train/`, `val/`, `test/`, `inference/` subfolders and
a `metadata.json` describing each sample — see the root
[README's data section](../../README.md#data--model-weights-not-in-this-repo).
This raw/preprocessed data is not included in this repository.
