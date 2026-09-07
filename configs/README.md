# `configs/`

JSON training configs, one per model/experiment. Each config has (at least) a
`"model"` block (constructor kwargs for the matching `src/architectures/` class) and a
`"data"` block (dataset path, batch size, augmentation settings). Keys starting with
`_` (e.g. `"_note_500hz_included"`) are comments, stripped before use — see
`strip_comments()` in `src/training/train_wavepinn_v10b.py`.

Load and pass to a trainer directly:

```bash
python -m src.training.train_wavepinn_v10b --config configs/config_wavepinn_v10b.json
```

## WavePINN main line

| Config | Trains with |
|---|---|
| `config_wavepinn_v3.json` … `config_wavepinn_v9.json` | `src/training/train_wavepinn_v{N}.py` |
| `config_wavepinn_v10.json` | `train_wavepinn_v10.py` |
| `config_wavepinn_v10b.json` | `train_wavepinn_v10b.py` — **production model config** |
| `config_wavepinn_v11.json` | `train_wavepinn_v11.py` |
| `config_wavepinn_v12.json` | `train_wavepinn_v12.py` |
| `config_wavepinn_v13.json` | `train_wavepinn_v13.py` |
| `config_wavepinn_v14.json`, `config_wavepinn_v14b.json`, `config_wavepinn_v14c.json`, `config_wavepinn_v14d.json` | `train_wavepinn_v14.py` (variant runs — see `docs/` or each file's `_note_*` comment keys for what differs between them) |

## Baselines / ablations

| Config | Trains with |
|---|---|
| `config_bilstm.json` | BiLSTM baseline (`src/architectures/bilstm_model.py`) |
| `config_complex_fno.json` | Complex FNO baseline |
| `config_curriculum_fno.json` | FNO with curriculum training schedule |
| `config_midpoint_fno.json` | `src/architectures/midpoint_fno.py` |
| `config_temporal_fno.json` | `src/architectures/temporal_fno.py` |
| `config_motion_aware.json` / `config_improved_motion.json` / `config_input_guided.json` | Motion-interpolation baselines |
| `config_pinn.json` | Generic PINN baseline |
| `config_velocity.json` / `config_iq_to_velocity.json` | Direct velocity-regression baselines |
| `config_non_cirs.json` | Non-CIRS phantom data variant |
| `config_multi_prf.json` | Multi-PRF training variant |

## Full-scale velocity dataset variants

| Config | Difference |
|---|---|
| `config_fullscale_velocity.json` | Base full-scale config |
| `config_fullscale_velocity_v2.json` | v2 iteration |
| `config_fullscale_velocity_v2_pinn_ablation.json` | v2 with PINN physics-loss ablation |
| `config_fullscale_velocity_v3_all_freq.json` | v3, trains across all acquisition frequencies |
| `config_fullscale_velocity_20x.json` / `config_fullscale_velocity_25x.json` | Fixed 20×/25× temporal upsampling factor variants |
