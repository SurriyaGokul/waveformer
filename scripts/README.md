# `scripts/`

Standalone, directly-runnable scripts (98 total) — as opposed to `src/`, which is the
importable library. Every script uses `argparse` with sensible defaults, so
`python scripts/<name>.py --help` always shows the exact options and defaults. All
scripts are meant to be run from the **repository root** (they reference `data/`,
`saved_models/`, `outputs/` by relative path).

Naming convention: a bare name (e.g. `swe_publication.py`) is the current/baseline
version; a `_v2`, `_v10b`, `_25x` suffix etc. marks a specific model version or
upsampling factor the script targets — check the docstring at the top of the file for
which model/config it expects.

## Preprocessing — raw acquisitions → training-ready `data/`

| Script | Purpose |
|---|---|
| `preprocess_1ft.py` | Preprocess the "1FT" phantom acquisition set into `data/fullscale_velocity/` (train/val split) |
| `preprocess_goat_heart.py` | Preprocess goat-heart acquisitions into `data/goat_heart_velocity/` |
| `preprocess_non_cirs.py` | Preprocess non-CIRS phantom data |
| `preprocess_ground_truth.py` | Compute Loupas ground-truth velocity fields for a raw acquisition set |
| `preprocess_higher_upsampling.py` | Build data variants for higher temporal-upsampling-factor experiments |
| `preprocess_velocity_data.py` | General velocity-dataset preprocessing entry point |
| `prepare_500hz_training_data.py` | Build the 500 Hz-inclusive training set used by v10b+ |
| `convert_mat_to_npy.py` | Convert MATLAB `.mat` acquisition files (see `matlab/`) to `.npy` |
| `recompute_loupas_targets.py` | Regenerate Loupas velocity targets (e.g. after a config change) with the shared `src/utils/loupas.py` kernel |
| `regroup_splits.py` | Rebuild train/val/test splits grouped by acquisition (fixes train/val leakage from window-level splitting) |
| `clean_empty_frames.py` / `clean_fullscale_velocity.py` | Remove degenerate/empty samples from a preprocessed dataset |

## Prediction — run a trained model on new/held-out IQ data

All `predict_<target-rate>_from_<source-rate>[_<Nx>].py` scripts load a checkpoint and
run temporal super-resolution inference (e.g. `predict_6khz_from_500hz_v10b.py` loads
a v10b checkpoint and upsamples 500 Hz IQ to 6 kHz velocity). The suffix identifies the
model version (`_v3`, `_v4`, `_v5`, `_v10b`, ...) or explicit upsampling factor
(`_20x`, `_25x`).

<details><summary>Full list (16 scripts)</summary>

`predict_10khz_from_500hz_20x.py`, `predict_1200hz_from_100hz.py`,
`predict_1200hz_from_100hz_v3.py`, `predict_1200hz_from_100hz_v4.py`,
`predict_12500hz_from_500hz_25x.py`, `predict_2500hz_from_100hz_25x.py`,
`predict_2khz_from_100hz_20x.py`, `predict_3khz_from_250hz.py`,
`predict_5khz_from_250hz_20x.py`, `predict_6250hz_from_250hz_25x.py`,
`predict_6khz_from_500hz.py`, `predict_6khz_from_500hz_v3.py`,
`predict_6khz_from_500hz_v4.py`, `predict_6khz_from_500hz_v5.py`,
`predict_6khz_from_500hz_v10b.py`, `predict_from_consecutive_500hz.py`

</details>

Common flags (see `predict_6khz_from_500hz_v4.py` for the canonical set):
`--model_path`, `--output_dir`, `--start_frame`, `--sample_name`, `--fps`, `--no_video`.

Related: `iq_to_velocity.py` (classical Loupas inference, no model), `inference_500hz_1khz.py`.

## Evaluation — score a checkpoint / pipeline end-to-end

| Script | Purpose |
|---|---|
| `evaluate_wavepinn_v9.py` / `evaluate_wavepinn_v10.py` | Full deployment evaluation (r_active, MAE, cs MAE vs GT) for the named model version |
| `evaluate_v10_swe_report.py` | Generates the SWE (shear-wave-speed) report for v10-line models |
| `evaluate_goat_heart_v10b.py` | External validation on goat-heart data |
| `evaluate_matched_gt.py` | Evaluation against a specific matched-GT split |
| `evaluate_mid_prf.py` | Evaluation at intermediate PRF (pulse repetition frequency) settings |
| `evaluate_pva_v10b_v12.py` / `evaluate_pva_simulated_rates.py` | PVA-phantom evaluation, incl. simulated-acquisition-rate variants |
| `evaluate_swe_v14.py` | v14-line evaluation |
| `eval_500hz_deployment.py` | Deployment-config evaluation at the native 500 Hz rate |
| `compute_interpolation_baselines.py` | Classical (non-learned) interpolation baselines for comparison |
| `compare_model_maps.py` | Side-by-side comparison of velocity maps from two models/checkpoints |
| `benchmark_inference.py` / `benchmark_v2_inference.py` | Inference-speed benchmarking |
| `generate_test_predictions.py` | Batch-generate predictions over a full test split for downstream analysis |

## Diagnosis — debugging a specific model/training issue

| Script | Purpose |
|---|---|
| `diagnose_v6_bias_variance.py` | Bias/variance decomposition of v6-backbone predictions |
| `diagnose_v6_phase_coherence.py` | Phase-coherence analysis (see project memory: band coherence, not power ratio, is the valid HF metric) |
| `diagnose_v6_plateau.py` | Investigates training-loss plateaus |
| `diagnose_v6_sparsity.py` | Investigates sparse-input handling issues |
| `diagnose_1ft_gap.py` | Investigates train/inference gap on the 1FT dataset |
| `diagnose_wave_dwell_time.py` | Wave dwell-time analysis in the ROI |
| `diagnose_video_vs_abstract.py` | Reconciles video-rendered results against reported/abstract numbers |
| `analyze_v8_vs_v2.py` / `analyze_v9_vs_v2.py` | Head-to-head comparison against the v2 baseline |

## SWE / dispersion analysis — velocity field → shear-wave-speed & viscoelastic parameters

| Script | Purpose |
|---|---|
| `swe_analysis.py`, `swe_analysis_v4.py`, `swe_analysis_20x.py`, `swe_analysis_25x.py` | Time-of-arrival shear-wave-speed (cs) analysis, per model version / upsampling factor |
| `swe_viscoelastic.py` / `swe_viscoelastic_v10b.py` | Kelvin-Voigt (G′, η) fitting from dispersion — **see project memory: pre-2026-08-24 numbers from this path are a known artifact; treat older outputs with caution** |
| `swe_publication.py` / `swe_publication_v10b.py` | Generates the full publication-ready SWE figure/table set |
| `validate_dispersion_gt.py` | Sanity-checks dispersion fitting against known/synthetic ground truth |
| `postprocess_muselab_report.py` | Post-processes MUSE-lab-format reports through `src/analysis/muselab_postproc.py` |
| `saus2026_units.py` | Unit conversions for the SAUS 2026 abstract/report numbers |

## Visualization

| Script | Purpose |
|---|---|
| `visualize_velocity_maps.py` / `visualize_velocity_predictions.py` / `visualize_velocity_video.py` | Render predicted velocity fields as maps/videos |
| `visualize_500hz_predictions.py` / `visualize_fullscale_predictions.py` / `visualize_non_cirs_predictions.py` | Render predictions for specific data subsets |
| `visualize_matched_gt_comparison_video.py` / `plot_3ft_reconstruction_comparison.py` / `visualize_v2_vs_v7.py` | Side-by-side prediction-vs-GT comparison videos/plots |
| `visualize_loupas_targets.py` / `visualize_training_targets.py` / `visualize_smooth_targets.py` | Inspect training-target generation |
| `visualize_preprocessed_data.py` | Sanity-check a preprocessed dataset before training |
| `visualize_magnitude_weighted_loss.py` | Visualize the spatial weighting used by magnitude-weighted loss terms |
| `visualize_v10_swe_video.py` | Render the v10-line SWE (cs map) video |
| `generate_demo_assets.py` | Generate demo images/videos for presentations |
| `generate_poster_wave_montage_500hz_vs_10khz.py` | Poster-figure montage comparing 500 Hz input vs 10 kHz output |
| `generate_figure3_comparison.py` / `generate_mae_ssim_figure.py` / `generate_final_tables_and_maps.py` | Publication figure/table generation |
| `generate_saus2026_figure.py` / `generate_saus2026_barpanel.py` / `generate_saus2026_stats.py` / `generate_saus2026_values.py` / `summarize_saus2026_values.py` | Figures/stats specifically for the SAUS 2026 abstract |
| `generate_smooth_targets.py` | Generate temporally-smoothed training targets |

## Tests

Lightweight correctness tests, run directly with `python scripts/<name>.py` (not
pytest-based):

| Script | Checks |
|---|---|
| `test_dt_per_sample.py` | Per-sample dt batching fix: batch-equivalence, float/tensor dt parity, strict checkpoint load (see `docs/`) |
| `test_dispersion_synthetic.py` | Dispersion fitting recovers known G′/η on synthetic data |
| `test_muselab_postproc_synthetic.py` | MUSE-lab post-processing on synthetic input |
| `test_wavepinn_v3.py` / `test_wavepinn_v4.py` | Forward-pass shape/gradient sanity checks for those architectures |
| `validate_deployment_scale.py` | Confirms the GT-free deployment velocity-scale proxy matches GT-based scaling and yields identical cs |

## Misc

| Script | Purpose |
|---|---|
| `train_baseline_velocity.py` | Standalone training entry point for the non-IQ velocity baseline (does not go through `src/training/`) |
