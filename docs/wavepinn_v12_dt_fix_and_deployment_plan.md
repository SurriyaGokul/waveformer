# WavePINN v12: dt-bug fix, GT-free deployment scale, and pipeline hygiene

## Context

WavePINN v10b (V6 backbone, `src/architectures/wavepinn_v6.py`, pure pixel loss) is the production model for 500 Hz → 6 kHz temporal super-resolution of SWE velocity. Two user-reported problems plus three discovered issues drive this work:

1. **Deployment normalization**: inference scripts denormalize with `vel_scale = 99th pct of GT velocity` (`scripts/predict_6khz_from_500hz_v4.py:95`) — unavailable in real deployment. Key insight: the 500 Hz "GT" velocity is itself Loupas-computed from the same input IQ, so the scale is recoverable from IQ alone. And the deployment output (cs via normalized-xcorr ToA, `scripts/swe_publication.py`) is **provably invariant to vel_scale** — user confirmed cs/elasticity is the only deployment output.
2. **dt batch-averaging bug (discovered)**: `wavepinn_v6.py:626` does `dt_seconds.mean().item()`. With per-sample PRF augmentation and batch_size=2, mixed-step batches feed a wrong averaged dt into the temporal positional encoding and physics refinement — diluting the dt-conditioning that drives cross-PRF generalization. User chose: fix + retrain.
3. **Train/val leakage (discovered)**: 12 of 
78 500 Hz acquisitions have window `w0` in one split and `w1` in the other (e.g. `3ft_15mmfocus_Plane_1_500Hz_...` w1 in train, w0 in val) → val metrics optimistic. Also no true held-out test set (only train/val/inference, with inference used during development).
4. Verified NON-issues: GT temporal resampling in `iq_velocity_dataset_v3.py:406-420` is correct (only a half-frame timestamp ambiguity — document, don't change). Stale comment at `wavepinn_v6.py:659-661` claims zero-init cross-attn output; actual init is deliberately Kaiming (rationale at lines 178-185) — fix comment, keep Kaiming.

User decisions: improve v10b directly (pure pixel loss, no physics losses — they caused the V9 collapse); full retrains + ablations OK.

## Task 0 — Freeze v10b baseline

Run existing eval scripts with the v10b checkpoint; save outputs under `outputs/baseline_v10b/`:
- `scripts/predict_6khz_from_500hz_v10b.py`, `scripts/swe_publication_v10b.py` (cs MAE ≈ 0.831 m/s), `scripts/evaluate_wavepinn_v10.py`.
Record r_active, MAE (mm/s), cs MAE, per-phantom cs.

## Task 1 — Per-sample dt fix (P1)

**Principle: forward-logic-only, zero state_dict changes** — v10b checkpoint must still load `strict=True`; float-dt path (used by inference scripts) stays bit-identical.

- `wavepinn_v6.py`: add helper `_dt_to_vector(dt_seconds, B, device) -> (B,)` (float → full; 0-dim/1-elem tensor → expand; `(B,)` → passthrough). Use at both call sites (bottleneck `:625`, `WavePINNv6.forward` `:911`).
- `SpatioTemporalPositionalEncoding.forward` (`:94`): accept `t_seconds` `(T,)` or `(B,T)`; batched path returns `(B, T*n_spatial, d)`.
- `SpatioTemporalSelfAttention` (`:207`) and `SparseToDenseCrossAttention` (`:295,333`): handle 2-D or 3-D positional tensors.
- `SpatioTemporalAttentionBottleneck.forward`: `t_sparse = arange(T)[None,:] * dt_vec[:,None]`; pass `dt_vec / upsampling_factor` (tensor `(B,)`) to physics refine. Note in comment: `_linear_interpolate` residual is dt-invariant → unchanged.
- `TemporalCrossAttentionSkip.forward` (`:440-501`): accept `(B,T)` time arrays; token layout is batch-major (`reshape(B*H*W, T, 2C)` at `:467`) so expand pos as `pos.unsqueeze(1).expand(B, H*W, T, d).reshape(-1, T, d)`.
- `wavepinn_v3.py` `PhysicsRefinementBlock.forward` (`:409`): accept float | `(B,)` tensor; tensor path `log_dt = log10(dt.clamp_min(1e-12)).reshape(-1,1)`; FiLM broadcast `(B_or_1,1,C2,1,1)`. Float path bit-identical for v3/v4/v5 callers.
- Fix stale comment at `:659-661` (Kaiming, not zero-init).

**New test `scripts/test_dt_per_sample.py`** (follows `scripts/test_wavepinn_v3.py` convention):
1. Batch-equivalence: eval-mode model, B=2 with `dt=[0.002, 0.024]` ≈ concat of two B=1 runs (atol 1e-5) — proves the fix.
2. Backward compat: float dt vs `tensor([dt])` identical.
3. v10b checkpoint loads `strict=True`.

## Task 2 — GT-free deployment denormalization (P2)

- **2a. `src/utils/loupas.py` (new)**: move `LoupasConfig`, `determine_roi`, `loupas_2d_displacement`, `smooth_displacement`, `compute_velocity` verbatim out of `scripts/iq_to_velocity.py`; that script imports from the new module (CLI unchanged). Point `scripts/recompute_loupas_targets.py` at it too if it duplicates the kernel.
- **2b. `src/utils/deployment_scale.py` (new)**: `estimate_velocity_scale_from_iq(iq_window (H,W,T) raw pre-clutter-filter, frame_time_s, config=None, calibration=1.0)` → Loupas → medfilt(12×6) → velocity → `max(percentile(|vel|, 99), 1.0)`. Training-time `vel_scale` was the 99th pct over the **full 50-frame window** (`iq_velocity_dataset.py:267-270`), so prefer running on the full available 500 Hz window (deployment records the full sequence; only the model input is 5 frames) → reproduces training normalization essentially exactly. Support T=5 worst case with optional `calibration` scalar.
- **2c.** Add `--scale-mode {loupas,gt}` (default `loupas`) to `scripts/predict_6khz_from_500hz_v10b.py` (`infer_one` `:92-119`), `scripts/predict_6khz_from_500hz_v4.py` (`:91-112`), `scripts/visualize_500hz_predictions.py` (`:146-173`). Record both scales in `summary.json`.
- **2d. `scripts/validate_deployment_scale.py` (new)**: over 500 Hz val/test samples, compare proxy (T=5 and full-window) vs GT scale (ratio median/IQR/extremes); run ToA cs on predictions denormalized both ways and assert cs identical. Acceptance: full-window ratio ≈ 1.0 ± 5%; cs equal across modes.

## Task 3 — v12 retrain + ablation (P3)

- **v12-core (run A)** = v6 backbone + Task 1 fix, recipe identical to v10b (pure pixel loss, 500 Hz in training with step override, early stop on `val_500hz_l1`, 200 epochs, lr 2e-4). New `configs/config_wavepinn_v12.json` (copy of v10b + `use_shallow_skip_refine: false`); new `src/training/train_wavepinn_v12.py` thin wrapper reusing `TrainerV10` (mirror `train_wavepinn_v10b.py`, ~30 lines). **Trains on the regrouped leakage-free split from Task 4 — Task 4.1 must land first.**
- **Ablation B (v12-skips)**: `ShallowSkipTemporalRefine` in `wavepinn_v6.py` — after `_interpolate_skip_temporal` on skip1/2, residual temporal conv: reshape `(B·H·W, 2C, T_dense)`, depthwise Conv1d(k=3)+GELU+pointwise Conv1d(k=1), `z + 0.5*refine(z)`, standard init (respect the codebase's no-zero-init lesson). Gated by config flag → checkpoint-compatible when off. (Full cross-attention on skip1 is ~0.5 GB activations — too heavy.)
- **Deferred (recommend NOT now)**: bottleneck at 16×20 tokens (encoder/decoder surgery, invalidates checkpoints → v13); more input frames (T_sparse=5 baked into data pipeline); half-frame GT timestamp shift (low value).
- **Schedule**: run A → run B → re-evaluate existing v10b on the new test split (inference only, re-bases the 0.831 baseline fairly) → compare on 500 Hz test L1, r_active/MAE, cs MAE vs 5 kHz GT (`evaluate_wavepinn_v10.py`), goat-heart eval. Ship A unless B beats it on **both** 500 Hz L1 and cs MAE. Drop B if GPU budget is tight.

## Task 4 — Pipeline hygiene (P4)

1. **`scripts/regroup_splits.py` (new)**: group 500 Hz samples by acquisition base name (strip `_w\d+$`); move whole acquisitions to repair the 12 straddling train/val; carve frozen `test` split (~15 of 78 acquisitions ≈ 20%, stratified by phantom source/stiffness, fixed seed); write `"test"` key to `metadata.json` (back up `.bak`), copy npy files into `data/fullscale_velocity/test/`. Supporting edits: `IQVelocityDatasetV3._load_sample_metadata` (`iq_velocity_dataset_v3.py:209`) add `"test"` to split tuple; eval scripts get `--split test`. Decision (moderate strictness): 5 kHz acquisitions of test phantoms remain in training (they're analysis GT, not prediction targets) — documented; goat heart stays the fully-external check.
2. Documenting comment at `iq_velocity_dataset_v3.py:406-420` (resampling convention is correct; note half-frame ambiguity).

## Ordering

Task 0 → Task 1 (+tests) → Task 4.1 (before any retrain) → Task 2 (parallel-safe) → Task 3 retrains → verification. Only Task 3 needs GPU (2 × ~200-epoch runs).

## Verification

- `python scripts/test_dt_per_sample.py` passes (batch-equivalence, float parity, strict checkpoint load) **before any retrain**.
- v10b checkpoint through fixed forward on 500 Hz val reproduces baseline exactly (B=1 ⇒ bug was inert at inference).
- `python scripts/validate_deployment_scale.py`: full-window proxy/GT ratio ≈ 1 ± 5%; cs identical across scale modes.
- Post-retrain vs re-based v10b on frozen test split: cs MAE ≤ baseline (hard gate +0.05 m/s), r_active ≥ baseline − 0.02, 500 Hz L1 ≤ baseline, goat-heart cs-MAE ≤ 2.37 m/s. Training uses v10b's G1–G4 gates.

## Risk flags

(a) Re-splitting changes val statistics — mitigated by re-evaluating v10b on the same test split. (b) T=5 Loupas proxy has only ~2 informative frames — prefer full-window mode; cs is scale-invariant regardless. (c) Ablation B is the only speculative item.
