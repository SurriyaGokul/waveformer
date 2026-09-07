# SWE Dispersion Post-Processing Pipeline (pre-MUSE Lab)

**Module:** `src/analysis/dispersion.py` (config `version = "v1"`)
**Validated by:** `scripts/validate_dispersion_gt.py`, `scripts/test_dispersion_synthetic.py`
**Used in production evaluation by:** `scripts/evaluate_swe_v14.py`, `scripts/swe_parameters()`
**Superseded by:** the MUSE Lab MATLAB port (`src/analysis/muselab_postproc.py`, 2026-08-29). This document covers the estimator that ran immediately before that switch — the one we built in-house to fix the broken original Voigt estimator.

This is the pipeline that turns a raw particle-velocity movie `(T, H, W)` into Kelvin-Voigt viscoelastic parameters (G′, η, E). Every number below (bins, thresholds, tolerances) is read directly from the frozen `DispersionConfig` in the source, not reconstructed from memory.

---

## 1. Why this pipeline exists

The original estimator (`scripts/evaluate_wavepinn_v10.py:estimate_voigt`) had a wavenumber unit bug: `np.fft.fftfreq` returns cycles/m, but phase velocity was computed as `omega / kx` with `omega` in rad/s and `kx` left in cycles/m instead of rad/m. That silently inflated every phase velocity by a factor of **2π**.

Consequences, measured directly:
- The inflated speeds collided with the `cs < 12 m/s` admissibility gate → **63% of 5 kHz ground-truth acquisitions returned NaN**.
- The survivors were fit from only 3–4 grossly-inflated dispersion points, producing viscosities of **20–37 Pa·s** where the gel phantoms actually sit near **1–2 Pa·s**.
- Every pre-2026-08-24 G′/η number in the repo is an artifact of this bug (see memory note `swe-old-voigt-estimator-was-broken`).

`src/analysis/dispersion.py` is the rebuild, written and frozen against ground-truth data only, before it was ever pointed at model output.

---

## 2. Acquisition geometry constants

```python
LATERAL_M_PX = 0.1013e-3   # lateral pixel pitch [m]  (≈101.3 µm/px)
RHO          = 1000.0      # tissue density [kg/m^3]
```

The ROI used throughout is **~16.2 mm wide** (measured aperture, see §4) — this single number is the binding constraint on the whole pipeline (see §9).

---

## 3. Pipeline stages, in order

```
depth-average
  -> push localisation
  -> one-directional (outward) segments
  -> detrend
  -> Tukey window (t, x)
  -> zero-pad
  -> 2-D FFT
  -> single quadrant (propagation direction)
  -> per-frequency wavenumber peak (SNR-gated, thresholded centroid)
  -> SNR / resolution / band-selection gates
  -> ToA-anchored, amplitude-weighted Kelvin-Voigt fit
```

A second, independent cross-check estimator (`estimate_dispersion_phase_gradient`) shares the same preprocessing but extracts the wavenumber by regressing unwrapped phase against lateral position instead of picking a 2-D spectral peak. Agreement between the two is evidence the dispersion curve is real, not a peak-picker artifact.

### 3.1 Depth averaging

```python
depth_frac = (0.25, 0.75)
```
`(T, H, W)` → `(T, W)` by averaging rows in the central 50% of the imaging depth (25%–75%). Collapses to a single lateral line per frame.

### 3.2 Push localisation + outward segmentation

- Push column = lateral index of peak `|velocity|` energy in the earliest frames (`n_early = max(3, 10% of T)`).
- Exclusion zone around the push: **`push_excl_px = 15` px** on each side (removes near-field push artifacts).
- Minimum usable segment width: **`min_segment_px = 30` px**.
- The left segment is mirrored (`[:, ::-1]`) so both segments read as propagating toward +x — this makes single-quadrant FFT selection an exact directional filter.
- **`segment_mode = "largest"`**: only the wider of the two outward segments is kept. Rationale (from the source comment): pooling a wide and a narrow segment averages a well-conditioned estimate with an aperture-starved one, so the pipeline deliberately throws away the shorter side rather than diluting the wavenumber resolution.

### 3.3 Windowing and zero-padding

- Detrend: subtract each column's temporal mean (removes DC/clutter residue).
- Separable **Tukey window**, `tukey_alpha = 0.25`, applied in both `t` and `x`.
- Zero-pad before the FFT:
  - `pad_factor_t = 4`, floored at `min_pad_t = 256` → `nt = max(next_pow2(4·T), 256)`
  - `pad_factor_x = 4` → `nx = next_pow2(4·W_segment)`
  - Padding only refines the interpolation grid; it adds no information. The estimator separately tracks how many *native*-resolution bins (`df_native_hz = 1/(T·dt)`) the retained band actually spans (`n_native_bins`), and gates on that, not on the padded bin count.

### 3.4 2-D FFT and quadrant selection

- `V = fft2(segment, s=(nt, nx))`, power `P = |V|²`.
- Temporal frequency axis: `f = fftfreq(nt, dt)` [Hz], restricted to `f > 0`.
- Wavenumber axis: **`k = 2π · fftfreq(nx, dx)`** [rad/m] — this `2π` is literally the line that fixes the original bug (`fftfreq` alone returns cycles/m).
- Both the `+k` and mirrored `-k` quadrants are computed; whichever actually carries the energy is kept (that's the true outward-propagating quadrant — sign convention isn't assumed, it's measured).

---

## 4. Admissible phase-speed window (ToA-narrowed search)

Two global bounds:
```python
c_min = 0.2 m/s
c_max = 15.0 m/s
```

But the effective per-acquisition search window is narrowed around the time-of-arrival (ToA) speed when available:
```python
use_toa_window   = True
toa_c_lo_factor  = 1.8   # c >= cs_toa / 1.8
toa_c_hi_factor  = 2.5   # c <= cs_toa * 2.5
```
Why: the ToA speed uses the *full* aperture with sub-frame lag interpolation, so it's far better conditioned than any single wavenumber bin. Without this narrowing, low-frequency bins latch onto the aperture's main lobe (a leakage artifact), which reads as huge apparent dispersion and drives G′ to its floor.

---

## 5. Per-frequency wavenumber peak extraction

For each frequency bin in the search band `f_min_hz = 75.0` to `f_hi_limit = min(f_max_abs_hz, f_max_nyq_frac · f_Nyquist)` (`f_max_abs_hz = 1500.0`, `f_max_nyq_frac = 0.45`):

1. **Aperture resolution floor**: a segment of length `L` cannot resolve wavenumbers below `k_resolvable = min_aperture_cycles · 2π / L`, with **`min_aperture_cycles = 0.8`**. Below this, the peak picker returns the aperture's main lobe instead of the real wave — verified on synthetic Kelvin-Voigt data (`test_dispersion_synthetic.py`).
2. Intersect the physically-admissible wavenumber range (`k_lo = max(ω/c_hi, k_resolvable)`, `k_hi = ω/c_lo`) with the row of the power spectrum; skip if fewer than 3 admissible bins.
3. **Noise floor** = median of the *entire* quadrant row (not just the admissible window, to avoid the peak dominating its own noise estimate).
4. **SNR gate**: peak power must exceed `snr_min_db = 10.0 dB` above the noise floor, i.e. peak ≥ 10^(10/10) × floor = 10× the median row power.
5. **Thresholded centroid refinement**: starting from the peak bin, expand outward while neighboring bins stay ≥ `centroid_frac = 0.5` (50%) of peak power, capped at `centroid_max_bins = 5` bins each direction. The wavenumber estimate for that frequency is the power-weighted centroid over this window, not the raw argmax — this reduces bin-quantization noise.
6. Reject if the centroid wavenumber falls back below `k_resolvable`, or if the resulting phase speed `c = ω/k` falls outside `[c_lo, c_hi]`.
7. Surviving `(freq, c_phase, weight=√peak_power, band_amplitude)` tuples are pooled across both outward segments (weighted average per frequency).

---

## 6. Band selection

After pooling, the dispersion curve is trimmed to its dominant contiguous run:

```python
band_db = 20.0            # keep bins within 20 dB of peak amplitude
min_native_bins = 6        # reject if fewer than 6 *native*-resolution bins survive
min_band_ratio = 3.0       # reject if f_hi / f_lo < 3.0 (no ToA anchor)
min_band_ratio_anchored = 1.6   # relaxed threshold when ToA anchor is used
```

`_select_band` keeps only the **largest contiguous run** of frequency bins whose pooled amplitude is within 20 dB of the run's peak amplitude — a single largest island, not just any bins above threshold.

Failure reasons emitted here (`FAILURE_REASONS` dict) are the same machine-readable codes reported in every validation run: `empty_input`, `no_segment`, `no_bins`, `few_native_bins`, `narrow_band`, `fit_failed`, `fit_at_bound`, `no_overlap`.

---

## 7. Time-of-arrival (ToA) anchor

`estimate_dispersion` optionally injects a low-frequency anchor point derived from the ToA cross-correlation speed:

```python
toa_anchor = True
group_phase_ratio = 1.125     # calibrated on synthetic KV waves
```

Rationale: the wavenumber-resolvable band sits *above* the elastic regime, where `c(ω) → sqrt(2ωη/ρ)` becomes asymptotically independent of G′. So the measured curve alone constrains η well but G′ poorly. The ToA anchor supplies the missing low-frequency constraint.

- Anchor frequency = energy-weighted spectral centroid of the segment in the search band (`_spectral_centroid_hz`) — this is the frequency the ToA cross-correlation is effectively sensitive to.
- Anchor phase speed = `cs_toa / group_phase_ratio` = `cs_toa / 1.125`. This group-to-phase conversion factor was **calibrated on synthetic Kelvin-Voigt waves** (`test_dispersion_synthetic.py`) and measured at **1.125 ± 0.062** over G′ 1–10 kPa, η 0.5–4 Pa·s.
- The anchor is given weight equal to the *sum* of all measured-curve weights, so it has equal total leverage: the anchor fixes the absolute level, the measured curve fixes the slope.

### 7.1 Time-of-arrival speed itself (`estimate_cs_toa`)

Independent of the FFT path — pairwise cross-correlation of two lateral traces:
```python
depth_frac (ToA)  = (0.20, 0.80)
min_amp_pct       = 15.0    # amplitude percentile threshold for "valid" columns
push_excl_px      = 15
min_lag_frames    = 0.15
```
Two-pass procedure:
1. **Pass 1**: fixed step of 5 px — gives a provisional speed estimate.
2. **Pass 2**: step size resized so the expected lag is ≈1 frame: `step = clip(ceil(cs_guess·dt/dx), 3, 12)`.

Sub-frame lag is refined by 3-point parabolic interpolation of the cross-correlation peak. Final `cs` = median over all valid column-pair speeds (`0.2 < c < 15 m/s`), robust spread = IQR/1.349 (normal-equivalent std). The original driver's step-sizing rule omitted the wave-speed factor entirely and collapsed to the 5-px floor at every PRF; this two-pass version is the fix.

---

## 8. Kelvin-Voigt fit

Model (`voigt_phase_velocity`):
```
c(ω) = sqrt( 2(G² + (ωη)²) / (ρ·(sqrt(G² + (ωη)²) + G)) )
```
G in Pa, η in Pa·s, ω in rad/s, ρ = 1000 kg/m³.

Fit bounds:
```python
g_min_pa   = 100.0
g_max_pa   = 5.0e4
eta_min    = 0.01
eta_max    = 20.0
g_init_pa  = 2500.0     # overridden by rho*cs_toa^2 when a ToA hint exists
eta_init   = 1.5
```
- `scipy.optimize.curve_fit`, amplitude-weighted (`sigma = 1/sqrt(weight)`), bounded, `maxfev=5000`.
- Two fits are always attempted and both retained:
  - **Curve-only fit** (`G_prime_curve_kpa`, `eta_curve`): honest dispersion-slope estimate, best for η, but weak on G′ alone.
  - **Anchored fit** (`G_prime_kpa`, `eta`): curve + ToA anchor point appended — this is the fit reported as the primary result.
- `E_kpa = 3 · G_prime_kpa` (incompressible-medium relation).
- `G_elastic_kpa = ρ·cs_toa² / 1000` is also reported alongside, independent of the KV fit, as the conventional single-number SWE estimate.
- A fit is flagged `fit_at_bound` (not discarded, but marked) if either parameter lands within 0.1% of its bound.
- `r_squared` is the weighted R² of the fit against the (curve or curve+anchor) points used.

---

## 9. Why G′ is weak and η is strong (the aperture limit)

Documented directly in the config comments and confirmed in synthetic tests:

> The ROI is only ~16.2 mm wide and the push often sits mid-ROI. At the frequencies this aperture can actually resolve, the medium's response is already in the viscosity-dominated (high-frequency) asymptotic regime, where phase speed depends on η but is nearly independent of G′.

Synthetic recovery tolerances (`test_dispersion_synthetic.py`) were set to reflect this measured limit, not an aspiration:
```
|error G'| <= 60%   (pass threshold per case)
|error eta| <= 35%   (pass threshold per case)
overall pass bar: |dG'| <= 30%, |d eta| <= 40% (median across all cases)
```
This is why every downstream comparison (§11) treats η as the trustworthy, cross-model-comparable quantity and G′ as directionally indicative only. (See memory note `swe-dispersion-aperture-limit`.)

---

## 10. Ground-truth validation results (frozen before touching model output)

Run: `scripts/validate_dispersion_gt.py`, output `outputs/dispersion_gt_validation/gt_validation_summary.json`, **38/38 native high-PRF acquisitions**, `dt = 0.2 ms` (5 kHz).

### Pre-registered acceptance gates (all passed)

| Gate | Threshold | Result |
|---|---|---|
| Success rate | ≥ 90% | **100%** (38/38, 0 failures) |
| Median η | in [0.3, 5] Pa·s | pass |
| Across-acquisition G′ CV | ≤ 15% | **0.37%** |
| Across-acquisition η CV | ≤ 25% | **0.23%** |
| Band perturbation \|ΔG′\| (f_hi ±20%) | ≤ 25% | **6.3–7.0%** |
| Band perturbation \|Δη\| (f_hi ±20%) | ≤ 25% | **12.6–12.8%** |

This is a dramatic improvement over the old estimator's **37% success rate** on the same class of data (cited directly in the validation script's docstring).

### Repeatability (the noise floor for every model-vs-GT claim)

- **Within-acquisition** (windows of one recording): only 2 multi-member groups — G′ CV 33.6%, η CV 76.0% (noisy; small n).
- **Across-acquisition** (repeat pushes, same imaging plane): 9 groups — G′ CV 0.37%, η CV 0.23%, cs CV 0.14%. Very tight — this is the number any model-vs-GT deviation should be judged against.

### Band-perturbation robustness

Perturbing the binding high-frequency limit (`f_hi` is set by the Nyquist fraction at 5 kHz, so `f_max_abs_hz` alone is a no-op; both `f_max_abs_hz` and `f_max_nyq_frac` are scaled together):

| Perturbation | median \|ΔG′\| | median \|Δη\| |
|---|---|---|
| f_hi −20% | 6.3% | 12.6% |
| f_hi +20% | 7.0% | 12.8% |

---

## 11. Comparing ground truth to model predictions

`compare_dispersion(gt, pred)` — used because GT (native 5 kHz, `dt=0.2ms`) and predictions (12× upsampled from 500 Hz sparse input, `dt = (1/500)/12 ≈ 0.167ms`, i.e. 6 kHz-equivalent) sit on **different temporal grids**.

- Find the overlapping frequency band: `f_lo = max(gt.band_lo, pred.band_lo)`, `f_hi = min(gt.band_hi, pred.band_hi)`.
- Build a common comparison grid at **`compare_df_hz = 25.0 Hz`** spacing across `[f_lo, f_hi]` (minimum 3 points).
- **Both curves are interpolated onto this grid and re-fit there** (not resampled at the movie/pixel level) — deliberately, because resampling one *movie* onto the other's grid would inject an interpolation operator into the very quantity being measured.
- Reports: `c_rmse`, `c_bias` (phase-speed curve comparison), matched-band `G_prime_kpa`/`eta` for both GT and prediction, absolute and relative differences.

### Production evaluation protocol (`scripts/evaluate_swe_v14.py`)

- 500 Hz sparse IQ (5 input frames, `dt=2ms`) → model → 49 dense frames at `dt = 2ms/12` (6 kHz-equivalent).
- Compared against a native 5 kHz acquisition of the same imaging plane (**different acquisition/day** — known confound; `evaluate_matched_gt.py` is the confound-free variant).
- GT dispersion is cached per acquisition file (`estimate_cs_toa` + `estimate_dispersion`) so repeated windows don't re-run it.
- `vel_scale` for denormalizing prediction output = mean of the 99th-percentile |velocity| across all matched GT acquisitions.
- Aggregation is **acquisition-level** (all matched-window pairs kept, tracked by `plane`/`ts`/`window` so nothing is double-counted).

### Latest results (`outputs/swe_final/swe_v14_report.txt`)

Matched-band viscoelastic comparison, n=66 windows across `1ft/2ft/3ft/cirs` phantoms:

| Model | cs MAE (m/s) | η rel. error (median) | η signed bias | G′ rel. error (median) | G′ signed bias | curve RMSE (m/s) | η corr. r |
|---|---|---|---|---|---|---|---|
| v12 | 0.901 | 35% | +20% | 28% | −5% | 1.178 | +0.431 |
| v14 | 0.836 | 65% | +55% | 18% | −3% | 1.510 | −0.093 |
| v14d | 1.028 | 44% | +29% | 36% | −10% | 1.712 | +0.392 |

Notes reflected in the report itself:
- Positive η bias across all three models = systematic **over-estimation** of viscosity by the model relative to GT.
- v14's near-zero η correlation (r = −0.093) despite the lowest cs MAE is the flag behind the memory note on the v12 HF amplitude deficit — a model can nail time-of-arrival while failing to track viscosity across phantoms.
- `few_native_bins` is the dominant prediction-side failure reason (9–16 windows depending on model) — the model output sometimes doesn't retain enough spectral bins in the admissible band to pass gate §6.

---

## 12. Config-as-contract

The entire estimator is one frozen `@dataclass(frozen=True) DispersionConfig`, `version = "v1"`. The module docstring is explicit: **"All estimator defaults are frozen against ground-truth data only... they are never tuned on model output."** Any change to a threshold requires re-running `validate_dispersion_gt.py` and bumping `version` — this is what makes every G′/η number in `swe_v14_report.txt` traceable to a specific, ground-truth-validated configuration rather than a hand-tuned fit against the thing being evaluated.

### Full parameter reference

| Parameter | Value | Role |
|---|---|---|
| `depth_frac` | (0.25, 0.75) | depth-averaging band |
| `push_excl_px` | 15 | push near-field exclusion, px |
| `min_segment_px` | 30 | minimum usable lateral segment, px |
| `segment_mode` | "largest" | use widest outward segment only |
| `tukey_alpha` | 0.25 | Tukey taper fraction (t and x) |
| `pad_factor_t` / `pad_factor_x` | 4 / 4 | zero-pad multiplier |
| `min_pad_t` | 256 | temporal pad floor |
| `c_min` / `c_max` | 0.2 / 15.0 m/s | global admissible phase-speed bounds |
| `use_toa_window` | True | narrow search window around ToA speed |
| `toa_c_lo_factor` / `toa_c_hi_factor` | 1.8 / 2.5 | ToA window multipliers |
| `f_min_hz` | 75.0 | search band floor |
| `f_max_abs_hz` | 1500.0 | search band absolute ceiling |
| `f_max_nyq_frac` | 0.45 | search band ceiling as fraction of Nyquist |
| `snr_min_db` | 10.0 | per-frequency peak SNR gate |
| `centroid_frac` | 0.5 | centroid refinement threshold (fraction of peak) |
| `centroid_max_bins` | 5 | centroid refinement window cap |
| `min_aperture_cycles` | 0.8 | wavenumber resolution floor (cycles across aperture) |
| `band_db` | 20.0 | band-selection amplitude window |
| `min_native_bins` | 6 | minimum native-resolution bins retained |
| `min_band_ratio` | 3.0 | f_hi/f_lo minimum, no anchor |
| `min_band_ratio_anchored` | 1.6 | f_hi/f_lo minimum, with ToA anchor |
| `toa_anchor` | True | inject ToA low-frequency anchor point |
| `group_phase_ratio` | 1.125 (±0.062 calibrated) | ToA group→phase speed conversion |
| `g_min_pa` / `g_max_pa` | 100.0 / 5.0e4 | KV fit G′ bounds |
| `eta_min` / `eta_max` | 0.01 / 20.0 | KV fit η bounds |
| `g_init_pa` / `eta_init` | 2500.0 / 1.5 | KV fit initial guess (G overridden by ρ·cs_toa² when available) |
| `compare_df_hz` | 25.0 Hz | GT-vs-prediction comparison grid spacing |

---

## 13. Handoff to the MUSE Lab pipeline

`src/analysis/muselab_postproc.py` (added 2026-08-29) is a from-scratch, statement-for-statement Python port of the lab's MATLAB script (`PostProcessing_L11_5v_ARFIViscoelastography_MUSELab_20260812.m`), **not** a continuation of `dispersion.py`. It picks up after Loupas/median-filter/spline steps that were already baked into `data/fullscale_velocity`, and diverges structurally from the pipeline documented here: Butterworth 10–800 Hz bandpass, single focal-depth line (not depth-averaged band), Maksuti-style band cutoffs (>50% phase-velocity jump low end, k-space intensity <30%-of-max high end), and an edge-trimmed KV fit iterated until R² ≥ 0.99, rather than the SNR/centroid/ToA-anchor scheme above.
