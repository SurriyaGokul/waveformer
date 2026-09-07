# `src/analysis/`

Post-hoc quantitative analysis of predicted (or Loupas) velocity fields — turning a
velocity field into physical tissue-property estimates.

| File | Key contents |
|---|---|
| `dispersion.py` | `DispersionConfig`, `DispersionResult` — phase-velocity dispersion curve fitting used to estimate shear storage/loss modulus (G′, η) from the predicted velocity field. **See project memory note:** the aperture of the current ROI (16 mm) makes η only weakly identifiable from G′ — read `docs/` before trusting absolute G′/η values from this module |
| `muselab_postproc.py` | `KVResult`, `bandpass_velocity`, `_pad_length_for_exact_bins` — post-processing for MUSE-lab-format acquisitions (bandpass filtering + Kelvin-Voigt fit) |

## Usage

```python
from src.analysis.dispersion import DispersionConfig, estimate_dispersion, fit_kelvin_voigt

result = estimate_dispersion(velocity_field, config=DispersionConfig(...))
G, eta = fit_kelvin_voigt(result.omega, result.phase_velocity)  # see file for exact signature
```
