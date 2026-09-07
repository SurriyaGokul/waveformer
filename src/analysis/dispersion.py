"""
Shear-wave dispersion analysis and Kelvin-Voigt parameter estimation.

This module replaces ``scripts/evaluate_wavepinn_v10.py:estimate_voigt``, which
had a wavenumber unit error (``np.fft.fftfreq`` returns cycles/m but the phase
velocity was computed as ``omega / kx`` with omega in rad/s), inflating every
phase velocity by 2*pi.  That inflation collided with the ``cs < 12 m/s``
admissibility gate, so 63% of 5 kHz ground-truth acquisitions returned NaN and
the survivors were fit from 3-4 grossly inflated points, producing viscosities
of 20-37 Pa.s where gel phantoms sit near 1-2 Pa.s.

The estimator here follows standard 2D-Fourier SWE dispersion practice:

    depth-average -> push localisation -> one-directional segments -> detrend
    -> Tukey window (t, x) -> zero-pad -> 2D FFT -> single quadrant
    -> per-frequency wavenumber peak (SNR-gated, thresholded centroid)
    -> SNR-based band selection -> amplitude-weighted Kelvin-Voigt fit

A second, independent estimator (``estimate_dispersion_phase_gradient``) fits
the phase ramp along x at each frequency.  It shares the preprocessing but not
the wavenumber extraction, so agreement between the two is evidence that the
dispersion curve is real rather than an artefact of the spectral peak picker.

Ground truth and predictions live on different temporal grids (5 kHz native GT
at dt=0.2 ms; model output at dt=dt_in/12).  ``compare_dispersion`` therefore
compares *dispersion curves* on the intersection of their valid bands rather
than resampling one movie onto the other's grid, which would inject the very
interpolation operator under test.

All estimator defaults are frozen against ground-truth data only (see
``scripts/validate_dispersion_gt.py``); they are never tuned on model output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Acquisition geometry / medium constants ───────────────────────────────────
LATERAL_M_PX = 0.1013e-3     # lateral pixel pitch [m]
RHO = 1000.0                 # density [kg/m^3]


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DispersionConfig:
    """Frozen estimator defaults.  Changing any of these invalidates all
    previously computed results — bump ``version`` when you do."""

    version: str = "v2"

    # ROI / geometry
    depth_frac: Tuple[float, float] = (0.25, 0.75)
    push_excl_px: int = 15
    min_segment_px: int = 30
    # Aperture sets the wavenumber resolution (dk ~ 2*pi/L), which is the
    # binding constraint here: the ROI is only 16.2 mm wide and the push often
    # sits mid-ROI.  Pooling a wide and a narrow segment averages a usable
    # estimate with an aperture-starved one, so use the widest segment.
    segment_mode: str = "largest"   # "largest" | "pooled"

    # Spectral estimation
    tukey_alpha: float = 0.25
    pad_factor_t: int = 4
    pad_factor_x: int = 4
    min_pad_t: int = 256

    # Admissible phase speeds [m/s]
    c_min: float = 0.2
    c_max: float = 15.0
    # When a time-of-arrival speed is available it is a far better conditioned
    # estimate than any single wavenumber bin (it uses the full aperture and
    # sub-frame lag interpolation), so it narrows the wavenumber search to a
    # physically plausible window around it.  Without this the low-frequency
    # bins latch onto the aperture main lobe, which reads as a huge apparent
    # dispersion and drives G' to its lower bound.
    use_toa_window: bool = True
    toa_c_lo_factor: float = 1.8    # c >= cs_toa / this
    toa_c_hi_factor: float = 2.5    # c <= cs_toa * this

    # Frequency search band
    f_min_hz: float = 75.0
    f_max_abs_hz: float = 1500.0
    f_max_nyq_frac: float = 0.45

    # Per-frequency wavenumber peak acceptance
    snr_min_db: float = 10.0
    centroid_frac: float = 0.5      # bins above this fraction of peak power
    centroid_max_bins: int = 5      # +/- padded bins around the peak

    # Wavenumber-resolution gate.  A segment of length L cannot resolve
    # wavenumbers below ~2*pi/L; below that the peak picker returns the
    # aperture main lobe instead of the wave, which drags c down at low
    # frequency and rotates the whole dispersion curve (verified on synthetic
    # Kelvin-Voigt waves in scripts/test_dispersion_synthetic.py).  Require at
    # least this many wavelengths across the analysed segment.
    min_aperture_cycles: float = 0.8

    # Band selection
    band_db: float = 20.0
    min_native_bins: int = 6
    min_band_ratio: float = 3.0

    # Time-of-arrival anchor.  The resolvable band sits above the elastic
    # regime, where c(w) -> sqrt(2*w*eta/rho) is asymptotically independent of
    # G, so the curve alone constrains eta well but G poorly.  The ToA speed
    # supplies the missing low-frequency anchor.  ``group_phase_ratio``
    # converts the ToA group speed to a phase speed at the measured spectral
    # centroid.
    #
    # It used to default to 1.125, calibrated on synthetic Kelvin-Voigt waves
    # with eta 0.5-4 Pa.s.  That calibration is circular: the group/phase ratio
    # is itself a function of the viscosity being estimated, and it tends to 1
    # as eta -> 0.  Applying 1.125 to a near-elastic medium plants the anchor
    # ~11% BELOW the true phase speed at mid-band, which on its own forces the
    # fit into positive dispersion and manufactures viscosity.  Default is now
    # 1.0 (no correction); the anchor is a level constraint only.
    toa_anchor: bool = True
    group_phase_ratio: float = 1.0
    min_band_ratio_anchored: float = 1.6

    # Kelvin-Voigt fit bounds
    g_min_pa: float = 100.0
    g_max_pa: float = 5.0e4
    eta_min: float = 0.01
    eta_max: float = 20.0
    g_init_pa: float = 2500.0
    eta_init: float = 1.5

    # Curve comparison grid
    compare_df_hz: float = 25.0


DEFAULT_CONFIG = DispersionConfig()


@dataclass
class DispersionResult:
    """Outcome of one dispersion estimate.

    ``ok`` is False when the estimate did not pass the quality gates; ``reason``
    then carries a machine-readable code (see ``FAILURE_REASONS``).
    """

    ok: bool = False
    reason: str = ""
    method: str = "fft2d"

    # Kelvin-Voigt parameters
    G_prime_kpa: float = float("nan")
    eta: float = float("nan")
    E_kpa: float = float("nan")
    G_prime_ci_kpa: float = float("nan")
    eta_ci: float = float("nan")
    r_squared: float = float("nan")

    # Curve-only fit (no ToA anchor).  Better for eta, much worse for G'.
    G_prime_curve_kpa: float = float("nan")
    eta_curve: float = float("nan")
    anchored: bool = False
    anchor_f_hz: float = float("nan")
    anchor_c: float = float("nan")

    # Elastic modulus from the ToA speed alone (rho * cs^2), the conventional
    # SWE estimate; reported alongside because it is independent of the fit.
    G_elastic_kpa: float = float("nan")

    # Dispersion curve actually used by the fit
    freqs_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c_phase: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Quality / provenance
    band_hz: Tuple[float, float] = (float("nan"), float("nan"))
    n_bins: int = 0
    n_native_bins: int = 0
    n_segments: int = 0
    aperture_mm: float = float("nan")
    df_native_hz: float = float("nan")
    cs_toa_hint: float = float("nan")

    def as_dict(self) -> Dict:
        """Flat, JSON-serialisable summary (drops the raw curve arrays)."""
        return dict(
            ok=self.ok,
            reason=self.reason,
            method=self.method,
            G_prime_kpa=_j(self.G_prime_kpa),
            eta=_j(self.eta),
            E_kpa=_j(self.E_kpa),
            G_prime_ci_kpa=_j(self.G_prime_ci_kpa),
            eta_ci=_j(self.eta_ci),
            r_squared=_j(self.r_squared),
            G_prime_curve_kpa=_j(self.G_prime_curve_kpa),
            eta_curve=_j(self.eta_curve),
            G_elastic_kpa=_j(self.G_elastic_kpa),
            anchored=self.anchored,
            anchor_f_hz=_j(self.anchor_f_hz),
            anchor_c=_j(self.anchor_c),
            band_lo_hz=_j(self.band_hz[0]),
            band_hi_hz=_j(self.band_hz[1]),
            n_bins=self.n_bins,
            n_native_bins=self.n_native_bins,
            n_segments=self.n_segments,
            aperture_mm=_j(self.aperture_mm),
            df_native_hz=_j(self.df_native_hz),
            cs_toa_hint=_j(self.cs_toa_hint),
        )


FAILURE_REASONS = {
    "": "no failure",
    "empty_input": "velocity array had no usable samples",
    "no_segment": "no lateral segment wide enough after push exclusion",
    "no_bins": "no frequency bin passed the SNR / admissible-speed gate",
    "few_native_bins": "band spans fewer than min_native_bins native bins",
    "narrow_band": "band ratio f_hi/f_lo below min_band_ratio",
    "fit_failed": "Kelvin-Voigt least squares did not converge",
    "fit_at_bound": "fitted parameters pinned to a bound",
    "no_overlap": "GT and prediction bands do not overlap",
}


def _j(x) -> float:
    """NaN-safe float for JSON."""
    x = float(x)
    return x if np.isfinite(x) else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════
def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def _tukey(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered-cosine) window; avoids a scipy import at call sites."""
    if n <= 1:
        return np.ones(max(n, 0))
    if alpha <= 0:
        return np.ones(n)
    alpha = min(alpha, 1.0)
    w = np.ones(n)
    taper = int(np.floor(alpha * (n - 1) / 2.0))
    if taper < 1:
        return w
    k = np.arange(taper + 1)
    ramp = 0.5 * (1 + np.cos(np.pi * (2 * k / (alpha * (n - 1)) - 1)))
    w[: taper + 1] = ramp
    w[n - taper - 1:] = ramp[::-1]
    return w


def depth_average(velocity: np.ndarray, depth_frac: Tuple[float, float]) -> np.ndarray:
    """(T, H, W) -> (T, W), averaging the central depth band."""
    T, H, W = velocity.shape
    r0, r1 = int(H * depth_frac[0]), int(H * depth_frac[1])
    r1 = max(r1, r0 + 1)
    return np.mean(velocity[:, r0:r1, :], axis=1)


def find_push_column(v_avg: np.ndarray) -> int:
    """Lateral column of the push, from peak energy in the earliest frames."""
    T = v_avg.shape[0]
    n_early = max(3, int(0.1 * T))
    return int(np.argmax(np.abs(v_avg[:n_early]).max(axis=0)))


def _outward_segments(
    v_avg: np.ndarray, cfg: DispersionConfig
) -> List[np.ndarray]:
    """Split at the push into segments where the wave travels outward.

    Each returned segment is oriented so propagation is toward increasing x,
    which makes single-quadrant selection an exact directional filter.
    """
    W = v_avg.shape[1]
    xp = find_push_column(v_avg)
    lo = max(0, xp - cfg.push_excl_px)
    hi = min(W, xp + cfg.push_excl_px + 1)

    segments = []
    left = v_avg[:, :lo]
    if left.shape[1] >= cfg.min_segment_px:
        # Wave travels toward -x on the left of the push: mirror it.
        segments.append(left[:, ::-1])
    right = v_avg[:, hi:]
    if right.shape[1] >= cfg.min_segment_px:
        segments.append(right)

    if cfg.segment_mode == "largest" and segments:
        segments = [max(segments, key=lambda s: s.shape[1])]
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# Kelvin-Voigt model and fit
# ══════════════════════════════════════════════════════════════════════════════
def voigt_phase_velocity(omega: np.ndarray, G: float, eta: float) -> np.ndarray:
    """Kelvin-Voigt shear phase velocity [m/s].

        c(w) = sqrt( 2 (G^2 + (w eta)^2) / (rho (sqrt(G^2 + (w eta)^2) + G)) )

    G in Pa, eta in Pa.s, omega in rad/s.
    """
    mag = np.sqrt(G ** 2 + (omega * eta) ** 2)
    num = 2.0 * (G ** 2 + (omega * eta) ** 2)
    den = RHO * (mag + G)
    return np.sqrt(np.maximum(num / np.maximum(den, 1e-30), 0.0))


def fit_kelvin_voigt(
    freqs_hz: np.ndarray,
    c_phase: np.ndarray,
    weights: np.ndarray,
    cfg: DispersionConfig = DEFAULT_CONFIG,
    cs_hint: float = float("nan"),
) -> Dict:
    """Amplitude-weighted Kelvin-Voigt fit of a dispersion curve."""
    from scipy.optimize import curve_fit

    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    c = np.asarray(c_phase, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w[w > 0]) if np.any(w > 0) else 1.0)

    g_init = cfg.g_init_pa
    if np.isfinite(cs_hint) and cs_hint > 0:
        g_init = float(np.clip(RHO * cs_hint ** 2, cfg.g_min_pa * 1.01, cfg.g_max_pa * 0.99))

    try:
        popt, pcov = curve_fit(
            voigt_phase_velocity,
            omega,
            c,
            p0=[g_init, cfg.eta_init],
            sigma=1.0 / np.sqrt(w),
            absolute_sigma=False,
            bounds=([cfg.g_min_pa, cfg.eta_min], [cfg.g_max_pa, cfg.eta_max]),
            maxfev=5000,
        )
    except Exception as exc:  # convergence failure, singular jacobian, ...
        logger.debug("Kelvin-Voigt fit failed: %s", exc)
        return dict(ok=False, reason="fit_failed")

    G_pa, eta = float(popt[0]), float(popt[1])

    # Weighted R^2 of the fit
    pred = voigt_phase_velocity(omega, G_pa, eta)
    ss_res = float(np.sum(w * (c - pred) ** 2))
    ss_tot = float(np.sum(w * (c - np.average(c, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    try:
        perr = np.sqrt(np.diag(pcov))
        g_ci, eta_ci = float(perr[0]), float(perr[1])
    except Exception:
        g_ci = eta_ci = float("nan")

    at_bound = (
        G_pa <= cfg.g_min_pa * 1.001 or G_pa >= cfg.g_max_pa * 0.999
        or eta <= cfg.eta_min * 1.001 or eta >= cfg.eta_max * 0.999
    )

    return dict(
        ok=True,
        reason="fit_at_bound" if at_bound else "",
        G_pa=G_pa,
        eta=eta,
        G_ci_pa=g_ci,
        eta_ci=eta_ci,
        r_squared=r2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Primary estimator: 2D Fourier
# ══════════════════════════════════════════════════════════════════════════════
def _segment_spectrum(
    seg: np.ndarray, dt_s: float, dx_m: float, cfg: DispersionConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D power spectrum of one outward segment, restricted to omega > 0 and a
    single (dominant) wavenumber quadrant.

    Returns
    -------
    power : (n_f, n_k) power over positive frequencies and |k| >= 0
    f_hz  : (n_f,) positive temporal frequencies [Hz]
    k_rad : (n_k,) positive angular wavenumbers [rad/m]
    """
    T, Wseg = seg.shape

    # Detrend: remove each column's temporal mean (kills DC / clutter residue)
    seg = seg - seg.mean(axis=0, keepdims=True)

    # Separable Tukey taper in t and x
    seg = seg * _tukey(T, cfg.tukey_alpha)[:, None]
    seg = seg * _tukey(Wseg, cfg.tukey_alpha)[None, :]

    nt = max(_next_pow2(cfg.pad_factor_t * T), cfg.min_pad_t)
    nx = _next_pow2(cfg.pad_factor_x * Wseg)

    V = np.fft.fft2(seg, s=(nt, nx))
    P = np.abs(V) ** 2

    f_all = np.fft.fftfreq(nt, dt_s)                      # Hz
    # THE FIX: fftfreq is cycles/m, angular wavenumber needs the 2*pi.
    k_all = 2.0 * np.pi * np.fft.fftfreq(nx, dx_m)        # rad/m

    f_pos = f_all > 0
    k_pos = k_all > 0
    k_neg = k_all < 0

    # A wave travelling toward +x occupies exactly one (omega>0, sign k)
    # quadrant.  Which sign that is depends on the FFT sign convention, so pick
    # the quadrant that actually carries the energy: that IS the outward-
    # propagating one, since the segment was oriented outward.
    P_pos = P[np.ix_(f_pos, k_pos)]
    P_neg = P[np.ix_(f_pos, k_neg)][:, ::-1]              # mirror to |k| ascending

    if P_neg.sum() > P_pos.sum():
        power = P_neg
        k_rad = -k_all[k_neg][::-1]
    else:
        power = P_pos
        k_rad = k_all[k_pos]

    return power, f_all[f_pos], k_rad


def _extract_wavenumbers(
    power: np.ndarray,
    f_hz: np.ndarray,
    k_rad: np.ndarray,
    cfg: DispersionConfig,
    f_hi_limit: float,
    aperture_m: float,
    c_lo: float,
    c_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frequency wavenumber peak with SNR and resolution gating.

    Returns (freqs, c_phase, weights, band_amplitude) for the bins that passed.
    """
    snr_lin = 10.0 ** (cfg.snr_min_db / 10.0)
    k_resolvable = cfg.min_aperture_cycles * 2.0 * np.pi / max(aperture_m, 1e-9)
    in_search = (f_hz >= cfg.f_min_hz) & (f_hz <= f_hi_limit)
    idx_search = np.flatnonzero(in_search)

    freqs, cs, wts, amps = [], [], [], []
    for i in idx_search:
        omega = 2.0 * np.pi * f_hz[i]
        row = power[i]

        # Physically admissible wavenumbers for this frequency.  The aperture
        # resolution limit is NOT applied here: intersecting it into the search
        # window does not reject an unresolvable bin, it forces the peak picker
        # to return the resolution lobe instead of the (excluded) true peak,
        # which biases c low at every frequency below ~c/L and manufactures
        # positive dispersion.  Search the full physical range, then reject.
        k_lo = omega / c_hi
        k_hi = omega / c_lo
        adm = np.flatnonzero((k_rad >= k_lo) & (k_rad <= k_hi))
        if adm.size < 3:
            continue

        # Noise floor from the whole quadrant row (mostly noise), not just the
        # admissible window, which can be dominated by the peak itself.
        floor = float(np.median(row))
        if floor <= 0:
            floor = float(np.mean(row)) or 1e-30

        j_rel = int(np.argmax(row[adm]))
        j_pk = int(adm[j_rel])
        p_pk = float(row[j_pk])
        if p_pk < snr_lin * floor:
            continue

        # Refine with a thresholded centroid over the contiguous bins around
        # the peak that stay above centroid_frac of it.
        thr = cfg.centroid_frac * p_pk
        lo = j_pk
        while lo - 1 >= adm[0] and row[lo - 1] >= thr and j_pk - (lo - 1) <= cfg.centroid_max_bins:
            lo -= 1
        hi = j_pk
        while hi + 1 <= adm[-1] and row[hi + 1] >= thr and (hi + 1) - j_pk <= cfg.centroid_max_bins:
            hi += 1
        sl = slice(lo, hi + 1)
        wsum = float(row[sl].sum())
        k_pk = float(np.dot(row[sl], k_rad[sl]) / wsum) if wsum > 0 else float(k_rad[j_pk])
        if k_pk <= 0:
            continue
        # Reject wavenumbers the aperture cannot resolve.  Now that the search
        # window is not floored at k_resolvable this is a live test: a bin whose
        # true peak sits below the resolution limit is discarded rather than
        # replaced by a spurious higher-k one.
        if k_pk < k_resolvable:
            continue

        c = omega / k_pk
        if not (c_lo <= c <= c_hi):
            continue

        freqs.append(float(f_hz[i]))
        cs.append(float(c))
        wts.append(float(np.sqrt(p_pk)))
        amps.append(float(row[adm].sum()))

    return (
        np.asarray(freqs), np.asarray(cs), np.asarray(wts), np.asarray(amps),
    )


def _spectral_centroid_hz(
    seg: np.ndarray, dt_s: float, cfg: DispersionConfig, f_hi_limit: float
) -> float:
    """Energy-weighted mean temporal frequency of a segment, in the search band.

    This is the frequency the ToA cross-correlation is effectively sensitive
    to, so it is where the ToA anchor point is placed.
    """
    s = seg - seg.mean(axis=0, keepdims=True)
    s = s * _tukey(s.shape[0], cfg.tukey_alpha)[:, None]
    nt = max(_next_pow2(cfg.pad_factor_t * s.shape[0]), cfg.min_pad_t)
    P = (np.abs(np.fft.rfft(s, n=nt, axis=0)) ** 2).sum(axis=1)
    f = np.fft.rfftfreq(nt, dt_s)
    m = (f >= cfg.f_min_hz) & (f <= f_hi_limit)
    if not np.any(m) or P[m].sum() <= 0:
        return float("nan")
    return float((f[m] * P[m]).sum() / P[m].sum())


def _speed_window(cfg: DispersionConfig, cs_toa_hint: float) -> Tuple[float, float]:
    """Admissible phase-speed window, narrowed around the ToA speed if known."""
    if cfg.use_toa_window and np.isfinite(cs_toa_hint) and cs_toa_hint > 0:
        return (
            max(cfg.c_min, cs_toa_hint / cfg.toa_c_lo_factor),
            min(cfg.c_max, cs_toa_hint * cfg.toa_c_hi_factor),
        )
    return cfg.c_min, cfg.c_max


def _select_band(
    freqs: np.ndarray, amps: np.ndarray, cfg: DispersionConfig
) -> np.ndarray:
    """Largest contiguous run of bins within ``band_db`` of the peak amplitude."""
    if freqs.size == 0:
        return np.zeros(0, dtype=int)
    keep = amps >= amps.max() * 10.0 ** (-cfg.band_db / 10.0)

    best, cur = [], []
    for i, k in enumerate(keep):
        if k:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return np.asarray(best, dtype=int)


def estimate_dispersion(
    velocity: np.ndarray,
    dt_s: float,
    dx_m: float = LATERAL_M_PX,
    cfg: DispersionConfig = DEFAULT_CONFIG,
    cs_toa_hint: float = float("nan"),
) -> DispersionResult:
    """Estimate the shear-wave dispersion curve and Kelvin-Voigt parameters.

    Parameters
    ----------
    velocity : (T, H, W) particle velocity [mm/s] (units cancel; only the
        spatio-temporal structure matters).
    dt_s : frame interval [s].
    dx_m : lateral pixel pitch [m].
    cs_toa_hint : optional time-of-arrival speed used to initialise G.
    """
    res = DispersionResult(method="fft2d", cs_toa_hint=cs_toa_hint)

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[0] < 8 or not np.any(np.isfinite(velocity)):
        res.reason = "empty_input"
        return res
    velocity = np.nan_to_num(velocity)

    T = velocity.shape[0]
    res.df_native_hz = 1.0 / (T * dt_s)
    f_nyq = 0.5 / dt_s
    f_hi_limit = min(cfg.f_max_abs_hz, cfg.f_max_nyq_frac * f_nyq)

    v_avg = depth_average(velocity, cfg.depth_frac)
    segments = _outward_segments(v_avg, cfg)
    res.n_segments = len(segments)
    if not segments:
        res.reason = "no_segment"
        return res
    res.aperture_mm = max(s.shape[1] for s in segments) * dx_m * 1e3

    c_lo, c_hi = _speed_window(cfg, cs_toa_hint)

    # Pool per-frequency estimates across both outward segments.
    pool: Dict[float, List[Tuple[float, float, float]]] = {}
    for seg in segments:
        power, f_hz, k_rad = _segment_spectrum(seg, dt_s, dx_m, cfg)
        f_s, c_s, w_s, a_s = _extract_wavenumbers(
            power, f_hz, k_rad, cfg, f_hi_limit, seg.shape[1] * dx_m, c_lo, c_hi
        )
        for f, c, w, a in zip(f_s, c_s, w_s, a_s):
            pool.setdefault(round(float(f), 6), []).append((c, w, a))

    if not pool:
        res.reason = "no_bins"
        return res

    freqs = np.array(sorted(pool))
    c_phase = np.array([np.average([v[0] for v in pool[f]],
                                   weights=[v[1] for v in pool[f]]) for f in freqs])
    weights = np.array([np.sum([v[1] for v in pool[f]]) for f in freqs])
    amps = np.array([np.sum([v[2] for v in pool[f]]) for f in freqs])

    sel = _select_band(freqs, amps, cfg)
    if sel.size == 0:
        res.reason = "no_bins"
        return res

    freqs, c_phase, weights = freqs[sel], c_phase[sel], weights[sel]
    f_lo, f_hi = float(freqs[0]), float(freqs[-1])
    res.freqs_hz, res.c_phase, res.weights = freqs, c_phase, weights
    res.band_hz = (f_lo, f_hi)
    res.n_bins = int(freqs.size)
    # Zero-padding refines the grid but adds no information: count how many
    # *native* resolution cells the retained band actually spans.
    res.n_native_bins = int(np.unique(np.round(freqs / res.df_native_hz)).size)

    if np.isfinite(cs_toa_hint) and cs_toa_hint > 0:
        res.G_elastic_kpa = RHO * cs_toa_hint ** 2 / 1000.0

    if res.n_native_bins < cfg.min_native_bins:
        res.reason = "few_native_bins"
        return res

    # The ToA anchor supplies the low-frequency constraint the aperture cannot,
    # which relaxes how wide the measured band itself has to be.
    anchor_f = float("nan")
    anchor_c = float("nan")
    use_anchor = (
        cfg.toa_anchor and np.isfinite(cs_toa_hint) and cs_toa_hint > 0
    )
    if use_anchor:
        anchor_f = _spectral_centroid_hz(segments[0], dt_s, cfg, f_hi_limit)
        anchor_c = cs_toa_hint / cfg.group_phase_ratio
        use_anchor = np.isfinite(anchor_f) and anchor_f > 0

    band_ratio_needed = cfg.min_band_ratio_anchored if use_anchor else cfg.min_band_ratio
    if f_lo <= 0 or f_hi / f_lo < band_ratio_needed:
        res.reason = "narrow_band"
        return res

    # Curve-only fit: the honest dispersion-slope estimate (best for eta).
    fit_curve = fit_kelvin_voigt(freqs, c_phase, weights, cfg, cs_toa_hint)
    if fit_curve["ok"]:
        res.G_prime_curve_kpa = fit_curve["G_pa"] / 1000.0
        res.eta_curve = fit_curve["eta"]

    if use_anchor:
        f_fit = np.append(freqs, anchor_f)
        c_fit = np.append(c_phase, anchor_c)
        # Give the anchor the same total leverage as the measured curve: it
        # fixes the absolute level, the curve fixes the slope.
        w_fit = np.append(weights, weights.sum())
        fit = fit_kelvin_voigt(f_fit, c_fit, w_fit, cfg, anchor_c)
        res.anchored = True
        res.anchor_f_hz = anchor_f
        res.anchor_c = anchor_c
    else:
        fit = fit_curve

    if not fit["ok"]:
        res.reason = fit["reason"]
        return res

    res.ok = True
    res.reason = fit["reason"]
    res.G_prime_kpa = fit["G_pa"] / 1000.0
    res.eta = fit["eta"]
    res.E_kpa = 3.0 * res.G_prime_kpa
    res.G_prime_ci_kpa = fit["G_ci_pa"] / 1000.0
    res.eta_ci = fit["eta_ci"]
    res.r_squared = fit["r_squared"]
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Cross-check estimator: phase gradient along x
# ══════════════════════════════════════════════════════════════════════════════
def estimate_dispersion_phase_gradient(
    velocity: np.ndarray,
    dt_s: float,
    dx_m: float = LATERAL_M_PX,
    cfg: DispersionConfig = DEFAULT_CONFIG,
    cs_toa_hint: float = float("nan"),
) -> DispersionResult:
    """Independent dispersion estimate from the phase ramp along x.

    Shares preprocessing with :func:`estimate_dispersion` but extracts the
    wavenumber by regressing unwrapped phase against lateral position, so the
    two disagree if the 2D spectral peak picker is latching onto an artefact.
    """
    res = DispersionResult(method="phase_gradient", cs_toa_hint=cs_toa_hint)

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[0] < 8 or not np.any(np.isfinite(velocity)):
        res.reason = "empty_input"
        return res
    velocity = np.nan_to_num(velocity)

    T = velocity.shape[0]
    res.df_native_hz = 1.0 / (T * dt_s)
    f_hi_limit = min(cfg.f_max_abs_hz, cfg.f_max_nyq_frac * (0.5 / dt_s))

    v_avg = depth_average(velocity, cfg.depth_frac)
    segments = _outward_segments(v_avg, cfg)
    res.n_segments = len(segments)
    if not segments:
        res.reason = "no_segment"
        return res
    res.aperture_mm = max(s.shape[1] for s in segments) * dx_m * 1e3

    nt = max(_next_pow2(cfg.pad_factor_t * T), cfg.min_pad_t)
    c_lo, c_hi = _speed_window(cfg, cs_toa_hint)
    pool: Dict[float, List[Tuple[float, float, float]]] = {}

    for seg in segments:
        Wseg = seg.shape[1]
        s = seg - seg.mean(axis=0, keepdims=True)
        s = s * _tukey(T, cfg.tukey_alpha)[:, None]

        V = np.fft.rfft(s, n=nt, axis=0)          # (n_f, Wseg)
        f_hz = np.fft.rfftfreq(nt, dt_s)
        x = np.arange(Wseg) * dx_m

        for i in np.flatnonzero((f_hz >= cfg.f_min_hz) & (f_hz <= f_hi_limit)):
            spec = V[i]
            amp = np.abs(spec)
            if amp.max() <= 0:
                continue
            # Only columns with real signal constrain the phase ramp.
            good = amp >= 0.25 * amp.max()
            if good.sum() < max(cfg.min_segment_px // 2, 10):
                continue

            phase = np.unwrap(np.angle(spec))
            xg, pg, wg = x[good], phase[good], amp[good]

            # Amplitude-weighted least-squares slope
            xbar = np.average(xg, weights=wg)
            pbar = np.average(pg, weights=wg)
            denom = float(np.sum(wg * (xg - xbar) ** 2))
            if denom <= 0:
                continue
            slope = float(np.sum(wg * (xg - xbar) * (pg - pbar)) / denom)

            k = abs(slope)                         # rad/m
            if k <= 0:
                continue
            # Same aperture-resolution gate as the 2D-FFT path: a phase ramp
            # spanning much less than a wavelength does not constrain k.
            if k * (Wseg * dx_m) < cfg.min_aperture_cycles * 2.0 * np.pi:
                continue
            omega = 2.0 * np.pi * f_hz[i]
            c = omega / k
            if not (c_lo <= c <= c_hi):
                continue

            # Weighted correlation: how linear is the ramp really?
            sp = float(np.sum(wg * (pg - pbar) ** 2))
            r2 = (np.sum(wg * (xg - xbar) * (pg - pbar)) ** 2 / (denom * sp)) if sp > 0 else 0.0
            if r2 < 0.5:
                continue

            w = float(np.sum(wg))
            pool.setdefault(round(float(f_hz[i]), 6), []).append((c, w, w))

    if not pool:
        res.reason = "no_bins"
        return res

    freqs = np.array(sorted(pool))
    c_phase = np.array([np.average([v[0] for v in pool[f]],
                                   weights=[v[1] for v in pool[f]]) for f in freqs])
    weights = np.array([np.sum([v[1] for v in pool[f]]) for f in freqs])

    sel = _select_band(freqs, weights ** 2, cfg)
    if sel.size == 0:
        res.reason = "no_bins"
        return res
    freqs, c_phase, weights = freqs[sel], c_phase[sel], weights[sel]

    res.freqs_hz, res.c_phase, res.weights = freqs, c_phase, weights
    res.band_hz = (float(freqs[0]), float(freqs[-1]))
    res.n_bins = int(freqs.size)
    res.n_native_bins = int(np.unique(np.round(freqs / res.df_native_hz)).size)

    if res.n_native_bins < cfg.min_native_bins:
        res.reason = "few_native_bins"
        return res
    if freqs[0] <= 0 or freqs[-1] / freqs[0] < cfg.min_band_ratio:
        res.reason = "narrow_band"
        return res

    fit = fit_kelvin_voigt(freqs, c_phase, weights, cfg, cs_toa_hint)
    if not fit["ok"]:
        res.reason = fit["reason"]
        return res

    res.ok = True
    res.reason = fit["reason"]
    res.G_prime_kpa = fit["G_pa"] / 1000.0
    res.eta = fit["eta"]
    res.E_kpa = 3.0 * res.G_prime_kpa
    res.G_prime_ci_kpa = fit["G_ci_pa"] / 1000.0
    res.eta_ci = fit["eta_ci"]
    res.r_squared = fit["r_squared"]
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Time of arrival
# ══════════════════════════════════════════════════════════════════════════════
def estimate_cs_toa(
    velocity: np.ndarray,
    dt_s: float,
    dx_m: float = LATERAL_M_PX,
    depth_frac: Tuple[float, float] = (0.20, 0.80),
    min_amp_pct: float = 15.0,
    push_excl_px: int = 15,
    min_lag_frames: float = 0.15,
) -> Dict:
    """Shear-wave speed from pairwise cross-correlation time-of-arrival.

    Two passes: the first fixes a provisional speed, the second sizes the
    column step so the expected lag is about one frame.  The original
    ``evaluate_wavepinn_v10.py:80`` step rule omitted the wave-speed factor and
    so collapsed to the 5 px floor at every PRF; the correct rule is the one in
    ``scripts/swe_publication_v10b.py:174-180``.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    T, H, W = velocity.shape
    v_avg = depth_average(velocity, depth_frac)
    v_abs = np.abs(v_avg)

    amp = v_abs.max(axis=0)
    thr = np.percentile(amp[amp > 0], min_amp_pct) if np.any(amp > 0) else 0.0
    valid = amp > thr

    xp = find_push_column(v_avg)
    valid[max(0, xp - push_excl_px): min(W, xp + push_excl_px + 1)] = False

    def _pass(step: int) -> Tuple[float, float, np.ndarray]:
        cs_map = np.full(W, np.nan)
        vals = []
        for x in range(0, W - step):
            x2 = x + step
            if not (valid[x] and valid[x2]):
                continue
            a, b = v_avg[:, x], v_avg[:, x2]
            ea, eb = np.sqrt(np.sum(a ** 2)), np.sqrt(np.sum(b ** 2))
            if ea < 1e-12 or eb < 1e-12:
                continue
            xc = np.correlate(a / ea, b / eb, mode="full")
            lags = np.arange(-(T - 1), T, dtype=float)
            pi = int(np.argmax(xc))
            if 0 < pi < len(xc) - 1:
                y0, y1, y2 = xc[pi - 1], xc[pi], xc[pi + 1]
                den = y0 - 2 * y1 + y2
                frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                lag = lags[pi] + frac
            else:
                lag = lags[pi]
            if abs(lag) < min_lag_frames:
                continue
            c = abs(step * dx_m / (lag * dt_s))
            if 0.2 < c < 15.0:
                vals.append(c)
                cs_map[x] = c
        if len(vals) < 3:
            return float("nan"), float("nan"), cs_map
        arr = np.asarray(vals)
        med = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        if not (0.2 < med < 15.0):
            return float("nan"), float("nan"), cs_map
        return med, iqr / 1.349, cs_map

    # Pass 1: provisional speed with a conservative step.
    cs1, _, _ = _pass(5)
    cs_guess = cs1 if np.isfinite(cs1) else 4.0

    # Pass 2: size the step so the expected lag is ~1 frame.
    step = int(min(12, max(3, np.ceil(cs_guess * dt_s / dx_m))))
    cs, cs_std, cs_map = _pass(step)
    return dict(cs=cs, cs_std=cs_std, cs_map=cs_map, step_px=step, cs_pass1=cs1)


# ══════════════════════════════════════════════════════════════════════════════
# Full SWE summary + GT/prediction comparison
# ══════════════════════════════════════════════════════════════════════════════
def swe_parameters(
    velocity: np.ndarray,
    dt_s: float,
    dx_m: float = LATERAL_M_PX,
    cfg: DispersionConfig = DEFAULT_CONFIG,
    with_phase_gradient: bool = False,
) -> Dict:
    """cs (ToA) + Kelvin-Voigt G', eta, E for one velocity movie."""
    toa = estimate_cs_toa(velocity, dt_s, dx_m)
    disp = estimate_dispersion(velocity, dt_s, dx_m, cfg, cs_toa_hint=toa["cs"])
    out = dict(
        cs=toa["cs"], cs_std=toa["cs_std"], toa_step_px=toa["step_px"],
        **disp.as_dict(),
    )
    out["_dispersion"] = disp
    if with_phase_gradient:
        pg = estimate_dispersion_phase_gradient(velocity, dt_s, dx_m, cfg,
                                                cs_toa_hint=toa["cs"])
        out["pg_G_prime_kpa"] = _j(pg.G_prime_kpa)
        out["pg_eta"] = _j(pg.eta)
        out["pg_ok"] = pg.ok
        out["pg_reason"] = pg.reason
        out["_dispersion_pg"] = pg
    return out


def compare_dispersion(
    gt: DispersionResult,
    pred: DispersionResult,
    cfg: DispersionConfig = DEFAULT_CONFIG,
) -> Dict:
    """Compare two dispersion curves on the intersection of their valid bands.

    Ground truth and prediction sit on different temporal grids, so both curves
    are interpolated onto a common frequency axis over the overlapping band and
    each is refit there.  Resampling one *movie* onto the other's grid would
    inject an interpolation operator into the quantity under test, which is
    exactly what this pipeline is measuring.
    """
    out = dict(ok=False, reason="", band_lo_hz=float("nan"), band_hi_hz=float("nan"),
               n_common=0, c_rmse=float("nan"), c_bias=float("nan"),
               gt_G_prime_kpa=float("nan"), gt_eta=float("nan"),
               pred_G_prime_kpa=float("nan"), pred_eta=float("nan"),
               d_G_prime_kpa=float("nan"), d_eta=float("nan"),
               rel_G_prime=float("nan"), rel_eta=float("nan"))

    if gt.freqs_hz.size == 0 or pred.freqs_hz.size == 0:
        out["reason"] = "no_bins"
        return out

    f_lo = max(gt.band_hz[0], pred.band_hz[0])
    f_hi = min(gt.band_hz[1], pred.band_hz[1])
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_hi <= f_lo:
        out["reason"] = "no_overlap"
        return out

    grid = np.arange(f_lo, f_hi + 1e-9, cfg.compare_df_hz)
    if grid.size < 3:
        grid = np.linspace(f_lo, f_hi, 3)

    c_gt = np.interp(grid, gt.freqs_hz, gt.c_phase)
    c_pr = np.interp(grid, pred.freqs_hz, pred.c_phase)
    w_gt = np.interp(grid, gt.freqs_hz, gt.weights)
    w_pr = np.interp(grid, pred.freqs_hz, pred.weights)

    fit_gt = fit_kelvin_voigt(grid, c_gt, w_gt, cfg, gt.cs_toa_hint)
    fit_pr = fit_kelvin_voigt(grid, c_pr, w_pr, cfg, pred.cs_toa_hint)
    if not (fit_gt["ok"] and fit_pr["ok"]):
        out["reason"] = "fit_failed"
        return out

    out.update(
        ok=True,
        band_lo_hz=float(f_lo), band_hi_hz=float(f_hi), n_common=int(grid.size),
        c_rmse=float(np.sqrt(np.mean((c_pr - c_gt) ** 2))),
        c_bias=float(np.mean(c_pr - c_gt)),
        gt_G_prime_kpa=fit_gt["G_pa"] / 1000.0, gt_eta=fit_gt["eta"],
        pred_G_prime_kpa=fit_pr["G_pa"] / 1000.0, pred_eta=fit_pr["eta"],
    )
    out["d_G_prime_kpa"] = out["pred_G_prime_kpa"] - out["gt_G_prime_kpa"]
    out["d_eta"] = out["pred_eta"] - out["gt_eta"]
    out["rel_G_prime"] = out["d_G_prime_kpa"] / out["gt_G_prime_kpa"] if out["gt_G_prime_kpa"] else float("nan")
    out["rel_eta"] = out["d_eta"] / out["gt_eta"] if out["gt_eta"] else float("nan")
    return out
