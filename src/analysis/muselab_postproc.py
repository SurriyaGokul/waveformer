"""
Faithful Python port of the MUSE Lab MATLAB post-processing script
``PostProcessing_L11_5v_ARFIViscoelastography_MUSELab_20260812.m``.

Entry point is the script's line 151 ("Bandpass filter velocity data between
10 Hz and 800 Hz").  Everything before that -- Loupas 2-D autocorrelation,
12x6 median filter, 10x slow-time spline interpolation -- was already applied
when the velocity arrays in ``data/fullscale_velocity`` were built, and the
model operates in the velocity domain, so those blocks are not re-run.

From the bandpass onward the port follows the MATLAB source statement for
statement:

    butter(3) 10 Hz high-pass -> butter(3) 800 Hz low-pass  (causal `filter`)
    -> single lateral/slow-time line at the axial focal depth
    -> lateral crop from x = 0 mm outward
    -> zero-padded 2-D FFT (nx = nt = 10000)
    -> per-frequency wavenumber peak with 3-point parabolic interpolation
    -> Maksuti low-frequency cutoff (>50% jump between consecutive phase
       velocities) and high-frequency cutoff (k-space intensity < 30% of max)
    -> bounded Kelvin-Voigt fit with edge trimming until R^2 >= 0.99

Three places where the MATLAB cannot be transcribed literally are called out
below and are the ONLY deviations; each is recorded in the returned result.

    (1) ``nt`` -- the script asserts that every frequency in 10:10:800 lands on
        an exact FFT bin (``error('error finding ftidx')`` otherwise).  With
        nt = 10000 that holds only when fs/1000 is an integer.  fs = 5000 Hz
        (native 5 kHz grid) satisfies it; fs = 6000 Hz (2 ms / 12 model output)
        does not.  ``nt`` is therefore rounded to the nearest multiple of
        fs/10, which keeps the pad length within 2% of 10000 and restores exact
        10 Hz bins.  See ``nt_used``.

    (2) lateral crop -- the script crops from ``axisWidth > 0`` to
        ``axisWidth > 10`` mm.  The ROI stored in this dataset extends only to
        +7.90 mm from the transducer centre, so the crop runs to the last
        available column.  See ``lat_span_mm``.

    (3) high-frequency cutoff quadrant -- see ``hf_mode`` and the note on
        ``_hf_cutoff_verbatim`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, lfilter

# ── Acquisition geometry (matches src/analysis/dispersion.py) ─────────────────
AXIAL_MM_PX = 0.0507
LATERAL_MM_PX = 0.1013
RHO = 1000.0                      # MATLAB: rho = 1000

# ── Script constants (MATLAB lines 157-158, 284, 316-320) ────────────────────
LOWERCUT_HZ = 10.0
HIGHCUT_HZ = 800.0
FREQS_HZ = np.arange(10.0, 801.0, 10.0)      # freqsToAnalyzeHz = 10:10:800
TARGET_R2 = 0.99
FIT_STEP_HZ = 10.0
MAX_TRIM_ITERS = 50
MIN_FIT_POINTS = 12
NX_PAD = 10000
NT_PAD_NOMINAL = 10000
LATERAL_CROP_MM = 10.0                       # axisWidth > 10 in the MATLAB


@dataclass
class KVResult:
    """Result of one pass of the MUSE Lab post-processing."""

    ok: bool
    reason: str = ""
    G_kpa: float = float("nan")
    G_pa: float = float("nan")
    eta: float = float("nan")
    r2: float = float("nan")
    min_f_cutoff: float = float("nan")
    max_f_cutoff: float = float("nan")
    n_fit_points: int = 0
    trim_iters: int = 0
    # diagnostics / deviation bookkeeping
    fs_hz: float = float("nan")
    nt_used: int = 0
    focal_row: int = -1
    focal_depth_mm: float = float("nan")
    lat_span_mm: Tuple[float, float] = (float("nan"), float("nan"))
    n_lat_cols: int = 0
    hf_mode: str = ""
    freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    phvel: np.ndarray = field(default_factory=lambda: np.array([]))
    freqs_fit: np.ndarray = field(default_factory=lambda: np.array([]))
    phvel_fit: np.ndarray = field(default_factory=lambda: np.array([]))
    vhat: np.ndarray = field(default_factory=lambda: np.array([]))


# ══════════════════════════════════════════════════════════════════════════════
# MATLAB line 151-170 -- bandpass filter material velocity
# ══════════════════════════════════════════════════════════════════════════════
def bandpass_velocity(vel_axlat_t: np.ndarray, fs_hz: float) -> np.ndarray:
    """``vel_filtL`` -- causal 3rd-order Butterworth high-pass then low-pass.

    MATLAB uses ``filter`` (causal IIR), not ``filtfilt``; the transient and
    phase lag it leaves in the first samples are part of the reference result,
    so ``scipy.signal.lfilter`` is used rather than ``filtfilt``.
    """
    nyq = fs_hz / 2.0
    bh, ah = butter(3, LOWERCUT_HZ / nyq, btype="high")
    bl, al = butter(3, min(HIGHCUT_HZ / nyq, 0.999), btype="low")
    v = lfilter(bh, ah, vel_axlat_t, axis=-1)
    return lfilter(bl, al, v, axis=-1)


def _pad_length_for_exact_bins(fs_hz: float, nominal: int = NT_PAD_NOMINAL) -> int:
    """Smallest deviation from ``nt = 10000`` that puts every 10 Hz multiple on
    an exact FFT bin, which the MATLAB's ``find(roundn(ft,-2) == f)`` requires.

    Bins are spaced fs/nt, so 10 Hz is a bin iff nt is a multiple of fs/10.
    """
    unit = fs_hz / 10.0
    if abs(unit - round(unit)) > 1e-9:
        return nominal                       # cannot be made exact; caller snaps
    unit = int(round(unit))
    k = max(1, int(round(nominal / unit)))
    nt = k * unit
    return nt + unit if nt % 2 else nt       # keep even for (-nt/2 : nt/2-1)


def _absfft2_shifted(cropped: np.ndarray, nx: int, nt: int,
                     chunk: int = 512) -> np.ndarray:
    """``abs(fftshift(fft2(cropped, nx, nt)))`` without materialising the
    complex plane.

    The literal expression allocates a 10000 x 10200 complex128 array (1.6 GB).
    The time-axis transform is cheap (the input has ~80 lateral samples), so it
    is done once; the wavenumber transform is then taken a column-block at a
    time and reduced to magnitude straight away, holding one float32 plane
    (~0.4 GB) instead.
    """
    x1 = np.fft.fft(np.asarray(cropped, dtype=np.float64), n=nt, axis=1)
    order = np.fft.fftshift(np.arange(nt))          # shifted col -> raw col
    out = np.empty((nx, nt), dtype=np.float32)
    for s0 in range(0, nt, chunk):
        cols = order[s0:s0 + chunk]
        blk = np.fft.fft(x1[:, cols], n=nx, axis=0)
        out[:, s0:s0 + cols.size] = np.abs(np.fft.fftshift(blk, axes=0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MATLAB line 283-307 -- dispersion curve from the 2-D FFT
# ══════════════════════════════════════════════════════════════════════════════
def _phase_velocity(absfftplane: np.ndarray, ft: np.ndarray,
                    flat: np.ndarray) -> np.ndarray:
    """``phVel`` -- peak wavenumber per temporal frequency, parabolic refined."""
    dflat = float(np.mean(np.diff(flat)))
    phvel = np.full(FREQS_HZ.size, np.nan)

    for i, f in enumerate(FREQS_HZ):
        ftidx = int(np.argmin(np.abs(ft - f)))
        profile = absfftplane[:, ftidx]
        flatidx = int(np.argmax(profile))

        if flatidx == 0 or flatidx == profile.size - 1:
            maxspf = flat[flatidx]
        else:
            y1, y2, y3 = profile[flatidx - 1], profile[flatidx], profile[flatidx + 1]
            den = y1 - 2 * y2 + y3
            interpval = 0.0 if den == 0 else (y1 - y3) / den / 2.0
            maxspf = flat[flatidx] + dflat * interpval

        phvel[i] = np.nan if maxspf == 0 else -f / maxspf
    return phvel


# ══════════════════════════════════════════════════════════════════════════════
# MATLAB line 338-353 -- high-frequency cutoff (30% of peak k-space intensity)
# ══════════════════════════════════════════════════════════════════════════════
def _hf_cutoff_verbatim(absfftplane: np.ndarray, ft: np.ndarray,
                        flat: np.ndarray) -> float:
    """Literal transcription of the MATLAB block.

    ``absfftplane`` is (wavenumber, frequency) -- that is how the script builds
    and indexes it everywhere else (``absfftplane(:,ftidx)`` at line 293, the
    ``absfftplane_quad`` crop at line 268).  Line 328 then slices it as
    ``absfftplane(ft_idx0Hz:end, 1:flat_idx0)``, i.e. as (frequency,
    wavenumber).  With nx = nt = 10000 both dimensions are the same length so
    MATLAB raises no error, but the resulting block is the (k >= 0, f <= 0)
    quadrant read with its axes swapped.  This function reproduces that.
    """
    ft_idx0 = int(np.argmin(np.abs(ft - 0.0)))
    flat_idx0 = int(np.argmin(np.abs(flat - 0.0)))
    ft_cropped = ft[ft_idx0:]
    block = absfftplane[ft_idx0:, : flat_idx0 + 1]

    y = block.max(axis=0)
    idx_ft_max = int(np.argmax(y))
    maxvalue = float(y[idx_ft_max])
    idx_flat_max = int(np.argmax(block[:, idx_ft_max]))

    xx = block[idx_flat_max, :]
    hit = np.flatnonzero(xx >= maxvalue * 0.30)
    if hit.size == 0:
        return float("nan")
    j = int(hit[-1])
    return float(ft_cropped[j]) if j < ft_cropped.size else float(ft_cropped[-1])


def _hf_cutoff_corrected(absfftplane: np.ndarray, ft: np.ndarray,
                         flat: np.ndarray) -> float:
    """Same rule applied to the quadrant the dispersion curve is read from.

    Keeps k <= 0 and f >= 0 (the propagating quadrant used by ``phVel``), finds
    the global k-space peak, walks along frequency at the peak wavenumber and
    returns the highest frequency whose intensity is still >= 30% of the peak.
    """
    ft_idx0 = int(np.argmin(np.abs(ft - 0.0)))
    flat_idx0 = int(np.argmin(np.abs(flat - 0.0)))
    block = absfftplane[: flat_idx0 + 1, ft_idx0:]     # (k <= 0, f >= 0)
    ft_pos = ft[ft_idx0:]

    y = block.max(axis=0)
    idx_ft_max = int(np.argmax(y))
    maxvalue = float(y[idx_ft_max])
    idx_flat_max = int(np.argmax(block[:, idx_ft_max]))

    xx = block[idx_flat_max, :]
    hit = np.flatnonzero(xx >= maxvalue * 0.30)
    return float(ft_pos[int(hit[-1])]) if hit.size else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# MATLAB line 390-475 -- Kelvin-Voigt fit with edge trimming
# ══════════════════════════════════════════════════════════════════════════════
def _kv_pred(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    g, eta = x[0], x[1]
    root = np.sqrt(g ** 2 + (w * eta) ** 2)
    return np.sqrt(2.0 * (g ** 2 + (w * eta) ** 2) / (RHO * (g + root)))


def _fit_kv(freqs_fit: np.ndarray, phvel_fit: np.ndarray, g_initial: float):
    """``lsqnonlin`` + the trim loop.

    MATLAB requests ``levenberg-marquardt`` while also passing lb/ub; MATLAB
    silently falls back to trust-region-reflective in that case (LM cannot
    handle bounds), so ``least_squares(method='trf')`` is the equivalent.
    """
    lb = np.array([g_initial * 0.4, 0.0])
    ub = np.array([np.inf, np.inf])
    x0 = np.array([g_initial, 1e-8])
    x0 = np.clip(x0, lb + 1e-12, ub)

    best = dict(x=None, r2=-np.inf, f=None, v=None, vhat=None, n=0)
    f_cur, v_cur = freqs_fit.copy(), phvel_fit.copy()
    trim = 0

    while trim <= MAX_TRIM_ITERS:
        w = 2 * np.pi * f_cur
        try:
            sol = least_squares(lambda x: _kv_pred(x, w) - v_cur, x0,
                                bounds=(lb, ub), method="trf",
                                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                max_nfev=50000)
            x = sol.x
        except Exception:
            break

        vhat = _kv_pred(x, w)
        ss_res = float(np.sum((v_cur - vhat) ** 2))
        ss_tot = float(np.sum((v_cur - np.mean(v_cur)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, np.finfo(float).eps)

        if r2 > best["r2"]:
            best = dict(x=x, r2=r2, f=f_cur.copy(), v=v_cur.copy(),
                        vhat=vhat.copy(), n=trim)

        if r2 >= TARGET_R2 or f_cur.size <= MIN_FIT_POINTS:
            break

        res = np.abs(v_cur - vhat)
        n = res.size
        edge_n = max(2, int(np.floor(0.2 * n)))
        low_err = float(np.mean(res[:edge_n]))
        high_err = float(np.mean(res[-edge_n:]))
        if high_err >= low_err:
            f_cur, v_cur = f_cur[:-1], v_cur[:-1]
        else:
            f_cur, v_cur = f_cur[1:], v_cur[1:]
        trim += 1

    return best


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def muselab_postprocess(vel_thw: np.ndarray, dt_s: float, focus_mm: float,
                        roi_min_row: int, roi_min_col: int, w_full: int,
                        hf_mode: str = "corrected") -> KVResult:
    """Run the MATLAB post-processing on one velocity movie.

    Parameters
    ----------
    vel_thw       : (T, H, W) particle velocity, mm/s, ROI-cropped.
    dt_s          : slow-time step of ``vel_thw``.
    focus_mm      : SWIFocusZ_mm -- push focal depth (10 or 15 mm here).
    roi_min_row   : first axial sample of the ROI in the full image, so that
                    absolute depth (``axisDepth``) can be reconstructed.
    roi_min_col   : first lateral sample of the ROI in the full image.
    w_full        : lateral size of the full image, for the ``axisWidth``
                    origin (the MATLAB axis is distance from image centre).
    hf_mode       : "corrected" (default) or "verbatim" -- see
                    ``_hf_cutoff_verbatim``.
    """
    vel = np.asarray(vel_thw, dtype=np.float64)
    if vel.ndim != 3 or vel.shape[0] < 8:
        return KVResult(ok=False, reason="velocity movie too short")

    T, H, W = vel.shape
    fs = 1.0 / dt_s

    # (ax, lat, t) -- the MATLAB axis order for vel_filtL
    v_axlat_t = np.transpose(vel, (1, 2, 0))
    v_filt = bandpass_velocity(v_axlat_t, fs)

    # MATLAB 218: indxZFocus = find(|axisDepth - SWIFocusZ_mm| == min, 1, 'last')
    axis_depth = (roi_min_row + np.arange(H)) * AXIAL_MM_PX
    d = np.abs(axis_depth - focus_mm)
    focal_row = int(np.flatnonzero(d == d.min())[-1])

    # MATLAB 221-227: lateral/slow-time plane at the focal depth, cropped to
    # the far side of the push (axisWidth > 0 .. > 10 mm).
    b = v_filt[focal_row, :, :]                       # (lat, t)
    axis_width = (roi_min_col + np.arange(W) - w_full / 2.0) * LATERAL_MM_PX

    pos = np.flatnonzero(axis_width > 0.0)
    if pos.size < MIN_FIT_POINTS:
        return KVResult(ok=False, reason="no lateral samples beyond the push")
    lat0 = int(pos[0])
    far = np.flatnonzero(axis_width > LATERAL_CROP_MM)
    lat1 = int(far[0]) if far.size else W - 1         # deviation (2)

    cropped = b[lat0:lat1 + 1, :]
    if cropped.shape[0] < 8:
        return KVResult(ok=False, reason="lateral aperture too narrow")

    # MATLAB 250-266: zero-padded 2-D FFT
    nt = _pad_length_for_exact_bins(fs)                # deviation (1)
    nx = NX_PAD
    ft = np.arange(-nt // 2, nt // 2) * (fs / nt)
    dlat_m = LATERAL_MM_PX / 1000.0                    # mean(diff(axisWidth/1000))
    fslat = 1.0 / dlat_m
    flat = np.arange(-nx // 2, nx // 2) * (fslat / nx)

    absfftplane = _absfft2_shifted(cropped, nx, nt)

    phvel = _phase_velocity(absfftplane, ft, flat)

    base = KVResult(ok=False, fs_hz=fs, nt_used=nt, focal_row=focal_row,
                    focal_depth_mm=float(axis_depth[focal_row]),
                    lat_span_mm=(float(axis_width[lat0]), float(axis_width[lat1])),
                    n_lat_cols=cropped.shape[0], hf_mode=hf_mode,
                    freqs=FREQS_HZ.copy(), phvel=phvel)

    # MATLAB 338-375: Maksuti cutoffs, then clamping
    max_f = (_hf_cutoff_corrected(absfftplane, ft, flat) if hf_mode == "corrected"
             else _hf_cutoff_verbatim(absfftplane, ft, flat))

    with np.errstate(invalid="ignore", divide="ignore"):
        pv_diff = np.abs((phvel[1:] - phvel[:-1]) / ((phvel[1:] + phvel[:-1]) / 2.0))
    low_hit = np.flatnonzero(pv_diff < 0.5)
    min_f = float(FREQS_HZ[low_hit[0]]) if low_hit.size else float(FREQS_HZ[0])

    min_f = max(min_f, float(FREQS_HZ[0]))
    max_f = min(max_f, float(FREQS_HZ[-1])) if np.isfinite(max_f) else float(FREQS_HZ[-1])
    if max_f <= min_f:
        min_f, max_f = float(FREQS_HZ[0]), float(FREQS_HZ[-1])
    base.min_f_cutoff, base.max_f_cutoff = min_f, max_f

    # MATLAB 377-388: build the fit vectors
    freqs_fit = np.arange(np.ceil(min_f), np.floor(max_f) + 1e-9, FIT_STEP_HZ)
    if freqs_fit.size < MIN_FIT_POINTS:
        freqs_fit = np.linspace(min_f, max_f, max(MIN_FIT_POINTS, 20))
    finite = np.isfinite(FREQS_HZ) & np.isfinite(phvel)
    if finite.sum() < 2:
        base.reason = "phase velocity curve is empty"
        return base
    phvel_fit = np.interp(freqs_fit, FREQS_HZ[finite], phvel[finite])

    good = np.isfinite(freqs_fit) & np.isfinite(phvel_fit) & (phvel_fit > 0)
    freqs_fit, phvel_fit = freqs_fit[good], phvel_fit[good]
    if freqs_fit.size < 4:
        base.reason = "fewer than 4 admissible phase velocities in band"
        return base

    # MATLAB 396-409: initial guess from the mid-band phase velocity
    midfreq = min_f + (max_f - min_f) / 2.0
    idx_mid = int(np.argmin(np.abs(FREQS_HZ - midfreq)))
    phvel_mid = phvel[idx_mid]
    if not np.isfinite(phvel_mid) or phvel_mid <= 0:
        phvel_mid = float(np.median(phvel_fit))
    g_initial = RHO * phvel_mid ** 2

    best = _fit_kv(freqs_fit, phvel_fit, g_initial)
    if best["x"] is None:
        base.reason = "Kelvin-Voigt solver did not converge"
        return base

    base.ok = True
    base.G_pa = float(best["x"][0])
    base.G_kpa = base.G_pa / 1e3
    base.eta = float(best["x"][1])
    base.r2 = float(best["r2"])
    base.n_fit_points = int(best["f"].size)
    base.trim_iters = int(best["n"])
    base.freqs_fit, base.phvel_fit, base.vhat = best["f"], best["v"], best["vhat"]
    return base
