# `matlab/`

Reference MATLAB script for the pre-Python (Verasonics-side) acquisition pipeline —
kept for documentation/reproducibility since the Python pipeline (`src/utils/loupas.py`
+ `scripts/preprocess_*.py`) reimplements the same math.

| File | Purpose |
|---|---|
| `IQ2VEl.m` | Reference implementation: raw Verasonics I/Q buffers → B-mode display → 2-D autocorrelation (Loupas et al., 1995) axial displacement → median filtering → slow-time interpolation → finite-difference velocity → bandpass filtering → velocity movie. This is the MATLAB-side origin of the Loupas velocity estimation reimplemented in Python at `src/utils/loupas.py`. |

## Usage

Run in MATLAB after loading a raw Verasonics acquisition workspace (providing
`IBuffer`, `QBuffer`, `na`, `SWIFocusX`, `Trans`, `Resource`, `DetectPRP`,
`Timeafterpush`, `axisWidth`, `axisDepth` in the workspace — these come from the
Verasonics acquisition `.mat` file, not included in this repo). Requires the Signal
Processing Toolbox (`butter`, `filter`, `medfilt2`).

This script is a working reference/scratch file, not a packaged function — read it
top-to-bottom and adapt the parameters (`M`, `N`, kernel sizes, filter cutoffs) to your
acquisition before running.
