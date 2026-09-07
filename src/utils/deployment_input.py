"""
Build the model's sparse IQ input from a raw acquisition at an arbitrary step.

The preprocessed ``sparse_<name>.npy`` files on disk were written at
``sparse_step = 12`` counted in NATIVE frames.  For a 500 Hz acquisition that
means the five input frames are 24 ms apart — but v14 was TRAINED with
``frequency_step_overrides: {"500hz": [1]}``, i.e. five *consecutive* frames
2 ms apart.  Evaluating from the stored files therefore feeds the model input it
never saw in training.

This module rebuilds the input from the raw acquisition at any step, following
``scripts/preprocess_1ft.py:process_iq_window`` exactly:

  * crop to the recorded ROI and analysis window,
  * clutter-filter by subtracting the temporal mean over the FULL window
    (not over the selected frames),
  * take ``n_sparse`` frames at the requested step,
  * trim to the velocity grid and transpose to (T, H, W).
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_RAW_ROOT = Path("500hz_raw_transfer")
H_ROI, W_ROI = 116, 160
N_SPARSE = 5

_RAW_INDEX: Optional[Dict[str, List[Path]]] = None


def resolve_raw(entry: Dict, raw_root: Path = DEFAULT_RAW_ROOT) -> Optional[Path]:
    """Locate a sample's raw acquisition.

    Some metadata entries carry an ``original_path`` from the machine the
    preprocessing was first run on, so fall back to indexing the local raw tree
    by timestamp and disambiguating on the focus/plane folders.
    """
    global _RAW_INDEX
    p = Path(entry.get("original_path", ""))
    if p.exists():
        return p
    if _RAW_INDEX is None:
        _RAW_INDEX = collections.defaultdict(list)
        if raw_root.exists():
            for f in raw_root.rglob("*.npy"):
                _RAW_INDEX[f.stem].append(f)
    cands = _RAW_INDEX.get(p.stem or "", [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    want = [x.lower() for x in p.parts[-4:-1]]
    for c in cands:
        if all(w in str(c).lower() for w in want):
            return c
    return cands[0]


def build_sparse_iq(entry: Dict, step: int, n_sparse: int = N_SPARSE,
                    h_roi: int = H_ROI, w_roi: int = W_ROI,
                    raw_root: Path = DEFAULT_RAW_ROOT) -> Optional[np.ndarray]:
    """Return the (n_sparse, H, W) complex64 input, or None if unavailable."""
    raw = resolve_raw(entry, raw_root)
    if raw is None:
        return None
    iq = np.load(raw, mmap_mode="r")
    w0, w1 = entry["window"]
    r = entry["roi"]
    iq_win = np.asarray(iq[r["min_row"]:r["max_row"] + 1,
                           r["min_col"]:r["max_col"] + 1, w0:w1])
    iq_filt = iq_win - iq_win.mean(axis=2, keepdims=True)
    idx = np.arange(n_sparse) * step
    if idx[-1] >= iq_filt.shape[2]:
        return None
    sparse = iq_filt[:h_roi, :w_roi, idx]
    return np.transpose(sparse, (2, 0, 1)).astype(np.complex64)
