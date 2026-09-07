#!/usr/bin/env python3
"""
SAUS 2026 abstract — viscoelastic values for the bar panel and the CSV.

Protocol A (the one used before the MUSE Lab post-processing port): real 500 Hz
test-split inputs, 5 sparse frames, upsampled 12x by the model and compared
against a native 5 kHz acquisition of the same imaging plane.

The input is rebuilt from the RAW 500 Hz IQ at ``--input-step`` (default 1 =
five CONSECUTIVE frames, dt_in = 2 ms).  That is how v14 was trained
(``frequency_step_overrides: {"500hz": [1]}``).  The stored sparse_*.npy files
were written at step 12 (24 ms apart), so reading them — as every earlier driver
did — feeds the model input it never saw in training.  Pass --input-step 12 to
reproduce the old, mismatched numbers.  Every
viscoelastic number -- ground truth, interpolation baselines and model alike --
comes from ``src/analysis/dispersion.py``, the estimator that superseded the
2*pi-bugged ``evaluate_wavepinn_v10.py:estimate_voigt``.  The MUSE Lab port
(``src/analysis/muselab_postproc.py``) is NOT used anywhere here.

Four methods per input window:
  GT (5 kHz)   native 5 kHz acquisition, analysed on its own 0.2 ms grid
  Linear       linear temporal interpolation of the 5 sparse velocity knots
  Cubic        cubic-spline temporal interpolation of the same 5 knots
  WaveFormer   the v14 checkpoint, 5 sparse IQ frames -> 49 frames

The baselines and the model share the dense output grid dt = 2 ms / 12, so they
are estimated identically; only the input differs.

Usage:
    python scripts/generate_saus2026_values.py \
        --model outputs/wavepinn_v14/best_model.pt \
        --out outputs/saus2026
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dispersion import (  # noqa: E402
    DEFAULT_CONFIG,
    compare_dispersion,
    estimate_cs_toa,
    estimate_dispersion,
)
from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402
from src.utils.deployment_input import build_sparse_iq  # noqa: E402

DT_500HZ = 1.0 / 500.0
UPSAMPLE = 12
DT_OUT = DT_500HZ / UPSAMPLE       # 6 kHz equivalent dense grid
DT_5KHZ_GT = 1.0 / 5000.0
H_ROI = 116
N_DENSE = 49
N_SPARSE = 5

CATEGORY_LABEL = {"1ft": "1FT", "2ft": "2FT", "3ft": "3FT", "cirs": "CIRS"}
METHOD_ORDER = ["GT_5kHz", "Linear", "Cubic", "WaveFormer"]


def pad16(a: np.ndarray) -> np.ndarray:
    *lead, H, W = a.shape
    pH, pW = (16 - H % 16) % 16, (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return a
    return np.pad(a, [(0, 0)] * len(lead) + [(0, pH), (0, pW)])


def phantom_of(name: str) -> str:
    low = name.lower()
    if "cirs" in low:
        return "cirs"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}ft" if m else "other"


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  WARNING missing keys: {list(missing)[:5]}")
    print(f"  checkpoint epoch={ckpt.get('epoch', '?')} "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.5f}")
    return m


def find_velocity_500hz(stem: str, data_dir: Path) -> Optional[Path]:
    """The native 500 Hz Loupas velocity movie for a sparse input window.  The
    sparse IQ frames are native frames 0, 12, 24, 36, 48 of this movie."""
    for split in ("test", "inference", "val", "train"):
        p = data_dir / split / f"velocity_{stem}.npy"
        if p.exists():
            return p
    return None


def interp_baseline(vel_native: np.ndarray, input_step: int, T_dense: int,
                    kind: str) -> np.ndarray:
    """Temporal interpolation of exactly the frames the model receives.

    The input frames are native indices 0, step, 2*step ... ; on the dense output
    grid they land at 0, UPSAMPLE, 2*UPSAMPLE ... regardless of step, because the
    model always upsamples the input spacing by UPSAMPLE."""
    src = np.arange(N_SPARSE) * input_step
    dst = np.arange(N_SPARSE) * UPSAMPLE
    if src[-1] >= vel_native.shape[0]:
        raise ValueError("input frames run past the velocity movie")
    f = interp1d(dst.astype(float), vel_native[src], axis=0, kind=kind,
                 bounds_error=False, fill_value="extrapolate")
    return f(np.arange(T_dense, dtype=float))


@torch.no_grad()
def predict(model, iq: np.ndarray, vel_scale: float,
            device: torch.device) -> np.ndarray:
    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    inp = pad16(np.stack([(iq / iq_s).real, (iq / iq_s).imag], axis=0))
    t = torch.from_numpy(inp).unsqueeze(0).to(device)
    out = model(t, dt_seconds=torch.tensor([DT_500HZ], device=device))
    return out[0, 0].float().cpu().numpy()[:, :H_ROI, :] * vel_scale


def pair_files(eval_dir: Path) -> List[Dict]:
    """Pair each 500 Hz sparse input with every 5 kHz acquisition of the same
    imaging plane (identical pairing to scripts/evaluate_swe_v14.py)."""
    sp_re = re.compile(r"sparse_(.+?)_500Hz_(\d{8}T\d+)(?:_w(\d+))?\.npy$")
    gt_re = re.compile(r"velocity_(.+?)_5kHz_(\d{8}T\d+)(?:_w(\d+))?\.npy$")

    gts = collections.defaultdict(list)
    for f in sorted(eval_dir.glob("velocity_*5kHz*.npy")):
        m = gt_re.match(f.name)
        if m:
            gts[m.group(1)].append(f)

    pairs = []
    for f in sorted(eval_dir.glob("sparse_*500Hz*.npy")):
        m = sp_re.match(f.name)
        if not m or m.group(1) not in gts:
            continue
        pairs.append(dict(plane=m.group(1), ts=m.group(2), window=m.group(3) or "0",
                          phantom=phantom_of(m.group(1)), sparse=f,
                          stem=f.name[len("sparse_"):-len(".npy")],
                          gts=gts[m.group(1)]))
    return pairs


def analyse(field: np.ndarray, dt_s: float) -> Dict:
    toa = estimate_cs_toa(field, dt_s)
    disp = estimate_dispersion(field, dt_s, cs_toa_hint=toa["cs"])
    return dict(cs=float(toa["cs"]), cs_std=float(toa["cs_std"]),
                eta=float(disp.eta), G_kpa=float(disp.G_prime_kpa),
                r2=float(disp.r_squared), ok=bool(disp.ok),
                reason=disp.reason, band=disp.band_hz, disp=disp)


def msd(vals) -> tuple:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=0)), int(a.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/wavepinn_v14/best_model.pt")
    ap.add_argument("--eval-dir", default="outputs/eval_test_split_merged")
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--out", default="outputs/saus2026")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-step", type=int, default=1,
                    help="native-frame spacing of the 5 input frames. 1 = "
                         "consecutive (dt_in = 2 ms, matches v14 training); "
                         "12 = the old mismatched path.")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    meta = json.loads((data_dir / "metadata.json").read_text())
    entries = {e["name"]: e for split in ("test", "inference", "val", "train")
               for e in meta.get(split, [])}

    dt_in = DT_500HZ * args.input_step
    print("=" * 100)
    print("  SAUS 2026 — v14 values, pre-MUSE-Lab dispersion estimator")
    print(f"  estimator {DEFAULT_CONFIG.version}   device {device}")
    print(f"  input step {args.input_step} -> dt_in {dt_in * 1e3:.1f} ms"
          f"   ({'CONSECUTIVE, matches v14 training' if args.input_step == 1 else 'legacy'})")
    print(f"  dt_out {DT_OUT * 1e3:.4f} ms  (6 kHz equivalent)")
    print("=" * 100)
    model = load_model(args.model, device)

    pairs = pair_files(Path(args.eval_dir))
    print(f"\n{len(pairs)} sparse 500 Hz windows paired with 5 kHz ground truth\n")

    gt_cache: Dict[str, Dict] = {}
    rows: List[Dict] = []
    n_skipped = 0

    for i, p in enumerate(pairs, 1):
        gt_results = []
        for g in p["gts"]:
            key = str(g)
            if key not in gt_cache:
                vel = np.load(g).astype(np.float64)[:, :H_ROI, :]
                res = analyse(vel, DT_5KHZ_GT)
                res["scale"] = float(max(np.percentile(np.abs(vel), 99), 1.0))
                gt_cache[key] = res
            gt_results.append(gt_cache[key])

        vel_scale = float(np.mean([g["scale"] for g in gt_results]))

        entry = entries.get(p["stem"])
        if entry is None:
            print(f"  [skip] {p['stem']}: not in metadata")
            continue
        iq = build_sparse_iq(entry, step=args.input_step)
        if iq is None:
            print(f"  [skip] {p['stem']}: raw IQ unavailable at step "
                  f"{args.input_step}")
            n_skipped += 1
            continue

        fields: Dict[str, np.ndarray] = {}

        vel500 = find_velocity_500hz(p["stem"], data_dir)
        if vel500 is not None:
            v = np.load(vel500).astype(np.float64)[:N_DENSE, :H_ROI, :]
            if v.shape[0] == N_DENSE:
                fields["Linear"] = interp_baseline(v, args.input_step, N_DENSE, "linear")
                fields["Cubic"] = interp_baseline(v, args.input_step, N_DENSE, "cubic")
        else:
            print(f"  [warn] no 500 Hz velocity movie for {p['stem']}")

        fields["WaveFormer"] = predict(model, iq, vel_scale, device)

        pred_res = {k: analyse(v, DT_OUT) for k, v in fields.items()}

        for g, gpath in zip(gt_results, p["gts"]):
            rows.append(dict(
                category=CATEGORY_LABEL.get(p["phantom"], p["phantom"]),
                plane=p["plane"], window=p["window"], input_500hz=p["stem"],
                gt_file=Path(gpath).name, method="GT_5kHz",
                cs_m_s=g["cs"], eta_Pa_s=g["eta"], G_prime_kPa=g["G_kpa"],
                fit_r2=g["r2"], fit_ok=g["ok"], fit_reason=g["reason"],
                cs_abs_err=0.0, eta_abs_err=0.0, G_abs_err=0.0,
                matched_band_ok=True,
            ))
            for name, res in pred_res.items():
                cmp = compare_dispersion(g["disp"], res["disp"])
                rows.append(dict(
                    category=CATEGORY_LABEL.get(p["phantom"], p["phantom"]),
                    plane=p["plane"], window=p["window"], input_500hz=p["stem"],
                    gt_file=Path(gpath).name, method=name,
                    cs_m_s=res["cs"], eta_Pa_s=res["eta"], G_prime_kPa=res["G_kpa"],
                    fit_r2=res["r2"], fit_ok=res["ok"], fit_reason=res["reason"],
                    cs_abs_err=abs(res["cs"] - g["cs"]),
                    eta_abs_err=abs(cmp["d_eta"]) if cmp["ok"] else float("nan"),
                    G_abs_err=(abs(cmp["d_G_prime_kpa"]) if cmp["ok"]
                               else float("nan")),
                    matched_band_ok=bool(cmp["ok"]),
                ))
        print(f"  [{i:2d}/{len(pairs)}] {p['stem'][:58]:60s} "
              f"GT cs={gt_results[0]['cs']:.2f}  "
              f"WF cs={pred_res['WaveFormer']['cs']:.2f}")

    # ── per-sample CSV ───────────────────────────────────────────────────────
    field_names = ["category", "plane", "window", "input_500hz", "gt_file",
                   "method", "cs_m_s", "eta_Pa_s", "G_prime_kPa", "fit_r2",
                   "fit_ok", "fit_reason", "cs_abs_err", "eta_abs_err",
                   "G_abs_err", "matched_band_ok"]
    per_sample = out_dir / "saus2026_per_sample.csv"
    with open(per_sample, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_names)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in field_names})

    # ── summary CSV: exactly the numbers the bar panel needs ─────────────────
    summary_rows = []
    for cat in ["1FT", "2FT", "3FT", "CIRS"]:
        for meth in METHOD_ORDER:
            sel = [r for r in rows if r["category"] == cat and r["method"] == meth]
            if not sel:
                continue
            cs_m, cs_s, cs_n = msd([r["cs_m_s"] for r in sel])
            et_m, et_s, et_n = msd([r["eta_Pa_s"] for r in sel])
            g_m, g_s, g_n = msd([r["G_prime_kPa"] for r in sel])
            cse_m, _, _ = msd([r["cs_abs_err"] for r in sel])
            ete_m, _, _ = msd([r["eta_abs_err"] for r in sel])
            summary_rows.append(dict(
                category=cat, method=meth, n_windows=len(sel),
                cs_mean_m_s=cs_m, cs_sd_m_s=cs_s, cs_n=cs_n,
                eta_mean_Pa_s=et_m, eta_sd_Pa_s=et_s, eta_n=et_n,
                G_prime_mean_kPa=g_m, G_prime_sd_kPa=g_s, G_prime_n=g_n,
                cs_MAE_vs_GT=cse_m if meth != "GT_5kHz" else 0.0,
                eta_MAE_vs_GT=ete_m if meth != "GT_5kHz" else 0.0,
            ))

    summary = out_dir / "saus2026_summary.csv"
    sum_fields = ["category", "method", "n_windows", "cs_mean_m_s", "cs_sd_m_s",
                  "cs_n", "eta_mean_Pa_s", "eta_sd_Pa_s", "eta_n",
                  "G_prime_mean_kPa", "G_prime_sd_kPa", "G_prime_n",
                  "cs_MAE_vs_GT", "eta_MAE_vs_GT"]
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    with open(out_dir / "saus2026_rows.json", "w") as fh:
        json.dump(rows, fh, indent=1, default=float)

    # ── console table ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  BAR-PANEL VALUES  (mean +- SD over test-split windows)")
    print("=" * 100)
    hdr = f"  {'cat':5s} {'method':11s} {'n':>3} {'cs (m/s)':>16} {'eta (Pa.s)':>16} {'G0 (kPa)':>16}"
    for cat in ["1FT", "2FT", "3FT", "CIRS"]:
        print(f"\n{hdr}")
        for r in [r for r in summary_rows if r["category"] == cat]:
            print(f"  {r['category']:5s} {r['method']:11s} {r['n_windows']:3d} "
                  f"{r['cs_mean_m_s']:7.2f}+-{r['cs_sd_m_s']:<7.2f} "
                  f"{r['eta_mean_Pa_s']:7.2f}+-{r['eta_sd_Pa_s']:<7.2f} "
                  f"{r['G_prime_mean_kPa']:7.2f}+-{r['G_prime_sd_kPa']:<7.2f}")

    print(f"\nWrote {per_sample}")
    print(f"Wrote {summary}")
    if n_skipped:
        print(f"NOTE: {n_skipped} windows skipped (raw IQ not available locally)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
