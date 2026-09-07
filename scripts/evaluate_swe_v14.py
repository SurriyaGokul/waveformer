#!/usr/bin/env python3
"""
Phase B1 — re-evaluate trained checkpoints with the REBUILT SWE estimator.

Supersedes scripts/evaluate_wavepinn_v10.py, whose Kelvin-Voigt routine had a
2*pi wavenumber unit error that made every published G'/eta number an artefact
(NaN on 63% of ground truth; eta 20-37 Pa.s where gels sit near 1 Pa.s).

Protocol: 500 Hz sparse IQ (5 frames, dt = 2 ms) -> 49 dense frames at
dt = 2 ms / 12 (6 kHz equivalent), compared against a native 5 kHz acquisition
of the same imaging plane.  Ground truth and prediction sit on different
temporal grids, so viscoelastic parameters are compared through
src.analysis.dispersion.compare_dispersion, which fits both curves on the
intersection of their valid bands rather than resampling either movie.

Known limitation carried over from the data: the 5 kHz ground truth is a
DIFFERENT acquisition (usually a different day) than the 500 Hz input, so
cross-acquisition biological/setup variation is folded into every difference
reported here.  scripts/evaluate_matched_gt.py is the confound-free protocol.

Usage:
    python scripts/evaluate_swe_v14.py --models v12=outputs/wavepinn_v12/best_model.pt
    python scripts/evaluate_swe_v14.py \
        --models v10b=SWE_PINN_outputs_wavepinn_v10b_best_model.pt \
                 v12=outputs/wavepinn_v12/best_model.pt \
                 v13=outputs/wavepinn_v13/best_model.pt
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dispersion import (  # noqa: E402
    DEFAULT_CONFIG,
    compare_dispersion,
    estimate_cs_toa,
    estimate_dispersion,
)
from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402

DT_500HZ = 1.0 / 500.0
UPSAMPLE = 12
DT_OUT = DT_500HZ / UPSAMPLE      # 6 kHz equivalent
DT_5KHZ_GT = 1.0 / 5000.0
H_ROI = 116


def pad16(a: np.ndarray) -> np.ndarray:
    *lead, H, W = a.shape
    pH, pW = (16 - H % 16) % 16, (16 - W % 16) % 16
    if pH == 0 and pW == 0:
        return a
    return np.pad(a, [(0, 0)] * len(lead) + [(0, pH), (0, pW)])


def phantom_of(name: str) -> str:
    """Actual phantom prefix.  evaluate_wavepinn_v10.py:545 labelled every
    1ft_* sample as '2ft'; this does not."""
    low = name.lower()
    if "cirs" in low:
        return "cirs"
    m = re.search(r"(\d+)ft", low)
    return f"{m.group(1)}ft" if m else "other"


def load_model(path: str, device: torch.device, label: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [{label}] WARNING missing keys: {list(missing)[:5]}")
    print(f"  [{label}] epoch={ckpt.get('epoch', '?')} "
          f"val_loss={ckpt.get('best_val_loss', float('nan')):.5f}")
    return m


@torch.no_grad()
def predict(model, iq_path: str, vel_scale: float, device: torch.device) -> np.ndarray:
    iq = np.load(iq_path).astype(np.complex64)
    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    inp = pad16(np.stack([(iq / iq_s).real, (iq / iq_s).imag], axis=0))
    t = torch.from_numpy(inp).unsqueeze(0).to(device)
    out = model(t, dt_seconds=torch.tensor([DT_500HZ], device=device))
    return out[0, 0].float().cpu().numpy()[:, :H_ROI, :] * vel_scale


def pair_files(eval_dir: Path) -> List[Dict]:
    """Pair each 500 Hz sparse input with the 5 kHz GT acquisitions of the same
    plane.  Unlike the old driver, ALL matching GT acquisitions are kept (not
    just gts[0]) and the w0/w1 windows are tracked so they can be aggregated at
    acquisition level instead of double counted."""
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
        if not m:
            continue
        plane, ts, win = m.group(1), m.group(2), m.group(3) or "0"
        if plane not in gts:
            continue
        pairs.append(dict(plane=plane, ts=ts, window=win, phantom=phantom_of(plane),
                          sparse=f, gts=gts[plane]))
    return pairs


def summarize(vals) -> str:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return "    n/a   "
    return f"{np.mean(a):6.2f}+-{np.std(a):<5.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--eval-dir", default="outputs/eval_test_split_merged")
    ap.add_argument("--out", default="outputs/swe_v14_eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("  Phase B1 — SWE evaluation with the rebuilt dispersion estimator")
    print(f"  estimator config: {DEFAULT_CONFIG.version}   device: {device}")
    print("=" * 104)

    models = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        models[label] = load_model(path, device, label)

    pairs = pair_files(Path(args.eval_dir))
    print(f"\n{len(pairs)} input windows paired with 5 kHz ground truth\n")

    gt_cache: Dict[str, Dict] = {}
    records: List[Dict] = []

    for i, p in enumerate(pairs, 1):
        # Ground truth: analyse every matching acquisition, keep them all.
        gt_results = []
        for g in p["gts"]:
            key = str(g)
            if key not in gt_cache:
                vel = np.load(g).astype(np.float64)
                toa = estimate_cs_toa(vel, DT_5KHZ_GT)
                disp = estimate_dispersion(vel, DT_5KHZ_GT, cs_toa_hint=toa["cs"])
                gt_cache[key] = dict(cs=toa["cs"], disp=disp, scale=float(
                    max(np.percentile(np.abs(vel), 99), 1.0)))
            gt_results.append(gt_cache[key])

        vel_scale = float(np.mean([g["scale"] for g in gt_results]))

        for label, model in models.items():
            pred = predict(model, str(p["sparse"]), vel_scale, device)
            toa_p = estimate_cs_toa(pred, DT_OUT)
            disp_p = estimate_dispersion(pred, DT_OUT, cs_toa_hint=toa_p["cs"])

            for g, gpath in zip(gt_results, p["gts"]):
                cmp = compare_dispersion(g["disp"], disp_p)
                records.append(dict(
                    plane=p["plane"], phantom=p["phantom"], window=p["window"],
                    ts=p["ts"], gt_file=Path(gpath).name, model=label,
                    cs_gt=g["cs"], cs_pred=toa_p["cs"],
                    cs_abs_err=abs(toa_p["cs"] - g["cs"]),
                    gt_G=g["disp"].G_prime_kpa, gt_eta=g["disp"].eta,
                    gt_ok=g["disp"].ok,
                    pred_G=disp_p.G_prime_kpa, pred_eta=disp_p.eta,
                    pred_ok=disp_p.ok, pred_reason=disp_p.reason,
                    **{f"m_{k}": v for k, v in cmp.items()},
                ))
        if i % 10 == 0:
            print(f"  ... {i}/{len(pairs)} windows")

    # ── Report ───────────────────────────────────────────────────────────────
    lines: List[str] = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 104)
    emit("  MATCHED-BAND VISCOELASTIC COMPARISON  (GT and prediction fit on their")
    emit("  overlapping frequency band; acquisition-level aggregation)")
    emit("=" * 104)

    for label in models:
        emit(f"\nMODEL: {label}")
        emit(f"  {'phantom':8s} {'n':>3} {'cs MAE':>9} {'GT eta':>14} {'pred eta':>14} "
             f"{'eta MAE':>9} {'GT G':>14} {'pred G':>14} {'G MAE':>8}")
        rows = [r for r in records if r["model"] == label]
        by_ph = collections.defaultdict(list)
        for r in rows:
            by_ph[r["phantom"]].append(r)
        by_ph["ALL"] = rows
        for ph in list(sorted(k for k in by_ph if k != "ALL")) + ["ALL"]:
            v = by_ph[ph]
            ok = [r for r in v if r["m_ok"]]
            if not ok:
                emit(f"  {ph:8s} {len(v):3d}   (no valid matched-band fits)")
                continue
            cs_mae = np.nanmean([r["cs_abs_err"] for r in v])
            eta_mae = np.nanmean([abs(r["m_d_eta"]) for r in ok])
            g_mae = np.nanmean([abs(r["m_d_G_prime_kpa"]) for r in ok])
            emit(f"  {ph:8s} {len(ok):3d} {cs_mae:9.3f} "
                 f"{summarize([r['m_gt_eta'] for r in ok]):>14} "
                 f"{summarize([r['m_pred_eta'] for r in ok]):>14} {eta_mae:9.3f} "
                 f"{summarize([r['m_gt_G_prime_kpa'] for r in ok]):>14} "
                 f"{summarize([r['m_pred_G_prime_kpa'] for r in ok]):>14} {g_mae:8.3f}")

        ok = [r for r in rows if r["m_ok"]]
        if ok:
            re_ = np.array([r["m_rel_eta"] for r in ok], dtype=float)
            rg = np.array([r["m_rel_G_prime"] for r in ok], dtype=float)
            re_, rg = re_[np.isfinite(re_)], rg[np.isfinite(rg)]
            emit(f"  matched-band fits: {len(ok)}/{len(rows)}   "
                 f"median relative error: eta {100 * np.median(np.abs(re_)):.0f}%  "
                 f"G' {100 * np.median(np.abs(rg)):.0f}%")
            emit(f"  median signed bias: eta {100 * np.median(re_):+.0f}%  "
                 f"G' {100 * np.median(rg):+.0f}%   "
                 f"(negative = model under-estimates)")
            crm = np.array([r["m_c_rmse"] for r in ok], dtype=float)
            emit(f"  dispersion-curve RMSE |c_pred - c_GT|: "
                 f"{np.nanmean(crm):.3f} m/s")

        # Does the model track viscosity ACROSS phantoms, or output a constant?
        pe = np.array([r["m_pred_eta"] for r in ok], dtype=float)
        ge = np.array([r["m_gt_eta"] for r in ok], dtype=float)
        m = np.isfinite(pe) & np.isfinite(ge)
        if m.sum() > 2:
            r_eta = float(np.corrcoef(ge[m], pe[m])[0, 1])
            emit(f"  eta correlation across samples r = {r_eta:+.3f}   "
                 f"(pred eta spread sd = {np.std(pe[m]):.3f}, "
                 f"GT spread sd = {np.std(ge[m]):.3f})")

    bad = collections.Counter(r["pred_reason"] for r in records if not r["pred_ok"])
    if bad:
        emit(f"\nprediction dispersion failures: {dict(bad)}")

    (out_dir / "swe_v14_report.txt").write_text("\n".join(lines))
    with open(out_dir / "per_sample.json", "w") as fh:
        json.dump(records, fh, indent=2, default=float)
    emit(f"\nWrote {out_dir}/swe_v14_report.txt and per_sample.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
