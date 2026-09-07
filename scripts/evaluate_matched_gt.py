#!/usr/bin/env python3
"""
Phase B2/B3 — matched ground-truth evaluation (the PRIMARY protocol).

Every other evaluation in this repo compares a 500 Hz input against a 5 kHz
acquisition recorded at a different time, so acquisition-to-acquisition
variation is folded into every reported error.  This protocol removes that
confound entirely.

For a native 5 kHz acquisition stored with sparse_step = 12:

    dt_in  = 12 * 0.2 ms = 2.4 ms   (417 Hz effective input PRF, deployment-like
                                     and inside the trained dt-FiLM range)
    dt_out = dt_in / 12  = 0.2 ms   (EXACTLY the native grid)

so the model's 49 output frames land frame-for-frame on the same acquisition's
own 49 ground-truth frames.  No temporal resampling, no cross-acquisition
mismatch, and the model's input is the very data the ground truth came from.

Baselines run through the identical path: linear and cubic-spline temporal
interpolation of the same 5 input velocity frames.  Linear interpolation is the
specific thing the model must beat, because it is both the architecture's
zero-effort solution and (for 500 Hz samples) the target it was trained against.

Also reports temporal power spectra (Phase B3): if the model merely smooths, its
spectrum collapses onto the linear baseline above a few hundred Hz.

Usage:
    python scripts/evaluate_matched_gt.py --models v12=outputs/wavepinn_v12/best_model.pt
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
    depth_average,
    estimate_cs_toa,
    estimate_dispersion,
)
from src.analysis.dispersion import _tukey  # noqa: E402
from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402

H_ROI = 116
UPSAMPLE = 12


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


def load_model(path: str, device: torch.device, label: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [{label}] WARNING missing keys: {list(missing)[:5]}")
    print(f"  [{label}] epoch={ckpt.get('epoch', '?')} "
          f"val_loss={ckpt.get('best_val_loss', float('nan')):.5f}")
    return m


def interp_baseline(vel_native: np.ndarray, knots: np.ndarray, T_dense: int,
                    kind: str) -> np.ndarray:
    """Temporal interpolation of the ground-truth frames at the INPUT sample
    times — i.e. exactly the information the model receives."""
    from scipy.interpolate import interp1d
    T, H, W = vel_native.shape
    y = vel_native[knots]                              # (5, H, W)
    f = interp1d(knots.astype(float), y, axis=0, kind=kind,
                 bounds_error=False, fill_value="extrapolate")
    return f(np.arange(T_dense, dtype=float))


def temporal_spectrum(vel: np.ndarray, dt_s: float, cfg=DEFAULT_CONFIG):
    """ROI-averaged temporal power spectrum of the depth-averaged field."""
    v = depth_average(np.asarray(vel, dtype=np.float64), cfg.depth_frac)
    v = v - v.mean(axis=0, keepdims=True)
    v = v * _tukey(v.shape[0], cfg.tukey_alpha)[:, None]
    P = (np.abs(np.fft.rfft(v, axis=0)) ** 2).sum(axis=1)
    f = np.fft.rfftfreq(v.shape[0], dt_s)
    return f, P


def band_power(f, P, lo, hi) -> float:
    m = (f >= lo) & (f <= hi)
    return float(P[m].sum()) if np.any(m) else float("nan")


def band_coherence(pred: np.ndarray, gt: np.ndarray, dt_s: float,
                   lo: float, hi: float, cfg=DEFAULT_CONFIG) -> float:
    """Correlation between prediction and ground truth AFTER band-passing both
    to [lo, hi] Hz in time.

    Raw band power is not enough: piecewise-linear interpolation puts a kink at
    every knot, and kinks radiate broadband energy, so a pure smoother can post
    a respectable high-frequency power ratio while carrying no correct
    high-frequency structure at all.  This metric asks whether the energy is in
    the right place, which is what actually matters for dispersion.
    """
    p = depth_average(np.asarray(pred, dtype=np.float64), cfg.depth_frac)
    g = depth_average(np.asarray(gt, dtype=np.float64), cfg.depth_frac)
    n = min(p.shape[0], g.shape[0])
    p, g = p[:n] - p[:n].mean(0, keepdims=True), g[:n] - g[:n].mean(0, keepdims=True)
    f = np.fft.rfftfreq(n, dt_s)
    keep = ((f >= lo) & (f <= hi)).astype(float)[:, None]
    pb = np.fft.irfft(np.fft.rfft(p, axis=0) * keep, n=n, axis=0).ravel()
    gb = np.fft.irfft(np.fft.rfft(g, axis=0) * keep, n=n, axis=0).ravel()
    if pb.std() < 1e-30 or gb.std() < 1e-30:
        return float("nan")
    return float(np.corrcoef(pb, gb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--split", default="val")
    ap.add_argument("--prf", default="5kHz", choices=["5kHz", "10kHz", "20kHz"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/matched_gt_eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.data_dir) / "metadata.json").read_text())
    entries = [e for e in meta[args.split]
               if e.get("frame_time_s") and f"_{args.prf}_" in e["name"]]
    if args.limit:
        entries = entries[:args.limit]

    print("=" * 104)
    print("  Phase B2 — MATCHED ground-truth evaluation (same acquisition, exact grid)")
    print(f"  {args.prf} {args.split} split, {len(entries)} acquisitions, device {device}")
    print("=" * 104)

    models = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        models[label] = load_model(path, device, label)

    records: List[Dict] = []
    spectra = collections.defaultdict(list)
    split_dir = Path(args.data_dir) / args.split

    for i, e in enumerate(entries, 1):
        name = e["name"]
        sp = split_dir / f"sparse_{name}.npy"
        gp = split_dir / f"velocity_{name}.npy"
        if not (sp.exists() and gp.exists()):
            continue

        step = int(e.get("sparse_step", UPSAMPLE))
        dt_native = float(e["frame_time_s"])
        dt_in = step * dt_native
        dt_out = dt_in / UPSAMPLE
        exact_grid = abs(dt_out - dt_native) < 1e-12

        gt = np.load(gp).astype(np.float64)[:, :H_ROI, :]
        T_dense = 1 + (5 - 1) * UPSAMPLE
        gt = gt[:T_dense]
        knots = np.arange(5) * UPSAMPLE          # dense indices of the input frames

        iq = np.load(sp).astype(np.complex64)
        iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
        vel_scale = max(float(np.percentile(np.abs(gt), 99)), 1.0)

        toa_gt = estimate_cs_toa(gt, dt_native)
        disp_gt = estimate_dispersion(gt, dt_native, cs_toa_hint=toa_gt["cs"])
        f_gt, P_gt = temporal_spectrum(gt, dt_native)
        spectra["GT"].append((f_gt, P_gt))

        fields: Dict[str, np.ndarray] = {}
        for kind, tag in [("linear", "linear"), ("cubic", "spline")]:
            try:
                fields[tag] = interp_baseline(gt, knots, T_dense, kind)
            except Exception:
                pass

        inp = pad16(np.stack([(iq / iq_s).real, (iq / iq_s).imag], axis=0))
        t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            for label, model in models.items():
                out = model(t, dt_seconds=torch.tensor([dt_in], device=device))
                fields[label] = (out[0, 0].float().cpu().numpy()[:, :H_ROI, :]
                                 * vel_scale)

        for tag, pred in fields.items():
            n = min(pred.shape[0], gt.shape[0])
            p, g = pred[:n], gt[:n]
            denom = float(np.mean(np.abs(g))) or 1.0
            l1 = float(np.mean(np.abs(p - g)))

            # Correlation over frames BETWEEN the input knots: the frames the
            # model actually had to invent.  Knot frames are given away by the
            # input, so including them flatters every method equally.
            interior = np.setdiff1d(np.arange(n), knots)
            gi, pi_ = g[interior].ravel(), p[interior].ravel()
            r_int = float(np.corrcoef(gi, pi_)[0, 1]) if gi.size > 2 else float("nan")
            l1_int = float(np.mean(np.abs(p[interior] - g[interior])))

            toa_p = estimate_cs_toa(p, dt_out)
            disp_p = estimate_dispersion(p, dt_out, cs_toa_hint=toa_p["cs"])
            cmp = compare_dispersion(disp_gt, disp_p)

            f_p, P_p = temporal_spectrum(p, dt_out)
            spectra[tag].append((f_p, P_p))

            records.append(dict(
                name=name, phantom=phantom_of(name), method=tag,
                exact_grid=exact_grid, dt_in=dt_in, dt_out=dt_out,
                l1=l1, l1_rel=l1 / denom, l1_interior=l1_int,
                r_interior=r_int,
                cs_gt=toa_gt["cs"], cs_pred=toa_p["cs"],
                cs_abs_err=abs(toa_p["cs"] - toa_gt["cs"]),
                gt_eta=disp_gt.eta, gt_G=disp_gt.G_prime_kpa, gt_ok=disp_gt.ok,
                pred_eta=disp_p.eta, pred_G=disp_p.G_prime_kpa, pred_ok=disp_p.ok,
                bp_250_1500=band_power(f_p, P_p, 250, 1500),
                bp_gt_250_1500=band_power(f_gt, P_gt, 250, 1500),
                coh_250_1500=band_coherence(p, g, dt_out, 250, 1500),
                coh_100_250=band_coherence(p, g, dt_out, 100, 250),
                **{f"m_{k}": v for k, v in cmp.items()},
            ))
        if i % 10 == 0:
            print(f"  ... {i}/{len(entries)}")

    # ── Report ───────────────────────────────────────────────────────────────
    lines: List[str] = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    methods = list(dict.fromkeys(r["method"] for r in records))
    emit()
    emit("=" * 104)
    emit(f"  MATCHED-GT RESULTS   ({args.prf} {args.split}, "
         f"n={len(set(r['name'] for r in records))} acquisitions)")
    emit(f"  exact output grid: {all(r['exact_grid'] for r in records)}")
    emit("=" * 104)
    emit()
    emit("  'interior' = the frames strictly between input samples, i.e. the ones")
    emit("  the model had to synthesise.  Knot frames are handed to every method.")
    emit()
    emit(f"  {'method':10s} {'L1':>9} {'L1 interior':>12} {'r interior':>11} "
         f"{'cs MAE':>8} {'eta MAE':>9} {'G MAE':>8} {'HFpow':>7} "
         f"{'coh 100-250':>12} {'coh 250-1500':>13}")
    for meth in methods:
        v = [r for r in records if r["method"] == meth]
        ok = [r for r in v if r["m_ok"]]
        hf = np.array([r["bp_250_1500"] / r["bp_gt_250_1500"]
                       for r in v if r["bp_gt_250_1500"]], dtype=float)
        emit(f"  {meth:10s} {np.nanmean([r['l1'] for r in v]):9.4f} "
             f"{np.nanmean([r['l1_interior'] for r in v]):12.4f} "
             f"{np.nanmean([r['r_interior'] for r in v]):11.3f} "
             f"{np.nanmean([r['cs_abs_err'] for r in v]):8.3f} "
             f"{(np.nanmean([abs(r['m_d_eta']) for r in ok]) if ok else float('nan')):9.3f} "
             f"{(np.nanmean([abs(r['m_d_G_prime_kpa']) for r in ok]) if ok else float('nan')):8.3f} "
             f"{np.nanmedian(hf) if hf.size else float('nan'):7.3f} "
             f"{np.nanmean([r['coh_100_250'] for r in v]):12.3f} "
             f"{np.nanmean([r['coh_250_1500'] for r in v]):13.3f}")

    emit()
    emit("  HFpow = predicted / ground-truth temporal power in 250-1500 Hz -- an ENERGY")
    emit("  ratio only.  Piecewise-linear interpolation puts a kink at every knot and")
    emit("  kinks radiate broadband energy, so a pure smoother scores respectably here")
    emit("  while carrying no correct structure.  'coh' is the band-passed correlation")
    emit("  with ground truth and is the metric that actually distinguishes them.")

    emit()
    emit("-" * 104)
    emit("VISCOELASTIC AGREEMENT (matched band, same acquisition)")
    emit("-" * 104)
    for meth in methods:
        ok = [r for r in records if r["method"] == meth and r["m_ok"]]
        if not ok:
            emit(f"  {meth:10s}  (no valid fits)")
            continue
        re_ = np.array([r["m_rel_eta"] for r in ok], dtype=float)
        rg = np.array([r["m_rel_G_prime"] for r in ok], dtype=float)
        re_, rg = re_[np.isfinite(re_)], rg[np.isfinite(rg)]
        pe = np.array([r["m_pred_eta"] for r in ok], dtype=float)
        ge = np.array([r["m_gt_eta"] for r in ok], dtype=float)
        m = np.isfinite(pe) & np.isfinite(ge)
        r_eta = float(np.corrcoef(ge[m], pe[m])[0, 1]) if m.sum() > 2 else float("nan")
        emit(f"  {meth:10s} n={len(ok):3d}  "
             f"eta: median|err| {100 * np.median(np.abs(re_)):5.0f}%  "
             f"bias {100 * np.median(re_):+5.0f}%  r={r_eta:+.3f}   |   "
             f"G': median|err| {100 * np.median(np.abs(rg)):5.0f}%  "
             f"bias {100 * np.median(rg):+5.0f}%")

    emit()
    emit("-" * 104)
    emit("PER-PHANTOM eta (matched band)")
    emit("-" * 104)
    phs = sorted(set(r["phantom"] for r in records))
    emit(f"  {'phantom':8s} {'n':>3} {'GT eta':>12} " +
         " ".join(f"{m:>12}" for m in methods))
    for ph in phs:
        row = [f"  {ph:8s}"]
        sub = [r for r in records if r["phantom"] == ph and r["m_ok"]]
        if not sub:
            continue
        n = len(set(r["name"] for r in sub))
        gt_eta = np.nanmean([r["m_gt_eta"] for r in sub])
        row.append(f"{n:3d} {gt_eta:12.3f}")
        for meth in methods:
            s = [r["m_pred_eta"] for r in sub if r["method"] == meth]
            row.append(f"{np.nanmean(s) if s else float('nan'):12.3f}")
        emit(" ".join(row))

    # Mean spectra for the figure
    spec_out = {}
    for tag, lst in spectra.items():
        if not lst:
            continue
        f0 = lst[0][0]
        P = np.mean([np.interp(f0, f, p) for f, p in lst], axis=0)
        spec_out[tag] = dict(f_hz=f0.tolist(), power=P.tolist())

    (out_dir / "matched_gt_report.txt").write_text("\n".join(lines))
    with open(out_dir / "per_sample.json", "w") as fh:
        json.dump(records, fh, indent=2, default=float)
    with open(out_dir / "mean_spectra.json", "w") as fh:
        json.dump(spec_out, fh)
    emit(f"\nWrote {out_dir}/matched_gt_report.txt, per_sample.json, mean_spectra.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
