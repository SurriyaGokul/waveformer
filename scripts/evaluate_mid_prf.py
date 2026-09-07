#!/usr/bin/env python3
"""
Phase D — the headline deliverable: low PRF in, MID PRF out (2-3 kHz).

The model maps 5 input frames to 49 output frames at dt_in/12, so a 500 Hz
input gives a 6 kHz output.  Mid-PRF outputs are taken by integer decimation of
that 6 kHz field:

    6 kHz / 3 -> 2 kHz   (every 3rd frame)
    6 kHz / 2 -> 3 kHz   (every 2nd frame)

Decimation rather than retraining at upsampling_factor 4 or 6 is deliberate:
the factors divide exactly so no interpolation is introduced, one checkpoint
serves every output rate, and -- critically -- the alternative path
(scripts/preprocess_higher_upsampling.py) builds INTERPOLATED dense targets,
which would reintroduce the fake-supervision defect V14 exists to remove.

This script verifies the claim that decimation is lossless for the quantities
that matter, by estimating cs / G' / eta at 6, 3 and 2 kHz and checking they
agree, and it reports mid-PRF accuracy against matched ground truth.

Usage:
    python scripts/evaluate_mid_prf.py --models v12=outputs/wavepinn_v12/best_model.pt
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
    compare_dispersion,
    estimate_cs_toa,
    estimate_dispersion,
)
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


def load_model(path: str, device, label: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    m.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  [{label}] epoch={ckpt.get('epoch', '?')}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="outputs/mid_prf_eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.data_dir) / "metadata.json").read_text())
    entries = [e for e in meta[args.split]
               if e.get("frame_time_s") and "_5kHz_" in e["name"]]

    print("=" * 100)
    print("  Phase D — mid-PRF output (2 / 3 kHz) by exact decimation of the 6 kHz field")
    print(f"  {len(entries)} matched-GT acquisitions, device {device}")
    print("=" * 100)

    models = {label: load_model(path, device, label)
              for label, path in (s.split("=", 1) for s in args.models)}

    split_dir = Path(args.data_dir) / args.split
    records: List[Dict] = []

    for i, e in enumerate(entries, 1):
        name = e["name"]
        sp, gp = split_dir / f"sparse_{name}.npy", split_dir / f"velocity_{name}.npy"
        if not (sp.exists() and gp.exists()):
            continue

        dt_native = float(e["frame_time_s"])
        step = int(e.get("sparse_step", UPSAMPLE))
        dt_in = step * dt_native
        dt_out = dt_in / UPSAMPLE

        T_dense = 1 + 4 * UPSAMPLE
        gt = np.load(gp).astype(np.float64)[:T_dense, :H_ROI, :]
        iq = np.load(sp).astype(np.complex64)
        iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
        vel_scale = max(float(np.percentile(np.abs(gt), 99)), 1.0)

        inp = pad16(np.stack([(iq / iq_s).real, (iq / iq_s).imag], axis=0))
        t = torch.from_numpy(inp).unsqueeze(0).to(device)

        # Fixed reference: the FULL-RATE ground truth, analysed once.  Comparing
        # a decimated prediction against a decimated ground truth moves both
        # sides of the comparison at once (the estimator's band follows the
        # Nyquist), which hides what deployment actually cares about: how much
        # viscoelastic accuracy is lost by shipping a mid-PRF product.
        toa_g_full = estimate_cs_toa(gt, dt_out)
        dg_full = estimate_dispersion(gt, dt_out, cs_toa_hint=toa_g_full["cs"])

        for label, model in models.items():
            with torch.no_grad():
                out = model(t, dt_seconds=torch.tensor([dt_in], device=device))
            pred = out[0, 0].float().cpu().numpy()[:, :H_ROI, :] * vel_scale

            for dec in (1, 2, 3):
                tag = f"{1.0 / (dt_out * dec) / 1000:.1f}kHz"
                p = pred[::dec]
                g = gt[::dec]
                dtd = dt_out * dec
                toa_p = estimate_cs_toa(p, dtd)
                toa_g = estimate_cs_toa(g, dtd)
                dp = estimate_dispersion(p, dtd, cs_toa_hint=toa_p["cs"])
                dg = estimate_dispersion(g, dtd, cs_toa_hint=toa_g["cs"])
                cmp = compare_dispersion(dg, dp)
                cmp_ref = compare_dispersion(dg_full, dp)
                records.append(dict(
                    name=name, phantom=phantom_of(name), model=label,
                    decim=dec, rate_tag=tag, out_rate_hz=1.0 / dtd,
                    n_frames=p.shape[0],
                    cs_gt=toa_g["cs"], cs_pred=toa_p["cs"],
                    cs_abs_err=abs(toa_p["cs"] - toa_g["cs"]),
                    cs_gt_full=toa_g_full["cs"],
                    cs_abs_err_ref=abs(toa_p["cs"] - toa_g_full["cs"]),
                    **{f"m_{k}": v for k, v in cmp.items()},
                    **{f"ref_{k}": v for k, v in cmp_ref.items()},
                ))
        if i % 10 == 0:
            print(f"  ... {i}/{len(entries)}")

    lines: List[str] = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 100)
    emit("  OUTPUT-RATE SWEEP (input dt = 2.4 ms, i.e. 417 Hz effective PRF)")
    emit("=" * 100)
    emit()
    emit("  The decimated field is literally the full-rate field subsampled, so any")
    emit("  change below is a change in what the ANALYSIS can see, not in the")
    emit("  reconstruction: the estimator's upper frequency limit tracks the output")
    emit("  Nyquist, so a mid-PRF product narrows the viscoelastic analysis band.")
    emit()
    emit("  'vs decimated GT' moves both sides of the comparison at once.")
    emit("  'vs FULL-RATE GT' holds the reference fixed and is the deployment")
    emit("  question: what is lost by shipping 2-3 kHz instead of the full rate?")
    emit()
    for label in models:
        emit(f"MODEL: {label}")
        emit(f"  {'':>10} {'':>7} |{'-- vs decimated GT --':^28}|"
             f"{'-- vs FULL-RATE GT --':^30}|")
        emit(f"  {'out rate':>10} {'frames':>7} {'cs MAE':>9} {'eta MAE':>9} {'G MAE':>8} "
             f"{'cs MAE':>9} {'eta MAE':>9} {'G MAE':>8} {'n fit':>6}")
        for dec in (1, 2, 3):
            v = [r for r in records if r["model"] == label and r["decim"] == dec]
            ok = [r for r in v if r["m_ok"]]
            okr = [r for r in v if r["ref_ok"]]
            if not v:
                continue

            def mn(rows, key):
                return np.nanmean([abs(r[key]) for r in rows]) if rows else float("nan")

            emit(f"  {v[0]['out_rate_hz']:9.0f}Hz {v[0]['n_frames']:7d} "
                 f"{np.nanmean([r['cs_abs_err'] for r in v]):9.3f} "
                 f"{mn(ok, 'm_d_eta'):9.3f} {mn(ok, 'm_d_G_prime_kpa'):8.3f} "
                 f"{np.nanmean([r['cs_abs_err_ref'] for r in v]):9.3f} "
                 f"{mn(okr, 'ref_d_eta'):9.3f} {mn(okr, 'ref_d_G_prime_kpa'):8.3f} "
                 f"{len(okr):6d}")
        emit()
        emit(f"  Reference (full-rate GT): cs = "
             f"{np.nanmean([r['cs_gt_full'] for r in records if r['model'] == label]):.3f} m/s, "
             f"eta = {np.nanmean([r['ref_gt_eta'] for r in records if r['model'] == label and r['ref_ok']]):.3f} Pa.s")
        emit()

    (out_dir / "mid_prf_report.txt").write_text("\n".join(lines))
    with open(out_dir / "per_sample.json", "w") as fh:
        json.dump(records, fh, indent=2, default=float)
    emit(f"Wrote {out_dir}/mid_prf_report.txt and per_sample.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
