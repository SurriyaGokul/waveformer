#!/usr/bin/env python3
"""
WavePINN v14 / v14d results, post-processed with the MUSE Lab MATLAB script.

Every viscoelastic number in this report -- ground truth and prediction alike --
comes from ``src/analysis/muselab_postproc.py``, a statement-for-statement port
of ``PostProcessing_L11_5v_ARFIViscoelastography_MUSELab_20260812.m`` entered at
its bandpass stage.  The repository's own dispersion estimator
(``src/analysis/dispersion.py``) is NOT used anywhere here.

Two protocols:

  A. Cross-acquisition, frozen test split.  Real 500 Hz test inputs (dt = 2 ms)
     upsampled 12x to dt = 1/6000 s, compared against a separate native 5 kHz
     acquisition of the same imaging plane.  Deployment-realistic; acquisition-
     to-acquisition variation is folded into every difference.

  B. Matched ground truth, 5 kHz validation split.  Each native 5 kHz
     acquisition is decimated to its own 5 input frames (dt_in = 2.4 ms), pushed
     through the model, and the 49 output frames land back on that same
     acquisition's grid.  No cross-acquisition confound.  The frozen test split
     holds 500 Hz entries only, so this section is the VALIDATION split.

Usage:
    python scripts/postprocess_muselab_report.py \
        --models v14=outputs/wavepinn_v14/best_model.pt \
                 v14d=outputs/wavepinn_v14d/best_model.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.muselab_postproc import muselab_postprocess  # noqa: E402
from src.architectures.wavepinn_v10 import create_wavepinn_v10  # noqa: E402

H_ROI = 116
UPSAMPLE = 12
DT_500HZ = 1.0 / 500.0
R2_GATE = 0.90


# ── plumbing shared with the existing evaluation drivers ─────────────────────
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


def group_of(name: str) -> str:
    """Finer grouping than ``phantom_of``: the CIRS acquisitions target four
    different inclusions (10/50/60/100 kPa), so pooling them under one 'cirs'
    label hides a 10x spread in the true modulus."""
    low = name.lower()
    if "cirs" in low:
        m = re.search(r"(\d+)kpa", low)
        return f"cirs {m.group(1)}kPa" if m else "cirs"
    return phantom_of(name)


def load_model(path: str, device: torch.device, label: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}).get("model", {}))
    cfg.pop("name", None)
    m = create_wavepinn_v10(config=cfg).to(device).eval()
    missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [{label}] WARNING missing keys: {list(missing)[:5]}")
    print(f"  [{label}] {path}  epoch={ckpt.get('epoch', '?')} "
          f"val_loss={ckpt.get('best_val_loss', float('nan')):.5f}")
    return m


@torch.no_grad()
def predict(model, iq_path: Path, vel_scale: float, dt_in: float,
            device: torch.device) -> np.ndarray:
    iq = np.load(iq_path).astype(np.complex64)
    iq_s = max(float(np.percentile(np.abs(iq), 99)), 1.0)
    inp = pad16(np.stack([(iq / iq_s).real, (iq / iq_s).imag], axis=0))
    t = torch.from_numpy(inp).unsqueeze(0).to(device)
    out = model(t, dt_seconds=torch.tensor([dt_in], device=device))
    return out[0, 0].float().cpu().numpy()[:, :H_ROI, :] * vel_scale


def geometry(entry: Dict) -> Dict:
    r = entry["roi"]
    return dict(focus_mm=float(entry["focus"].replace("mmfocus", "")),
                roi_min_row=int(r["min_row"]), roi_min_col=int(r["min_col"]),
                w_full=int(entry["original_iq_shape"][1]))


def run_script(vel: np.ndarray, dt_s: float, geo: Dict, hf_mode: str):
    return muselab_postprocess(vel, dt_s, geo["focus_mm"], geo["roi_min_row"],
                               geo["roi_min_col"], geo["w_full"], hf_mode=hf_mode)


def as_row(res, prefix: str) -> Dict:
    return {f"{prefix}_G_kpa": res.G_kpa, f"{prefix}_eta": res.eta,
            f"{prefix}_R2": res.r2, f"{prefix}_ok": bool(res.ok),
            f"{prefix}_band_lo": res.min_f_cutoff,
            f"{prefix}_band_hi": res.max_f_cutoff,
            f"{prefix}_n_fit": res.n_fit_points}


# ══════════════════════════════════════════════════════════════════════════════
# Protocol A -- cross-acquisition, frozen test split
# ══════════════════════════════════════════════════════════════════════════════
def protocol_a(models, lut, eval_dir: Path, hf_mode: str,
               device: torch.device) -> List[Dict]:
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
        if m and m.group(1) in gts:
            pairs.append(dict(plane=m.group(1), ts=m.group(2),
                              window=m.group(3) or "0", sparse=f,
                              gts=gts[m.group(1)]))

    dt_out = DT_500HZ / UPSAMPLE
    gt_cache: Dict[str, Dict] = {}
    records: List[Dict] = []

    for i, p in enumerate(pairs, 1):
        sname = p["sparse"].name[len("sparse_"):-4]
        sp_entry = lut.get(sname)
        if sp_entry is None:
            print(f"  skip {sname}: no metadata entry")
            continue
        sp_geo = geometry(sp_entry)

        gt_res = []
        for g in p["gts"]:
            key = g.name
            if key not in gt_cache:
                gname = g.name[len("velocity_"):-4]
                g_entry = lut.get(gname)
                if g_entry is None:
                    print(f"  skip GT {gname}: no metadata entry")
                    continue
                vel = np.load(g).astype(np.float64)
                gt_cache[key] = dict(
                    res=run_script(vel, float(g_entry["frame_time_s"]),
                                   geometry(g_entry), hf_mode),
                    scale=max(float(np.percentile(np.abs(vel), 99)), 1.0),
                    peak=float(np.percentile(np.abs(vel), 99)))
            if key in gt_cache:
                gt_res.append((g.name, gt_cache[key]))
        if not gt_res:
            continue

        vel_scale = float(np.mean([c["scale"] for _, c in gt_res]))
        preds = {}
        for label, model in models.items():
            pv = predict(model, p["sparse"], vel_scale, DT_500HZ, device)
            preds[label] = dict(res=run_script(pv, dt_out, sp_geo, hf_mode),
                                peak=float(np.percentile(np.abs(pv), 99)))

        for gname, c in gt_res:
            row = dict(protocol="A", sample=sname, gt_file=gname,
                       phantom=phantom_of(sname), window=p["window"],
                       focus_mm=sp_geo["focus_mm"],
                       gt_peak_mms=c["peak"], **as_row(c["res"], "gt"))
            for label in models:
                row.update(as_row(preds[label]["res"], label))
                row[f"{label}_peak_mms"] = preds[label]["peak"]
            records.append(row)

        print(f"  A {i}/{len(pairs)}  {sname}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Protocol B -- matched ground truth, 5 kHz validation split
# ══════════════════════════════════════════════════════════════════════════════
def protocol_b(models, meta, data_dir: Path, split: str, hf_mode: str,
               device: torch.device) -> List[Dict]:
    entries = [e for e in meta[split]
               if e.get("frame_time_s") and "_5kHz_" in e["name"]]
    split_dir = data_dir / split
    T_dense = 1 + (5 - 1) * UPSAMPLE
    records: List[Dict] = []

    for i, e in enumerate(entries, 1):
        name = e["name"]
        sp, gp = split_dir / f"sparse_{name}.npy", split_dir / f"velocity_{name}.npy"
        if not (sp.exists() and gp.exists()):
            continue

        geo = geometry(e)
        dt_native = float(e["frame_time_s"])
        dt_in = int(e.get("sparse_step", UPSAMPLE)) * dt_native
        dt_out = dt_in / UPSAMPLE

        gt = np.load(gp).astype(np.float64)[:T_dense, :H_ROI, :]
        vel_scale = max(float(np.percentile(np.abs(gt), 99)), 1.0)
        gt_res = run_script(gt, dt_native, geo, hf_mode)

        row = dict(protocol="B", sample=name, gt_file=gp.name,
                   phantom=phantom_of(name), window="-",
                   focus_mm=geo["focus_mm"],
                   gt_peak_mms=float(np.percentile(np.abs(gt), 99)),
                   **as_row(gt_res, "gt"))
        for label, model in models.items():
            pv = predict(model, sp, vel_scale, dt_in, device)
            row.update(as_row(run_script(pv, dt_out, geo, hf_mode), label))
            row[f"{label}_peak_mms"] = float(np.percentile(np.abs(pv), 99))
        records.append(row)
        print(f"  B {i}/{len(entries)}  {name}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════
def collapse_windows(rows: List[Dict], labels: List[str]) -> List[Dict]:
    """One record per input window.

    In protocol A a plane may have been re-acquired at 5 kHz several times (the
    10 kPa CIRS inclusion has seven), and the driver emits one row per
    (window, ground-truth acquisition) pairing.  Aggregating over those rows
    would weight a window by how often its plane happened to be re-imaged and
    would repeat one prediction seven times inside a mean.  Here the ground
    truth is averaged over the pairings of a window -- the prediction is
    identical across them -- so every window counts once.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[r["sample"]].append(r)

    out = []
    for sample, group in by.items():
        base = dict(group[0])
        base["n_pairings"] = len(group)
        for fld in ("gt_G_kpa", "gt_eta", "gt_peak_mms"):
            vals = np.array([g[fld] for g in group], float)
            vals = vals[np.isfinite(vals)]
            base[fld] = float(vals.mean()) if vals.size else float("nan")
        r2 = np.array([g["gt_R2"] for g in group], float)
        r2 = r2[np.isfinite(r2)]
        base["gt_R2"] = float(r2.min()) if r2.size else float("nan")
        out.append(base)
    return sorted(out, key=lambda r: (r["phantom"], r["sample"]))


def msd(vals) -> str:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    return "     n/a      " if a.size == 0 else f"{a.mean():7.2f}+-{a.std():<6.2f}"


def agg_table(emit, rows: List[Dict], label: str, subset_name: str,
              keyfn=None, keyname: str = "phantom"):
    keyfn = keyfn or (lambda r: r["phantom"])
    emit(f"\n  MODEL {label}   [{subset_name}]")
    emit(f"    {keyname:11s} {'n':>3}  {'GT G (kPa)':>14} {'pred G (kPa)':>14} "
         f"{'G MAE':>7} {'G med|err|':>10}  {'GT eta (Pa.s)':>14} "
         f"{'pred eta':>14} {'eta MAE':>8} {'eta med|err|':>12}")
    by = collections.defaultdict(list)
    for r in rows:
        by[keyfn(r)].append(r)
    for ph in sorted(by) + ["ALL"]:
        v = rows if ph == "ALL" else by[ph]
        if not v:
            continue
        gG = np.array([r["gt_G_kpa"] for r in v], float)
        pG = np.array([r[f"{label}_G_kpa"] for r in v], float)
        ge = np.array([r["gt_eta"] for r in v], float)
        pe = np.array([r[f"{label}_eta"] for r in v], float)
        mg, me = np.isfinite(gG) & np.isfinite(pG), np.isfinite(ge) & np.isfinite(pe)
        g_mae = np.mean(np.abs(pG[mg] - gG[mg])) if mg.any() else np.nan
        e_mae = np.mean(np.abs(pe[me] - ge[me])) if me.any() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            g_rel = np.abs((pG[mg] - gG[mg]) / gG[mg]) if mg.any() else np.array([])
            e_rel = np.abs((pe[me] - ge[me]) / np.where(ge[me] == 0, np.nan, ge[me])) \
                if me.any() else np.array([])
        g_rel, e_rel = g_rel[np.isfinite(g_rel)], e_rel[np.isfinite(e_rel)]
        emit(f"    {ph:11s} {len(v):3d}  {msd(gG):>14} {msd(pG):>14} "
             f"{g_mae:7.2f} {100 * np.median(g_rel) if g_rel.size else np.nan:9.0f}% "
             f" {msd(ge):>14} {msd(pe):>14} {e_mae:8.3f} "
             f"{100 * np.median(e_rel) if e_rel.size else np.nan:11.0f}%")

    gG = np.array([r["gt_G_kpa"] for r in rows], float)
    pG = np.array([r[f"{label}_G_kpa"] for r in rows], float)
    ge = np.array([r["gt_eta"] for r in rows], float)
    pe = np.array([r[f"{label}_eta"] for r in rows], float)
    for nm, a, b in [("G'", gG, pG), ("eta", ge, pe)]:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (b[m] - a[m]) / np.where(a[m] == 0, np.nan, a[m])
            rel = rel[np.isfinite(rel)]
            r_corr = float(np.corrcoef(a[m], b[m])[0, 1]) if np.std(a[m]) > 0 else np.nan
            emit(f"    {nm:>4} : signed bias {100 * np.median(rel):+5.0f}%   "
                 f"correlation with GT r = {r_corr:+.3f}   "
                 f"(pred sd {np.std(b[m]):.3f}, GT sd {np.std(a[m]):.3f})")
    r2 = np.array([r[f"{label}_R2"] for r in rows], float)
    r2 = r2[np.isfinite(r2)]
    if r2.size:
        emit(f"    KV fit quality on predictions: median R2 = {np.median(r2):.3f}, "
             f"{int((r2 >= R2_GATE).sum())}/{r2.size} at R2 >= {R2_GATE}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["v14=outputs/wavepinn_v14/best_model.pt",
                             "v14d=outputs/wavepinn_v14d/best_model.pt"])
    ap.add_argument("--eval-dir", default="outputs/eval_test_split_merged")
    ap.add_argument("--data-dir", default="data/fullscale_velocity")
    ap.add_argument("--matched-split", default="val")
    ap.add_argument("--hf-mode", default="verbatim",
                    choices=["verbatim", "corrected"])
    ap.add_argument("--out", default="outputs/muselab_v14_report")
    ap.add_argument("--from-json", action="store_true",
                    help="re-render the text report from a previous run's "
                         "per_sample.json instead of re-running inference")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    meta = json.loads((data_dir / "metadata.json").read_text())
    lut = {e["name"]: e for sp in ["train", "val", "inference", "test"]
           for e in meta[sp]}

    print("=" * 118)
    print("  MUSE Lab MATLAB post-processing applied to WavePINN v14 / v14d")
    print(f"  hf-cutoff mode: {args.hf_mode}   device: {device}")
    print("=" * 118)

    labels = [spec.split("=", 1)[0] for spec in args.models]

    if args.from_json:
        cached = json.loads((out_dir / "per_sample.json").read_text())
        rec_a, rec_b = cached["protocol_A"], cached["protocol_B"]
        print(f"  re-rendering from {out_dir}/per_sample.json "
              f"(A n={len(rec_a)}, B n={len(rec_b)})")
    else:
        models = {}
        for spec in args.models:
            label, path = spec.split("=", 1)
            models[label] = load_model(path, device, label)
        rec_a = protocol_a(models, lut, Path(args.eval_dir), args.hf_mode, device)
        rec_b = protocol_b(models, meta, data_dir, args.matched_split,
                           args.hf_mode, device)

    lines: List[str] = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 118)
    emit("  SHEAR WAVE VISCOELASTOGRAPHY RESULTS -- WavePINN v14 and v14d")
    emit("  post-processing: PostProcessing_L11_5v_ARFIViscoelastography_MUSELab_20260812.m")
    emit("=" * 118)
    emit()
    emit("  Ground truth and prediction are pushed through the identical MATLAB")
    emit("  pipeline (10-800 Hz Butterworth -> lateral/slow-time line at the axial")
    emit("  focus -> zero-padded 2-D FFT -> Maksuti band cutoffs -> bounded")
    emit("  Kelvin-Voigt fit).  No other estimator is used.")
    emit(f"  High-frequency-cutoff mode: {args.hf_mode}.")

    # ── test-set composition ────────────────────────────────────────────────
    emit()
    emit("-" * 118)
    emit("  SAMPLE COUNTS")
    emit("-" * 118)
    test_names = [e["name"] for e in meta["test"]]
    test_ph = collections.Counter(phantom_of(n) for n in test_names)
    merged = {f.name[len("sparse_"):-4]
              for f in Path(args.eval_dir).glob("sparse_*500Hz*.npy")}
    merged_ph = collections.Counter(phantom_of(n) for n in test_names if n in merged)
    a_ph = collections.Counter(r["phantom"] for r in rec_a)
    a_windows = collections.Counter(phantom_of(s)
                                    for s in {r["sample"] for r in rec_a})
    b_ph = collections.Counter(r["phantom"] for r in rec_b)
    quiet = collections.Counter(phantom_of(r["sample"]) for r in rec_a
                                if r["gt_peak_mms"] < 1.0)

    emit()
    emit(f"  {'phantom':10s} {'(1) frozen':>11} {'(2) 5 kHz':>11} "
         f"{'(3) same-':>11} {'(4) GT':>11} {'(5) no-wave':>12} {'(6) matched':>12}")
    emit(f"  {'':10s} {'test split':>11} {'copy on':>11} {'plane GT':>11} "
         f"{'pairings':>11} {'windows in':>12} {'GT 5 kHz':>12}")
    emit(f"  {'':10s} {'500 Hz':>11} {'disk':>11} {'exists':>11} "
         f"{'(protocol A)':>11} {'(3)':>12} {'val split':>12}")
    emit("  " + "-" * 84)
    for ph in sorted(set(test_ph) | set(a_ph) | set(b_ph)):
        emit(f"  {ph:10s} {test_ph.get(ph, 0):11d} {merged_ph.get(ph, 0):11d} "
             f"{a_windows.get(ph, 0):11d} {a_ph.get(ph, 0):11d} "
             f"{quiet.get(ph, 0):12d} {b_ph.get(ph, 0):12d}")
    emit(f"  {'TOTAL':10s} {sum(test_ph.values()):11d} {sum(merged_ph.values()):11d} "
         f"{sum(a_windows.values()):11d} {len(rec_a):11d} "
         f"{sum(quiet.values()):12d} {len(rec_b):12d}")
    emit()
    emit("  (1) every 500 Hz window in the frozen test split.")
    emit("  (2) those with a copy in the merged evaluation directory.")
    emit("  (3) those whose imaging plane also has a native 5 kHz acquisition; the")
    emit("      remaining windows have no ground truth to compare against.")
    emit("  (4) one row per (window, 5 kHz acquisition) pairing -- several planes were")
    emit("      acquired at 5 kHz more than once, so (4) exceeds (3).")
    emit("  (5) windows in (3) whose ground truth peaks below 1 mm/s: the shear wave")
    emit("      sits in the w0 time window, so w1 records only the quiet tail.")
    emit("  (6) protocol B runs on the 5 kHz VALIDATION split -- the frozen test split")
    emit("      contains 500 Hz entries only, so matched ground truth is impossible")
    emit("      there.  These acquisitions were used for model selection, not training.")

    # ── aggregated results ──────────────────────────────────────────────────
    for tag, rows, title in [
            ("A", rec_a, "PROTOCOL A -- cross-acquisition, FROZEN TEST SPLIT "
                         "(500 Hz in, 6 kHz out vs separate 5 kHz acquisition)"),
            ("B", rec_b, "PROTOCOL B -- matched ground truth, 5 kHz VALIDATION SPLIT "
                         "(same acquisition, exact output grid)")]:
        if not rows:
            continue
        units = collapse_windows(rows, labels) if tag == "A" else rows
        keep = [r for r in units
                if r["gt_R2"] >= R2_GATE
                and all(r[f"{l}_R2"] >= R2_GATE for l in labels)]
        emit()
        emit("=" * 118)
        emit(f"  {title}")
        emit("=" * 118)
        if tag == "A":
            emit()
            emit(f"  Aggregated per INPUT WINDOW (n = {len(units)}), not per pairing")
            emit(f"  (n = {len(rows)}): where a plane was acquired at 5 kHz more than")
            emit("  once, the ground-truth fits are averaged over those acquisitions so")
            emit("  that a repeatedly-imaged plane does not dominate the mean.")
        emit()
        emit(f"  ALL SAMPLES  (n = {len(units)})")
        for label in labels:
            agg_table(emit, units, label, "all samples")
        emit()
        emit(f"  WELL-CONDITIONED SUBSET  (n = {len(keep)} of {len(units)}: "
             f"Kelvin-Voigt R2 >= {R2_GATE} on the ground truth AND on both models,")
        emit("  so every model is compared on an identical set of samples)")
        if keep:
            for label in labels:
                agg_table(emit, keep, label, f"R2 >= {R2_GATE}")
        else:
            emit("    (empty)")

        if any(r["phantom"] == "cirs" for r in units):
            emit()
            emit("  CIRS BROKEN OUT BY INCLUSION  (all samples) -- the four CIRS")
            emit("  inclusions span 10 to 100 kPa, which the pooled 'cirs' row hides")
            for label in labels:
                agg_table(emit, [r for r in units if r["phantom"] == "cirs"], label,
                          "all samples", keyfn=lambda r: group_of(r["sample"]),
                          keyname="inclusion")

    # ── per-sample tables ───────────────────────────────────────────────────
    for tag, rows, title in [("A", rec_a, "PROTOCOL A"), ("B", rec_b, "PROTOCOL B")]:
        if not rows:
            continue
        emit()
        emit("=" * 118)
        emit(f"  PER-SAMPLE GROUND TRUTH vs PREDICTION -- {title}")
        emit("=" * 118)
        emit()
        w_s = max(len(r["sample"]) for r in rows)
        gt_col = tag == "A"
        w_g = max((len(r["gt_file"]) - len("velocity_") - 4 for r in rows),
                  default=0) if gt_col else 0
        head = f"  {'#':>3} {'phantom':7s} {'input window':{w_s}s} "
        if gt_col:
            head += f"{'5 kHz ground truth':{w_g}s} "
        head += f"{'peak':>6} {'GT G':>8} {'GT eta':>8} {'GT R2':>6}"
        for l in labels:
            head += f" {l + ' G':>9} {l + ' eta':>9} {l + ' R2':>7}"
        emit(head)
        sub = f"  {'':3} {'':7} {'':{w_s}s} " + (f"{'':{w_g}s} " if gt_col else "")
        sub += f"{'mm/s':>6} {'kPa':>8} {'Pa.s':>8} {'':6}"
        emit(sub + "".join(f" {'kPa':>9} {'Pa.s':>9} {'':7}" for _ in labels))
        emit("  " + "-" * (len(head) - 2))
        for i, r in enumerate(sorted(rows, key=lambda x: (x["phantom"], x["sample"],
                                                          x["gt_file"])), 1):
            flag = "" if (r["gt_R2"] >= R2_GATE
                          and all(r[f"{l}_R2"] >= R2_GATE for l in labels)) else " *"
            s = f"  {i:3d} {r['phantom']:7s} {r['sample']:{w_s}s} "
            if gt_col:
                s += f"{r['gt_file'][len('velocity_'):-4]:{w_g}s} "
            s += (f"{r['gt_peak_mms']:6.2f} {r['gt_G_kpa']:8.2f} "
                  f"{r['gt_eta']:8.3f} {r['gt_R2']:6.3f}")
            for l in labels:
                s += (f" {r[f'{l}_G_kpa']:9.2f} {r[f'{l}_eta']:9.3f} "
                      f"{r[f'{l}_R2']:7.3f}")
            emit(s + flag)
        emit()
        emit(f"  * = outside the well-conditioned subset (a Kelvin-Voigt fit below "
             f"R2 {R2_GATE}).")
        emit("  'peak' is the 99th percentile of |ground-truth particle velocity|.")
        emit("  Windows near 0.1 mm/s contain no shear wave: the push sits in the w0")
        emit("  time window and w1 records only the quiet tail.  The MATLAB script has")
        emit("  no rejection rule, so it still emits a G and eta for those.")
        if gt_col:
            emit("  Rows sharing an input window differ only in which 5 kHz acquisition")
            emit("  they are scored against, so their predicted columns repeat.")

    # ── estimator validation appendix ───────────────────────────────────────
    synth = out_dir / "synthetic_validation.txt"
    emit()
    emit("=" * 118)
    emit("  HOW FAR TO TRUST THESE NUMBERS -- ESTIMATOR VALIDATION AND LIMITS")
    emit("=" * 118)
    emit()
    emit("  The 'ground truth' columns are the same MATLAB pipeline applied to a")
    emit("  higher-PRF acquisition, so they cannot certify the pipeline itself.")
    emit("  scripts/test_muselab_postproc_synthetic.py puts Kelvin-Voigt waves with")
    emit("  KNOWN G' and eta through it on this exact ROI geometry.  Findings:")
    emit()
    emit("  1. Soft media (1-3 kPa) recover to within ~20% in G' and ~19% in eta.")
    emit("  2. G' is systematically UNDER-estimated as stiffness rises: -38% at")
    emit("     6 kPa and -41% at 10 kPa, with eta over-estimated by +17 to +32%.")
    emit("     The ROI is 16.2 mm wide and the crop keeps only the 7.9 mm beyond")
    emit("     the push, so the wavenumber resolution runs out before the fit can")
    emit("     separate G' from eta.  Both models are scored against a ground")
    emit("     truth carrying the same bias, so the COMPARISON stays fair, but the")
    emit("     absolute kPa values are not calibrated.")
    emit("  3. R2 is not a fit-quality measure for elastic media.  A 20 kPa,")
    emit("     eta = 0.05 Pa.s synthetic gives G' within 6% -- a correct fit -- yet")
    emit("     R2 = 0.000, because a flat dispersion curve has no variance to")
    emit("     explain.  The 'well-conditioned' gate therefore removes genuinely")
    emit("     non-dispersive samples along with the failed ones.")
    emit("  4. The measured CIRS ground truth lands at G' = 2.2-3.3 kPa (c ~ 1.6")
    emit("     m/s) where a 10-100 kPa inclusion should give c = 3.2-10 m/s.  The")
    emit("     synthetic test recovers 20 kPa correctly on the same geometry, so")
    emit("     this is a property of the CIRS acquisitions as cropped here, not of")
    emit("     the port.  Treat every CIRS row as unresolved.")
    if synth.exists():
        emit()
        emit("  Full synthetic table: " + str(synth))
        emit()
        for ln in synth.read_text().splitlines():
            emit("  | " + ln)

    (out_dir / "muselab_report.txt").write_text("\n".join(lines))
    for tag, rows in [("A", rec_a), ("B", rec_b)]:
        if not rows:
            continue
        keys = list(rows[0])
        with open(out_dir / f"per_sample_protocol_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / "per_sample.json", "w") as fh:
        json.dump(dict(protocol_A=rec_a, protocol_B=rec_b), fh, indent=2, default=float)

    emit()
    emit(f"Wrote {out_dir}/muselab_report.txt")
    emit(f"      {out_dir}/per_sample_protocol_A.csv, per_sample_protocol_B.csv")
    emit(f"      {out_dir}/per_sample.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
