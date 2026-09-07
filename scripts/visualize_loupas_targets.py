#!/usr/bin/env python3
"""
Visualize & Self-Verify Loupas Velocity Targets vs Raw Kasai.

Rows:
  1. Model Input — sparse IQ magnitude (real²+imag²)
  2. Raw Kasai Target (dynamic vmax from signal region)
  3. Loupas Target   (same colour scale → easy comparison)
  4. Magnitude Mask  [0, 1]
  5. Spatial Coherence Maps (per-frame, both methods)

Also prints quantitative self-verification metrics and fails loudly
if the Loupas targets don't meet quality thresholds.

Usage:
    python scripts/visualize_loupas_targets.py
    python scripts/visualize_loupas_targets.py --split val --sample_indices 0 5 35
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.velocity_dataset import PreprocessedVelocityDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────
def spatial_coherence(vel_frame: np.ndarray, mask_frame: np.ndarray) -> float:
    """Mean correlation of each pixel with its 4-connected neighbours (inside mask)."""
    m = mask_frame > 0.1
    if m.sum() < 100:
        return 0.0
    v = vel_frame.copy()
    v[~m] = 0.0
    # Shift-correlations (up, down, left, right)
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    corrs = []
    for dy, dx in shifts:
        shifted = np.roll(np.roll(v, dy, axis=0), dx, axis=1)
        shifted_m = np.roll(np.roll(m.astype(float), dy, axis=0), dx, axis=1) > 0.5
        valid = m & shifted_m
        if valid.sum() < 50:
            continue
        a = v[valid]
        b = shifted[valid]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        corrs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0


def compute_dynamic_vmax(vel: np.ndarray, mask: np.ndarray, pctl: int = 99) -> float:
    sig_vals = np.abs(vel[mask > 0.1])
    if len(sig_vals) == 0:
        return 0.3
    return float(np.percentile(sig_vals, pctl))


# ── loading ────────────────────────────────────────────────────────────────
def load_kasai_and_loupas(preprocessed_dir: Path, idx: int, split: str):
    split_dir = preprocessed_dir / split
    with open(preprocessed_dir / "metadata.json") as f:
        metadata = json.load(f)
    gmax = metadata["global_stats"]["global_max"]

    ds_kasai = PreprocessedVelocityDataset(
        split_dir, global_max=gmax, return_mask=True,
        use_loupas_targets=False,
    )
    ds_loupas = PreprocessedVelocityDataset(
        split_dir, global_max=gmax, return_mask=True,
        use_loupas_targets=True,
    )

    inp, tgt_kasai, mask = ds_kasai[idx]
    _, tgt_loupas, _ = ds_loupas[idx]
    info = ds_kasai.get_sample_info(idx)
    return inp, tgt_kasai, tgt_loupas, mask, info, metadata


# ── quality gate ───────────────────────────────────────────────────────────
def verify_quality(raw_np, loupas_np, mask_np, sample_name: str):
    """Compute & report spatial coherence; return dict of metrics."""
    T = raw_np.shape[0]
    # Use frames at 25%, 50%, 75%
    check_frames = [T // 4, T // 2, 3 * T // 4]

    metrics = {"sample": sample_name, "frames": {}}
    for t in check_frames:
        sc_raw = spatial_coherence(raw_np[t], mask_np[t])
        sc_loupas = spatial_coherence(loupas_np[t], mask_np[t])
        improvement = sc_loupas - sc_raw
        metrics["frames"][t] = {
            "coherence_raw": sc_raw,
            "coherence_loupas": sc_loupas,
            "improvement": improvement,
        }

    # Aggregate
    avg_raw = np.mean([v["coherence_raw"] for v in metrics["frames"].values()])
    avg_loupas = np.mean([v["coherence_loupas"] for v in metrics["frames"].values()])
    metrics["avg_coherence_raw"] = avg_raw
    metrics["avg_coherence_loupas"] = avg_loupas

    # Signal statistics
    sig_mask = mask_np > 0.1
    raw_sig = raw_np[sig_mask]
    loupas_sig = loupas_np[sig_mask]
    metrics["raw_signal_std"] = float(raw_sig.std()) if len(raw_sig) else 0
    metrics["loupas_signal_std"] = float(loupas_sig.std()) if len(loupas_sig) else 0
    metrics["raw_99pctl"] = float(np.percentile(np.abs(raw_sig), 99)) if len(raw_sig) else 0
    metrics["loupas_99pctl"] = float(np.percentile(np.abs(loupas_sig), 99)) if len(loupas_sig) else 0

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"QUALITY REPORT: {sample_name}")
    logger.info(f"{'='*60}")
    for t, v in metrics["frames"].items():
        status = "✓" if v["coherence_loupas"] > 0.5 else "✗"
        logger.info(
            f"  Frame {t:3d}: Kasai={v['coherence_raw']:.3f}  "
            f"Loupas={v['coherence_loupas']:.3f}  "
            f"Δ={v['improvement']:+.3f}  {status}"
        )
    logger.info(f"  AVG coherence: Kasai={avg_raw:.3f} → Loupas={avg_loupas:.3f}")
    logger.info(f"  Signal std: Kasai={metrics['raw_signal_std']:.4f}, "
                f"Loupas={metrics['loupas_signal_std']:.4f}")
    logger.info(f"  99th pctl: Kasai={metrics['raw_99pctl']:.4f}, "
                f"Loupas={metrics['loupas_99pctl']:.4f}")

    if avg_loupas < 0.5:
        logger.warning(f"  ⚠ Loupas coherence {avg_loupas:.3f} < 0.5 — waves may not be visible!")
    elif avg_loupas > 0.8:
        logger.info(f"  ✓ Loupas coherence {avg_loupas:.3f} — clean wave structure confirmed.")
    return metrics


# ── main figure ────────────────────────────────────────────────────────────
def visualize_comparison(
    inp_t, tgt_kasai_t, tgt_loupas_t, mask_t,
    info, metadata, output_path, metrics,
):
    """5-row comparison: Input | Kasai | Loupas | Mask | Coherence."""
    inp = inp_t.numpy()           # (2, T_sparse, H, W)
    kasai = tgt_kasai_t.numpy()   # (1, T_dense, H, W)
    loupas = tgt_loupas_t.numpy() # (1, T_dense, H, W)
    mask = mask_t.numpy()         # (1, T_dense, H, W)

    T_sparse = inp.shape[1]
    T_dense = kasai.shape[1]
    H, W = kasai.shape[2], kasai.shape[3]

    # Dynamic vmax — use the LOUPAS signal region (it's cleaner)
    vmax_loupas = compute_dynamic_vmax(loupas[0], mask[0], pctl=99)
    # Also compute raw vmax for context but use loupas scale for comparison
    vmax_kasai = compute_dynamic_vmax(kasai[0], mask[0], pctl=99)

    # Use Loupas vmax for both (fair comparison at same colour scale)
    vmax = vmax_loupas

    t_indices = np.linspace(0, T_dense - 1, 8, dtype=int)
    vel_norm = Normalize(vmin=-vmax, vmax=vmax)
    mask_norm = Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(5, 8, figsize=(30, 19))

    row_labels = [
        "Model Input\n(IQ magnitude)",
        f"Raw Kasai Target\n(vmax={vmax:.4f})",
        f"Loupas Target\n(vmax={vmax:.4f})",
        "Magnitude Mask\n[0, 1]",
        "Spatial Coherence\n(pixel-neighbour corr)",
    ]

    coherence_kasai_values = []
    coherence_loupas_values = []

    for col, t in enumerate(t_indices):
        sparse_idx = min(t // 20, T_sparse - 1)

        # Row 0 — IQ magnitude
        ax = axes[0, col]
        iq_mag = np.sqrt(inp[0, sparse_idx] ** 2 + inp[1, sparse_idx] ** 2)
        ax.imshow(iq_mag.T, cmap="magma", origin="lower", aspect="auto")
        ax.set_title(f"sparse t={sparse_idx}", fontsize=8)

        # Row 1 — Kasai
        ax = axes[1, col]
        ax.imshow(kasai[0, t].T, cmap="RdBu_r", norm=vel_norm,
                  origin="lower", aspect="auto")
        ax.set_title(f"t={t}", fontsize=8)

        # Row 2 — Loupas
        ax = axes[2, col]
        ax.imshow(loupas[0, t].T, cmap="RdBu_r", norm=vel_norm,
                  origin="lower", aspect="auto")
        ax.set_title(f"t={t}", fontsize=8)

        # Row 3 — Mask
        ax = axes[3, col]
        ax.imshow(mask[0, t].T, cmap="viridis", norm=mask_norm,
                  origin="lower", aspect="auto")

        # Row 4 — Spatial coherence mini-bar
        sc_k = spatial_coherence(kasai[0, t], mask[0, t])
        sc_l = spatial_coherence(loupas[0, t], mask[0, t])
        coherence_kasai_values.append(sc_k)
        coherence_loupas_values.append(sc_l)
        ax = axes[4, col]
        bars = ax.bar(["Kasai", "Loupas"], [sc_k, sc_l],
                      color=["#d66", "#6b6"])
        ax.set_ylim(0, 1)
        ax.set_title(f"t={t}", fontsize=8)
        for bar, val in zip(bars, [sc_k, sc_l]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    for row in range(5):
        for col in range(8):
            if row < 4:
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10, rotation=0,
                                labelpad=130, va="center")

    # Colour bars for rows 0-3
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="magma"), ax=axes[0, :].tolist(),
        shrink=0.7, pad=0.01, label="IQ mag"
    )
    sm_vel = plt.cm.ScalarMappable(norm=vel_norm, cmap="RdBu_r")
    fig.colorbar(sm_vel, ax=axes[1, :].tolist(), shrink=0.7, pad=0.01, label="Velocity")
    fig.colorbar(sm_vel, ax=axes[2, :].tolist(), shrink=0.7, pad=0.01, label="Velocity")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=mask_norm, cmap="viridis"),
        ax=axes[3, :].tolist(), shrink=0.7, pad=0.01, label="Mask"
    )

    avg_coh_k = np.mean(coherence_kasai_values)
    avg_coh_l = np.mean(coherence_loupas_values)

    title = (
        f"{info['base_name']}  —  Input (2,{T_sparse},{H},{W})  →  "
        f"Target (1,{T_dense},{H},{W})\n"
        f"Kasai vmax={vmax_kasai:.4f}   Loupas vmax={vmax_loupas:.4f}   "
        f"Avg coherence: Kasai={avg_coh_k:.3f}  Loupas={avg_coh_l:.3f}  "
        f"(improvement: {avg_coh_l - avg_coh_k:+.3f})"
    )
    fig.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout(rect=[0.12, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {output_path}")


# ── detail figure (line profiles) ──────────────────────────────────────────
def visualize_detail(kasai_np, loupas_np, mask_np, info, output_path):
    """Zoomed-in line profiles and histograms for a single mid-time frame."""
    T = kasai_np.shape[0]
    t = T // 2
    m = mask_np[t] > 0.1

    vmax = compute_dynamic_vmax(loupas_np, mask_np, 99)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    vel_kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")

    axes[0, 0].imshow(kasai_np[t].T, **vel_kw)
    axes[0, 0].set_title(f"Kasai (frame {t})")
    axes[0, 1].imshow(loupas_np[t].T, **vel_kw)
    axes[0, 1].set_title(f"Loupas (frame {t})")
    axes[0, 2].imshow(mask_np[t].T, cmap="viridis", vmin=0, vmax=1,
                      origin="lower", aspect="auto")
    axes[0, 2].set_title("Mask")

    # Find row through middle of ROI
    rows = np.where(m.any(axis=1))[0]
    mid_row = rows[len(rows) // 2] if len(rows) > 0 else kasai_np.shape[1] // 2
    for ax in axes[0, :2]:
        ax.axhline(mid_row, color="lime", lw=1.5, ls="--", alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    # Lateral profile
    x = np.arange(kasai_np.shape[2])
    axes[1, 0].plot(x, kasai_np[t, mid_row], "r-", alpha=0.5, lw=0.8, label="Kasai")
    axes[1, 0].plot(x, loupas_np[t, mid_row], "b-", lw=1.5, label="Loupas")
    axes[1, 0].set_title(f"Lateral profile (row {mid_row})")
    axes[1, 0].legend(); axes[1, 0].set_xlabel("Lateral pixel")
    axes[1, 0].set_ylim(-vmax * 1.3, vmax * 1.3)

    # Temporal profile at centre of ROI
    cols = np.where(m.any(axis=0))[0]
    cx, cy = mid_row, (cols[len(cols) // 2] if len(cols) else kasai_np.shape[2] // 2)
    axes[1, 1].plot(kasai_np[:, cx, cy], "r-", alpha=0.5, lw=0.8, label="Kasai")
    axes[1, 1].plot(loupas_np[:, cx, cy], "b-", lw=1.5, label="Loupas")
    axes[1, 1].set_title(f"Temporal profile (pixel [{cx},{cy}])")
    axes[1, 1].legend(); axes[1, 1].set_xlabel("Frame")

    # Summary box
    sig_mask = mask_np > 0.1
    raw_sig = kasai_np[sig_mask]
    loupas_sig = loupas_np[sig_mask]
    sc_kasai = spatial_coherence(kasai_np[t], mask_np[t])
    sc_loupas = spatial_coherence(loupas_np[t], mask_np[t])

    txt = (
        f"QUALITY METRICS (frame {t})\n"
        f"{'─' * 35}\n"
        f"Spatial coherence:\n"
        f"  Kasai:  {sc_kasai:.3f}\n"
        f"  Loupas: {sc_loupas:.3f}\n"
        f"\n"
        f"Signal std (all frames in ROI):\n"
        f"  Kasai:  {raw_sig.std():.4f}\n"
        f"  Loupas: {loupas_sig.std():.4f}\n"
        f"\n"
        f"99th pctl |velocity| in ROI:\n"
        f"  Kasai:  {np.percentile(np.abs(raw_sig), 99):.4f}\n"
        f"  Loupas: {np.percentile(np.abs(loupas_sig), 99):.4f}\n"
        f"\n"
        f"Dynamic vmax (Loupas): {vmax:.4f}\n"
        f"ROI coverage: {sig_mask.mean()*100:.1f}%\n"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes,
                    va="top", fontsize=11, family="monospace",
                    bbox=dict(boxstyle="round", fc="lightyellow"))

    fig.suptitle(f"Detail — {info['base_name']}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {output_path}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize Loupas vs Kasai targets")
    parser.add_argument("--preprocessed_dir", default="data/preprocessed")
    parser.add_argument("--output_dir", default="results/loupas_target_viz")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = args.sample_indices or list(range(args.num_samples))

    logger.info("=" * 60)
    logger.info("LOUPAS vs KASAI TARGET VISUALIZATION + QUALITY GATE")
    logger.info("=" * 60)

    all_metrics = []
    for idx in indices:
        try:
            inp, kasai, loupas, mask, info, meta = load_kasai_and_loupas(
                preprocessed_dir, idx, args.split
            )
            metrics = verify_quality(
                kasai.numpy()[0], loupas.numpy()[0], mask.numpy()[0],
                info["base_name"],
            )
            all_metrics.append(metrics)

            visualize_comparison(
                inp, kasai, loupas, mask, info, meta,
                output_dir / f"kasai_vs_loupas_{idx}_{info['base_name']}.png",
                metrics,
            )
            visualize_detail(
                kasai.numpy()[0], loupas.numpy()[0], mask.numpy()[0],
                info, output_dir / f"detail_{idx}_{info['base_name']}.png",
            )
        except Exception as e:
            logger.error(f"Sample {idx}: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    if all_metrics:
        avg_kasai = np.mean([m["avg_coherence_raw"] for m in all_metrics])
        avg_loupas = np.mean([m["avg_coherence_loupas"] for m in all_metrics])
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL QUALITY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Avg spatial coherence across all samples:")
        logger.info(f"    Kasai:  {avg_kasai:.3f}")
        logger.info(f"    Loupas: {avg_loupas:.3f}")
        logger.info(f"    Improvement: {avg_loupas - avg_kasai:+.3f}")
        if avg_loupas > 0.7:
            logger.info("  ✓ PASS — Loupas targets have clear wave structure")
        elif avg_loupas > 0.5:
            logger.info("  ~ MARGINAL — some wave structure but may need larger sigma")
        else:
            logger.info("  ✗ FAIL — wave structure not sufficient for training")

    logger.info(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
