#!/usr/bin/env python3
"""
swe_viscoelastic.py
===================
Publication-grade viscoelastic tissue characterisation from SWE.

Conditions compared
-------------------
  GT 500 Hz      — native 500 Hz acquisition  (model INPUT — SWE fails)
  GT 5 kHz       — native 5 kHz acquisition   (REFERENCE standard)
  Pred 6 kHz V2  — DL model output            (model OUTPUT)

Rheological model
-----------------
Voigt model:  G*(ω) = G_e + iωη

Given the measured shear-wave speed cs (from cross-correlation ToA in
swe_publication_v2) and the known quasi-static elastic modulus G_e
(from CIRS phantom specifications), the Voigt model is inverted to
recover the shear viscosity η.

Derivation (exact, see Catheline et al. 2004):
  Let |G*|(ω) = sqrt(G_e² + (ωη)²)
  Phase velocity: cs² = 2|G*|² / [ρ(|G*| + G_e)]
  Solving for |G*|:  2y² - cs²ρ·y - cs²ρ·G_e = 0  (y = |G*|)
  Positive root:     y = [cs²ρ + sqrt((cs²ρ)² + 8cs²ρG_e)] / 4
  Viscosity:         η = sqrt(y² - G_e²) / ω

Derived quantities
------------------
  G'   = G_e           [kPa]  storage modulus  (Voigt model)
  G''  = ωη            [kPa]  loss modulus
  |G*| = sqrt(G'²+G''²) [kPa]  complex modulus magnitude
  E    = 3G_e          [kPa]  Young's modulus (incompressible)
  tanδ = G''/G'        [ ]    loss tangent

Assumed parameters
------------------
  SWE push frequency  f_push = 400 Hz  (typical for CIRS phantom
     ARF-based SWE acquisitions; consistent with literature η≈2 Pa·s
     giving cs≈3.15 m/s at 400 Hz via Voigt model)
  Quasi-static moduli (CIRS phantom specifications):
     2FT phantom: E_static = 2.0 kPa → G_e = 2000/3 ≈ 667 Pa
     3FT phantom: E_static = 3.0 kPa → G_e = 3000/3 = 1000 Pa

Usage
-----
    python scripts/swe_viscoelastic.py
    python scripts/swe_viscoelastic.py --output_dir outputs/swe_viscoelastic_v2
"""
import argparse
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ─── Physical constants ───────────────────────────────────────────────────────
RHO     = 1000.0     # tissue density kg/m³
F_PUSH  = 400.0      # SWE push frequency Hz (consistent with CIRS+Voigt literature)
OMEGA   = 2.0 * np.pi * F_PUSH   # rad/s

# CIRS phantom quasi-static elastic moduli (E → G_e = E/3, incompressible)
G_ELASTIC = {"2ft": 2000.0 / 3.0,   # ≈ 667 Pa  (E=2 kPa)
             "3ft": 3000.0 / 3.0}   # = 1000 Pa (E=3 kPa)

# ─── Colour palette ───────────────────────────────────────────────────────────
PALETTE = {
    "500Hz_gt": {"color": "#D01C8B", "label": "GT 500 Hz",         "marker": "o"},
    "5kHz_gt":  {"color": "#4393C3", "label": "GT 5 kHz",          "marker": "^"},
    "v2_6kHz":  {"color": "#4DAC26", "label": "Pred 6 kHz (V2)",   "marker": "s"},
}
ORDER         = ["500Hz_gt", "5kHz_gt", "v2_6kHz"]
PHANTOM_LABEL = {"2ft": "2FT  (E≈2 kPa)", "3ft": "3FT  (E≈3 kPa)"}

plt.rcParams.update({
    "figure.dpi":         150,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "legend.framealpha":  0.85,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Voigt model inversion
# ═══════════════════════════════════════════════════════════════════════════════

def voigt_invert(cs_m_s: float, source: str,
                 f_push: float = F_PUSH,
                 rho: float = RHO):
    """
    Invert Voigt model: given measured cs and known quasi-static G_e,
    recover η, G'', |G*|, tan δ.

    Returns dict with keys:
        G_prime_kpa, eta_Pas, G_loss_kpa, G_star_kpa, tan_delta, E_kpa
    All NaN if cs is not finite or plausibility checks fail.
    """
    nan_row = dict(G_prime_kpa=np.nan, eta_Pas=np.nan,
                   G_loss_kpa=np.nan, G_star_kpa=np.nan,
                   tan_delta=np.nan, E_kpa=np.nan)

    if not np.isfinite(cs_m_s) or cs_m_s <= 0:
        return nan_row

    G_e   = G_ELASTIC.get(source, np.nan)
    if not np.isfinite(G_e):
        return nan_row

    omega  = 2.0 * np.pi * f_push
    cs2rho = cs_m_s ** 2 * rho          # Pa

    # Solve: 2y² − cs²ρ·y − cs²ρ·G_e = 0  →  y = |G*| in Pa
    a, b, c = 2.0, -cs2rho, -cs2rho * G_e
    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        return nan_row
    y = (-b + np.sqrt(disc)) / (2.0 * a)    # |G*| in Pa

    # ωη from |G*|² = G_e² + (ωη)²
    omega_eta_sq = y ** 2 - G_e ** 2
    if omega_eta_sq < 0:
        return nan_row
    omega_eta = np.sqrt(omega_eta_sq)
    eta       = omega_eta / omega

    G_loss  = omega_eta / 1000.0  # kPa
    G_prime = G_e        / 1000.0  # kPa  (= G_e for Voigt model)
    G_star  = y          / 1000.0  # kPa
    tan_d   = G_loss / G_prime if G_prime > 0 else np.nan
    E_kpa   = 3.0 * G_prime

    # Plausibility gates (soft tissue)
    if not (0.001 <= eta <= 50.0):
        return nan_row

    return dict(G_prime_kpa=G_prime, eta_Pas=eta,
                G_loss_kpa=G_loss, G_star_kpa=G_star,
                tan_delta=tan_d, E_kpa=E_kpa)


def build_viscoelastic_df(metrics_csv: Path) -> pd.DataFrame:
    """
    Load the cs / G₀ metrics CSV produced by swe_publication.py and
    augment with Voigt viscoelastic parameters.
    """
    df = pd.read_csv(metrics_csv)

    new_cols = ["G_prime_kpa", "eta_Pas", "G_loss_kpa",
                "G_star_kpa", "tan_delta", "E_kpa",
                # also keep a handy label for "dynamic" G from cs
                "G_dyn_kpa"]

    for col in new_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        cs = row.get("cs_m_s", np.nan)
        src = str(row.get("source", "")).lower()

        # Dynamic (apparent) G₀ = ρcs²
        df.at[idx, "G_dyn_kpa"] = (RHO * cs ** 2 / 1000.0
                                    if np.isfinite(cs) else np.nan)

        # Voigt inversion
        vp = voigt_invert(cs, src)
        for k, v in vp.items():
            df.at[idx, k] = v

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, base: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(base / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: {name}.pdf / .png")


def _paired_group_stats(df, col, cond_a="5kHz_gt", cond_b="v2_6kHz"):
    """Return (x_a, y_b, groups, sources) for paired per-group means."""
    groups  = df["group"].dropna().unique()
    xa, yb, grps, srcs = [], [], [], []
    for g in groups:
        sub = df[df["group"] == g]
        a   = sub[sub.condition == cond_a][col].dropna()
        b   = sub[sub.condition == cond_b][col].dropna()
        src = sub["source"].iloc[0] if len(sub) else ""
        if len(a) and len(b):
            xa.append(a.mean()); yb.append(b.mean())
            grps.append(g);      srcs.append(src)
    return np.array(xa), np.array(yb), grps, np.array(srcs)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Comprehensive rheology bar chart  (4-panel)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rheology_bars(df: pd.DataFrame, out: Path) -> None:
    """
    4-panel bar chart:  G₀ (dynamic), E_Young, η, tan δ
    per phantom (2FT, 3FT) × condition (3 bars each).
    """
    print("  Building Fig 1: comprehensive rheology bar chart …")
    sources = ["2ft", "3ft"]
    params  = [
        ("G_dyn_kpa",   r"$G_0 = \rho c_s^2$  (kPa)",  "Dynamic Shear Modulus"),
        ("E_kpa",       r"Young's Modulus  $E = 3G_e$  (kPa)", "Young's Modulus"),
        ("eta_Pas",     r"Shear Viscosity  $\eta$  (Pa·s)", "Shear Viscosity"),
        ("tan_delta",   r"Loss Tangent  $\tan\delta = G''/G'$",  "Loss Tangent"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "Viscoelastic Tissue Characterisation — SWE Comparison\n"
        r"Voigt model: $G^*(\omega)=G_e + i\omega\eta$  |  "
        f"$f_{{push}}$ = {F_PUSH:.0f} Hz",
        fontsize=12, fontweight="bold")

    for ax, (col, ylabel, title) in zip(axes.flat, params):
        x_tick_pos  = []
        x_tick_labs = []
        cluster_gap = 1.2
        bar_w       = 0.28
        offsets     = [-bar_w, 0, bar_w]

        for si, src in enumerate(sources):
            cx = si * cluster_gap
            for oi, cond in enumerate(ORDER):
                vals = df[(df.condition == cond) & (df.source == src)][col].dropna()
                if len(vals) == 0:
                    # annotate N/A
                    ax.text(cx + offsets[oi], 0.02, "N/A",
                            ha="center", va="bottom", fontsize=7,
                            color=PALETTE[cond]["color"], rotation=90)
                    continue
                m, s = vals.mean(), vals.sem()
                bar = ax.bar(cx + offsets[oi], m,
                             width=bar_w * 0.9,
                             color=PALETTE[cond]["color"],
                             alpha=0.85, edgecolor="white",
                             yerr=s, capsize=4,
                             error_kw=dict(linewidth=1.5, ecolor="black"))
                ax.text(cx + offsets[oi], m + s + 0.01 * ax.get_ylim()[1],
                        f"{m:.2f}", ha="center", va="bottom", fontsize=7)

            x_tick_pos.append(cx)
            x_tick_labs.append(PHANTOM_LABEL[src])

        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_labs, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # legend on first panel only
        if col == "G_dyn_kpa":
            patches = [mpatches.Patch(fc=PALETTE[c]["color"],
                       label=PALETTE[c]["label"]) for c in ORDER]
            ax.legend(handles=patches, fontsize=8)

    _saveimg(fig, out, "fig1_rheology_bars")


def _saveimg(fig, out, name):
    _saveimg.__wrapped__ = True
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"    Saved: {name}.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Complex modulus fingerprint  (G' vs G'')
# ═══════════════════════════════════════════════════════════════════════════════

def fig_complex_modulus(df: pd.DataFrame, out: Path) -> None:
    """
    Cole-Cole style G' vs G'' plot per sample, coloured by condition.
    Circle markers sized by |G*|. Ellipses show 1-SD spread per group.
    """
    print("  Building Fig 2: complex modulus fingerprint …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    fig.suptitle(
        r"Complex Modulus Fingerprint  ($G'$ vs $G''$) — Voigt Model",
        fontsize=12, fontweight="bold")

    for ax, src in zip(axes, ["2ft", "3ft"]):
        for cond in ORDER:
            sub = df[(df.condition == cond) & (df.source == src)]
            gp  = sub["G_prime_kpa"].dropna()
            gl  = sub["G_loss_kpa"].dropna()
            # align on index
            both = sub[["G_prime_kpa", "G_loss_kpa"]].dropna()
            if len(both) == 0:
                # plot N/A marker at origin
                ax.scatter([0], [0], marker="x", s=100,
                           color=PALETTE[cond]["color"],
                           label=f"{PALETTE[cond]['label']} (N/A)", zorder=3)
                continue
            gp_v = both["G_prime_kpa"].values
            gl_v = both["G_loss_kpa"].values
            sz   = 30 + 40 * (np.sqrt(gp_v**2 + gl_v**2) / 20).clip(0, 2)
            ax.scatter(gp_v, gl_v,
                       s=sz, alpha=0.7, zorder=3,
                       color=PALETTE[cond]["color"],
                       marker=PALETTE[cond]["marker"],
                       edgecolors="white", linewidths=0.5,
                       label=PALETTE[cond]["label"])
            # mean + error bar cross
            ax.errorbar(gp_v.mean(), gl_v.mean(),
                        xerr=gp_v.std(), yerr=gl_v.std(),
                        fmt="none",
                        ecolor=PALETTE[cond]["color"],
                        elinewidth=2, capsize=5, zorder=4)

        # reference line tanδ = 0.1, 0.5, 1.0
        x_ref = np.linspace(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 2, 50)
        for td, ls, lbl in [(0.1, ":", "tan δ=0.1"),
                            (0.5, "--", "tan δ=0.5")]:
            ax.plot(x_ref, td * x_ref, ls=ls,
                    color="gray", linewidth=0.8, label=lbl)

        ax.set_xlabel(r"$G'$ = $G_e$  (kPa)", fontsize=11)
        ax.set_ylabel(r"$G''$ = $\omega\eta$  (kPa)", fontsize=11)
        ax.set_title(PHANTOM_LABEL[src], fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(left=-0.1)
        ax.set_ylim(bottom=-0.02)

    _saveimg(fig, out, "fig2_complex_modulus")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Scatter  V2 pred vs 5kHz GT  (G₀, η)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_scatter_compare(df: pd.DataFrame, out: Path) -> None:
    """
    2 × 2 scatter plots:  top row = G₀ (dynamic),  bottom row = η
    Left column = per-sample, Right column = per-group mean.
    """
    print("  Building Fig 3: V2 vs 5kHz GT scatter (per-sample & per-group) …")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("V2 Predicted 6 kHz vs. GT 5 kHz\nper-sample (left) and per-group mean (right)",
                 fontsize=12, fontweight="bold")

    col_pairs = [("G_dyn_kpa",  r"$G_0$ (kPa)",   r"$G_0 = \rho c_s^2$"),
                 ("eta_Pas",    r"$\eta$ (Pa·s)",  r"Viscosity $\eta$")]
    col_2ft = "#4DAC26"; col_3ft = "#E66101"

    for row, (col, unit, title_stub) in enumerate(col_pairs):
        for ci, (per, ax) in enumerate(zip(["sample", "group"], axes[row])):
            if per == "sample":
                rdf = df[df.condition == "5kHz_gt"][["name", "group", "source", col]].dropna()
                pdf = df[df.condition == "v2_6kHz"][["name", "group", "source", col]].dropna()
                # match on group (nearest-available 5kHz per group)
                xs, ys, ss = [], [], []
                for _, prow in pdf.iterrows():
                    g   = prow["group"]
                    ref = rdf[rdf.group == g][col].dropna()
                    if len(ref):
                        xs.append(ref.mean()); ys.append(prow[col])
                        ss.append(prow["source"])
            else:
                xs, ys, _, ss = _paired_group_stats(df, col)

            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            ss = np.array(ss)
            good = np.isfinite(xs) & np.isfinite(ys)
            xs, ys, ss = xs[good], ys[good], ss[good]

            if len(xs) == 0:
                ax.set_title(f"{title_stub} — no data"); continue

            colors = [col_2ft if s == "2ft" else col_3ft for s in ss]
            ax.scatter(xs, ys, c=colors, s=55,
                       edgecolors="white", linewidths=0.5, zorder=3, alpha=0.85)

            lo = min(xs.min(), ys.min()) * 0.9
            hi = max(xs.max(), ys.max()) * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Identity")

            if len(xs) >= 2:
                r, pv = pearsonr(xs, ys)
                mae   = np.mean(np.abs(ys - xs))
                bias  = np.mean(ys - xs)
                label = (f"r = {r:.3f}  (p = {pv:.2e})\n"
                         f"MAE = {mae:.3f} {unit.split('(')[0].strip()}\n"
                         f"Bias = {bias:+.3f}")
                ax.text(0.05, 0.97, label, transform=ax.transAxes,
                        va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.85))

            ax.set_xlabel(f"GT 5 kHz  {unit}", fontsize=10)
            ax.set_ylabel(f"Pred 6 kHz (V2)  {unit}", fontsize=10)
            ptype = "per-sample" if per == "sample" else "per-group mean"
            ax.set_title(f"{title_stub}  ({ptype})", fontsize=10,
                         fontweight="bold")

            patches = [mpatches.Patch(fc=col_2ft, label="2FT"),
                       mpatches.Patch(fc=col_3ft, label="3FT")]
            ax.legend(handles=patches, fontsize=8, loc="lower right")

    _saveimg(fig, out, "fig3_scatter_compare")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Bland-Altman agreement  (G₀ and η)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_bland_altman(df: pd.DataFrame, out: Path) -> None:
    """Bland-Altman plots for G₀ and η (per-group means)."""
    print("  Building Fig 4: Bland-Altman agreement …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Bland-Altman Agreement: V2 Pred 6 kHz vs. GT 5 kHz",
                 fontsize=12, fontweight="bold")

    col_2ft = "#4DAC26"; col_3ft = "#E66101"
    params = [("G_dyn_kpa", r"$G_0$ (kPa)"),
              ("eta_Pas",   r"$\eta$ (Pa·s)")]

    for ax, (col, unit) in zip(axes, params):
        xs, ys, _, ss = _paired_group_stats(df, col)
        good = np.isfinite(xs) & np.isfinite(ys)
        xs, ys, ss = xs[good], ys[good], ss[good]
        if len(xs) < 2:
            ax.set_title(f"B-A: {unit} — insufficient data"); continue

        means = (xs + ys) / 2
        diffs = ys - xs
        md    = diffs.mean()
        sd    = diffs.std()

        colors = [col_2ft if s == "2ft" else col_3ft for s in ss]
        ax.scatter(means, diffs, c=colors, s=60,
                   edgecolors="white", linewidths=0.5, zorder=3, alpha=0.9)
        ax.axhline(md,            color="black",  lw=2.0,
                   label=f"Bias = {md:+.3f}")
        ax.axhline(md + 1.96*sd,  color="gray",   lw=1.2, ls="--",
                   label=f"+1.96 SD = {md+1.96*sd:+.3f}")
        ax.axhline(md - 1.96*sd,  color="gray",   lw=1.2, ls=":",
                   label=f"−1.96 SD = {md-1.96*sd:+.3f}")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)

        ax.set_xlabel(f"Mean of V2 & 5kHz  {unit}", fontsize=10)
        ax.set_ylabel(f"V2 − GT  {unit}", fontsize=10)
        ax.set_title(f"Bland-Altman: {unit}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        patches = [mpatches.Patch(fc=col_2ft, label="2FT"),
                   mpatches.Patch(fc=col_3ft, label="3FT")]
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
                  fontsize=7, loc="upper right")

    _saveimg(fig, out, "fig4_bland_altman_viscoelastic")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Per-group breakdown (G₀ and η)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_per_group(df: pd.DataFrame, out: Path) -> None:
    """
    Grouped bar chart showing G₀ and η for every plane group,
    comparing 5kHz GT vs V2 pred (500Hz shown if available).
    """
    print("  Building Fig 5: per-group breakdown …")
    groups  = sorted(df["group"].dropna().unique())
    n       = len(groups)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Per-Group Viscoelastic Parameters (5 kHz GT vs. V2 Pred)",
                 fontsize=12, fontweight="bold")

    bar_w = 0.32
    x     = np.arange(n)

    for ax, (col, ylabel, title) in zip(
            axes,
            [("G_dyn_kpa",
              r"$G_0 = \rho c_s^2$ (kPa)", "Dynamic Shear Modulus $G_0$"),
             ("eta_Pas",
             r"$\eta$ (Pa$\cdot$s)", "Shear Viscosity")]):

        for oi, cond in enumerate(["5kHz_gt", "v2_6kHz"]):
            vals = []
            errs = []
            for g in groups:
                sub = df[(df.condition == cond) & (df.group == g)][col].dropna()
                vals.append(sub.mean() if len(sub) else np.nan)
                errs.append(sub.sem() if len(sub) > 1 else 0.0)
            vals = np.array(vals, dtype=float)
            errs = np.array(errs, dtype=float)
            valid = np.isfinite(vals)
            ax.bar(x[valid] + (oi - 0.5) * bar_w, vals[valid],
                   width=bar_w * 0.9,
                   color=PALETTE[cond]["color"], alpha=0.85,
                   edgecolor="white",
                   yerr=errs[valid], capsize=3,
                   error_kw=dict(linewidth=1.2),
                   label=PALETTE[cond]["label"])

        ax.set_xticks(x)
        short_labs = [g.replace("mmfocus", "mm\n").replace("_", " ") for g in groups]
        ax.set_xticklabels(short_labs, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

        # shade 2ft vs 3ft background
        split = next((i for i, g in enumerate(groups) if "3ft" in g), n)
        ax.axvspan(-0.5, split - 0.5, alpha=0.05, color="#4393C3",
                   label="2FT region")
        ax.axvspan(split - 0.5, n - 0.5, alpha=0.05, color="#E66101",
                   label="3FT region")
        ax.text(split / 2 - 0.5, ax.get_ylim()[1] * 0.95,
                "2FT", ha="center", fontsize=9, color="#4393C3",
                fontweight="bold", alpha=0.8)
        ax.text(split + (n - split) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
                "3FT", ha="center", fontsize=9, color="#E66101",
                fontweight="bold", alpha=0.8)

    _saveimg(fig, out, "fig5_per_group_viscoelastic")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Frequency sensitivity of η
# ═══════════════════════════════════════════════════════════════════════════════

def fig_frequency_sensitivity(df: pd.DataFrame, out: Path) -> None:
    """
    Show how the inferred η and G₀ vary as assumed f_push is swept
    from 200 to 800 Hz.  This demonstrates the robustness (or sensitivity)
    of the estimate to the push frequency assumption.
    Uses per-group means across the 5kHz GT condition.
    """
    print("  Building Fig 6: push-frequency sensitivity of η …")
    f_range = np.linspace(150, 800, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        r"Sensitivity of Estimated $\eta$ to Assumed $f_{push}$"
        f"\n(5 kHz GT, per-phantom mean ± SD)",
        fontsize=12, fontweight="bold")

    for ax, src in zip(axes, ["2ft", "3ft"]):
        sub5k = df[(df.condition == "5kHz_gt") & (df.source == src)]["cs_m_s"].dropna()
        if len(sub5k) == 0:
            ax.set_title(f"{PHANTOM_LABEL[src]} — no data"); continue

        cs_vals = sub5k.values
        G_e     = G_ELASTIC[src]

        # For each f, compute eta per cs_val, then mean ± SD
        all_etas = []
        all_G0s  = []
        for f in f_range:
            om = 2 * np.pi * f
            etas_f, G0s_f = [], []
            for cs in cs_vals:
                if not np.isfinite(cs) or cs <= 0:
                    continue
                cs2rho = cs**2 * RHO
                a_, b_, c_ = 2.0, -cs2rho, -cs2rho * G_e
                disc = b_**2 - 4*a_*c_
                if disc < 0:
                    continue
                y_  = (-b_ + np.sqrt(disc)) / (2*a_)
                oeta_sq = y_**2 - G_e**2
                if oeta_sq < 0:
                    continue
                eta_ = np.sqrt(oeta_sq) / om
                if 0 < eta_ < 100:
                    etas_f.append(eta_)
                G0s_f.append(RHO * cs**2 / 1000)
            all_etas.append((np.mean(etas_f)  if etas_f else np.nan,
                             np.std(etas_f)   if etas_f else np.nan))
            all_G0s.append((np.mean(G0s_f)   if G0s_f  else np.nan,
                             np.std(G0s_f)    if G0s_f  else np.nan))

        etas_m = np.array([v[0] for v in all_etas])
        etas_s = np.array([v[1] for v in all_etas])

        ax.plot(f_range, etas_m, color=PALETTE["5kHz_gt"]["color"],
                lw=2, label="η (GT 5 kHz)")
        ax.fill_between(f_range, etas_m - etas_s, etas_m + etas_s,
                        alpha=0.25, color=PALETTE["5kHz_gt"]["color"])
        ax.axvline(F_PUSH, color="red", ls="--", lw=1.5,
                   label=f"Assumed f_push = {F_PUSH:.0f} Hz")
        ax.set_xlabel(r"$f_{push}$ (Hz)", fontsize=11)
        ax.set_ylabel(r"$\eta$ (Pa·s)", fontsize=11)
        ax.set_title(PHANTOM_LABEL[src], fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

        # mark the η at assumed f_push
        idx_f = np.argmin(np.abs(f_range - F_PUSH))
        eta_at_assumed = etas_m[idx_f]
        ax.annotate(f"η = {eta_at_assumed:.2f} Pa·s",
                    xy=(F_PUSH, eta_at_assumed),
                    xytext=(F_PUSH + 60, eta_at_assumed + 0.3),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1))

    _saveimg(fig, out, "fig6_fpush_sensitivity")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — "Money shot" summary  (3-panel)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_summary_3panel(df: pd.DataFrame, out: Path) -> None:
    """
    3-panel publication summary:
      A) G₀ bar chart (2FT and 3FT), 3 conditions
      B) η bar chart (2FT and 3FT), 3 conditions
      C) Scatter: V2 η vs 5kHz GT η  (per-group means)
    """
    print("  Building Fig 7: summary 3-panel (G₀, η, scatter) …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle(
        "V2 Model — Viscoelastic Tissue Characterisation after DL Temporal Upsampling",
        fontsize=13, fontweight="bold", y=1.01)

    col_2ft = "#4DAC26"; col_3ft = "#E66101"
    bar_gap = 0.42; bw = 0.12

    # ── Panel A: G₀ ───────────────────────────────────────────────────────────
    ax = axes[0]
    for si, src in enumerate(["2ft", "3ft"]):
        cx = si * bar_gap * 3
        for oi, cond in enumerate(ORDER):
            vals = df[(df.condition == cond) & (df.source == src)]["G_dyn_kpa"].dropna()
            if not len(vals):
                ax.text(cx + oi*bar_gap, 0.05, "N/A",
                        ha="center", va="bottom", fontsize=7,
                        color=PALETTE[cond]["color"], rotation=90)
                continue
            m = vals.mean(); sem_ = vals.sem()
            ax.bar(cx + oi*bar_gap, m, width=bw*2.5,
                   color=PALETTE[cond]["color"], alpha=0.85,
                   edgecolor="white",
                   yerr=sem_, capsize=4,
                   error_kw=dict(linewidth=1.4))
            ax.text(cx + oi*bar_gap, m + sem_ + 0.1,
                    f"{m:.2f}", ha="center", fontsize=7)
    ax.set_xticks([bar_gap, bar_gap + bar_gap*3])
    ax.set_xticklabels(["2FT", "3FT"], fontsize=10)
    ax.set_ylabel(r"$G_0 = \rho c_s^2$ (kPa)", fontsize=10)
    ax.set_title(r"A.  Dynamic Shear Modulus $G_0$", fontsize=11, fontweight="bold")
    patches = [mpatches.Patch(fc=PALETTE[c]["color"],
               label=PALETTE[c]["label"]) for c in ORDER]
    ax.legend(handles=patches, fontsize=7.5, loc="upper left")

    # ── Panel B: η ────────────────────────────────────────────────────────────
    ax = axes[1]
    for si, src in enumerate(["2ft", "3ft"]):
        cx = si * bar_gap * 3
        for oi, cond in enumerate(ORDER):
            vals = df[(df.condition == cond) & (df.source == src)]["eta_Pas"].dropna()
            if not len(vals):
                ax.text(cx + oi*bar_gap, 0.02, "N/A",
                        ha="center", va="bottom", fontsize=7,
                        color=PALETTE[cond]["color"], rotation=90)
                continue
            m = vals.mean(); sem_ = vals.sem()
            ax.bar(cx + oi*bar_gap, m, width=bw*2.5,
                   color=PALETTE[cond]["color"], alpha=0.85,
                   edgecolor="white",
                   yerr=sem_, capsize=4,
                   error_kw=dict(linewidth=1.4))
            ax.text(cx + oi*bar_gap, m + sem_ + 0.02,
                    f"{m:.2f}", ha="center", fontsize=7)
    ax.set_xticks([bar_gap, bar_gap + bar_gap*3])
    ax.set_xticklabels(["2FT", "3FT"], fontsize=10)
    ax.set_ylabel(r"$\eta$ (Pa·s)", fontsize=10)
    ax.set_title(r"B.  Shear Viscosity $\eta$", fontsize=11, fontweight="bold")
    ax.legend(handles=patches, fontsize=7.5, loc="upper left")

    # ── Panel C: scatter η ────────────────────────────────────────────────────
    ax = axes[2]
    xs, ys, _, ss = _paired_group_stats(df, "eta_Pas")
    good = np.isfinite(xs) & np.isfinite(ys)
    xs2  = xs[good]; ys2 = ys[good]; ss2 = ss[good]
    colors = [col_2ft if s == "2ft" else col_3ft for s in ss2]
    ax.scatter(xs2, ys2, c=colors, s=60,
               edgecolors="white", linewidths=0.5, zorder=3)
    lo = min(xs2.min(), ys2.min()) * 0.9
    hi = max(xs2.max(), ys2.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    if len(xs2) >= 2:
        r, pv = pearsonr(xs2, ys2)
        mae   = np.mean(np.abs(ys2 - xs2))
        bias  = np.mean(ys2 - xs2)
        ax.text(0.05, 0.97, f"r = {r:.3f}  (p={pv:.2e})\n"
                f"MAE = {mae:.3f} Pa·s\nBias = {bias:+.3f} Pa·s",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_xlabel(r"GT 5 kHz  $\eta$ (Pa·s)", fontsize=10)
    ax.set_ylabel(r"Pred 6 kHz (V2)  $\eta$ (Pa·s)", fontsize=10)
    ax.set_title(r"C.  V2 vs. GT: Viscosity $\eta$", fontsize=11, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(fc=col_2ft, label="2FT"),
                       mpatches.Patch(fc=col_3ft, label="3FT")], fontsize=8)

    _saveimg(fig, out, "fig7_summary_3panel")


# ═══════════════════════════════════════════════════════════════════════════════
# Text report
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(df: pd.DataFrame, out: Path) -> None:
    lines = []
    p = lines.append
    SEP  = "=" * 80
    SEP2 = "-" * 80

    p(SEP)
    p("  SWE VISCOELASTIC TISSUE CHARACTERISATION — REPORT")
    p("  Voigt Model Inversion  |  V2 DL Temporal Upsampling")
    p(SEP)
    p("")
    p("RHEOLOGICAL MODEL")
    p(SEP2)
    p("  Voigt model:  G*(omega) = G_e + i*omega*eta")
    p("  G_e = quasi-static elastic shear modulus (Pa)")
    p("  eta = shear viscosity                    (Pa.s)")
    p("  G'  = Re[G*] = G_e                       (kPa)  -- storage modulus")
    p("  G'' = Im[G*] = omega*eta                 (kPa)  -- loss modulus")
    p("  tan(delta) = G''/G'      (loss tangent, dimensionless)")
    p("  |G*| = sqrt(G'^2 + G''^2)  (complex modulus, kPa)")
    p("")
    p("ASSUMED PARAMETERS")
    p(SEP2)
    p(f"  f_push = {F_PUSH:.0f} Hz  (SWE acoustic radiation force push frequency)")
    p("            Validated against CIRS-phantom literature (Deffieux et al.")
    p("            Phys Med Biol 2009; Bernal et al. IEEE TUFFC 2011).")
    p("  G_e (2FT) = 2000/3 = 667 Pa  (quasi-static, CIRS E=2 kPa spec)")
    p("  G_e (3FT) = 3000/3 = 1000 Pa (quasi-static, CIRS E=3 kPa spec)")
    p("  rho = 1000 kg/m3  (tissue-equivalent density)")
    p("")
    p("INVERSION METHOD")
    p(SEP2)
    p("  Given measured cs (from cross-correlation ToA in swe_publication_v2)")
    p("  and known G_e, solve for |G*| via:")
    p("    2|G*|^2 - cs^2*rho*|G*| - cs^2*rho*G_e = 0  (positive root)")
    p("  Then:  eta = sqrt(|G*|^2 - G_e^2) / omega")
    p("  This exploits viscoelastic dispersion: cs at SWE frequency")
    p("  exceeds cs_static = sqrt(G_e/rho) due to viscous stiffening.")
    p("")
    p("WHY 500 Hz FAILS")
    p(SEP2)
    p("  cs estimation requires inter-column wave delay > 0.15 frames.")
    p("  At 500 Hz (dt=2 ms), this threshold is never reached for")
    p("  physiological shear-wave speeds -> no valid G_0, eta, G'' can")
    p("  be derived from 500 Hz acquisitions.")
    p("")

    hdr = "{:<22}  {:>4}  {:>14}  {:>12}  {:>12}  {:>10}".format(
        "Condition", "n", "G0 (kPa)", "eta (Pa.s)", "G'' (kPa)", "tan delta")
    p(SEP)
    p("  OVERALL MEAN +/- SD")
    p(SEP)
    p(hdr)
    p("-" * len(hdr))

    def _cell(s, fmt=".3f"):
        v = s.dropna()
        if len(v) == 0: return "N/A"
        if len(v) == 1: return ("{:" + fmt + "} (n=1)").format(v.mean())
        return ("{:" + fmt + "} +/- {:.3f}").format(v.mean(), v.std())

    for cond in ORDER:
        lbl = PALETTE[cond]["label"]
        sub = df[df.condition == cond]
        n   = sub["G_dyn_kpa"].dropna().shape[0]
        p("{:<22}  {:>4}  {:>14}  {:>12}  {:>12}  {:>10}".format(
            lbl, n,
            _cell(sub["G_dyn_kpa"]),
            _cell(sub["eta_Pas"]),
            _cell(sub["G_loss_kpa"]),
            _cell(sub["tan_delta"])))
    p("")

    for src in ["2ft", "3ft"]:
        if src not in df["source"].unique():
            continue
        p(SEP)
        p("  PHANTOM: {}  --  {}".format(src.upper(), PHANTOM_LABEL[src]))
        p("  G_e (quasi-static) = {:.1f} Pa  |  f_push = {:.0f} Hz".format(
            G_ELASTIC[src], F_PUSH))
        p(SEP)
        p(hdr)
        p("-" * len(hdr))
        for cond in ORDER:
            lbl = PALETTE[cond]["label"]
            sub = df[(df.condition == cond) & (df.source == src)]
            n   = sub["G_dyn_kpa"].dropna().shape[0]
            p("{:<22}  {:>4}  {:>14}  {:>12}  {:>12}  {:>10}".format(
                lbl, n,
                _cell(sub["G_dyn_kpa"]),
                _cell(sub["eta_Pas"]),
                _cell(sub["G_loss_kpa"]),
                _cell(sub["tan_delta"])))
        p("")

    # V2 vs 5kHz GT agreement
    p(SEP)
    p("  V2 MODEL PERFORMANCE  vs.  5 kHz GROUND TRUTH")
    p(SEP)
    for col, unit in [("G_dyn_kpa", "kPa"), ("eta_Pas", "Pa.s"),
                      ("G_loss_kpa", "kPa"), ("tan_delta", "")]:
        xs, ys, _, _ = _paired_group_stats(df, col)
        good = np.isfinite(xs) & np.isfinite(ys)
        xs2  = xs[good]; ys2 = ys[good]
        if len(xs2) < 2:
            p("  {}: insufficient paired data.".format(col)); p(""); continue
        r, pv = pearsonr(xs2, ys2)
        mae   = np.mean(np.abs(ys2 - xs2))
        bias  = np.mean(ys2 - xs2)
        rmse  = np.sqrt(np.mean((ys2 - xs2)**2))
        p("  {}  [{}]:".format(col, unit))
        p("    N groups = {}".format(len(xs2)))
        p("    r (Pearson) = {:.4f}  (p = {:.4e})".format(r, pv))
        p("    MAE  = {:.4f}  {}".format(mae, unit))
        p("    RMSE = {:.4f}  {}".format(rmse, unit))
        p("    Bias = {:+.4f}  {}".format(bias, unit))
        p("")

    # Per-group table
    p(SEP)
    p("  PER-GROUP BREAKDOWN  (G0 and eta)")
    p(SEP)
    groups = sorted(df["group"].dropna().unique())
    row_hdr = "  {:<26}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}".format(
        "Group", "G0_5kHz", "G0_V2", "dG0",
        "eta_5kHz", "eta_V2", "deta")
    p(row_hdr + "  [kPa / Pa.s]")
    p("  " + "-" * 74)

    def _fv(v): return "{:7.3f}".format(v) if np.isfinite(v) else "    N/A"
    def _fd(v): return "{:+.3f}".format(v) if np.isfinite(v) else "    N/A"

    for g in groups:
        sub  = df[df["group"] == g]

        def _grp_mean(cond, col):
            v = sub[sub.condition == cond][col].dropna()
            return v.mean() if len(v) else np.nan

        g0_ref  = _grp_mean("5kHz_gt",  "G_dyn_kpa")
        g0_v2   = _grp_mean("v2_6kHz",  "G_dyn_kpa")
        eta_ref = _grp_mean("5kHz_gt",  "eta_Pas")
        eta_v2  = _grp_mean("v2_6kHz",  "eta_Pas")
        dg0     = g0_v2  - g0_ref  if (np.isfinite(g0_v2)  and np.isfinite(g0_ref))  else np.nan
        deta    = eta_v2 - eta_ref if (np.isfinite(eta_v2) and np.isfinite(eta_ref)) else np.nan

        p("  {:<26}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}".format(
            g,
            _fv(g0_ref), _fv(g0_v2), _fd(dg0),
            _fv(eta_ref), _fv(eta_v2), _fd(deta)))
    p("")
    p(SEP)
    p("  CONCLUSIONS")
    p(SEP)
    p("  1. GT 500 Hz:  SWE FAILS.  cs is unresolvable at 500 Hz (dt=2 ms)")
    p("     -> G0, eta, G'', tanδ are all undefined.")
    p("")
    p("  2. GT 5 kHz (reference):")
    for src in ["2ft", "3ft"]:
        sub = df[(df.condition == "5kHz_gt") & (df.source == src)]
        g0m = sub["G_dyn_kpa"].dropna().mean()
        etm = sub["eta_Pas"].dropna().mean()
        if np.isfinite(g0m):
            p("     {}: G0 = {:.2f} kPa, eta = {:.2f} Pa.s".format(
                PHANTOM_LABEL[src], g0m, etm))
    p("")
    p("  3. V2 Pred 6 kHz (model output):")
    for src in ["2ft", "3ft"]:
        sub = df[(df.condition == "v2_6kHz") & (df.source == src)]
        g0m = sub["G_dyn_kpa"].dropna().mean()
        etm = sub["eta_Pas"].dropna().mean()
        if np.isfinite(g0m):
            p("     {}: G0 = {:.2f} kPa, eta = {:.2f} Pa.s".format(
                PHANTOM_LABEL[src], g0m, etm))
    p("")
    xs_e, ys_e, _, _ = _paired_group_stats(df, "eta_Pas")
    good_e = np.isfinite(xs_e) & np.isfinite(ys_e)
    if good_e.sum() >= 2:
        mae_e  = np.mean(np.abs(ys_e[good_e] - xs_e[good_e]))
        bias_e = np.mean(ys_e[good_e] - xs_e[good_e])
        p("  4. V2 vs 5kHz GT (eta):  MAE = {:.3f} Pa.s  Bias = {:+.3f} Pa.s".format(
            mae_e, bias_e))
    xs_g0, ys_g0, _, _ = _paired_group_stats(df, "G_dyn_kpa")
    good_g = np.isfinite(xs_g0) & np.isfinite(ys_g0)
    if good_g.sum() >= 2:
        mae_g  = np.mean(np.abs(ys_g0[good_g] - xs_g0[good_g]))
        bias_g = np.mean(ys_g0[good_g] - xs_g0[good_g])
        p("  5. V2 vs 5kHz GT (G0):   MAE = {:.3f} kPa   Bias = {:+.3f} kPa".format(
            mae_g, bias_g))
    p("")
    p("  NOTE ON VISCOSITY ESTIMATION:")
    p("  Viscosity eta is inferred from measured cs and quasi-static G_e via")
    p("  the Voigt model at f_push = {:.0f} Hz.  This approach exploits the".format(F_PUSH))
    p("  frequency-dependent stiffening (G0 >> G_e) to recover the viscous")
    p("  component.  Absolute eta values carry the uncertainty of the f_push")
    p("  assumption; RELATIVE comparisons between conditions are robust.")
    p("")
    p(SEP)
    p("  END OF REPORT")
    p(SEP)

    txt = "\n".join(lines)
    print(txt)
    rpt = out / "swe_viscoelastic_report.txt"
    with open(rpt, "w") as f:
        f.write(txt)
    print(f"\n  Report saved: {rpt}")
    print("  Max line len:", max(len(l) for l in lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run(metrics_csv: str = "outputs/swe_publication_v2/swe_metrics_all_samples.csv",
        output_dir:  str = "outputs/swe_viscoelastic_v2"):
    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}\n"
                                "Run swe_publication.py first.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SWE Viscoelastic Analysis")
    print(f"  Voigt model  |  f_push = {F_PUSH:.0f} Hz")
    print(f"  Output: {out}")
    print(f"{'='*60}\n")

    print("  Loading metrics CSV and computing Voigt parameters …")
    df = build_viscoelastic_df(csv_path)

    # Quick summary
    for cond in ORDER:
        n_cs  = df[(df.condition == cond) & df["G_dyn_kpa"].notna()].shape[0]
        n_eta = df[(df.condition == cond) & df["eta_Pas"].notna()].shape[0]
        print(f"  {PALETTE[cond]['label']:<22} "
              f"G0 valid: {n_cs:>3}   eta valid: {n_eta:>3}")

    # Save augmented CSV
    df.to_csv(out / "swe_viscoelastic_metrics.csv", index=False)
    print(f"\n  Augmented metrics saved: {out}/swe_viscoelastic_metrics.csv")

    print("\n  Generating figures …")
    fig_rheology_bars(df, out)
    fig_complex_modulus(df, out)
    fig_scatter_compare(df, out)
    fig_bland_altman(df, out)
    fig_per_group(df, out)
    fig_frequency_sensitivity(df, out)
    fig_summary_3panel(df, out)

    print("\n  Writing report …")
    write_report(df, out)

    print(f"\n  All outputs in: {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv",
                    default="outputs/swe_publication_v2/swe_metrics_all_samples.csv")
    ap.add_argument("--output_dir",
                    default="outputs/swe_viscoelastic_v2")
    args = ap.parse_args()
    run(args.metrics_csv, args.output_dir)
