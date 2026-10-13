#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrum-part figures (reference-level), saved under
results/figures/spectrum_stats/.

Figures:
  - modal_energy_magnitude_by_condition.png  grouped bars of energy-weighted
    |Lambda| mean per condition (from condition_participation.csv).
  - mode_matching_similarity_distribution.png histogram of matched-pair cosine
    vs random-pair baseline (from pre_post_matching.csv).
  - fold_topography_stability.png  per-fold mean similarity to the across-fold
    centroid, flagging subjects 09/15/19 (from fold_stability_by_fold.csv and
    fold_stability_by_rank.csv).

Usage:
    py -3.11 analysis/spectrum_figures.py
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures" / "spectrum_stats"
DATADIR = ROOT / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
COND_ORDER = ["vim_pre", "vim_post", "zi_pre", "zi_post"]
COND_LABEL = {"vim_pre": "VIM-pre", "vim_post": "VIM-post",
              "zi_pre": "ZI-pre", "zi_post": "ZI-post"}
PALETTE = {"vim_pre": "#4472C4", "vim_post": "#8FAADC",
           "zi_pre": "#ED7D31", "zi_post": "#F4B183"}


def read_csv_rows(name):
    p = DATADIR / name
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fig_energy_magnitude():
    rows = read_csv_rows("spectrum_stats/condition_participation.csv")
    groups = {c: [] for c in COND_ORDER}
    for r in rows:
        key = f"{r['target']}_{r['condition']}"
        if key in groups:
            groups[key].append(float(r["mag_mean"]))
    # subject-level paired values
    pairs = {c: {} for c in ("vim", "zi")}
    for r in rows:
        pairs.setdefault(r["target"], {}).setdefault(r["subject_id"], {})[r["condition"]] = float(r["mag_mean"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(len(COND_ORDER))
    means = [np.mean(groups[c]) for c in COND_ORDER]
    sems = [np.std(groups[c]) / np.sqrt(len(groups[c])) for c in COND_ORDER]
    bars = ax.bar(positions, means, yerr=sems, capsize=4, width=0.62,
                  color=[PALETTE[c] for c in COND_ORDER], edgecolor="black", linewidth=0.6)
    # paired connecting lines between pre and post within each target
    for target, off in (("vim", -0.5), ("zi", -0.5)):
        pass  # paired lines drawn per subject below
    for i, (target, cpre, cpost) in enumerate(
            (("vim", "vim_pre", "vim_post"), ("zi", "zi_pre", "zi_post"))):
        d = pairs[target]
        for sid, v in d.items():
            if "pre" in v and "post" in v:
                x0 = COND_ORDER.index(cpre) + (-0.08 if target == "vim" else 0.08)
                x1 = COND_ORDER.index(cpost) + (-0.08 if target == "vim" else 0.08)
                ax.plot([x0, x1], [v["pre"], v["post"]], color="0.55", lw=0.5, zorder=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    ax.set_ylabel("Energy-weighted $|\\Lambda|$ (mean)")
    ax.set_title("Modal energy persistence by condition")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.004, f"{m:.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "modal_energy_magnitude_by_condition.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "modal_energy_magnitude_by_condition.svg", bbox_inches="tight")
    print("Saved modal_energy_magnitude_by_condition.{png,svg}")


def fig_matching_distribution():
    rows = read_csv_rows("mode_participation/pre_post_matching.csv")
    matched = np.array([float(r["cos_sim"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.hist(matched, bins=24, color="#4472C4", edgecolor="white", alpha=0.85,
            label="Matched pairs (Hungarian)")
    ax.axvline(np.mean(matched), color="tab:red", lw=1.6, ls="--",
               label=f"Mean matched = {np.mean(matched):.3f}")
    ax.axvline(0.005, color="0.4", lw=1.2, ls=":", label="Random baseline = 0.005")
    ax.set_xlabel("Mean-centred cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Pre/post mode matching similarity")
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "mode_matching_similarity_distribution.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "mode_matching_similarity_distribution.svg", bbox_inches="tight")
    print("Saved mode_matching_similarity_distribution.{png,svg}")


def fig_fold_stability():
    folds = read_csv_rows("mode_participation/fold_stability_by_fold.csv")
    folds_sorted = sorted(folds, key=lambda r: float(r["mean_mode_sim_to_others"]))
    names = [f["fold"] for f in folds_sorted]
    vals = [float(f["mean_mode_sim_to_others"]) for f in folds_sorted]
    colors = []
    for n in names:
        sid = n.replace("sub-", "")
        colors.append("#ED7D31" if sid in ("fuspd09", "fuspd15", "fuspd19") else "#4472C4")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(np.median(vals), color="0.3", ls="--", lw=1,
               label=f"Median = {np.median(vals):.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean mode-similarity to across-fold centroid")
    ax.set_title("Across-fold mode-topography stability")
    ax.legend(fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4472C4", label="Retained folds"),
                       Patch(color="#ED7D31", label="Divergent folds (09/15/19)")],
              fontsize=9)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fold_topography_stability.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "fold_topography_stability.svg", bbox_inches="tight")
    print("Saved fold_topography_stability.{png,svg}")


if __name__ == "__main__":
    fig_energy_magnitude()
    fig_matching_distribution()
    fig_fold_stability()