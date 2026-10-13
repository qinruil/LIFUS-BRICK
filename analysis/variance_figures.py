#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variance-decomposition figures, saved under
results/figures/variance_decomposition/.

Figures:
  - percent_K_by_band_and_condition.png  grouped bars of %K per frequency band
    for the four conditions (from freq_band_pooled.csv).
  - covariance_split_sensitivity.png    interval lines of %K as the shared
    covariance is assigned to K with theta in {0, 0.5, 1} (from
    split_rule_sensitivity.csv).
  - u_t_autocorrelation_pre_post.png     paired scatter of median lag-1
    autocorrelation pre vs post per subject (from u_autocorrelation.csv).
  - percent_K_vs_training_size.png       scatter of %K vs N_train
    (from scaling_points.csv).

Usage:
    py -3.11 analysis/variance_figures.py
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures" / "variance_decomposition"
DATADIR = ROOT / "results" / "variance_decomposition"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PALETTE = {"vim_pre": "#4472C4", "vim_post": "#8FAADC",
           "zi_pre": "#ED7D31", "zi_post": "#F4B183"}
COND_ORDER = ["vim_pre", "vim_post", "zi_pre", "zi_post"]
COND_LABEL = {"vim_pre": "VIM-pre", "vim_post": "VIM-post",
              "zi_pre": "ZI-pre", "zi_post": "ZI-post"}


def read_csv(name):
    with open(DATADIR / name, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fig_band_k():
    rows = read_csv("freq_band_pooled.csv")
    bands = ["delta", "theta", "alpha", "beta_fast"]
    data = {c: {b: float("nan") for b in bands} for c in COND_ORDER}
    for r in rows:
        key = f"{r['target']}_{r['condition']}"
        if key in data:
            data[key][r["band"]] = float(r["pct_K_pooled"])

    x = np.arange(len(bands))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, c in enumerate(COND_ORDER):
        vals = [data[c][b] for b in bands]
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=PALETTE[c],
               edgecolor="black", linewidth=0.5, label=COND_LABEL[c])
    ax.axhline(1.24, color="0.3", ls="--", lw=1.1,
               label="Broadband %K = 1.24")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("%K (variance share of propagated history)")
    ax.set_title("%K by frequency band and condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "percent_K_by_band_and_condition.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "percent_K_by_band_and_condition.svg", bbox_inches="tight")
    print("Saved percent_K_by_band_and_condition.{png,svg}")


def fig_split_sensitivity():
    rows = read_csv("split_rule_sensitivity.csv")
    thetas = [0.0, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, r in enumerate(rows):
        key = f"{r['target']}_{r['condition']}"
        vals = [float(r[f"pct_K_theta{th:g}"]) for th in thetas]
        ax.plot(thetas, vals, marker="o", ms=5, lw=1.6,
                color=PALETTE[key], label=COND_LABEL[key])
    ax.set_xlabel(r"$\theta$ (share of 2Cov assigned to K)")
    ax.set_ylabel("%K")
    ax.set_xticks(thetas)
    ax.set_xticklabels(["0\n(all to C)", "0.5", "1\n(all to K)"])
    ax.set_title("Sensitivity of %K to covariance assignment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "covariance_split_sensitivity.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "covariance_split_sensitivity.svg", bbox_inches="tight")
    print("Saved covariance_split_sensitivity.{png,svg}")


def fig_u_autocorr():
    rows = read_csv("u_autocorrelation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, tgt in zip(axes, ("vim", "zi")):
        d = {}
        for r in rows:
            if r["target"] != tgt:
                continue
            d.setdefault(r["subject_id"], {})[r["condition"]] = float(r["median_rho1"])
        pairs = [(v["pre"], v["post"]) for v in d.values() if "pre" in v and "post" in v]
        pre = np.array([p[0] for p in pairs])
        post = np.array([p[1] for p in pairs])
        ax.scatter(pre, post, s=28, color=PALETTE[f"{tgt}_pre"], edgecolor="black",
                   linewidth=0.4, alpha=0.85)
        lim = (min(pre.min(), post.min()) - 0.02, max(pre.max(), post.max()) + 0.02)
        ax.plot(lim, lim, ls="--", color="0.4", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Pre median $\\rho_1$(u)")
        ax.set_ylabel("Post median $\\rho_1$(u)")
        ax.set_title(f"{tgt.upper()}: u autocorrelation pre/post")
        ax.text(0.03, 0.95, f"n = {len(pairs)}",
                transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "u_t_autocorrelation_pre_post.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "u_t_autocorrelation_pre_post.svg", bbox_inches="tight")
    print("Saved u_t_autocorrelation_pre_post.{png,svg}")


def fig_scaling():
    rows = read_csv("scaling_points.csv")
    families = {}
    for r in rows:
        fam = r["checkpoint"].split("/")[0]
        families.setdefault(fam, []).append(r)
    fam_colors = {"loso": "#4472C4", "seed": "#70AD47", "taskstate": "#ED7D31"}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for fam, rs in families.items():
        xs = [int(r["n_train"]) + (0.08 if fam == "loso" else -0.08 if fam == "seed" else 0)
              for r in rs]
        ys = [float(r["pct_K"]) for r in rs]
        ax.scatter(xs, ys, s=34, color=fam_colors.get(fam, "#7F7F7F"),
                   label=fam, edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
        # annotate the two divergent folds
        if fam == "loso":
            for r, y in zip(rs, ys):
                sid = r["checkpoint"].split("/")[-1].replace("fold_sub-", "")
                if y > 50:
                    ax.annotate(sid, (int(r["n_train"]), y), textcoords="offset points",
                                xytext=(4, 4), fontsize=8)
    ax.set_xticks([13, 17])
    ax.set_xlabel("Training-set size, N_train")
    ax.set_ylabel("%K (variance share)")
    ax.set_title("%K vs training-set size (multi-checkpoint)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "percent_K_vs_training_size.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "percent_K_vs_training_size.svg", bbox_inches="tight")
    print("Saved percent_K_vs_training_size.{png,svg}")


if __name__ == "__main__":
    fig_band_k()
    fig_split_sensitivity()
    fig_u_autocorr()
    fig_scaling()