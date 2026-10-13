#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph-topology figures, saved under
results/figures/graph_topology/.

Figures:
  - K_betweenness_hub_ranking.png  horizontal bar of betweenness centrality
    per ROI on the K graph (from partA_K_graph_metrics.csv).
  - threshold_scan_density_assortativity.png  two-panel lines of mean degree
    and assortativity vs retained fraction for K and delta-K (from
    partB_threshold_scan_K.csv and partB_threshold_scan_deltaK.csv).
  - sliding_window_components_by_condition.png  box plot of mean connected
    components per condition (from partD_topology_timeseries_summary.csv).

Usage:
    py -3.11 analysis/graph_topology_figures.py
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures" / "graph_topology"
DATADIR = ROOT / "results" / "graph_topology"

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


def fig_betweenness():
    rows = [r for r in read_csv("partA_K_graph_metrics.csv")
            if r["roi"] != "GLOBAL_efficiency"]
    rows.sort(key=lambda r: float(r["betweenness"]))
    names = [r["roi"] for r in rows]
    vals = [float(r["betweenness"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    colors = []
    net_colors = {"BasalGanglia": "#4472C4", "Cerebellum": "#70AD47",
                  "Cortex": "#ED7D31", "TUS": "#A5A5A5"}
    for r in rows:
        colors.append(net_colors.get(r["network"], "#7F7F7F"))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Betweenness centrality (thresholded K graph)")
    ax.set_title("K network: betweenness hub ranking")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=n) for n, c in net_colors.items()]
    ax.legend(handles=handles, fontsize=8, title="Network", title_fontsize=9)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "K_betweenness_hub_ranking.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "K_betweenness_hub_ranking.svg", bbox_inches="tight")
    print("Saved K_betweenness_hub_ranking.{png,svg}")


def fig_threshold_scan():
    k_rows = read_csv("partB_threshold_scan_K.csv")
    d_rows = read_csv("partB_threshold_scan_deltaK.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, col, ylab, title in (
            (axes[0], "mean_degree", "Mean degree", "Mean degree vs retained fraction"),
            (axes[1], "assortativity", "Assortativity (deg-neighbor corr.)",
             "Assortativity vs retained fraction")):
        ax.plot([float(r["threshold_frac"]) for r in k_rows],
                [float(r[col]) for r in k_rows], marker="o", ms=4, lw=1.6,
                color="#4472C4", label="K")
        ax.plot([float(r["threshold_frac"]) for r in d_rows],
                [float(r[col]) for r in d_rows], marker="s", ms=4, lw=1.6,
                color="#ED7D31", label="$\\Delta$K")
        ax.set_xlabel("Retained connection fraction")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "threshold_scan_density_assortativity.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "threshold_scan_density_assortativity.svg", bbox_inches="tight")
    print("Saved threshold_scan_density_assortativity.{png,svg}")


def fig_sliding_window():
    rows = read_csv("partD_topology_timeseries_summary.csv")
    data = {c: [] for c in COND_ORDER}
    for r in rows:
        key = f"{r['target']}_{r['condition']}"
        if key in data:
            data[key].append(float(r["mean_components"]))
    positions = np.arange(len(COND_ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot([data[c] for c in COND_ORDER], positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], COND_ORDER):
        patch.set_facecolor(PALETTE[c])
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    ax.set_ylabel("Mean connected components (windowed correlation graph)")
    ax.set_title("Sliding-window topology by condition")
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "sliding_window_components_by_condition.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "sliding_window_components_by_condition.svg", bbox_inches="tight")
    print("Saved sliding_window_components_by_condition.{png,svg}")


if __name__ == "__main__":
    fig_betweenness()
    fig_threshold_scan()
    fig_sliding_window()