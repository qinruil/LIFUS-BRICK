#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K network visualisation.

Draws the reference-operator K coupling matrix as a weighted directed graph:
  - node radius  = betweenness centrality (top hubs labelled)
  - node colour  = network group (BasalGanglia / Cerebellum / Cortex / TUS)
  - edge opacity = coupling strength (top 30% kept)

Uses a fixed Kamada-Kawai-like circular-plus-core layout implemented with
numpy only (no networkx dependency). Edge positions are straight lines with
alpha scaled by weight.

Usage:
    py -3.11 analysis/K_network_graph.py
Output:
    results/figures/graph_topology/K_network_graph.png
    results/figures/graph_topology/K_network_graph.svg
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures" / "graph_topology"
DATADIR = ROOT / "results" / "graph_topology"
KCSV = ROOT / "results" / "final_model" / "figures_and_stats" / "K_region_coupling.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.5,
    "figure.dpi": 150,
})

NET_COLORS = {"BasalGanglia": "#4472C4", "Cerebellum": "#70AD47",
              "Cortex": "#ED7D31", "TUS": "#7030A0"}
NET_GROUPS = {
    "BasalGanglia": ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STH",
                     "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu", "rh_STH"],
    "Cerebellum": ["lh_cerebellum_dentate", "lh_cerebellum_motor",
                   "rh_cerebellum_dentate", "rh_cerebellum_motor"],
    "Cortex": ["lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
               "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
               "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
               "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"],
    "TUS": ["lh_vim", "lh_zi"],
}
NET_OF_ROI = {r: n for n, rois in NET_GROUPS.items() for r in rois}
HUB_THRESHOLD = 20.0


def read_k():
    with open(KCSV, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    labels = [r[0].strip() for r in rows[1:]]
    mat = np.array([[float(c) for c in r[1:]] for r in rows[1:]])
    return labels, mat


def read_metrics():
    with open(DATADIR / "partA_K_graph_metrics.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["roi"]: float(r["betweenness"]) for r in rows if r["roi"] != "GLOBAL_efficiency"}


def circular_pos(labels):
    n = len(labels)
    # group-averaged angular anchoring
    net_angles = {}
    for net, rois in NET_GROUPS.items():
        idx = [labels.index(r) for r in rois if r in labels]
        if idx:
            net_angles[net] = 2 * np.pi * np.mean(idx) / n
    pos = {}
    for i, r in enumerate(labels):
        base = 2 * np.pi * i / n
        pos[r] = np.array([np.cos(base), np.sin(base)])
    return pos


def main():
    labels, K = read_k()
    betw = read_metrics()
    pos = circular_pos(labels)

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    # edges: keep top 30% strong |K|, alpha scaled by weight
    A = np.abs(K)
    np.fill_diagonal(A, 0)
    thr = np.quantile(A[A > 0], 0.7) if (A > 0).any() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            w = A[i, j]
            if w >= thr and w > 0:
                p0, p1 = pos[labels[i]], pos[labels[j]]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color="0.55", lw=0.4 + 1.6 * (w / A.max()),
                        alpha=0.25 + 0.5 * (w / A.max()), zorder=1)
    # nodes
    for r in labels:
        x, y = pos[r]
        s = 260 + 320 * (betw.get(r, 0) / 70.0)
        net = NET_OF_ROI.get(r, "Other")
        ax.scatter(x, y, s=s, color=NET_COLORS.get(net, "#7F7F7F"),
                   edgecolor="black", linewidth=0.6, zorder=3, alpha=0.92)
        short = r.replace("lh_", "").replace("rh_", "").replace("_smooth3mm", "")
        if betw.get(r, 0) >= HUB_THRESHOLD or r in ("lh_vim", "lh_zi"):
            ax.annotate(short, (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, fontweight="bold",
                        zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("K network: betweenness hubs and coupling structure")
    handles = [Patch(color=c, label=n) for n, c in NET_COLORS.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=2, fontsize=9, frameon=False)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "K_network_graph.png", bbox_inches="tight")
    fig.savefig(FIGDIR / "K_network_graph.svg", bbox_inches="tight")
    print("Saved K_network_graph.{png,svg}")


if __name__ == "__main__":
    main()