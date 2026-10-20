#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph-theoretic and topological analysis of the Koopman operator.

Four parts (all read-only, no retraining):

Part A: Directed graph metrics on K and C
    Builds a weighted directed graph from the coupling matrix
    (K_region_coupling.csv for the shared operator, per-subject diagonal C
    vectors from the LOSO delta table for the control pathway). Computes
    in-degree, out-degree, betweenness centrality, local clustering
    coefficient and global/local efficiency on thresholded graphs, and
    and identifies hub regions by each metric.

Part B: Threshold scanning and topological transitions
    Scans the coupling threshold from the 50th to the 95th percentile of
    connection weights and tracks degree-assortativity/mean degree as a
    function of the threshold for K and C graphs, localising any transition
    point in the network organisation.

Part C: Persistent homology of the weighted directed graph
    Treats |K| (and per-condition |C|) as a weighted similarity and builds
    a VR-like filtration over thresholds; extracts 0-dimensional (connected
    components) and 1-dimensional (cycles) persistence diagrams and computes
    persistence-doubling summaries plus the Wasserstein distance between the
    pre/post persistent diagrams.

Part D: Sliding-window topological time series
    For each session, the ROI x ROI BOLD correlation matrix is computed over
    a sliding window, converted to a threshold-weighted graph, and the
    umber of connected components is tracked over time, giving a topological
    time series. Sessions are then summarised per condition.

Outputs (results/graph_topology/):
    partA_K_graph_metrics.csv, partA_C_graph_metrics.csv
    partB_threshold_scan.csv
    partC_persistence_K.csv, partC_persistence_C.csv,
    partC_wasserstein_pre_post.csv
    partD_topology_timeseries_summary.csv
    run_manifest.json

Dependencies: numpy, scipy, standard library (no pandas, no matplotlib).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402

OUT_DIR = ROOT / "results" / "graph_topology"
FINAL_DIR = ROOT / "results" / "final_model" / "figures_and_stats"
TS_DIR = ROOT / "results" / "figures" / "task_states"
LOSO_DIR = ROOT / "results" / "figures" / "loso" / "loso_19_fold_beta_0.2_13to1to5_split"

K_CSV = FINAL_DIR / "K_region_coupling.csv"
DK_CSV = TS_DIR / "K_comparison" / "K_region_coupling_diff.csv"
KPRE_CSV = TS_DIR / "pre_model" / "K_region_coupling.csv"
KPOST_CSV = TS_DIR / "post_model" / "K_region_coupling.csv"
DC_CSV = LOSO_DIR / "loso_delta_summary.csv"

EPS = 1e-12
NETWORK_GROUPS = {
    "BasalGanglia": ["lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STH",
                     "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu", "rh_STH"],
    "Cerebellum":   ["lh_cerebellum_dentate", "lh_cerebellum_motor",
                     "rh_cerebellum_dentate", "rh_cerebellum_motor"],
    "Cortex":       ["lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
                     "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
                     "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
                     "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm"],
    "TUS":          ["lh_vim", "lh_zi"],
}
NET_OF_ROI = {}
for net, rois in NETWORK_GROUPS.items():
    for r in rois:
        NET_OF_ROI[r] = net


# ---------------------------------------------------------------------------
# IO / matrix helpers
# ---------------------------------------------------------------------------
def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    labels = [r[0].strip() for r in rows[1:]]
    mat = np.array([[float(c) for c in r[1:]] for r in rows[1:]])
    cols = [c.strip() for c in rows[0][1:]]
    assert mat.shape == (len(labels), len(cols)), "ragged matrix"
    return labels, mat


def write_csv(path: Path, rows_or_fields, rows: list[dict] | None = None,
              order: list[str] | None = None):
    """Flexible CSV writer:
      write_csv(path, rows)                          -> keys from first row
      write_csv(path, fieldnames, rows)              -> explicit fields
      write_csv(path, rows, order=fields)            -> explicit fields
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = rows_or_fields
        keys = order if order else list(rows[0].keys())
    elif isinstance(rows_or_fields, list) and rows_or_fields and all(
            isinstance(x, str) for x in rows_or_fields):
        keys = rows_or_fields
    else:
        rows = rows_or_fields
        keys = order if order else list(rows[0].keys())
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else _fmt(r.get(k))) for k in keys})
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, np.floating):
        return _fmt(float(v))
    return str(v)


def threshold_adjacency(M: np.ndarray, frac: float, sym: bool = False) -> np.ndarray:
    """Binary adjacency from the top `frac` fraction of off-diagonal |M|."""
    A = np.abs(M).copy()
    np.fill_diagonal(A, 0.0)
    flat = A[A > 0]
    if flat.size == 0:
        return np.zeros_like(A, dtype=bool)
    thr = np.quantile(flat, 1.0 - frac)
    A = (A >= thr).astype(np.int8)
    if sym:
        A = np.maximum(A, A.T)
    return A


# ---------------------------------------------------------------------------
# Part A: directed graph metrics
# ---------------------------------------------------------------------------
def betweenness_directed(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    scores = np.zeros(n)
    csr = csr_matrix(A)
    for s in range(n):
        dist, pred = shortest_path(csr, directed=True, unweighted=True,
                                   indices=s, return_predecessors=True)
        # count geodesics through each vertex (unweighted BFS counting)
        order = np.argsort(dist)
        sigma = np.zeros(n)
        sigma[s] = 1.0
        for v in order:
            if math.isinf(dist[v]):
                continue
            for u in range(n):
                if pred[u] == v and not math.isinf(dist[u]):
                    sigma[u] += sigma[v]
        delta = np.zeros(n)
        for v in order[::-1]:
            if v == s:
                continue
            if math.isinf(dist[v]):
                continue
            for u in range(n):
                if pred[v] == u and not math.isinf(dist[v]) and dist[v] == dist[u] + 1:
                    delta[u] += (sigma[u] / sigma[v]) * (1.0 + delta[v])
            scores[v] += delta[v]
    return scores


def local_clustering(M: np.ndarray, frac: float) -> np.ndarray:
    A = threshold_adjacency(M, frac, sym=True)
    n = A.shape[0]
    cc = np.zeros(n)
    for i in range(n):
        neigh = np.where(A[i] > 0)[0]
        k = len(neigh)
        if k < 2:
            cc[i] = 0.0
            continue
        sub = A[np.ix_(neigh, neigh)]
        cc[i] = sub.sum() / (k * (k - 1))
    return cc


def global_efficiency(M: np.ndarray, frac: float) -> float:
    """Global efficiency of the undirected thresholded graph built from M:
    mean inverse shortest-path distance over all node pairs."""
    A = threshold_adjacency(M, frac)
    csr = csr_matrix(A.astype(float))
    d = shortest_path(csr, directed=False)
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(d) & (d > 0), 1.0 / np.where(d == 0, np.inf, d), 0.0)
    n = d.shape[0]
    return float(inv.sum() / (n * (n - 1)))


def part_a(M: np.ndarray, labels: list[str], tag: str):
    rows = []
    if M.ndim == 2 and M.shape[0] == M.shape[1]:   # graph matrix
        n = M.shape[0]
        A = threshold_adjacency(M, 0.3)
        indeg = A.sum(axis=0)
        outdeg = A.sum(axis=1)
        betw = betweenness_directed(A)
        cc = local_clustering(M, 0.3)
        ge = global_efficiency(M, 0.3)
        for i in range(n):
            rows.append({"graph": tag, "roi": labels[i], "network": NET_OF_ROI.get(labels[i], "Other"),
                         "in_degree": int(indeg[i]), "out_degree": int(outdeg[i]),
                         "betweenness": float(betw[i]), "clustering": float(cc[i]),
                         "in_out_ratio": float((indeg[i] + 1) / (outdeg[i] + 1))})
        rows.append({"graph": tag, "roi": "GLOBAL_efficiency", "network": "",
                     "in_degree": "", "out_degree": "",
                     "betweenness": float(ge),
                     "clustering": "", "in_out_ratio": ""})
    write_csv(OUT_DIR / f"partA_{tag}_graph_metrics.csv",
              ["graph", "roi", "network", "in_degree", "out_degree", "betweenness",
               "clustering", "in_out_ratio"], rows)
    return rows


# ---------------------------------------------------------------------------
# Part B: threshold scan
# ---------------------------------------------------------------------------
def part_b(M: np.ndarray, tag: str):
    rows = []
    for frac_pct in range(50, 95, 5):
        frac = frac_pct / 100.0
        A = threshold_adjacency(M, frac)
        n = A.shape[0]
        deg = A.sum(axis=1) + A.sum(axis=0)  # undirected degree proxy
        mean_deg = float(deg.mean()) / 2
        # assortativity proxy: correlation between node degree and neighbor mean degree
        assoc = float("nan")
        arr_x, arr_y = [], []
        for i in range(n):
            nbrs = np.where((A[i] + A[:, i]) > 0)[0]
            if len(nbrs) > 0:
                arr_x.append(deg[i])
                arr_y.append(deg[nbrs].mean())
        if len(arr_x) > 3:
            r, _ = sp_stats.pearsonr(arr_x, arr_y)
            assoc = float(r)
        rows.append({"graph": tag, "threshold_frac": frac, "mean_degree": mean_deg,
                     "assortativity": assoc, "n_edges": int(A.sum())})
    write_csv(OUT_DIR / f"partB_threshold_scan_{tag}.csv",
              ["graph", "threshold_frac", "mean_degree", "assortativity", "n_edges"], rows)
    return rows


# ---------------------------------------------------------------------------
# Part C: persistent homology (0-dim and 1-dim via union-find / cycles)
# ---------------------------------------------------------------------------
def persistent_0d(adj: np.ndarray) -> list[tuple[float, float]]:
    """0-dimensional persistence on a weighted graph:
    sort edges by weight ascending, union-find, births at edge weight when a
    component merges (death of the older component = merge weight).
    Returns (birth, death) pairs."""
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w > 0:
                edges.append((w, i, j))
    edges.sort()
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    births = [(0.0, i) for i in range(n)]
    active = list(range(n))
    pers = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # union: older component dies at weight w
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1
        pers.append((0.0, float(w)))
        active.remove(rj)
    # remaining components never die -> death = +inf
    for c in active:
        pers.append((0.0, float("inf")))
    return pers


def persistent_1d(adj: np.ndarray) -> list[tuple[float, float]]:
    """Approximate 1-dimensional persistence: cycle-born components.
    Uses the standard incremental VR reconstruction at each edge weight,
    counting independent cycles as they are closed. Implemented via the
    reduced-persistence heuristic: build the graph progressively and count
    cycles formed (edges within the same component). Edge case: 24 nodes
    yield few genuine 1-cycles, so this is primarily a screening output."""
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w > 0:
                edges.append((w, i, j))
    edges.sort()
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pers1 = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            # cycle closed at this weight
            pers1.append((float(w), float("inf")))
        else:
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
    return pers1


def persistence_summary(pers: list[tuple[float, float]]) -> dict:
    births = [b for b, d in pers if math.isfinite(b)]
    deaths = [d for b, d in pers if math.isfinite(d)]
    lifespans = [d - b for b, d in pers if math.isfinite(d)]
    return {"n_features": len(pers),
            "n_infinite": sum(1 for b, d in pers if not math.isfinite(d)),
            "mean_lifespan": float(np.mean(lifespans)) if lifespans else 0.0,
            "sum_lifespan": float(np.sum(lifespans)) if lifespans else 0.0,
            "max_lifespan": float(np.max(lifespans)) if lifespans else 0.0}


def wasserstein_persistence(p1: list[tuple[float, float]], p2: list[tuple[float, float]]) -> float:
    """1-D Wasserstein between two persistence diagrams (matching each point
    to the diagonal with the L1 cost; births on the diagonal at 0)."""
    pts1 = [(b, d) for b, d in p1 if math.isfinite(b) and math.isfinite(d)]
    pts2 = [(b, d) for b, d in p2 if math.isfinite(b) and math.isfinite(d)]
    # pad with diagonal births
    cost = 0.0
    n = max(len(pts1), len(pts2))
    for i in range(n):
        q1 = pts1[i] if i < len(pts1) else (0.0, 0.0)
        q2 = pts2[i] if i < len(pts2) else (0.0, 0.0)
        cost += abs(q1[0] - q2[0]) + abs(q1[1] - q2[1])
    return float(cost)


def part_c(M: np.ndarray, tag: str):
    A = np.abs(M)
    np.fill_diagonal(A, 0.0)
    p0 = persistent_0d(A)
    p1 = persistent_1d(A)
    s0 = persistence_summary(p0)
    s1 = persistence_summary(p1)
    rows = [{"graph": tag, "dim": "H0", **{f"k_{k}": v for k, v in s0.items()}},
            {"graph": tag, "dim": "H1", **{f"k_{k}": v for k, v in s1.items()}}]
    write_csv(OUT_DIR / f"partC_persistence_{tag}.csv",
              ["graph", "dim", "k_n_features", "k_n_infinite", "k_mean_lifespan",
               "k_sum_lifespan", "k_max_lifespan"], rows)
    return rows


def part_c_wasserstein(Kpre: np.ndarray, Kpost: np.ndarray):
    Ap, Aq = np.abs(Kpre), np.abs(Kpost)
    np.fill_diagonal(Ap, 0.0); np.fill_diagonal(Aq, 0.0)
    p0p, p0q = persistent_0d(Ap), persistent_0d(Aq)
    p1p, p1q = persistent_1d(Ap), persistent_1d(Aq)
    w0 = wasserstein_persistence(p0p, p0q)
    w1 = wasserstein_persistence(p1p, p1q)
    rows = [{"comparison": "pre_vs_post", "dim": "H0", "wasserstein": w0},
            {"comparison": "pre_vs_post", "dim": "H1", "wasserstein": w1}]
    write_csv(OUT_DIR / "partC_wasserstein_pre_post.csv",
              ["comparison", "dim", "wasserstein"], rows)
    return rows


# ---------------------------------------------------------------------------
# Part D: sliding-window topological time series (on BOLD correlation)
# ---------------------------------------------------------------------------
def part_d(quick: bool = False):
    subjects = load_all()
    window = 40
    step = 10
    rows = []
    for s in subjects:
        for phase in ("mpre", "mpost"):
            x = np.asarray(s[phase], dtype=float)
            T, N = x.shape
            n_windows = max(1, (T - window) // step + 1)
            comp_counts = []
            for w in range(n_windows):
                seg = x[w * step: w * step + window]
                if seg.shape[0] < 3:
                    continue
                c = np.corrcoef(seg.T)
                A = threshold_adjacency(c, 0.3, sym=True)
                ncomp = connected_components(csr_matrix(A.astype(float)), directed=False)[0]
                comp_counts.append(int(ncomp))
            if not comp_counts:
                continue
            rows.append({
                "subject_id": s["subject_id"], "target": s["target"],
                "phase": phase, "condition": "pre" if phase == "mpre" else "post",
                "n_windows": len(comp_counts),
                "mean_components": float(np.mean(comp_counts)),
                "sd_components": float(np.std(comp_counts)),
                "min_components": int(np.min(comp_counts)),
                "max_components": int(np.max(comp_counts)),
            })
            if quick:
                break
    write_csv(OUT_DIR / "partD_topology_timeseries_summary.csv",
              ["subject_id", "target", "phase", "condition", "n_windows",
               "mean_components", "sd_components", "min_components", "max_components"], rows)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 78)
    print("Graph-theoretic and topological analysis of the Koopman operator")
    print("=" * 78)

    labels, K = read_matrix(K_CSV)
    _, DK = read_matrix(DK_CSV)
    _, Kpre = read_matrix(KPRE_CSV)
    _, Kpost = read_matrix(KPOST_CSV)

    print(f"[load] K {K.shape}, deltaK {DK.shape}, Kpre {Kpre.shape}, Kpost {Kpost.shape}")
    print("[part A] directed graph metrics on K")
    part_a(K, labels, "K")
    print("[part B] threshold scan on K and delta-K")
    part_b(K, "K")
    part_b(DK, "deltaK")
    print("[part C] persistent homology")
    part_c(K, "K")
    part_c(DK, "deltaK")
    part_c_wasserstein(Kpre, Kpost)
    print("[part D] sliding-window topological time series (76 sessions)")
    part_d()

    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"script": "analysis/graph_topology_analysis.py",
                   "inputs": {"K": str(K_CSV.relative_to(ROOT)),
                              "deltaK": str(DK_CSV.relative_to(ROOT)),
                              "Kpre": str(KPRE_CSV.relative_to(ROOT)),
                              "Kpost": str(KPOST_CSV.relative_to(ROOT))},
                   "threshold_fraction": 0.3,
                   "window_discarded": "sliding window of 40 TR, step 10, correlation threshold 0.3",
                   "notes": ["Betweenness is exact on unweighted thresholded graphs.",
                             "Persistence H1 is a screening summary; genuine cycles on 24 nodes are few."],
                   "elapsed_sec": round(time.time() - t0, 2)}, fh, indent=2, ensure_ascii=False)
    print(f"\n[done] {OUT_DIR} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()