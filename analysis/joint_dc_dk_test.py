#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint test of delta-C (local control gain) and delta-K (network propagation).

Purpose
-------
The main report keeps delta-C and delta-K in separate chapters and never couples
them on a common unit. This script performs the coupling tests that the existing
saved tables support, WITHOUT retraining any model.

Inputs (all pre-existing files, read only)
    results/figures/loso/loso_19_fold_beta_0.2_13to1to5_split/loso_delta_summary.csv
        19 subjects x {vim, zi} x 24 ROIs of roi_c_pre / roi_c_post / delta.
        Inference unit: subject (held-out LOSO fold).
    results/figures/task_states/K_comparison/K_region_coupling_diff.csv
        24 x 24 matrix, value = ||K_post block|| - ||K_pre block||,
        rows = target ROI, cols = source ROI.
    results/figures/task_states/{pre_model,post_model}/K_region_coupling.csv
        Absolute block-coupling norms per model, used for relative (scale-free)
        change features. Optional; skipped if absent.

Outputs (written to results/delta_c_k_joint/, created if needed)
    region_feature_table.csv      per-ROI delta-K features + mean delta-C
    subject_level_correlation.csv Spearman rho per subject x target x feature
    pooled_regression.csv         slope with subject-clustered bootstrap CI
    region_permutation.csv        ROI-permutation test for every feature
    target_dissociation.csv       rho(dC_vim, dK) vs rho(dC_zi, dK) contrast
    fdr_summary.csv               BH correction over the declared families
    run_manifest.json             provenance, versions, decisions

HARD CONSTRAINT ON THE INFERENCE UNIT (stated in every output table)
    delta-K is a single shared 24x24 matrix computed once from two independently
    trained models (pre_model, post_model). It carries NO subject dimension.
    Therefore "same-subject coupling of delta-C and delta-K" is NOT computable.
    What this script can legitimately test is whether, ACROSS THE 24 REGIONS,
    a region's delta-K profile predicts its delta-C profile. The delta-K side of
    that test has n = 24 regions with n = 1 operator, so the operator-level
    uncertainty (two independently trained nets, run-to-run optimisation noise)
    is entirely unrepresented. Every p-value below must be reported as
    hypothesis-generating, consistent with the main report's own downgrade of
    delta-K to descriptive status.

Usage
    python analysis/joint_dc_dk_test.py [--n-perm 5000] [--seed 20261006]

Dependencies: numpy, scipy, standard library only (pandas and matplotlib are
NOT importable in this environment).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
LOSO_DIR = ROOT / "results" / "figures" / "loso" / "loso_19_fold_beta_0.2_13to1to5_split"
DC_CSV = LOSO_DIR / "loso_delta_summary.csv"
TASK_DIR = ROOT / "results" / "figures" / "task_states"
DK_DIFF_CSV = TASK_DIR / "K_comparison" / "K_region_coupling_diff.csv"
DK_PRE_CSV = TASK_DIR / "pre_model" / "K_region_coupling.csv"
DK_POST_CSV = TASK_DIR / "post_model" / "K_region_coupling.csv"
OUT_DIR = ROOT / "results" / "delta_c_k_joint"

# Feature definitions. signed = post - pre sums; abs = total rewired magnitude;
# rel = signed change divided by pre level (scale free).
FEATURES = [
    "dk_self",          # diagonal block change (within-region)
    "dk_in_signed",     # net change in influence received by ROI (row sum)
    "dk_out_signed",    # net change in influence emitted by ROI (col sum)
    "dk_in_abs",        # sum |change| over incoming edges, off-diagonal
    "dk_out_abs",       # sum |change| over outgoing edges, off-diagonal
    "dk_in_rel",        # (post - pre) received / pre received
    "dk_out_rel",       # (post - pre) emitted / pre emitted
]
PRIMARY_TARGETS = ["vim", "zi"]


# --------------------------------------------------------------------------- #
# Generic CSV helpers
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fieldnames})


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (np.floating, float)):
        if math.isnan(float(v)):
            return "nan"
        return f"{float(v):.6g}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.ndarray, list, tuple)):
        return ";".join(_fmt(x) for x in v)
    return str(v)


def read_square_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a CSV whose first column holds row labels and remaining cells are numeric."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    header = rows[0]
    labels = [r[0].strip() for r in rows[1:]]
    mat = np.array([[float(c) for c in r[1:]] for r in rows[1:]], dtype=float)
    cols = [h.strip() for h in header[1:]]
    assert mat.shape == (len(labels), len(cols)), f"{path.name}: ragged matrix"
    return labels, cols, mat


# --------------------------------------------------------------------------- #
# Statistics helpers (hand written: no statsmodels, no pandas)
# --------------------------------------------------------------------------- #
def wilcoxon_p(a, b=None) -> float:
    """Two-sided Wilcoxon p-value, tolerant of scipy version differences.

    scipy >= 1.26 dropped the ``zero_mode`` keyword, so the parameter is not
    passed; zero differences are dropped up front instead (ztest default).
    Returns nan when the test is undefined (all differences zero, or too few).
    """
    a = np.asarray(a, dtype=float)
    if b is None:
        d = a
    else:
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0.0]
    if d.size < 5:
        return float("nan")
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values, same convention as koopman_spectrum_stats.py."""
    n = len(pvals)
    order = np.argsort(np.asarray(pvals, dtype=float))
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        val = min(prev, pvals[idx] * n / rank)
        q[idx] = val
        prev = val
    return [float(x) for x in q]


def spearman_or_nan(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    res = stats.spearmanr(x, y)
    return float(res.correlation), float(res.pvalue)


def fisher_z(rho: float) -> float:
    rho = max(min(rho, 0.999999), -0.999999)
    return float(np.arctanh(rho))


def cluster_bootstrap_slope(x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                            n_boot: int, rng: np.random.Generator) -> dict:
    """OLS slope of y ~ x, resampling whole clusters (subjects) with replacement."""
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}

    def slope(sel):
        xs, ys = x[sel], y[sel]
        if len(xs) < 3 or np.allclose(xs, xs[0]):
            return float("nan")
        xm, ym = xs - xs.mean(), ys - ys.mean()
        denom = float((xm * xm).sum())
        return float((xm * ym).sum() / denom) if denom > 0 else float("nan")

    full = np.arange(len(x))
    obs = slope(full)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_cluster[c] for c in pick])
        s = slope(sel)
        if not math.isnan(s):
            boots.append(s)
    boots = np.asarray(boots, dtype=float)
    if boots.size < 20:
        return {"slope": obs, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_boot": float("nan"), "n_boot_valid": int(boots.size)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled slope crosses zero
    p_boot = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"slope": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": float(min(1.0, p_boot)), "n_boot_valid": int(boots.size)}


def permutation_rho(x: np.ndarray, y: np.ndarray, n_perm: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    """ROI-label permutation test: shuffle y across regions, compare |rho|."""
    obs_r, _ = spearman_or_nan(x, y)
    if math.isnan(obs_r):
        return float("nan"), float("nan")
    n = len(y)
    hits = 0
    done = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r, _ = spearman_or_nan(x, yp)
        if math.isnan(r):
            continue
        done += 1
        if abs(r) >= abs(obs_r) - 1e-12:
            hits += 1
    if done == 0:
        return obs_r, float("nan")
    return obs_r, float((hits + 1) / (done + 1))


# --------------------------------------------------------------------------- #
# Load delta-C
# --------------------------------------------------------------------------- #
def load_delta_c() -> dict:
    rows = read_rows(DC_CSV)
    if not rows:
        raise SystemExit(f"empty delta-C table: {DC_CSV}")
    roi_order = []
    for r in rows:
        if r["roi"] not in roi_order:
            roi_order.append(r["roi"])
    subjects = sorted({r["subject_id"] for r in rows})
    targets = sorted({r["target"] for r in rows})

    delta = {}          # (subject, target) -> np.array over roi_order
    pre = {}
    post = {}
    checks = []
    for subj in subjects:
        for tgt in targets:
            cell = {(r["roi"]): r for r in rows if r["subject_id"] == subj and r["target"] == tgt}
            if not cell:
                continue
            miss = [roi for roi in roi_order if roi not in cell]
            if miss:
                checks.append(f"INCOMPLETE {subj}/{tgt} missing {len(miss)} ROIs -> skipped")
                continue
            p = np.array([float(cell[r]["roi_c_pre"]) for r in roi_order])
            q = np.array([float(cell[r]["roi_c_post"]) for r in roi_order])
            d = np.array([float(cell[r]["delta"]) for r in roi_order])
            # integrity check: stored delta must equal post - pre
            resid = float(np.max(np.abs(d - (q - p))))
            if resid > 1e-6:
                checks.append(f"DELTA MISMATCH {subj}/{tgt} max|stored-(post-pre)|={resid:.2e}")
            delta[(subj, tgt)] = d
            pre[(subj, tgt)] = p
            post[(subj, tgt)] = q
    return {"roi_order": roi_order, "subjects": subjects, "targets": targets,
            "delta": delta, "pre": pre, "post": post, "checks": checks,
            "n_rows": len(rows)}


# --------------------------------------------------------------------------- #
# Load delta-K features
# --------------------------------------------------------------------------- #
def load_delta_k(roi_order: list[str]) -> tuple[dict, dict]:
    if not DK_DIFF_CSV.exists():
        raise SystemExit(f"delta-K matrix not found: {DK_DIFF_CSV}")
    r_lab, c_lab, D = read_square_matrix(DK_DIFF_CSV)
    n = len(roi_order)
    if len(r_lab) != n or len(c_lab) != n:
        raise SystemExit(f"delta-K is {len(r_lab)}x{len(c_lab)}, expected {n}x{n}")
    if set(r_lab) != set(roi_order) or set(c_lab) != set(roi_order):
        extra_k = sorted(set(r_lab) ^ set(roi_order))
        raise SystemExit(f"ROI label mismatch between delta-C and delta-K: {extra_k}")

    # reindex into roi_order for both axes
    ridx = [r_lab.index(r) for r in roi_order]
    cidx = [c_lab.index(r) for r in roi_order]
    D = D[np.ix_(ridx, cidx)]
    # D rows = target ROI (receiver), cols = source ROI (emitter)
    notes = {"dk_shape": list(D.shape),
             "dk_row_mean": float(D.mean()),
             "dk_abs_mean": float(np.abs(D).mean()),
             "dk_max_inc": float(D.max()),
             "dk_max_dec": float(D.min())}

    P = Q = None
    if DK_PRE_CSV.exists() and DK_POST_CSV.exists():
        pr_lab, pr_col, P = read_square_matrix(DK_PRE_CSV)
        qo_lab, qo_col, Q = read_square_matrix(DK_POST_CSV)
        if set(pr_lab) == set(roi_order) and set(pr_col) == set(roi_order):
            P = P[np.ix_([pr_lab.index(r) for r in roi_order],
                         [pr_col.index(r) for r in roi_order])]
            Q = Q[np.ix_([qo_lab.index(r) for r in roi_order],
                         [qo_col.index(r) for r in roi_order])]
        else:
            notes["abs_tables"] = "ROI labels of absolute coupling tables differ; rel features skipped"
            P = Q = None

    off = ~np.eye(n, dtype=bool)
    feats: dict[str, np.ndarray] = {}
    feats["dk_self"] = np.diag(D).copy()
    feats["dk_in_signed"] = D.sum(axis=1)                      # received
    feats["dk_out_signed"] = D.sum(axis=0)                     # emitted
    feats["dk_in_abs"] = (np.abs(D) * off).sum(axis=1)
    feats["dk_out_abs"] = (np.abs(D) * off).sum(axis=0)

    if P is not None:
        pre_in = (P * off).sum(axis=1)
        pre_out = (P * off).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            feats["dk_in_rel"] = ((D * off).sum(axis=1)) / np.where(pre_in == 0, np.nan, pre_in)
            feats["dk_out_rel"] = ((D * off).sum(axis=0)) / np.where(pre_out == 0, np.nan, pre_out)
        notes["abs_tables_used"] = True
    else:
        feats["dk_in_rel"] = np.full(n, np.nan)
        feats["dk_out_rel"] = np.full(n, np.nan)
        notes["abs_tables_used"] = False

    return feats, notes


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def subject_level(dc: dict, feats: dict, rng, n_boot: int) -> list[dict]:
    """For each (subject, target): Spearman rho across 24 ROIs, then one-sample
    test of Fisher-z over the 19 subjects."""
    out = []
    roi_order = dc["roi_order"]
    for tgt in PRIMARY_TARGETS:
        for feat in FEATURES:
            x = feats[feat]
            if np.all(np.isnan(x)):
                continue
            zs, rhos, cells = [], [], []
            for subj in dc["subjects"]:
                d = dc["delta"].get((subj, tgt))
                if d is None:
                    continue
                mask = ~np.isnan(x)
                rho, _ = spearman_or_nan(d[mask], x[mask])
                if math.isnan(rho):
                    continue
                rhos.append(rho)
                zs.append(fisher_z(rho))
                cells.append({"layer": "subject_level", "target": tgt, "feature": feat,
                              "unit": "subject", "id": subj, "rho": rho,
                              "inference_note": "rho over 24 ROIs; shared delta-K"})
            if len(zs) < 3:
                continue
            z = np.asarray(zs)
            tt = stats.ttest_1samp(z, 0.0)
            p_w = wilcoxon_p(z)
            mean_rho = float(np.mean(rhos))
            # bootstrap CI on mean rho, resampling subjects
            boots = []
            for _ in range(n_boot):
                pick = rng.integers(0, len(rhos), size=len(rhos))
                boots.append(float(np.mean(np.asarray(rhos)[pick])))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({"layer": "subject_level", "target": tgt, "feature": feat,
                        "unit": "subject", "id": "AGGREGATE",
                        "n_subjects": len(zs), "mean_rho": mean_rho,
                        "ci_lo": float(lo), "ci_hi": float(hi),
                        "mean_fisher_z": float(z.mean()),
                        "t_stat": float(tt.statistic), "p_t": float(tt.pvalue),
                        "p_wilcoxon": p_w,
                        "sign_consistency_frac": float(np.mean(np.asarray(rhos) > 0)),
                        "inference_note": ("delta-C n=19 subjects; delta-K shared (n=1 operator). "
                                           "Unit of the rho is subject, but the predictor has no subject variance")})
            out.extend(cells)
    return out


def region_permutation(dc: dict, feats: dict, n_perm: int, rng) -> list[dict]:
    """Pool delta-C per ROI (mean over subjects, within target) and test rho
    against each delta-K feature by permuting ROI labels."""
    out = []
    roi_order = dc["roi_order"]
    n = len(roi_order)
    for tgt in PRIMARY_TARGETS:
        stacks = []
        for subj in dc["subjects"]:
            d = dc["delta"].get((subj, tgt))
            if d is not None:
                stacks.append(d)
        if not stacks:
            continue
        mean_d = np.mean(np.vstack(stacks), axis=0)
        for feat in FEATURES:
            x = feats[feat]
            mask = ~np.isnan(x)
            if mask.sum() < 5:
                continue
            rho, p_param = permutation_rho(mean_d[mask], x[mask], n_perm, rng)
            out.append({"layer": "region_permutation", "target": tgt, "feature": feat,
                        "unit": "region", "id": "AGGREGATE",
                        "n_regions": int(mask.sum()), "rho": rho,
                        "p_perm": p_param, "p_parametric": spearman_or_nan(mean_d[mask], x[mask])[1],
                        "inference_note": f"predictor is one shared operator; n={int(mask.sum())} regions, n=1 operator"})
    return out


def pooled_regression(dc: dict, feats: dict, rng, n_boot: int) -> list[dict]:
    """All 19x24 rows per target stacked; slope of delta-C on the delta-K feature
    with subject-clustered bootstrap CI (clusters are dependent within subject)."""
    out = []
    for tgt in PRIMARY_TARGETS:
        rows_y, rows_x, rows_c = {f: [] for f in FEATURES}, None, []
        for i, subj in enumerate(dc["subjects"]):
            d = dc["delta"].get((subj, tgt))
            if d is None:
                continue
            rows_c.append(np.full(len(d), subj))
            for f in FEATURES:
                rows_x = feats[f]
                rows_y[f].append(d)
        clusters_all = np.concatenate(rows_c) if rows_c else np.array([])
        for f in FEATURES:
            if not rows_y[f]:
                continue
            x = feats[f]
            y = np.concatenate(rows_y[f])
            cl = clusters_all
            mask_full = ~np.isnan(x)
            if mask_full.sum() < 5:
                continue
            rep = cluster_bootstrap_slope(np.tile(x[mask_full], len(rows_y[f])),
                                          y.reshape(len(rows_y[f]), -1)[:, mask_full].ravel(),
                                          np.repeat(np.arange(len(rows_y[f])), int(mask_full.sum())),
                                          n_boot, rng)
            rho, _ = spearman_or_nan(np.tile(x[mask_full], len(rows_y[f])),
                                     y.reshape(len(rows_y[f]), -1)[:, mask_full].ravel())
            out.append({"layer": "pooled_regression", "target": tgt, "feature": f,
                        "unit": "subject x region", "id": "AGGREGATE",
                        "n_obs": int(y.size), "n_clusters": int(len(set(cl.tolist()))),
                        "slope": rep["slope"], "ci_lo": rep["ci_lo"], "ci_hi": rep["ci_hi"],
                        "p_cluster_boot": rep["p_boot"], "spearman_rho": rho,
                        "inference_note": "regions within a subject are not independent; CI from subject-cluster bootstrap"})
    return out


def target_dissociation(dc: dict, feats: dict, rng, n_boot: int) -> list[dict]:
    """Is rho(deltaC_vim, dKfeature) systematically different from
    rho(deltaC_zi, dKfeature)? Compare at subject level with paired resampling."""
    out = []
    per_subject = {}
    for tgt in PRIMARY_TARGETS:
        for f in FEATURES:
            x = feats[f]
            if np.all(np.isnan(x)):
                continue
            mask = ~np.isnan(x)
            for subj in dc["subjects"]:
                d = dc["delta"].get((subj, tgt))
                if d is None:
                    continue
                rho, _ = spearman_or_nan(d[mask], x[mask])
                per_subject.setdefault(f, {}).setdefault(tgt, {})[subj] = rho
    for f in per_subject:
        v = per_subject[f].get("vim", {})
        z = per_subject[f].get("zi", {})
        common = sorted(set(v) & set(z))
        rv = np.array([v[s] for s in common])
        rz = np.array([z[s] for s in common])
        good = ~(np.isnan(rv) | np.isnan(rz))
        rv, rz = rv[good], rz[good]
        if rv.size < 5:
            continue
        obs_diff = float(np.mean(rz - rv))
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, rv.size, size=rv.size)
            boots.append(float(np.mean((rz - rv)[pick])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_boot = 2.0 * min((np.asarray(boots) <= 0).mean(), (np.asarray(boots) >= 0).mean())
        p_wil = wilcoxon_p(rz, rv)
        out.append({"layer": "target_dissociation", "target": "zi_minus_vim", "feature": f,
                    "unit": "subject", "id": "AGGREGATE", "n_subjects": int(rv.size),
                    "mean_rho_vim": float(np.mean(rv)), "mean_rho_zi": float(np.mean(rz)),
                    "diff": obs_diff, "ci_lo": float(lo), "ci_hi": float(hi),
                    "p_wilcoxon": p_wil, "p_boot": float(min(1.0, p_boot)),
                    "inference_note": "same shared delta-K vector used for both targets; "
                                      "a difference here reflects only delta-C side"})
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="joint delta-C / delta-K test")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--dc-csv", type=str, default=None,
                    help="override delta-C table (e.g. loso_delta_summary_filtered.csv "
                         "with the 3 divergent folds excluded)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="append to output dir name, e.g. _filtered")
    args = ap.parse_args()

    global DC_CSV, OUT_DIR
    if args.dc_csv:
        cand = Path(args.dc_csv)
        DC_CSV = cand if cand.is_absolute() else (ROOT / cand)
        if not DC_CSV.exists():
            raise SystemExit(f"--dc-csv not found: {DC_CSV}")
    if args.out_suffix:
        OUT_DIR = ROOT / "results" / f"delta_c_k_joint{args.out_suffix}"

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for p in (DC_CSV, DK_DIFF_CSV):
        if not p.exists():
            raise SystemExit(f"required input missing: {p}")

    print("=" * 78)
    print("Joint test: delta-C (control gain) x delta-K (network propagation)")
    print("=" * 78)

    dc = load_delta_c()
    print(f"[load] delta-C rows={dc['n_rows']} subjects={len(dc['subjects'])} "
          f"targets={dc['targets']} roi={len(dc['roi_order'])}")
    for c in dc["checks"]:
        print(f"  [check] {c}")
    if not dc["checks"]:
        print("  [check] stored delta equals roi_c_post - roi_c_pre for every cell")

    feats, dk_notes = load_delta_k(dc["roi_order"])
    print(f"[load] delta-K {dk_notes['dk_shape']} mean={dk_notes['dk_row_mean']:+.4f} "
          f"mean|d|={dk_notes['dk_abs_mean']:.4f} max_inc={dk_notes['dk_max_inc']:+.4f} "
          f"max_dec={dk_notes['dk_max_dec']:+.4f}")
    print(f"[load] absolute coupling tables used for rel features: {dk_notes.get('abs_tables_used')}")

    # ---- region feature table ------------------------------------------------
    fields = ["roi", "mean_dC_vim", "mean_dC_zi", "rank_dC_vim", "rank_dC_zi"] + FEATURES
    vstacks = [dc["delta"][(s, "vim")] for s in dc["subjects"] if (s, "vim") in dc["delta"]]
    zstacks = [dc["delta"][(s, "zi")] for s in dc["subjects"] if (s, "zi") in dc["delta"]]
    mv = np.mean(np.vstack(vstacks), axis=0) if vstacks else np.full(len(dc["roi_order"]), np.nan)
    mz = np.mean(np.vstack(zstacks), axis=0) if zstacks else np.full(len(dc["roi_order"]), np.nan)

    def ranks(a):
        order = np.argsort(np.argsort(a))
        return order.astype(float)

    rows = []
    for i, roi in enumerate(dc["roi_order"]):
        row = {"roi": roi, "mean_dC_vim": mv[i], "mean_dC_zi": mz[i],
               "rank_dC_vim": ranks(mv)[i], "rank_dC_zi": ranks(mz)[i]}
        for f in FEATURES:
            row[f] = feats[f][i]
        rows.append(row)
    write_rows(OUT_DIR / "region_feature_table.csv", fields, rows)
    print(f"[out ] region_feature_table.csv ({len(rows)} ROIs x {len(fields)} cols)")

    # ---- tests ---------------------------------------------------------------
    subj_rows = subject_level(dc, feats, rng, args.n_boot)
    perm_rows = region_permutation(dc, feats, args.n_perm, rng)
    pool_rows = pooled_regression(dc, feats, rng, args.n_boot)
    diss_rows = target_dissociation(dc, feats, rng, args.n_boot)

    write_rows(OUT_DIR / "subject_level_correlation.csv",
               ["layer", "target", "feature", "unit", "id", "n_subjects", "mean_rho",
                "ci_lo", "ci_hi", "mean_fisher_z", "t_stat", "p_t", "p_wilcoxon",
                "sign_consistency_frac", "rho", "inference_note"], subj_rows)
    write_rows(OUT_DIR / "region_permutation.csv",
               ["layer", "target", "feature", "unit", "id", "n_regions", "rho",
                "p_perm", "p_parametric", "inference_note"], perm_rows)
    write_rows(OUT_DIR / "pooled_regression.csv",
               ["layer", "target", "feature", "unit", "id", "n_obs", "n_clusters",
                "slope", "ci_lo", "ci_hi", "p_cluster_boot", "spearman_rho",
                "inference_note"], pool_rows)
    write_rows(OUT_DIR / "target_dissociation.csv",
               ["layer", "target", "feature", "unit", "id", "n_subjects",
                "mean_rho_vim", "mean_rho_zi", "diff", "ci_lo", "ci_hi",
                "p_wilcoxon", "p_boot", "inference_note"], diss_rows)

    # ---- FDR over declared families -----------------------------------------
    agg_subj = [r for r in subj_rows if r["id"] == "AGGREGATE"]
    fams = {
        "subject_level_p_t": [(r["target"], r["feature"], r["p_t"]) for r in agg_subj],
        "subject_level_p_wilcoxon": [(r["target"], r["feature"], r["p_wilcoxon"]) for r in agg_subj],
        "region_permutation_p_perm": [(r["target"], r["feature"], r["p_perm"]) for r in perm_rows],
        "pooled_p_cluster_boot": [(r["target"], r["feature"], r["p_cluster_boot"]) for r in pool_rows],
        "dissociation_p_wilcoxon": [(r["feature"], "zi-vim", r["p_wilcoxon"]) for r in diss_rows],
    }
    fdr_rows, all_p = [], []
    for fam, items in fams.items():
        vals = [(a, b, p) for a, b, p in items if p is not None and not math.isnan(float(p))]
        if not vals:
            continue
        q = bh_fdr([float(p) for _, _, p in vals])
        for (a, b, p), qq in zip(vals, q):
            fdr_rows.append({"family": fam, "target": a, "feature": b,
                             "p_raw": float(p), "q_bh_within_family": qq,
                             "n_tests_in_family": len(vals)})
            all_p.append(float(p))
    q_all = bh_fdr(all_p) if all_p else []
    for r, qq in zip(fdr_rows, q_all):
        r["q_bh_pooled_all_families"] = qq
    write_rows(OUT_DIR / "fdr_summary.csv",
               ["family", "target", "feature", "p_raw", "q_bh_within_family",
                "q_bh_pooled_all_families", "n_tests_in_family"], fdr_rows)

    # ---- console summary ------------------------------------------------------
    def show(title, items, key, cols):
        print(f"\n--- {title} ---")
        if not items:
            print("  (no testable entries)")
            return
        for r in sorted(items, key=key):
            vals = "  ".join(f"{c}={_fmt(r.get(c))}" for c in cols)
            print(f"  {r['target']:>6} {r['feature']:<14} {vals}")

    show("subject level (mean rho over subjects, BH q within family)",
         agg_subj, lambda r: r["p_t"],
         ["mean_rho", "ci_lo", "ci_hi", "p_t", "p_wilcoxon", "sign_consistency_frac"])
    show("region permutation (n=24 regions, single shared operator)",
         perm_rows, lambda r: r["p_perm"], ["rho", "p_perm", "n_regions"])
    show("pooled regression (subject-cluster bootstrap)",
         pool_rows, lambda r: r["p_cluster_boot"],
         ["slope", "ci_lo", "ci_hi", "p_cluster_boot", "n_obs"])
    show("target dissociation (rho_zi - rho_vim per subject)",
         diss_rows, lambda r: r["p_wilcoxon"],
         ["mean_rho_vim", "mean_rho_zi", "diff", "ci_lo", "ci_hi", "p_wilcoxon"])

    print("\n--- BH-FDR survivors (q < 0.05) ---")
    surv = [r for r in fdr_rows if float(r["q_bh_pooled_all_families"]) < 0.05]
    if not surv:
        pmin = min(all_p) if all_p else float("nan")
        qmin = min(q_all) if q_all else float("nan")
        print(f"  none. most significant raw p={pmin:.4g}; pooled q={qmin:.4g} "
              f"over {len(all_p)} tests")
    else:
        for r in surv:
            print(f"  {r['family']} {r['target']}/{r['feature']} p={r['p_raw']:.4g} "
                  f"q={r['q_bh_pooled_all_families']:.4g}")

    manifest = {
        "script": "analysis/joint_dc_dk_test.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                    "pandas_available": _probe("pandas"),
                    "statsmodels_available": _probe("statsmodels")},
        "inputs": {"delta_c": str(DC_CSV.relative_to(ROOT)),
                   "delta_k_diff": str(DK_DIFF_CSV.relative_to(ROOT)),
                   "delta_k_pre": str(DK_PRE_CSV.relative_to(ROOT)) if DK_PRE_CSV.exists() else None,
                   "delta_k_post": str(DK_POST_CSV.relative_to(ROOT)) if DK_POST_CSV.exists() else None},
        "delta_k_metadata": dk_notes,
        "features": FEATURES,
        "n_perm": args.n_perm, "n_boot": args.n_boot, "seed": args.seed,
        "counts": {"delta_c_rows": dc["n_rows"], "subjects": len(dc["subjects"]),
                   "rois": len(dc["roi_order"])},
        "caveats": [
            "delta-K is one shared 24x24 operator built from two independently trained "
            "models (pre_model, post_model); it has no subject dimension, so a literal "
            "same-subject delta-C x delta-K test is not computable.",
            "Operator-level uncertainty of delta-K (run-to-run optimisation noise, two "
            "training runs) is NOT represented in any p-value here.",
            "The subject-level family treats 19 held-out folds of delta-C against a fixed "
            "predictor; a nonzero mean rho therefore reflects region ordering agreement, "
            "not subject-specific coupling.",
            "All results are hypothesis-generating and must be reported with effect sizes "
            "and CIs, consistent with the main report's N=19 power boundary.",
        ],
        "elapsed_sec": round(time.time() - t0, 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\n[out ] run_manifest.json  elapsed={manifest['elapsed_sec']}s")
    print(f"[done] results written to {OUT_DIR}")


def _probe(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
