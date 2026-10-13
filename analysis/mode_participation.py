#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Koopman mode participation scores: spatial quantification, pre/post
topography matching, and cross-fold stability.

Motivation
----------
The report (p58-60) looked at mode spatial loadings and resting networks and
judged "no correspondence" by eye, and it never quantified WHICH modal spatial
patterns shift with treatment. This script turns that layer into numbers. It
adds no retraining: every quantity is read from saved checkpoints (W_bar_x,
nu_log, theta_log) or from the spectrum-analysis energy tables on disk.

Four outputs
------------
1) reference_mode_participation.csv
   For the shared reference operator (results/final_model): each mode's L1
   normalised |loading| profile over 24 ROIs -> top-1 ROI, top-1 share, top-3
   share, normalised entropy, and network mass over the four groups defined by
   the project's own compare_pre_post.py (BasalGanglia 10 / Cerebellum 4 /
   Cortex 8 / TUS targets 2; no right-side targets exist in the 24-ROI set).
   Plus the mean per-mode energy share per condition (from the
   spectrum analysis, condition_participation_by_mode.csv), and the resulting ROI-level
   participation score P[roi|condition] = sum_m energy_share[m|condition] *
   loadnorm[m|roi]: "which regions does each condition actually run through".

2) pre_post_matching.csv
   The two task-state operators have no shared mode ordering (report p54-56),
   so mode correspondence is ESTIMATED by optimal assignment on loading-space
   cosine similarity (Hungarian). For matched pairs we report the similarity,
   the pre/post persistence |Lambda| and frequency, and the shift. Summary
   lines compare matched similarity against the random-assignment baseline.
   This is the quantified version of "which modal spatial patterns changed".

3) fold_stability.csv
   The report claims persistence-rank mode topography is "highly consistent"
   across LOSO folds, broken by sub-fuspd09/15/19. We reproduce that
   definitionally: rank modes by |Lambda| descending within each fold, then
   for each rank compute centroid cosine similarity across folds, and for each
   fold its mean similarity to the grand centroid.

4) condition_roi_participation.csv / _sessions.csv
   Group-mean ROI participation per condition, plus per-session ROI
   participation vectors (76 sessions) recomputed by forward pass under the
   reference operator (no training), so later work can test region-level
   effects without re-running the model.

Interpretation notes written into the files
-------------------------------------------
* All topography is |loading|: sign/phase information of the complex readout
  is discarded by design (mode phase is absorbed by the eigenvalue).
* Pre/post matching similarity is NOT a statistical test; it characterises how
  much spatial correspondence exists between independently trained operators.
* N=19: nothing in this script is a confirmatory inference (per the frozen
  rule of this round, no new hypothesis-test families are opened here).

Usage
    python analysis/mode_participation.py
Dependencies: numpy, torch, scipy (no pandas/matplotlib/statsmodels).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import N_ROIS, H, M  # noqa: E402
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402
from analysis.koopman_spectrum_stats import (  # noqa: E402
    load_spectrum, load_brick_model, znorm, mode_energy_from_model, spectral_quantities,
)

REF_CKPT = ROOT / "results" / "final_model" / "best_model_cls.pt"
PRE_CKPT = ROOT / "results" / "trained_models" / "task_states" / "pre_model" / "best_model_cls.pt"
POST_CKPT = ROOT / "results" / "trained_models" / "task_states" / "post_model" / "best_model_cls.pt"
LOSO_DIR = ROOT / "results" / "trained_models" / "loso_19_fold_beta_0.2_13to1to5_split"
COND_BY_MODE = ROOT / "results" / "spectrum_stats" / "condition_participation_by_mode.csv"
OUT_DIR = ROOT / "results" / "mode_participation"

TR = 2.0  # confirmed by the user on 2026-10-06

# verbatim network grouping from analysis/compare_pre_post.py (lines 164-183),
# restricted to regions actually present in TARGET_ROIS
NETWORKS = {
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
for net, rois in NETWORKS.items():
    for r in rois:
        NET_OF_ROI[r] = net
GVEC = np.array([NET_OF_ROI[r] for r in TARGET_ROIS])
NET_NAMES = list(NETWORKS.keys())
GIDX = {n: i for i, n in enumerate(NET_NAMES)}


# ------------------------------------------------------------- utilities ----
def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            (f"{float(r[k]):.6g}" if isinstance(r[k], (float, np.floating))
                             and np.isfinite(r[k]) else str(r.get(k, ""))))
                        for k in fields})


def loadnorm(W_bar_x):
    """Columns of |W_bar_x|, L1-normalised per mode -> (M, N_ROIS)."""
    L = np.abs(W_bar_x).astype(np.float64)
    s = L.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return L / s


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def entropy_norm(p):
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(len(p)))


# --------------------------------------------------------------- outputs ----
def reference_table(Ln, Lambda, energy_by_cond):
    q = spectral_quantities(Lambda, TR)
    rows = []
    for m in range(Ln.shape[1]):
        x = Ln[:, m]
        order = np.argsort(x)[::-1]
        mass = {n: float(x[GVEC == n].sum()) for n in NET_NAMES}
        row = {"mode_index": m,
               "top1_roi": TARGET_ROIS[order[0]], "top1_share": float(x[order[0]]),
               "top3_share": float(x[order[:3]].sum()),
               "entropy_norm": entropy_norm(x),
               "mass_BasalGanglia": mass["BasalGanglia"], "mass_Cerebellum": mass["Cerebellum"],
               "mass_Cortex": mass["Cortex"], "mass_TUS": mass["TUS"],
               "eigenvalue_mag": float(q["mag"][m]),
               "freq_hz": float(q["freq_hz"][m])}
        for cond, share in energy_by_cond.items():
            row[f"energy_share_{cond}"] = float(share[m])
        rows.append(row)
    return rows


def condition_participation(Ln, energy_by_cond):
    """P[roi|cond] = sum_m E[m|cond] * L1norm loading; rows: ROI x condition."""
    out = []
    for cond, share in energy_by_cond.items():
        P = Ln @ share          # (N_ROIS,) = sum_m loadnorm[roi,m]*E[m]
        P = P / P.sum() if P.sum() > 0 else P
        for i, roi in enumerate(TARGET_ROIS):
            out.append({"condition": cond, "roi": roi, "network": NET_OF_ROI[roi],
                        "participation": float(P[i])})
    return out


def network_mass_by_condition(Ln, energy_by_cond):
    rows = []
    for cond, share in energy_by_cond.items():
        P = Ln @ share
        P = P / P.sum() if P.sum() > 0 else P
        for n in NET_NAMES:
            rows.append({"condition": cond, "network": n,
                         "participation_mass": float(P[GVEC == n].sum())})
    return rows


def per_session_participation(model, Ln, sessions):
    rows = []
    with torch.no_grad():
        for s in sessions:
            x_z = znorm(torch.tensor(s["mpre"], dtype=torch.float32))
            e_pre = mode_energy_from_model(model, x_z)["energy_share"]
            x_q = znorm(torch.tensor(s["mpost"], dtype=torch.float32))
            e_post = mode_energy_from_model(model, x_q)["energy_share"]
            for cond, E in (("pre", e_pre), ("post", e_post)):
                P = Ln @ E
                P = P / P.sum() if P.sum() > 0 else P
                net = {n: float(P[GVEC == n].sum()) for n in NET_NAMES}
                rows.append({"subject_id": s["subject_id"], "target": s["target"],
                             "order_group": s["group"], "condition": cond,
                             **{f"net_{n}": v for n, v in net.items()},
                             **{f"roi_{r}": float(P[i]) for i, r in enumerate(TARGET_ROIS)}})
    return rows


def pre_post_matching(spec_a, spec_b, name_a, name_b):
    a = loadnorm(spec_a["W_bar_x"]); b = loadnorm(spec_b["W_bar_x"])
    # mean-centred cosine: plain cosine over non-negative profiles saturates
    # (~0.8 even for random pairs) because all profiles share a positive
    # common component; removing the per-vector mean restores discrimination.
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    S = (ac / np.linalg.norm(ac, axis=0, keepdims=True)).T @ \
        (bc / np.linalg.norm(bc, axis=0, keepdims=True))
    qa, qb = spectral_quantities(spec_a["Lambda"], TR), spectral_quantities(spec_b["Lambda"], TR)
    ri, ci = linear_sum_assignment(-S)
    matched = S[ri, ci]
    rnd = float(S.mean())
    rows = []
    for r, c, s in zip(ri, ci, matched):
        rows.append({"mode_a": int(r), "mode_b": int(c), "cos_sim": float(s),
                     "mag_a": float(qa["mag"][r]), "mag_b": float(qb["mag"][c]),
                     "dlog_mag": (float(math.log(qb["mag"][c]) - math.log(qa["mag"][r]))
                                  if qa["mag"][r] > 0 and qb["mag"][c] > 0 else float("nan")),
                     "freq_a_hz": float(qa["freq_hz"][r]), "freq_b_hz": float(qb["freq_hz"][c]),
                     "abs_d_freq_hz": float(abs(qb["freq_hz"][c] - qa["freq_hz"][r]))})
    summary = {"pair": f"{name_a}->{name_b}", "n_modes": int(S.shape[0]),
               "mean_matched_cos": float(matched.mean()),
               "mean_random_cos": rnd,
               "lift_over_random": float(matched.mean() - rnd),
               "frac_matched_ge_0p5": float((matched >= 0.5).mean()),
               "frac_matched_ge_0p2": float((matched >= 0.2).mean()),
               "frac_random_ge_0p5": float((S >= 0.5).mean()),
               "mean_abs_d_freq_matched_hz": float(np.nanmean([r["abs_d_freq_hz"] for r in rows])),
               "median_dlog_mag_matched": float(np.nanmedian([r["dlog_mag"] for r in rows]))}
    return rows, summary


def fold_stability():
    folds = sorted(LOSO_DIR.glob("fold_sub-*"))
    data = {}
    for f in folds:
        ck = f / "best_model_cls.pt"
        if not ck.exists():
            continue
        spec = load_spectrum(ck)
        Ln = loadnorm(spec["W_bar_x"])          # (24 ROIs, M)
        P = Ln.T                                  # (M, 24) profiles
        P = P - P.mean(axis=1, keepdims=True)     # ROI-mean-centred per mode
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        rank = np.argsort(np.abs(spec["Lambda"]))[::-1]   # persistence descending
        data[f.name.replace("fold_", "")] = P[rank, :]
    names = sorted(data)
    mats = np.stack([data[n] for n in names])            # (F, M, 24) unit profiles
    centroid = mats.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid, axis=-1, keepdims=True)
    rows = []
    for m in range(mats.shape[1]):
        sims = [float(mats[i, m] @ centroid[m]) for i in range(len(names))]
        rows.append({"persistence_rank": m, "mean_sim_to_centroid": float(np.mean(sims)),
                     "min_sim_to_centroid": float(np.min(sims)),
                     "min_fold": names[int(np.argmin(sims))],
                     "pairwise_mean_cos": float(np.mean([
                         mats[i, m] @ mats[j, m] for i in range(len(names)) for j in range(i + 1, len(names))]))})
    fold_rows = []
    for i, n in enumerate(names):
        sims = [float((mats[i] * mats[j]).sum(axis=1).mean()) for j in range(len(names)) if j != i]
        fold_rows.append({"fold": n, "mean_mode_sim_to_others": float(np.mean(sims))})
    return rows, fold_rows


# -------------------------------------------------------------------- main --
def main():
    t0 = time.time()
    print("=" * 78)
    print("mode participation: spatial quantification, matching, fold stability")
    print("=" * 78)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_spectrum(REF_CKPT)
    Ln = loadnorm(ref["W_bar_x"])

    # spectrum-analysis energy table
    energy_by_cond = {}
    with open(COND_BY_MODE, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            key = f"{r['target']}_{r['condition']}"
            energy_by_cond.setdefault(key, {})[int(r["mode_index"])] = float(r["mean_energy_share"])
    energy_by_cond = {k: np.array([v[i] for i in range(M)]) for k, v in energy_by_cond.items()}
    print(f"[load ] energy table: conditions={sorted(energy_by_cond)}")

    ref_rows = reference_table(Ln, ref["Lambda"], energy_by_cond)
    write_rows(OUT_DIR / "reference_mode_participation.csv",
               ["mode_index", "eigenvalue_mag", "freq_hz", "top1_roi", "top1_share",
                "top3_share", "entropy_norm", "mass_BasalGanglia", "mass_Cerebellum",
                "mass_Cortex", "mass_TUS"] + [f"energy_share_{c}" for c in sorted(energy_by_cond)],
               ref_rows)
    print(f"[out ] reference_mode_participation.csv ({len(ref_rows)} modes)")

    cond_rows = condition_participation(Ln, energy_by_cond)
    write_rows(OUT_DIR / "condition_roi_participation.csv",
               ["condition", "roi", "network", "participation"], cond_rows)
    net_rows = network_mass_by_condition(Ln, energy_by_cond)
    write_rows(OUT_DIR / "condition_network_mass.csv",
               ["condition", "network", "participation_mass"], net_rows)
    print("[out ] condition_roi_participation.csv (4x24), condition_network_mass.csv (4x4)")
    print("--- network participation mass by condition ---")
    for n in NET_NAMES:
        line = "  ".join(f"{c}={next(r['participation_mass'] for r in net_rows if r['condition']==c and r['network']==n):.4f}"
                         for c in sorted(energy_by_cond))
        print(f"  {n:<13} {line}")

    # per-session vectors
    model, nc, _ = load_brick_model(REF_CKPT, verbose=False)
    subjects = load_all()
    sess_rows = per_session_participation(model, Ln, subjects)
    fields = (["subject_id", "target", "order_group", "condition"] +
              [f"net_{n}" for n in NET_NAMES] +
              [f"roi_{r}" for r in TARGET_ROIS])
    write_rows(OUT_DIR / "session_roi_participation.csv", fields, sess_rows)
    print(f"[out ] session_roi_participation.csv ({len(sess_rows)} session x target rows)")

    # pre/post matching
    pre = load_spectrum(PRE_CKPT); post = load_spectrum(POST_CKPT)
    mrows, msum = pre_post_matching(pre, post, "pre_model", "post_model")
    write_rows(OUT_DIR / "pre_post_matching.csv",
               ["mode_a", "mode_b", "cos_sim", "mag_a", "mag_b", "dlog_mag",
                "freq_a_hz", "freq_b_hz", "abs_d_freq_hz"],
               sorted(mrows, key=lambda r: -r["cos_sim"]))
    write_rows(OUT_DIR / "pre_post_matching_summary.csv", list(msum.keys()), [msum])
    top_movers = sorted([r for r in mrows if r["cos_sim"] >= 0.4],
                        key=lambda r: -r["abs_d_freq_hz"])[:8]
    print(f"[out ] pre_post_matching.csv  mean matched cos={msum['mean_matched_cos']:.3f} "
          f"(random {msum['mean_random_cos']:.3f}); >=0.5: "
          f"{msum['frac_matched_ge_0p5']:.1%} matched vs {msum['frac_random_ge_0p5']:.1%} of all cells")
    print("       largest frequency shifts among spatially matched pairs (sim>=0.4):")
    for r in top_movers:
        print(f"         a{r['mode_a']}<->b{r['mode_b']} cos={r['cos_sim']:.2f} "
              f"pre {r['freq_a_hz']:.3f}Hz -> post {r['freq_b_hz']:.3f}Hz "
              f"(|L| {r['mag_a']:.2f}->{r['mag_b']:.2f})")

    # fold stability
    rank_rows, fold_rows = fold_stability()
    write_rows(OUT_DIR / "fold_stability_by_rank.csv",
               ["persistence_rank", "mean_sim_to_centroid", "min_sim_to_centroid",
                "min_fold", "pairwise_mean_cos"], rank_rows)
    write_rows(OUT_DIR / "fold_stability_by_fold.csv",
               ["fold", "mean_mode_sim_to_others"], fold_rows)
    lo = sorted(rank_rows, key=lambda r: r["mean_sim_to_centroid"])[:6]
    fl = sorted(fold_rows, key=lambda r: r["mean_mode_sim_to_others"])[:5]
    print("[out ] fold_stability_by_rank.csv / _by_fold.csv")
    print("       least stable persistence ranks:",
          ", ".join(f"r{r['persistence_rank']}={r['mean_sim_to_centroid']:.3f}" for r in lo))
    print("       least stable folds:",
          ", ".join(f"{r['fold']}={r['mean_mode_sim_to_others']:.3f}" for r in fl))

    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"script": "analysis/mode_participation.py",
                   "reference_checkpoint": str(REF_CKPT.relative_to(ROOT)),
                   "tr_s": TR,
                   "networks": {k: len(v) for k, v in NETWORKS.items()},
                   "note_sign": "topography uses |W_bar_x| columns; complex phase of the readout is not used",
                   "note_matching": "Hungarian assignment on loading-space cosine is a descriptive correspondence estimate, not a test",
                   "elapsed_sec": round(time.time() - t0, 2)},
                  fh, indent=2, ensure_ascii=False)
    print(f"[done] {OUT_DIR} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
