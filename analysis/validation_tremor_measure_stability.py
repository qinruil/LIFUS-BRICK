#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stability of the tremor outcome measure near zero baseline power.

The published outcome is the normalized difference
(Post - Pre) / (Post + Pre) of 4-8 Hz accelerometer power, which is bounded
in [-1, 1] for non-negative powers and degenerates to +/-1 whenever either
session contributes nothing. This script rebuilds the outcome from the
cleaned accelerometer recordings and reports it under three definitions so
the clinical association can be read against each:

  ratio        (Post - Pre) / (Post + Pre), the published convention
  log_ratio    log(Post) - log(Pre), unbounded but scale-stable
  ratio_clean  the ratio restricted to sessions whose two power values are
               both well away from zero

Reconstruction follows the recorded calculation order: for each of the six
task blocks and each three-axis sensor set, the first principal component,
Welch power spectral density with a 4-second window and 50% overlap evaluated
on the 0 to 100 Hz grid in 0.1 Hz steps, the mean over the 4-8 Hz band; the
per-sensor normalized difference is formed first, then averaged over the
non-postural blocks for the rest composite, then over sensors, dropping the
sensor flagged as faulty for the two subjects documented in the original
calculation file.

Aggregating in that order matters: taking power differences before the
per-sensor ratio would not reproduce the stored values.

A replication table compares the rebuilt ratio against the stored outcome,
and an association table recomputes the region-level Spearman correlations
of delta-C against each outcome definition.

Inputs (read only):
    fhqwd-osfstorage-archive/Behaviour/accelerometer_data/
        sub-fuspd*_{VIM,ZI}_Data_cleaned.mat
    fhqwd-osfstorage-archive/rsfMRI/tremor/tremor_pos_rest.mat
    results/figures/loso/.../loso_delta_summary_filtered.csv

Outputs (results/validation/tremor_measure_stability/):
    tremor_measures.csv                 rebuilt outcome per session and band
    baseline_noise_audit.csv            zero-floor cells and ratio extremes
    stored_vs_recomputed.csv            replication against stored outcome
    correlation_replication.csv         rank agreement per target x band
    outcome_association_by_definition.csv
    fig_tremor_measure_stability.png
    run_manifest.json

Usage:
    python analysis/validation_tremor_measure_stability.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
ACC = ARC / "Behaviour" / "accelerometer_data"
LOSO = (ROOT / "results" / "figures" / "loso"
        / "loso_19_fold_beta_0.2_13to1to5_split"
        / "loso_delta_summary_filtered.csv")
OUT = ROOT / "results" / "validation" / "tremor_measure_stability"

FS = 200.0
NPERSEG = int(4 * FS)
NOVERLAP = int(NPERSEG * 0.5)
GRID = np.arange(0.0, FS / 2 + 1e-9, 0.1)
BAND = (GRID >= 4.0) & (GRID <= 8.0)
N_TASKS = 6
POSTURAL_INDEX = 0
FAULTY_SENSOR = {"fuspd03": 0, "fuspd09": 0}
ZERO_TOL = 1e-12
NEAR_LIMIT = 0.90           # |ratio| at or above this means one session ~ zero


def pc1_per_sensor(x):
    """First principal component of each three-axis sensor block."""
    out = np.zeros((x.shape[0], 3))
    for i in range(3):
        block = x[:, i * 3:i * 3 + 3]
        block = block - block.mean(0)
        u, s, _ = np.linalg.svd(block, full_matrices=False)
        out[:, i] = u[:, 0] * s[0]
    return out


def band_power(pcs):
    """Mean 4-8 Hz Welch power per sensor column."""
    out = np.zeros(pcs.shape[1])
    for j in range(pcs.shape[1]):
        nper = NPERSEG if pcs.shape[0] >= NPERSEG else pcs.shape[0]
        f, pxx = signal.welch(pcs[:, j], fs=FS, nperseg=nper,
                              noverlap=int(nper * 0.5), detrend=False,
                              scaling="density")
        out[j] = np.interp(GRID, f, pxx)[BAND].mean()
    return out


def unpack_tasks(struct):
    """The (1, 6) object row of per-task arrays inside Pre_clean/Post_clean."""
    raw = np.asarray(struct["data"][0, 0], dtype=object)
    out = []
    for s in range(N_TASKS):
        a = raw[0, s]
        while isinstance(a, np.ndarray) and a.dtype == object:
            a = a.flat[0]
        out.append(np.asarray(a, dtype=float))
    return out


def rebuild_session(fp):
    parts = fp.stem.split("_")
    subj, tgt = parts[0].replace("sub-", ""), parts[1].lower()
    d = sio.loadmat(str(fp))
    pre_t = unpack_tasks(d["Pre_clean"])
    post_t = unpack_tasks(d["Post_clean"])
    pre = np.array([band_power(pc1_per_sensor(a)) for a in pre_t])
    post = np.array([band_power(pc1_per_sensor(a)) for a in post_t])
    keep = [c for c in range(3) if not (subj in FAULTY_SENSOR
                                       and c == FAULTY_SENSOR[subj])]
    p0, p1 = pre[:, keep], post[:, keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_cell = (p1 - p0) / (p0 + p1)
    ratio_cell[~np.isfinite(ratio_cell)] = np.nan
    # silent in both sessions carries no information for either definition;
    # silent in one session saturates the ratio at +/-1 while the log form
    # keeps it as an extreme value, which is the point of the comparison.
    log_cell = np.log(np.maximum(p1, ZERO_TOL)) - np.log(np.maximum(p0, ZERO_TOL))
    log_cell[(p0 <= ZERO_TOL) & (p1 <= ZERO_TOL)] = np.nan
    rows = []
    for band, idx in (("pos", [POSTURAL_INDEX]),
                      ("rest", [i for i in range(N_TASKS)
                                if i != POSTURAL_INDEX])):
        # aggregation order of the recorded calculation: mean over the task
        # blocks within a sensor, then mean over sensors. A block whose two
        # sessions are both silent has no defined difference and contributes
        # no value, which is the one case where the mean of an empty slice is
        # the intended answer.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = float(np.nanmean(np.nanmean(ratio_cell[idx], axis=0)))
            lg = float(np.nanmean(np.nanmean(log_cell[idx], axis=0)))
        rows.append({
            "subject_id": subj, "target": tgt, "band": band,
            "power_pre": float(np.nanmean(p0[idx])),
            "power_post": float(np.nanmean(p1[idx])),
            "ratio": float(r),
            "log_ratio": float(lg),
            "n_cells": int(np.isfinite(ratio_cell[idx]).sum()),
            "n_zero_cells": int((p0[idx] <= ZERO_TOL).sum()
                                + (p1[idx] <= ZERO_TOL).sum()),
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(ACC.glob("sub-fuspd*_Data_cleaned.mat"))
    if not files:
        raise SystemExit(f"no cleaned accelerometer files under {ACC}")

    rows = []
    for fp in files:
        rows.extend(rebuild_session(fp))
    built = pd.DataFrame(rows)
    built.to_csv(OUT / "tremor_measures.csv", index=False)

    # ---- baseline / zero-floor audit ----
    audit = (built.groupby(["subject_id", "target"])
             .agg(n_zero_cells=("n_zero_cells", "sum"),
                  n_cells=("n_cells", "sum"),
                  power_pre_min=("power_pre", "min"),
                  ratio_range=("ratio", lambda s: float(s.max() - s.min())))
             .reset_index())
    audit["near_limit"] = (audit["ratio_range"] >= NEAR_LIMIT) | \
        (audit["n_zero_cells"] > 0)
    audit.to_csv(OUT / "baseline_noise_audit.csv", index=False)

    # ---- replication against the stored outcome ----
    ref = sio.loadmat(ARC / "rsfMRI" / "tremor" / "tremor_pos_rest.mat")
    subj_list = [x.strip().replace("sub-", "") for x in
                 (ARC / "rsfMRI" / "scripts" / "subject_list.txt")
                 .read_text(encoding="utf-8", errors="replace")
                 .splitlines() if x.strip()]
    ref_long = []
    for i, s in enumerate(subj_list):
        for tgt in ("vim", "zi"):
            for band in ("pos", "rest"):
                ref_long.append({"subject_id": s, "target": tgt, "band": band,
                                 "stored_ratio": float(
                                     np.asarray(ref[f"{tgt}_{band}"]).ravel()[i])})
    ref_df = pd.DataFrame(ref_long)
    cmp_df = ref_df.merge(built, on=["subject_id", "target", "band"],
                          how="inner")
    cmp_df["abs_diff"] = (cmp_df["stored_ratio"]
                          - cmp_df["ratio"]).abs()
    cmp_df["signed_diff"] = cmp_df["stored_ratio"] - cmp_df["ratio"]
    cmp_df.to_csv(OUT / "stored_vs_recomputed.csv", index=False)

    rep_rows = []
    for (tgt, band), g in cmp_df.groupby(["target", "band"]):
        r_log, p_log = spearmanr(g["stored_ratio"], g["log_ratio"])
        r_rec, p_rec = spearmanr(g["stored_ratio"], g["ratio"])
        rep_rows.append({"target": tgt, "band": band, "n": int(len(g)),
                         "pearson_stored_vs_recomputed": float(
                             np.corrcoef(g["stored_ratio"], g["ratio"])[0, 1]),
                         "median_abs_diff": float(g["abs_diff"].median()),
                         "max_abs_diff": float(g["abs_diff"].max()),
                         "spearman_stored_vs_log_ratio": float(r_log),
                         "spearman_stored_vs_recomputed": float(r_rec)})
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(OUT / "correlation_replication.csv", index=False)
    print("=== replication of the stored outcome ===")
    print(rep.to_string(index=False))

    # ---- association of delta-C with each outcome definition ----
    dsum = pd.read_csv(LOSO)
    for c in ("subject_id", "roi", "target"):
        dsum[c] = dsum[c].astype(str).str.strip()
    dsum["target"] = dsum["target"].str.lower()
    dsum["subject_id"] = dsum["subject_id"].str.replace("sub-", "", regex=False)
    clean = audit[~audit["near_limit"]][["subject_id", "target"]]

    assoc = []
    for tgt in ("vim", "zi"):
        for band in ("pos", "rest"):
            m = built[(built["target"] == tgt) & (built["band"] == band)][
                ["subject_id", "ratio", "log_ratio"]]
            mc = m.merge(clean[clean["target"] == tgt], on="subject_id",
                         how="inner").rename(columns={"ratio":
                                                      "ratio_clean_baseline"})
            for roi in ("lh_GPi", "lh_STN", "rh_STN", "lh_vim", "lh_zi"):
                dd = dsum[(dsum["target"] == tgt) & (dsum["roi"] == roi)][
                    ["subject_id", "delta"]]
                for label, src in (("ratio", m), ("log_ratio", m),
                                   ("ratio_clean_baseline", mc)):
                    j = dd.merge(src[["subject_id", label]].dropna(),
                                 on="subject_id")
                    if len(j) < 8:
                        assoc.append({"target": tgt, "band": band, "roi": roi,
                                      "definition": label, "n": int(len(j)),
                                      "rho": np.nan, "p": np.nan,
                                      "note": "below the eight-session floor"})
                        continue
                    r, p = spearmanr(j["delta"], j[label])
                    assoc.append({"target": tgt, "band": band, "roi": roi,
                                  "definition": label, "n": int(len(j)),
                                  "rho": float(r), "p": float(p), "note": ""})
    adf = pd.DataFrame(assoc)
    adf.to_csv(OUT / "outcome_association_by_definition.csv", index=False)
    print("\n=== delta-C association by outcome definition, VIM rest ===")
    print(adf[(adf["target"] == "vim") & (adf["band"] == "rest")]
          .to_string(index=False))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.hist(built["power_pre"], bins=16, color="C0", alpha=0.85)
    ax.axvline(ZERO_TOL, color="C3", ls="--", lw=1.0, label="zero floor")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("baseline 4-8 Hz power (log)")
    ax.set_ylabel("task x sensor sessions")
    ax.set_title("Baseline power distribution")
    ax.legend(fontsize=7)

    ax = axes[1]
    key = adf[(adf["band"] == "rest") & (adf["target"] == "vim")
              & (adf["roi"].isin(["lh_GPi", "lh_STN", "rh_STN"]))]
    piv = key.pivot(index="roi", columns="definition", values="rho")
    piv = piv.reindex(["ratio", "log_ratio", "ratio_clean_baseline"], axis=1)
    xs = np.arange(len(piv))
    for k, col in enumerate(piv.columns):
        vals = piv[col].to_numpy(float)
        ax.bar(xs + k * 0.27, vals, 0.25,
               color=("0.55", "C2", "C0")[k], label=col)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs + 0.27)
    ax.set_xticklabels(piv.index, fontsize=8)
    ax.set_ylabel("Spearman rho with VIM-rest outcome")
    ax.set_title("Same association, three outcome definitions")
    ax.legend(fontsize=7)

    ax = axes[2]
    j = (dsum[(dsum["target"] == "vim") & (dsum["roi"] == "lh_STN")]
         [["subject_id", "delta"]]
         .merge(built[(built["target"] == "vim") & (built["band"] == "rest")]
                [["subject_id", "ratio", "n_zero_cells"]], on="subject_id"))
    flag = j["n_zero_cells"] > 0
    ax.scatter(j.loc[~flag, "ratio"], j.loc[~flag, "delta"], s=30,
               label="baseline away from zero")
    if flag.any():
        ax.scatter(j.loc[flag, "ratio"], j.loc[flag, "delta"], s=52,
                   marker="D", color="C3", label="zero-floor cells present")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("ratio outcome")
    ax.set_ylabel("lh_STN delta-C")
    ax.set_title("VIM rest: where the zero floor bites")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tremor_measure_stability.png", dpi=150)
    plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "pipeline": "per-sensor PCA first component; Welch PSD, 4 s window, "
                    "50% overlap, 0-100 Hz grid in 0.1 Hz steps; 4-8 Hz band "
                    "mean; per-sensor ratio formed first, averaged over "
                    "non-postural blocks, then over sensors with the "
                    "documented faulty-sensor column dropped",
        "definitions": ["ratio", "log_ratio",
                        "ratio_clean_baseline (no zero-floor cells)"],
        "near_limit_criterion": f"session excluded when any task x sensor "
                                f"cell has power <= {ZERO_TOL:g}, or when the "
                                f"ratio range across bands reaches "
                                f"{NEAR_LIMIT:g}",
        "note": "ratio and log-ratio are monotone transforms of the same "
                "power pair, so rank results are expected to agree; the "
                "definitions differ in extreme-value handling and in which "
                "sessions survive",
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
