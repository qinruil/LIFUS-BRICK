#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration robustness of the delta-C versus tremor association, and a
single convention for held-out model fit.

Every trained checkpoint already on disk is evaluated under one protocol,
identical to the leave-one-subject-out analysis this project publishes: the
control matrix diagonal is projected to regions by the decoder weights, the
per-subject feature is delta-C = C(post) - C(pre), and each session is scored
by a normalized mean-square error against its own z-scored input.

Analysis units. A leave-one-subject-out study is one unit: each subject is
scored by the fold that never trained on that subject, so the 19 folds supply
one delta-C vector per subject. A whole-data checkpoint is one unit by itself,
scoring all subjects. Families covered:

  loso_beta0.2 / loso_beta0.05   the two published fold studies;
  sweep_batch / beta / epsilon / lambda_noise / weight_decay / train_seed /
  split_seed / beta_grid          the full-model sweeps on disk;
  reference_full_data             the reference checkpoint;
  relabel_shuffle / constant / swap   label-perturbation checkpoints.

Two subject cohorts are reported for every unit: the 16 converging folds that
the paper treats as primary, and the full 19. Correlations with the normalized
tremor change use Spearman's rho with BH-FDR applied inside each declared
family of 24 regions, cell by cell.

A reproduction check compares the recomputed leave-one-subject-out deltas
against the published delta table, so the protocol here is demonstrably the
same one that produced it.

Outputs (results/validation/config_robustness/):
    unit_deltas.csv                per unit x subject x target x roi: delta-C
    unit_nmse.csv                  per unit: session NMSE and loss identity
    unit_cell_correlations.csv     rho/p/q per unit x target x task x roi
    key_roi_ranges.csv             rho min/median/max across units, per cohort
    family_survivor_summary.csv    survivor counts per family x cell x cohort
    reproduction_check.csv         recomputed vs published fold deltas
    fig_config_robustness.png      rho spread per unit and survivor ranges
    run_manifest.json

Usage:
    python analysis/validation_config_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARC = ROOT.parent / "fhqwd-osfstorage-archive"
TM = ROOT / "results" / "trained_models"
OUT = ROOT / "results" / "validation" / "config_robustness"

from config import LAMBDA_NOISE  # noqa: E402
from preprocessing.load_preprocessed_data import TARGET_ROIS  # noqa: E402
from analysis.analysis_helper_functions import (  # noqa: E402
    load_model, compute_roi_projection_weights, project_to_roi,
)
from training.dataset import BRICKDataset  # noqa: E402
from training.train import DATA_DIR  # noqa: E402

TARGETS = ("vim", "zi")
TASKS = ("pos", "rest")
DIVERGENT = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}
KEY_ROIS = ["lh_GPi", "lh_STN", "rh_STN", "lh_vim", "lh_zi"]
PUBLISHED_LOSO = (ROOT / "results" / "figures" / "loso"
                  / "loso_19_fold_beta_0.2_13to1to5_split"
                  / "loso_delta_summary.csv")


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    q = np.ones(len(p))
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ok] = ranked
    return q


def load_tremor():
    # subject_list.txt carries the "sub-" prefix; the score table below keys
    # on the bare identifier so both sides match.
    subj = [s.strip().replace("sub-", "") for s in
            (ARC / "rsfMRI" / "scripts" / "subject_list.txt")
            .read_text(encoding="utf-8").splitlines() if s.strip()]
    m = sio.loadmat(ARC / "rsfMRI" / "tremor" / "tremor_pos_rest.mat")
    return pd.DataFrame({
        "subject_id": subj,
        "vim_pos": np.asarray(m["vim_pos"]).ravel().astype(float),
        "vim_rest": np.asarray(m["vim_rest"]).ravel().astype(float),
        "zi_pos": np.asarray(m["zi_pos"]).ravel().astype(float),
        "zi_rest": np.asarray(m["zi_rest"]).ravel().astype(float)})


def load_all_items():
    """One dict per (subject, target, session), z-scored as in training."""
    ds = BRICKDataset(DATA_DIR)
    items = []
    for i in range(len(ds)):
        it = ds[i]
        items.append({
            "subject_id": it["subject_id"],
            "target": it["target"],
            "session": "pre" if it["condition_str"] == "mpre" else "post",
            "x": it["x"],
            "lifus_condition": it["lifus_condition"],
        })
    return items


def score_checkpoint(ckpt, items):
    """Projected C and session NMSE for a checkpoint over its items."""
    model = load_model(ckpt)
    roi_weights = compute_roi_projection_weights(
        model.W_bar_x.detach().cpu().numpy())
    delta_rows, nmse_rows = [], []
    with torch.no_grad():
        cache = {}
        for it in items:
            out = model(it["x"], it["lifus_condition"],
                        kl_g0_weight=1.0, kl_u_weight=1.0,
                        apply_free_bits=False)
            c_diag = torch.diagonal(out["C"])
            if torch.is_complex(c_diag):
                c_diag = c_diag.real
            roi_c = np.asarray(
                project_to_roi(c_diag.unsqueeze(0), roi_weights)[0],
                dtype=float)
            x = it["x"]
            denom = float(torch.mean(x ** 2))
            nmse = float(torch.mean((out["x_recon"] - x) ** 2)) / max(denom, 1e-12)
            key = (it["subject_id"], it["target"])
            cache.setdefault(key, {})[it["session"]] = (roi_c, nmse)
        for (subj, tgt), phases in cache.items():
            if "pre" not in phases or "post" not in phases:
                continue
            d = phases["post"][0] - phases["pre"][0]
            finite = np.isfinite(d).all()
            for j, roi in enumerate(TARGET_ROIS):
                delta_rows.append({"subject_id": subj, "target": tgt,
                                   "roi": roi, "delta": float(d[j]),
                                   "finite": bool(finite)})
            nmse_rows.append({"subject_id": subj, "target": tgt,
                              "nmse_pre": float(phases["pre"][1]),
                              "nmse_post": float(phases["post"][1])})
    del model
    return delta_rows, nmse_rows


def discover_units():
    """[(unit name, family, mode, {subject or None: checkpoint})]."""
    units = []
    for tag, fam in [("loso_19_fold_beta_0.2_13to1to5_split", "loso_beta0.2"),
                     ("loso_19_fold_beta_0.05_13to1to5_split", "loso_beta0.05")]:
        folds = {}
        for d in sorted((TM / tag).glob("fold_*")):
            ck = d / "best_model_cls.pt"
            if ck.exists():
                folds[d.name[len("fold_"):]] = ck
        if folds:
            units.append((fam, fam, "heldout_folds", folds))
    for fam_dir in sorted(TM.glob("sweep_*")):
        for run in sorted(fam_dir.glob("sweep_*")):
            ck = run / "best_model_cls.pt"
            if not ck.exists():
                continue
            knob = run.name.split("_", 1)[1]
            # the same knob value appears in more than one sweep directory
            # (batch size 4 in both sweep_1 and sweep_2) as an independently
            # trained checkpoint, so the unit name carries the sweep folder
            # to keep every checkpoint a distinct analysis unit.
            units.append((f"{fam_dir.name}/{run.name}",
                          f"sweep_{knob.split('_')[0].lower()}",
                          "whole_data", {None: ck}))
    ref = ROOT / "results" / "final_model" / "best_model_cls.pt"
    if ref.exists():
        units.append(("reference_full_data", "reference", "whole_data",
                      {None: ref}))
    for c in ("shuffle", "constant", "swap"):
        ck = ROOT / "results" / "relabel_check" / c / "best_model_cls.pt"
        if ck.exists():
            units.append((f"relabel_{c}", "relabel", "whole_data", {None: ck}))
    return units


def correlate_unit(delta_df, tremor, cohort_label):
    """rho/p/q per (target, task, roi) across subjects of one unit."""
    rows = []
    for tgt in TARGETS:
        for task in TASKS:
            recs = []
            for roi in TARGET_ROIS:
                sub = delta_df[(delta_df["target"] == tgt) &
                               (delta_df["roi"] == roi)]
                pairs = []
                for _, rr in sub.iterrows():
                    tw = tremor[tremor["subject_id"] ==
                                rr["subject_id"].replace("sub-", "")]
                    if len(tw) and np.isfinite(rr["delta"]):
                        pairs.append((float(rr["delta"]),
                                      float(tw[f"{tgt}_{task}"].iloc[0])))
                if len(pairs) < 12:
                    continue
                x = np.array([q[0] for q in pairs])
                y = np.array([q[1] for q in pairs])
                r, p = spearmanr(x, y)
                recs.append({"target": tgt, "task": task, "roi": roi,
                             "n": len(pairs), "rho": float(r), "p": float(p)})
            if recs:
                g = pd.DataFrame(recs)
                g["q"] = bh_fdr(g["p"].to_numpy())
                g["cohort"] = cohort_label
                rows.append(g)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tremor = load_tremor()
    all_items = load_all_items()
    units = discover_units()
    print(f"{len(units)} analysis units discovered")

    delta_all, nmse_all, corr_all = [], [], []
    for name, family, mode, ckpts in units:
        if mode == "heldout_folds":
            # one subject per fold: pool the folds into a single unit
            for subj, ck in ckpts.items():
                items = [it for it in all_items if it["subject_id"] == subj]
                d_rows, n_rows = score_checkpoint(ck, items)
                for r in d_rows:
                    delta_all.append({**r, "unit": name, "family": family,
                                      "mode": mode})
                for r in n_rows:
                    nmse_all.append({**r, "unit": name, "family": family,
                                     "mode": mode})
        else:
            ck = list(ckpts.values())[0]
            d_rows, n_rows = score_checkpoint(ck, all_items)
            for r in d_rows:
                delta_all.append({**r, "unit": name, "family": family,
                                  "mode": mode})
            for r in n_rows:
                nmse_all.append({**r, "unit": name, "family": family,
                                 "mode": mode})
        print(f"  {name}: scored")

    deltas = pd.DataFrame(delta_all)
    deltas.to_csv(OUT / "unit_deltas.csv", index=False)
    nmse = pd.DataFrame(nmse_all)
    nmse.to_csv(OUT / "unit_subject_nmse.csv", index=False)

    # ---- correlations per unit, both cohorts ----
    for (unit, family, mode), g in deltas.groupby(["unit", "family", "mode"]):
        for label, sub in (("16-subject", g[~g["subject_id"].isin(DIVERGENT)]),
                           ("19-subject", g)):
            rows = correlate_unit(sub, tremor, label)
            for r in rows:
                r["unit"] = unit
                r["family"] = family
                r["mode"] = mode
                corr_all.append(r)
    corr = pd.concat(corr_all, ignore_index=True)
    corr.to_csv(OUT / "unit_cell_correlations.csv", index=False)

    # ---- reproduction check against the published fold table ----
    pub = pd.read_csv(PUBLISHED_LOSO)
    for c in ("subject_id", "roi", "target"):
        pub[c] = pub[c].astype(str).str.strip()
    pub["target"] = pub["target"].str.lower()
    mine = deltas[deltas["unit"] == "loso_beta0.2"]
    mg = mine.merge(pub[["subject_id", "target", "roi", "delta"]],
                    on=["subject_id", "target", "roi"], suffixes=("_new", "_pub"))
    mg["abs_diff"] = (mg["delta_new"] - mg["delta_pub"]).abs()
    rep = mg.groupby(["subject_id", "target"]).agg(
        n_roi=("abs_diff", "size"), max_abs_diff=("abs_diff", "max"),
        mean_abs_diff=("abs_diff", "mean")).reset_index()
    rep.to_csv(OUT / "reproduction_check.csv", index=False)
    print("\nreproduction vs published fold deltas: "
          f"{len(mg)} rows, max abs diff {mg['abs_diff'].max():.2e}")

    # ---- rho ranges across units for the key regions ----
    kr = corr[corr["roi"].isin(KEY_ROIS)]
    rng = (kr.groupby(["cohort", "family", "target", "task", "roi"])["rho"]
           .agg(n_units="size", rho_min="min", rho_median="median",
                rho_max="max").reset_index())
    surv_map = (kr.assign(surv=(kr["q"] < 0.05).astype(int))
                .groupby(["cohort", "family", "target", "task", "roi"])["surv"]
                .sum().reset_index().rename(columns={"surv": "units_with_survivor"}))
    rng = rng.merge(surv_map, on=["cohort", "family", "target", "task", "roi"])
    rng.to_csv(OUT / "key_roi_ranges.csv", index=False)
    print("\n=== rho range across configurations, VIM rest, 16-subject cohort ===")
    show = rng[(rng["cohort"] == "16-subject") & (rng["target"] == "vim")
               & (rng["task"] == "rest")]
    print(show.to_string(index=False))

    # ---- survivor counts per family, cell, cohort ----
    fs_rows = []
    for (coh, fam, tgt, task), g in corr.groupby(
            ["cohort", "family", "target", "task"]):
        per_unit = (g.assign(s=(g["q"] < 0.05).astype(int))
                    .groupby("unit")["s"].sum())
        fs_rows.append({"cohort": coh, "family": fam, "target": tgt,
                        "task": task, "n_units": int(g["unit"].nunique()),
                        "survivors_min": int(per_unit.min()),
                        "survivors_median": float(per_unit.median()),
                        "survivors_max": int(per_unit.max()),
                        "units_with_any": int((per_unit > 0).sum()),
                        "median_key_roi_rho": float(
                            g[g["roi"].isin(KEY_ROIS)]["rho"].median())})
    fs = pd.DataFrame(fs_rows)
    fs.to_csv(OUT / "family_survivor_summary.csv", index=False)
    print("\n=== survivor counts per family (VIM rest) ===")
    print(fs[fs["task"] == "rest"][fs["target"] == "vim"].to_string(index=False))

    # ---- unified NMSE per unit ----
    nm_rows = []
    for (unit, family, mode), g in nmse.groupby(["unit", "family", "mode"]):
        worst = np.maximum(g["nmse_pre"], g["nmse_post"])
        nm_rows.append({"unit": unit, "family": family, "mode": mode,
                        "n_sessions": int(2 * len(g)),
                        "nmse_median": float(np.nanmedian(worst)),
                        "nmse_p10": float(np.nanquantile(worst, 0.10)),
                        "nmse_p90": float(np.nanquantile(worst, 0.90)),
                        "nmse_mean": float(np.nanmean(np.r_[
                            g["nmse_pre"].values, g["nmse_post"].values])),
                        })
    nm = pd.DataFrame(nm_rows)
    nm["loss_recon_equivalent"] = nm["nmse_mean"] / (2 * LAMBDA_NOISE)
    nm.to_csv(OUT / "unit_nmse.csv", index=False)
    print("\n=== NMSE by unit (median over sessions, worst phase kept) ===")
    print(nm.sort_values("nmse_median").to_string(index=False))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    sel = corr[(corr["cohort"] == "16-subject") & (corr["target"] == "vim")
               & (corr["task"] == "rest") & (corr["roi"] == "lh_GPi")]
    sel = sel.sort_values("rho")
    axes[0].scatter(np.arange(len(sel)), sel["rho"], s=26, alpha=0.85,
                    edgecolor="k", linewidth=0.3)
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].axhline(-0.5, color="C3", ls=":", lw=0.9)
    axes[0].set_xticks(np.arange(len(sel)))
    axes[0].set_xticklabels(sel["unit"], rotation=55, ha="right", fontsize=6.5)
    axes[0].set_ylabel("Spearman rho, lh_GPi delta-C vs VIM-rest tremor")
    axes[0].set_title("One point per configuration unit (16-subject cohort)")
    f16 = fs[(fs["cohort"] == "16-subject") & (fs["target"] == "vim")
             & (fs["task"] == "rest")].sort_values("family")
    xs = np.arange(len(f16))
    axes[1].vlines(xs, f16["survivors_min"], f16["survivors_max"],
                   color="C3", lw=2.5)
    axes[1].scatter(xs, f16["survivors_median"], color="C3", zorder=3, s=40)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(f16["family"], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("FDR survivors in the 24-region family")
    axes[1].set_title("Survivor count range within each family")
    fig.tight_layout()
    fig.savefig(OUT / "fig_config_robustness.png", dpi=150)
    plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "units": int(deltas["unit"].nunique()),
        "families": sorted(deltas["family"].unique().tolist()),
        "delta_convention": "decoder-projected diagonal control gain, "
                            "post minus pre, per session",
        "nmse_convention": "mean((x_hat - x)^2) / mean(x^2) per session on "
                           "the z-scored input; the reported session value is "
                           "the worse of pre and post; loss_recon_equivalent "
                           "= nmse / (2 * lambda_noise)",
        "cohorts": ["16-subject primary", "19-subject"],
        "correlation": "Spearman, BH-FDR inside each 24-region family, "
                       "per target x task cell",
        "caveat": "whole-data units score every subject in sample; only the "
                  "fold units are held out per subject",
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
