#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variance share of the shared operator under one convention, per configuration.

The share is the variance-weighted percentage contributed by the
K-propagated component of the reconstruction, using the rollout
decomposition established in variance_decomposition_extended: at every step
the state update splits into k_term = Lambda * g_prev (carried by the shared
operator alone) and c_term = P_inv (C u_t) (the freshly injected control);
both are mapped to ROI space by the decoder and the covariance between the
two images is split evenly (theta = 0.5). Percentages are pooled over all 76
sessions with variance weighting, exactly as in the published decomposition.

This script computes that share for every whole-data checkpoint on disk:
the reference model, the three label-perturbation checkpoints, and the
complete parameter-sweep runs (batch size, beta grid, epsilon, lambda_noise,
weight decay, train and split seeds). Fold-level shares of the published
leave-one-subject-out study and the seed table already computed under the
same convention are carried into the same register, so a single file ties
every reported share to the model object it describes.

Usage:
    python analysis/validation_k_share_by_configuration.py
    python analysis/validation_k_share_by_configuration.py --quick
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(5)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TM = ROOT / "results" / "trained_models"
OUT = ROOT / "results" / "validation" / "k_share_by_configuration"

from analysis.variance_decomposition_extended import (  # noqa: E402
    load_brick, encode_and_control, manual_rollout, znorm,
    variance_components,
)
from models.koopman_utils import compute_lambda  # noqa: E402
from preprocessing.load_preprocessed_data import load_all  # noqa: E402

EPS = 1e-12


def pct_K(model, sessions):
    """Variance-weighted pooled %K over sessions, theta = 0.5 convention."""
    tot_k = tot_v = 0.0
    with torch.no_grad():
        Lambda = compute_lambda(model.nu_log, model.theta_log)
        P_inv = model.P_inv
        W_bar_x = model.W_bar_x
        for s in sessions:
            for key in ("mpre", "mpost"):
                x = znorm(torch.tensor(np.asarray(s[key], dtype=np.float32)))
                g_0, C_base, u = encode_and_control(model, x)
                _, x_K, x_C = manual_rollout(
                    Lambda, P_inv, g_0, u, W_bar_x, C_seq=C_base,
                    return_decomposition=True)
                vk, vc, cov, vt = variance_components(
                    x_K.detach().numpy(), x_C.detach().numpy())
                tot_k += float(np.sum(vk) + np.sum(cov))
                tot_v += float(np.sum(vt))
    return 100.0 * tot_k / (tot_v + EPS)


def whole_data_units():
    units = []
    ref = ROOT / "results" / "final_model" / "best_model_cls.pt"
    if ref.exists():
        units.append(("reference_full_data", "reference", ref))
    for c in ("shuffle", "constant", "swap"):
        ck = ROOT / "results" / "relabel_check" / c / "best_model_cls.pt"
        if ck.exists():
            units.append((f"relabel_{c}", "relabel", ck))
    for fam_dir in sorted(TM.glob("sweep_*")):
        for run in sorted(fam_dir.glob("sweep_*")):
            ck = run / "best_model_cls.pt"
            if ck.exists():
                knob = run.name.split("_", 1)[1]
                # the same knob value can appear in two sweep directories as
                # an independently trained checkpoint, so the folder name is
                # kept in the unit label to avoid merging distinct models
                units.append((f"{fam_dir.name}/{run.name}",
                              f"sweep_{knob.split('_')[0].lower()}", ck))
    return units


def main():
    quick = "--quick" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if quick:
        sessions = sessions[:3]

    rows = []
    for name, family, ck in whole_data_units():
        t0 = time.time()
        try:
            model, _ = load_brick(Path(ck), verbose=False)
        except Exception as e:
            print(f"  {name}: cannot load ({e})")
            continue
        pk = pct_K(model, sessions)
        rows.append({"unit": name, "family": family, "kind": "whole_data",
                     "pct_K": pk, "n_sessions": 2 * len(sessions),
                     "source": "computed here"})
        print(f"  {name}: %K = {pk:.3f}  ({time.time()-t0:.0f}s)")
        del model

    # carry the existing table, computed under the identical convention
    prev = (ROOT / "results" / "variance_decomposition" / "scaling_points.csv")
    if prev.exists():
        old = pd.read_csv(prev)
        for _, r in old.iterrows():
            rows.append({"unit": str(r["checkpoint"]),
                         "family": ("loso_folds" if str(r["checkpoint"])
                                    .startswith("loso/") else
                                    ("task_state" if str(r["checkpoint"])
                                     .startswith("taskstate") else "seeds")),
                         "kind": ("fold" if str(r["checkpoint"])
                                  .startswith("loso/") else "existing"),
                         "pct_K": float(r["pct_K"]),
                         "n_sessions": int(r["n_sessions"]),
                         "source": "variance_decomposition table"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "k_share_by_configuration.csv", index=False)
    print("\n=== K share register ===")
    print(df.sort_values(["family", "pct_K"]).to_string(index=False))

    if not quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = df.sort_values("pct_K")
        fig, ax = plt.subplots(figsize=(9, 10))
        colors = {"reference": "C3", "relabel": "C0", "loso_folds": "C1",
                  "seeds": "0.5", "task_state": "C4"}
        for fam in d["family"].unique():
            sub = d[d["family"] == fam]
            ax.barh(np.arange(len(d)), np.where(d["family"] == fam,
                                                d["pct_K"], np.nan),
                    label=fam, color=colors.get(fam, "grey"))
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d["unit"], fontsize=6.5)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlabel("variance share of K (%), pooled theta=0.5")
        ax.set_title("One convention, every checkpoint")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "fig_k_share_by_configuration.png", dpi=150)
        plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "convention": "rollout k_term/c_term split, decoder images, theta=0.5 "
                      "covariance split, variance-weighted pool over all 76 "
                      "sessions (reference) or the recorded session count",
        "computed_here": len([r for r in rows if r["source"] == "computed here"]),
        "carried_from_table": len([r for r in rows
                                   if r["source"] != "computed here"]),
        "quick": quick,
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
