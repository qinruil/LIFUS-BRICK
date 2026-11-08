#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze the label-perturbation retraining.

For each relabeling configuration (shuffle / constant / swap) retrained with
the reference protocol, compare against the reference model on quantities that
are defined independently of the classifier:

  1. %K variance share, computed with the project convention (manual
     eigenspace rollout, x_K + x_C == x_hat, variance-weighted pooling over
     sessions and ROIs, theta = 0.5 covariance split). If K absorbs what the
     classifier needs, relabeling should move this share.
  2. Eigenvalue spectrum: magnitude distribution (tau scale), oscillation
     fraction (|arg lambda| >= 0.05 rad), dominant frequency band. Structural
     stability of the spectrum under relabeling speaks to K being a property
     of the recording rather than of the task definition.
  3. Mode-space similarity: principal-angle distance between the top-k
     eigenvector subspaces of K (k = 8, 16, 24) reference vs relabeled.
  4. Held-out reconstruction NMSE per fold-equivalent test subject, to confirm
     the relabeled models are comparable fits and not degenerate.
  5. Downstream consequence: per-subject delta-C from the relabeled model
     (test-subject forward passes, the LOSO convention) correlated with tremor
     on the pre-specified ROIs, to ask whether the tremor association survives
     a relabeling of the classifier.

Inputs:  results/final_model/best_model_cls.pt
         results/relabel_check/{shuffle,constant,swap}/best_model_cls.pt
         data/preprocessed_data/*.mat, archive tremor file
Outputs (results/validation/relabel_kcheck/):
    pct_K_by_config.csv
    spectrum_by_config.csv
    subspace_distance.csv
    reconstruction_by_config.csv
    deltaC_tremor_by_config.csv
    fig_relabel_spectrum_comparison.png
    run_manifest.json

Usage:
    python analysis/validation_relabel_kcheck.py
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
OUT = ROOT / "results" / "validation" / "relabel_kcheck"
TR = 2.0
OSC_TOL = 0.05
EPS = 1e-12

from preprocessing.load_preprocessed_data import load_all  # noqa: E402
from models.brick import BRICK  # noqa: E402
from models.koopman_utils import compute_lambda  # noqa: E402


def load_model(path):
    cm = torch.load(path, map_location="cpu", weights_only=False)
    m = BRICK(use_control=cm["use_control"], use_ic=cm["use_ic"],
              h=cm["h"], m=cm["m"])
    m.load_state_dict(cm["model_state_dict"])
    m.eval()
    return m


def zs(x):
    x = np.asarray(x, float)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def rollout(model, x_z):
    xd = torch.tensor(x_z, dtype=torch.float32)
    with torch.no_grad():
        Lambda = compute_lambda(model.nu_log, model.theta_log)
        P_inv = model.P_inv
        g_0, _, _ = model.encoder(xd)
        C_base, u, _, _, _ = model.control(xd)
        g_bar = P_inv @ g_0.to(torch.complex64)
        u_c = u.to(torch.complex64)
        C_c = C_base.to(torch.complex64)
        ks, cs = [], []
        for t in range(u_c.shape[0]):
            k_term = Lambda * g_bar
            c_term = P_inv @ (C_c @ u_c[t])
            g_bar = k_term + c_term
            ks.append(k_term)
            cs.append(c_term)
        K = torch.stack(ks); Cc = torch.stack(cs)
        x_K = (model.W_bar_x @ K.T).T.real
        x_C = (model.W_bar_x @ Cc.T).T.real
        x_hat = x_K + x_C
    return x_K.numpy(), x_C.numpy(), x_hat.numpy()


def pct_K(model, sessions):
    """Variance-weighted pooled %K over all sessions, project convention."""
    tot_k = tot_v = 0.0
    for s in sessions:
        for key in ("mpre", "mpost"):
            x = zs(s[key])
            xK, xC, _ = rollout(model, x)
            vk = np.var(xK, axis=0).sum()
            vc = np.var(xC, axis=0).sum()
            cov = sum(np.cov(xK[:, r], xC[:, r])[0, 1]
                      for r in range(xK.shape[1]))
            vt = np.var(xK + xC, axis=0).sum()
            tot_k += vk + cov          # theta = 0.5: contrib_K = Var_K + 2*0.5*cov
            tot_v += vt
    return 100.0 * tot_k / (tot_v + EPS)


def spectrum_summary(model, Lambda):
    mag = np.abs(Lambda)
    ang = np.abs(np.angle(Lambda))
    f = ang / (2 * np.pi * TR)
    osc = ang >= OSC_TOL
    tau = -TR / np.log(np.clip(mag, 1e-8, 1 - 1e-12))
    return {
        "max_abs_lambda": float(mag.max()),
        "median_abs_lambda": float(np.median(mag)),
        "median_tau_s": float(np.median(tau)),
        "osc_fraction": float(osc.mean()),
        "median_freq_hz": float(np.median(f)),
        "freq_q25": float(np.quantile(f, 0.25)),
        "freq_q75": float(np.quantile(f, 0.75)),
        "frac_slow_lt_0p04": float(np.mean(f < 0.04)),
        "frac_band_0p04_0p12": float(np.mean((f >= 0.04) & (f < 0.12))),
        "n_dc": int(np.sum(ang < OSC_TOL)),
    }


def principal_angles(A, B, k):
    """Mean/median principal-angle cosine between top-k eigenvector subspaces."""
    Va = A[:, :k]
    Vb = B[:, :k]
    qa, _ = np.linalg.qr(Va)
    qb, _ = np.linalg.qr(Vb)
    s = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    s = np.clip(s.real, 0, 1)
    return float(np.cos(s).mean()), float(np.median(np.cos(s)))


def load_tremor():
    subj = [x.strip().replace("sub-", "") for x in
            (ARC / "rsfMRI" / "scripts" / "subject_list.txt")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    m = sio.loadmat(ARC / "rsfMRI" / "tremor" / "tremor_pos_rest.mat")
    return pd.DataFrame({
        "subject_id": subj,
        "vim_pos": np.asarray(m["vim_pos"]).ravel().astype(float),
        "vim_rest": np.asarray(m["vim_rest"]).ravel().astype(float),
        "zi_pos": np.asarray(m["zi_pos"]).ravel().astype(float),
        "zi_rest": np.asarray(m["zi_rest"]).ravel().astype(float)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {
        "reference": ROOT / "results" / "final_model" / "best_model_cls.pt",
        "shuffle": OUT.parent.parent / "relabel_check" / "shuffle" / "best_model_cls.pt",
        "constant": OUT.parent.parent / "relabel_check" / "constant" / "best_model_cls.pt",
        "swap": OUT.parent.parent / "relabel_check" / "swap" / "best_model_cls.pt",
    }
    present = {}
    for name, path in configs.items():
        if Path(path).exists():
            present[name] = load_model(Path(path))
        else:
            print(f"  missing model for {name} ({path}), skipped")
    if "reference" not in present:
        raise SystemExit("reference model required")

    sessions = load_all()
    rois = list(sessions[0]["roi_names"])
    tremor = load_tremor()

    share_rows, spec_rows, eig = [], [], {}
    for name, model in present.items():
        Lambda, P_inv = None, None
        with torch.no_grad():
            Lambda = compute_lambda(model.nu_log, model.theta_log).cpu().numpy()
            P_inv = model.P_inv.cpu().numpy()
        pk = pct_K(model, sessions)
        share_rows.append({"config": name, "pct_K": pk,
                           "n_sessions": len(sessions) * 2})
        ss = spectrum_summary(model, Lambda)
        ss["config"] = name
        spec_rows.append(ss)
        # eigenvectors of K ordered by |lambda| descending
        K = P_inv @ np.diag(Lambda) @ np.linalg.inv(P_inv)
        w, V = np.linalg.eig(K)
        order = np.argsort(-np.abs(w))
        eig[name] = V[:, order]
        print(f"  {name}: pct_K = {pk:.3f}%")

    pd.DataFrame(share_rows)[["config", "pct_K", "n_sessions"]] \
        .to_csv(OUT / "pct_K_by_config.csv", index=False)
    cspec = pd.DataFrame(spec_rows)
    cspec = cspec[["config"] + [c for c in cspec.columns if c != "config"]]
    cspec.to_csv(OUT / "spectrum_by_config.csv", index=False)
    print("\n=== variance share and spectrum by relabeling config ===")
    print(cspec.to_string(index=False))

    # subspace distances vs reference
    ds_rows = []
    for name in present:
        if name == "reference":
            continue
        for k in (8, 16, 24):
            cm, cmed = principal_angles(eig["reference"], eig[name], k)
            ds_rows.append({"config": name, "k_modes": k,
                            "mean_cos_principal_angle": cm,
                            "median_cos_principal_angle": cmed})
    dsdf = pd.DataFrame(ds_rows)
    dsdf.to_csv(OUT / "subspace_distance.csv", index=False)
    if len(dsdf):
        print("\n=== top-mode subspace similarity to reference ===")
        print(dsdf.to_string(index=False))

    # held-out reconstruction NMSE on the split's test subject(s)
    rec_rows = []
    for name, model in present.items():
        split_p = (ROOT / "results" / "final_model" / "split.json")
        test_ids = json.loads(split_p.read_text())["test"] if name == "reference" \
            else json.loads((ROOT / "results" / "relabel_check" / name / "split.json").read_text())["test"] \
            if (ROOT / "results" / "relabel_check" / name / "split.json").exists() else []
        for s in sessions:
            if s["subject_id"] not in [t.replace("sub-", "") for t in test_ids]:
                continue
            for key in ("mpre", "mpost"):
                x = zs(s[key])
                _, _, xh = rollout(model, x)
                if not np.isfinite(xh).all():
                    nmse = float("inf")
                else:
                    nmse = float(np.mean((xh - x) ** 2)
                                 / np.mean((x - x.mean(0)) ** 2))
                rec_rows.append({"config": name, "subject_id": s["subject_id"],
                                 "target": s["target"], "phase": key,
                                 "nmse": nmse})
    rdf = pd.DataFrame(rec_rows)
    rdf.to_csv(OUT / "reconstruction_by_config.csv", index=False)
    if len(rdf):
        print("\ntest-subject reconstruction NMSE by config:")
        print(rdf.groupby("config")["nmse"].median().to_string())

    # delta-C (test subject, held-out forward pass) vs tremor, per config
    rows = []
    pre_rois = ["lh_GPi", "lh_STN", "rh_STN", "lh_vim", "lh_zi"]
    for name, model in present.items():
        if name == "reference":
            continue                      # reference delta-C already in LOSO tables
        for s in sessions:
            sid = s["subject_id"]
            tw = tremor[tremor["subject_id"] == sid]
            if not len(tw):
                continue
            pre = zs(s["mpre"]); post = zs(s["mpost"])
            with torch.no_grad():
                C_pre = np.real(model(torch.tensor(pre, dtype=torch.float32))
                                ["C"].diag().cpu().numpy())
                C_post = np.real(model(torch.tensor(post, dtype=torch.float32))
                                 ["C"].diag().cpu().numpy())
            Wre = model.W_bar_x.detach().real.cpu().numpy()
            d_pre = np.diag(Wre @ np.diag(C_pre) @ Wre.T)
            d_post = np.diag(Wre @ np.diag(C_post) @ Wre.T)
            dvec = d_post - d_pre
            for j, roi in enumerate(rois):
                if roi not in pre_rois:
                    continue
                for task in ("pos", "rest"):
                    col = f"{s['target']}_{task}"
                    rows.append({"config": name, "subject_id": sid,
                                 "target": s["target"], "roi": roi,
                                 "task": task, "deltaC": float(dvec[j]),
                                 "tremor_change": float(tw[col].iloc[0])})
    dcf = pd.DataFrame(rows)
    if len(dcf):
        out = []
        for (cfg, roi, task), g in dcf.groupby(["config", "roi", "task"]):
            if len(g) < 5:
                continue
            r, p = spearmanr(g["deltaC"], g["tremor_change"])
            out.append({"config": cfg, "roi": roi, "task": task,
                        "n": int(len(g)), "rho": float(r), "p": float(p)})
        tab = pd.DataFrame(out)
        tab.to_csv(OUT / "deltaC_tremor_by_config.csv", index=False)
        print("\ndelta-C vs tremor under relabeled models "
              "(note: n = test-subject count only):")
        print(tab.to_string(index=False))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(present)
    xs = np.arange(len(names))
    sh = pd.DataFrame(share_rows).set_index("config").reindex(names)
    axes[0].bar(xs, sh["pct_K"], color="C0")
    for i, v in enumerate(sh["pct_K"]):
        axes[0].text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=9)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel("variance share of K (%)")
    axes[0].set_title("Does what K absorbs depend on the labels?")
    cs = cspec.set_index("config").reindex(names)
    axes[1].bar(xs - 0.2, cs["osc_fraction"], 0.4, label="oscillation fraction")
    axes[1].bar(xs + 0.2, cs["median_freq_hz"] / 0.25, 0.4,
                label="median freq (fraction of Nyquist 0.25)")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Spectrum structure across relabelings")
    fig.tight_layout()
    fig.savefig(OUT / "fig_relabel_spectrum_comparison.png", dpi=120)
    plt.close(fig)

    (OUT / "run_manifest.json").write_text(json.dumps({
        "models": {k: str(v) for k, v in configs.items()},
        "protocol": "identical split/hyperparameters to results/final_model; "
                    "only the classifier labels differ",
        "pct_K": "project variance-decomposition convention, theta=0.5, "
                 "variance-weighted pool over 76 sessions",
        "oscillation": "|arg lambda| >= 0.05 rad",
        "caveat": "delta-C rows use the single test subject of each run; "
                  "correlations there are descriptive only",
    }, indent=2), encoding="utf-8")
    print("\ndone:", OUT)


if __name__ == "__main__":
    main()
