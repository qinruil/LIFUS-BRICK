#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-domain characterization of the dominant Koopman modes.

For the reference operator (final model checkpoint) and the two task-state
models (pre_model, post_model), the 96 eigenvalues of K are converted into
their time-domain kernels

    c_m(t) = |lambda_m|^t * cos(omega_m * t),   t = 0..T_steps

with physical units obtained by multiplying the discrete step by TR = 2.0 s:

    frequency f_m = |arg lambda_m| / (2 pi TR)  [Hz]
    decay time  tau_m = -TR / ln|lambda_m|      [s]

Each mode is classified as:
    oscillatory    if |arg lambda_m| >= 0.05 rad
    non-oscillatory (pure decay) otherwise
and the stability check verifies that all |lambda_m| < 1 (all modes decay).
A near-unit real pair with vanishing phase is reported as the DC-like
component. The energy-weighted top-10 modes per condition are taken from
condition_participation_by_mode.csv (mean_energy_share).

Outputs (results/validation/k_filter_modes/):
    k_mode_time_analysis.csv        per-model per-mode lambda/arg/f/tau/type
    dominant_modes_summary.csv      top-10 by energy share per model/condition
    stability_dc_check.csv          max|lambda|, counts of near-unit real modes
    fig_mode_panels_{pre,post,reference}.png   8-panel time kernels (s)
    fig_mode_classification.png     |lambda| vs phase scatter per model

Usage:
    python analysis/validation_k_filter_modes.py

Dependencies: numpy, torch, pandas, matplotlib (model loading via repo modules).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TR = 2.0                      # seconds; hard-coded per acquisition protocol
OSC_MIN_RAD = 0.05            # phase threshold for oscillatory classification
NEAR_UNIT = 0.98              # |lambda| threshold for slow/persistent labelling
DC_MAG_MIN = 0.999            # near-unit magnitude for DC-like detection
DC_PHASE_MAX = 0.02           # near-zero phase for DC-like detection

TS_DIR = ROOT / "results" / "figures" / "task_states"
REF_CKPT = ROOT / "results" / "final_model" / "best_model_cls.pt"
OUT = ROOT / "results" / "validation" / "k_filter_modes"


def load_ref_lambda():
    from analysis.koopman_spectrum_stats import load_spectrum
    return load_spectrum(REF_CKPT)["Lambda"]


def mode_table(lam, model):
    mag = np.abs(lam)
    ph = np.angle(lam)
    osc = np.abs(ph) >= OSC_MIN_RAD
    f_hz = np.where(osc, np.abs(ph) / (2 * np.pi * TR), 0.0)
    with np.errstate(divide="ignore"):
        tau = np.where(mag < 1.0, -TR / np.log(np.maximum(mag, 1e-12)), np.inf)
    dc = (np.abs(ph) < DC_PHASE_MAX) & (mag > DC_MAG_MIN)
    return pd.DataFrame({
        "model": model,
        "mode": np.arange(len(lam)),
        "abs_lambda": mag,
        "phase_rad": ph,
        "freq_hz": f_hz,
        "tau_s": tau,
        "oscillatory": osc.astype(int),
        "dc_like": dc.astype(int),
    })


def plot_panels(df, model, out_path, top_n=8):
    d = df.sort_values("abs_lambda", ascending=False).head(top_n)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    t_steps = np.arange(0, 31)
    t_s = t_steps * TR
    for ax, (_, r) in zip(axes.ravel(), d.iterrows()):
        lam = r["abs_lambda"] * np.exp(1j * r["phase_rad"])
        kern = np.abs(lam) ** t_steps * np.cos(r["phase_rad"] * t_steps)
        env = np.abs(lam) ** t_steps
        ax.plot(t_s, env, color="k", lw=0.8, ls="--")
        ax.plot(t_s, -env, color="k", lw=0.8, ls="--")
        ax.plot(t_s, kern, lw=1.3, color=("C3" if r["oscillatory"] else "C0"))
        tag = "osc" if r["oscillatory"] else "decay"
        ax.set_title(f"m{int(r['mode'])}  |L|={r['abs_lambda']:.3f}\n"
                     f"f={r['freq_hz']:.3f}Hz tau={r['tau_s']:.1f}s [{tag}]", fontsize=11)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlim(0, t_s[-1])
    fig.suptitle(f"{model}: time kernels of top-8 modes (|lambda| ranking), TR=2.0 s",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    for model in ["pre_model", "post_model"]:
        ev = pd.read_csv(TS_DIR / model / "koopman_eigenvalues.csv")
        mag = ev["eigenvalue_mag"].to_numpy()
        ph = ev["phase_rad"].to_numpy()
        osc = np.abs(ph) >= OSC_MIN_RAD
        f_hz = np.where(osc, np.abs(ph) / (2 * np.pi * TR), 0.0)
        tau = np.where(mag < 1.0, -TR / np.log(np.maximum(mag, 1e-12)), np.inf)
        dc = (np.abs(ph) < DC_PHASE_MAX) & (mag > DC_MAG_MIN)
        tables.append(pd.DataFrame({
            "model": model, "mode": ev["mode_index"],
            "abs_lambda": mag, "phase_rad": ph, "freq_hz": f_hz, "tau_s": tau,
            "oscillatory": osc.astype(int), "dc_like": dc.astype(int)}))
    lam_ref = load_ref_lambda()
    tables.append(mode_table(lam_ref, "reference"))

    df = pd.concat(tables, ignore_index=True)
    df.to_csv(OUT / "k_mode_time_analysis.csv", index=False)

    stab = df.groupby("model").apply(lambda g: pd.Series({
        "n_modes": len(g),
        "max_abs_lambda": g["abs_lambda"].max(),
        "all_stable_lt_1": bool((g["abs_lambda"] < 1.0).all()),
        "n_oscillatory": int(g["oscillatory"].sum()),
        "n_pure_decay": int((g["oscillatory"] == 0).sum()),
        "n_dc_like": int(g["dc_like"].sum()),
        "n_near_unit": int((g["abs_lambda"] >= NEAR_UNIT).sum()),
        "tau_min_s": g["tau_s"].min(), "tau_max_s": g["tau_s"].max(),
    }), include_groups=False).reset_index()
    stab.to_csv(OUT / "stability_dc_check.csv", index=False)

    cp = pd.read_csv(ROOT / "results" / "spectrum_stats" /
                     "condition_participation_by_mode.csv")
    top = (cp.sort_values(["target", "condition", "mean_energy_share"],
                          ascending=[True, True, False])
             .groupby(["target", "condition"]).head(10)
             .reset_index(drop=True))
    top.to_csv(OUT / "dominant_modes_summary.csv", index=False)

    for model in ["pre_model", "post_model", "reference"]:
        plot_panels(df[df["model"] == model], model,
                    OUT / f"fig_mode_panels_{model.replace('_model','')}.png")

    fig, axes = plt.subplots(3, 1, figsize=(9, 15))
    for ax, model in zip(axes, ["reference", "pre_model", "post_model"]):
        g = df[df["model"] == model]
        c = g["freq_hz"] * 2 * np.pi * TR
        sc = ax.scatter(g["phase_rad"], g["abs_lambda"], s=12, c=c,
                        cmap="viridis", vmin=0, vmax=0.125)
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xlabel("phase (rad)"); ax.set_ylabel("|lambda|")
        ax.set_title(f"{model}")
    fig.colorbar(sc, ax=axes.tolist(), label="frequency (Hz) within Nyquist 0.125",
                 fraction=0.03, pad=0.03)
    fig.suptitle("K spectra: every mode decays (|lambda|<1); colour = frequency",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig_mode_classification.png")
    plt.close(fig)

    print(stab.to_string(index=False))
    print("done:", OUT)


if __name__ == "__main__":
    main()