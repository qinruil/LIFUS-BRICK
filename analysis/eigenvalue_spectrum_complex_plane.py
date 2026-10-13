#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure: eigenvalue spectrum of the reference operator in the complex plane.

Rather than reading the per-model records CSV (which covers the LOSO folds,
seed variants and task-state models), this script loads the reference operator
checkpoint directly (results/final_model/best_model_cls.pt) and recomputes its
96 eigenvalues via the model's own parameterisation, so the figure depicts the
operator that underlies the energy-participation analyses.

Usage:
    py -3.11 analysis/eigenvalue_spectrum_complex_plane.py
Output:
    results/figures/spectrum_stats/eigenvalue_spectrum_complex_plane.png
    results/figures/spectrum_stats/eigenvalue_spectrum_complex_plane.svg
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.koopman_utils import compute_lambda  # noqa: E402
from analysis.koopman_spectrum_stats import load_spectrum  # noqa: E402

REF_CKPT = ROOT / "results" / "final_model" / "best_model_cls.pt"
OUTDIR = ROOT / "results" / "figures" / "spectrum_stats"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def main():
    spec = load_spectrum(REF_CKPT, known_role=None) if False else load_spectrum(REF_CKPT)
    lam = spec["Lambda"]
    re_ = lam.real
    im_ = lam.imag
    mag = np.abs(lam)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    sc = ax.scatter(re_, im_, s=26, c=mag, cmap="viridis", alpha=0.9,
                    edgecolors="white", linewidths=0.3, zorder=3)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="tab:red", ls="--", lw=1.0,
            label="Unit circle (|$\Lambda$| = 1, no decay)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"Re($\Lambda$)")
    ax.set_ylabel(r"Im($\Lambda$)")
    ax.set_title("Reference Koopman operator: eigenvalue spectrum")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"|$\Lambda$| (persistence)")
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "eigenvalue_spectrum_complex_plane.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "eigenvalue_spectrum_complex_plane.svg", bbox_inches="tight")
    print(f"Saved -> {OUTDIR / 'eigenvalue_spectrum_complex_plane.png'}")
    print(f"        {OUTDIR / 'eigenvalue_spectrum_complex_plane.svg'}")


if __name__ == "__main__":
    main()