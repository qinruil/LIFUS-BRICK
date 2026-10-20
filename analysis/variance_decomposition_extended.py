#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended variance decomposition (four parts).

Part A (freq):  Frequency-band resolved %C / %K. The exact K/C split of the
                BOLD reconstruction (x_K + x_C == x_hat, from the manual
                eigenspace rollout) is band-passed per ROI into four bands
                below the Nyquist limit (TR = 2.0 s, Nyquist = 0.25 Hz), and
                the variance-weighted %C / %K is recomputed inside each band
                for the four conditions (vim_pre, vim_post, zi_pre, zi_post).

Part B (split): Sensitivity of %C / %K to the treatment of the shared
                covariance term. With total = Var_K + Var_C + 2*Cov and
                contributions defined by a parameter theta in [0,1]:
                    contrib_K = Var_K + 2*theta*Cov
                    contrib_C = Var_C + 2*(1-theta)*Cov
                which sums to total exactly for every theta. theta=0.5 is the
                half-split used by the existing variance_decomposition();
                theta=1 and theta=0 give the extremes. Reported as a band of
                %C / %K values per condition.

Part C (autocorr): Temporal autocorrelation scale of the control inputs u_t.
                For every session, per-mode lag-1 autocorrelation rho1 is
                computed, and the session-level effective sample size /
                decorrelation time is derived (TR * (1+rho)/(1-rho)).
                pre/post differences are reported descriptively for VIM and ZI.

Part D (scaling): How the %K share depends on training-set size. Existing
                checkpoints provide only two N_train levels (13 LOSO/seed
                runs and 17 task-state runs); the shared-operator
                variance-weighted %K over the full 76-session set is computed
                per checkpoint and reported as two scaling points, with the
                explicit caveat that a genuine scaling curve requires
                retraining a gradient of N_train values.

Dependencies: numpy, torch, scipy, standard library (no pandas, no
statsmodels, no matplotlib). All helper functions that the existing
analysis_helper_functions provides are re-implemented locally to avoid its
pandas dependency.

Output directory: results/variance_decomposition/
    freq_band_decomposition.csv
    split_rule_sensitivity.csv
    u_autocorrelation.csv
    scaling_points.csv
    session_rollout_cache.pt   (x_K, x_C per reference model, reused by A/B)
    run_manifest.json

Usage:
    python analysis/variance_decomposition_extended.py --part freq
    python analysis/variance_decomposition_extended.py --part all
    python analysis/variance_decomposition_extended.py --quick   (2 folds only)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sp_signal
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import M, N_ROIS, H  # noqa: E402
from models.koopman_utils import compute_lambda  # noqa: E402
from models.brick import BRICK  # noqa: E402
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402

TR = 2.0
BANDS = [
    ("delta", 0.01, 0.04),
    ("theta", 0.04, 0.08),
    ("alpha", 0.08, 0.12),
    ("beta_fast", 0.12, 0.25),
]
NYQUIST = 0.25

REF_CKPT = ROOT / "results" / "final_model" / "best_model_cls.pt"
TASK_DIR = ROOT / "results" / "trained_models" / "task_states"
LOSO_DIR = ROOT / "results" / "trained_models" / "loso_19_fold_beta_0.2_13to1to5_split"
SEED_DIR = ROOT / "results" / "trained_models" / "sweep_3_seeds"
OUT_DIR = ROOT / "results" / "variance_decomposition"
EPS = 1e-12


# --------------------------------------------------------------------------
# IO helpers (no pandas)
# --------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], order: list[str] | None = None):
    """
    Write rows dicts to CSV. signature: (path, rows, optional order).
    All call sites pass (path, fieldnames, rows) historically; the flexible
    form below accepts both: if the second argument is a list of strings it is
    treated as fieldnames and the third as rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # flexible signature: (path, fieldnames, rows) or (path, rows, order)
    if isinstance(rows, list) and rows and all(isinstance(x, str) for x in rows):
        if not order:
            return
        fieldnames = rows
        rows = order
    else:
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows:
        return
    keys = fieldnames
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else _fmt(r.get(k))) for k in keys})
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (np.floating,)):
        return _fmt(float(v))
    return str(v)


def load_brick(path: Path, verbose=True):
    """Load a BRICK checkpoint with num_classes guessed from the state dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    for nc in (2, 3, 4):
        m = BRICK(h=int(ckpt.get("h", H)), m=int(ckpt.get("m", M)),
                  use_control=bool(ckpt.get("use_control", True)),
                  use_ic=bool(ckpt.get("use_ic", True)), num_classes=nc)
        try:
            m.load_state_dict(sd, strict=True)
        except Exception:
            continue
        m.eval()
        if verbose:
            print(f"    loaded {Path(path).name}: num_classes={nc}, "
                  f"use_control={m.use_control}, use_ic={m.use_ic}")
        return m, ckpt
    raise RuntimeError(f"cannot load {path} with num_classes in (2,3,4)")


def sessions_from_load_all():
    """Return list of session dicts with x (T,N) float32, label fields."""
    subs = load_all()
    out = []
    for s in subs:
        for phase in ("mpre", "mpost"):
            x = np.asarray(s[phase], dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError(f"non-finite BOLD for {s['subject_id']}/{s['target']}/{phase}")
            out.append({
                "subject_id": s["subject_id"], "target": s["target"],
                "condition": "pre" if phase == "mpre" else "post",
                "order_group": s["group"],
                "x": x,
            })
    return out


# --------------------------------------------------------------------------
# Core functions reproduced locally (from C_stability_perturbation_analysis,
# without its pandas dependency)
# --------------------------------------------------------------------------
def znorm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


def to_complex_tensor(x):
    return torch.as_tensor(x, dtype=torch.complex64)


def encode_and_control(model, x):
    with torch.no_grad():
        if model.use_ic:
            g_0, _, _ = model.encoder(x)
        else:
            g_0 = torch.zeros(model.m, device=x.device)
        C_base, u, _, _, _ = model.control(x)
    return g_0, C_base, u


def manual_rollout(Lambda, P_inv, g_0, u, W_bar_x, C_seq, return_decomposition=False):
    with torch.no_grad():
        Tlen = u.shape[0]
        g0_c = g_0.to(torch.complex64)
        u_c = u.to(torch.complex64)
        g_bar = P_inv @ g0_c
        traj, k_terms, c_terms = [], [], []
        for t in range(Tlen):
            C_t = C_seq[t] if isinstance(C_seq, (list, tuple)) else C_seq
            C_t_c = C_t.to(torch.complex64)
            k_term = Lambda * g_bar
            c_term = P_inv @ (C_t_c @ u_c[t])
            g_bar = k_term + c_term
            traj.append(g_bar)
            if return_decomposition:
                k_terms.append(k_term)
                c_terms.append(c_term)
        g_bar_traj = torch.stack(traj, dim=0)
        x_hat = (W_bar_x @ g_bar_traj.T).T.real
        if not return_decomposition:
            return x_hat
        k_traj = torch.stack(k_terms, dim=0)
        c_traj = torch.stack(c_terms, dim=0)
        x_K = (W_bar_x @ k_traj.T).T.real
        x_C = (W_bar_x @ c_traj.T).T.real
        return x_hat, x_K, x_C


def variance_components(x_K: np.ndarray, x_C: np.ndarray):
    """Per-ROI Var_K, Var_C, Cov, Var_total. Cov here is the full covariance
    (np.cov off-diagonal), so Var_total == Var_K + Var_C + 2*Cov exactly."""
    Tlen, Nrois = x_K.shape
    vk = np.var(x_K, axis=0)
    vc = np.var(x_C, axis=0)
    cov = np.zeros(Nrois)
    for r in range(Nrois):
        cov[r] = np.cov(x_K[:, r], x_C[:, r])[0, 1]
    vt = np.var(x_K + x_C, axis=0)
    return vk, vc, cov, vt


def pct_from_components(vk, vc, cov, vt, theta):
    """theta in [0,1]: share of 2*Cov attributed to K. Sums to 100% exactly."""
    contrib_k = vk + 2 * theta * cov
    contrib_c = vc + 2 * (1 - theta) * cov
    pct_k = 100 * contrib_k / (vt + EPS)
    pct_c = 100 * contrib_c / (vt + EPS)
    return pct_k, pct_c


def pooled_pct(rows_by_component, theta):
    """variance-weighted pooled %K/%C over (session, ROI) components.
    contributions are summed over ROIs first, then over sessions, so every
    value here is a scalar (per-ROI vectors are summed)."""
    tot = float(sum(np.sum(r["vt"]) for r in rows_by_component))
    if tot <= 0:
        return float("nan"), float("nan")
    ok = sum(float(np.sum(r["vk"] + 2 * theta * r["cov"])) for r in rows_by_component)
    pct_k = 100 * ok / tot
    return pct_k, 100 - pct_k


# --------------------------------------------------------------------------
# Band-pass filter helper (zero-phase, per ROI time series)
# --------------------------------------------------------------------------
def bandpass_filter(x: np.ndarray, lo_hz: float, hi_hz: float, tr: float = TR):
    """x: (T, N) real. Zero-phase butterworth band-pass per column.
    Returns filtered (T, N). Edges are handled by filtfilt padding."""
    fs = 1.0 / tr
    nyq = fs / 2.0
    lo = max(lo_hz, 1e-6) / nyq
    hi = min(hi_hz, nyq - 1e-6) / nyq
    if hi <= lo:
        return np.zeros_like(x)
    sos = sp_signal.butter(4, [lo, hi], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, x, axis=0)


# --------------------------------------------------------------------------
# Sessions and reference decomposition cache
# --------------------------------------------------------------------------
def run_reference_decomposition(quick: bool):
    """Compute x_K, x_C per session under the reference operator and return
    (sessions, list_of dicts with subject/target/condition/xK/xC)."""
    sessions = sessions_from_load_all()
    model, _ = load_brick(REF_CKPT)
    with torch.no_grad():
        Lam = compute_lambda(model.nu_log.double(), model.theta_log.double()).to(torch.complex64)
        P_inv = model.P_inv
        W_bar_x = model.W_bar_x
    recs = []
    t0 = time.time()
    for i, s in enumerate(sessions):
        x = znorm(torch.from_numpy(s["x"]))
        g_0, C_base, u = encode_and_control(model, x)
        x_hat, x_K, x_C = manual_rollout(
            Lam, P_inv, g_0, u, W_bar_x, C_seq=C_base, return_decomposition=True)
        recs.append({
            "subject_id": s["subject_id"], "target": s["target"],
            "condition": s["condition"], "order_group": s["order_group"],
            "xK": x_K.numpy().astype(np.float64),
            "xC": x_C.numpy().astype(np.float64),
            "u": u.numpy().astype(np.float64),
        })
        if (i + 1) % 19 == 0:
            print(f"    reference decomposition {i + 1}/{len(sessions)} "
                  f"({time.time() - t0:.1f}s)")
        if quick and i >= 5:
            break
    return sessions, recs


# --------------------------------------------------------------------------
# Part A: frequency bands
# --------------------------------------------------------------------------
def part_freq(recs, quick=False):
    rows = []
    for r in recs:
        xK, xC = r["xK"], r["xC"]
        for bname, lo, hi in BANDS:
            fK = bandpass_filter(xK, lo, hi)
            fC = bandpass_filter(xC, lo, hi)
            vk, vc, cov, vt = variance_components(fK, fC)
            # theta = 0.5 half-split: contrib_K = vk + cov, scalar per session x band
            total_var = float(np.sum(vt))
            contrib_k = float(np.sum(vk + cov))
            pct_K = 100 * contrib_k / (total_var + EPS) if total_var > 0 else float("nan")
            rows.append({
                "subject_id": r["subject_id"], "target": r["target"],
                "condition": r["condition"], "order_group": r["order_group"],
                "band": bname, "band_lo_hz": lo, "band_hi_hz": hi,
                "roi": "pooled", "pct_K": pct_K,
                "pct_C": 100 - pct_K if total_var > 0 else float("nan"),
            })
    write_csv(OUT_DIR / "freq_band_decomposition.csv",
              ["subject_id", "target", "condition", "order_group", "band",
               "band_lo_hz", "band_hi_hz", "roi", "pct_K", "pct_C"], rows)
    # condition x band variance-weighted pooled
    agg = {}
    for r in recs:
        for bname, lo, hi in BANDS:
            fK = bandpass_filter(r["xK"], lo, hi)
            fC = bandpass_filter(r["xC"], lo, hi)
            vk, vc, cov, vt = variance_components(fK, fC)
            key = (r["target"], r["condition"], bname)
            agg.setdefault(key, []).append({"vk": vk, "vc": vc, "cov": cov, "vt": vt})
    pooled = []
    for (tgt, cond, bname), comps in sorted(agg.items()):
        pctk, pctc = pooled_pct(comps, 0.5)
        pooled.append({"target": tgt, "condition": cond, "band": bname,
                       "pct_K_pooled": pctk, "pct_C_pooled": pctc})
    write_csv(OUT_DIR / "freq_band_pooled.csv",
              ["target", "condition", "band", "pct_K_pooled", "pct_C_pooled"], pooled)
    print("\n--- Part A: condition x band pooled %K ---")
    for r in pooled:
        print(f"  {r['target']:<4} {r['condition']:<5} {r['band']:<10} %K={r['pct_K_pooled']:.2f}%")


# --------------------------------------------------------------------------
# Part B: covariance-assignment sensitivity
# --------------------------------------------------------------------------
def part_split(recs, thetas=(0.0, 0.5, 1.0)):
    # per (session) components
    agg_rows = []
    comps_by_key = {}
    for r in recs:
        vk, vc, cov, vt = variance_components(r["xK"], r["xC"])
        key = (r["target"], r["condition"])
        comps_by_key.setdefault(key, []).append({"vk": vk, "vc": vc, "cov": cov, "vt": vt})
    for key, comps in sorted(comps_by_key.items()):
        tgt, cond = key
        row = {"target": tgt, "condition": cond}
        for th in thetas:
            pk, pc = pooled_pct(comps, th)
            row[f"pct_K_theta{th:g}"] = pk
            row[f"pct_C_theta{th:g}"] = pc
        row["pct_K_range"] = f"{row['pct_K_theta0']:.2f} to {row['pct_K_theta1']:.2f}"
        agg_rows.append(row)
    write_csv(OUT_DIR / "split_rule_sensitivity.csv",
              ["target", "condition", "pct_K_theta0", "pct_C_theta0",
               "pct_K_theta0.5", "pct_C_theta0.5",
               "pct_K_theta1", "pct_C_theta1", "pct_K_range"], agg_rows)
    print("\n--- Part B: covariance-assignment sensitivity (pooled %K) ---")
    for r in agg_rows:
        print(f"  {r['target']:<4} {r['condition']:<5} "
              f"theta0={r['pct_K_theta0']:.2f}  theta0.5={r['pct_K_theta0.5']:.2f}  "
              f"theta1={r['pct_K_theta1']:.2f}")


# --------------------------------------------------------------------------
# Part C: u_t autocorrelation
# --------------------------------------------------------------------------
def part_autocorr(recs):
    rows = []
    for r in recs:
        u = r["u"]  # (T, M)
        u = u - u.mean(axis=0, keepdims=True)
        denom = np.sum(u ** 2, axis=0)
        num = np.sum(u[1:] * u[:-1], axis=0)
        rho1 = np.where(denom > EPS, num / (denom + EPS), np.nan)
        rho1 = rho1[~np.isnan(rho1)]
        if rho1.size == 0:
            continue
        rho_med = float(np.median(rho1))
        eff_n = float(M * (1 - rho_med) / (1 + rho_med + EPS))
        decorr_s = float(TR * (1 + rho_med) / (1 - rho_med + EPS))
        rows.append({
            "subject_id": r["subject_id"], "target": r["target"],
            "condition": r["condition"], "order_group": r["order_group"],
            "n_modes": int(rho1.size),
            "median_rho1": rho_med,
            "eff_sample_n_equivalent": eff_n,
            "decorrelation_time_s": decorr_s,
        })
    write_csv(OUT_DIR / "u_autocorrelation.csv",
              ["subject_id", "target", "condition", "order_group", "n_modes",
               "median_rho1", "eff_sample_n_equivalent", "decorrelation_time_s"], rows)
    # pre/post paired description per target
    print("\n--- Part C: u_t lag-1 autocorrelation (median over modes) ---")
    for tgt in ("vim", "zi"):
        pre = [r for r in rows if r["target"] == tgt and r["condition"] == "pre"]
        post = [r for r in rows if r["target"] == tgt and r["condition"] == "post"]
        d = {}
        for r in pre:
            d.setdefault(r["subject_id"], {})["pre"] = r["median_rho1"]
        for r in post:
            d.setdefault(r["subject_id"], {})["post"] = r["median_rho1"]
        pairs = [(v["pre"], v["post"]) for v in d.values() if "pre" in v and "post" in v]
        if not pairs:
            continue
        a = np.array([x[0] for x in pairs]); b = np.array([x[1] for x in pairs])
        delta = b - a
        try:
            w = sp_stats.wilcoxon(b, a)
            p_w = float(w.pvalue)
        except Exception:
            p_w = float("nan")
        print(f"  {tgt:<4} n={len(pairs)}  pre rho1 med={np.median(a):.4f}  "
              f"post rho1 med={np.median(b):.4f}  delta med={np.median(delta):+.4f}  "
              f"wilcoxon p={p_w:.4f}")

    return rows


# --------------------------------------------------------------------------
# Part D: scaling with training-set size
# --------------------------------------------------------------------------
def collect_checkpoint_groups():
    groups = []
    for fold in sorted(LOSO_DIR.glob("fold_sub-*")):
        ck = fold / "best_model_cls.pt"
        if ck.exists():
            groups.append((f"loso/{fold.name}", ck, 13))
    for sd in sorted(SEED_DIR.iterdir()):
        ck = sd / "best_model_cls.pt"
        if ck.is_file():
            groups.append((f"seed/{sd.name}", ck, 13))
    for cond in ("pre_model", "post_model"):
        ck = TASK_DIR / cond / "best_model_cls.pt"
        if ck.exists():
            groups.append((f"taskstate/{cond}", ck, 17))
    return groups


def part_scaling(quick=False):
    groups = collect_checkpoint_groups()
    if quick:
        groups = groups[:6]
    sessions = sessions_from_load_all()
    all_rows = []
    print(f"\n--- Part D: %K vs N_train over {len(groups)} checkpoints ---")
    for name, ck, n_train in groups:
        model, _ = load_brick(ck, verbose=False)
        with torch.no_grad():
            Lam = compute_lambda(model.nu_log.double(), model.theta_log.double()).to(torch.complex64)
            P_inv = model.P_inv
            W_bar_x = model.W_bar_x
        comps = []
        for s in sessions:
            x = znorm(torch.from_numpy(s["x"]))
            g_0, C_base, u = encode_and_control(model, x)
            _, x_K, x_C = manual_rollout(
                Lam, P_inv, g_0, u, W_bar_x, C_seq=C_base, return_decomposition=True)
            vk, vc, cov, vt = variance_components(x_K.numpy(), x_C.numpy())
            comps.append({"vk": vk, "vc": vc, "cov": cov, "vt": vt})
        pk, pc = pooled_pct(comps, 0.5)
        all_rows.append({"checkpoint": name, "n_train": n_train, "pct_K": pk, "pct_C": pc,
                         "n_sessions": len(sessions)})
        print(f"  {name:<42} N_train={n_train:<3} %K={pk:.2f}  %C={pc:.2f}")
    write_csv(OUT_DIR / "scaling_points.csv",
              ["checkpoint", "n_train", "pct_K", "pct_C", "n_sessions"], all_rows)
    return all_rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="extended variance decomposition")
    ap.add_argument("--part", choices=["freq", "split", "autocorr", "scaling", "all"],
                    default="all")
    ap.add_argument("--quick", action="store_true",
                    help="small subset for a fast smoke test")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("Extended variance decomposition: frequency bands, covariance split,")
    print("u_t autocorrelation, and scaling with training-set size")
    print("=" * 78)
    print(f"TR = {TR} s, Nyquist = {NYQUIST} Hz, bands = "
          f"{[b[0] for b in BANDS]}, M = {M}, N_ROIS = {N_ROIS}")

    # cache reference decomposition once for freq/split
    recs = None
    if args.part in ("freq", "split", "all"):
        print("\n[reference] decomposing 76 sessions under the reference operator")
        _, recs = run_reference_decomposition(args.quick)

    if args.part in ("freq", "all") and recs:
        part_freq(recs, args.quick)
    if args.part in ("split", "all") and recs:
        part_split(recs)
    if args.part in ("autocorr", "all"):
        uses = sessions_from_load_all()
        # autocorr only needs u, which is in the reference decomposition; reuse it
        _, recs_u = run_reference_decomposition(args.quick) if recs is None else (None, recs)
        part_autocorr(recs_u if recs_u else [])
    if args.part in ("scaling", "all"):
        part_scaling(args.quick)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({
            "script": "analysis/variance_decomposition_extended.py",
            "tr_s": TR, "nyquist_hz": NYQUIST,
            "bands_hz": {b[0]: [b[1], b[2]] for b in BANDS},
            "theta_values": [0.0, 0.5, 1.0],
            "reference_checkpoint": str(REF_CKPT.relative_to(ROOT)),
            "task_states_dir": str(TASK_DIR.relative_to(ROOT)),
            "loso_dir": str(LOSO_DIR.relative_to(ROOT)),
            "seed_dir": str(SEED_DIR.relative_to(ROOT)),
            "elapsed_sec": round(time.time() - t0, 2),
        }, fh, indent=2, ensure_ascii=False)
    print(f"\n[done] {OUT_DIR} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()