# analysis/koopman_spectrum_stats.py
"""
================================================================================
Cross-condition statistical analysis of the Koopman eigenvalue spectrum
================================================================================

Overview
--------
Statistical comparison of the Koopman eigenvalue spectrum across conditions.
The eigenvalues Lambda of K -- magnitude encodes decay rate, phase encodes
oscillation frequency -- are compared statistically, motivated by the cohort
design (2 targets x 2 conditions, plus a 4-hour washout and two treatment
orders).

Identifiability constraints
---------------------------
In BRICK, K is a SINGLE SHARED operator per trained model. There is therefore
NO per-subject, per-target or per-session version of Lambda, which means:

  * A four-condition contrast "VIM-pre vs VIM-post vs ZI-pre vs ZI-post" of the
    spectrum itself is NOT representable in one model. It is only available
    through the two independently trained task-state models (pre_model,
    post_model), i.e. n = 1 model per condition. Such a contrast is
    DESCRIPTIVE; no inferential p-value exists for it.
  * What IS testable, and what this script provides:
      (A) fold-level replication -- the distribution of spectra across the 19
          LOSO folds, and across seed/beta variants, answers "how much of the
          spectral shift is smaller than the spread between independently
          trained operators?"
      (B) condition-level mode ENERGETICS -- holding ONE operator fixed and
          running its forward pass on each of the four conditions yields a
          per-mode energy participation profile. The operator's spectrum is
          shared, but WHICH modes a given condition excites is condition
          specific, with subject x target as a legitimate unit of replication
          (19 subjects x 2 targets = 38 sessions per condition). This is the
          testable version of "the spectrum shifts across conditions".

The script reports both: spectral-quantity distributions per model
(A) and paired cross-condition tests of spectral energy participation (B).

Statistical design
------------------
* Mode ordering is NOT matched between independently trained models (no shared
  basis exists). All cross-model comparisons are distribution-level, using the
  1-D Wasserstein-2 distance between empirical samples of |Lambda| and of
  |arg(Lambda)|.
* The null for "is the pre/post operator difference larger than ordinary
  retraining noise?" is the empirical distribution of pairwise distances within
  the pool of independently trained models (LOSO folds, seeds, betas). The
  observed pre-vs-post distance is scored against that pool by percentile and
  by a standardised z (descriptive convention).
* Within-condition paired tests (pre vs post per subject) use the paired t-test
  and Wilcoxon signed-rank test; the four-condition within-subject contrast
  uses the Friedman test. Because treatment order is a design-level confound, every test is
  additionally stratified by
  VIM_first / ZI_first.
* Multiple comparisons across the 24 ROIs / test families are BH-FDR corrected.
  statsmodels is not available in this project's environment, so BH-FDR is
  implemented here.

Units and the TR caveat
-----------------------
|Lambda| and arg(Lambda) are scale-free per-step quantities and are reported as
primary. Converting to Hz / seconds requires the repetition time:

    freq_hz = |arg| / (2*pi*TR),  period_s = 2*pi*TR/|arg|,  tau_s = -TR/ln|Lambda|

config.py defines no TR, the .mat headers carry no acquisition metadata, and
analysis/compare_pre_post.py hard-codes TR = 2.0 s. Every physical-unit column
written here is therefore CONDITIONAL on --tr and is labelled as such. Confirm
the real TR before quoting any Hz or second value.

Usage
-----
    python analysis/koopman_spectrum_stats.py                       # full run
    python analysis/koopman_spectrum_stats.py --skip-participation   # spectra only, fast
    python analysis/koopman_spectrum_stats.py --tr 2.85 --reference results/final_model/best_model_cls.pt
    python analysis/koopman_spectrum_stats.py --folds loso_19_fold_beta_0.2_13to1to5_split

Outputs (all under results/spectrum_stats/, nothing existing is overwritten)
    model_spectrum_summary.csv      per-model spectral summaries + scale-free stats
    eigenvalue_records.csv          long-format, one row per model per mode
    pairwise_wasserstein_models.csv  all cross-model spectral distances
    pre_post_vs_retraining_null.csv observed pre/post shift scored against that null
    condition_participation.csv     per subject/target/condition mode-energy metrics
    condition_participation_by_mode.csv per-mode participation, 4 conditions
    condition_stats_tests.csv       paired / Friedman tests + BH-FDR
    tremor_band_modes.csv           near-unit oscillatory modes vs tremor bands
    run_manifest.json               exact inputs, settings and caveats of the run

Dependencies: numpy, torch, scipy, stdlib only (no pandas, no matplotlib).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402
from scipy import stats  # noqa: E402

from config import N_ROIS, H, M, R_MIN, R_MAX  # noqa: E402
from models.koopman_utils import compute_lambda  # noqa: E402
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS  # noqa: E402

TRAINED_MODELS_DIR = ROOT_DIR / "results" / "trained_models"
OUT_DIR = ROOT_DIR / "results" / "spectrum_stats"

# Primary LOSO run (beta = 0.2).
DEFAULT_LOSO_RUN = "loso_19_fold_beta_0.2_13to1to5_split"
ALT_BETA_RUN = "loso_19_fold_beta_0.05_13to1to5_split"
TASK_STATES_RUN = "task_states"
SEED_SWEEP_RUN = "sweep_3_seeds"

# The three model kinds.
ROLE_MAIN = "loso_fold"
ROLE_PRE = "task_state_pre"
ROLE_POST = "task_state_post"

# Behavioural tremor bands, for the near-unit oscillatory screen only.
# These are clinical priors, not measurements from this dataset.
TREMOR_BANDS_HZ = (("postural", 4.0, 6.0), ("resting", 4.0, 7.0))
NEAR_UNIT_QUANTILE = 0.90       # "near-unit" = |Lambda| at/above this quantile
PERSISTENCE_SLOW = 0.95         # |Lambda| threshold used for slow-mode labelling
OSCILLATION_MIN_RAD = 0.05      # below this |arg| a mode is treated as non-oscillatory


# ================================================================================
# 1. EXTRACTION -- spectrum from a checkpoint, without touching the full model
# ================================================================================

def lambda_from_state_dict(nu_log, theta_log):
    """
    Recompute the complex spectrum from the trained eigenvalue parameters, using
    the model's own parameterisation

        Lambda_j = exp(-exp(nu_log_j) + i * exp(theta_log_j))

    which guarantees |Lambda_j| < 1 by construction. Computed in float64 for
    numerical fidelity; the ordering convention is the model's own, i.e.
    mode_index m = the diagonal slot of Lambda, with no sorting applied.
    """
    if nu_log.shape != theta_log.shape:
        raise ValueError(f"nu_log{tuple(nu_log.shape)} vs theta_log{tuple(theta_log.shape)}")
    with torch.no_grad():
        lam = compute_lambda(nu_log.double(), theta_log.double())
    return lam.detach().cpu().numpy().astype(np.complex128)


def load_spectrum(ckpt_path: Path) -> dict:
    """
    Read only the Koopman spectral parameters from a checkpoint.

    Deliberately avoids instantiating BRICK: the whole point of this analysis is
    the operator's spectrum, and loading state_dict into a freshly built model
    would couple this script to classifier head sizes (2 / 3 / 4 classes across
    runs) that are irrelevant here.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    missing = [k for k in ("nu_log", "theta_log", "P_inv", "W_bar_x") if k not in sd]
    if missing:
        raise KeyError(f"{ckpt_path.name} lacks {missing}; not a BRICK checkpoint")

    Lambda = lambda_from_state_dict(sd["nu_log"], sd["theta_log"])
    return {
        "path": ckpt_path,
        "Lambda": Lambda,
        "P_inv": sd["P_inv"].detach().cpu().numpy().astype(np.complex128),
        "W_bar_x": sd["W_bar_x"].detach().cpu().numpy().astype(np.complex128),
        "h": int(ckpt.get("h", H)),
        "m": int(ckpt.get("m", M)),
        "use_control": bool(ckpt.get("use_control", True)),
        "use_ic": bool(ckpt.get("use_ic", True)),
        "val_loss_total": ckpt.get("val_loss_total"),
        "val_loss_cls": ckpt.get("val_loss_cls"),
        "epoch": ckpt.get("epoch"),
    }


def collect_model_paths(folds_run: str, include_alt_beta: bool, include_seeds: bool) -> list:
    """
    Enumerate every checkpoint whose spectrum we can use, tagged with its role.

    Only best_model_cls.pt is used, to keep one checkpoint convention per run.
    Returns a list of dicts with
    keys: name, role, group, path.
    """
    entries = []

    def add(dir_path: Path, name: str, role: str, group: str):
        ckpt = dir_path / "best_model_cls.pt"
        if ckpt.exists():
            entries.append({"name": name, "role": role, "group": group, "path": ckpt})

    primary = TRAINED_MODELS_DIR / folds_run
    if not primary.exists():
        raise FileNotFoundError(f"primary LOSO run not found: {primary}")
    for fold in sorted(primary.iterdir()):
        if fold.is_dir() and fold.name.startswith("fold_"):
            add(fold, fold.name, ROLE_MAIN, folds_run)

    if include_alt_beta:
        alt = TRAINED_MODELS_DIR / ALT_BETA_RUN
        if alt.exists():
            for fold in sorted(alt.iterdir()):
                if fold.is_dir() and fold.name.startswith("fold_"):
                    add(fold, f"{ALT_BETA_RUN}/{fold.name}", ROLE_MAIN, ALT_BETA_RUN)

    if include_seeds:
        seeds = TRAINED_MODELS_DIR / SEED_SWEEP_RUN
        if seeds.exists():
            for d in sorted(seeds.iterdir()):
                if d.is_dir():
                    add(d, f"{SEED_SWEEP_RUN}/{d.name}", "seed_variant", SEED_SWEEP_RUN)

    ts = TRAINED_MODELS_DIR / TASK_STATES_RUN
    if (ts / "pre_model").exists():
        add(ts / "pre_model", "pre_model", ROLE_PRE, TASK_STATES_RUN)
    if (ts / "post_model").exists():
        add(ts / "post_model", "post_model", ROLE_POST, TASK_STATES_RUN)

    return entries


# ================================================================================
# 2. SPECTRAL QUANTITIES -- scale-free primary, TR-conditional physical units
# ================================================================================

def spectral_quantities(Lambda: np.ndarray, tr: float) -> dict:
    """
    Convert discrete-time eigenvalues to interpretable quantities.

    Same three conversions as analysis/compare_pre_post.py, restated here so this
    script stays self-contained:
        freq_hz  = |arg| / (2*pi*TR)
        period_s = 2*pi*TR / |arg|
        tau_s    = -TR / ln|Lambda|
    mag / phase / arg are per-step and need no TR; they are the primary outputs.
    """
    mag = np.abs(Lambda)
    phase = np.angle(Lambda)
    aphase = np.abs(phase)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq_hz = aphase / (2.0 * math.pi * tr)
        period_s = np.where(aphase > 1e-12, 2.0 * math.pi * tr / aphase, np.inf)
        tau_s = np.where((mag > 0) & (mag < 1.0), -tr / np.log(mag), np.inf)
    return {"mag": mag, "phase": phase, "aphase": aphase,
            "freq_hz": freq_hz, "period_s": period_s, "tau_s": tau_s}


def distribution_summary(x: np.ndarray) -> dict:
    """Quantiles + moments of a finite 1-D sample, for writing a compact row."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    return {
        "n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]), "median": float(qs[3]),
        "p75": float(qs[4]), "p90": float(qs[5]), "p99": float(qs[6]), "max": float(qs[7]),
    }


def w2(a: np.ndarray, b: np.ndarray) -> float:
    """
    1-D Wasserstein-2 distance between two empirical samples.

    Squared-magnitude lift keeps the metric on the same scale as the inputs
    (scipy's wasserstein_distance is W1); finite values only, so an inf in a
    derived column can never silently dominate a comparison.
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.wasserstein_distance(a, b))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple:
    """Two-sample Kolmogorov-Smirnov statistic and p-value on finite samples."""
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=np.float64); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    res = stats.ks_2samp(a, b, method="exact") if min(a.size, b.size) <= 100 else stats.ks_2samp(a, b)
    return float(res.statistic), float(res.pvalue)


def bh_fdr(p_values) -> np.ndarray:
    """
    Benjamini-Hochberg q-values (FDR-adjusted p). Implemented locally because
    statsmodels is not installed in this project's environment.

    Standard step-up procedure with monotonicity enforced from the largest p
    downwards; q-values are clipped to 1.
    """
    p = np.asarray(p_values, dtype=np.float64)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    scaled = ranked * n / (np.arange(1, n + 1))
    mono = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(mono, 0.0, 1.0)
    q[ok] = out
    return q


def score_against_null(observed: float, null_pool) -> dict:
    """
    Where an observed distance sits inside the distribution of distances between
    independently trained operators. Percentile + one-sided empirical p + z.
    """
    pool = np.asarray(null_pool, dtype=np.float64)
    pool = pool[np.isfinite(pool)]
    row = {"observed": float(observed) if np.isfinite(observed) else float("nan")}
    if pool.size == 0 or not np.isfinite(observed):
        row.update({"null_n": int(pool.size)})
        return row
    row.update({
        "null_n": int(pool.size),
        "null_median": float(np.median(pool)),
        "null_p90": float(np.quantile(pool, 0.90)),
        "null_p95": float(np.quantile(pool, 0.95)),
        "null_max": float(pool.max()),
        "percentile": float(100.0 * (pool <= observed).mean()),
        # one-sided empirical p: how often retraining alone produces a shift at
        # least this large. +1 smoothing so p never reads as exactly 0.
        "p_empirical_one_sided": float((1 + (pool >= observed).sum()) / (1 + pool.size)),
        "z_vs_null": float((observed - pool.mean()) / pool.std(ddof=1)) if pool.std(ddof=1) > 0 else float("nan"),
    })
    return row


# ================================================================================
# 3. CONDITION-LEVEL MODE ENERGETICS -- the testable part
# ================================================================================

def load_brick_model(ckpt_path: Path, verbose: bool = True):
    """
    Instantiate BRICK from a checkpoint for forward passes.

    num_classes is not stored in the checkpoints, so it is inferred by trying the
    values actually present across runs (2 = single-model classifier, 3 and 4 =
    the task-state models) and keeping the one whose state_dict loads strictly.
    """
    from models.brick import BRICK

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    errors = {}
    for nc in (2, 3, 4):
        model = BRICK(h=int(ckpt.get("h", H)), m=int(ckpt.get("m", M)),
                      use_control=bool(ckpt.get("use_control", True)),
                      use_ic=bool(ckpt.get("use_ic", True)),
                      num_classes=nc)
        try:
            model.load_state_dict(sd, strict=True)
        except Exception as exc:  # size mismatch on the classifier head
            errors[nc] = str(exc).splitlines()[0]
            continue
        model.eval()
        if verbose:
            print(f"[reference] {ckpt_path.name}: num_classes={nc}, "
                  f"use_control={model.use_control}, use_ic={model.use_ic}")
        return model, nc, errors
    raise RuntimeError(f"Could not load {ckpt_path} with num_classes in (2,3,4):\n"
                       + "\n".join(f"  num_classes={k}: {v}" for k, v in errors.items()))


def znorm(x: torch.Tensor) -> torch.Tensor:
    """Per-ROI z-score over time. Identical to the convention used by the C extraction."""
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


def mode_energy_from_model(model, x_z: torch.Tensor) -> dict:
    """
    Eigenspace energy per mode for one session, under the SHARED operator.

    forward() returns g_trajectory (T, M) complex, i.e. the state already in the
    eigenbasis, so per-mode energy needs no extra machinery:

        E_total[m] = mean_t |g_bar[t, m]|^2      (with control input, full model)

    The autonomous part is available in closed form from the recurrence
    g_bar(t) = Lambda^t * (P_inv g_0) + sum_{k<=t} Lambda^(t-k) (P_inv C u_k):
    dropping the driven sum leaves

        E_auto[m]  = mean_t |Lambda_m|^t * |a_m|^2 ,  a = P_inv g_0

    E_auto is a LOWER BOUND on the K-only contribution: the exact decomposition
    of |total|^2 contains cross terms that are not attributed here. Report it as
    such; do not read 1 - E_auto/E_total as a clean control share.
    """
    with torch.no_grad():
        out = model(x_z)
        g_bar = out["g_trajectory"]                     # (T, M) complex
        if model.use_ic:
            g_0 = model.encoder(x_z)[0]                # (M,) real
            a = (model.P_inv @ g_0.to(torch.complex64)).abs().cpu().numpy() ** 2
        else:
            a = np.zeros(g_bar.shape[1], dtype=np.float64)
        lam = compute_lambda(model.nu_log.double(), model.theta_log.double())
        t = torch.arange(1, g_bar.shape[0] + 1, dtype=torch.float64)
        decay = lam.real.abs() ** t.unsqueeze(1)        # (T, M), |Lambda|^t
        e_auto = float(decay.mean(dim=0).sum())
        e_auto_vec = (decay.detach().cpu().numpy() * a[None, :]).mean(axis=0)

    e_total = (g_bar.abs() ** 2).mean(dim=0).detach().cpu().numpy().astype(np.float64)
    e_total = np.asarray(e_total, dtype=np.float64)
    total = float(e_total.sum())
    return {
        "energy": e_total,
        "energy_share": e_total / total if total > 0 else np.zeros_like(e_total),
        "energy_auto": e_auto_vec.astype(np.float64),
        "g_bar_std": g_bar.std(dim=0).abs().detach().cpu().numpy().astype(np.float64),
    }


def participation_metrics(energy_share: np.ndarray, Lambda: np.ndarray, tr: float) -> dict:
    """
    Reduce a per-mode energy profile to a handful of testable scalars.

    All metrics are computed on the SAME operator's spectrum, so they are directly
    comparable across the four conditions (that is the point): differences here
    are condition differences, not retraining differences.

      mag_mean  -- energy-weighted mean |Lambda| ("how persistent is what this
                    condition actually uses")
      mag_median-- distribution median over modes carrying >= q10 of the energy
      osc_frac  -- fraction of energy in modes with |arg| beyond OSCILLATION_MIN_RAD
      slow_frac -- fraction of energy in modes with |Lambda| >= PERSISTENCE_SLOW
      near_unit_slow_frac -- energy in modes that are BOTH near-unit (top
                    NEAR_UNIT_QUANTILE of |Lambda|) and oscillatory: the
                    tremor-relevant class targeted by the near-unit screen
      freq_mean -- energy-weighted frequency in Hz (TR-conditional)
      entropy   -- Shannon entropy of the energy share over modes (high = flat /
                    no dominant mode; low = energy concentrated in few modes)
      w2_vs_flat-- Wasserstein distance of the |Lambda| energy-weighted sample
                    against the unweighted spectrum of the same operator
    """
    mag, aphase = np.abs(Lambda), np.abs(np.angle(Lambda))
    freq = aphase / (2.0 * math.pi * tr)
    w = np.asarray(energy_share, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = w / w.sum() if w.sum() > 0 else w

    keep = w >= np.quantile(w[w > 0], 0.10) if (w > 0).any() else np.ones_like(w, dtype=bool)
    near_unit = mag >= np.quantile(mag, NEAR_UNIT_QUANTILE)
    osc = aphase >= OSCILLATION_MIN_RAD

    nz = w[w > 0]
    entropy = float(-(nz * np.log(nz)).sum() / math.log(w.size)) if nz.size else float("nan")

    return {
        "mag_mean": float((w * mag).sum()),
        "mag_median": float(np.median(mag[keep])) if keep.any() else float("nan"),
        "osc_frac": float(w[osc].sum()),
        "slow_frac": float(w[mag >= PERSISTENCE_SLOW].sum()),
        "near_unit_osc_frac": float(w[near_unit & osc].sum()),
        "freq_mean_hz": float((w * freq).sum()),
        "energy_entropy": entropy,
        "top1_share": float(w.max()),
    }


def condition_label(target: str, phase_key: str, group: str) -> dict:
    """
    Map a session to the four conditions, plus the design-level order stratum.

    phase_key is 'mpre' | 'mpost'. group is 'VIM_first' | 'ZI_first' exactly as
    stored in patient_treatment_mapping.xlsx via combine_and_align_rois.py.
    """
    return {
        "target": target,
        "condition": phase_key.replace("m", ""),          # 'pre' / 'post'
        "order_group": "vim_first" if group == "VIM_first" else "zi_first",
        "tx_order": "1st" if (group == "VIM_first" and target == "vim") or \
                             (group == "ZI_first" and target == "zi") else "2nd",
    }


def collect_sessions() -> list:
    """
    Load every subject-target session and split it into pre / post rows.

    Returns list of dicts: subject_id, target, condition, order_group, tx_order,
    x_pre/x_post as float32 arrays. ROI order is checked against TARGET_ROIS
    before anything is used, because a mismatch would silently mislabel every
    spatial readout.
    """
    subjects = load_all()
    target_order = list(TARGET_ROIS)
    rows = []
    for s in subjects:
        if list(s["roi_names"]) != target_order:
            raise ValueError(
                f"ROI order mismatch for {s['subject_id']}/{s['target']}: "
                f"roi_names != TARGET_ROIS. Spatial labels would be wrong."
            )
        for phase_key in ("mpre", "mpost"):
            arr = np.asarray(s[phase_key], dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite BOLD in {s['subject_id']}/{s['target']}/{phase_key}")
            meta = condition_label(s["target"], phase_key, s["group"])
            rows.append({"subject_id": s["subject_id"], **meta, "x": arr})
    return rows


def run_participation(model, sessions, Lambda, tr) -> list:
    """Per-session participation metrics. Unit of replication: subject x target."""
    out = []
    for row in sessions:
        x_z = znorm(torch.from_numpy(row["x"]))
        me = mode_energy_from_model(model, x_z)
        m = participation_metrics(me["energy_share"], Lambda, tr)
        m.update({"subject_id": row["subject_id"], "target": row["target"],
                  "condition": row["condition"], "order_group": row["order_group"],
                  "tx_order": row["tx_order"],
                  "energy_total": float(me["energy"].sum())})
        out.append(m)
    return out


# ================================================================================
# 4. TESTING -- paired and within-subject four-condition contrasts
# ================================================================================

METRICS_TESTED = ("mag_mean", "mag_median", "osc_frac", "slow_frac",
                  "near_unit_osc_frac", "freq_mean_hz", "energy_entropy", "top1_share")


def paired_test(rows, target: str, metric: str):
    """
    Pre vs post within the same target, paired on subject.

    Paired because the two sessions come from the same person on the same day;
    an unpaired test would throw away the design's main asset. n is reported so
    a small overlap cannot be missed.
    """
    pre, post, subj = {}, {}, []
    for r in rows:
        if r["target"] != target:
            continue
        if r["condition"] == "pre":
            pre.setdefault(r["subject_id"], []).append(r[metric])
        elif r["condition"] == "post":
            post.setdefault(r["subject_id"], []).append(r[metric])
    for s in sorted(set(pre) & set(post)):
        if len(pre[s]) == 1 and len(post[s]) == 1:
            subj.append(s)
    if len(subj) < 3:
        return {"target": target, "metric": metric, "n_pairs": len(subj),
                "note": "insufficient pairs"}
    a = np.array([pre[s][0] for s in subj], dtype=np.float64)
    b = np.array([post[s][0] for s in subj], dtype=np.float64)
    d = b - a
    row = {
        "target": target, "metric": metric, "n_pairs": len(subj),
        "mean_pre": float(a.mean()), "mean_post": float(b.mean()),
        "mean_delta": float(d.mean()), "sem_delta": float(d.std(ddof=1) / math.sqrt(len(subj))),
        "cohens_d_paired": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
    }
    try:
        t = stats.ttest_rel(b, a)
        row["t"], row["p_t"] = float(t.statistic), float(t.pvalue)
    except Exception:
        row["t"], row["p_t"] = float("nan"), float("nan")
    try:
        w = stats.wilcoxon(b, a, zero_method="zsplit")
        row["wilcoxon_stat"], row["p_wilcoxon"] = float(w.statistic), float(w.pvalue)
    except Exception:
        row["wilcoxon_stat"], row["p_wilcoxon"] = float("nan"), float("nan")
    # 95% CI of the paired mean delta, t-based (n=19 -> df=18, so this is honest
    # at this sample size rather than a normal-approximation shortcut).
    if len(subj) > 1 and d.std(ddof=1) > 0:
        crit = float(stats.t.ppf(0.975, len(subj) - 1))
        row["ci95_lo"] = row["mean_delta"] - crit * row["sem_delta"]
        row["ci95_hi"] = row["mean_delta"] + crit * row["sem_delta"]
    return row


def friedman_by_order(rows, metric: str):
    """
    Within-subject four-condition contrast (VIM-pre, VIM-post, ZI-pre, ZI-post),
    stratified by treatment order.

    Stratification is mandatory here: the second sonication acts on a brain that
    already received the first, and the design shows an order-dependent
    pattern. Pooling the two orders would average over that confound and make the
    statistic uninterpretable. n per stratum is small (11 / 8), so results are
    reported with consistency counts alongside p-values.
    """
    per_order = {}
    for og in ("vim_first", "zi_first"):
        table, subs = {}, []
        for r in rows:
            if r["order_group"] != og:
                continue
            key = (r["subject_id"], r["target"], r["condition"])
            table.setdefault(key, {})[key[1:] ] = r[metric]
        # assemble one vector of 4 values per subject
        by_subj = {}
        for (sid, tgt, cond), vals in table.items():
            by_subj.setdefault(sid, {})[(tgt, cond)] = vals.get((tgt, cond), list(vals.values())[0])
        needed = [("vim", "pre"), ("vim", "post"), ("zi", "pre"), ("zi", "post")]
        arr = np.array([[by_subj[s][k] for k in needed] for s in sorted(by_subj)
                        if all(k in by_subj[s] for k in needed)], dtype=np.float64)
        row = {"order_group": og, "metric": metric, "n_subjects": int(arr.shape[0])}
        if arr.shape[0] >= 4:
            try:
                f = stats.friedmanchisquare(*arr.T)
                row["friedman_chi2"], row["p_friedman"] = float(f.statistic), float(f.pvalue)
            except Exception:
                row["friedman_chi2"], row["p_friedman"] = float("nan"), float("nan")
            for j, k in enumerate(needed):
                row[f"median_{k[0]}_{k[1]}"] = float(np.median(arr[:, j]))
        per_order[og] = row
    return per_order


def order_interaction(rows, metric: str):
    """
    Is the pre->post shift different between the two treatment orders?

    Two-sample test on delta = post - pre, per target. This is the direct test of
    the order/priming effect at the spectral-energy level.
    """
    out = []
    for target in ("vim", "zi"):
        d = {}
        for r in rows:
            if r["target"] != target:
                continue
            d.setdefault(r["subject_id"], {})[r["condition"]] = r[metric]
        g = {}
        for sid, v in d.items():
            if "pre" in v and "post" in v:
                og = next(r["order_group"] for r in rows if r["subject_id"] == sid)
                g.setdefault(og, []).append(v["post"] - v["pre"])
        a = np.array(g.get("vim_first", []), dtype=np.float64)
        b = np.array(g.get("zi_first", []), dtype=np.float64)
        row = {"target": target, "metric": metric,
               "n_vim_first": int(a.size), "n_zi_first": int(b.size),
               "mean_delta_vim_first": float(a.mean()) if a.size else float("nan"),
               "mean_delta_zi_first": float(b.mean()) if b.size else float("nan")}
        if a.size >= 3 and b.size >= 3:
            try:
                tt = stats.ttest_ind(a, b, equal_var=False)
                row["welch_t"], row["p_welch"] = float(tt.statistic), float(tt.pvalue)
            except Exception:
                row["welch_t"], row["p_welch"] = float("nan"), float("nan")
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            row["mannwhitney_u"], row["p_mw"] = float(mw.statistic), float(mw.pvalue)
        out.append(row)
    return out


# ================================================================================
# 5. NEAR-UNIT OSCILLATORY SCREEN
# ================================================================================

def tremor_band_screen(models, tr) -> list:
    """
    Do any near-unit modes carry an oscillation in the behavioural tremor bands?

    Selection is deliberately conservative: a mode qualifies as near-unit if its
    |Lambda| is at or above the NEAR_UNIT_QUANTILE of that model's own spectrum,
    and its frequency falls inside a clinical tremor band. Because the band
    definition is TR-dependent, --tr must be confirmed before this table is used
    for any claim; the scale-free columns (mag, phase) are unaffected by it.
    """
    rows = []
    for name, spec in models.items():
        q = spectral_quantities(spec["Lambda"], tr)
        mag, freq, aphase = q["mag"], q["freq_hz"], q["aphase"]
        cut = np.quantile(mag, NEAR_UNIT_QUANTILE)
        for band, lo, hi in TREMOR_BANDS_HZ:
            sel = np.where((mag >= cut) & (freq >= lo) & (freq <= hi))[0]
            rows.append({
                "model": name, "role": spec["role"], "tr_s": tr,
                "band": band, "band_lo_hz": lo, "band_hi_hz": hi,
                "near_unit_mag_cutoff": float(cut),
                "n_modes": int(sel.size),
                "mode_index": ";".join(str(i) for i in sel[:20]),
                "median_freq_hz": float(np.median(freq[sel])) if sel.size else float("nan"),
                "median_mag": float(np.median(mag[sel])) if sel.size else float("nan"),
                "median_tau_s": float(np.median(q["tau_s"][sel])) if sel.size else float("nan"),
                "total_mag_mass": float(mag[sel].sum() / mag.sum()) if mag.sum() > 0 else float("nan"),
            })
    return rows


# ================================================================================
# 6. OUTPUT
# ================================================================================

def write_csv(path: Path, rows, field_order=None):
    """Write a list of dicts as CSV, union of keys, utf-8-sig for Excel-safe read."""
    rows = [r for r in rows if r]
    if not rows:
        return
    keys = list(field_order) if field_order else []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    print(f"  wrote {path.relative_to(ROOT_DIR)}  ({len(rows)} rows x {len(keys)} cols)")


def fmt(v, nd=4):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    return str(v)


# ================================================================================
# 7. MAIN
# ================================================================================

def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):     # Windows console may not be UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="cross-condition Koopman spectrum statistics")
    ap.add_argument("--tr", type=float, default=2.0,
                    help="seconds per volume; ONLY an assumption until acquisition metadata "
                         "is confirmed (config.py defines none, compare_pre_post.py hard-codes 2.0)")
    ap.add_argument("--folds", default=DEFAULT_LOSO_RUN, help="primary LOSO run directory name")
    ap.add_argument("--no-alt-beta", action="store_true", help="exclude the beta=0.05 fold pool")
    ap.add_argument("--no-seeds", action="store_true", help="exclude sweep_3_seeds variants")
    ap.add_argument("--skip-participation", action="store_true",
                    help="spectrum extraction and cross-model distances only (no forward passes)")
    ap.add_argument("--reference", default=None,
                    help="checkpoint used for the condition-participation analysis; "
                         "default results/final_model/best_model_cls.pt")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    tr = args.tr
    out_dir = Path(args.out)
    print("=" * 78)
    print("Koopman eigenvalue spectrum: cross-condition statistical analysis")
    print("=" * 78)
    print(f"TR used for physical units : {tr} s  "
          f"(ASSUMED -- confirm acquisition TR before quoting Hz or seconds)")
    print(f"Scale-free primary units   : |Lambda| per step, arg(Lambda) rad per step")
    print(f"Latent dimension M         : {M} = N_ROIS({N_ROIS}) x H({H})")
    print(f"Eigenvalue init ring       : R_MIN={R_MIN}, R_MAX={R_MAX} "
          f"(observed |Lambda| below R_MIN means training left the init ring)")
    print()

    # ---------------- 1. collect spectra ---------------- #
    entries = collect_model_paths(args.folds, not args.no_alt_beta, not args.no_seeds)
    print(f"[1] spectra: {len(entries)} checkpoints")
    models, per_model_rows, eig_rows = {}, [], []
    for e in entries:
        try:
            spec = load_spectrum(e["path"])
        except Exception as exc:
            print(f"    SKIP {e['name']}: {type(exc).__name__}: {exc}")
            continue
        spec["role"] = e["role"]
        spec["group"] = e["group"]
        models[e["name"]] = spec
        q = spectral_quantities(spec["Lambda"], tr)
        smag = distribution_summary(q["mag"])
        sph = distribution_summary(q["phase"])
        sfq = distribution_summary(q["freq_hz"])
        stau = distribution_summary(q["tau_s"])
        mag_finite = q["mag"][np.isfinite(q["mag"])]
        row = {
            "model": e["name"], "role": e["role"], "run_group": e["group"],
            "checkpoint": str(e["path"].relative_to(ROOT_DIR)),
            "M": spec["m"], "h": spec["h"],
            "use_control": spec["use_control"], "use_ic": spec["use_ic"],
            "tr_s": tr,
            "mag_min": smag.get("min"), "mag_median": smag.get("median"),
            "mag_mean": smag.get("mean"), "mag_p90": smag.get("p90"), "mag_max": smag.get("max"),
            "mag_std": smag.get("std"),
            "n_below_R_MIN": int((mag_finite < R_MIN).sum()),
            "n_near_unit_ge_0.95": int((mag_finite >= 0.95).sum()),
            "phase_min": sph.get("min"), "phase_median": sph.get("median"), "phase_max": sph.get("max"),
            "n_real_like": int((q["aphase"] < OSCILLATION_MIN_RAD).sum()),
            "frac_oscillatory": float((q["aphase"] >= OSCILLATION_MIN_RAD).mean()),
            "freq_median_hz": sfq.get("median"), "freq_max_hz": sfq.get("max"),
            "tau_median_s": stau.get("median"), "tau_max_s": stau.get("max"),
            "val_loss_total": spec["val_loss_total"], "val_loss_cls": spec["val_loss_cls"],
        }
        per_model_rows.append(row)
        for i in range(spec["Lambda"].size):
            eig_rows.append({
                "model": e["name"], "role": e["role"], "run_group": e["group"], "tr_s": tr,
                "mode_index": i,
                "lambda_real": float(spec["Lambda"][i].real),
                "lambda_imag": float(spec["Lambda"][i].imag),
                "eigenvalue_mag": float(q["mag"][i]), "phase_rad": float(q["phase"][i]),
                "freq_hz": float(q["freq_hz"][i]), "period_s": float(q["period_s"][i]),
                "tau_s": float(q["tau_s"][i]),
            })

    print(f"    loaded {len(models)} spectra "
          f"({sum(1 for v in models.values() if v['role'] == ROLE_MAIN)} folds, "
          f"{sum(1 for v in models.values() if v['role'] in (ROLE_PRE, ROLE_POST))} task-state, "
          f"{sum(1 for v in models.values() if v['role'] == 'seed_variant')} seed variants)")

    # ---------------- 2. cross-model distances + retraining null ---------------- #
    main_folds = [n for n, v in models.items() if v["role"] == ROLE_MAIN and v["group"] == args.folds]
    other_folds = [n for n, v in models.items() if v["role"] == ROLE_MAIN and v["group"] != args.folds]
    seeds = [n for n, v in models.items() if v["role"] == "seed_variant"]

    def dist(a, b, kind):
        qa, qb = spectral_quantities(models[a]["Lambda"], tr), spectral_quantities(models[b]["Lambda"], tr)
        vec = {"mag": (qa["mag"], qb["mag"]),
               "phase": (qa["phase"], qb["phase"]),
               "freq": (qa["freq_hz"], qb["freq_hz"])}[kind]
        return w2(vec[0], vec[1])

    pw_rows = []
    pool = {"mag": [], "phase": [], "freq": []}
    import itertools
    for a, b in itertools.combinations(main_folds, 2):
        for kind in pool:
            d = dist(a, b, kind)
            pool[kind].append(d)
            pw_rows.append({"pair": f"{a} | {b}", "kind": kind, "distance": d,
                            "pool": "primary_LOSO_folds"})
    for a, b in itertools.combinations(seeds, 2):
        for kind in pool:
            pw_rows.append({"pair": f"{a} | {b}", "kind": kind, "distance": dist(a, b, kind),
                            "pool": "seed_variants"})
    if other_folds:
        for a, b in itertools.combinations(other_folds, 2):
            for kind in pool:
                pw_rows.append({"pair": f"{a} | {b}", "kind": kind, "distance": dist(a, b, kind),
                               "pool": "alt_beta_folds"})

    print(f"\n[2] cross-model spectral distances: "
          f"{len(main_folds)} primary folds -> {len(pool['mag'])} fold-pair distances per quantity")
    for kind in ("mag", "phase"):
        arr = np.array(pool[kind], dtype=np.float64)
        print(f"    {kind:<6} null median {fmt(np.median(arr))}, p95 {fmt(np.quantile(arr,0.95))}, "
              f"max {fmt(arr.max())}")

    # pre vs post -- n=1 model each. Descriptive, scored against the null.
    pp_rows = []
    if ROLE_PRE in [v["role"] for v in models.values()] and ROLE_POST in [v["role"] for v in models.values()]:
        pre_name = next(n for n, v in models.items() if v["role"] == ROLE_PRE)
        post_name = next(n for n, v in models.items() if v["role"] == ROLE_POST)
        for kind in ("mag", "phase", "freq"):
            obs = dist(pre_name, post_name, kind)
            scored = score_against_null(obs, pool[kind] if pool[kind] else [np.nan])
            scored.update({"pair": f"{pre_name} | {post_name}", "kind": kind,
                           "interpretation": "descriptive: n=1 model per condition; percentile "
                                             "is against retraining spread, NOT a significance test"})
            pp_rows.append(scored)
        qa, qb = spectral_quantities(models[pre_name]["Lambda"], tr), spectral_quantities(models[post_name]["Lambda"], tr)
        ks_stat, ks_p = ks_two_sample(qa["mag"], qb["mag"])
        print(f"\n[2b] pre_model vs post_model (the only operator-level pre/post contrast):")
        for r in pp_rows:
            print(f"    {r['kind']:<6} W2={fmt(r['observed'])}  null_median={fmt(r.get('null_median'))}  "
                  f"percentile={fmt(r.get('percentile'),1)}  z={fmt(r.get('z_vs_null'),2)}")
        print(f"    KS on |Lambda|: D={fmt(ks_stat,3)} p={fmt(ks_p,4)} "
              f"(descriptive: 96 modes per model are not independent observations)")
        pp_rows.append({"pair": f"{pre_name} | {post_name}", "kind": "KS_mag",
                        "observed": ks_stat, "p_empirical_one_sided": ks_p,
                        "interpretation": "descriptive only"})

    # ---------------- 3. condition participation ---------------- #
    part_rows, mode_part_rows, test_rows = [], [], []
    ref_path = Path(args.reference) if args.reference else ROOT_DIR / "results" / "final_model" / "best_model_cls.pt"
    if not ref_path.exists():
        print(f"\n[3] reference checkpoint missing ({ref_path}); participation skipped")
    elif args.skip_participation:
        print("\n[3] participation skipped (--skip-participation)")
    else:
        print(f"\n[3] condition-level mode energetics under ONE shared operator: {ref_path.name}")
        model, nc, load_errors = load_brick_model(ref_path)
        ref_spec = load_spectrum(ref_path)
        sessions = collect_sessions()
        n_subj = len({r["subject_id"] for r in sessions})
        print(f"    {len(sessions)} sessions from {n_subj} subjects x 2 targets x pre/post; "
              f"num_classes={nc}")
        part_rows = run_participation(model, sessions, ref_spec["Lambda"], tr)

        # per-mode table for the "which modes does each condition use" question
        agg = {}
        for row in sessions:
            x_z = znorm(torch.from_numpy(row["x"]))
            e = mode_energy_from_model(model, x_z)["energy_share"]
            agg.setdefault((row["target"], row["condition"]), []).append(e)
        q_ref = spectral_quantities(ref_spec["Lambda"], tr)
        for (tgt, cond), vecs in sorted(agg.items()):
            mean_share = np.mean(np.vstack(vecs), axis=0)
            for m in range(mean_share.size):
                mode_part_rows.append({
                    "target": tgt, "condition": cond, "mode_index": m, "n_sessions": len(vecs),
                    "mean_energy_share": float(mean_share[m]),
                    "eigenvalue_mag": float(q_ref["mag"][m]),
                    "phase_rad": float(q_ref["phase"][m]),
                    "freq_hz": float(q_ref["freq_hz"][m]), "tau_s": float(q_ref["tau_s"][m]),
                    "roi_top3": roi_top3_from_mode(ref_spec["W_bar_x"], m),
                })

        print("\n[3b] paired pre-vs-post tests of spectral energy participation "
              "(unit of replication = subject):")
        for target in ("vim", "zi"):
            for metric in METRICS_TESTED:
                test_rows.append(paired_test(part_rows, target, metric))
        if test_rows:
            qs = bh_fdr([r.get("p_t") for r in test_rows])
            for r, q in zip(test_rows, qs):
                r["p_t_fdr_bh"] = q
            qs2 = bh_fdr([r.get("p_wilcoxon") for r in test_rows])
            for r, q in zip(test_rows, qs2):
                r["p_wilcoxon_fdr_bh"] = q
            shown = 0
            for r in test_rows:
                if "mean_delta" not in r:
                    continue
                star = "  <== smallest q" if r.get("p_t_fdr_bh") == min(
                    [x.get("p_t_fdr_bh", 1) for x in test_rows if "p_t_fdr_bh" in x]) else ""
                print(f"    {r['target']:<4} {r['metric']:<19} n={r.get('n_pairs')} "
                      f"delta={fmt(r.get('mean_delta'))} CI[{fmt(r.get('ci95_lo'))},{fmt(r.get('ci95_hi'))}] "
                      f"d={fmt(r.get('cohens_d_paired'),2)} p_t={fmt(r.get('p_t'),4)} "
                      f"p_w={fmt(r.get('p_wilcoxon'),4)}{star}")
                shown += 1
            if shown == 0:
                print("    (no testable metric)")

        print("\n[3c] four-condition within-subject contrast, stratified by treatment order:")
        fr_rows = []
        for metric in METRICS_TESTED:
            per = friedman_by_order(part_rows, metric)
            for og, r in per.items():
                fr_rows.append(r)
                if "p_friedman" in r:
                    print(f"    {og:<10} {metric:<19} n={r['n_subjects']} "
                          f"chi2={fmt(r['friedman_chi2'],2)} p={fmt(r['p_friedman'],4)}")

        print("\n[3d] order interaction on the pre->post shift "
              "(is the shift different for VIM-first vs ZI-first?):")
        oi_rows = order_interaction(part_rows, "mag_mean") + \
                  order_interaction(part_rows, "near_unit_osc_frac")
        for r in oi_rows:
            if "p_welch" in r:
                print(f"    {r['target']:<4} {r['metric']:<19} "
                      f"vim_first(n={r['n_vim_first']})={fmt(r['mean_delta_vim_first'])} "
                      f"zi_first(n={r['n_zi_first']})={fmt(r['mean_delta_zi_first'])} "
                      f"p_welch={fmt(r['p_welch'],4)} p_mw={fmt(r['p_mw'],4)}")
        test_rows.extend(fr_rows)
        test_rows.extend(oi_rows)

    # ---------------- 4. tremor band screen ---------------- #
    tb_rows = tremor_band_screen(models, tr)

    # ---------------- 5. write ---------------- #
    print(f"\n[5] writing to {out_dir.relative_to(ROOT_DIR)}")
    write_csv(out_dir / "model_spectrum_summary.csv", per_model_rows,
              field_order=["model", "role", "run_group", "tr_s", "M", "mag_min", "mag_median",
                           "mag_mean", "mag_p90", "mag_max", "n_below_R_MIN", "n_near_unit_ge_0.95",
                           "frac_oscillatory", "n_real_like", "freq_median_hz", "freq_max_hz",
                           "tau_median_s", "tau_max_s", "checkpoint"])
    write_csv(out_dir / "eigenvalue_records.csv", eig_rows)
    write_csv(out_dir / "pairwise_wasserstein_models.csv", pw_rows)
    write_csv(out_dir / "pre_post_vs_retraining_null.csv", pp_rows)
    write_csv(out_dir / "condition_participation.csv", part_rows)
    write_csv(out_dir / "condition_participation_by_mode.csv", mode_part_rows)
    write_csv(out_dir / "condition_stats_tests.csv", test_rows)
    write_csv(out_dir / "tremor_band_modes.csv", tb_rows)

    manifest = {
        "tool": "analysis/koopman_spectrum_stats.py",
        "purpose": "cross-condition Koopman eigenvalue spectrum statistics",
        "tr_s_assumed": tr,
        "tr_caveat": "config.py defines no TR and the .mat headers carry no acquisition metadata; "
                     "compare_pre_post.py hard-codes TR=2.0. All Hz/second columns are conditional "
                     "on --tr and must be re-derived once the acquisition TR is confirmed.",
        "reference_checkpoint": str(ref_path),
        "primary_folds": args.folds,
        "n_models_loaded": len(models),
        "n_paired_sessions": len(part_rows),
        "identifiability_note": "K is a single shared operator per model. The operator-level "
                                "pre/post contrast is n=1 model per condition and is DESCRIPTIVE. "
                                "Inference is available only on condition-level mode energetics "
                                "under a fixed operator, with subject as the unit of replication.",
        "near_unit_quantile": NEAR_UNIT_QUANTILE,
        "persistence_slow_threshold": PERSISTENCE_SLOW,
        "oscillation_min_rad": OSCILLATION_MIN_RAD,
        "tremor_bands_hz_prior": TREMOR_BANDS_HZ,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"  wrote {out_dir.relative_to(ROOT_DIR)}/run_manifest.json")

    print("\n" + "=" * 78)
    print("Interpretation guardrails")
    print("=" * 78)
    print(" * pre_model vs post_model is one model each: treat as hypothesis-generating,")
    print("   score against the fold-pair null, do not attach a significance claim.")
    print(" * Mode indices are not comparable between independently trained models;")
    print("   only distribution-level quantities are.")
    print(" * Any Hz or second value depends on the assumed TR (see run_manifest.json).")
    print(" * N=19 caps inferential strength: use delta, CI and consistency counts,")
    print("   not just p-values.")
    return 0


def roi_top3_from_mode(W_bar_x: np.ndarray, mode_index: int) -> str:
    """
    Top 3 ROIs loading on a mode, by |column| of W_bar_x.

    W_bar_x = W_x @ P is the (N_ROIS, M) readout in the eigenbasis, so column m
    IS the mode's spatial profile. This helper is kept here because the spectrum and
    mode-participation analyses consume the same object; the values are |.| only, so no sign information is
    implied by the ranking.
    """
    col = np.abs(W_bar_x[:, mode_index])
    order = np.argsort(col)[::-1][:3]
    names = [TARGET_ROIS[i] for i in order if i < len(TARGET_ROIS)]
    return ";".join(f"{n}({col[i]:.3f})" for n, i in zip(names, order))


if __name__ == "__main__":
    sys.exit(main())
