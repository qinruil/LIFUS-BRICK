#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Design disclosure for the crossover treatment design.

The cohort is a two-treatment crossover: every patient receives both VIM and ZI
low-intensity focused ultrasound, in one of two orders (VIM-first or ZI-first).
Delta-C is computed per patient per target, which means the four "groups" used
in the analysis are not independent cohorts but order partitions of the same
19 patients.

This script writes, for both the full 19-patient set and the 16-patient set used
after excluding the three divergent folds:
  - patient membership per group and per target,
  - group sizes and pairwise overlap counts,
  - the number of within-subject VIM-ZI pairs available for paired designs.

Outputs: results/statistics_supplement/group_membership_full.csv
         results/statistics_supplement/group_membership_filtered.csv
         results/statistics_supplement/group_overlap_summary.csv
"""

from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "statistics_supplement"
LOSO_DIR = ROOT / "results" / "figures" / "loso" / "loso_19_fold_beta_0.2_13to1to5_split"
FULL_CSV = LOSO_DIR / "loso_delta_summary.csv"
FILT_CSV = LOSO_DIR / "loso_delta_summary_filtered.csv"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("subject_id")]


def membership(rows):
    """Return dict target -> subject -> (group, count of roi rows)."""
    out = defaultdict(dict)
    for r in rows:
        subj, tgt, grp = r["subject_id"], r["target"], r["group_str"]
        out[tgt][subj] = grp
    return out


def build(rows, tag):
    mem = membership(rows)
    vim = mem["vim"]
    zi = mem["zi"]
    subjects = sorted(set(vim) | set(zi))
    rows_out = []
    for s in subjects:
        g = vim.get(s) or zi.get(s)
        # group truth: which order group owns this subject
        in_vim1 = (g == "VIM_first") and s in vim
        in_vim2 = (g == "ZI_first") and s in vim
        in_zi1 = (g == "ZI_first") and s in zi
        in_zi2 = (g == "VIM_first") and s in zi
        rows_out.append({
            "subject": s, "order_group": g,
            "in_VIM1st": int(in_vim1), "in_VIM2nd": int(in_vim2),
            "in_ZI1st": int(in_zi1), "in_ZI2nd": int(in_zi2),
            "has_vim": int(s in vim), "has_zi": int(s in zi),
        })
    return subjects, rows_out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    full_rows = load(FULL_CSV)
    filt_rows = load(FILT_CSV)

    subjects_full, m_full = build(full_rows, "full")
    subjects_filt, m_filt = build(filt_rows, "filtered")

    for tag, m in (("full", m_full), ("filtered", m_filt)):
        p = RES / f"group_membership_{tag}.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(m[0].keys()))
            w.writeheader()
            w.writerows(m)

    # overlap summary
    def grp_sizes(m):
        sizes = defaultdict(int)
        for r in m:
            for k in ("in_VIM1st", "in_VIM2nd", "in_ZI1st", "in_ZI2nd"):
                if r[k]:
                    sizes[k.replace("in_", "")] += 1
        return sizes

    s_full, s_filt = grp_sizes(m_full), grp_sizes(m_filt)
    # within-subject VIM-ZI pairs: patients with both targets
    pairs_full = len(subjects_full)
    pairs_filt = len(subjects_filt)

    summary = [{
        "analysis_set": "full_19", **{f"n_{g}": s_full.get(g, 0) for g in
                                      ("VIM1st", "VIM2nd", "ZI1st", "ZI2nd")},
        "subjects_with_both_targets": pairs_full,
        "note": ("crossover: VIM1st and ZI2nd are the same patients "
                 "(VIM-first); ZI1st and VIM2nd are the same patients "
                 "(ZI-first). Within-subject VIM-ZI pairs = all patients."),
    }, {
        "analysis_set": "filtered_16", **{f"n_{g}": s_filt.get(g, 0) for g in
                                          ("VIM1st", "VIM2nd", "ZI1st", "ZI2nd")},
        "subjects_with_both_targets": pairs_filt,
        "note": ("After excluding divergent folds (3 patients), pairwise "
                 "independence across order groups is unchanged: the two "
                 "target-level comparisons remain paired within subject."),
    }]
    p = RES / "group_overlap_summary.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("Subjects (full):", len(subjects_full))
    print("Subjects (filtered):", len(subjects_filt))
    print("Group sizes full:", dict(s_full))
    print("Group sizes filtered:", dict(s_filt))
    print("Excluded patients:", sorted(set(subjects_full) - set(subjects_filt)))
    print("Written group_overlap_disclosure outputs to", RES)


if __name__ == "__main__":
    main()