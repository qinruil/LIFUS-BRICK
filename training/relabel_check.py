#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label-perturbation check for the shared operator K.

Question: does what the shared Koopman operator K absorbs depend on how the
pre/post classifier labels are defined? If K is a generic property of the
recording, its structure should be stable under label changes; if K is shaped
by the task definition, relabeling should move the %K variance share and the
eigenmode spectrum.

Three relabeling configurations are trained. Everything else (subject split,
hyperparameters, epochs, seeds) is identical to the reference run in
results/final_model:

  shuffle   labels randomly permuted over the 76 sessions, counts kept 38/38
            (fixed rng seed);
  constant  all labels set to 0 -- the classifier has nothing to discriminate;
  swap      pre <-> post (labels inverted, counts unchanged).

The mainline dataset module is not modified: a BRICKDataset subclass overrides
__getitem__, and training.train is pointed at it.

Usage:
  python training/relabel_check.py shuffle
  python training/relabel_check.py constant
  python training/relabel_check.py swap
"""

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(5)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.dataset import BRICKDataset   # noqa: E402
import training.train as tr                 # noqa: E402

N_EPOCHS = 2700            # reference run stopped at epoch 2700
SHUFFLE_SEED = 20261106

SPLIT = {
    "train": ["sub-fuspd01", "sub-fuspd03", "sub-fuspd04", "sub-fuspd06",
              "sub-fuspd08", "sub-fuspd12", "sub-fuspd13", "sub-fuspd15",
              "sub-fuspd16", "sub-fuspd18", "sub-fuspd20", "sub-fuspd24",
              "sub-fuspd26"],
    "val":   ["sub-fuspd09", "sub-fuspd11", "sub-fuspd19", "sub-fuspd22",
              "sub-fuspd23"],
    "test":  ["sub-fuspd07"],
}


class RelabeledDataset(BRICKDataset):
    """BRICKDataset with the classifier label remapped by a strategy."""

    def __init__(self, data_dir, strategy: str, cache_fc: bool = True):
        super().__init__(data_dir, cache_fc=cache_fc)
        self.strategy = strategy
        n = len(self)
        if strategy == "shuffle":
            rng = np.random.default_rng(SHUFFLE_SEED)
            lab = np.array([0] * (n - n // 2) + [1] * (n // 2))
            self._perm = rng.permutation(lab)
        elif strategy in ("constant", "swap"):
            self._perm = None
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        orig = int(item["lifus_condition"])
        if self.strategy == "shuffle":
            new = int(self._perm[idx])
        elif self.strategy == "constant":
            new = 0
        else:
            new = 1 - orig
        item["lifus_condition"] = torch.tensor(new, dtype=torch.long)
        return item


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("shuffle", "constant", "swap"):
        raise SystemExit("usage: python training/relabel_check.py {shuffle|constant|swap}")
    strategy = sys.argv[1]

    def patched_dataset(data_dir, cache_fc=True):
        return RelabeledDataset(data_dir, strategy, cache_fc=cache_fc)

    tr.BRICKDataset = patched_dataset   # train() resolves the class at call time

    tr.train(n_epochs=N_EPOCHS, use_control=True, use_ic=True,
             run_name=strategy, subject_split=SPLIT,
             base_results_dir=ROOT_DIR / "results" / "relabel_check")


if __name__ == "__main__":
    main()
