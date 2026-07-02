"""Shared helpers for the experiment scripts (path setup + ranking metrics)."""
from __future__ import annotations

import pathlib
import sys

# make `import coax` work when running `python experiments/<script>.py` from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def candidates(model, exclude):
    exclude = set(exclude)
    return [u for u in range(model.num_units) if model.layer_head(u) not in exclude]


def backup_auc(scores, model, positives, exclude) -> float:
    """ROC-AUC for ranking the ``positives`` heads among all candidates (``exclude`` removed)."""
    positives = set(positives)
    units = candidates(model, exclude)
    y = np.array([model.layer_head(u) in positives for u in units], dtype=int)
    s = np.nan_to_num(np.array([scores[u] for u in units]))
    return float(roc_auc_score(y, s))


def top_units(scores, model, exclude, k):
    """The k highest-scoring candidate heads, as (layer, head) tuples."""
    units = candidates(model, exclude)
    order = sorted(units, key=lambda u: np.nan_to_num(scores[u]), reverse=True)
    return [model.layer_head(u) for u in order[:k]]


def recovered_backups(model, coax, comp_scores, seed, k, min_ratio=1.05):
    """The CoAx-recovered backup set used downstream: the top-``k`` candidates by conditional
    growth that also *wake up* (output-norm ratio > ``min_ratio`` once the seed is ablated). The
    wake-up signature removes the high-energy early-layer false positives raw ranking would
    include, so the set behaves like real backups for knockout / attribution (paper Sec. 3.2)."""
    ratios = coax.wakeup_ratios(seed)
    units = candidates(model, seed)
    order = sorted(units, key=lambda u: np.nan_to_num(comp_scores[u]), reverse=True)
    kept = [u for u in order if ratios[u] > min_ratio][:k]
    return [model.layer_head(u) for u in kept]
