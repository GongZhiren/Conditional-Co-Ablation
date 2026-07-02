#!/usr/bin/env python3
"""Reproduce the headline result: recovering the GPT-2-small IOI backup name-movers.

Conditioning CoAx on the three documented name-mover *primaries* and ranking the remaining
141 candidate heads by conditional-growth recovers the eight documented backups that
single-ablation saliency misses:

    single ablation (1st order) : backup ROC-AUC ~ 0.33   (dormant -> ranked *below* chance)
    CoAx conditional (2nd order): backup ROC-AUC ~ 0.91

Usage
-----
    python reproduce_ioi.py                       # GPT-2-small from the Hugging Face hub
    python reproduce_ioi.py --model <path>        # a local checkpoint
    python reproduce_ioi.py --num-prompts 96      # match the paper's discovery setting

The score is label-free (labels are used only to *evaluate* the ranking) and reproduces on
CPU in a few minutes; a single GPU takes seconds.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from coax import CoAx, Model
from coax.ioi import BACKUPS, PRIMARIES, ioi_prompts


def backup_auc(scores: np.ndarray, model: Model, positives, exclude) -> float:
    """ROC-AUC for ranking the ``positives`` heads among all candidates (``exclude`` removed)."""
    positives, exclude = set(positives), set(exclude)
    units = [u for u in range(model.num_units) if model.layer_head(u) not in exclude]
    labels = np.array([model.layer_head(u) in positives for u in units], dtype=int)
    values = np.nan_to_num(np.array([scores[u] for u in units]))
    return float(roc_auc_score(labels, values))


def topk_recall(scores: np.ndarray, model: Model, positives, exclude, ks=(8, 10, 15, 20)):
    positives, exclude = set(positives), set(exclude)
    units = [u for u in range(model.num_units) if model.layer_head(u) not in exclude]
    order = sorted(units, key=lambda u: np.nan_to_num(scores[u]), reverse=True)
    ranked = [model.layer_head(u) for u in order]
    return {k: sum(h in positives for h in ranked[:k]) for k in ks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2", help="HF id or local path (default: gpt2)")
    ap.add_argument("--num-prompts", type=int, default=96,
                    help="IOI calibration prompts (paper setting: 96; 32 already reaches ~0.90)")
    ap.add_argument("--top-r", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model = Model(args.model)
    prompts = ioi_prompts(args.num_prompts, seed=args.seed)
    print(f"Scoring {model.num_units} heads on {len(prompts)} IOI prompts "
          f"(seed {args.seed}, top-r {args.top_r}, device {model.device}) ...")

    coax = CoAx(model, prompts, top_r=args.top_r)
    result = coax.conditional_compensation(PRIMARIES)

    auc_single = backup_auc(result["single"], model, BACKUPS, exclude=PRIMARIES)
    auc_coax = backup_auc(result["compensation"], model, BACKUPS, exclude=PRIMARIES)
    recall = topk_recall(result["compensation"], model, BACKUPS, exclude=PRIMARIES)

    bar = "=" * 60
    print(f"\n{bar}\n  Backup-name-mover recovery (8 documented backups, 141 candidates)\n{bar}")
    print(f"  single ablation (1st order)   backup ROC-AUC = {auc_single:.3f}")
    print(f"  CoAx conditional (2nd order)  backup ROC-AUC = {auc_coax:.3f}")
    print(f"{bar}")
    print("  CoAx top-k recall of documented backups: "
          + ", ".join(f"{n}/8 @ top-{k}" for k, n in recall.items()))

    order = sorted((u for u in range(model.num_units)
                    if model.layer_head(u) not in set(PRIMARIES)),
                   key=lambda u: np.nan_to_num(result["compensation"][u]), reverse=True)
    print("\n  Top-10 heads by CoAx score (* = documented backup):")
    backups = set(BACKUPS)
    for rank, u in enumerate(order[:10], 1):
        l, h = model.layer_head(u)
        star = "*" if (l, h) in backups else " "
        print(f"    {rank:2d}. {star} L{l}.H{h:<2d}  comp={result['compensation'][u]:+.4f}")


if __name__ == "__main__":
    main()
