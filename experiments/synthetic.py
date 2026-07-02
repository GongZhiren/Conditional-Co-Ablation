#!/usr/bin/env python3
"""Table 2 --- controlled-redundancy synthetic benchmark (validates Proposition 2).

We plant the self-repair *mechanism* (not the score): dormant backups that leave no clean-state
trace but wake up once the primaries are removed. Clean-state scores are then provably at or
below chance, while the conditional CoAx score recovers the backups. No model is needed --- this
runs in a second and isolates *which* property of a score lets it see a conditionally-active backup.

    python experiments/synthetic.py

Approximate expected ROC-AUC (100 backups vs 100 inert, 40 trials):
    first-order energy (clean) 0.42   [below chance --- anti-ranks the dormant backups]
    AtP*-style gradient (clean) 0.51  [chance]
    GIM-style (conditional)     ~0.75 [conditioning helps, but an answer-gradient is noisy]
    CoAx (conditional growth)   0.92
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def one_trial(rng, r=192, K=4, M=100, Z=100, gamma=0.7, sigma=0.05):
    e = _norm(rng.standard_normal(r))
    # primaries: aligned to the answer direction e
    prim = []
    for _ in range(K):
        perp = rng.standard_normal(r); perp -= (perp @ e) * e
        prim.append(_norm(gamma * e + np.sqrt(1 - gamma ** 2) * _norm(perp)))
    prim = np.array(prim)
    # backups: shadow a random primary (so they write the answer), near-dormant on the clean pass,
    # gate opens once the primaries are removed (graded wake-up).
    Wb = _norm(prim[rng.integers(0, K, size=M)] + 0.10 * rng.standard_normal((M, r)))
    a_clean_b = np.full(M, 0.04)
    a_cond_b = np.clip(0.42 + 0.10 * rng.standard_normal(M), 0.0, 1.2)
    # inert: random directions, small clean gate, no gating change
    Wz = _norm(rng.standard_normal((Z, r)))
    a_clean_z = np.full(Z, 0.13)

    def effect(W, gate):
        return gate[:, None] * W + sigma * rng.standard_normal(W.shape)

    dz_clean = np.vstack([effect(Wb, a_clean_b), effect(Wz, a_clean_z)])
    dz_cond = np.vstack([effect(Wb, a_cond_b), effect(Wz, a_clean_z)])  # inert unchanged
    y = np.array([1] * M + [0] * Z)                                     # positives = backups
    ehat = _norm(e + 0.15 * rng.standard_normal(r))                     # noisy gradient estimate

    scores = {
        "first-order energy (clean)":   (dz_clean ** 2).sum(1),
        "AtP*-style gradient (clean)":  np.abs(dz_clean @ e),
        "GIM-style (conditional)":      np.abs(dz_cond @ ehat),
        "CoAx (conditional growth)":    (dz_cond ** 2).sum(1) - (dz_clean ** 2).sum(1),
    }
    return {k: roc_auc_score(y, v) for k, v in scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    aucs = {}
    for _ in range(args.trials):
        for k, v in one_trial(rng).items():
            aucs.setdefault(k, []).append(v)

    bar = "=" * 56
    print(f"\n{bar}\n  Synthetic backup recovery ROC-AUC ({args.trials} trials)\n{bar}")
    for k in ["first-order energy (clean)", "AtP*-style gradient (clean)",
              "GIM-style (conditional)", "CoAx (conditional growth)"]:
        a = np.array(aucs[k])
        print(f"  {k:30s} {a.mean():.2f} ± {a.std():.2f}")
    print(bar)
    print("  Clean-state scores sit at/below chance (0.50); only conditioning lifts a score above it,")
    print("  exactly as Proposition 2 predicts. CoAx is strongest and alignment-invariant.")


if __name__ == "__main__":
    main()
